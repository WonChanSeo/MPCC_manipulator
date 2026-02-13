#include "osqp.h"
#include "auxil.h"
#include "lin_alg.h"
#include "scaling.h"
#include "util.h"
#include "printing.h"
#include "timing.h"
#include "osqp_api_functions.h"
#include <stdint.h>
#include <limits.h>
#include <string.h>

/* =========[ Rounding mode support (same pattern as qdldl.c) ]========= */
#if defined(OSQP_USE_TRUNCATE) || defined(OSQP_USE_ROUND_TO_EVEN)
#include <fenv.h>
#endif

#if defined(OSQP_USE_TRUNCATE)
  #define OSQP_ROUNDING_MODE FE_TOWARDZERO
#elif defined(OSQP_USE_ROUND_TO_EVEN)
  #define OSQP_ROUNDING_MODE FE_TONEAREST
#endif

#ifdef OSQP_ROUNDING_MODE
#include "algebra_impl.h"  /* for OSQPMatrix_->csc access */

/* =========[ ASIC bit-exact: static globals ]========= */
static OSQPFloat* g_A_dns = NULL;    /* A dense row-major [m x n] */
static OSQPFloat* g_P_dns = NULL;    /* P dense full-symmetric row-major [n x n] */
static int*       g_dinv_exp = NULL; /* Dinv exponent array [n] */
static int*       g_einv_exp = NULL; /* Einv exponent array [m] */
static OSQPInt    g_dns_m = 0, g_dns_n = 0;
static int        g_dns_initialized = 0;

/* =========[ ASIC bit-exact: FP32 primitives ]========= */

static inline OSQPFloat fp32_mul_conv(OSQPFloat a, OSQPFloat b) {
    volatile OSQPFloat r = a * b;
    return r;
}

static inline OSQPFloat fp32_add_conv(OSQPFloat a, OSQPFloat b) {
    volatile OSQPFloat r = a + b;
    return r;
}

static inline OSQPFloat fp32_sub_conv(OSQPFloat a, OSQPFloat b) {
    /* RTL: fp32_add(a, b ^ 0x80000000) — sign bit flip */
    union { float f; uint32_t u; } v;
    v.f = (float)b;
    v.u ^= 0x80000000u;
    volatile OSQPFloat r = (OSQPFloat)((float)a + v.f);
    return r;
}

static inline OSQPFloat exp_adjust_conv(OSQPFloat val, int adj) {
    union { float f; uint32_t u; } v;
    v.f = (float)val;
    uint32_t exp_field = (v.u >> 23) & 0xFFu;
    if (exp_field == 0) return 0.0f;  /* zero/denorm -> 0 */
    uint32_t new_exp = (exp_field + (uint32_t)(uint8_t)adj) & 0xFFu;
    v.u = (v.u & 0x807FFFFFu) | (new_exp << 23);
    return (OSQPFloat)v.f;
}

static inline OSQPFloat update_max_abs_conv(OSQPFloat a, OSQPFloat b) {
    /* Returns max(|a|, |b|) via unsigned integer comparison */
    union { float f; uint32_t u; } va, vb;
    va.f = (float)a; vb.f = (float)b;
    uint32_t abs_a = va.u & 0x7FFFFFFFu;
    uint32_t abs_b = vb.u & 0x7FFFFFFFu;
    if (abs_a >= abs_b) { va.u = abs_a; return (OSQPFloat)va.f; }
    else                { vb.u = abs_b; return (OSQPFloat)vb.f; }
}

/* =========[ ASIC bit-exact: pairwise reduction (128-lane tree) ]========= */

static OSQPFloat pairwise_reduce_128(OSQPFloat* buf, int count) {
    /* Zero-pad to 128 */
    for (int i = count; i < 128; i++) buf[i] = 0.0f;

    int size = 128;
    while (size > 1) {
        int half = size / 2;
        for (int i = 0; i < half; i++) {
            buf[i] = fp32_add_conv(buf[2*i], buf[2*i+1]);
        }
        size = half;
    }
    return buf[0];
}

/* =========[ ASIC bit-exact: 2-pass dot product (reverse permuted) ]========= */
/* PASS 1: cols 51-178 (128 products), PASS 2: cols 0-50 (51 products, pad to 128) */

static OSQPFloat two_pass_dot_179(const OSQPFloat* row, const OSQPFloat* x) {
    OSQPFloat buf[128];

    /* PASS 1: cols 51-178 (128 products, ASIC reverse load: 178,177,...,51) */
    for (int j = 0; j < 128; j++)
        buf[j] = fp32_mul_conv(row[178 - j], x[178 - j]);
    OSQPFloat pass1 = pairwise_reduce_128(buf, 128);

    /* PASS 2: cols 0-50 (51 products, ASIC reverse load: 50,49,...,0) */
    for (int j = 0; j < 51; j++)
        buf[j] = fp32_mul_conv(row[50 - j], x[50 - j]);
    OSQPFloat pass2 = pairwise_reduce_128(buf, 51);

    /* Stage 4: accumulate */
    return fp32_add_conv(pass1, pass2);
}

/* =========[ ASIC bit-exact: A'*y outer product accumulation ]========= */

static void compute_Aty_asic(const OSQPFloat* A_dns, OSQPInt m, OSQPInt n,
                              const OSQPFloat* y, OSQPFloat* Aty) {
    for (OSQPInt j = 0; j < n; j++) Aty[j] = 0.0f;

    for (OSQPInt i = 0; i < m; i++) {
        const OSQPFloat* A_row = A_dns + i * n;
        for (OSQPInt j = 0; j < n; j++) {
            OSQPFloat prod = fp32_mul_conv(A_row[j], y[i]);
            Aty[j] = fp32_add_conv(prod, Aty[j]);
        }
    }
}

/* =========[ ASIC bit-exact: dual residual 2-step PE ]========= */

static void compute_dual_res_pe(const OSQPFloat* q, const OSQPFloat* Px,
                                 const OSQPFloat* Aty, OSQPInt n, OSQPFloat* dr) {
    for (OSQPInt j = 0; j < n; j++) {
        OSQPFloat temp = fp32_add_conv(fp32_mul_conv(1.0f, q[j]), Px[j]);
        dr[j] = fp32_add_conv(fp32_mul_conv(1.0f, temp), Aty[j]);
    }
}

/* =========[ ASIC bit-exact: dense matrix initialization ]========= */

static void extract_scaling_exponents(const OSQPFloat* vec, OSQPInt len, int* exp_out) {
    for (OSQPInt i = 0; i < len; i++) {
        union { float f; uint32_t u; } v;
        v.f = (float)vec[i];
        uint32_t exp_field = (v.u >> 23) & 0xFFu;
        if (exp_field == 0) {
            exp_out[i] = 0;
        } else {
            exp_out[i] = (int)exp_field - 127;
        }
    }
}

static const OSQPFloat* g_last_A_x_ptr = NULL;  /* track A CSC data for change detection */

/* Call at start of each osqp_solve to force dense matrix rebuild
 * (A/P values are updated in-place so pointer comparison cannot detect changes) */
void auxil_reset_dense_cache(void) {
    g_dns_initialized = 0;
}

static void auxil_init_dense_matrices(const OSQPSolver* solver) {
    if (g_dns_initialized) return;  /* already built for this solve */

    /* Free old allocations if re-initializing */
    if (g_A_dns)    { c_free(g_A_dns);    g_A_dns = NULL; }
    if (g_P_dns)    { c_free(g_P_dns);    g_P_dns = NULL; }
    if (g_dinv_exp) { c_free(g_dinv_exp); g_dinv_exp = NULL; }
    if (g_einv_exp) { c_free(g_einv_exp); g_einv_exp = NULL; }

    OSQPWorkspace* work = solver->work;
    OSQPInt m = work->data->m;
    OSQPInt n = work->data->n;

    /* Dense A: CSC -> row-major [m x n] */
    OSQPCscMatrix* A_csc = work->data->A->csc;
    g_A_dns = (OSQPFloat*)c_calloc(m * n, sizeof(OSQPFloat));
    for (OSQPInt j = 0; j < n; j++) {
        for (OSQPInt k = A_csc->p[j]; k < A_csc->p[j+1]; k++) {
            OSQPInt i = A_csc->i[k];
            g_A_dns[i * n + j] = A_csc->x[k];
        }
    }

    /* Dense P: CSC upper-tri -> full symmetric row-major [n x n] */
    OSQPCscMatrix* P_csc = work->data->P->csc;
    g_P_dns = (OSQPFloat*)c_calloc(n * n, sizeof(OSQPFloat));
    for (OSQPInt j = 0; j < n; j++) {
        for (OSQPInt k = P_csc->p[j]; k < P_csc->p[j+1]; k++) {
            OSQPInt i = P_csc->i[k];
            g_P_dns[i * n + j] = P_csc->x[k];
            if (i != j) {
                g_P_dns[j * n + i] = P_csc->x[k];  /* mirror */
            }
        }
    }

    /* Exponent arrays from Dinv/Einv */
    if (solver->settings->scaling && work->scaling) {
        const OSQPFloat* dinv_data = OSQPVectorf_data(work->scaling->Dinv);
        const OSQPFloat* einv_data = OSQPVectorf_data(work->scaling->Einv);
        g_dinv_exp = (int*)c_malloc(n * sizeof(int));
        g_einv_exp = (int*)c_malloc(m * sizeof(int));
        extract_scaling_exponents(dinv_data, n, g_dinv_exp);
        extract_scaling_exponents(einv_data, m, g_einv_exp);
    }

    g_dns_m = m;
    g_dns_n = n;
    g_last_A_x_ptr = work->data->A->csc->x;
    g_dns_initialized = 1;
}

#endif /* OSQP_ROUNDING_MODE */

/* =========[ Termination check debug info (exported for logging) ]========= */
OSQPFloat g_last_eps_prim = 0.0;
OSQPFloat g_last_eps_dual = 0.0;
OSQPInt   g_last_prim_check = -1;
OSQPInt   g_last_dual_check = -1;

/* =========[ FP32 exponent helpers for rho estimation ]========= */

/* Get unbiased exponent of |x| (IEEE-754 float):
 *  - x == 0      -> INT_MIN (ignored in max calculation)
 *  - subnormal   -> -126 (coarse approximation)
 *  - +Inf/NaN    -> INT_MAX (treated as very large)
 */
static inline int osqp_float_unbiased_exp_abs(OSQPFloat x) {
  union { float f; uint32_t u; } v = { (float)x };
  v.u &= 0x7FFFFFFFu;                   // |x|
  uint32_t e = (v.u >> 23) & 0xFFu;     // exponent(8 bits)
  uint32_t f =  v.u        & 0x7FFFFFu; // fraction(23 bits)
  if (e == 0) {
    if (f == 0) return INT_MIN;         // zero
    return -126;                        // subnormal
  }
  if (e == 0xFFu) return INT_MAX;       // Inf/NaN
  return (int)e - 127;                  // unbiased exponent
}

/* Exponent limits for rho adaptation */
#define EXP_MIN -14
#define EXP_MAX +14

/* Compute 2^k as float */
static inline OSQPFloat pow2_k(int k) {
  return scalbnf(1.0f, k);
}

/* Compute max exponent in a vector */
static inline int OSQPVectorf_max_exponent(const OSQPVectorf* v) {
  OSQPInt n = OSQPVectorf_length(v);
  const OSQPFloat* data = OSQPVectorf_data(v);
  int max_exp = INT_MIN;

  for (OSQPInt i = 0; i < n; ++i) {
    int exp = osqp_float_unbiased_exp_abs(data[i]);
    if (exp > max_exp) max_exp = exp;
  }

  return max_exp;
}

/* Compute coarse magnitude estimate from max exponent
 * Returns approximate log2(norm_inf) for hardware-friendly implementation
 */
static inline int compute_coarse_magnitude(const OSQPVectorf* v) {
  int exp = OSQPVectorf_max_exponent(v);

  // Clamp to valid range
  if (exp == INT_MIN) return -127;  // Zero vector
  if (exp == INT_MAX) return 127;   // Inf/NaN
  if (exp > EXP_MAX) return EXP_MAX;
  if (exp < EXP_MIN) return EXP_MIN;

  return exp;
}

/***********************************************************
* Auxiliary functions needed to compute ADMM iterations * *
***********************************************************/
#if OSQP_EMBEDDED_MODE != 1

OSQPFloat compute_rho_estimate(const OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Get primal and dual residuals (already computed)
  OSQPFloat prim_res = work->scaled_prim_res;
  OSQPFloat dual_res = work->scaled_dual_res;

  // ===== Hardware-friendly approach: Use exponents only =====
  // Original: rho_new = rho * sqrt(prim_res_norm / dual_res_norm)
  // Our approach: rho_new = rho * 2^((log2(prim) - log2(dual)) / 2)

  // Extract exponents from residuals themselves
  int exp_prim_res = osqp_float_unbiased_exp_abs(prim_res);
  int exp_dual_res = osqp_float_unbiased_exp_abs(dual_res);

  // Compute magnitude exponents for normalization terms
  int exp_z  = compute_coarse_magnitude(work->z);
  int exp_Ax = compute_coarse_magnitude(work->Ax);
  int exp_prim_norm = (exp_z > exp_Ax) ? exp_z : exp_Ax;

  int exp_q   = compute_coarse_magnitude(work->data->q);
  int exp_Aty = compute_coarse_magnitude(work->Aty);
  int exp_Px  = compute_coarse_magnitude(work->Px);
  int exp_dual_norm = exp_q;
  exp_dual_norm = (exp_Aty > exp_dual_norm) ? exp_Aty : exp_dual_norm;
  exp_dual_norm = (exp_Px > exp_dual_norm) ? exp_Px : exp_dual_norm;

  // Compute normalized exponents:
  // exp(prim_res / prim_norm) ≈ exp(prim_res) - exp(prim_norm)
  int exp_prim_normalized = exp_prim_res - exp_prim_norm;
  int exp_dual_normalized = exp_dual_res - exp_dual_norm;

  // Compute ratio exponent:
  // exp(prim_norm / dual_norm) ≈ exp(prim_norm) - exp(dual_norm)
  int exp_ratio = exp_prim_normalized - exp_dual_normalized;

  // Handle extreme cases
  if (exp_prim_res == INT_MIN || exp_dual_res == INT_MIN ||
      exp_prim_res == INT_MAX || exp_dual_res == INT_MAX) {
    // Residuals are 0, inf, or NaN - keep current rho
    return settings->rho;
  }

  // Halve the exponent (sqrt in log space)
  int exp_sqrt_ratio = exp_ratio >> 1;

  // Clamp to reasonable range [-14, +14] to avoid extreme rho values
  exp_sqrt_ratio = (exp_sqrt_ratio > EXP_MAX) ? EXP_MAX :
                   (exp_sqrt_ratio < -EXP_MAX) ? -EXP_MAX : exp_sqrt_ratio;

  // Compute 2^(exp_sqrt_ratio) as sqrt approximation
  OSQPFloat sqrt_approx = pow2_k(exp_sqrt_ratio);

  // Update rho estimate
  #ifdef OSQP_ROUNDING_MODE
    OSQPFloat rho_estimate = fp32_mul_conv(settings->rho, sqrt_approx);
    if (rho_estimate < OSQP_RHO_MIN) rho_estimate = OSQP_RHO_MIN;
    if (rho_estimate > OSQP_RHO_MAX) rho_estimate = OSQP_RHO_MAX;
  #else
    OSQPFloat rho_estimate = settings->rho * sqrt_approx;
    rho_estimate = c_min(c_max(rho_estimate, OSQP_RHO_MIN), OSQP_RHO_MAX);
  #endif



  return rho_estimate;
}

OSQPInt adapt_rho(OSQPSolver* solver) {

  OSQPInt   exitflag; // Exitflag
  OSQPFloat rho_new;  // New rho value

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;

  exitflag = 0;     // Initialize exitflag to 0

  // Compute new rho
  rho_new = compute_rho_estimate(solver);

  // Set rho estimate in info
  info->rho_estimate = rho_new;

  // Check if the new rho is large or small enough and update it in case
  #ifdef OSQP_ROUNDING_MODE
    OSQPFloat upper_threshold = fp32_mul_conv(settings->rho, settings->adaptive_rho_tolerance);
    OSQPFloat lower_threshold = settings->rho / settings->adaptive_rho_tolerance;
    OSQPInt should_update = (rho_new > upper_threshold) || (rho_new < lower_threshold);
  #else
    OSQPFloat upper_threshold = settings->rho * settings->adaptive_rho_tolerance;
    OSQPFloat lower_threshold = settings->rho / settings->adaptive_rho_tolerance;
    OSQPInt should_update = (rho_new > upper_threshold) || (rho_new < lower_threshold);
  #endif

    osqp_log_rho_debug(info->iter, "THRESHOLD_CHECK", should_update, 0.0, 0.0, rho_new, settings->rho);

    if (should_update) {
      OSQPFloat rho_old = settings->rho;
      osqp_log_rho_debug(info->iter, "UPDATING_RHO", 1, 0.0, 0.0, rho_new, rho_old);
      exitflag                 = osqp_update_rho(solver, rho_new);
      info->rho_updates += 1;
      solver->work->rho_updated = 1;
      osqp_log_rho_update(info->iter, rho_old, rho_new);
    }

  return exitflag;
}

OSQPInt set_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_types_changed = 0;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  settings->rho = c_min(c_max(settings->rho, OSQP_RHO_MIN), OSQP_RHO_MAX);

  constr_types_changed = OSQPVectorf_ew_bounds_type(work->constr_type,
                                                    work->data->l,
                                                    work->data->u,
                                                    OSQP_RHO_TOL,
                                                    OSQP_INFTY * OSQP_MIN_SCALING);


  //NB: Always refresh the complete rho vector, since the rho_vals
  //might be garbage if they have not been initialised yet.  This means
  //that there is some wasted effort in the case that the constraint types
  //haven't changed and the rho values are already correct, but such is life.
  OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                     work->constr_type,
                                     OSQP_RHO_MIN,                             //const  == -1
                                     settings->rho,                       //constr == 0
                                     OSQP_RHO_EQ_OVER_RHO_INEQ*settings->rho); //constr == 1

  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);

  return constr_types_changed;
}

OSQPInt update_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_type_changed;
  OSQPInt exitflag = 0;
  OSQPWorkspace* work = solver->work;

  //update rho_vec and see if anything changed
  constr_type_changed = set_rho_vec(solver);

  // Update rho_vec in KKT matrix if constraints type has changed
  if (constr_type_changed == 1) {
    exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, solver->settings->rho);
  }

  return exitflag;
}

#endif // OSQP_EMBEDDED_MODE != 1


void swap_vectors(OSQPVectorf** a,
                  OSQPVectorf** b) {
  OSQPVectorf* temp;

  temp = *b;
  *b   = *a;
  *a   = temp;
}

static void compute_rhs(OSQPSolver* solver) {

  OSQPWorkspace* work     = solver->work;
  OSQPSettings*  settings = solver->settings;

  //part related to x variables
  OSQPVectorf_add_scaled(work->xtilde_view,
                        settings->sigma,work->x_prev,
                        -1., work->data->q);

  //part related to dual variable in the equality constrained QP (nu)
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->ztilde_view, work->rho_inv_vec, work->y);
    OSQPVectorf_add_scaled(work->ztilde_view,
                          -1.0, work->ztilde_view,
                          1.0, work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled(work->ztilde_view,
                          1.0, work->z_prev,
                          -work->rho_inv, work->y);
  }
}

void update_xz_tilde(OSQPSolver* solver,
                     OSQPInt     admm_iter) {

  OSQPWorkspace* work = solver->work;

  // Compute right-hand side
  compute_rhs(solver);

  // Solve linear system
  work->linsys_solver->solve(work->linsys_solver, work->xz_tilde, admm_iter);
}

void update_x(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update x
  OSQPVectorf_add_scaled(work->x,
                        settings->alpha,work->xtilde_view,
                        (1.0 - settings->alpha),work->x_prev);

  // update delta_x
  OSQPVectorf_minus(work->delta_x,work->x,work->x_prev);
}

void update_z(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update z
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->z, work->rho_inv_vec,work->y);
    OSQPVectorf_add_scaled3(work->z,
                            1., work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled3(work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev,
                            work->rho_inv, work->y);
  }

  // project z onto C = [l,u]
  OSQPVectorf_ew_bound_vec(work->z, work->z, work->data->l, work->data->u);
}

void update_y(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update y
  OSQPVectorf_add_scaled3(work->delta_y,
                          settings->alpha, work->ztilde_view,
                          (1.0 - settings->alpha), work->z_prev,
                          -1.0, work->z);

  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->delta_y, work->delta_y, work->rho_vec);
  }
  else {
    OSQPVectorf_mult_scalar(work->delta_y, settings->rho);
  }

  OSQPVectorf_plus(work->y, work->y, work->delta_y);

}

void compute_obj_val_dual_gap(const OSQPSolver*  solver,
                              const OSQPVectorf* x,
                              const OSQPVectorf* y,
                                    OSQPFloat*   prim_obj_val,
                                    OSQPFloat*   dual_obj_val,
                                    OSQPFloat*   duality_gap) {
  OSQPFloat quad_term = 0.0;
  OSQPFloat lin_term  = 0.0;
  OSQPFloat sup_term  = 0.0;
  OSQPWorkspace* work = solver->work;

  /* NB: The function is always called after dual_res is computed */
  quad_term = OSQPVectorf_dot_prod(work->Px, x);
  lin_term  = OSQPVectorf_dot_prod(work->data->q, x);

  /* Compute the support function of the constraints, SC(y) = u'*max(y, 0) + l'*min(y, 0)
    by projecting y onto the polar of the recession cone of C=[l,u], then doing the dot products */
  OSQPVectorf_copy(work->z_prev, y);
  OSQPVectorf_project_polar_reccone(work->z_prev,
                                    work->data->l,
                                    work->data->u,
                                    OSQP_INFTY * OSQP_MIN_SCALING);

  // Round anything in the range [-OSQP_ZERO_DEADZONE, OSQP_ZERO_DEADZONE] to 0 to
  // prevent very small (i.e., 1e-20) values from blowing up the numerics.
  OSQPVectorf_round_to_zero(work->z_prev, OSQP_ZERO_DEADZONE);

  sup_term  = OSQPVectorf_dot_prod_signed(work->data->u, work->z_prev, +1);
  sup_term += OSQPVectorf_dot_prod_signed(work->data->l, work->z_prev, -1);

  /* Primal objective value is 0.5*x^T P x + q^T x */
  *prim_obj_val = 0.5 * quad_term + lin_term;

  /* Dual objective value is -0.5*x^T P x - SC(y)*/
  *dual_obj_val = -0.5 * quad_term - sup_term;

  /* Duality gap is x^T P x + q^T x + SC(y) */
  work->scaled_dual_gap = quad_term + lin_term + sup_term;

  if (solver->settings->scaling) {
    *prim_obj_val *= work->scaling->cinv;
    *dual_obj_val *= work->scaling->cinv;

    // We always store the duality gap in the info as unscaled (since it is for the user),
    // but we keep the scaled version to use as a termination check when requested.
    *duality_gap = work->scaling->cinv * work->scaled_dual_gap;
  } else {
    *duality_gap = work->scaled_dual_gap;
  }

  /* Save cost values for later use in termination tolerance computation */
  work->xtPx = quad_term;
  work->qtx  = lin_term;
  work->SC   = sup_term;
}

static OSQPFloat compute_duality_gap_tol(const OSQPSolver* solver,
                                               OSQPFloat   eps_abs,
                                               OSQPFloat   eps_rel) {
  OSQPFloat max_rel_eps = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  /* Compute max{ |x'*P*x|, |q'*x|, |SC(y)|} */
  max_rel_eps = c_absval(work->xtPx);                     /* |x'P*x| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->qtx));  /* |q'*x| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->SC));   /* |SC(y)| */

  /* Unscale the termination tolerance if required*/
  if (settings->scaling && !settings->scaled_termination) {
    max_rel_eps = work->scaling->cinv * max_rel_eps;
  }

  // eps_duality_gap
  return eps_abs + eps_rel * max_rel_eps;
}

static OSQPFloat compute_prim_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* z) {

  // NB: Use z_prev as working vector
  // pr = Ax - z

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat prim_res;

  #ifdef OSQP_ROUNDING_MODE
  {
    auxil_init_dense_matrices(solver);
    OSQPInt m = g_dns_m;
    OSQPInt n = g_dns_n;
    const OSQPFloat* x_data = OSQPVectorf_data(x);
    const OSQPFloat* z_data = OSQPVectorf_data(z);
    OSQPFloat* Ax_data = OSQPVectorf_data(work->Ax);
    OSQPFloat* pr_data = OSQPVectorf_data(work->z_prev);

    // Ax = A*x via 2-pass adder tree
    for (OSQPInt i = 0; i < m; i++)
      Ax_data[i] = two_pass_dot_179(&g_A_dns[i * n], x_data);

    // pr = Ax - z
    for (OSQPInt i = 0; i < m; i++)
      pr_data[i] = fp32_sub_conv(Ax_data[i], z_data[i]);

    // scaled_prim_res = ||pr||_inf
    OSQPFloat max_val = 0.0f;
    for (OSQPInt i = 0; i < m; i++)
      max_val = update_max_abs_conv(max_val, pr_data[i]);
    work->scaled_prim_res = max_val;

    // Unscale: prim_res = ||Einv * pr||_inf (Einv = 2^exp, so use exponent shift)
    if (settings->scaling && !settings->scaled_termination) {
      OSQPFloat max_scaled = 0.0f;
      for (OSQPInt i = 0; i < m; i++) {
        OSQPFloat scaled = exp_adjust_conv(pr_data[i], g_einv_exp[i]);
        max_scaled = update_max_abs_conv(max_scaled, scaled);
      }
      prim_res = max_scaled;
    } else {
      prim_res = work->scaled_prim_res;
    }
  }
  #else
    OSQPMatrix_Axpy(work->data->A,x,work->Ax, 1.0, 0.0); //Ax = A*x
    OSQPVectorf_minus(work->z_prev, work->Ax, z);

    work->scaled_prim_res = OSQPVectorf_norm_inf(work->z_prev);

    // If scaling active -> rescale residual
    if (settings->scaling && !settings->scaled_termination) {
      prim_res =  OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z_prev);
    }
    else{
      prim_res  = work->scaled_prim_res;
    }
  #endif

  return prim_res;
}

static OSQPFloat compute_prim_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  #ifdef OSQP_ROUNDING_MODE
  {
    OSQPInt m = g_dns_m;
    const OSQPFloat* z_data  = OSQPVectorf_data(work->z);
    const OSQPFloat* ax_data = OSQPVectorf_data(work->Ax);

    if (settings->scaling && !settings->scaled_termination) {
      // ||Einv * z|| (Einv = 2^exp, so use exponent shift)
      OSQPFloat max_einv_z = 0.0f;
      for (OSQPInt i = 0; i < m; i++)
        max_einv_z = update_max_abs_conv(max_einv_z, exp_adjust_conv(z_data[i], g_einv_exp[i]));

      // ||Einv * Ax||
      OSQPFloat max_einv_ax = 0.0f;
      for (OSQPInt i = 0; i < m; i++)
        max_einv_ax = update_max_abs_conv(max_einv_ax, exp_adjust_conv(ax_data[i], g_einv_exp[i]));

      max_rel_eps = update_max_abs_conv(max_einv_z, max_einv_ax);
    } else {
      // ||z||
      max_rel_eps = 0.0f;
      for (OSQPInt i = 0; i < m; i++)
        max_rel_eps = update_max_abs_conv(max_rel_eps, z_data[i]);

      // ||Ax||
      temp_rel_eps = 0.0f;
      for (OSQPInt i = 0; i < m; i++)
        temp_rel_eps = update_max_abs_conv(temp_rel_eps, ax_data[i]);

      max_rel_eps = update_max_abs_conv(max_rel_eps, temp_rel_eps);
    }

    return fp32_add_conv(eps_abs, fp32_mul_conv(eps_rel, max_rel_eps));
  }
  #else
    // max_rel_eps = max(||z||, ||A x||)
    if (settings->scaling && !settings->scaled_termination) {
      // ||Einv * z||
      max_rel_eps =
      OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z);

      // ||Einv * A * x||
      temp_rel_eps =
      OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->Ax);

      // Choose maximum
      max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
    }

    else { // No unscaling required
      // ||z||
      max_rel_eps = OSQPVectorf_norm_inf(work->z);

      // ||A * x||
      temp_rel_eps = OSQPVectorf_norm_inf(work->Ax);

      // Choose maximum
      max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
    }

    // eps_prim
    return eps_abs + eps_rel * max_rel_eps;
  #endif
}

static OSQPFloat compute_dual_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* y) {

  // NB: Use x_prev as temporary vector
  // NB: Only upper triangular part of P is stored.
  // dr = q + A'*y + P*x

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat      dual_res;

  // dr = q
  OSQPVectorf_copy(work->x_prev, work->data->q);

  #ifdef OSQP_ROUNDING_MODE
  {
    auxil_init_dense_matrices(solver);
    OSQPInt m = g_dns_m;
    OSQPInt n = g_dns_n;
    const OSQPFloat* x_data = OSQPVectorf_data(x);
    const OSQPFloat* y_data = OSQPVectorf_data(y);
    const OSQPFloat* q_data = OSQPVectorf_data(work->data->q);
    OSQPFloat* Px_data  = OSQPVectorf_data(work->Px);
    OSQPFloat* Aty_data = OSQPVectorf_data(work->Aty);
    OSQPFloat* dr_data  = OSQPVectorf_data(work->x_prev);

    // Px = P*x via 2-pass adder tree
    for (OSQPInt i = 0; i < n; i++)
      Px_data[i] = two_pass_dot_179(&g_P_dns[i * n], x_data);

    // Aty = A'*y via outer product accumulation
    if (work->data->m) {
      compute_Aty_asic(g_A_dns, m, n, y_data, Aty_data);
    } else {
      for (OSQPInt j = 0; j < n; j++) Aty_data[j] = 0.0f;
    }

    // dr = q + Px + Aty via 2-step PE
    compute_dual_res_pe(q_data, Px_data, Aty_data, n, dr_data);

    // scaled_dual_res = ||dr||_inf
    OSQPFloat max_val = 0.0f;
    for (OSQPInt j = 0; j < n; j++)
      max_val = update_max_abs_conv(max_val, dr_data[j]);
    work->scaled_dual_res = max_val;

    // Unscale: dual_res = cinv * ||Dinv * dr||_inf (Dinv = 2^exp, cinv = 1.0)
    if (settings->scaling && !settings->scaled_termination) {
      OSQPFloat max_scaled = 0.0f;
      for (OSQPInt j = 0; j < n; j++) {
        OSQPFloat scaled = exp_adjust_conv(dr_data[j], g_dinv_exp[j]);
        max_scaled = update_max_abs_conv(max_scaled, scaled);
      }
      dual_res = max_scaled;  // cinv = 1.0, no-op
    } else {
      dual_res = work->scaled_dual_res;
    }
  }
  #else

    // Px = P * x
    OSQPMatrix_Axpy(work->data->P, x, work->Px, 1.0, 0.0);

    // dr += Px
    OSQPVectorf_plus(work->x_prev, work->x_prev, work->Px);

    // dr += A' * y
    if (work->data->m) {
      OSQPMatrix_Atxpy(work->data->A, y, work->Aty, 1.0, 0.0);
      OSQPVectorf_plus(work->x_prev, work->x_prev, work->Aty);
    }

    work->scaled_dual_res = OSQPVectorf_norm_inf(work->x_prev);

    // If scaling active -> rescale residual
    if (settings->scaling && !settings->scaled_termination) {
      dual_res =  work->scaling->cinv * OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                                                    work->x_prev);
    }
    else {
      dual_res = work->scaled_dual_res;
    }
  #endif

  return dual_res;
}

static OSQPFloat compute_dual_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  #ifdef OSQP_ROUNDING_MODE
  {
    OSQPInt n = g_dns_n;
    const OSQPFloat* q_data   = OSQPVectorf_data(work->data->q);
    const OSQPFloat* aty_data = OSQPVectorf_data(work->Aty);
    const OSQPFloat* px_data  = OSQPVectorf_data(work->Px);

    if (settings->scaling && !settings->scaled_termination) {
      // ||Dinv * q|| (Dinv = 2^exp, so use exponent shift)
      max_rel_eps = 0.0f;
      for (OSQPInt j = 0; j < n; j++)
        max_rel_eps = update_max_abs_conv(max_rel_eps, exp_adjust_conv(q_data[j], g_dinv_exp[j]));

      // ||Dinv * A'y||
      temp_rel_eps = 0.0f;
      for (OSQPInt j = 0; j < n; j++)
        temp_rel_eps = update_max_abs_conv(temp_rel_eps, exp_adjust_conv(aty_data[j], g_dinv_exp[j]));
      max_rel_eps = update_max_abs_conv(max_rel_eps, temp_rel_eps);

      // ||Dinv * Px||
      temp_rel_eps = 0.0f;
      for (OSQPInt j = 0; j < n; j++)
        temp_rel_eps = update_max_abs_conv(temp_rel_eps, exp_adjust_conv(px_data[j], g_dinv_exp[j]));
      max_rel_eps = update_max_abs_conv(max_rel_eps, temp_rel_eps);

      // cinv = 1.0, no-op
    } else {
      // ||q||
      max_rel_eps = 0.0f;
      for (OSQPInt j = 0; j < n; j++)
        max_rel_eps = update_max_abs_conv(max_rel_eps, q_data[j]);

      // ||A'*y||
      temp_rel_eps = 0.0f;
      for (OSQPInt j = 0; j < n; j++)
        temp_rel_eps = update_max_abs_conv(temp_rel_eps, aty_data[j]);
      max_rel_eps = update_max_abs_conv(max_rel_eps, temp_rel_eps);

      // ||P*x||
      temp_rel_eps = 0.0f;
      for (OSQPInt j = 0; j < n; j++)
        temp_rel_eps = update_max_abs_conv(temp_rel_eps, px_data[j]);
      max_rel_eps = update_max_abs_conv(max_rel_eps, temp_rel_eps);
    }

    return fp32_add_conv(eps_abs, fp32_mul_conv(eps_rel, max_rel_eps));
  }
  #else

    // max_rel_eps = max(||q||, ||A' y|, ||P x||)
    if (settings->scaling && !settings->scaled_termination) {
      // || Dinv q||
      max_rel_eps =
      OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                  work->data->q);

      // || Dinv A' y ||
      temp_rel_eps =
      OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                  work->Aty);

      max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

      // || Dinv P x||
      temp_rel_eps =
      OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                  work->Px);

      max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

      // Multiply by cinv
      max_rel_eps *= work->scaling->cinv;
    } else { // No scaling required
      // ||q||
      max_rel_eps = OSQPVectorf_norm_inf(work->data->q);

      // ||A'*y||
      temp_rel_eps = OSQPVectorf_norm_inf(work->Aty);
      max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);

      // ||P*x||
      temp_rel_eps = OSQPVectorf_norm_inf(work->Px);
      max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
    }

    // eps_dual
    return eps_abs + eps_rel * max_rel_eps;

  #endif
}

OSQPInt is_primal_infeasible(OSQPSolver* solver,
                             OSQPFloat   eps_prim_inf) {

  /* This function checks the primal infeasibility criteria
     1) || A' * delta_y || < eps * ||delta_y||
     2) u'*max(delta_y, 0) + l'*min(delta_y, 0) < 0
   */

  OSQPFloat norm_delta_y;
  OSQPFloat ineq_lhs = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Project delta_y onto the polar of the recession cone of C=[l,u]
  OSQPVectorf_project_polar_reccone(work->delta_y,
                                    work->data->l,
                                    work->data->u,
                                    OSQP_INFTY * OSQP_MIN_SCALING);

  // Compute infinity norm of delta_y (unscale if necessary)
  if (settings->scaling && !settings->scaled_termination) {
    // Use work->Adelta_x as temporary vector
    OSQPVectorf_ew_prod(work->Adelta_x,
                        work->scaling->E,
                        work->delta_y);
    norm_delta_y = OSQPVectorf_norm_inf(work->Adelta_x);
  }
  else
    norm_delta_y = OSQPVectorf_norm_inf(work->delta_y);

  if (norm_delta_y > OSQP_DIVISION_TOL) {

    ineq_lhs  = OSQPVectorf_dot_prod_signed(work->data->u, work->delta_y, +1);
    ineq_lhs += OSQPVectorf_dot_prod_signed(work->data->l, work->delta_y, -1);

    /* Check if the condition is satisfied */
    if (ineq_lhs < 0.0) {
      OSQPMatrix_Atxpy(work->data->A, work->delta_y, work->Atdelta_y, 1.0, 0.0);

      // Unscale if necessary
      if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Atdelta_y,
                            work->Atdelta_y,
                            work->scaling->Dinv);
      }

      return OSQPVectorf_norm_inf(work->Atdelta_y) < eps_prim_inf * norm_delta_y;
    }
  }

  // Conditions not satisfied -> not primal infeasible
  return 0;
}

OSQPInt is_dual_infeasible(OSQPSolver* solver,
                           OSQPFloat   eps_dual_inf) {

  /* This function checks the dual infeasibility criteria
     1) q * delta_x < 0
     2) ||P * delta_x || < eps * || delta_x ||
     3) (A * delta_x)_i > -eps * || delta_x ||,    l_i != -inf
        (A * delta_x)_i <  eps * || delta_x ||,    u_i != +inf
   */

  OSQPFloat norm_delta_x;
  OSQPFloat cost_scaling;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  // Compute norm of delta_x
  if (settings->scaling && !settings->scaled_termination) { // Unscale if needed

    norm_delta_x =
    OSQPVectorf_scaled_norm_inf(work->scaling->D,
                                work->delta_x);
    cost_scaling = work->scaling->c;
  }
  else {
    norm_delta_x = OSQPVectorf_norm_inf(work->delta_x);
    cost_scaling = 1.0;
  }

  // Prevent 0 division || delta_x || > 0
  if (norm_delta_x > OSQP_DIVISION_TOL) {
    // Normalize delta_x by its norm

    /* vec_mult_scalar(work->delta_x, 1./norm_delta_x, work->data->n); */

    // Check first if q'*delta_x < 0
    if (OSQPVectorf_dot_prod(work->data->q, work->delta_x) < 0.0) {
      // Compute product P * delta_x
      OSQPMatrix_Axpy(work->data->P, work->delta_x, work->Pdelta_x, 1.0, 0.0);

      // Scale if necessary
      if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Pdelta_x,
                            work->Pdelta_x,
                            work->scaling->Dinv);
      }

      // Check if || P * delta_x || = 0
      if (OSQPVectorf_norm_inf(work->Pdelta_x) <
          cost_scaling * eps_dual_inf * norm_delta_x) {
        // Compute A * delta_x
        OSQPMatrix_Axpy(work->data->A, work->delta_x, work->Adelta_x,1.0,0.0);

        // Scale if necessary
        if (settings->scaling && !settings->scaled_termination) {
          OSQPVectorf_ew_prod(work->Adelta_x, work->Adelta_x, work->scaling->Einv);
        }

        // De Morgan Law Applied to dual infeasibility conditions for A * x
        // NB: Note that MIN_SCALING is used to adjust the infinity value
        // in case the problem is scaled.

        // If you get this far, then all tests passed, so return results from final test
        // Test whether Adelta_x is in the recession cone of C = [l, u]
        return OSQPVectorf_in_reccone(work->Adelta_x,
                                      work->data->l,
                                      work->data->u,
                                      OSQP_INFTY * OSQP_MIN_SCALING,
                                      eps_dual_inf * norm_delta_x);
      }
    }
  }

  // Conditions not satisfied -> not dual infeasible
  return 0;
}

OSQPInt has_solution(const OSQPInfo* info) {
  return ((info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_NON_CVX));
}

void store_solution(OSQPSolver *solver, OSQPSolution* solution) {

#ifndef OSQP_EMBEDDED_MODE
  OSQPFloat norm_vec;
#endif /* ifndef OSQP_EMBEDDED_MODE */

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  /* Bypass function if solution wasn't allocated */
  if (!solution)
    return;


  if (has_solution(info)) {
    // Unscale solution if scaling has been performed
      if (settings->scaling){
          //use x_prev and z_prev as scratch space
          unscale_solution(work->x_prev,work->z_prev, //unscaled solution
                           work->x,work->y,           //scaled solution
                           work);
          OSQPVectorf_to_raw(solution->x, work->x_prev); // primal
          OSQPVectorf_to_raw(solution->y, work->z_prev); // dual
      }
      else{
          OSQPVectorf_to_raw(solution->x, work->x); // primal
          OSQPVectorf_to_raw(solution->y, work->y); // dual
      }
      /* Set infeasibility certificates to NaN */
      OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
  }

  else {

    // No solution present. Solution is NaN
    OSQPVectorf_set_scalar(work->x, OSQP_NAN);
    OSQPVectorf_set_scalar(work->y, OSQP_NAN);
    OSQPVectorf_to_raw(solution->x, work->x); // primal
    OSQPVectorf_to_raw(solution->y, work->y); // dual

    // reset iterates to 0 for next run (they cannot start from NaN)
    osqp_cold_start(solver);


#ifndef OSQP_EMBEDDED_MODE

    // Normalize infeasibility certificates if embedded is off
    // NB: It requires a division
    if ((info->status_val == OSQP_PRIMAL_INFEASIBLE) ||
        ((info->status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_y);
      OSQPVectorf_mult_scalar(work->delta_y, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);

      /* Set dual infeasibility certificate to NaN */
      OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
    }

    if ((info->status_val == OSQP_DUAL_INFEASIBLE) ||
        ((info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_x);
      OSQPVectorf_mult_scalar(work->delta_x, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);

      /* Set primal infeasibility certificate to NaN */
      OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
    }

#endif /* ifndef OSQP_EMBEDDED_MODE */
  }
}

void update_info(OSQPSolver* solver,
                 OSQPInt     iter,
                 OSQPInt     polishing) {

  OSQPVectorf* x;
  OSQPVectorf* z;
  OSQPVectorf* y;                   // Allocate pointers to vectors

  // objective value, residuals
  OSQPFloat* prim_obj_val;
  OSQPFloat* dual_obj_val;
  OSQPFloat* dual_gap;
  OSQPFloat* prim_res;
  OSQPFloat* dual_res;

  OSQPInfo*      info     = solver->info;
  OSQPWorkspace* work     = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  OSQPFloat* run_time;                    // Execution time
#endif /* ifdef OSQP_ENABLE_PROFILING */

#ifndef OSQP_EMBEDDED_MODE

  if (polishing) {
    x            = work->pol->x;
    y            = work->pol->y;
    z            = work->pol->z;
    prim_obj_val = &work->pol->obj_val;
    dual_obj_val = &work->pol->dual_obj_val;
    dual_gap     = &work->pol->duality_gap;
    prim_res     = &work->pol->prim_res;
    dual_res     = &work->pol->dual_res;
# ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->polish_time;
# endif /* ifdef OSQP_ENABLE_PROFILING */
  }
  else {
#endif // OSQP_EMBEDDED_MODE
    x            = work->x;
    y            = work->y;
    z            = work->z;
    prim_obj_val = &info->obj_val;
    dual_obj_val = &info->dual_obj_val;
    dual_gap     = &info->duality_gap;
    prim_res     = &info->prim_res;
    dual_res     = &info->dual_res;
    info->iter   = iter;
#ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->solve_time;
#endif /* ifdef OSQP_ENABLE_PROFILING */
#ifndef OSQP_EMBEDDED_MODE
}

#endif /* ifndef OSQP_EMBEDDED_MODE */

  // Compute primal residual
  if (work->data->m == 0) {
    // No constraints -> Always primal feasible
    *prim_res = 0.;
  } else {
    *prim_res = compute_prim_res(solver, x, z);
  }

  // Compute dual residual; store P*x in work->Px
  *dual_res = compute_dual_res(solver, x, y);

  // Compute the objective and duality gap, store various temp values in work
  compute_obj_val_dual_gap(solver, x, y, prim_obj_val, dual_obj_val, dual_gap);

  // Log residuals for debugging
  osqp_log_residuals(info->iter, *prim_res, *dual_res, *dual_gap);

  // Compute the duality gap integral
  if (!polishing) {
    info->primdual_int += c_absval(*dual_gap);
  }

  // Update timing
#ifdef OSQP_ENABLE_PROFILING
  *run_time = osqp_toc(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */

  // Compute the relative KKT error
  info->rel_kkt_error = c_max( c_max(*dual_res, *prim_res), *dual_gap);

#ifdef OSQP_ENABLE_PRINTING
  work->summary_printed = 0; // The just updated info have not been printed
#endif /* ifdef OSQP_ENABLE_PRINTING */
}


void reset_info(OSQPInfo *info) {
#ifdef OSQP_ENABLE_PROFILING

  // Initialize info values.
  info->solve_time = 0.0;  // Solve time to zero
# ifndef OSQP_EMBEDDED_MODE
  info->polish_time = 0.0; // Polish time to zero
# endif /* ifndef OSQP_EMBEDDED_MODE */

  // NB: We do not reset the setup_time because it is performed only once
#endif /* ifdef OSQP_ENABLE_PROFILING */

  update_status(info, OSQP_UNSOLVED); // Problem is unsolved

#if OSQP_EMBEDDED_MODE != 1
  info->rho_updates = 0;              // Rho updates are now 0
#endif /* if OSQP_EMBEDDED_MODE != 1 */
}

const char *OSQP_STATUS_MESSAGE[] = {
  "",   // Status messages start from 1, so add a buffer
  "solved",
  "solved inaccurate",
  "primal infeasible",
  "primal infeasible inaccurate",
  "dual infeasible",
  "dual infeasible inaccurate",
  "maximum iterations reached",
  "run time limit reached",
  "problem non convex",
  "interrupted",
  "unsolved"
};

void update_status(OSQPInfo* info,
                   OSQPInt   status_val) {
  // Update status value
  info->status_val = status_val;

  // Update status string depending on status val
  c_strcpy(info->status, OSQP_STATUS_MESSAGE[status_val]);
}

OSQPInt check_termination(OSQPSolver* solver,
                          OSQPInt     approximate) {

  OSQPFloat eps_prim, eps_dual, eps_duality_gap, eps_prim_inf, eps_dual_inf;
  OSQPInt   exitflag;
  OSQPInt   prim_res_check, dual_res_check, duality_gap_check, prim_inf_check, dual_inf_check;
  OSQPFloat eps_abs, eps_rel;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Initialize variables to 0
  exitflag       = 0;
  prim_res_check = 0; dual_res_check = 0;
  prim_inf_check = 0; dual_inf_check = 0;
  duality_gap_check = 0;

  // Initialize tolerances
  eps_abs      = settings->eps_abs;
  eps_rel      = settings->eps_rel;
  eps_prim_inf = settings->eps_prim_inf;
  eps_dual_inf = settings->eps_dual_inf;

  // If residuals are too large, the problem is probably non convex
  if ((info->prim_res > OSQP_INFTY) ||
      (info->dual_res > OSQP_INFTY)){
    // Looks like residuals are diverging. Probably the problem is non convex!
    // Terminate and report it
    update_status(info, OSQP_NON_CVX);
    info->obj_val = OSQP_NAN;
    return 1;
  }

  // If approximate solution required, increase tolerances by 10
  if (approximate) {
    eps_abs      *= 8; // MODI 10
    eps_rel      *= 8; // MODI 10
    eps_prim_inf *= 8; // MODI 10
    eps_dual_inf *= 8; // MODI 10
  }

  // Check residuals
  if (work->data->m == 0) {
    prim_res_check = 1; // No constraints -> Primal feasibility always satisfied
  }
  else {
    // Compute primal tolerance
    eps_prim = compute_prim_tol(solver, eps_abs, eps_rel);

    // Primal feasibility check
    if (info->prim_res < eps_prim) {
      prim_res_check = 1;
    } else {
      // Primal infeasibility check
      prim_inf_check = is_primal_infeasible(solver, eps_prim_inf);
    }
  } // End check if m == 0

  // Compute dual tolerance
  eps_dual = compute_dual_tol(solver, eps_abs, eps_rel);

  // Dual feasibility check
  if (info->dual_res < eps_dual) {
    dual_res_check = 1;
  } else {
    // Check dual infeasibility
    dual_inf_check = is_dual_infeasible(solver, eps_dual_inf);
  }

  // Export termination check debug info
  g_last_eps_prim  = eps_prim;
  g_last_eps_dual  = eps_dual;
  g_last_prim_check = prim_res_check;
  g_last_dual_check = dual_res_check;

  if (settings->check_dualgap ) {
    // Compute duality gap tolerance
    eps_duality_gap = compute_duality_gap_tol(solver, eps_abs, eps_rel);

    // Duality gap check
    if (settings->scaling && !settings->scaled_termination) {
      // Use the unscaled duality gap value
      if (c_absval(info->duality_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    } else {
      // Use the scaled duality gap value
      if (c_absval(work->scaled_dual_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    }
  } else {
    // Force to 1 to bypass the check
    duality_gap_check = 1;
  }

  // Compare checks to determine solver status
  if (prim_res_check && dual_res_check && duality_gap_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_SOLVED_INACCURATE);
    } else {
      update_status(info, OSQP_SOLVED);
    }
    exitflag = 1;
  }
  else if (prim_inf_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_PRIMAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_PRIMAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
      // Update infeasibility certificate
      OSQPVectorf_ew_prod(work->delta_y,
                          work->delta_y,
                          work->scaling->E);
    }
    info->obj_val = OSQP_INFTY;
    exitflag            = 1;
  }
  else if (dual_inf_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_DUAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_DUAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
      // Update infeasibility certificate
      OSQPVectorf_ew_prod(work->delta_x,
                          work->delta_x,
                          work->scaling->D);
    }
    info->obj_val = -OSQP_INFTY;
    exitflag            = 1;
  }

  return exitflag;
}


#ifndef OSQP_EMBEDDED_MODE

OSQPInt validate_data(const OSQPCscMatrix* P,
                      const OSQPFloat*     q,
                      const OSQPCscMatrix* A,
                      const OSQPFloat*     l,
                      const OSQPFloat*     u,
                            OSQPInt        m,
                            OSQPInt        n) {
  OSQPInt j, ptr;

  if (!P) {
    c_eprint("Missing quadratic cost matrix P");
    return 1;
  }

  if (!A) {
    c_eprint("Missing constraint matrix A");
    return 1;
  }

  if (!q) {
    c_eprint("Missing linear cost vector q");
    return 1;
  }

  // General dimensions Tests
  if ((n <= 0) || (m < 0)) {
    c_eprint("n must be positive and m nonnegative; n = %i, m = %i",
             (int)n, (int)m);
    return 1;
  }

  // Matrix P
  if (P->m != n) {
    c_eprint("P does not have dimension n x n with n = %i", (int)n);
    return 1;
  }

  if (P->m != P->n) {
    c_eprint("P is not square");
    return 1;
  }

  for (j = 0; j < n; j++) { // COLUMN
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) {
      if (P->i[ptr] > j) {  // if ROW > COLUMN
        c_eprint("P is not upper triangular");
        return 1;
      }
    }
  }

  // Matrix A
  if ((A->m != m) || (A->n != n)) {
    c_eprint("A does not have dimension %i x %i", (int)m, (int)n);
    return 1;
  }

  // Lower and upper bounds
  for (j = 0; j < m; j++) {
    if (l[j] > u[j]) {
      c_eprint("Lower bound at index %d is greater than upper bound: %.4e > %.4e",
               (int)j, l[j], u[j]);
      return 1;
    }
  }

  return 0;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */


OSQPInt validate_linsys_solver(OSQPInt linsys_solver) {
  /* Verify the algebra backend supports the requested indirect solver */
  if ( (linsys_solver == OSQP_INDIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_INDIRECT_SOLVER) ) {
    return 0;
  }

  /* Verify the algebra backend supports the requested direct solver */
  if ( (linsys_solver == OSQP_DIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_DIRECT_SOLVER) ) {
    return 0;
  }

  // Invalid solver
  return 1;
}


OSQPInt validate_settings(const OSQPSettings* settings,
                          OSQPInt             from_setup) {

  if (!settings) {
    c_eprint("Missing settings!");
    return 1;
  }

  if (from_setup &&
      validate_linsys_solver(settings->linsys_solver)) {
    c_eprint("linsys_solver not recognized");
    return 1;
  }

  if (from_setup &&
      settings->allocate_solution != 0 &&
      settings->allocate_solution != 1) {
    c_eprint("allocate_solution must be either 0 or 1");
    return 1;
  }

  if (settings->verbose != 0 &&
      settings->verbose != 1) {
    c_eprint("verbose must be either 0 or 1");
    return 1;
  }

  if (settings->profiler_level != 0 &&
      settings->profiler_level != 1 &&
      settings->profiler_level != 2) {
    c_eprint("profiler_level must be either 0, 1 or 2");
    return 1;
  }

  if (settings->warm_starting != 0 &&
      settings->warm_starting != 1) {
    c_eprint("warm_start must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->scaling < 0) {
    c_eprint("scaling must be nonnegative");
    return 1;
  }

  if (settings->polishing != 0 &&
      settings->polishing != 1) {
    c_eprint("polishing must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->rho <= 0.0) {
    c_eprint("rho must be positive");
    return 1;
  }

  if (from_setup &&
      (settings->rho_is_vec != 0) &&
      (settings->rho_is_vec != 1)) {
    c_eprint("rho_is_vec must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->sigma <= 0.0) {
    c_eprint("sigma must be positive");
    return 1;
  }

  if (settings->alpha <= 0.0 ||
      settings->alpha >= 2.0) {
    c_eprint("alpha must be strictly between 0 and 2");
    return 1;
  }

  if (settings->cg_max_iter <= 0) {
    c_eprint("cg_max_iter must be positive");
    return 1;
  }

  if (settings->cg_tol_reduction <= 0) {
    c_eprint("cg_tol_reduction must be positive");
    return 1;
  }

  if (settings->cg_tol_fraction <= 0.0 ||
      settings->cg_tol_fraction >= 1.0) {
    c_eprint("cg_tol_fraction must be strictly between 0 and 1");
    return 1;
  }

  if (from_setup &&
      settings->adaptive_rho < 0 ||
      settings->adaptive_rho >= _OSQP_ADAPTIVE_RHO_UPDATE_LAST_VALUE) {
    c_eprint("adaptive_rho not a valid choice");
    return 1;
  }

#if !defined(OSQP_ENABLE_PROFILING)
  if (from_setup && settings->adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_TIME) {
    c_eprint("adaptive_rho time-based adaptation requires profiling to be enabled");
    return 1;
  }
#endif /* !defined(OSQP_ENABLE_PROFILING) */

  if (from_setup && settings->adaptive_rho_interval < 0) {
    c_eprint("adaptive_rho_interval must be nonnegative");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_fraction <= 0) {
    c_eprint("adaptive_rho_fraction must be positive");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_tolerance < 1.0) {
    c_eprint("adaptive_rho_tolerance must be >= 1");
    return 1;
  }

  if (settings->max_iter <= 0) {
    c_eprint("max_iter must be positive");
    return 1;
  }

  if (settings->eps_abs < 0.0) {
    c_eprint("eps_abs must be nonnegative");
    return 1;
  }

  if (settings->eps_rel < 0.0) {
    c_eprint("eps_rel must be nonnegative");
    return 1;
  }

  if (settings->eps_rel == 0.0 &&
      settings->eps_abs == 0.0) {
    c_eprint("at least one of eps_abs and eps_rel must be positive");
    return 1;
  }

  if (settings->eps_prim_inf <= 0.0) {
    c_eprint("eps_prim_inf must be positive");
    return 1;
  }

  if (settings->eps_dual_inf <= 0.0) {
    c_eprint("eps_dual_inf must be positive");
    return 1;
  }

  if (settings->scaled_termination != 0 &&
      settings->scaled_termination != 1) {
    c_eprint("scaled_termination must be either 0 or 1");
    return 1;
  }

  if (settings->check_termination < 0) {
    c_eprint("check_termination must be nonnegative");
    return 1;
  }

  if (settings->check_dualgap != 0 &&
      settings->check_dualgap != 1) {
    c_eprint("check_dualgap must be either 0 or 1");
    return 1;
  }

  if (settings->time_limit <= 0.0) {
    c_eprint("time_limit must be positive\n");
    return 1;
  }

  if (settings->delta <= 0.0) {
    c_eprint("delta must be positive");
    return 1;
  }

  if (settings->polish_refine_iter < 0) {
    c_eprint("polish_refine_iter must be nonnegative");
    return 1;
  }

  return 0;
}

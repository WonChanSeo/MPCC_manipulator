// #include "scaling.h"

// #if OSQP_EMBEDDED_MODE != 1


// // Set values lower than threshold SCALING_REG to 1

// OSQPFloat limit_scaling_scalar(OSQPFloat v) {
//     v = v < OSQP_MIN_SCALING ? 1.0 : v;
//     v = v > OSQP_MAX_SCALING ? OSQP_MAX_SCALING : v;
//     return v;
// }

// void limit_scaling_vector(OSQPVectorf* v) {
//   OSQPVectorf_set_scalar_if_lt(v,v,OSQP_MIN_SCALING,1.0);
//   OSQPVectorf_set_scalar_if_gt(v,v,OSQP_MAX_SCALING,OSQP_MAX_SCALING);
// }

// /**
//  * Compute infinite norm of the columns of the KKT matrix without forming it
//  *
//  * The norm is stored in the vector v = (D, E)
//  *
//  * @param P        Cost matrix
//  * @param A        Constraints matrix
//  * @param D        Norm of columns related to variables
//  * @param D_temp_A Temporary vector for norm of columns of A
//  * @param E        Norm of columns related to constraints
//  */
// void compute_inf_norm_cols_KKT(const OSQPMatrix*  P,
//                                const OSQPMatrix*  A,
//                                      OSQPVectorf* D,
//                                      OSQPVectorf* D_temp_A,
//                                      OSQPVectorf* E) {
//   // First half
//   //  [ P ]
//   //  [ A ]
//   OSQPMatrix_col_norm_inf(P,D);
//   OSQPMatrix_col_norm_inf(A, D_temp_A);
//   OSQPVectorf_ew_max_vec(D, D_temp_A, D);

//   // Second half
//   //  [ A']
//   //  [ 0 ]
//   OSQPMatrix_row_norm_inf(A,E);
// }

// OSQPInt scale_data(OSQPSolver* solver) {
//   // Scale KKT matrix
//   //
//   //    [ P   A']
//   //    [ A   0 ]
//   //
//   // with diagonal matrix
//   //
//   //  S = [ D    ]
//   //      [    E ]
//   //

//   OSQPInt   i;          // Iterations index
//   OSQPInt   n;          // Number of variables
//   OSQPFloat c_temp;     // Objective function scaling
//   OSQPFloat inf_norm_q; // Infinity norm of q

//   OSQPSettings*  settings = solver->settings;
//   OSQPWorkspace* work     = solver->work;

//   n = work->data->n;

//   // Initialize scaling to 1
//   work->scaling->c = 1.0;
//   OSQPVectorf_set_scalar(work->scaling->D,    1.);
//   OSQPVectorf_set_scalar(work->scaling->Dinv, 1.);
//   OSQPVectorf_set_scalar(work->scaling->E,    1.);
//   OSQPVectorf_set_scalar(work->scaling->Einv, 1.);


//   for (i = 0; i < settings->scaling; i++) {
//     //
//     // First Ruiz step
//     //

//     // Compute norm of KKT columns
//     compute_inf_norm_cols_KKT(work->data->P, work->data->A,
//                               work->D_temp,
//                               work->D_temp_A,
//                               work->E_temp);

//     // Set to 1 values with 0 norms (avoid crazy scaling)
//     limit_scaling_vector(work->D_temp);
//     limit_scaling_vector(work->E_temp);

//     // Take square root of norms
//     OSQPVectorf_ew_sqrt(work->D_temp);
//     OSQPVectorf_ew_sqrt(work->E_temp);

//     // Copy inverses of D/E over themselves
//     OSQPVectorf_ew_reciprocal(work->D_temp, work->D_temp);
//     OSQPVectorf_ew_reciprocal(work->E_temp, work->E_temp);

//     // Equilibrate matrices P and A and vector q
//     // P <- DPD
//     OSQPMatrix_lmult_diag(work->data->P,work->D_temp);
//     OSQPMatrix_rmult_diag(work->data->P,work->D_temp);

//     // A <- EAD
//     OSQPMatrix_lmult_diag(work->data->A,work->E_temp);
//     OSQPMatrix_rmult_diag(work->data->A,work->D_temp);

//     // q <- Dq
//     OSQPVectorf_ew_prod(work->data->q, work->data->q, work->D_temp);

//     // Update equilibration matrices D and E
//     OSQPVectorf_ew_prod(work->scaling->D, work->scaling->D, work->D_temp);
//     OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, work->E_temp);

//     //
//     // Cost normalization step
//     //

//     // Compute avg norm of cols of P.
//     OSQPMatrix_col_norm_inf(work->data->P, work->D_temp);
//     c_temp = OSQPVectorf_norm_1(work->D_temp);
//     c_temp = c_temp / n;

//     // Compute inf norm of q
//     inf_norm_q = OSQPVectorf_norm_inf(work->data->q);

//     // If norm_q == 0, set it to 1 (ignore it in the scaling)
//     // NB: Using the same function as with vectors here
//     inf_norm_q = limit_scaling_scalar(inf_norm_q);

//     // Compute max between avg norm of cols of P and inf norm of q
//     c_temp = c_max(c_temp, inf_norm_q);

//     // Limit scaling (use same function as with vectors)
//     c_temp = limit_scaling_scalar(c_temp);

//     // Invert scaling c = 1 / cost_measure
//     c_temp = 1. / c_temp;

//     // Scale P
//     OSQPMatrix_mult_scalar(work->data->P,c_temp);

//     // Scale q
//     OSQPVectorf_mult_scalar(work->data->q, c_temp);

//     // Update cost scaling
//     work->scaling->c *= c_temp;
//   }


//   // Store cinv, Dinv, Einv
//   work->scaling->cinv = 1. / work->scaling->c;
//   OSQPVectorf_ew_reciprocal(work->scaling->Dinv, work->scaling->D);
//   OSQPVectorf_ew_reciprocal(work->scaling->Einv, work->scaling->E);


//   // Scale problem vectors l, u
//   OSQPVectorf_ew_prod(work->data->l, work->data->l, work->scaling->E);
//   OSQPVectorf_ew_prod(work->data->u, work->data->u, work->scaling->E);

//   return 0;
// }

// #endif /* if OSQP_EMBEDDED_MODE != 1 */


// OSQPInt unscale_data(OSQPSolver* solver) {

//   OSQPWorkspace* work     = solver->work;

//   // Unscale cost
//   OSQPMatrix_mult_scalar(work->data->P, work->scaling->cinv);
//   OSQPMatrix_lmult_diag(work->data->P,  work->scaling->Dinv);
//   OSQPMatrix_rmult_diag(work->data->P,  work->scaling->Dinv);
//   OSQPVectorf_mult_scalar(work->data->q,work->scaling->cinv);
//   OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->Dinv);

//   // Unscale constraints
//   OSQPMatrix_lmult_diag(work->data->A,work->scaling->Einv);
//   OSQPMatrix_rmult_diag(work->data->A,work->scaling->Dinv);

//   OSQPVectorf_ew_prod(work->data->l,
//                       work->data->l,
//                       work->scaling->Einv);
//   OSQPVectorf_ew_prod(work->data->u,
//                       work->data->u,
//                       work->scaling->Einv);

//   return 0;
// }

// OSQPInt unscale_solution(OSQPVectorf*       usolx,
//                          OSQPVectorf*       usoly,
//                          const OSQPVectorf* solx,
//                          const OSQPVectorf* soly,
//                          OSQPWorkspace*     work) {

//   // primal
//   OSQPVectorf_ew_prod(usolx,solx,work->scaling->D);

//   // dual
//   OSQPVectorf_ew_prod(usoly,soly,work->scaling->E);

//   OSQPVectorf_mult_scalar(usoly,work->scaling->cinv);
//   return 0;
// }

#include "scaling.h"
#include <limits.h>
#include <stdint.h>
#include "algebra_vector.h"

// ===== 통일된 벡터 읽기 접근자 =====
// ---- 통일된 벡터 읽기 접근자 ----
// (builtin 백엔드 기준: 함수 이름은 algebra_vector.h에 선언돼 있음)
// 통일된 읽기 접근자
#define VEC_LEN(v)        OSQPVectorf_length((v))
#define VEC_PTR(v)        OSQPVectorf_data((v))      // ← data_const 말고 data
#define VEC_GET(v,i)      (VEC_PTR((v))[(i)])

static OSQPInt check_bounds(const OSQPVectorf* l, const OSQPVectorf* u) {
  OSQPInt m = OSQPVectorf_length(l);
  for (OSQPInt i=0;i<m;++i) if (VEC_GET(l,i) > VEC_GET(u,i)) return i+1;
  return 0;
}


/* =========[ FP32 exponent helpers ]========= */

/* |x|의 unbiased exponent(IEEE-754):
 *  - x == 0      -> INT_MIN (max 계산에서 무시)
 *  - subnormal   -> -126 (coarse)
 *  - +Inf/NaN    -> INT_MAX (아주 큼으로 간주)
 */
static inline int osqp_float_unbiased_exp_abs(OSQPFloat x) {
  union { float f; uint32_t u; } v = { (float)x };
  v.u &= 0x7FFFFFFFu;                   // |x|
  uint32_t e = (v.u >> 23) & 0xFFu;     // exponent(8)
  uint32_t f =  v.u        & 0x7FFFFFu; // fraction(23)
  if (e == 0) {
    if (f == 0) return INT_MIN;         // zero
    return -126;                        // subnormal
  }
  if (e == 0xFFu) return INT_MAX;       // Inf/NaN
  return (int)e - 127;                  // unbiased
}

/* 2^{-k} as float */
static inline OSQPFloat pow2_neg_k_int(int k) { return scalbnf(1.0f, -k); }

/* =========[ Exponent from norms (KKT-aware) ]========= */

/* e_col[j] = max( exp(||P(:,j)||_inf), exp(||A(:,j)||_inf) ) */
static void col_expmax_KKT_using_norms(const OSQPMatrix* P,
                                       const OSQPMatrix* A,
                                       OSQPVectorf* Pcol_inf,
                                       OSQPVectorf* Acol_inf,
                                       int* e_col) {
  OSQPMatrix_col_norm_inf(P, Pcol_inf);
  OSQPMatrix_col_norm_inf(A, Acol_inf);

  const OSQPInt n = VEC_LEN(Pcol_inf);
  const OSQPFloat* pc = VEC_PTR(Pcol_inf);   // ★ data()
  const OSQPFloat* ac = VEC_PTR(Acol_inf);   // ★ data()

  for (OSQPInt j = 0; j < n; ++j) {
    int eP = osqp_float_unbiased_exp_abs(pc[j]);
    int eA = osqp_float_unbiased_exp_abs(ac[j]);
    e_col[j] = (eP > eA ? eP : eA);
  }
}

static void row_expmax_A_using_norms(const OSQPMatrix* A,
                                     OSQPVectorf* Arow_inf,
                                     int* e_row) {
  OSQPMatrix_row_norm_inf(A, Arow_inf);

  const OSQPInt m = VEC_LEN(Arow_inf);
  const OSQPFloat* ar = VEC_PTR(Arow_inf);   // ★ data()

  for (OSQPInt i = 0; i < m; ++i)
    e_row[i] = osqp_float_unbiased_exp_abs(ar[i]);
}


/* =========[ SCALE ]========= */

OSQPInt scale_data(OSQPSolver* solver) {
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  const OSQPInt n = work->data->n;
  const OSQPInt m = work->data->m;

  /* 0) 누적 스케일 초기화 */
  work->scaling->c = (OSQPFloat)1.0;
  OSQPVectorf_set_scalar(work->scaling->D,    1.0);
  OSQPVectorf_set_scalar(work->scaling->Dinv, 1.0);
  OSQPVectorf_set_scalar(work->scaling->E,    1.0);
  OSQPVectorf_set_scalar(work->scaling->Einv, 1.0);

  /* 버퍼 */
  int*       kD    = (int*)       c_malloc(sizeof(int)       * (n ? n : 1));
  int*       kE    = (int*)       c_malloc(sizeof(int)       * (m ? m : 1));
  OSQPFloat* Darr  = (OSQPFloat*) c_malloc(sizeof(OSQPFloat) * (n ? n : 1));
  OSQPFloat* Earr  = (OSQPFloat*) c_malloc(sizeof(OSQPFloat) * (m ? m : 1));
  int*       e_col = (int*)       c_malloc(sizeof(int)       * (n ? n : 1));
  int*       e_row = (int*)       c_malloc(sizeof(int)       * (m ? m : 1));
  if (!kD || !kE || !Darr || !Earr || !e_col || !e_row) return OSQP_MEM_ALLOC_ERROR;

  for (OSQPInt j = 0; j < n; ++j) kD[j] = 0;
  for (OSQPInt i = 0; i < m; ++i) kE[i] = 0;

  /* OSQP 스크래치 벡터 재사용 */
  OSQPVectorf* Pcol_inf = work->D_temp;    /* size n */
  OSQPVectorf* Acol_inf = work->D_temp_A;  /* size n */
  OSQPVectorf* Arow_inf = work->E_temp;    /* size m */

  // const OSQPInt T = (settings->scaling > 0) ? settings->scaling : 2;
  const OSQPInt T = 100;
  const int EXP_MIN = -13, EXP_MAX = +13;

  /* ==================== 여기부터 루프 전체 교체 ==================== */
  for (OSQPInt t = 0; t < T; ++t) {
    /* 1) 열 패스: KKT 열 최대 exponent */
    if (n > 0) {
      col_expmax_KKT_using_norms(work->data->P, work->data->A,
                                 Pcol_inf, Acol_inf, e_col);

      /* ★ 누적이 아니라 '증분' Δk만 이번 스텝에 곱한다 */
      for (OSQPInt j = 0; j < n; ++j) {
        const int emax  = e_col[j];
        const int add   = (emax == INT_MIN || emax <= 0) ? 0 : emax;
        const int prev  = kD[j];
        int cum         = prev + add;
        if (cum < EXP_MIN) cum = EXP_MIN;
        if (cum > EXP_MAX) cum = EXP_MAX;
        const int delta  = cum - prev;           /* ★ Δk */
        kD[j]   = cum;
        Darr[j] = pow2_neg_k_int(delta);         /* ★ Dstep = 2^{-Δk} */
      }

      OSQPVectorf* Dstep = OSQPVectorf_new(Darr, n);
      if (!Dstep) return OSQP_MEM_ALLOC_ERROR;

      /* ★ P ← Dstep * P * Dstep  (합동 스케일: PSD/대칭 보존) */
      OSQPMatrix_lmult_diag(work->data->P, Dstep);
      OSQPMatrix_rmult_diag(work->data->P, Dstep);

      /* A, q는 기존처럼 우측/원소곱 */
      OSQPMatrix_rmult_diag(work->data->A, Dstep);
      OSQPVectorf_ew_prod(work->data->q, work->data->q, Dstep);

      /* 누적 기록은 곱으로 */
      OSQPVectorf_ew_prod(work->scaling->D, work->scaling->D, Dstep);
      OSQPVectorf_free(Dstep);
    }

    /* 2) 행 패스: A 행 최대 exponent */
    if (m > 0) {
      row_expmax_A_using_norms(work->data->A, Arow_inf, e_row);

      /* ★ 누적이 아니라 '증분' Δk만 이번 스텝에 곱한다 */
      for (OSQPInt i = 0; i < m; ++i) {
        const int emax  = e_row[i];
        const int add   = (emax == INT_MIN || emax <= 0) ? 0 : emax;
        const int prev  = kE[i];
        int cum         = prev + add;
        if (cum < EXP_MIN) cum = EXP_MIN;
        if (cum > EXP_MAX) cum = EXP_MAX;
        const int delta  = cum - prev;           /* ★ Δk */
        kE[i]   = cum;
        Earr[i] = pow2_neg_k_int(delta);         /* ★ Estep = 2^{-Δk} */
      }

      OSQPVectorf* Estep = OSQPVectorf_new(Earr, m);
      if (!Estep) return OSQP_MEM_ALLOC_ERROR;

      /* A ← Estep * A,  l,u ← Estep ⊙ (l,u) */
      OSQPMatrix_lmult_diag(work->data->A, Estep);
      OSQPVectorf_ew_prod(work->data->l, work->data->l, Estep);
      OSQPVectorf_ew_prod(work->data->u, work->data->u, Estep);

      /* ★★★ bounds 무결성 즉시 체크 (E-step 직후) ★★★ */
      {
        OSQPInt bad = check_bounds(work->data->l, work->data->u);
        if (bad) {
          printf("[scaling] invalid bounds at i=%d: l=%g > u=%g\n",
                 (int)bad-1,
                 VEC_GET(work->data->l, bad-1),
                 VEC_GET(work->data->u, bad-1));
          OSQPVectorf_free(Estep);
          return OSQP_DATA_VALIDATION_ERROR;
        }
      }

      /* 누적 기록은 곱으로 */
      OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, Estep);
      OSQPVectorf_free(Estep);
    }
  }
  /* ==================== 여기까지 루프 전체 교체 ==================== */

  /* 루프 종료 후 한 번 더 bounds 체크 (안전망) */
  {
    OSQPInt bad = check_bounds(work->data->l, work->data->u);
    if (bad) return OSQP_DATA_VALIDATION_ERROR;
  }

  /* 3) 비용 정규화 (AP2 스칼라) — 기존 그대로 사용 가능 */
  OSQPMatrix_col_norm_inf(work->data->P, Pcol_inf);
  OSQPFloat Pcol_inf_max = (n > 0) ? OSQPVectorf_norm_inf(Pcol_inf) : 0.0f;
  int eP = osqp_float_unbiased_exp_abs(Pcol_inf_max);

  OSQPFloat qinf = (n > 0) ? OSQPVectorf_norm_inf(work->data->q) : 0.0f;
  int eq = osqp_float_unbiased_exp_abs(qinf);

  int ecost = (eP > eq ? eP : eq);
  int kc = (ecost == INT_MIN || ecost <= 0) ? 0 : ecost;
  if (kc < EXP_MIN) kc = EXP_MIN;
  if (kc > EXP_MAX) kc = EXP_MAX;

  OSQPFloat c = pow2_neg_k_int(kc); /* 2^{-kc} */
  if (kc != 0 && n > 0) {
    OSQPMatrix_mult_scalar(work->data->P, c);
    OSQPVectorf_mult_scalar(work->data->q, c);
  }
  work->scaling->c *= c;

  /* 4) 역스케일 준비 */
  work->scaling->cinv = 1.0f / work->scaling->c;
  OSQPVectorf_ew_reciprocal(work->scaling->Dinv, work->scaling->D);
  OSQPVectorf_ew_reciprocal(work->scaling->Einv, work->scaling->E);

  c_free(kD); c_free(kE); c_free(Darr); c_free(Earr); c_free(e_col); c_free(e_row);
  return 0;
}


/* =========[ UNSCALE ]========= */

OSQPInt unscale_data(OSQPSolver* solver) {
  OSQPWorkspace* work = solver->work;

  /* 목적함수 역스케일 */
  OSQPMatrix_mult_scalar(work->data->P, work->scaling->cinv);
  OSQPMatrix_lmult_diag(work->data->P,  work->scaling->Dinv);
  OSQPMatrix_rmult_diag(work->data->P,  work->scaling->Dinv);

  OSQPVectorf_mult_scalar(work->data->q, work->scaling->cinv);
  OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->Dinv);

  /* 제약 역스케일 */
  OSQPMatrix_lmult_diag(work->data->A, work->scaling->Einv);
  OSQPMatrix_rmult_diag(work->data->A, work->scaling->Dinv);

  OSQPVectorf_ew_prod(work->data->l, work->data->l, work->scaling->Einv);
  OSQPVectorf_ew_prod(work->data->u, work->data->u, work->scaling->Einv);

  return 0;
}

OSQPInt unscale_solution(OSQPVectorf*       usolx,
                         OSQPVectorf*       usoly,
                         const OSQPVectorf* solx,
                         const OSQPVectorf* soly,
                         OSQPWorkspace*     work) {
  /* x_orig = D .* x_scaled */
  OSQPVectorf_ew_prod(usolx, solx, work->scaling->D);

  /* y_orig = (E .* y_scaled) / c */
  OSQPVectorf_ew_prod(usoly, soly, work->scaling->E);
  OSQPVectorf_mult_scalar(usoly, work->scaling->cinv);
  return 0;
}

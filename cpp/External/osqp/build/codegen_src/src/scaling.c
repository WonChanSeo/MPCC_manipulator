// #include "scaling.h"
// #include <limits.h>
// #include <stdint.h>
// #include "algebra_vector.h"

// // ===== 통일된 벡터 읽기 접근자 =====
// // ---- 통일된 벡터 읽기 접근자 ----
// // (builtin 백엔드 기준: 함수 이름은 algebra_vector.h에 선언돼 있음)
// // 통일된 읽기 접근자
// #define VEC_LEN(v)        OSQPVectorf_length((v))
// #define VEC_PTR(v)        OSQPVectorf_data((v))      // ← data_const 말고 data
// #define VEC_GET(v,i)      (VEC_PTR((v))[(i)])
// #define EXP_MIN -14
// #define EXP_MAX +14

// static OSQPInt check_bounds(const OSQPVectorf* l, const OSQPVectorf* u) {
//   OSQPInt m = OSQPVectorf_length(l);
//   for (OSQPInt i=0;i<m;++i) if (VEC_GET(l,i) > VEC_GET(u,i)) return i+1;
//   return 0;
// }

// OSQPFloat limit_scaling_scalar(OSQPFloat v) {
//     v = v < OSQP_MIN_SCALING ? 1.0 : v;
//     v = v > OSQP_MAX_SCALING ? OSQP_MAX_SCALING : v;
//     return v;
// }


// /* =========[ FP32 exponent helpers ]========= */

// /* |x|의 unbiased exponent(IEEE-754):
//  *  - x == 0      -> INT_MIN (max 계산에서 무시)
//  *  - subnormal   -> -126 (coarse)
//  *  - +Inf/NaN    -> INT_MAX (아주 큼으로 간주)
//  */
// static inline int osqp_float_unbiased_exp_abs(OSQPFloat x) {
//   union { float f; uint32_t u; } v = { (float)x };
//   v.u &= 0x7FFFFFFFu;                   // |x|
//   uint32_t e = (v.u >> 23) & 0xFFu;     // exponent(8)
//   uint32_t f =  v.u        & 0x7FFFFFu; // fraction(23)
//   if (e == 0) {
//     if (f == 0) return INT_MIN;         // zero
//     return -126;                        // subnormal
//   }
//   if (e == 0xFFu) return INT_MAX;       // Inf/NaN
//   return (int)e - 127;                  // unbiased
// }

// /* 2^{-k} as float */
// static inline OSQPFloat pow2_k_int(int k) { 
//   // printf("2^(%d) = %e\n", k, scalbnf(1.0f, -k));
//   return scalbnf(1.0f, k); }

// /* =========[ Exponent from norms (KKT-aware) ]========= */

// static void OSQPMatrix_col_norm_inf_by_exponent(const OSQPMatrix* M,
//                                            OSQPVectorf*    Mcol_inf) {
//   OSQPMatrix_col_norm_inf(M, Mcol_inf);

//   const OSQPInt n = VEC_LEN(Mcol_inf);
//   const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data
//   int e_col_tmp;

//   for (OSQPInt j = 0; j < n; ++j) {
//     e_col_tmp = osqp_float_unbiased_exp_abs(mc[j]) > EXP_MAX ? EXP_MAX :
//                osqp_float_unbiased_exp_abs(mc[j]) < EXP_MIN ? 0 :
//                osqp_float_unbiased_exp_abs(mc[j]);
//     VEC_PTR(Mcol_inf)[j] = pow2_k_int(e_col_tmp); // 2^{-k
//   }
// }

// static inline OSQPFloat OSQPMatrix_col_norm_inf_avg_by_exponent(const OSQPMatrix* M) {
  
//   OSQPVectorf* Mcol_inf = OSQPVectorf_malloc(179);
//   if (!Mcol_inf) { /* OOM 처리 */ }

//   /* 필요하면 0으로 초기화 */
//   OSQPVectorf_set_scalar(Mcol_inf, (OSQPFloat)0.0);
  
//   OSQPMatrix_col_norm_inf(M, Mcol_inf);

//   const OSQPInt n = VEC_LEN(Mcol_inf);
//   const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data
  
//   int e_col_tmp = 0;

//   for (OSQPInt j = 0; j < n; ++j) {
//     e_col_tmp += osqp_float_unbiased_exp_abs(mc[j]);
//   }

//   OSQPVectorf_free(Mcol_inf);

//   return pow2_k_int(e_col_tmp >> 9); // 2^{-k
// }

// static void OSQPMatrix_col_norm_inf_by_exponent_half(const OSQPMatrix* M,
//                                            OSQPVectorf*    Mcol_inf) {
//   OSQPMatrix_col_norm_inf(M, Mcol_inf);

//   const OSQPInt n = VEC_LEN(Mcol_inf);
//   const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data
//   int e_col_tmp;

//   for (OSQPInt j = 0; j < n; ++j) {
//     e_col_tmp = osqp_float_unbiased_exp_abs(mc[j]) > EXP_MAX ? EXP_MAX :
//                osqp_float_unbiased_exp_abs(mc[j]) < EXP_MIN ? 0 :
//                osqp_float_unbiased_exp_abs(mc[j]);
//     // printf("Column %d: norm_inf = %e, exp = %d, half exp = %d, scaling = %e\n",
//     //        j, mc[j], e_col_tmp, e_col_tmp >> 1, pow2_k_int(e_col_tmp >> 1));
//     VEC_PTR(Mcol_inf)[j] = pow2_k_int(e_col_tmp >> 1); // 2^{-k
//   }
// }

// static void OSQPMatrix_row_norm_inf_by_exponent_half(const OSQPMatrix* A,
//                                      OSQPVectorf* Arow_inf) {
//   OSQPMatrix_row_norm_inf(A, Arow_inf);

//   const OSQPInt m = VEC_LEN(Arow_inf);
//   const OSQPFloat* ar = VEC_PTR(Arow_inf);   // ★ data()

//   int e_row_tmp;

//   for (OSQPInt i = 0; i < m; ++i) {
//     e_row_tmp = osqp_float_unbiased_exp_abs(ar[i]) > EXP_MAX ? EXP_MAX :
//                osqp_float_unbiased_exp_abs(ar[i]) < EXP_MIN ? 0 :
//                osqp_float_unbiased_exp_abs(ar[i]);
//     VEC_PTR(Arow_inf)[i] = pow2_k_int(e_row_tmp >> 1); // 2^{-k
//   }
// }

// /* e_col[j] = max( exp(||P(:,j)||_inf), exp(||A(:,j)||_inf) ) */
// static void col_expmax_KKT_using_norms(const OSQPMatrix* P,
//                                         const OSQPMatrix* A,
//                                         OSQPVectorf* D,
//                                         OSQPVectorf* D_temp_A,
//                                         OSQPVectorf* E) {
//   OSQPMatrix_col_norm_inf_by_exponent_half(P,D);
//   OSQPMatrix_col_norm_inf_by_exponent_half(A, D_temp_A);
//   OSQPVectorf_ew_max_vec(D, D_temp_A, D); // D = max(D, D_temp_A)

//   OSQPMatrix_row_norm_inf_by_exponent_half(A, E);
// }

// static void col_expmax_KKT_using_norms_half(const OSQPMatrix* M,
//                                        OSQPVectorf* Mcol_inf) {
//   OSQPMatrix_col_norm_inf(M, Mcol_inf);

//   const OSQPInt n = VEC_LEN(Mcol_inf);
//   const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data()

//   int e_col_tmp;

//   for (OSQPInt j = 0; j < n; ++j) {
//     e_col_tmp = osqp_float_unbiased_exp_abs(mc[j]) > EXP_MAX ? EXP_MAX :
//                osqp_float_unbiased_exp_abs(mc[j]) < EXP_MIN ? 0 :
//                osqp_float_unbiased_exp_abs(mc[j]);
//     VEC_PTR(Mcol_inf)[j] = pow2_k_int(e_col_tmp >> 1); // 2^{-k}
//   }
// }



// static inline OSQPFloat col_expmax_KKT_using_norms_vec(const OSQPVectorf* v) {
//   const OSQPInt n = VEC_LEN(v);
//   const OSQPFloat* vp = VEC_PTR(v);   // ★ data()

//   int inf_norm_v_tmp = OSQPVectorf_norm_inf(v); // v의 무한 노름을 v에 저장


//   inf_norm_v_tmp = osqp_float_unbiased_exp_abs(inf_norm_v_tmp) > EXP_MAX ? EXP_MAX :
//               osqp_float_unbiased_exp_abs(inf_norm_v_tmp) < EXP_MIN ? 0 :
//               osqp_float_unbiased_exp_abs(inf_norm_v_tmp);
//   return pow2_k_int(inf_norm_v_tmp); // 2^{-k
// }


// /* =========[ SCALE ]========= */

// OSQPInt scale_data(OSQPSolver* solver) {
//   OSQPSettings*  settings = solver->settings;
//   OSQPWorkspace* work     = solver->work;

//   const OSQPInt n = work->data->n;
//   const OSQPInt m = work->data->m;

//   /* 0) 누적 스케일 초기화 */
//   work->scaling->c = (OSQPFloat)1.0;
//   OSQPVectorf_set_scalar(work->scaling->D,    1.0);
//   OSQPVectorf_set_scalar(work->scaling->Dinv, 1.0);
//   OSQPVectorf_set_scalar(work->scaling->E,    1.0);
//   OSQPVectorf_set_scalar(work->scaling->Einv, 1.0);

//   OSQPFloat c_temp;     // Objective function scaling
//   OSQPFloat inf_norm_q; // Infinity norm of q

//   // const OSQPInt T = (settings->scaling > 0) ? settings->scaling : 2;
//   const OSQPInt T = 10; // ★ 0 허용


//   /* ==================== 여기부터 루프 전체 교체 ==================== */
//   for (OSQPInt t = 0; t < T; ++t) {
//   /* 1) 열 패스: KKT 열 최대 exponent */

//     col_expmax_KKT_using_norms(work->data->P,
//                                work->data->A,
//                                work->D_temp,    // D_temp: P,A 열 노름의 최대값
//                                work->D_temp_A,  // E_temp_A: A 행 노름의 최대값
//                                work->E_temp);   // E_temp: A 행 노름

//     // printf("D_temp (col expmax of P,A): \n");
//     // for (OSQPInt i = 0; i < n; ++i) {
//     //   printf("%e ", VEC_GET(work->D_temp, i));
//     // }
//     // printf("\n");
//     // printf("E_temp (row expmax of A): \n");
//     // for (OSQPInt i = 0; i < m; ++i) {
//     //   printf("%e ", VEC_GET(work->E_temp, i));
//     // }
//     // printf("\n");

    
//     // Copy inverses of D/E over themselves
//     OSQPVectorf_ew_reciprocal(work->D_temp, work->D_temp);
//     OSQPVectorf_ew_reciprocal(work->E_temp, work->E_temp);

//     // P <- DPD
//     OSQPMatrix_lmult_diag(work->data->P, work->D_temp);
//     OSQPMatrix_rmult_diag(work->data->P, work->D_temp);

//     // A <- EAD
//     OSQPMatrix_lmult_diag(work->data->A, work->E_temp);
//     OSQPMatrix_rmult_diag(work->data->A, work->D_temp);

//     // q <- Dq
//     OSQPVectorf_ew_prod(work->data->q, work->data->q, work->D_temp);

//     // Update equilibration matrices D and E
//     OSQPVectorf_ew_prod(work->scaling->D, work->scaling->D, work->D_temp);
//     OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, work->E_temp);

//     //
//     // Cost normalization step
//     //

//     // Compute avg norm of cols of P.
//     // OSQPMatrix_col_norm_inf_by_exponent(work->data->P, work->D_temp);
//     OSQPMatrix_col_norm_inf(work->data->P, work->D_temp); // original
//     c_temp = OSQPVectorf_norm_1(work->D_temp);
//     // printf("c_temp (norm of cols of P): %e\n", c_temp);
    
//     c_temp = c_temp / n;
//     printf("c_temp (avg norm of cols of P): %e\n", c_temp);

//     // c_temp = OSQPMatrix_col_norm_inf_avg_by_exponent(work->data->P);

//     // Compute inf norm of q
//     inf_norm_q = OSQPVectorf_norm_inf(work->data->q); // original
//     // inf_norm_q = col_expmax_KKT_using_norms_vec(work->data->q);
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
//   /* ==================== 여기까지 루프 전체 교체 ==================== */

//     // Store cinv, Dinv, Einv
//   work->scaling->cinv = 1. / work->scaling->c;
//   OSQPVectorf_ew_reciprocal(work->scaling->Dinv, work->scaling->D);
//   OSQPVectorf_ew_reciprocal(work->scaling->Einv, work->scaling->E);


//   // Scale problem vectors l, u
//   OSQPVectorf_ew_prod(work->data->l, work->data->l, work->scaling->E);
//   OSQPVectorf_ew_prod(work->data->u, work->data->u, work->scaling->E);

//   return 0;
// }


// /* =========[ UNSCALE ]========= */

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

#if OSQP_EMBEDDED_MODE != 1


// Set values lower than threshold SCALING_REG to 1

OSQPFloat limit_scaling_scalar(OSQPFloat v) {
    v = v < OSQP_MIN_SCALING ? 1.0 : v;
    v = v > OSQP_MAX_SCALING ? OSQP_MAX_SCALING : v;
    return v;
}

void limit_scaling_vector(OSQPVectorf* v) {
  OSQPVectorf_set_scalar_if_lt(v,v,OSQP_MIN_SCALING,1.0);
  OSQPVectorf_set_scalar_if_gt(v,v,OSQP_MAX_SCALING,OSQP_MAX_SCALING);
}

/**
 * Compute infinite norm of the columns of the KKT matrix without forming it
 *
 * The norm is stored in the vector v = (D, E)
 *
 * @param P        Cost matrix
 * @param A        Constraints matrix
 * @param D        Norm of columns related to variables
 * @param D_temp_A Temporary vector for norm of columns of A
 * @param E        Norm of columns related to constraints
 */
void compute_inf_norm_cols_KKT(const OSQPMatrix*  P,
                               const OSQPMatrix*  A,
                                     OSQPVectorf* D,
                                     OSQPVectorf* D_temp_A,
                                     OSQPVectorf* E) {
  // First half
  //  [ P ]
  //  [ A ]
  OSQPMatrix_col_norm_inf(P,D);
  OSQPMatrix_col_norm_inf(A, D_temp_A);
  OSQPVectorf_ew_max_vec(D, D_temp_A, D);

  // Second half
  //  [ A']
  //  [ 0 ]
  OSQPMatrix_row_norm_inf(A,E);
}

OSQPInt scale_data(OSQPSolver* solver) {
  // Scale KKT matrix
  //
  //    [ P   A']
  //    [ A   0 ]
  //
  // with diagonal matrix
  //
  //  S = [ D    ]
  //      [    E ]
  //

  OSQPInt   i;          // Iterations index
  OSQPInt   n;          // Number of variables
  OSQPFloat c_temp;     // Objective function scaling
  OSQPFloat inf_norm_q; // Infinity norm of q

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  n = work->data->n;

  // Initialize scaling to 1
  work->scaling->c = 1.0;
  OSQPVectorf_set_scalar(work->scaling->D,    1.);
  OSQPVectorf_set_scalar(work->scaling->Dinv, 1.);
  OSQPVectorf_set_scalar(work->scaling->E,    1.);
  OSQPVectorf_set_scalar(work->scaling->Einv, 1.);


  for (i = 0; i < settings->scaling; i++) {
    //
    // First Ruiz step
    //

    // Compute norm of KKT columns
    compute_inf_norm_cols_KKT(work->data->P, work->data->A,
                              work->D_temp,
                              work->D_temp_A,
                              work->E_temp);

    // Set to 1 values with 0 norms (avoid crazy scaling)
    limit_scaling_vector(work->D_temp);
    limit_scaling_vector(work->E_temp);

    // Take square root of norms
    OSQPVectorf_ew_sqrt(work->D_temp);
    OSQPVectorf_ew_sqrt(work->E_temp);

    // Copy inverses of D/E over themselves
    OSQPVectorf_ew_reciprocal(work->D_temp, work->D_temp);
    OSQPVectorf_ew_reciprocal(work->E_temp, work->E_temp);

    // Equilibrate matrices P and A and vector q
    // P <- DPD
    OSQPMatrix_lmult_diag(work->data->P,work->D_temp);
    OSQPMatrix_rmult_diag(work->data->P,work->D_temp);

    // A <- EAD
    OSQPMatrix_lmult_diag(work->data->A,work->E_temp);
    OSQPMatrix_rmult_diag(work->data->A,work->D_temp);

    // q <- Dq
    OSQPVectorf_ew_prod(work->data->q, work->data->q, work->D_temp);

    // Update equilibration matrices D and E
    OSQPVectorf_ew_prod(work->scaling->D, work->scaling->D, work->D_temp);
    OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, work->E_temp);

    //
    // Cost normalization step
    //

    // // Compute avg norm of cols of P.
    // OSQPMatrix_col_norm_inf(work->data->P, work->D_temp);
    // c_temp = OSQPVectorf_norm_1(work->D_temp);
    // c_temp = c_temp / n;

    // // Compute inf norm of q
    // inf_norm_q = OSQPVectorf_norm_inf(work->data->q);

    // // If norm_q == 0, set it to 1 (ignore it in the scaling)
    // // NB: Using the same function as with vectors here
    // inf_norm_q = limit_scaling_scalar(inf_norm_q);

    // // Compute max between avg norm of cols of P and inf norm of q
    // c_temp = c_max(c_temp, inf_norm_q);

    // // Limit scaling (use same function as with vectors)
    // c_temp = limit_scaling_scalar(c_temp);

    // // Invert scaling c = 1 / cost_measure
    // c_temp = 1. / c_temp;

    // // Scale P
    // OSQPMatrix_mult_scalar(work->data->P,c_temp);

    // // Scale q
    // OSQPVectorf_mult_scalar(work->data->q, c_temp);

    // // Update cost scaling
    // work->scaling->c *= c_temp;
  }


  // Store cinv, Dinv, Einv
  work->scaling->cinv = 1. / work->scaling->c;
  OSQPVectorf_ew_reciprocal(work->scaling->Dinv, work->scaling->D);
  OSQPVectorf_ew_reciprocal(work->scaling->Einv, work->scaling->E);


  // Scale problem vectors l, u
  OSQPVectorf_ew_prod(work->data->l, work->data->l, work->scaling->E);
  OSQPVectorf_ew_prod(work->data->u, work->data->u, work->scaling->E);

  return 0;
}

#endif /* if OSQP_EMBEDDED_MODE != 1 */


OSQPInt unscale_data(OSQPSolver* solver) {

  OSQPWorkspace* work     = solver->work;

  // Unscale cost
  OSQPMatrix_mult_scalar(work->data->P, work->scaling->cinv);
  OSQPMatrix_lmult_diag(work->data->P,  work->scaling->Dinv);
  OSQPMatrix_rmult_diag(work->data->P,  work->scaling->Dinv);
  OSQPVectorf_mult_scalar(work->data->q,work->scaling->cinv);
  OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->Dinv);

  // Unscale constraints
  OSQPMatrix_lmult_diag(work->data->A,work->scaling->Einv);
  OSQPMatrix_rmult_diag(work->data->A,work->scaling->Dinv);

  OSQPVectorf_ew_prod(work->data->l,
                      work->data->l,
                      work->scaling->Einv);
  OSQPVectorf_ew_prod(work->data->u,
                      work->data->u,
                      work->scaling->Einv);

  return 0;
}

OSQPInt unscale_solution(OSQPVectorf*       usolx,
                         OSQPVectorf*       usoly,
                         const OSQPVectorf* solx,
                         const OSQPVectorf* soly,
                         OSQPWorkspace*     work) {

  // primal
  OSQPVectorf_ew_prod(usolx,solx,work->scaling->D);

  // dual
  OSQPVectorf_ew_prod(usoly,soly,work->scaling->E);

  OSQPVectorf_mult_scalar(usoly,work->scaling->cinv);
  return 0;
}

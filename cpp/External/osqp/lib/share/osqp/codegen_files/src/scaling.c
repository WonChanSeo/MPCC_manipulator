#include "scaling.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "algebra_vector.h"
#include "algebra_matrix.h"

// ===== ASIC Testcase: Iteration-by-iteration saving =====
#define OSQP_TESTCASE_SAVE_INTERVAL 50
#define OSQP_TESTCASE_DIR "../result/asic_testcases/osqp/scaling"

// Use the counter from osqp_api.c (shared across both files)
extern OSQPInt g_testcase_count;

static int g_scaling_dir_initialized = 0;
static void init_scaling_dir(void) {
  if (!g_scaling_dir_initialized) {
    system("mkdir -p " OSQP_TESTCASE_DIR);
    g_scaling_dir_initialized = 1;
  }
}

// Helper: Save CSC matrix in float format
static void scaling_save_csc_float(FILE* f, const char* name, const OSQPMatrix* M) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPInt* Mi = OSQPMatrix_get_i(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);
  OSQPInt m = OSQPMatrix_get_m(M);
  OSQPInt nnz = Mp[n];

  fprintf(f, "# %s (CSC) - %lld x %lld, nnz=%lld\n", name, (long long)m, (long long)n, (long long)nnz);
  fprintf(f, "# col_ptr\n");
  for (OSQPInt j = 0; j <= n; j++) {
    fprintf(f, "%lld", (long long)Mp[j]);
    if (j < n) fprintf(f, ",");
  }
  fprintf(f, "\n# row_idx\n");
  for (OSQPInt k = 0; k < nnz; k++) {
    fprintf(f, "%lld", (long long)Mi[k]);
    if (k < nnz - 1) fprintf(f, ",");
  }
  fprintf(f, "\n# values\n");
  for (OSQPInt k = 0; k < nnz; k++) {
    fprintf(f, "%.8e", (double)Mx[k]);
    if (k < nnz - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");
}

// Helper: Save CSC matrix in hex bits format
static void scaling_save_csc_bits(FILE* f, const char* name, const OSQPMatrix* M) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPInt* Mi = OSQPMatrix_get_i(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);
  OSQPInt m = OSQPMatrix_get_m(M);
  OSQPInt nnz = Mp[n];

  fprintf(f, "# %s (CSC, hex FP32) - %lld x %lld, nnz=%lld\n", name, (long long)m, (long long)n, (long long)nnz);
  fprintf(f, "# col_ptr\n");
  for (OSQPInt j = 0; j <= n; j++) {
    fprintf(f, "%lld", (long long)Mp[j]);
    if (j < n) fprintf(f, ",");
  }
  fprintf(f, "\n# row_idx\n");
  for (OSQPInt k = 0; k < nnz; k++) {
    fprintf(f, "%lld", (long long)Mi[k]);
    if (k < nnz - 1) fprintf(f, ",");
  }
  fprintf(f, "\n# values (hex FP32)\n");
  for (OSQPInt k = 0; k < nnz; k++) {
    union { float f; uint32_t u; } v;
    v.f = (float)Mx[k];
    fprintf(f, "%08x", v.u);
    if (k < nnz - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");
}

// Helper: Save dense matrix (expand CSC) - float format
static void scaling_save_dense_float(FILE* f, const char* name, const OSQPMatrix* M, int is_symmetric) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPInt* Mi = OSQPMatrix_get_i(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);
  OSQPInt m = OSQPMatrix_get_m(M);

  fprintf(f, "# %s (dense) - %lld x %lld\n", name, (long long)m, (long long)n);
  for (OSQPInt i = 0; i < m; i++) {
    for (OSQPInt j = 0; j < n; j++) {
      OSQPFloat val = 0.0;
      for (OSQPInt k = Mp[j]; k < Mp[j + 1]; k++) {
        if (Mi[k] == i) { val = Mx[k]; break; }
      }
      if (is_symmetric && val == 0.0 && i != j && j < m) {
        for (OSQPInt k = Mp[i]; k < Mp[i + 1]; k++) {
          if (Mi[k] == j) { val = Mx[k]; break; }
        }
      }
      fprintf(f, "%.8e", (double)val);
      if (j < n - 1) fprintf(f, ",");
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
}

// Helper: Save dense matrix (expand CSC) - bits format
static void scaling_save_dense_bits(FILE* f, const char* name, const OSQPMatrix* M, int is_symmetric) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPInt* Mi = OSQPMatrix_get_i(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);
  OSQPInt m = OSQPMatrix_get_m(M);

  fprintf(f, "# %s (dense, hex FP32) - %lld x %lld\n", name, (long long)m, (long long)n);
  for (OSQPInt i = 0; i < m; i++) {
    for (OSQPInt j = 0; j < n; j++) {
      OSQPFloat val = 0.0;
      for (OSQPInt k = Mp[j]; k < Mp[j + 1]; k++) {
        if (Mi[k] == i) { val = Mx[k]; break; }
      }
      if (is_symmetric && val == 0.0 && i != j && j < m) {
        for (OSQPInt k = Mp[i]; k < Mp[i + 1]; k++) {
          if (Mi[k] == j) { val = Mx[k]; break; }
        }
      }
      union { float f; uint32_t u; } v;
      v.f = (float)val;
      fprintf(f, "%08x", v.u);
      if (j < n - 1) fprintf(f, ",");
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
}

// Helper: Save vector - float format
static void scaling_save_vector_float(FILE* f, const char* name, const OSQPVectorf* v) {
  OSQPInt len = OSQPVectorf_length(v);
  const OSQPFloat* data = OSQPVectorf_data(v);
  fprintf(f, "# %s - %lld values\n", name, (long long)len);
  for (OSQPInt i = 0; i < len; i++) {
    fprintf(f, "%.8e", (double)data[i]);
    if (i < len - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");
}

// Helper: Save vector - bits format
static void scaling_save_vector_bits(FILE* f, const char* name, const OSQPVectorf* v) {
  OSQPInt len = OSQPVectorf_length(v);
  const OSQPFloat* data = OSQPVectorf_data(v);
  fprintf(f, "# %s (hex FP32) - %lld values\n", name, (long long)len);
  for (OSQPInt i = 0; i < len; i++) {
    union { float f; uint32_t u; } val;
    val.f = (float)data[i];
    fprintf(f, "%08x", val.u);
    if (i < len - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");
}

// Save scaling iteration data
static void save_scaling_iteration(OSQPWorkspace* work, OSQPInt sample_id, OSQPInt iter) {
  char path_float[256], path_bits[256];

  // Use "init" for initial data (iter == -1), otherwise use iteration number
  if (iter < 0) {
    snprintf(path_float, sizeof(path_float), "%s/sample_%lld_scaling_init.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id);
    snprintf(path_bits, sizeof(path_bits), "%s/sample_%lld_scaling_init_bits.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id);
  } else {
    snprintf(path_float, sizeof(path_float), "%s/sample_%lld_scaling_iter_%lld.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id, (long long)iter);
    snprintf(path_bits, sizeof(path_bits), "%s/sample_%lld_scaling_iter_%lld_bits.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id, (long long)iter);
  }

  FILE* f_float = fopen(path_float, "w");
  if (f_float) {
    if (iter < 0) {
      fprintf(f_float, "# OSQP Scaling Initial (before scaling) - Sample %lld\n", (long long)sample_id);
    } else {
      fprintf(f_float, "# OSQP Scaling Iteration %lld - Sample %lld\n", (long long)iter, (long long)sample_id);
    }
    fprintf(f_float, "# n=%lld, m=%lld\n\n", (long long)work->data->n, (long long)work->data->m);

    // Save matrices (CSC and dense)
    scaling_save_csc_float(f_float, "P_csc", work->data->P);
    scaling_save_csc_float(f_float, "A_csc", work->data->A);
    scaling_save_dense_float(f_float, "P", work->data->P, 1);
    scaling_save_dense_float(f_float, "A", work->data->A, 0);

    // Save vectors
    scaling_save_vector_float(f_float, "q", work->data->q);
    scaling_save_vector_float(f_float, "l", work->data->l);
    scaling_save_vector_float(f_float, "u", work->data->u);

    // Save D and E scaling vectors
    scaling_save_vector_float(f_float, "D", work->scaling->D);
    scaling_save_vector_float(f_float, "E", work->scaling->E);

    fclose(f_float);
  }

  FILE* f_bits = fopen(path_bits, "w");
  if (f_bits) {
    if (iter < 0) {
      fprintf(f_bits, "# OSQP Scaling Initial (hex FP32, before scaling) - Sample %lld\n", (long long)sample_id);
    } else {
      fprintf(f_bits, "# OSQP Scaling Iteration %lld (hex FP32) - Sample %lld\n", (long long)iter, (long long)sample_id);
    }
    fprintf(f_bits, "# n=%lld, m=%lld\n\n", (long long)work->data->n, (long long)work->data->m);

    // Save matrices (CSC and dense)
    scaling_save_csc_bits(f_bits, "P_csc", work->data->P);
    scaling_save_csc_bits(f_bits, "A_csc", work->data->A);
    scaling_save_dense_bits(f_bits, "P", work->data->P, 1);
    scaling_save_dense_bits(f_bits, "A", work->data->A, 0);

    // Save vectors
    scaling_save_vector_bits(f_bits, "q", work->data->q);
    scaling_save_vector_bits(f_bits, "l", work->data->l);
    scaling_save_vector_bits(f_bits, "u", work->data->u);

    // Save D and E scaling vectors
    scaling_save_vector_bits(f_bits, "D", work->scaling->D);
    scaling_save_vector_bits(f_bits, "E", work->scaling->E);

    fclose(f_bits);
  }
}

// Helper: Save integer exponent array
static void scaling_save_exp_array(FILE* f, const char* name, const int32_t* exp_arr, OSQPInt len) {
  fprintf(f, "# %s (integer exponents) - %lld values\n", name, (long long)len);
  for (OSQPInt i = 0; i < len; i++) {
    fprintf(f, "%d", exp_arr[i]);
    if (i < len - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");
}

// Save scaling iteration data with integer exponents (for ASIC bit-level operations)
static void save_scaling_iteration_exp(OSQPWorkspace* work, OSQPInt sample_id, OSQPInt iter,
                                        const int32_t* D_exp, const int32_t* E_exp,
                                        OSQPInt n, OSQPInt m) {
  init_scaling_dir();
  char path_float[256], path_bits[256];

  // iter == -1: init (before scaling), iter == -2: done (after all scaling), else: iteration number
  if (iter == -1) {
    snprintf(path_float, sizeof(path_float), "%s/sample_%lld_scaling_init.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id);
    snprintf(path_bits, sizeof(path_bits), "%s/sample_%lld_scaling_init_bits.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id);
  } else if (iter == -2) {
    snprintf(path_float, sizeof(path_float), "%s/sample_%lld_scaling_done.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id);
    snprintf(path_bits, sizeof(path_bits), "%s/sample_%lld_scaling_done_bits.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id);
  } else {
    snprintf(path_float, sizeof(path_float), "%s/sample_%lld_scaling_iter_%lld.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id, (long long)iter);
    snprintf(path_bits, sizeof(path_bits), "%s/sample_%lld_scaling_iter_%lld_bits.csv",
             OSQP_TESTCASE_DIR, (long long)sample_id, (long long)iter);
  }

  const char* phase_str = (iter == -1) ? "Initial (before scaling)" :
                           (iter == -2) ? "Done (after all scaling)" : NULL;

  FILE* f_float = fopen(path_float, "w");
  if (f_float) {
    if (phase_str) {
      fprintf(f_float, "# OSQP Scaling %s - Sample %lld\n", phase_str, (long long)sample_id);
    } else {
      fprintf(f_float, "# OSQP Scaling Iteration %lld - Sample %lld\n", (long long)iter, (long long)sample_id);
    }
    fprintf(f_float, "# n=%lld, m=%lld\n\n", (long long)n, (long long)m);

    // Save matrices (CSC and dense)
    scaling_save_csc_float(f_float, "P_csc", work->data->P);
    scaling_save_csc_float(f_float, "A_csc", work->data->A);
    scaling_save_dense_float(f_float, "P", work->data->P, 1);
    scaling_save_dense_float(f_float, "A", work->data->A, 0);

    // Save vectors
    scaling_save_vector_float(f_float, "q", work->data->q);
    scaling_save_vector_float(f_float, "l", work->data->l);
    scaling_save_vector_float(f_float, "u", work->data->u);

    // Save D and E as integer exponents (accumulated)
    scaling_save_exp_array(f_float, "D_exp", D_exp, n);
    scaling_save_exp_array(f_float, "E_exp", E_exp, m);

    fclose(f_float);
  }

  FILE* f_bits = fopen(path_bits, "w");
  if (f_bits) {
    if (phase_str) {
      fprintf(f_bits, "# OSQP Scaling %s (hex FP32) - Sample %lld\n", phase_str, (long long)sample_id);
    } else {
      fprintf(f_bits, "# OSQP Scaling Iteration %lld (hex FP32) - Sample %lld\n", (long long)iter, (long long)sample_id);
    }
    fprintf(f_bits, "# n=%lld, m=%lld\n\n", (long long)n, (long long)m);

    // Save matrices (CSC and dense)
    scaling_save_csc_bits(f_bits, "P_csc", work->data->P);
    scaling_save_csc_bits(f_bits, "A_csc", work->data->A);
    scaling_save_dense_bits(f_bits, "P", work->data->P, 1);
    scaling_save_dense_bits(f_bits, "A", work->data->A, 0);

    // Save vectors
    scaling_save_vector_bits(f_bits, "q", work->data->q);
    scaling_save_vector_bits(f_bits, "l", work->data->l);
    scaling_save_vector_bits(f_bits, "u", work->data->u);

    // Save D and E as integer exponents (same in bits file)
    scaling_save_exp_array(f_bits, "D_exp", D_exp, n);
    scaling_save_exp_array(f_bits, "E_exp", E_exp, m);

    fclose(f_bits);
  }
}

// EnvCol row indices for analysis
// N_eq=99, N_ineqb=259, polytopic_start=358
// EnvCol rows: polytopic_start + NPC*i + [2..10], where NPC=11, i=0..N-1, N=10
#define ENVCOL_POLYTOPIC_START 358
#define ENVCOL_NPC 11
#define ENVCOL_N 10
#define ENVCOL_CON_START 2  // con_envcol1
#define ENVCOL_CON_END 10   // con_envcol9

static int is_envcol_row(int row) {
    if (row < ENVCOL_POLYTOPIC_START) return 0;
    int local = row - ENVCOL_POLYTOPIC_START;
    int step = local / ENVCOL_NPC;
    int offset = local % ENVCOL_NPC;
    if (step >= ENVCOL_N) return 0;  // k=N has zero Jacobian, exclude
    return (offset >= ENVCOL_CON_START && offset <= ENVCOL_CON_END);
}

// Check if EnvCol rows affect column inf-norm calculation for matrix A
// Returns number of columns where EnvCol row provides the max element
static int check_envcol_affects_col_norm(const OSQPMatrix* A) {
    OSQPInt* Ap = OSQPMatrix_get_p(A);
    OSQPInt* Ai = OSQPMatrix_get_i(A);
    OSQPFloat* Ax = OSQPMatrix_get_x(A);
    OSQPInt n = OSQPMatrix_get_n(A);

    int envcol_max_count = 0;

    for (OSQPInt j = 0; j < n; j++) {
        OSQPFloat max_val = 0.0;
        int max_row = -1;

        for (OSQPInt ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
            OSQPFloat abs_val = (Ax[ptr] >= 0) ? Ax[ptr] : -Ax[ptr];
            if (abs_val > max_val) {
                max_val = abs_val;
                max_row = Ai[ptr];
            }
        }

        if (max_row >= 0 && is_envcol_row(max_row)) {
            envcol_max_count++;
        }
    }

    return envcol_max_count;
}

// ===== 통일된 벡터 읽기 접근자 =====
// ---- 통일된 벡터 읽기 접근자 ----
// (builtin 백엔드 기준: 함수 이름은 algebra_vector.h에 선언돼 있음)
// 통일된 읽기 접근자
#define VEC_LEN(v)        OSQPVectorf_length((v))
#define VEC_PTR(v)        OSQPVectorf_data((v))      // ← data_const 말고 data
#define VEC_GET(v,i)      (VEC_PTR((v))[(i)])
#define EXP_MIN -14
#define EXP_MAX +14

static OSQPInt check_bounds(const OSQPVectorf* l, const OSQPVectorf* u) {
  OSQPInt m = OSQPVectorf_length(l);
  for (OSQPInt i=0;i<m;++i) if (VEC_GET(l,i) > VEC_GET(u,i)) return i+1;
  return 0;
}

OSQPFloat limit_scaling_scalar(OSQPFloat v) {
    v = v < OSQP_MIN_SCALING ? 1.0 : v;
    v = v > OSQP_MAX_SCALING ? OSQP_MAX_SCALING : v;
    return v;
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

/* 2^{k} as float */
static inline OSQPFloat pow2_k_int(int k) {
  // printf("2^(%d) = %e\n", k, scalbnf(1.0f, k));
  return scalbnf(1.0f, k);
}

/* =========[ Bit-level exponent operations for ASIC ]========= */

// Extract exponent from FP32 bits (returns unbiased exponent, i.e., actual power of 2)
// FP32: [31:sign][30:23:exp][22:0:mantissa], exp_biased = exp + 127
static inline int32_t fp32_get_exp(float x) {
  union { float f; uint32_t u; } v = { x };
  uint32_t exp_biased = (v.u >> 23) & 0xFF;
  if (exp_biased == 0) return -127;  // zero or subnormal
  if (exp_biased == 255) return 128; // inf or nan
  return (int32_t)exp_biased - 127;
}

// Scale float by adding exp_delta to its exponent (bit-level, no float multiplication)
// This is equivalent to: x * 2^exp_delta
static inline float fp32_scale_by_exp(float x, int32_t exp_delta) {
  if (x == 0.0f) return 0.0f;
  union { float f; uint32_t u; } v = { x };
  uint32_t sign = v.u & 0x80000000u;
  uint32_t exp_biased = (v.u >> 23) & 0xFF;
  uint32_t mantissa = v.u & 0x7FFFFFu;

  // Handle special cases
  if (exp_biased == 0 || exp_biased == 255) return x;  // zero, subnormal, inf, nan

  int32_t new_exp = (int32_t)exp_biased + exp_delta;

  // Clamp to valid range
  if (new_exp <= 0) return 0.0f;        // underflow to zero
  if (new_exp >= 255) {                  // overflow to inf
    v.u = sign | 0x7F800000u;
    return v.f;
  }

  v.u = sign | ((uint32_t)new_exp << 23) | mantissa;
  return v.f;
}

// Scale vector elements by exponent array (bit-level operation)
// result[i] = vec[i] * 2^exp_arr[i]
static void vec_scale_by_exp_array(OSQPVectorf* result, const OSQPVectorf* vec, const int32_t* exp_arr) {
  OSQPInt len = OSQPVectorf_length(vec);
  const OSQPFloat* src = OSQPVectorf_data(vec);
  OSQPFloat* dst = OSQPVectorf_data(result);
  for (OSQPInt i = 0; i < len; i++) {
    dst[i] = fp32_scale_by_exp((float)src[i], exp_arr[i]);
  }
}

// Scale CSC matrix by left diagonal exponent array (bit-level): M = diag(2^exp_arr) * M
static void matrix_lmult_diag_exp(OSQPMatrix* M, const int32_t* exp_arr) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPInt* Mi = OSQPMatrix_get_i(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);

  for (OSQPInt j = 0; j < n; j++) {
    for (OSQPInt k = Mp[j]; k < Mp[j + 1]; k++) {
      OSQPInt row = Mi[k];
      Mx[k] = fp32_scale_by_exp((float)Mx[k], exp_arr[row]);
    }
  }
}

// Scale CSC matrix by right diagonal exponent array (bit-level): M = M * diag(2^exp_arr)
static void matrix_rmult_diag_exp(OSQPMatrix* M, const int32_t* exp_arr) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);

  for (OSQPInt j = 0; j < n; j++) {
    int32_t exp_j = exp_arr[j];
    for (OSQPInt k = Mp[j]; k < Mp[j + 1]; k++) {
      Mx[k] = fp32_scale_by_exp((float)Mx[k], exp_j);
    }
  }
}

// Compute column inf-norm exponents for matrix (returns half of max exponent for each column)
static void matrix_col_exp_half(const OSQPMatrix* M, int32_t* exp_arr) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);

  for (OSQPInt j = 0; j < n; j++) {
    int32_t max_exp = -127;
    for (OSQPInt k = Mp[j]; k < Mp[j + 1]; k++) {
      int32_t e = fp32_get_exp((float)Mx[k]);
      if (e > max_exp) max_exp = e;
    }
    // Clamp and take half (arithmetic shift right by 1)
    if (max_exp < EXP_MIN) max_exp = 0;
    if (max_exp > EXP_MAX) max_exp = EXP_MAX;
    exp_arr[j] = max_exp >> 1;
  }
}

// Compute row inf-norm exponents for CSC matrix (returns half of max exponent for each row)
static void matrix_row_exp_half(const OSQPMatrix* M, int32_t* exp_arr, OSQPInt m) {
  OSQPInt* Mp = OSQPMatrix_get_p(M);
  OSQPInt* Mi = OSQPMatrix_get_i(M);
  OSQPFloat* Mx = OSQPMatrix_get_x(M);
  OSQPInt n = OSQPMatrix_get_n(M);

  // Initialize to minimum
  for (OSQPInt i = 0; i < m; i++) exp_arr[i] = -127;

  // Scan all elements
  for (OSQPInt j = 0; j < n; j++) {
    for (OSQPInt k = Mp[j]; k < Mp[j + 1]; k++) {
      OSQPInt row = Mi[k];
      int32_t e = fp32_get_exp((float)Mx[k]);
      if (e > exp_arr[row]) exp_arr[row] = e;
    }
  }

  // Clamp and take half
  for (OSQPInt i = 0; i < m; i++) {
    if (exp_arr[i] < EXP_MIN) exp_arr[i] = 0;
    if (exp_arr[i] > EXP_MAX) exp_arr[i] = EXP_MAX;
    exp_arr[i] = exp_arr[i] >> 1;
  }
}

// Compute KKT column exponents: max of P column exp and A column exp (for D)
// Also compute A row exponents (for E)
static void col_expmax_KKT_exp(const OSQPMatrix* P, const OSQPMatrix* A,
                                int32_t* D_exp, int32_t* E_exp, OSQPInt n, OSQPInt m) {
  // Temporary array for A column exponents
  int32_t A_col_exp[1024];  // Assuming n <= 1024

  matrix_col_exp_half(P, D_exp);
  matrix_col_exp_half(A, A_col_exp);

  // D_exp = max(P_col_exp, A_col_exp)
  for (OSQPInt j = 0; j < n; j++) {
    if (A_col_exp[j] > D_exp[j]) D_exp[j] = A_col_exp[j];
  }

  // E_exp = A row exponents
  matrix_row_exp_half(A, E_exp, m);
}

/* =========[ Exponent from norms (KKT-aware) ]========= */

static void OSQPMatrix_col_norm_inf_by_exponent(const OSQPMatrix* M,
                                           OSQPVectorf*    Mcol_inf) {
  OSQPMatrix_col_norm_inf(M, Mcol_inf);

  const OSQPInt n = VEC_LEN(Mcol_inf);
  const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data
  int e_col_tmp;

  for (OSQPInt j = 0; j < n; ++j) {
    e_col_tmp = osqp_float_unbiased_exp_abs(mc[j]) > EXP_MAX ? EXP_MAX :
               osqp_float_unbiased_exp_abs(mc[j]) < EXP_MIN ? 0 :
               osqp_float_unbiased_exp_abs(mc[j]);
    VEC_PTR(Mcol_inf)[j] = pow2_k_int(e_col_tmp); // 2^{-k
  }
}

static inline OSQPFloat OSQPMatrix_col_norm_inf_avg_by_exponent(const OSQPMatrix* M) {
  
  OSQPVectorf* Mcol_inf = OSQPVectorf_malloc(179);
  if (!Mcol_inf) { /* OOM 처리 */ }

  /* 필요하면 0으로 초기화 */
  OSQPVectorf_set_scalar(Mcol_inf, (OSQPFloat)0.0);
  
  OSQPMatrix_col_norm_inf(M, Mcol_inf);

  const OSQPInt n = VEC_LEN(Mcol_inf);
  const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data
  
  int e_col_tmp = 0;

  for (OSQPInt j = 0; j < n; ++j) {
    e_col_tmp += osqp_float_unbiased_exp_abs(mc[j]);
  }

  OSQPVectorf_free(Mcol_inf);

  return pow2_k_int(e_col_tmp >> 9); // 2^{-k
}

static void OSQPMatrix_col_norm_inf_by_exponent_half(const OSQPMatrix* M,
                                           OSQPVectorf*    Mcol_inf) {
  OSQPMatrix_col_norm_inf(M, Mcol_inf);

  const OSQPInt n = VEC_LEN(Mcol_inf);
  const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data
  int e_col_tmp;

  for (OSQPInt j = 0; j < n; ++j) {
    e_col_tmp = osqp_float_unbiased_exp_abs(mc[j]) > EXP_MAX ? EXP_MAX :
               osqp_float_unbiased_exp_abs(mc[j]) < EXP_MIN ? 0 :
               osqp_float_unbiased_exp_abs(mc[j]);
    // printf("Column %d: norm_inf = %e, exp = %d, half exp = %d, scaling = %e\n",
    //        j, mc[j], e_col_tmp, e_col_tmp >> 1, pow2_k_int(e_col_tmp >> 1));
    VEC_PTR(Mcol_inf)[j] = pow2_k_int(e_col_tmp >> 1); // 2^{-k
  }
}

static void OSQPMatrix_row_norm_inf_by_exponent_half(const OSQPMatrix* A,
                                     OSQPVectorf* Arow_inf) {
  OSQPMatrix_row_norm_inf(A, Arow_inf);

  const OSQPInt m = VEC_LEN(Arow_inf);
  const OSQPFloat* ar = VEC_PTR(Arow_inf);   // ★ data()

  int e_row_tmp;

  for (OSQPInt i = 0; i < m; ++i) {
    e_row_tmp = osqp_float_unbiased_exp_abs(ar[i]) > EXP_MAX ? EXP_MAX :
               osqp_float_unbiased_exp_abs(ar[i]) < EXP_MIN ? 0 :
               osqp_float_unbiased_exp_abs(ar[i]);
    VEC_PTR(Arow_inf)[i] = pow2_k_int(e_row_tmp >> 1); // 2^{-k
  }
}

/* e_col[j] = max( exp(||P(:,j)||_inf), exp(||A(:,j)||_inf) ) */
static void col_expmax_KKT_using_norms(const OSQPMatrix* P,
                                        const OSQPMatrix* A,
                                        OSQPVectorf* D,
                                        OSQPVectorf* D_temp_A,
                                        OSQPVectorf* E) {
  OSQPMatrix_col_norm_inf_by_exponent_half(P,D);
  OSQPMatrix_col_norm_inf_by_exponent_half(A, D_temp_A);
  OSQPVectorf_ew_max_vec(D, D_temp_A, D); // D = max(D, D_temp_A)

  OSQPMatrix_row_norm_inf_by_exponent_half(A, E);
}

static void col_expmax_KKT_using_norms_half(const OSQPMatrix* M,
                                       OSQPVectorf* Mcol_inf) {
  OSQPMatrix_col_norm_inf(M, Mcol_inf);

  const OSQPInt n = VEC_LEN(Mcol_inf);
  const OSQPFloat* mc = VEC_PTR(Mcol_inf);   // ★ data()

  int e_col_tmp;

  for (OSQPInt j = 0; j < n; ++j) {
    e_col_tmp = osqp_float_unbiased_exp_abs(mc[j]) > EXP_MAX ? EXP_MAX :
               osqp_float_unbiased_exp_abs(mc[j]) < EXP_MIN ? 0 :
               osqp_float_unbiased_exp_abs(mc[j]);
    VEC_PTR(Mcol_inf)[j] = pow2_k_int(e_col_tmp >> 1); // 2^{-k}
  }
}



static inline OSQPFloat col_expmax_KKT_using_norms_vec(const OSQPVectorf* v) {
  const OSQPInt n = VEC_LEN(v);
  const OSQPFloat* vp = VEC_PTR(v);   // ★ data()

  int inf_norm_v_tmp = OSQPVectorf_norm_inf(v); // v의 무한 노름을 v에 저장


  inf_norm_v_tmp = osqp_float_unbiased_exp_abs(inf_norm_v_tmp) > EXP_MAX ? EXP_MAX :
              osqp_float_unbiased_exp_abs(inf_norm_v_tmp) < EXP_MIN ? 0 :
              osqp_float_unbiased_exp_abs(inf_norm_v_tmp);
  return pow2_k_int(inf_norm_v_tmp); // 2^{-k
}


/* =========[ SCALE (Bit-level exponent operations for ASIC) ]========= */

// Static arrays for integer exponents (assuming max dimension 1024)
static int32_t g_D_exp[1024];      // Column scaling exponents (negated for inverse)
static int32_t g_E_exp[1024];      // Row scaling exponents (negated for inverse)
static int32_t g_D_accum[1024];    // Accumulated D exponents
static int32_t g_E_accum[1024];    // Accumulated E exponents

OSQPInt scale_data(OSQPSolver* solver) {
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  const OSQPInt n = work->data->n;
  const OSQPInt m = work->data->m;

  /* 0) 누적 스케일 초기화 (정수 지수로) */
  work->scaling->c = (OSQPFloat)1.0;
  for (OSQPInt i = 0; i < n; i++) g_D_accum[i] = 0;
  for (OSQPInt i = 0; i < m; i++) g_E_accum[i] = 0;

  // Also initialize the float versions for compatibility
  OSQPVectorf_set_scalar(work->scaling->D,    1.0);
  OSQPVectorf_set_scalar(work->scaling->Dinv, 1.0);
  OSQPVectorf_set_scalar(work->scaling->E,    1.0);
  OSQPVectorf_set_scalar(work->scaling->Einv, 1.0);

  OSQPFloat c_temp;     // Objective function scaling
  OSQPFloat inf_norm_q; // Infinity norm of q

  const OSQPInt T = 10;

  // Check if we should save iteration data for this sample
  int save_iterations = (g_testcase_count % OSQP_TESTCASE_SAVE_INTERVAL == 0);

  // Save initial data (before any scaling iteration)
  if (save_iterations) {
    save_scaling_iteration_exp(work, g_testcase_count, -1, g_D_accum, g_E_accum, n, m);
  }

  for (OSQPInt t = 0; t < T; ++t) {
    /* 1) Compute column/row exponents using bit-level operations */
    // g_D_exp, g_E_exp contain half of max exponent (for sqrt effect)
    col_expmax_KKT_exp(work->data->P, work->data->A, g_D_exp, g_E_exp, n, m);

    /* 2) Apply scaling using bit-level exponent operations
     * We want: P <- D^{-1} P D^{-1}, A <- E^{-1} A D^{-1}, q <- D^{-1} q
     * where D = diag(2^{D_exp}), E = diag(2^{E_exp})
     * So D^{-1} = diag(2^{-D_exp}), E^{-1} = diag(2^{-E_exp})
     * => multiply matrix elements by 2^{-D_exp} or 2^{-E_exp}
     */

    // Compute negated exponents for inverse scaling
    int32_t neg_D_exp[1024], neg_E_exp[1024];
    for (OSQPInt i = 0; i < n; i++) neg_D_exp[i] = -g_D_exp[i];
    for (OSQPInt i = 0; i < m; i++) neg_E_exp[i] = -g_E_exp[i];

    // P <- D^{-1} P D^{-1} (bit-level: add -D_exp to exponents of matrix elements)
    matrix_lmult_diag_exp(work->data->P, neg_D_exp);
    matrix_rmult_diag_exp(work->data->P, neg_D_exp);

    // A <- E^{-1} A D^{-1} (bit-level)
    matrix_lmult_diag_exp(work->data->A, neg_E_exp);
    matrix_rmult_diag_exp(work->data->A, neg_D_exp);

    // Update accumulated exponents: D_accum += neg_D_exp, E_accum += neg_E_exp
    for (OSQPInt i = 0; i < n; i++) g_D_accum[i] += neg_D_exp[i];
    for (OSQPInt i = 0; i < m; i++) g_E_accum[i] += neg_E_exp[i];

    // // Save iteration data with integer exponents
    // if (save_iterations) {
    //   save_scaling_iteration_exp(work, g_testcase_count, t, g_D_accum, g_E_accum, n, m);
    // }
  }
  /* ==================== 루프 끝 ==================== */

  /* 3) After all iterations, compute final D, E float vectors from accumulated exponents
   * D[i] = 2^{D_accum[i]}, E[i] = 2^{E_accum[i]}
   * Dinv[i] = 2^{-D_accum[i]}, Einv[i] = 2^{-E_accum[i]}
   */
  OSQPFloat* D_data = OSQPVectorf_data(work->scaling->D);
  OSQPFloat* Dinv_data = OSQPVectorf_data(work->scaling->Dinv);
  OSQPFloat* E_data = OSQPVectorf_data(work->scaling->E);
  OSQPFloat* Einv_data = OSQPVectorf_data(work->scaling->Einv);

  for (OSQPInt i = 0; i < n; i++) {
    // D[i] = 2^{D_accum[i]} using bit-level operation
    D_data[i] = fp32_scale_by_exp(1.0f, g_D_accum[i]);
    Dinv_data[i] = fp32_scale_by_exp(1.0f, -g_D_accum[i]);
  }
  for (OSQPInt i = 0; i < m; i++) {
    // E[i] = 2^{E_accum[i]} using bit-level operation
    E_data[i] = fp32_scale_by_exp(1.0f, g_E_accum[i]);
    Einv_data[i] = fp32_scale_by_exp(1.0f, -g_E_accum[i]);
  }

  // Store cinv (c remains 1.0 since we don't do cost normalization)
  work->scaling->cinv = 1. / work->scaling->c;

  /* 4) Scale problem vectors q, l, u using bit-level operations
   * q <- D^{-1} q = diag(2^{D_accum}) * q  (D_accum is already negated)
   * l <- E * l = diag(2^{E_accum}) * l
   * u <- E * u = diag(2^{E_accum}) * u
   */
  OSQPFloat* q_data = OSQPVectorf_data(work->data->q);
  for (OSQPInt i = 0; i < n; i++) {
    q_data[i] = fp32_scale_by_exp((float)q_data[i], g_D_accum[i]);
  }

  OSQPFloat* l_data = OSQPVectorf_data(work->data->l);
  OSQPFloat* u_data = OSQPVectorf_data(work->data->u);
  for (OSQPInt i = 0; i < m; i++) {
    l_data[i] = fp32_scale_by_exp((float)l_data[i], g_E_accum[i]);
    u_data[i] = fp32_scale_by_exp((float)u_data[i], g_E_accum[i]);
  }

  // Save final scaled data (after all scaling applied to P, A, q, l, u)
  if (save_iterations) {
    save_scaling_iteration_exp(work, g_testcase_count, -2, g_D_accum, g_E_accum, n, m);
  }

  return 0;
}


/* =========[ UNSCALE ]========= */

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

  OSQPInt n = OSQPVectorf_length(solx);
  OSQPInt m = OSQPVectorf_length(soly);

  // primal: x_out[i] = x_scaled[i] * 2^{D_accum[i]}  (exponent bit-shift)
  const OSQPFloat* sx = OSQPVectorf_data((OSQPVectorf*)solx);
  OSQPFloat*       ux = OSQPVectorf_data(usolx);
  for (OSQPInt i = 0; i < n; i++) {
    ux[i] = fp32_scale_by_exp((float)sx[i], g_D_accum[i]);
  }

  // dual: y_out[i] = y_scaled[i] * 2^{E_accum[i]} * cinv
  // cinv = 1.0 (no cost normalization), so just exponent shift
  const OSQPFloat* sy = OSQPVectorf_data((OSQPVectorf*)soly);
  OSQPFloat*       uy = OSQPVectorf_data(usoly);
  for (OSQPInt i = 0; i < m; i++) {
    uy[i] = fp32_scale_by_exp((float)sy[i], g_E_accum[i]);
  }

  return 0;
}


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

//     // // Compute avg norm of cols of P.
//     // OSQPMatrix_col_norm_inf(work->data->P, work->D_temp);
//     // c_temp = OSQPVectorf_norm_1(work->D_temp);
//     // c_temp = c_temp / n;

//     // // Compute inf norm of q
//     // inf_norm_q = OSQPVectorf_norm_inf(work->data->q);

//     // // If norm_q == 0, set it to 1 (ignore it in the scaling)
//     // // NB: Using the same function as with vectors here
//     // inf_norm_q = limit_scaling_scalar(inf_norm_q);

//     // // Compute max between avg norm of cols of P and inf norm of q
//     // c_temp = c_max(c_temp, inf_norm_q);

//     // // Limit scaling (use same function as with vectors)
//     // c_temp = limit_scaling_scalar(c_temp);

//     // // Invert scaling c = 1 / cost_measure
//     // c_temp = 1. / c_temp;

//     // // Scale P
//     // OSQPMatrix_mult_scalar(work->data->P,c_temp);

//     // // Scale q
//     // OSQPVectorf_mult_scalar(work->data->q, c_temp);

//     // // Update cost scaling
//     // work->scaling->c *= c_temp;
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

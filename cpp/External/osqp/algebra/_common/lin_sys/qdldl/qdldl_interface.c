#include "glob_opts.h"
#include "algebra_impl.h"
#include "printing.h"
#include "profilers.h"
#include "timing.h"

#include "error.h"
#include "qdldl.h"
#include "qdldl_interface.h"
#include "util.h"

#ifndef OSQP_EMBEDDED_MODE
#include "amd.h"
#endif

#if OSQP_EMBEDDED_MODE != 1
#include "kkt.h"
#endif

#include <stdio.h>
#include "qdldl_interface.h"

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <fenv.h>

#if defined(OSQP_USE_TRUNCATE)
  #define OSQP_ROUNDING_MODE FE_TOWARDZERO
#elif defined(OSQP_USE_ROUND_TO_EVEN)
  #define OSQP_ROUNDING_MODE FE_TONEAREST
#endif

// External timing setter functions (defined in osqp_api.c)
extern void osqp_set_permutation_time(OSQPFloat time);
extern void osqp_set_factorization_time(OSQPFloat time);

// Use the same testcase counter from osqp_api.c (shared for consistent sampling)
extern OSQPInt g_testcase_count;

// ===== ASIC Testcase: Factorization result saving =====
#define FACTORIZATION_TESTCASE_SAVE_INTERVAL 50
#define FACTORIZATION_TESTCASE_DIR "../result/asic_testcases/osqp/factorization"

// ===== ASIC Testcase: Solve result saving =====
#define SOLVE_TESTCASE_SAVE_INTERVAL 50
#define SOLVE_TESTCASE_DIR "../result/asic_testcases/osqp/solve"

// Track solve calls within each sample for intermediate debug saving
static OSQPInt g_solve_call_count = 0;  // reset at each factorization

// Helper: Save CSC matrix in float format
static void factor_save_csc_float(FILE* f, const char* name, OSQPInt n, OSQPInt m,
                                   const OSQPInt* Mp, const OSQPInt* Mi, const OSQPFloat* Mx) {
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
static void factor_save_csc_bits(FILE* f, const char* name, OSQPInt n, OSQPInt m,
                                  const OSQPInt* Mp, const OSQPInt* Mi, const OSQPFloat* Mx) {
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

// Helper: Save vector in float format
static void factor_save_vector_float(FILE* f, const char* name, const OSQPFloat* data, OSQPInt len) {
  fprintf(f, "# %s - %lld values\n", name, (long long)len);
  for (OSQPInt i = 0; i < len; i++) {
    fprintf(f, "%.8e", (double)data[i]);
    if (i < len - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");
}

// Helper: Save vector in hex bits format
static void factor_save_vector_bits(FILE* f, const char* name, const OSQPFloat* data, OSQPInt len) {
  fprintf(f, "# %s (hex FP32) - %lld values\n", name, (long long)len);
  for (OSQPInt i = 0; i < len; i++) {
    union { float f; uint32_t u; } val;
    val.f = (float)data[i];
    fprintf(f, "%08x", val.u);
    if (i < len - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");
}

// Helper: Save CSC matrix as CSR format (float)
static void factor_save_csr_float(FILE* f, const char* name, OSQPInt n,
                                   const OSQPInt* Cp, const OSQPInt* Ci, const OSQPFloat* Cx) {
  OSQPInt nnz = Cp[n];
  OSQPInt i, j, k;

  OSQPInt* row_ptr  = (OSQPInt*)calloc(n + 1, sizeof(OSQPInt));
  OSQPInt* col_idx  = (OSQPInt*)malloc(nnz * sizeof(OSQPInt));
  OSQPFloat* values = (OSQPFloat*)malloc(nnz * sizeof(OSQPFloat));
  OSQPInt* cursor   = (OSQPInt*)calloc(n, sizeof(OSQPInt));
  if (!row_ptr || !col_idx || !values || !cursor) { free(row_ptr); free(col_idx); free(values); free(cursor); return; }

  for (k = 0; k < nnz; k++) row_ptr[Ci[k] + 1]++;
  for (i = 0; i < n; i++) row_ptr[i + 1] += row_ptr[i];
  for (j = 0; j < n; j++) {
    for (k = Cp[j]; k < Cp[j + 1]; k++) {
      OSQPInt row = Ci[k];
      OSQPInt dest = row_ptr[row] + cursor[row];
      col_idx[dest] = j;
      values[dest] = Cx[k];
      cursor[row]++;
    }
  }

  fprintf(f, "# %s (CSR) - %lld x %lld, nnz=%lld\n", name, (long long)n, (long long)n, (long long)nnz);
  fprintf(f, "# row_ptr\n");
  for (i = 0; i <= n; i++) { fprintf(f, "%lld", (long long)row_ptr[i]); if (i < n) fprintf(f, ","); }
  fprintf(f, "\n# col_idx\n");
  for (k = 0; k < nnz; k++) { fprintf(f, "%lld", (long long)col_idx[k]); if (k < nnz-1) fprintf(f, ","); }
  fprintf(f, "\n# values\n");
  for (k = 0; k < nnz; k++) { fprintf(f, "%.8e", (double)values[k]); if (k < nnz-1) fprintf(f, ","); }
  fprintf(f, "\n\n");

  free(row_ptr); free(col_idx); free(values); free(cursor);
}

// Helper: Save CSC matrix as CSR format (hex FP32)
static void factor_save_csr_bits(FILE* f, const char* name, OSQPInt n,
                                  const OSQPInt* Cp, const OSQPInt* Ci, const OSQPFloat* Cx) {
  OSQPInt nnz = Cp[n];
  OSQPInt i, j, k;

  OSQPInt* row_ptr  = (OSQPInt*)calloc(n + 1, sizeof(OSQPInt));
  OSQPInt* col_idx  = (OSQPInt*)malloc(nnz * sizeof(OSQPInt));
  OSQPFloat* values = (OSQPFloat*)malloc(nnz * sizeof(OSQPFloat));
  OSQPInt* cursor   = (OSQPInt*)calloc(n, sizeof(OSQPInt));
  if (!row_ptr || !col_idx || !values || !cursor) { free(row_ptr); free(col_idx); free(values); free(cursor); return; }

  for (k = 0; k < nnz; k++) row_ptr[Ci[k] + 1]++;
  for (i = 0; i < n; i++) row_ptr[i + 1] += row_ptr[i];
  for (j = 0; j < n; j++) {
    for (k = Cp[j]; k < Cp[j + 1]; k++) {
      OSQPInt row = Ci[k];
      OSQPInt dest = row_ptr[row] + cursor[row];
      col_idx[dest] = j;
      values[dest] = Cx[k];
      cursor[row]++;
    }
  }

  fprintf(f, "# %s (CSR, hex FP32) - %lld x %lld, nnz=%lld\n", name, (long long)n, (long long)n, (long long)nnz);
  fprintf(f, "# row_ptr\n");
  for (i = 0; i <= n; i++) { fprintf(f, "%lld", (long long)row_ptr[i]); if (i < n) fprintf(f, ","); }
  fprintf(f, "\n# col_idx\n");
  for (k = 0; k < nnz; k++) { fprintf(f, "%lld", (long long)col_idx[k]); if (k < nnz-1) fprintf(f, ","); }
  fprintf(f, "\n# values (hex FP32)\n");
  for (k = 0; k < nnz; k++) {
    union { float f; uint32_t u; } v; v.f = (float)values[k];
    fprintf(f, "%08x", v.u); if (k < nnz-1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");

  free(row_ptr); free(col_idx); free(values); free(cursor);
}

// Row offset for L full matrix (non-zero values start from this row)
#define L_FULL_MATRIX_START_ROW 479

// Helper: Save symmetric CSC matrix as full matrix (entire n x n) - float format
// KKT is symmetric, so we need to fill both upper and lower triangular parts
static void factor_save_KKT_full_float(FILE* f, const char* name, OSQPInt n,
                                        const OSQPInt* Kp, const OSQPInt* Ki, const OSQPFloat* Kx) {
  fprintf(f, "# %s (full matrix) - %lld x %lld\n", name, (long long)n, (long long)n);

  // For each row 0 to n-1
  for (OSQPInt i = 0; i < n; i++) {
    // For each column 0 to n-1
    for (OSQPInt j = 0; j < n; j++) {
      OSQPFloat val = 0.0;
      // KKT stores upper triangular part, so K[i][j] is stored at column max(i,j), row min(i,j)
      OSQPInt col = (i <= j) ? j : i;
      OSQPInt row = (i <= j) ? i : j;
      // Search for row in column col
      for (OSQPInt k = Kp[col]; k < Kp[col + 1]; k++) {
        if (Ki[k] == row) {
          val = Kx[k];
          break;
        }
      }
      fprintf(f, "%.8e", (double)val);
      if (j < n - 1) fprintf(f, ",");
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
}

// Helper: Save symmetric CSC matrix as full matrix (entire n x n) - bits format
static void factor_save_KKT_full_bits(FILE* f, const char* name, OSQPInt n,
                                       const OSQPInt* Kp, const OSQPInt* Ki, const OSQPFloat* Kx) {
  fprintf(f, "# %s (full matrix, hex FP32) - %lld x %lld\n", name, (long long)n, (long long)n);

  // For each row 0 to n-1
  for (OSQPInt i = 0; i < n; i++) {
    // For each column 0 to n-1
    for (OSQPInt j = 0; j < n; j++) {
      OSQPFloat val = 0.0;
      // KKT stores upper triangular part, so K[i][j] is stored at column max(i,j), row min(i,j)
      OSQPInt col = (i <= j) ? j : i;
      OSQPInt row = (i <= j) ? i : j;
      // Search for row in column col
      for (OSQPInt k = Kp[col]; k < Kp[col + 1]; k++) {
        if (Ki[k] == row) {
          val = Kx[k];
          break;
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

// Helper: Save lower triangular CSC matrix as full matrix (rows from start_row, all columns) - float format
// L is lower triangular: L[i][j] = 0 for i < j
static void factor_save_L_full_float(FILE* f, const char* name, OSQPInt n,
                                      const OSQPInt* Lp, const OSQPInt* Li, const OSQPFloat* Lx,
                                      OSQPInt start_row) {
  OSQPInt num_rows = n - start_row;
  fprintf(f, "# %s (rows %lld to %lld, all columns) - %lld x %lld\n", name, (long long)start_row, (long long)(n-1), (long long)num_rows, (long long)n);

  // For each row from start_row to n-1
  for (OSQPInt i = start_row; i < n; i++) {
    // For each column 0 to n-1
    for (OSQPInt j = 0; j < n; j++) {
      OSQPFloat val = 0.0;
      // L is lower triangular, so L[i][j] exists only if i >= j
      if (i >= j) {
        // Search for row i in column j
        for (OSQPInt k = Lp[j]; k < Lp[j + 1]; k++) {
          if (Li[k] == i) {
            val = Lx[k];
            break;
          }
        }
      }
      fprintf(f, "%.8e", (double)val);
      if (j < n - 1) fprintf(f, ",");
    }
    fprintf(f, "\n");
  }
  fprintf(f, "\n");
}

// Helper: Save lower triangular CSC matrix as full matrix (rows from start_row, all columns) - bits format
static void factor_save_L_full_bits(FILE* f, const char* name, OSQPInt n,
                                     const OSQPInt* Lp, const OSQPInt* Li, const OSQPFloat* Lx,
                                     OSQPInt start_row) {
  OSQPInt num_rows = n - start_row;
  fprintf(f, "# %s (rows %lld to %lld, all columns, hex FP32) - %lld x %lld\n", name, (long long)start_row, (long long)(n-1), (long long)num_rows, (long long)n);

  // For each row from start_row to n-1
  for (OSQPInt i = start_row; i < n; i++) {
    // For each column 0 to n-1
    for (OSQPInt j = 0; j < n; j++) {
      OSQPFloat val = 0.0;
      // L is lower triangular, so L[i][j] exists only if i >= j
      if (i >= j) {
        // Search for row i in column j
        for (OSQPInt k = Lp[j]; k < Lp[j + 1]; k++) {
          if (Li[k] == i) {
            val = Lx[k];
            break;
          }
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

// Save factorization results (KKT, L, D, Dinv)
static void save_factorization_result(OSQPInt sample_id, OSQPInt n,
                                       const OSQPInt* KKTp, const OSQPInt* KKTi, const OSQPFloat* KKTx,
                                       const OSQPInt* Lp, const OSQPInt* Li, const OSQPFloat* Lx,
                                       const OSQPFloat* D, const OSQPFloat* Dinv) {
  char path_float[256], path_bits[256];
  snprintf(path_float, sizeof(path_float), "%s/sample_%lld_factorization.csv",
           FACTORIZATION_TESTCASE_DIR, (long long)sample_id);
  snprintf(path_bits, sizeof(path_bits), "%s/sample_%lld_factorization_bits.csv",
           FACTORIZATION_TESTCASE_DIR, (long long)sample_id);

  FILE* f_float = fopen(path_float, "w");
  if (f_float) {
    fprintf(f_float, "# OSQP Factorization Result - Sample %lld\n", (long long)sample_id);
    fprintf(f_float, "# KKT dimension n=%lld\n\n", (long long)n);

    factor_save_csc_float(f_float, "KKT", n, n, KKTp, KKTi, KKTx);
    factor_save_KKT_full_float(f_float, "KKT_full", n, KKTp, KKTi, KKTx);
    factor_save_csc_float(f_float, "L", n, n, Lp, Li, Lx);
    factor_save_csr_float(f_float, "L", n, Lp, Li, Lx);
    factor_save_L_full_float(f_float, "L_full", n, Lp, Li, Lx, L_FULL_MATRIX_START_ROW);
    factor_save_vector_float(f_float, "D", D, n);
    factor_save_vector_float(f_float, "Dinv", Dinv, n);

    fclose(f_float);
  }

  FILE* f_bits = fopen(path_bits, "w");
  if (f_bits) {
    fprintf(f_bits, "# OSQP Factorization Result (hex FP32) - Sample %lld\n", (long long)sample_id);
    fprintf(f_bits, "# KKT dimension n=%lld\n\n", (long long)n);

    factor_save_csc_bits(f_bits, "KKT", n, n, KKTp, KKTi, KKTx);
    factor_save_KKT_full_bits(f_bits, "KKT_full", n, KKTp, KKTi, KKTx);
    factor_save_csc_bits(f_bits, "L", n, n, Lp, Li, Lx);
    factor_save_csr_bits(f_bits, "L", n, Lp, Li, Lx);
    factor_save_L_full_bits(f_bits, "L_full", n, Lp, Li, Lx, L_FULL_MATRIX_START_ROW);
    factor_save_vector_bits(f_bits, "D", D, n);
    factor_save_vector_bits(f_bits, "Dinv", Dinv, n);

    fclose(f_bits);
  }

  printf("[OSQP] Saved factorization sample_%lld\n", (long long)sample_id);
}

// ===== Solve result saving =====

static int g_solve_dir_created = 0;
static void ensure_solve_dir(void) {
  if (!g_solve_dir_created) {
    // Use system() for portability; harmless if dir already exists
    system("mkdir -p " SOLVE_TESTCASE_DIR);
    g_solve_dir_created = 1;
  }
}

// Save a single vector to file (float + hex)
static void save_solve_vector(const char* filepath, const char* label,
                               OSQPInt sample_id, OSQPInt n,
                               const OSQPFloat* x) {
  ensure_solve_dir();
  FILE* f = fopen(filepath, "w");
  if (!f) return;

  fprintf(f, "# OSQP Solve Result - Sample %lld - %s\n", (long long)sample_id, label);
  fprintf(f, "# n=%lld\n\n", (long long)n);

  // Float format
  fprintf(f, "# x (float) - %lld values\n", (long long)n);
  for (OSQPInt i = 0; i < n; i++) {
    fprintf(f, "%.8e", (double)x[i]);
    if (i < n - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");

  // Hex format
  fprintf(f, "# x (hex FP32) - %lld values\n", (long long)n);
  for (OSQPInt i = 0; i < n; i++) {
    union { float f; uint32_t u; } v;
    v.f = (float)x[i];
    fprintf(f, "%08x", v.u);
    if (i < n - 1) fprintf(f, ",");
  }
  fprintf(f, "\n\n");

  fclose(f);
}

// Save solve result x (called every SOLVE_TESTCASE_SAVE_INTERVAL samples)
static void save_solve_result(OSQPInt sample_id, OSQPInt n,
                               const OSQPFloat* x) {
  char path[256];
  snprintf(path, sizeof(path), "%s/sample_%lld_solve.csv",
           SOLVE_TESTCASE_DIR, (long long)sample_id);
  save_solve_vector(path, "final", sample_id, n, x);
  printf("[OSQP] Saved solve result sample_%lld\n", (long long)sample_id);
}

// Save intermediate solve steps (sample 0, first solve only)
static void save_solve_intermediate(OSQPInt sample_id, const char* step_name,
                                     OSQPInt n, const OSQPFloat* x) {
  char path[256];
  snprintf(path, sizeof(path), "%s/sample_%lld_solve_%s.csv",
           SOLVE_TESTCASE_DIR, (long long)sample_id, step_name);
  save_solve_vector(path, step_name, sample_id, n, x);
  printf("[OSQP] Saved solve intermediate sample_%lld_%s\n",
         (long long)sample_id, step_name);
}

// Save L matrix (CSR format) and Dinv vector at solve time
static void save_solve_L_Dinv(OSQPInt sample_id, OSQPInt n,
                               const OSQPInt* Lp, const OSQPInt* Li, const OSQPFloat* Lx,
                               const OSQPFloat* Dinv) {
  ensure_solve_dir();

  // Float file
  char path[256];
  snprintf(path, sizeof(path), "%s/sample_%lld_solve_L_Dinv.csv",
           SOLVE_TESTCASE_DIR, (long long)sample_id);
  FILE* f = fopen(path, "w");
  if (f) {
    fprintf(f, "# L matrix (CSR) and Dinv at solve time - Sample %lld\n", (long long)sample_id);
    fprintf(f, "# n=%lld\n\n", (long long)n);
    factor_save_csr_float(f, "L", n, Lp, Li, Lx);
    factor_save_vector_float(f, "Dinv", Dinv, n);
    fclose(f);
  }

  // Hex file
  char path_bits[256];
  snprintf(path_bits, sizeof(path_bits), "%s/sample_%lld_solve_L_Dinv_bits.csv",
           SOLVE_TESTCASE_DIR, (long long)sample_id);
  FILE* fb = fopen(path_bits, "w");
  if (fb) {
    fprintf(fb, "# L matrix (CSR, FP32 Hex) and Dinv at solve time - Sample %lld\n", (long long)sample_id);
    fprintf(fb, "# n=%lld\n\n", (long long)n);
    factor_save_csr_bits(fb, "L", n, Lp, Li, Lx);
    factor_save_vector_bits(fb, "Dinv", Dinv, n);
    fclose(fb);
  }

  printf("[OSQP] Saved solve L(CSR)/Dinv sample_%lld\n", (long long)sample_id);
}

// Save metadata (rho_updates, iter) for a sample
// ADMM iteration log entry (must match definition in osqp_api.c)
typedef struct {
  OSQPInt   iter;
  OSQPFloat prim_res;
  OSQPFloat dual_res;
  OSQPFloat duality_gap;
  OSQPFloat rel_kkt_error;
  OSQPFloat rho;
  OSQPFloat last_rel_kkt;
  OSQPFloat kkt_threshold;
  OSQPInt   kkt_check;
  OSQPInt   can_adapt_rho;
  OSQPFloat eps_prim;
  OSQPFloat eps_dual;
  OSQPInt   prim_check;
  OSQPInt   dual_check;
  OSQPInt   terminated;
} admm_log_entry_t;

void save_solve_metadata(OSQPInt sample_id, OSQPInt rho_updates, OSQPInt iter,
                          const OSQPInt* rho_update_iters,
                          const OSQPFloat* rho_update_old,
                          const OSQPFloat* rho_update_new,
                          OSQPInt n_rho_updates_logged,
                          const void* admm_log_ptr, OSQPInt n_admm_log) {
  if (sample_id < 0 || (sample_id % SOLVE_TESTCASE_SAVE_INTERVAL != 0))
    return;

  ensure_solve_dir();

  const admm_log_entry_t* admm_log = (const admm_log_entry_t*)admm_log_ptr;

  char path[256];
  snprintf(path, sizeof(path), "%s/sample_%lld_metadata.csv",
           SOLVE_TESTCASE_DIR, (long long)sample_id);

  FILE* f = fopen(path, "w");
  if (!f) return;

  fprintf(f, "# OSQP Solve Metadata - Sample %lld\n", (long long)sample_id);
  fprintf(f, "# rho_updates,iter\n");
  fprintf(f, "%lld,%lld\n", (long long)rho_updates, (long long)iter);

  // Helper macro for float-to-hex
  #define F2H(val) do { union { float f; uint32_t u; } _v; _v.f = (float)(val); fprintf(f, "%08x", _v.u); } while(0)

  // Rho update history (float)
  fprintf(f, "\n# rho_update_history (%lld entries)\n", (long long)n_rho_updates_logged);
  fprintf(f, "# admm_iter,rho_old,rho_new\n");
  for (OSQPInt i = 0; i < n_rho_updates_logged; i++) {
    fprintf(f, "%lld,%.8e,%.8e\n",
            (long long)rho_update_iters[i],
            (double)rho_update_old[i],
            (double)rho_update_new[i]);
  }

  // Rho update history (hex)
  fprintf(f, "\n# rho_update_history_hex (%lld entries)\n", (long long)n_rho_updates_logged);
  fprintf(f, "# admm_iter,rho_old_hex,rho_new_hex\n");
  for (OSQPInt i = 0; i < n_rho_updates_logged; i++) {
    fprintf(f, "%lld,", (long long)rho_update_iters[i]);
    F2H(rho_update_old[i]); fprintf(f, ",");
    F2H(rho_update_new[i]); fprintf(f, "\n");
  }

  // ADMM iteration log (float)
  fprintf(f, "\n# admm_iteration_log (%lld entries)\n", (long long)n_admm_log);
  fprintf(f, "# iter,prim_res,dual_res,duality_gap,rel_kkt_error,rho,"
             "last_rel_kkt,kkt_threshold,kkt_check,can_adapt_rho,"
             "eps_prim,eps_dual,prim_check,dual_check,terminated\n");
  for (OSQPInt i = 0; i < n_admm_log; i++) {
    const admm_log_entry_t* e = &admm_log[i];
    fprintf(f, "%lld,%.8e,%.8e,%.8e,%.8e,%.8e,"
               "%.8e,%.8e,%lld,%lld,"
               "%.8e,%.8e,%lld,%lld,%lld\n",
            (long long)e->iter,
            (double)e->prim_res, (double)e->dual_res,
            (double)e->duality_gap, (double)e->rel_kkt_error,
            (double)e->rho,
            (double)e->last_rel_kkt, (double)e->kkt_threshold,
            (long long)e->kkt_check, (long long)e->can_adapt_rho,
            (double)e->eps_prim, (double)e->eps_dual,
            (long long)e->prim_check, (long long)e->dual_check,
            (long long)e->terminated);
  }

  // ADMM iteration log (hex)
  fprintf(f, "\n# admm_iteration_log_hex (%lld entries)\n", (long long)n_admm_log);
  fprintf(f, "# iter,prim_res_hex,dual_res_hex,duality_gap_hex,rel_kkt_error_hex,rho_hex,"
             "last_rel_kkt_hex,kkt_threshold_hex,kkt_check,can_adapt_rho,"
             "eps_prim_hex,eps_dual_hex,prim_check,dual_check,terminated\n");
  for (OSQPInt i = 0; i < n_admm_log; i++) {
    const admm_log_entry_t* e = &admm_log[i];
    fprintf(f, "%lld,", (long long)e->iter);
    F2H(e->prim_res);      fprintf(f, ",");
    F2H(e->dual_res);      fprintf(f, ",");
    F2H(e->duality_gap);   fprintf(f, ",");
    F2H(e->rel_kkt_error); fprintf(f, ",");
    F2H(e->rho);           fprintf(f, ",");
    F2H(e->last_rel_kkt);  fprintf(f, ",");
    F2H(e->kkt_threshold); fprintf(f, ",");
    fprintf(f, "%lld,%lld,", (long long)e->kkt_check, (long long)e->can_adapt_rho);
    F2H(e->eps_prim);      fprintf(f, ",");
    F2H(e->eps_dual);      fprintf(f, ",");
    fprintf(f, "%lld,%lld,%lld\n",
            (long long)e->prim_check, (long long)e->dual_check,
            (long long)e->terminated);
  }

  #undef F2H

  fclose(f);
  printf("[OSQP] Saved solve metadata sample_%lld (rho_updates=%lld, iter=%lld, admm_log=%lld)\n",
         (long long)sample_id, (long long)rho_updates, (long long)iter, (long long)n_admm_log);
}

#ifdef OSQP_USE_LONG
#  define OSQP_INT_FMT "%" PRId64
#else
#  define OSQP_INT_FMT "%d"
#endif

// 구분자: 콤마/공백/탭/세미콜론
static const char* CSV_DELIMS = " ,\t;";

// ===== External permutation reader =====
#ifndef OSQP_PERM_PATH
// 기본 경로를 매크로로 지정해두고 필요 시 컴파일 옵션으로 바꿔도 됨
#define OSQP_PERM_PATH "../reference_matrix/P_rev.txt"
#endif

// 파일에서 정수 토큰 하나를 읽어 n개 채운다. 구분자는 공백/탭/콤마/세미콜론 허용.
static int _read_int_tokens(FILE* f, OSQPInt *out, OSQPInt n_expected) {
    OSQPInt count = 0;
    char buf[1<<15];
    while (count < n_expected && fgets(buf, sizeof(buf), f)) {
        char *tok = strtok(buf, " ,\t;\r\n");
        while (tok && count < n_expected) {
#ifdef OSQP_USE_LONG
            long long v; 
            if (sscanf(tok, "%lld", &v) != 1) return -1;
#else
            int v;
            if (sscanf(tok, "%d", &v) != 1) return -1;
#endif
            out[count++] = (OSQPInt)v;
            tok = strtok(NULL, " ,\t;\r\n");
        }
    }
    return (count == n_expected) ? 0 : -2; // 부족하면 에러
}

// P의 0/1 기반 판별 및 0-based 변환, 유효성 검사(중복/범위 체크)
static int normalize_and_validate_perm(OSQPInt *P, OSQPInt n) {
    // 범위 스캔
    OSQPInt minv = P[0], maxv = P[0];
    for (OSQPInt i=1;i<n;++i){ if (P[i]<minv) minv=P[i]; if (P[i]>maxv) maxv=P[i]; }

    int one_based = (minv == 1 && maxv == n);
    int zero_based= (minv == 0 && maxv == n-1);
    if (!one_based && !zero_based) {
        // 혼합/범위 이상
        return -3;
    }
    if (one_based) {
        for (OSQPInt i=0;i<n;++i) P[i] = P[i]-1;
    }

    // 중복 체크
    unsigned char *seen = (unsigned char*)c_calloc((size_t)n, 1);
    if (!seen) return -4;
    for (OSQPInt i=0;i<n;++i){
        OSQPInt v = P[i];
        if (v < 0 || v >= n) { c_free(seen); return -5; }
        if (seen[v]) { c_free(seen); return -6; }  // duplicate
        seen[v] = 1;
    }
    c_free(seen);
    return 0;
}

// 외부 파일에서 P를 읽어와 P_out에 0-based로 채움
static int read_perm_from_file(const char *path, OSQPInt n, OSQPInt *P_out){
    if (!path || !P_out) return -10;
    FILE *fp = fopen(path, "r");
    if (!fp) return -11;

    int rc = _read_int_tokens(fp, P_out, n);
    fclose(fp);
    if (rc) return rc;

    rc = normalize_and_validate_perm(P_out, n);
    return rc; // 0이면 OK
}

// 한 줄에서 토큰을 최대 n개까지 파싱하여 double로 채움.
// expected_cols<0 이면 개수 제한 없음, >=0 이면 정확히 expected_cols개여야 함.
// 반환: 읽은 개수(>=0). 오류시 -1
static int parse_numeric_line(char *line, int expected_cols, double *out_vals) {
    int count = 0;
    for (char *tok = strtok(line, CSV_DELIMS); tok; tok = strtok(NULL, CSV_DELIMS)) {
        char *endp = NULL;
        double v = strtod(tok, &endp);
        if (endp == tok) return -1; // 숫자 파싱 실패
        if (expected_cols >= 0 && count >= expected_cols) return -1;
        out_vals[count++] = v;
    }
    return count;
}

/**
 * Dense CSV -> OSQPCscMatrix (CSC)
 *  - path: 숫자만 있는 CSV/TSV(구분자: , \t 공백 ; 허용)
 *  - tol : |v| <= tol 이면 0으로 간주(무시)
 *  - store_values!=0 이면 x[]에 실제 값 저장, 0이면 모두 1.0 저장(패턴만)
 *  - out : 결과 CSC (csc_spalloc로 할당됨). 실패 시 NULL.
 *
 * 반환: 0 성공, <0 오류
 */
static int read_dense_csv_to_csc(const char *path, double tol, int store_values,
                                 OSQPCscMatrix **out)
{
    if (!path || !out) return -1;
    *out = OSQP_NULL;

    FILE *f = fopen(path, "r");
    if (!f) return -2;

    const int MAX_LINE = 1<<20; // 1MB 라인 버퍼
    char *line = (char*)c_malloc(MAX_LINE);
    if (!line){ fclose(f); return -3; }

    // --- 1pass: 첫 줄 읽고 열 개수 n 결정 ---
    if (!fgets(line, MAX_LINE, f)){ c_free(line); fclose(f); return -4; }

    // 첫 줄을 파싱하기 위해 임시 버퍼 준비
    int max_tokens = (int)strlen(line);
    if (max_tokens < 1) { c_free(line); fclose(f); return -5; }
    double *rowbuf = (double*)c_malloc(sizeof(double)*max_tokens);
    if (!rowbuf){ c_free(line); fclose(f); return -6; }

    int n = parse_numeric_line(line, -1, rowbuf);
    if (n <= 0){ c_free(rowbuf); c_free(line); fclose(f); return -7; }

    // 열별 nnz 카운트
    OSQPInt *col_counts = (OSQPInt*)c_calloc((size_t)n, sizeof(OSQPInt));
    if (!col_counts){ c_free(rowbuf); c_free(line); fclose(f); return -8; }

    // 첫 줄의 nnz 반영
    OSQPInt nnz = 0;
    for (int j=0;j<n;++j){
        if (fabs(rowbuf[j]) > tol) { col_counts[j]++; nnz++; }
    }

    // --- 나머지 줄 스캔: m/nnz/열카운트 ---
    OSQPInt m = 1;
    while (fgets(line, MAX_LINE, f)) {
        // 공백/빈 줄 스킵
        int only_ws = 1;
        for (char *q=line; *q; ++q) {
            if (!isspace((unsigned char)*q) && *q!=',' && *q!=';') { only_ws=0; break; }
        }
        if (only_ws) continue;

        int k = parse_numeric_line(line, n, rowbuf);
        if (k != n){ c_free(col_counts); c_free(rowbuf); c_free(line); fclose(f); return -9; }
        m++;
        for (int j=0;j<n;++j){
            if (fabs(rowbuf[j]) > tol) { col_counts[j]++; nnz++; }
        }
    }

    // --- CSC 메모리 할당 ---
    OSQPCscMatrix *A = csc_spalloc(m, n, nnz, /*values*/1, /*triplet*/0);
    if (!A){ c_free(col_counts); c_free(rowbuf); c_free(line); fclose(f); return -10; }

    // p: 누적합
    A->p[0] = 0;
    for (OSQPInt j=0;j<n;++j){
        A->p[j+1] = A->p[j] + col_counts[j];
    }

    // 열별 삽입 cursor
    OSQPInt *cursor = (OSQPInt*)c_malloc(sizeof(OSQPInt)*(size_t)n);
    if (!cursor){ csc_spfree(A); c_free(col_counts); c_free(rowbuf); c_free(line); fclose(f); return -11; }
    for (OSQPInt j=0;j<n;++j) cursor[j] = A->p[j];

    // --- 2pass: 다시 읽어서 i/x 채우기 ---
    rewind(f);
    if (!fgets(line, MAX_LINE, f)){} // 첫 줄 재독

    // 첫 줄 채우기 (row=0)
    {
        int k = parse_numeric_line(line, n, rowbuf);
        if (k != n){ c_free(cursor); csc_spfree(A); c_free(col_counts); c_free(rowbuf); c_free(line); fclose(f); return -12; }
        for (OSQPInt j=0;j<n;++j){
            double v = rowbuf[j];
            if (fabs(v) > tol){
                OSQPInt pos = cursor[j]++;
                A->i[pos] = 0;
                A->x[pos] = store_values ? (OSQPFloat)v : (OSQPFloat)1.0;
            }
        }
    }

    // 나머지 행들
    OSQPInt row = 1;
    while (fgets(line, MAX_LINE, f)) {
        int only_ws = 1;
        for (char *q=line; *q; ++q) {
            if (!isspace((unsigned char)*q) && *q!=',' && *q!=';') { only_ws=0; break; }
        }
        if (only_ws) continue;

        int k = parse_numeric_line(line, n, rowbuf);
        if (k != n){ c_free(cursor); csc_spfree(A); c_free(col_counts); c_free(rowbuf); c_free(line); fclose(f); return -13; }

        for (OSQPInt j=0;j<n;++j){
            double v = rowbuf[j];
            if (fabs(v) > tol){
                OSQPInt pos = cursor[j]++;
                A->i[pos] = row;
                A->x[pos] = store_values ? (OSQPFloat)v : (OSQPFloat)1.0;
            }
        }
        row++;
    }

    // 정리
    c_free(cursor);
    c_free(col_counts);
    c_free(rowbuf);
    c_free(line);
    fclose(f);

    *out = A;
    return 0;
}

// === CSC 덤프: 텍스트 1파일 형식 ===
// 파일 포맷 (0-based index):
//   % CSC matrix dump
//   m n nnz
//   p: p0 p1 ... p_n
//   i: i0 i1 ... i_(nnz-1)
//   x: x0 x1 ... x_(nnz-1)   (x가 NULL이면 "x:none")
static int dump_csc_text(const OSQPCscMatrix* M, const char* filepath) {
    if (!M || !filepath) return -1;
    FILE* f = fopen(filepath, "w");
    if (!f) return -2;

    const OSQPInt m = M->m;
    const OSQPInt n = M->n;
    const OSQPInt nnz = M->p[n];

    fprintf(f, "%% CSC matrix dump\n");
    fprintf(f, OSQP_INT_FMT " " OSQP_INT_FMT " " OSQP_INT_FMT "\n",
            (long long)m, (long long)n, (long long)nnz);

    // p array
    fputs("p:", f);
    for (OSQPInt j = 0; j <= n; ++j) {
        fprintf(f, " " OSQP_INT_FMT, (long long)M->p[j]);
    }
    fputc('\n', f);

    // i array
    fputs("i:", f);
    for (OSQPInt k = 0; k < nnz; ++k) {
        fprintf(f, " " OSQP_INT_FMT, (long long)M->i[k]);
    }
    fputc('\n', f);

    // x array (값이 NULL이면 구조만)
    if (M->x) {
        fputs("x:", f);
        for (OSQPInt k = 0; k < nnz; ++k) {
            // double/float 상관없이 손실 적은 출력
            fprintf(f, " %.17g", (double)M->x[k]);
        }
        fputc('\n', f);
    } else {
        fputs("x:none\n", f);
    }

    fclose(f);
    return 0;
}

// CSC -> CSR 변환 (대칭 확장 옵션 포함)
// 입력: M (CSC: m x n), expand_sym!=0이면 (i,j)와 (j,i) 모두 반영(대각은 중복 제거)
typedef struct {
    OSQPInt m, n;
    OSQPInt *rp;  // row pointer size m+1
    OSQPInt *ci;  // col indices size nnz'
    OSQPFloat *x; // values size nnz'
} CSR;

static void csr_free(CSR* R){
    if(!R) return;
    c_free(R->rp); c_free(R->ci); c_free(R->x);
    memset(R, 0, sizeof(*R));
}

static int csc_to_csr_with_sym(const OSQPCscMatrix* M, int expand_sym, CSR* out){
    if(!M || !out) return -1;
    const OSQPInt m = M->m;
    const OSQPInt n = M->n;
    const OSQPInt nnz = M->p[n];

    out->m = m; out->n = n;
    out->rp = c_calloc(m+1, sizeof(OSQPInt));
    if(!out->rp) return -2;

    // 1) row counts
    if (!expand_sym){
        for(OSQPInt j=0;j<n;++j){
            for(OSQPInt k=M->p[j]; k<M->p[j+1]; ++k){
                OSQPInt i = M->i[k];
                if(i<0 || i>=m) return -3;
                out->rp[i]++;  // count per row
            }
        }
    } else {
        for(OSQPInt j=0;j<n;++j){
            for(OSQPInt k=M->p[j]; k<M->p[j+1]; ++k){
                OSQPInt i = M->i[k];
                if(i<0 || i>=m) return -3;
                out->rp[i]++;           // (i,j)
                if(i!=j) out->rp[j]++;  // (j,i) for off-diagonal
            }
        }
    }

    // prefix sum -> rp
    OSQPInt total = 0;
    for(OSQPInt i=0;i<m;++i){
        OSQPInt c = out->rp[i];
        out->rp[i] = total;
        total += c;
    }
    out->rp[m] = total;

    out->ci = c_malloc(total * sizeof(OSQPInt));
    out->x  = c_malloc(total * sizeof(OSQPFloat));
    if(!out->ci || !out->x){ csr_free(out); return -4; }

    // 2) fill rows
    // use a working copy of rp as insertion cursor
    OSQPInt* wr = c_malloc((m+1)*sizeof(OSQPInt));
    if(!wr){ csr_free(out); return -5; }
    memcpy(wr, out->rp, (m+1)*sizeof(OSQPInt));

    if(!expand_sym){
        for(OSQPInt j=0;j<n;++j){
            for(OSQPInt k=M->p[j]; k<M->p[j+1]; ++k){
                OSQPInt i = M->i[k];
                OSQPInt pos = wr[i]++;
                out->ci[pos] = j;
                out->x[pos]  = M->x ? M->x[k] : (OSQPFloat)1.0;
            }
        }
    } else {
        for(OSQPInt j=0;j<n;++j){
            for(OSQPInt k=M->p[j]; k<M->p[j+1]; ++k){
                OSQPInt i = M->i[k];
                OSQPFloat v = M->x ? M->x[k] : (OSQPFloat)1.0;

                OSQPInt pos1 = wr[i]++;
                out->ci[pos1] = j;
                out->x[pos1]  = v;

                if(i!=j){
                    OSQPInt pos2 = wr[j]++;
                    out->ci[pos2] = i;
                    out->x[pos2]  = v;
                }
            }
        }
    }

    c_free(wr);
    return 0;
}

// CSR(희소) -> 행렬 모양(행 기준)으로 파일에 저장 (dense stream, 한 행씩 버퍼)
// sep: ","면 CSV, "\t"면 TSV
// precision: 예) 17 -> "%.17g"
static int dump_dense_rows_from_csr(const CSR* R, const char* path,
                                    const char* sep, int precision){
    if(!R || !path || !sep) return -1;
    FILE* f = fopen(path, "w");
    if(!f) return -2;

    // 헤더(선택): 첫 줄에 m,n
    // fprintf(f, OSQP_INT_FMT "," OSQP_INT_FMT "\n", (long long)R->m, (long long)R->n);

    OSQPFloat* rowbuf = c_calloc(R->n, sizeof(OSQPFloat));
    if(!rowbuf){ fclose(f); return -3; }

    char fmt[32];
    // "%.17g" 형태 포맷 문자열 구성
    snprintf(fmt, sizeof(fmt), "%%.%dg", precision > 0 ? precision : 17);

    for(OSQPInt i=0;i<R->m;++i){
        // 0으로 초기화 (이미 calloc했지만 매 행마다 reset)
        memset(rowbuf, 0, R->n * sizeof(OSQPFloat));

        for(OSQPInt k=R->rp[i]; k<R->rp[i+1]; ++k){
            OSQPInt j = R->ci[k];
            rowbuf[j] = R->x[k];
        }

        // 한 행 출력
        for(OSQPInt j=0;j<R->n;++j){
            // 값 출력
            // CSV에서 0도 명시적으로 찍는다. (희소가 아닌 "행렬 모양" 보존)
            fprintf(f, fmt, (double)rowbuf[j]);
            if(j+1 < R->n) fputs(sep, f);
        }
        fputc('\n', f);
    }

    c_free(rowbuf);
    fclose(f);
    return 0;
}

// 통합: CSC 행렬을 행렬 모양(행 기준)으로 저장
// expand_sym: 0 그대로, 1 대칭확장
static int dump_matrix_shape(const OSQPCscMatrix* M, const char* path,
                             int expand_sym, const char* sep, int precision){
    CSR R = {0};
    int st = csc_to_csr_with_sym(M, expand_sym, &R);
    if(st) { csr_free(&R); return st; }
    st = dump_dense_rows_from_csr(&R, path, sep, precision);
    csr_free(&R);
    return st;
}


static void dump_L_to_file(const qdldl_solver *s, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        c_eprint("Failed to open %s for writing L dump\n", filename);
        return;
    }

    int n = s->L->n;
    fprintf(f, "%% L factor (lower triangular, CSC format)\n");
    fprintf(f, "%% Format: row  col  value\n");

    for (int col = 0; col < n; col++) {
        int p_start = s->L->p[col];
        int p_end   = s->L->p[col + 1];
        for (int idx = p_start; idx < p_end; idx++) {
            int row = s->L->i[idx];
            double val = s->L->x[idx];
            // "row col val" 형태로 출력
            fprintf(f, "%d %d %.15e\n", row, col, val);
        }
    }

    fclose(f);
}


// ==== Dense 0/1 mask dumper ====
// CSC -> CSR 변환은 기존 함수를 재사용(csc_to_csr_with_sym).
// tol > 0 이면 |x| <= tol 은 0 으로 간주.

static int dump_dense_mask_from_csr(const CSR* R,
                                    const char* path,
                                    const char* sep) {
    if(!R || !path || !sep) return -1;
    FILE* f = fopen(path, "w");
    if(!f) return -2;

    // 한 행씩 0/1 출력 (메모리 O(n))
    unsigned char* rowmask = (unsigned char*)c_calloc(R->n, sizeof(unsigned char));
    if(!rowmask){ fclose(f); return -3; }

    for(OSQPInt i=0;i<R->m;++i){
        memset(rowmask, 0, (size_t)R->n * sizeof(unsigned char));

        for(OSQPInt k=R->rp[i]; k<R->rp[i+1]; ++k){
            OSQPInt j = R->ci[k];
            rowmask[j] = 1; // 값 무시, 존재 여부만 표시
        }

        for(OSQPInt j=0;j<R->n;++j){
            fputc(rowmask[j] ? '1' : '0', f);
            if(j+1 < R->n) fputs(sep, f);
        }
        fputc('\n', f);
    }

    c_free(rowmask);
    fclose(f);
    return 0;
}

// tol 적용 버전: |x| > tol 인 항만 1로 인정.
// (CSR 값 배열을 tol에 따라 걸러서 마스크 생성)
static int dump_dense_mask_from_csr_with_tol(const CSR* R,
                                             const char* path,
                                             const char* sep,
                                             double tol) {
    if(!R || !path || !sep) return -1;
    FILE* f = fopen(path, "w");
    if(!f) return -2;

    unsigned char* rowmask = (unsigned char*)c_calloc(R->n, sizeof(unsigned char));
    if(!rowmask){ fclose(f); return -3; }

    for(OSQPInt i=0;i<R->m;++i){
        memset(rowmask, 0, (size_t)R->n * sizeof(unsigned char));

        for(OSQPInt k=R->rp[i]; k<R->rp[i+1]; ++k){
            OSQPInt j = R->ci[k];
            OSQPFloat v = R->x ? R->x[k] : (OSQPFloat)1.0;
            if ((double)((v>=0)?v:-v) > tol) rowmask[j] = 1;
        }

        for(OSQPInt j=0;j<R->n;++j){
            fputc(rowmask[j] ? '1' : '0', f);
            if(j+1 < R->n) fputs(sep, f);
        }
        fputc('\n', f);
    }

    c_free(rowmask);
    fclose(f);
    return 0;
}

// 통합 엔트리: CSC -> CSR(필요시 대칭 확장) -> 0/1 마스크 저장
// expand_sym: 0 원본 그대로, 1 대칭 확장(A + A^T - diag(A))
// sep: ","(CSV) 또는 "\t"(TSV) 등
// tol <= 0  이면 값 무관 존재여부(구조)만 1, 0 처리
// tol >  0  이면 |x| > tol 일 때만 1
static int dump_matrix_mask_shape(const OSQPCscMatrix* M, const char* path,
                                  int expand_sym, const char* sep, double tol) {
    CSR R = {0};
    int st = csc_to_csr_with_sym(M, expand_sym, &R);
    if(st) { csr_free(&R); return st; }

    if (tol > 0.0)
        st = dump_dense_mask_from_csr_with_tol(&R, path, sep, tol);
    else
        st = dump_dense_mask_from_csr(&R, path, sep);

    csr_free(&R);
    return st;
}

// ========================================
// ADMM iteration tracking wrapper for osqp_api.c
// Define the globals HERE (in OSQP library) instead of in qdldl.c
// ========================================
int g_qdldl_current_admm_iteration = 0;
int g_qdldl_enable_admm_solve_logging = 0;
int g_qdldl_current_sample_id = -1;

void osqp_set_qdldl_admm_iteration(OSQPInt iter, OSQPInt sample_id, OSQPInt enable) {
    g_qdldl_current_admm_iteration = (int)iter;
    g_qdldl_current_sample_id = (int)sample_id;
    g_qdldl_enable_admm_solve_logging = (int)enable;
}

void update_settings_linsys_solver_qdldl(qdldl_solver*       s,
                                         const OSQPSettings* settings) {
    /* No settings to update */
    OSQP_UnusedVar(s);
    OSQP_UnusedVar(settings);
    return;
}

void warm_start_linsys_solver_qdldl(qdldl_solver*      s,
                                    const OSQPVectorf* x) {
    /* Warm starting not used by direct solvers */
    OSQP_UnusedVar(s);
    OSQP_UnusedVar(x);
    return;
}

#ifndef OSQP_EMBEDDED_MODE

// Free LDL Factorization structure
void free_linsys_solver_qdldl(qdldl_solver* s) {
    if (s) {
        if (s->L) {
            if (s->L->p) c_free(s->L->p);
            if (s->L->i) c_free(s->L->i);
            if (s->L->x) c_free(s->L->x);
            c_free(s->L);
        }

        if (s->P)           c_free(s->P);
        if (s->Dinv)        c_free(s->Dinv);
        if (s->bp)          c_free(s->bp);
        if (s->sol)         c_free(s->sol);
        if (s->rho_inv_vec) c_free(s->rho_inv_vec);

        // These are required for matrix updates
        if (s->KKT)       csc_spfree(s->KKT);
        if (s->PtoKKT)    c_free(s->PtoKKT);
        if (s->AtoKKT)    c_free(s->AtoKKT);
        if (s->rhotoKKT)  c_free(s->rhotoKKT);

        if (s->adj)         c_free(s->adj);

        // QDLDL workspace
        if (s->D)         c_free(s->D);
        if (s->etree)     c_free(s->etree);
        if (s->Lnz)       c_free(s->Lnz);
        if (s->iwork)     c_free(s->iwork);
        if (s->bwork)     c_free(s->bwork);
        if (s->fwork)     c_free(s->fwork);
        c_free(s);

    }
}


/**
 * Compute LDL factorization of matrix A
 * @param  A    Matrix to be factorized
 * @param  p    Private workspace
 * @param  nvar Number of QP variables
 * @return      exitstatus (0 is good)
 */
static OSQPInt LDL_factor(OSQPCscMatrix* A,
                          qdldl_solver*  p,
                          OSQPInt        nvar) {

    OSQPInt sum_Lnz;
    OSQPInt factor_status;

    // Compute elimination tree
    osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_SYM_FAC);
    sum_Lnz = QDLDL_etree(A->n, A->p, A->i, p->iwork, p->Lnz, p->etree);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_SYM_FAC);

    if (sum_Lnz < 0){
      // Error
      c_eprint("Error in KKT matrix LDL factorization when computing the elimination tree.");
      if(sum_Lnz == -1){
        c_eprint("Matrix is not perfectly upper triangular.");
      }
      else if(sum_Lnz == -2){
        c_eprint("Integer overflow in L nonzero count.");
      }
      return sum_Lnz;
    }

    // Allocate memory for Li and Lx
    p->L->i = (OSQPInt *)c_malloc(sizeof(OSQPInt)*sum_Lnz);
    p->L->x = (OSQPFloat *)c_malloc(sizeof(OSQPFloat)*sum_Lnz);
    p->L->nzmax = sum_Lnz;

    // Factor matrix
    osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);
    factor_status = QDLDL_factor_right_looking(A->n, A->p, A->i, A->x,
                                 p->L->p, p->L->i, p->L->x,
                                 p->D, p->Dinv, p->Lnz,
                                 p->etree, p->bwork, p->iwork, p->fwork);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);
    g_solve_call_count = 0;  // Reset solve counter after factorization

#ifdef QDLDL_ENABLE_SAMPLE_LOGGING
    // Store factor results for sample logging (will be saved on first solve)
    if (factor_status >= 0) {
        QDLDL_store_factor_for_sample(A->n, A->p, A->i, A->x,
                                       p->L->p, p->L->i, p->L->x,
                                       p->D, p->Dinv, p->etree, p->Lnz);
    }
#endif

    if (factor_status < 0){
      // Error
      c_eprint("Error in KKT matrix LDL factorization when computing the nonzero elements. There are zeros in the diagonal matrix");
      return factor_status;
    } else if (factor_status < nvar) {
      // Error: Number of positive elements of D should be equal to nvar
      c_eprint("Error in KKT matrix LDL factorization when computing the nonzero elements. The problem seems to be non-convex");
      return -2;
    }

    return 0;

}

/* MODIFIED */
static OSQPInt permute_KKT(OSQPCscMatrix** KKT,
                           qdldl_solver*   p,
                           OSQPInt         Pnz,
                           OSQPInt         Anz,
                           OSQPInt         m,
                           OSQPInt*        PtoKKT,
                           OSQPInt*        AtoKKT,
                           OSQPInt*        rhotoKKT) {
    OSQPFloat* info;
    OSQPInt*   Pinv;
    OSQPInt*   KtoPKPt;
    OSQPInt    i;

    OSQPCscMatrix* KKT_temp;

    static int run_counter = 0;

    info = (OSQPFloat *)c_malloc(AMD_INFO * sizeof(OSQPFloat));

    // P_ref.txt에서 permutation vector 읽기
    {
        FILE* file = fopen(OSQP_PERM_PATH, "r");
        if (file) {
            OSQPInt n_kkt = (*KKT)->n;
            for (i = 0; i < n_kkt; i++) {
#ifdef OSQP_USE_LONG
                if (fscanf(file, "%lld", (long long*)&p->P[i]) != 1) {
                    c_eprint("Failed to read permutation value at index %lld from %s.", (long long)i, OSQP_PERM_PATH);
                    fclose(file);
                    c_free(info);
                    return -1;
                }
#else
                if (fscanf(file, "%d", &p->P[i]) != 1) {
                    c_eprint("Failed to read permutation value at index %d from %s.", i, OSQP_PERM_PATH);
                    fclose(file);
                    c_free(info);
                    return -1;
                }
#endif
            }
            fclose(file);
            if (run_counter == 0) {
                printf("[OSQP] Permutation vector loaded from %s (n=%lld)\n",
                       OSQP_PERM_PATH, (long long)n_kkt);
            }
        } else {
            c_eprint("Failed to open file %s for reading permutation vector.", OSQP_PERM_PATH);
            c_free(info);
            return -1;
        }
    }

    // (D) Pinv 및 대칭 퍼뮤테이션
    Pinv = csc_pinv(p->P, (*KKT)->n);
    if (!PtoKKT && !AtoKKT && !rhotoKKT){
        KKT_temp = csc_symperm((*KKT), Pinv, OSQP_NULL, 1);
    } else {
        KtoPKPt = c_malloc((*KKT)->p[(*KKT)->n] * sizeof(OSQPInt));
        KKT_temp = csc_symperm((*KKT), Pinv, KtoPKPt, 1);

        if (PtoKKT){
            for (i = 0; i < Pnz; i++){
                PtoKKT[i] = KtoPKPt[PtoKKT[i]];
            }
        }
        if (AtoKKT){
            for (i = 0; i < Anz; i++){
                AtoKKT[i] = KtoPKPt[AtoKKT[i]];
            }
        }
        if (rhotoKKT){
            for (i = 0; i < m; i++){
                rhotoKKT[i] = KtoPKPt[rhotoKKT[i]];
            }
        }
        c_free(KtoPKPt);
    }

    // // (E) 퍼뮤테이션 후 KKT를 "행렬 모양"으로 저장
    // snprintf(path, sizeof(path),
    //          "../result/KKT_dense/KKT_after_run_%d.csv", run_counter);
    // if (dump_matrix_shape(KKT_temp, path, /*expand_sym=*/1, /*sep=*/",", /*precision=*/17) != 0){
    //     c_eprint("Failed to dump dense matrix (after) to %s.", path);
    // }
    // snprintf(path, sizeof(path),
    //          "../result/KKT_csc/KKT_after_run_%d.csc.txt", run_counter);
    // if (dump_csc_text(KKT_temp, path) != 0) {
    //     c_eprint("Failed to dump CSC (after) to %s.", path);
    // }
    // // (E) 퍼뮤테이션 후: 0/1 마스크 저장
    // snprintf(path, sizeof(path),
    //         "../result/KKT_mask/KKT_mask_after_run_%d.csv", run_counter);
    // if (dump_matrix_mask_shape(KKT_temp, path, /*expand_sym=*/1, ",", /*tol=*/0.0) != 0){
    //     c_eprint("Failed to dump dense mask (after) to %s.", path);
    // }

    // (F) 교체/정리
    csc_spfree((*KKT));
    (*KKT) = KKT_temp;
    c_free(Pinv);
    c_free(info);

    // (G) run id 증가 (before/after/P 파일이 동일 인덱스로 묶임)
    run_counter++;

    return 0;
}

// static OSQPInt permute_KKT(OSQPCscMatrix** KKT,
//                            qdldl_solver*   p,
//                            OSQPInt         Pnz,
//                            OSQPInt         Anz,
//                            OSQPInt         m,
//                            OSQPInt*        PtoKKT,
//                            OSQPInt*        AtoKKT,
//                            OSQPInt*        rhotoKKT) {
//     OSQPFloat* info;
//     OSQPInt    amd_status;
//     OSQPInt*   Pinv;
//     OSQPInt*   KtoPKPt;
//     OSQPInt    i; // Indexing

//     OSQPCscMatrix* KKT_temp;

//     info = (OSQPFloat *)c_malloc(AMD_INFO * sizeof(OSQPFloat));

//     // Compute permutation matrix P using AMD
// #ifdef OSQP_USE_LONG
//     amd_status = amd_l_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (OSQPFloat *)OSQP_NULL, info);
// #else
//     amd_status = amd_order((*KKT)->n, (*KKT)->p, (*KKT)->i, p->P, (OSQPFloat *)OSQP_NULL, info);
// #endif
//     if (amd_status < 0) {
//         // Free Amd info and return an error
//         c_free(info);
//         return amd_status;
//     }


//     // Inverse of the permutation vector
//     Pinv = csc_pinv(p->P, (*KKT)->n);

//     // Permute KKT matrix
//     if (!PtoKKT && !AtoKKT && !rhotoKKT){  // No vectors to be stored
//         // Assign values of mapping
//         KKT_temp = csc_symperm((*KKT), Pinv, OSQP_NULL, 1);
//     }
//     else {
//         // Allocate vector of mappings from unpermuted to permuted
//         KtoPKPt = c_malloc((*KKT)->p[(*KKT)->n] * sizeof(OSQPInt));
//         KKT_temp = csc_symperm((*KKT), Pinv, KtoPKPt, 1);

//         // Update vectors PtoKKT, AtoKKT and rhotoKKT
//         if (PtoKKT){
//             for (i = 0; i < Pnz; i++){
//                 PtoKKT[i] = KtoPKPt[PtoKKT[i]];
//             }
//         }
//         if (AtoKKT){
//             for (i = 0; i < Anz; i++){
//                 AtoKKT[i] = KtoPKPt[AtoKKT[i]];
//             }
//         }
//         if (rhotoKKT){
//             for (i = 0; i < m; i++){
//                 rhotoKKT[i] = KtoPKPt[rhotoKKT[i]];
//             }
//         }

//         // Cleanup vector of mapping
//         c_free(KtoPKPt);
//     }

//     // Cleanup
//     // Free previous KKT matrix and assign pointer to new one
//     csc_spfree((*KKT));
//     (*KKT) = KKT_temp;
//     // Free Pinv
//     c_free(Pinv);
//     // Free Amd info
//     c_free(info);

//     return 0;
// }



// Initialize LDL Factorization structure
OSQPInt init_linsys_solver_qdldl(qdldl_solver**      sp,
                                 const OSQPMatrix*   P,
                                 const OSQPMatrix*   A,
                                 const OSQPVectorf*  rho_vec,
                                 const OSQPSettings* settings,
                                 OSQPInt             polishing) {

    // Define Variables
    OSQPCscMatrix* KKT_temp; // Temporary KKT pointer
    OSQPInt    i;         // Loop counter
    OSQPInt    m, n;      // Dimensions of A
    OSQPInt    n_plus_m;  // Define n_plus_m dimension
    OSQPFloat* rhov;      // used for direct access to rho_vec data when polishing=false
    OSQPFloat  sigma = settings->sigma;

    // Allocate private structure to store KKT factorization
    qdldl_solver* s = c_calloc(1, sizeof(qdldl_solver));
    *sp = s;

    // Size of KKT
    n = P->csc->n;
    m = A->csc->m;
    s->n = n;
    s->m = m;
    n_plus_m = n + m;

    // Scalar parameters
    s->sigma = sigma;
    s->rho_inv = 1. / settings->rho;

    // Polishing flag
    s->polishing = polishing;

    // Link Functions
    s->name            = &name_qdldl;
    s->solve           = &solve_linsys_qdldl;
    s->update_settings = &update_settings_linsys_solver_qdldl;
    s->warm_start      = &warm_start_linsys_solver_qdldl;
    s->adjoint_derivative = &adjoint_derivative_qdldl;


#ifndef OSQP_EMBEDDED_MODE
    s->free = &free_linsys_solver_qdldl;
#endif

#if OSQP_EMBEDDED_MODE != 1
    s->update_matrices = &update_linsys_solver_matrices_qdldl;
    s->update_rho_vec  = &update_linsys_solver_rho_vec_qdldl;
#endif

    // Assign type
    s->type = OSQP_DIRECT_SOLVER;

    // Set number of threads to 1 (single threaded)
    s->nthreads = 1;

    // Sparse matrix L (lower triangular)
    // NB: We don not allocate L completely (CSC elements)
    //      L will be allocated during the factorization depending on the
    //      resulting number of elements.
    s->L = c_calloc(1, sizeof(OSQPCscMatrix));
    s->L->m  = n_plus_m;
    s->L->n  = n_plus_m;
    s->L->nz = -1;
    s->L->p  = (OSQPInt *)c_malloc((n_plus_m+1) * sizeof(QDLDL_int));

    // Diagonal matrix stored as a vector D
    s->Dinv = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);
    s->D    = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Permutation vector P
    s->P    = (QDLDL_int *)c_malloc(sizeof(QDLDL_int) * n_plus_m);

    // Working vector
    s->bp   = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Solution vector
    s->sol  = (QDLDL_float *)c_malloc(sizeof(QDLDL_float) * n_plus_m);

    // Parameter vector
    if (rho_vec)
      s->rho_inv_vec = (OSQPFloat *)c_malloc(sizeof(OSQPFloat) * m);
    // else it is NULL

    // Elimination tree workspace
    s->etree = (QDLDL_int *)c_malloc(n_plus_m * sizeof(QDLDL_int));
    s->Lnz   = (QDLDL_int *)c_malloc(n_plus_m * sizeof(QDLDL_int));

    // Lx and Li are sparsity dependent, so set them to
    // null initially so we don't try to free them prematurely
    s->L->i = OSQP_NULL;
    s->L->x = OSQP_NULL;

    // Preallocate workspace
    s->iwork = (QDLDL_int *)c_malloc(sizeof(QDLDL_int)*(3*n_plus_m));
    s->bwork = (QDLDL_bool *)c_malloc(sizeof(QDLDL_bool)*n_plus_m);
    s->fwork = (QDLDL_float *)c_malloc(sizeof(QDLDL_float)*n_plus_m);

    // Form and permute KKT matrix
    if (polishing){ // Called from polish()

        KKT_temp = form_KKT(P->csc,A->csc,
                            0, //format = 0 means CSC
                            sigma, s->rho_inv_vec, sigma,
                            OSQP_NULL, OSQP_NULL, OSQP_NULL);

        // Permute matrix
        if (KKT_temp)
            permute_KKT(&KKT_temp, s, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indices
        s->PtoKKT = c_malloc(P->csc->p[n] * sizeof(OSQPInt));
        s->AtoKKT = c_malloc(A->csc->p[n] * sizeof(OSQPInt));
        s->rhotoKKT = c_malloc(m * sizeof(OSQPInt));

        // Use p->rho_inv_vec for storing param2 = rho_inv_vec
        if (rho_vec) {
          rhov = rho_vec->values;
          for (i = 0; i < m; i++){
              s->rho_inv_vec[i] = 1. / rhov[i];
          }
        }
        else {
          s->rho_inv = 1. / settings->rho;
        }

        KKT_temp = form_KKT(P->csc,A->csc,
                            0, //format = 0 means CSC format
                            sigma, s->rho_inv_vec, s->rho_inv,
                            s->PtoKKT, s->AtoKKT,s->rhotoKKT);

        // Permute matrix with timing
        if (KKT_temp){
            OSQPTimer* perm_timer = OSQPTimer_new();
            osqp_tic(perm_timer);
            permute_KKT(&KKT_temp, s, P->csc->p[n], A->csc->p[n], m, s->PtoKKT, s->AtoKKT, s->rhotoKKT);
            osqp_set_permutation_time(osqp_toc(perm_timer));
            OSQPTimer_free(perm_timer);
        }
    }

    // Check if matrix has been created
    if (!KKT_temp){
        c_eprint("Error forming and permuting KKT matrix");
        free_linsys_solver_qdldl(s);
        *sp = OSQP_NULL;
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Factorize the KKT matrix with timing
    OSQPTimer* fact_timer = OSQPTimer_new();
    osqp_tic(fact_timer);
    OSQPInt factor_result = LDL_factor(KKT_temp, s, n);
    osqp_set_factorization_time(osqp_toc(fact_timer));
    OSQPTimer_free(fact_timer);

    if (factor_result < 0) {
        csc_spfree(KKT_temp);
        free_linsys_solver_qdldl(s);
        *sp = OSQP_NULL;
        // printf("Error in KKT matrix LDL factorization. The problem seems to be non-convex.\n");
        return OSQP_NONCVX_ERROR;
    } else {
        // Successful factorization
        // printf("KKT matrix LDL factorization successful.\n");
    }

    // dump_L_to_file(s, "/home/mms-wonchan/Studies/OSQP/Precision/L.txt");

    if (polishing){ // If KKT passed, assign it to KKT_temp
        // Polish, no need for KKT_temp
        csc_spfree(KKT_temp);
    }
    else { // If not embedded option 1 copy pointer to KKT_temp. Do not free it.
        s->KKT = KKT_temp;

        // Save factorization result for ASIC testcase (every FACTORIZATION_TESTCASE_SAVE_INTERVAL runs)
        // This is for init_linsys_solver_qdldl which is called during osqp_setup
        // Note: g_testcase_count is incremented after scaling in scale_data()
        // Since init is called after scaling, g_testcase_count - 1 gives us the current sample ID
        OSQPInt sample_id = g_testcase_count - 1;
        if (sample_id >= 0 && (sample_id % FACTORIZATION_TESTCASE_SAVE_INTERVAL == 0)) {
            save_factorization_result(sample_id, s->KKT->n,
                                       s->KKT->p, s->KKT->i, s->KKT->x,
                                       s->L->p, s->L->i, s->L->x,
                                       s->D, s->Dinv);
        }
    }


    // No error
    return 0;
}

#endif  // OSQP_EMBEDDED_MODE

const char* name_qdldl(qdldl_solver* s) {
    OSQP_UnusedVar(s);

    return "QDLDL v" STRINGIZE(QDLDL_VERSION_MAJOR) "." STRINGIZE(QDLDL_VERSION_MINOR) "." STRINGIZE(QDLDL_VERSION_PATCH);
}


/* solve P'LDL'P x = b for x */
static void LDLSolve(OSQPFloat*           x,
                     const OSQPFloat*     b,
                     const OSQPCscMatrix* L,
                     const OSQPFloat*     Dinv,
                     const OSQPInt*       P,
                     OSQPFloat*           bp) {

  OSQPInt j;
  OSQPInt n = L->n;

  osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);

  // permute_x(L->n, bp, b, P);
  for (j = 0 ; j < n ; j++) bp[j] = b[P[j]];

  QDLDL_solve(L->n, L->p, L->i, L->x, Dinv, bp);

  // permutet_x(L->n, x, bp, P);
  for (j = 0 ; j < n ; j++) x[P[j]] = bp[j];

  osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);
}

#ifdef QDLDL_ENABLE_SAMPLE_LOGGING
/* Extended LDLSolve that saves sample data only if factor was recently called */
static void LDLSolve_with_logging(OSQPFloat*           x,
                                   const OSQPFloat*     b,
                                   qdldl_solver*        s) {

  OSQPInt j;
  OSQPInt n = s->L->n;
  OSQPFloat* bp = s->bp;

  osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);

  // permute_x(L->n, bp, b, P);
  for (j = 0 ; j < n ; j++) bp[j] = b[s->P[j]];

  // Save b_input (permuted) before solve
  OSQPFloat* b_input_copy = (OSQPFloat*)c_malloc(sizeof(OSQPFloat) * n);
  if (b_input_copy) {
    for (j = 0; j < n; j++) b_input_copy[j] = bp[j];
  }

  QDLDL_solve(s->L->n, s->L->p, s->L->i, s->L->x, s->Dinv, bp);

  // Save sample only if there's a pending factor (factor was recently called)
  // This will save once per factor call, not on every solve
  QDLDL_save_sample_on_solve(b_input_copy, bp);

  if (b_input_copy) c_free(b_input_copy);

  // permutet_x(L->n, x, bp, P);
  for (j = 0 ; j < n ; j++) x[s->P[j]] = bp[j];

  osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);
}
#endif


OSQPInt solve_linsys_qdldl(qdldl_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter) {

  OSQPInt    j;
  OSQPInt    n = s->n;
  OSQPInt    m = s->m;
  OSQPInt    N = s->L->n;  // KKT dimension (n + m)
  OSQPFloat* bv = b->values;

  // Direct solver doesn't care about the ADMM iteration
  OSQP_UnusedVar(admm_iter);

  // Determine sample_id and whether to save
  OSQPInt sample_id = g_testcase_count - 1;
  OSQPInt is_save_sample = (sample_id >= 0 &&
                            (sample_id % SOLVE_TESTCASE_SAVE_INTERVAL == 0));
  OSQPInt is_first_solve = (g_solve_call_count == 0);

  // Use static flags to ensure saves happen only once per QP
  // (prevents overwriting by rho update refactorizations that reset g_solve_call_count)
  static OSQPInt s_intermediate_saved = 0;
  static OSQPInt s_final_saved_id = -1;  // sample_id of last saved final
  if (sample_id != 0) s_intermediate_saved = 0;  // reset for next QP
  OSQPInt save_intermediate = (sample_id == 0 && is_first_solve && !s_intermediate_saved);
  OSQPInt save_final = (is_save_sample && is_first_solve && sample_id != s_final_saved_id);

  g_solve_call_count++;

  osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_SOLVE);

#ifndef OSQP_EMBEDDED_MODE
  if (s->polishing) {
    /* stores solution to the KKT system in b */
    LDLSolve(bv, bv, s->L, s->Dinv, s->P, s->bp);
  } else {
#endif
    if (save_intermediate) {
      // === Sample 0, first solve: step-by-step with intermediate saves ===
      // This runs only once per QP (guarded by s_intermediate_saved + s_final_saved_id)
      s_intermediate_saved = 1;
      s_final_saved_id = sample_id;  // save_final 경로의 덮어쓰기도 방지
      OSQPFloat* bp = s->bp;

      // Save L matrix and Dinv at solve time
      save_solve_L_Dinv(sample_id, N, s->L->p, s->L->i, s->L->x, s->Dinv);

      osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);

      // Permute input
      for (j = 0; j < N; j++) bp[j] = bv[s->P[j]];
      save_solve_intermediate(sample_id, "rhs", N, bp);

      // Step 1: Lsolve
      QDLDL_Lsolve(N, s->L->p, s->L->i, s->L->x, bp);
      save_solve_intermediate(sample_id, "after_Lsolve", N, bp);

      // Step 2: Dinv multiply
      #ifdef OSQP_ROUNDING_MODE
      {
        int old_round = fegetround();
        fesetround(OSQP_ROUNDING_MODE);
        for (j = 0; j < N; j++) bp[j] = bp[j] * s->Dinv[j];
        fesetround(old_round);
      }
      #else
      for (j = 0; j < N; j++) bp[j] *= s->Dinv[j];
      #endif
      save_solve_intermediate(sample_id, "after_Dinv", N, bp);

      // Step 3: Ltsolve
      QDLDL_Ltsolve(N, s->L->p, s->L->i, s->L->x, bp);
      save_solve_intermediate(sample_id, "after_Ltsolve", N, bp);

      // Unpermute → s->sol
      for (j = 0; j < N; j++) s->sol[s->P[j]] = bp[j];

      osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);

      // Also save final solve result
      save_solve_result(sample_id, N, bp);
    } else if (save_final) {
      // === Every 50th sample, first solve: normal solve + save rhs & result ===
      OSQPFloat* bp = s->bp;

      // Save L matrix and Dinv at solve time
      save_solve_L_Dinv(sample_id, N, s->L->p, s->L->i, s->L->x, s->Dinv);

      osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);

      for (j = 0; j < N; j++) bp[j] = bv[s->P[j]];
      save_solve_intermediate(sample_id, "rhs", N, bp);

      QDLDL_solve(N, s->L->p, s->L->i, s->L->x, s->Dinv, bp);
      for (j = 0; j < N; j++) s->sol[s->P[j]] = bp[j];

      osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_BACKSOLVE);

      save_solve_result(sample_id, N, bp);
      s_final_saved_id = sample_id;  // 이 sample에 대해 중복 저장 방지
    } else {
      // === Normal path ===
#ifdef QDLDL_ENABLE_SAMPLE_LOGGING
      LDLSolve_with_logging(s->sol, bv, s);
#else
      LDLSolve(s->sol, bv, s->L, s->Dinv, s->P, s->bp);
#endif
    }

    /* copy x_tilde from s->sol */
    for (j = 0 ; j < n ; j++) {
      bv[j] = s->sol[j];
    }

    /* compute z_tilde from b and s->sol */
    if (s->rho_inv_vec) {
      for (j = 0 ; j < m ; j++) {
        bv[j + n] += s->rho_inv_vec[j] * s->sol[j + n];
      }
    }
    else {
      for (j = 0 ; j < m ; j++) {
        bv[j + n] += s->rho_inv * s->sol[j + n];
      }
    }
#ifndef OSQP_EMBEDDED_MODE
  }
#endif
  osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_SOLVE);
  return 0;
}


#if OSQP_EMBEDDED_MODE != 1

// Update private structure with new P and A
OSQPInt update_linsys_solver_matrices_qdldl(qdldl_solver*     s,
                                            const OSQPMatrix* P,
                                            const OSQPInt*    Px_new_idx,
                                            OSQPInt           P_new_n,
                                            const OSQPMatrix* A,
                                            const OSQPInt*    Ax_new_idx,
                                            OSQPInt           A_new_n) {

    OSQPInt pos_D_count;

    // Update KKT matrix with new P
    update_KKT_P(s->KKT, P->csc, Px_new_idx, P_new_n, s->PtoKKT, s->sigma, 0);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT, A->csc, Ax_new_idx, A_new_n, s->AtoKKT);

    osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);
    pos_D_count = QDLDL_factor_right_looking(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);
    g_solve_call_count = 0;  // Reset solve counter after refactorization

#ifdef QDLDL_ENABLE_SAMPLE_LOGGING
    // Store factor results for sample logging (will be saved on first solve)
    if (pos_D_count >= 0) {
        QDLDL_store_factor_for_sample(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
                                       s->L->p, s->L->i, s->L->x,
                                       s->D, s->Dinv, s->etree, s->Lnz);
    }
#endif

    // Save factorization result for ASIC testcase (every FACTORIZATION_TESTCASE_SAVE_INTERVAL runs)
    // NOTE: 이 함수는 matrix update 시 refactorization에서 호출됨.
    // 초기 factorization은 init_linsys_solver_qdldl에서 이미 저장하므로,
    // 여기서는 저장하지 않음 (덮어쓰기 방지).
    // OSQPInt sample_id = g_testcase_count - 1;
    // if (pos_D_count >= 0 && sample_id >= 0 && (sample_id % FACTORIZATION_TESTCASE_SAVE_INTERVAL == 0)) {
    //     save_factorization_result(sample_id, ...);
    // }

    //number of positive elements in D should match the
    //dimension of P if P + \sigma I is PD.   Error otherwise.
    return (pos_D_count == P->csc->n) ? 0 : 1;
}


OSQPInt update_linsys_solver_rho_vec_qdldl(qdldl_solver*      s,
                                           const OSQPVectorf* rho_vec,
                                           OSQPFloat          rho_sc) {

    OSQPInt i;
    OSQPInt retval = 0;
    OSQPInt m = s->m;
    OSQPFloat* rhov;

    // Update internal rho_inv_vec
    if (s->rho_inv_vec) {
      rhov = rho_vec->values;
      for (i = 0; i < m; i++){
          s->rho_inv_vec[i] = 1. / rhov[i];
      }
    }
    else {
      s->rho_inv = 1. / rho_sc;
    }

    // Update KKT matrix with new rho_vec
    update_KKT_param2(s->KKT, s->rho_inv_vec, s->rho_inv, s->rhotoKKT, s->m);

    osqp_profiler_sec_push(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);
    retval = QDLDL_factor_right_looking(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
        s->L->p, s->L->i, s->L->x, s->D, s->Dinv, s->Lnz,
        s->etree, s->bwork, s->iwork, s->fwork);
    osqp_profiler_sec_pop(OSQP_PROFILER_SEC_LINSYS_NUM_FAC);
    g_solve_call_count = 0;  // Reset solve counter after rho refactorization

#ifdef QDLDL_ENABLE_SAMPLE_LOGGING
    // Store factor results for sample logging (will be saved on first solve)
    if (retval >= 0) {
        QDLDL_store_factor_for_sample(s->KKT->n, s->KKT->p, s->KKT->i, s->KKT->x,
                                       s->L->p, s->L->i, s->L->x,
                                       s->D, s->Dinv, s->etree, s->Lnz);
    }
#endif

    return (retval < 0);
}

#endif

#ifndef OSQP_EMBEDDED_MODE

// --------- Derivative functions -------- //

//increment the D colptr by the number of nonzeros
//in a square diagonal matrix.
static void _colcount_diag(OSQPCscMatrix* D,
                           OSQPInt        initcol,
                           OSQPInt        blockcols) {

    OSQPInt j;
    for(j = initcol; j < (initcol + blockcols); j++){
        D->p[j]++;
    }
}

//increment D colptr by the number of nonzeros in M
static void _colcount_block(OSQPCscMatrix* D,
                            OSQPCscMatrix* M,
                            OSQPInt        initcol,
                            OSQPInt        istranspose) {

    OSQPInt nnzM, j;

    if(istranspose){
        nnzM = M->p[M->n];
        for (j = 0; j < nnzM; j++){
            D->p[M->i[j] + initcol]++;
        }
    }
    else {
        //just add the column count
        for (j = 0; j < M->n; j++){
            D->p[j + initcol] += M->p[j+1] - M->p[j];
        }
    }
}

static void _colcount_to_colptr(OSQPCscMatrix* D) {

    OSQPInt j, count;
    OSQPInt currentptr = 0;

    for(j = 0; j <= D->n; j++){
        count        = D->p[j];
        D->p[j]      = currentptr;
        currentptr  += count;
    }
}

//populate values from M using the K colptr as indicator of
//next fill location in each row
static void _fill_block(OSQPCscMatrix* K,
                        OSQPCscMatrix* M,
                        OSQPInt*       index_mapping,
                        OSQPInt        initrow,
                        OSQPInt        initcol,
                        OSQPInt        istranspose) {
    OSQPInt ii, jj, row, col, dest;

    for(ii=0; ii < M->n; ii++){
        for(jj = M->p[ii]; jj < M->p[ii+1]; jj++){
            if(istranspose){
                col = M->i[jj] + initcol;
                row = ii + initrow;
            }
            else {
                col = ii + initcol;
                row = M->i[jj] + initrow;
            }

            dest       = K->p[col]++;
            K->i[dest] = row;
            K->x[dest] = M->x[jj];
            if (index_mapping != OSQP_NULL) { index_mapping[jj] = dest; }
        }
    }
}

static void _fill_diag_values(OSQPCscMatrix* K,
                              OSQPInt*       index_mapping,
                              OSQPInt        initrow,
                              OSQPInt        initcol,
                              OSQPFloat*     values,
                              OSQPFloat      value_scalar,
                              OSQPInt        n) {

    OSQPInt j, dest, row, col;
    for (j = 0; j < n; j++) {
        row         = j + initrow;
        col         = j + initcol;
        dest        = K->p[col];
        K->i[dest]  = row;
        if (values != OSQP_NULL) {
            K->x[dest] = values[j];
        } else {
            K->x[dest] = value_scalar;
        }
        K->p[col]++;
        if (index_mapping != OSQP_NULL) { index_mapping[j] = dest; }
    }
}

static void _backshift_colptrs(OSQPCscMatrix* K) {

    int j;
    for(j = K->n; j > 0; j--){
        K->p[j] = K->p[j-1];
    }
    K->p[0] = 0;
}

static void _adj_perturb(OSQPCscMatrix* D,
                         OSQPFloat      eps) {
    OSQPInt j, dest;

    dest = 0;
    for (j = 0; j < D->m / 2; j++) {
        dest = D->p[j+1]-1;
        D->x[dest] += eps;
    }
    for (j = D->m / 2; j < D->m; j++) {
        dest = D->p[j+1]-1;
        D->x[dest] -= eps;
    }
}

static void _adj_assemble_csc(OSQPCscMatrix*     D,
                              const OSQPMatrix*  P_full,
                              const OSQPMatrix*  G,
                              const OSQPMatrix*  A_eq,
                              const OSQPMatrix*  GDiagLambda,
                              const OSQPVectorf* slacks) {

    OSQPInt n = OSQPMatrix_get_m(P_full);
    OSQPInt x = OSQPMatrix_get_m(G);        // No. of inequality constraints
    OSQPInt y = OSQPMatrix_get_m(A_eq);     // No. of equality constraints

    OSQPInt j;
    //use D.p to hold nnz entries in each column of the D matrix
    for (j=0; j <= 2*(n+x+y); j++){D->p[j] = 0;}

    _colcount_diag(D, 0, n+x+y);
    _colcount_block(D, P_full->csc, n+x+y, 0);
    _colcount_block(D, G->csc, n+x+y, 0);
    _colcount_block(D, A_eq->csc, n+x+y, 0);
    _colcount_block(D, GDiagLambda->csc, n+x+y+n, 1);
    _colcount_diag(D, n+x+y+n, x);
    _colcount_block(D, A_eq->csc, n+x+y+n+x, 1);
    _colcount_diag(D, n+x+y, n+x+y);

    //cumsum total entries to convert to D.p
    _colcount_to_colptr(D);

    _fill_diag_values(D, OSQP_NULL, 0, 0, OSQP_NULL, 1, n+x+y);
    _fill_block(D, P_full->csc, OSQP_NULL, 0, n+x+y, 0);
    _fill_block(D, G->csc, OSQP_NULL, n, n+x+y, 0);
    _fill_block(D, A_eq->csc, OSQP_NULL, n+x, n+x+y, 0);
    _fill_block(D, GDiagLambda->csc, OSQP_NULL, 0, n+x+y+n, 1);
    _fill_diag_values(D, OSQP_NULL, n, n+x+y+n, slacks->values, 0, x);
    _fill_block(D, A_eq->csc, OSQP_NULL, 0, n+x+y+n+x, 1);
    _fill_diag_values(D, OSQP_NULL, n+x+y, n+x+y, OSQP_NULL, 0, n+x+y);

    _backshift_colptrs(D);

}

OSQPInt adjoint_derivative_qdldl(qdldl_solver**     s,
                                 const OSQPMatrix*  P_full,
                                 const OSQPMatrix*  G,
                                 const OSQPMatrix*  A_eq,
                                 const OSQPMatrix*  GDiagLambda,
                                 const OSQPVectorf* slacks,
                                       OSQPVectorf* rhs) {
    /* We don't currently reuse the solver for the adjoint computations */
    OSQP_UnusedVar(s);

    OSQPInt retval = 0;

    OSQPInt n = OSQPMatrix_get_m(P_full);
    OSQPInt n_ineq = OSQPMatrix_get_m(G);
    OSQPInt n_eq = OSQPMatrix_get_m(A_eq);

    // Get maximum number of nonzero elements (only upper triangular part)
    OSQPInt P_full_nnz = OSQPMatrix_get_nz(P_full);
    OSQPInt G_nnz = OSQPMatrix_get_nz(G);
    OSQPInt A_eq_nnz = OSQPMatrix_get_nz(A_eq);

    OSQPInt nnzKKT = n + n_ineq + n_eq +           // Number of diagonal elements in I (+eps)
                   P_full_nnz +                  // Number of elements in P_full
                   G_nnz +                       // Number of nonzeros in G
                   A_eq_nnz +                    // Number of nonzeros in A_eq
                   G_nnz +                       // Number of nonzeros in G'
                   n_ineq +                      // Number of diagonal elements in slacks
                   A_eq_nnz +                    // Number of nonzeros in A_eq'
                   n + n_ineq + n_eq;            // Number of -eps entries on diagonal

    OSQPInt dim = 2 * (n + n_ineq + n_eq);
    OSQPCscMatrix* adj = csc_spalloc(dim, dim, nnzKKT, 1, 0);
    if (!adj) return osqp_error(OSQP_MEM_ALLOC_ERROR);

    _adj_assemble_csc(adj, P_full, G, A_eq, GDiagLambda, slacks);

    OSQPMatrix *adj_matrix = OSQPMatrix_new_from_csc(adj, 1);

    if (!adj_matrix) {
        retval = OSQP_MEM_ALLOC_ERROR;
        goto adj_alloc_fail;
    }

    _adj_perturb(adj, 1e-6);

    // ----------------------------
    // QDLDL formulation + solve
    // ----------------------------
    const QDLDL_int   An   = dim;
    QDLDL_int i; // Counter

    //data for L and D factors
    QDLDL_int Ln = An;

    QDLDL_float *Lx = OSQP_NULL;
    QDLDL_int   *Li = OSQP_NULL;

    QDLDL_int   *Lp   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
    QDLDL_float *D    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    QDLDL_float *Dinv = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //permutation
    QDLDL_int   *P    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An));
    QDLDL_int   *Pinv = OSQP_NULL;

    //data for elim tree calculation
    QDLDL_int *etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    QDLDL_int *Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    QDLDL_int  sumLnz;

    //working data for factorisation
    QDLDL_int   *iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
    QDLDL_bool  *bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
    QDLDL_float *fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    //Data for results of A\b
    QDLDL_float *x = OSQP_NULL;
    QDLDL_float *x_work = OSQP_NULL;

    if (!Lp || !D || !Dinv || !P || !etree || !Lnz || !iwork || !bwork || !fwork) {
        retval = OSQP_MEM_ALLOC_ERROR;
        goto mat_comp_alloc_error;
    }

    OSQPInt amd_status;
#ifdef OSQP_USE_LONG
    amd_status = amd_l_order(An, adj->p, adj->i, P, (OSQPFloat *)OSQP_NULL, (OSQPFloat *)OSQP_NULL);
#else
    amd_status = amd_order(An, adj->p, adj->i, P, (OSQPFloat *)OSQP_NULL, (OSQPFloat *)OSQP_NULL);
#endif
    if (amd_status < 0) {
        retval = amd_status;
        goto mat_comp_alloc_error;
    }

    // Inverse of the permutation vector
    Pinv = csc_pinv(P, An);
    if (!Pinv) {
        retval = OSQP_MEM_ALLOC_ERROR;
        goto inv_perm_fail;
    }

    OSQPCscMatrix* adj_permuted = OSQP_NULL;
    adj_permuted = csc_symperm(adj, Pinv, OSQP_NULL, 1);
    if (!adj_permuted) {
        retval = OSQP_MEM_ALLOC_ERROR;
        goto sym_perm_fail;
    }

    sumLnz = QDLDL_etree(An, adj_permuted->p, adj_permuted->i, iwork, Lnz, etree);

    Li    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*sumLnz);
    Lx    = (QDLDL_float*)malloc(sizeof(QDLDL_float)*sumLnz);

    if (!Li || !Lx) {
        retval = OSQP_MEM_ALLOC_ERROR;
        goto csc_alloc_fail;
    }

    QDLDL_factor_right_looking(An, adj_permuted->p, adj_permuted->i, adj_permuted->x, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);

    x = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    x_work = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);

    if (!x || !x_work) {
        retval = OSQP_MEM_ALLOC_ERROR;
        goto x_alloc_fail;
    }

    //when solving A\b, start with x = b
    for (i = 0 ; i < An ; i++) x_work[i] = rhs->values[P[i]];
    QDLDL_solve(Ln, Lp, Li, Lx, Dinv, x_work);
    for (i = 0 ; i < An ; i++) x[P[i]] = x_work[i];

    OSQPVectorf *sol = OSQPVectorf_new(x, An);
    OSQPVectorf *residual = OSQPVectorf_malloc(An);

    if (!sol || !residual) {
        retval = OSQP_MEM_ALLOC_ERROR;
        goto vec_alloc_fail;
    }

    OSQPInt k;
    for (k=0; k<200; k++) {
        OSQPVectorf_copy(residual, rhs);
        OSQPMatrix_Axpy(adj_matrix, sol, residual, 1, -1);
        if (OSQPVectorf_norm_2(residual) < 1e-12) break;

        for (i = 0 ; i < An ; i++) x_work[i] = residual->values[P[i]];
        QDLDL_solve(Ln, Lp, Li, Lx, Dinv, x_work);
        for (i = 0 ; i < An ; i++) residual->values[P[i]] = x_work[i];

        OSQPVectorf_minus(sol, sol, residual);
    }

    // rhs is sized to be the largest possible size needed, so sol might be smaller
    // Therefore, we have to subassign into rhs
    OSQPVectorf_subvector_assign(rhs, OSQPVectorf_data(sol), 0, OSQPVectorf_length(sol), 1.0);

/* Free data based on what failed */
vec_alloc_fail:
    OSQPVectorf_free(sol);
    OSQPVectorf_free(residual);

x_alloc_fail:
    c_free(x);
    c_free(x_work);

csc_alloc_fail:
    c_free(Li);
    c_free(Lx);

sym_perm_fail:
    csc_spfree(adj_permuted);

inv_perm_fail:
    c_free(Pinv);

mat_comp_alloc_error:
    c_free(Lp);
    c_free(D);
    c_free(Dinv);
    c_free(P);
    c_free(etree);
    c_free(Lnz);
    c_free(iwork);
    c_free(bwork);
    c_free(fwork);

adj_alloc_fail:
    OSQPMatrix_free(adj_matrix);
    csc_spfree(adj);

    return retval;
}

#endif

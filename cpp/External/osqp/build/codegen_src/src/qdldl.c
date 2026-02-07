/*
 * This file is part of QDLDL, a library for performing the LDL^T factorization
 * of a symmetric indefinite matrix.
 *
 * QDLDL is part of the OSQP project, and is available at https://github.com/osqp/qdldl.
 *
 * Copyright 2018, Paul Goulart, Bartolomeo Stellato, Goran Banjac, The OSQP developers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-ExternalRef: PACKAGE_MANAGER purl pkg:github/osqp/qdldl
 */
#include "qdldl.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>

#if defined(OSQP_USE_TRUNCATE) || defined(OSQP_USE_ROUND_TO_EVEN)
#include <fenv.h>
#endif

/* Unified rounding mode macro:
 *   OSQP_USE_TRUNCATE     → FE_TOWARDZERO
 *   OSQP_USE_ROUND_TO_EVEN → FE_TONEAREST
 */
#if defined(OSQP_USE_TRUNCATE)
  #define OSQP_ROUNDING_MODE FE_TOWARDZERO
#elif defined(OSQP_USE_ROUND_TO_EVEN)
  #define OSQP_ROUNDING_MODE FE_TONEAREST
#endif

#define QDLDL_UNKNOWN (-1)
#define QDLDL_USED (1)
#define QDLDL_UNUSED (0)

// ========================================
// L의 CSR 저장 구조 (Lsolve용 adder tree 지원)
// L을 CSR로 저장하면 L^T 관점에서는 CSC가 됨
// ========================================
static QDLDL_int* g_L_csr_rowptr = NULL;   // row pointers (size n+1)
static QDLDL_int* g_L_csr_colind = NULL;   // column indices (size nnz_L)
static QDLDL_float* g_L_csr_values = NULL; // values (size nnz_L)
static QDLDL_int g_L_csr_n = 0;            // matrix dimension
static QDLDL_int g_L_csr_nnz = 0;          // number of nonzeros
static int g_L_csr_initialized = 0;        // initialization flag

// Adder tree용 임시 버퍼 (동적 할당 최소화)
static QDLDL_float* g_adder_tree_buffer = NULL;
static QDLDL_int g_adder_tree_buffer_size = 0;

// CSR 메모리 해제
static void free_L_csr(void) {
    if (g_L_csr_rowptr) { free(g_L_csr_rowptr); g_L_csr_rowptr = NULL; }
    if (g_L_csr_colind) { free(g_L_csr_colind); g_L_csr_colind = NULL; }
    if (g_L_csr_values) { free(g_L_csr_values); g_L_csr_values = NULL; }
    if (g_adder_tree_buffer) { free(g_adder_tree_buffer); g_adder_tree_buffer = NULL; }
    g_L_csr_n = 0;
    g_L_csr_nnz = 0;
    g_L_csr_initialized = 0;
    g_adder_tree_buffer_size = 0;
}

// CSC에서 CSR로 변환 (L matrix)
// CSC: Lp (col ptr), Li (row ind), Lx (values)
// CSR: rowptr, colind, values
static void convert_csc_to_csr(QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li, const QDLDL_float* Lx) {
    QDLDL_int nnz = Lp[n];
    QDLDL_int i, j, col, row, dest;
    QDLDL_int* row_counts;

    // 기존 CSR 해제
    free_L_csr();

    // 메모리 할당
    g_L_csr_rowptr = (QDLDL_int*)malloc(sizeof(QDLDL_int) * (n + 1));
    g_L_csr_colind = (QDLDL_int*)malloc(sizeof(QDLDL_int) * nnz);
    g_L_csr_values = (QDLDL_float*)malloc(sizeof(QDLDL_float) * nnz);
    row_counts = (QDLDL_int*)calloc(n, sizeof(QDLDL_int));

    if (!g_L_csr_rowptr || !g_L_csr_colind || !g_L_csr_values || !row_counts) {
        free_L_csr();
        if (row_counts) free(row_counts);
        return;
    }

    g_L_csr_n = n;
    g_L_csr_nnz = nnz;

    // Step 1: 각 row의 nnz 카운트
    for (i = 0; i < nnz; i++) {
        row_counts[Li[i]]++;
    }

    // Step 2: rowptr 계산 (cumsum)
    g_L_csr_rowptr[0] = 0;
    for (i = 0; i < n; i++) {
        g_L_csr_rowptr[i + 1] = g_L_csr_rowptr[i] + row_counts[i];
    }

    // Step 3: row_counts를 현재 삽입 위치로 리셋
    for (i = 0; i < n; i++) {
        row_counts[i] = g_L_csr_rowptr[i];
    }

    // Step 4: CSC 순회하며 CSR에 데이터 채우기
    for (col = 0; col < n; col++) {
        for (j = Lp[col]; j < Lp[col + 1]; j++) {
            row = Li[j];
            dest = row_counts[row];
            g_L_csr_colind[dest] = col;
            g_L_csr_values[dest] = Lx[j];
            row_counts[row]++;
        }
    }

    free(row_counts);
    g_L_csr_initialized = 1;

    // Adder tree 버퍼 할당 (최대 row nnz 크기)
    QDLDL_int max_row_nnz = 0;
    for (i = 0; i < n; i++) {
        QDLDL_int row_nnz = g_L_csr_rowptr[i + 1] - g_L_csr_rowptr[i];
        if (row_nnz > max_row_nnz) max_row_nnz = row_nnz;
    }
    // 2의 거듭제곱으로 올림
    QDLDL_int buf_size = 1;
    while (buf_size < max_row_nnz) buf_size <<= 1;
    g_adder_tree_buffer = (QDLDL_float*)malloc(sizeof(QDLDL_float) * buf_size);
    g_adder_tree_buffer_size = buf_size;
}

// Adder tree 합산 (in-place, 버퍼 사용)
// products 배열의 값들을 트리 구조로 합산
static QDLDL_float adder_tree_sum_qdldl(QDLDL_float* buffer, QDLDL_int count) {
    if (count == 0) return 0.0f;
    if (count == 1) return buffer[0];

    // 2의 거듭제곱으로 패딩
    QDLDL_int padded_size = 1;
    while (padded_size < count) padded_size <<= 1;

    // 남은 공간은 0으로 채움
    for (QDLDL_int i = count; i < padded_size; i++) {
        buffer[i] = 0.0f;
    }

    // Adder tree 레벨별 수행
    QDLDL_int current_size = padded_size;
    while (current_size > 1) {
        QDLDL_int half = current_size / 2;
        for (QDLDL_int i = 0; i < half; i++) {
            QDLDL_float sum = buffer[2*i] + buffer[2*i + 1];
#ifdef OSQP_ROUNDING_MODE
            // rounding mode는 fesetround(OSQP_ROUNDING_MODE)로 이미 설정됨
            volatile QDLDL_float truncated = sum;
            buffer[i] = truncated;
#else
            buffer[i] = sum;
#endif
        }
        current_size = half;
    }

    return buffer[0];
}

// ========================================
// QDLDL Sample Data Logging
// ========================================
// Enable/disable sample logging via compile flag QDLDL_ENABLE_SAMPLE_LOGGING
#ifdef QDLDL_ENABLE_SAMPLE_LOGGING

#include <sys/stat.h>
#include <errno.h>
#include <stdlib.h>

static int g_qdldl_sample_counter = 0;
static char g_qdldl_sample_dir[512] = {0};
static int g_qdldl_sample_initialized = 0;

// Pending factor data - stores factorization results until first solve
static int g_qdldl_factor_pending = 0;  // Flag: 1 = factor was called, waiting for solve
static QDLDL_int g_pending_n = 0;
static QDLDL_int* g_pending_Ap = NULL;
static QDLDL_int* g_pending_Ai = NULL;
static QDLDL_float* g_pending_Ax = NULL;
static QDLDL_int* g_pending_Lp = NULL;
static QDLDL_int* g_pending_Li = NULL;
static QDLDL_float* g_pending_Lx = NULL;
static QDLDL_float* g_pending_D = NULL;
static QDLDL_float* g_pending_Dinv = NULL;
static QDLDL_int* g_pending_etree = NULL;
static QDLDL_int* g_pending_Lnz = NULL;

// Default sample output base directory (absolute path)
#define QDLDL_DEFAULT_SAMPLE_BASE_DIR "/home/mms-wonchan/git/MPCC_manipulator/result"

// Get precision string based on build configuration
static const char* get_precision_string(void) {
#ifdef OSQP_ROUNDING_MODE
    return "truncate";
#else
    // Standard precision based on QDLDL_float type
    if (sizeof(QDLDL_float) == 8) {
        return "double";
    } else if (sizeof(QDLDL_float) == 4) {
        return "float";
    } else {
        return "unknown";
    }
#endif
}

// Initialize sample logging directory
void QDLDL_init_sample_logging(const char* output_dir) {
    strncpy(g_qdldl_sample_dir, output_dir, sizeof(g_qdldl_sample_dir) - 1);
    g_qdldl_sample_dir[sizeof(g_qdldl_sample_dir) - 1] = '\0';
    g_qdldl_sample_counter = 0;
    g_qdldl_sample_initialized = 1;

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0755);
    }
    printf("[QDLDL] Sample logging initialized: %s\n", output_dir);
}

// Auto-initialize with default directory if not initialized
// Directory name includes precision info: qdldl_samples_<precision>
static void ensure_sample_logging_initialized(void) {
    if (!g_qdldl_sample_initialized) {
        char default_dir[512];
        const char* precision = get_precision_string();

        // Check environment variable first
        const char* env_dir = getenv("QDLDL_SAMPLE_OUTPUT_DIR");
        if (env_dir && env_dir[0] != '\0') {
            // If env var is set, append precision to it
            snprintf(default_dir, sizeof(default_dir), "%s_%s", env_dir, precision);
        } else {
            // Use default base directory with precision suffix
            snprintf(default_dir, sizeof(default_dir), "%s/qdldl_samples_%s",
                     QDLDL_DEFAULT_SAMPLE_BASE_DIR, precision);
        }
        QDLDL_init_sample_logging(default_dir);
    }
}

// Reset sample counter
void QDLDL_reset_sample_counter(void) {
    g_qdldl_sample_counter = 0;
}

// Get current sample count
int QDLDL_get_sample_count(void) {
    return g_qdldl_sample_counter;
}

// Free pending factor data
static void free_pending_factor_data(void) {
    if (g_pending_Ap) { free(g_pending_Ap); g_pending_Ap = NULL; }
    if (g_pending_Ai) { free(g_pending_Ai); g_pending_Ai = NULL; }
    if (g_pending_Ax) { free(g_pending_Ax); g_pending_Ax = NULL; }
    if (g_pending_Lp) { free(g_pending_Lp); g_pending_Lp = NULL; }
    if (g_pending_Li) { free(g_pending_Li); g_pending_Li = NULL; }
    if (g_pending_Lx) { free(g_pending_Lx); g_pending_Lx = NULL; }
    if (g_pending_D) { free(g_pending_D); g_pending_D = NULL; }
    if (g_pending_Dinv) { free(g_pending_Dinv); g_pending_Dinv = NULL; }
    if (g_pending_etree) { free(g_pending_etree); g_pending_etree = NULL; }
    if (g_pending_Lnz) { free(g_pending_Lnz); g_pending_Lnz = NULL; }
    g_qdldl_factor_pending = 0;
    g_pending_n = 0;
}

// Store factor results for later save (called after QDLDL_factor)
void QDLDL_store_factor_for_sample(
    QDLDL_int n,
    const QDLDL_int* Ap, const QDLDL_int* Ai, const QDLDL_float* Ax,
    const QDLDL_int* Lp, const QDLDL_int* Li, const QDLDL_float* Lx,
    const QDLDL_float* D, const QDLDL_float* Dinv,
    const QDLDL_int* etree, const QDLDL_int* Lnz
) {
    // Auto-initialize if not done yet
    ensure_sample_logging_initialized();

    // Free any previous pending data
    free_pending_factor_data();

    QDLDL_int nnz_A = Ap ? Ap[n] : 0;
    QDLDL_int nnz_L = Lp ? Lp[n] : 0;

    g_pending_n = n;

    // Copy Ap
    g_pending_Ap = (QDLDL_int*)malloc(sizeof(QDLDL_int) * (n + 1));
    if (g_pending_Ap) memcpy(g_pending_Ap, Ap, sizeof(QDLDL_int) * (n + 1));

    // Copy Ai
    g_pending_Ai = (QDLDL_int*)malloc(sizeof(QDLDL_int) * nnz_A);
    if (g_pending_Ai) memcpy(g_pending_Ai, Ai, sizeof(QDLDL_int) * nnz_A);

    // Copy Ax
    g_pending_Ax = (QDLDL_float*)malloc(sizeof(QDLDL_float) * nnz_A);
    if (g_pending_Ax) memcpy(g_pending_Ax, Ax, sizeof(QDLDL_float) * nnz_A);

    // Copy Lp
    g_pending_Lp = (QDLDL_int*)malloc(sizeof(QDLDL_int) * (n + 1));
    if (g_pending_Lp) memcpy(g_pending_Lp, Lp, sizeof(QDLDL_int) * (n + 1));

    // Copy Li
    g_pending_Li = (QDLDL_int*)malloc(sizeof(QDLDL_int) * nnz_L);
    if (g_pending_Li) memcpy(g_pending_Li, Li, sizeof(QDLDL_int) * nnz_L);

    // Copy Lx
    g_pending_Lx = (QDLDL_float*)malloc(sizeof(QDLDL_float) * nnz_L);
    if (g_pending_Lx) memcpy(g_pending_Lx, Lx, sizeof(QDLDL_float) * nnz_L);

    // Copy D
    g_pending_D = (QDLDL_float*)malloc(sizeof(QDLDL_float) * n);
    if (g_pending_D) memcpy(g_pending_D, D, sizeof(QDLDL_float) * n);

    // Copy Dinv
    g_pending_Dinv = (QDLDL_float*)malloc(sizeof(QDLDL_float) * n);
    if (g_pending_Dinv) memcpy(g_pending_Dinv, Dinv, sizeof(QDLDL_float) * n);

    // Copy etree
    g_pending_etree = (QDLDL_int*)malloc(sizeof(QDLDL_int) * n);
    if (g_pending_etree) memcpy(g_pending_etree, etree, sizeof(QDLDL_int) * n);

    // Copy Lnz
    g_pending_Lnz = (QDLDL_int*)malloc(sizeof(QDLDL_int) * n);
    if (g_pending_Lnz) memcpy(g_pending_Lnz, Lnz, sizeof(QDLDL_int) * n);

    g_qdldl_factor_pending = 1;
}

// Check if there's a pending factor and save sample with solve results
void QDLDL_save_sample_on_solve(const QDLDL_float* x_input, const QDLDL_float* x_output) {
    if (!g_qdldl_factor_pending) return;  // No pending factor, skip

    // Save the complete sample with factor + solve results
    QDLDL_save_sample(
        g_pending_n,
        g_pending_Ap, g_pending_Ai, g_pending_Ax,
        g_pending_Lp, g_pending_Li, g_pending_Lx,
        g_pending_D, g_pending_Dinv,
        g_pending_etree, g_pending_Lnz,
        x_input, x_output
    );

    // Clear the pending flag (only save once per factor)
    free_pending_factor_data();
}

// Helper to write int array in hex format
static void write_int_hex_sample(const char* filepath, const QDLDL_int* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if (!f) { perror(filepath); return; }
    if (sizeof(QDLDL_int) == 8) {
        for (QDLDL_int i = 0; i < len; ++i)
            fprintf(f, "%016" PRIx64 "\n", (uint64_t)arr[i]);
    } else {
        for (QDLDL_int i = 0; i < len; ++i)
            fprintf(f, "%08" PRIx32 "\n", (uint32_t)arr[i]);
    }
    fclose(f);
}

// Helper to write float array in hex format
static void write_float_hex_sample(const char* filepath, const QDLDL_float* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if (!f) { perror(filepath); return; }
    if (sizeof(QDLDL_float) == 8) {
        for (QDLDL_int i = 0; i < len; ++i) {
            uint64_t u; memcpy(&u, &arr[i], 8);
            fprintf(f, "%016" PRIx64 "\n", u);
        }
    } else {
        for (QDLDL_int i = 0; i < len; ++i) {
            uint32_t u; memcpy(&u, &arr[i], 4);
            fprintf(f, "%08" PRIx32 "\n", u);
        }
    }
    fclose(f);
}

// Helper to write metadata
static void write_sample_metadata(const char* filepath, QDLDL_int n, QDLDL_int nnz_A, QDLDL_int nnz_L) {
    FILE* f = fopen(filepath, "w");
    if (!f) { perror(filepath); return; }
    fprintf(f, "n=%jd\n", (intmax_t)n);
    fprintf(f, "nnz_A=%jd\n", (intmax_t)nnz_A);
    fprintf(f, "nnz_L=%jd\n", (intmax_t)nnz_L);
    fprintf(f, "sizeof_int=%zu\n", sizeof(QDLDL_int));
    fprintf(f, "sizeof_float=%zu\n", sizeof(QDLDL_float));
    fprintf(f, "precision=%s\n", get_precision_string());
#ifdef OSQP_ROUNDING_MODE
    fprintf(f, "truncate=ON\n");
#else
    fprintf(f, "truncate=OFF\n");
#endif
    fclose(f);
}

// Save complete QDLDL sample - public API
void QDLDL_save_sample(
    QDLDL_int n,
    const QDLDL_int* Ap, const QDLDL_int* Ai, const QDLDL_float* Ax,
    const QDLDL_int* Lp, const QDLDL_int* Li, const QDLDL_float* Lx,
    const QDLDL_float* D, const QDLDL_float* Dinv,
    const QDLDL_int* etree, const QDLDL_int* Lnz,
    const QDLDL_float* x_input, const QDLDL_float* x_output
) {
    // Auto-initialize if not done yet
    ensure_sample_logging_initialized();

    if (!g_qdldl_sample_dir) return;

    char sample_dir[512];
    snprintf(sample_dir, sizeof(sample_dir), "%s/sample_%06d", g_qdldl_sample_dir, g_qdldl_sample_counter);

    // Create sample directory
    struct stat st = {0};
    if (stat(sample_dir, &st) == -1) {
        mkdir(sample_dir, 0755);
    }

    char filepath[576];

    // Get array sizes
    QDLDL_int nnz_A = Ap ? Ap[n] : 0;
    QDLDL_int nnz_L = Lp ? Lp[n] : 0;

    // Save metadata
    snprintf(filepath, sizeof(filepath), "%s/metadata.txt", sample_dir);
    write_sample_metadata(filepath, n, nnz_A, nnz_L);

    // Save input matrix A (CSC format)
    snprintf(filepath, sizeof(filepath), "%s/Ap.hex", sample_dir);
    write_int_hex_sample(filepath, Ap, n + 1);

    snprintf(filepath, sizeof(filepath), "%s/Ai.hex", sample_dir);
    write_int_hex_sample(filepath, Ai, nnz_A);

    snprintf(filepath, sizeof(filepath), "%s/Ax.hex", sample_dir);
    write_float_hex_sample(filepath, Ax, nnz_A);

    // Save L factor (CSC format)
    snprintf(filepath, sizeof(filepath), "%s/Lp.hex", sample_dir);
    write_int_hex_sample(filepath, Lp, n + 1);

    snprintf(filepath, sizeof(filepath), "%s/Li.hex", sample_dir);
    write_int_hex_sample(filepath, Li, nnz_L);

    snprintf(filepath, sizeof(filepath), "%s/Lx.hex", sample_dir);
    write_float_hex_sample(filepath, Lx, nnz_L);

    // Save D and Dinv
    snprintf(filepath, sizeof(filepath), "%s/D.hex", sample_dir);
    write_float_hex_sample(filepath, D, n);

    snprintf(filepath, sizeof(filepath), "%s/Dinv.hex", sample_dir);
    write_float_hex_sample(filepath, Dinv, n);

    // Save etree and Lnz
    snprintf(filepath, sizeof(filepath), "%s/etree.hex", sample_dir);
    write_int_hex_sample(filepath, etree, n);

    snprintf(filepath, sizeof(filepath), "%s/Lnz.hex", sample_dir);
    write_int_hex_sample(filepath, Lnz, n);

    // Save x (input: b vector before solve, output: solution after solve)
    if (x_input) {
        snprintf(filepath, sizeof(filepath), "%s/x_input.hex", sample_dir);
        write_float_hex_sample(filepath, x_input, n);
    }

    snprintf(filepath, sizeof(filepath), "%s/x_output.hex", sample_dir);
    write_float_hex_sample(filepath, x_output, n);

    g_qdldl_sample_counter++;
}

#endif // QDLDL_ENABLE_SAMPLE_LOGGING

static inline float init_reciprocal(float d) {
    union { uint32_t i; float f; } u = { .f = d };
    u.i = 0x7EEEEBB3U - u.i;
    return u.f;
}

#ifdef OSQP_ROUNDING_MODE
// Truncate version: use actual division with truncation (toward zero)
float reciprocal_nr_fma(float d) {
    int old_round = fegetround();
    fesetround(OSQP_ROUNDING_MODE);

    // Use actual division instead of Newton-Raphson MAC
    volatile float result = 1.0f / d;

    fesetround(old_round);
    return result;
}
#else
// Standard float version - also use actual division for consistency
float reciprocal_nr_fma(float d) {
    return 1.0f / d;
}
#endif

/* one-hex-word-per-line, MSB-first, width matches the C type */
static void dump_int_hex(const char *path, const QDLDL_int *a, QDLDL_int n) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    if (sizeof(QDLDL_int) == 8) {
        for (QDLDL_int i = 0; i < n; ++i)
            fprintf(f, "%016" PRIx64 "\n", (uint64_t)a[i]);
    } else if (sizeof(QDLDL_int) == 4) {
        for (QDLDL_int i = 0; i < n; ++i)
            fprintf(f, "%08"  PRIx32 "\n", (uint32_t)a[i]);
    } else {
        /* rare case: non 32/64-bit ints */
        for (QDLDL_int i = 0; i < n; ++i) {
            unsigned char b[sizeof(QDLDL_int)];
            memcpy(b, &a[i], sizeof(QDLDL_int));
            for (int k = (int)sizeof(QDLDL_int) - 1; k >= 0; --k) fprintf(f, "%02x", b[k]);
            fputc('\n', f);
        }
    }
    fclose(f);
}

static void dump_float_hex(const char *path, const QDLDL_float *a, QDLDL_int n) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    if (sizeof(QDLDL_float) == 8) {
        for (QDLDL_int i = 0; i < n; ++i) {
            uint64_t u; memcpy(&u, &a[i], 8);
            fprintf(f, "%016" PRIx64 "\n", u);
        }
    } else if (sizeof(QDLDL_float) == 4) {
        for (QDLDL_int i = 0; i < n; ++i) {
            uint32_t u; memcpy(&u, &a[i], 4);
            fprintf(f, "%08"  PRIx32 "\n", u);
        }
    } else {
        /* if someone builds with an exotic float size */
        for (QDLDL_int i = 0; i < n; ++i) {
            unsigned char b[sizeof(QDLDL_float)];
            memcpy(b, &a[i], sizeof(QDLDL_float));
            for (int k = (int)sizeof(QDLDL_float) - 1; k >= 0; --k) fprintf(f, "%02x", b[k]);
            fputc('\n', f);
        }
    }
    fclose(f);
}


static void append_dec_qint(const char *fname, QDLDL_int v) {
    FILE *f = fopen(fname, "a");           // append mode => cumulative log
    if (!f) { perror(fname); return; }
    fprintf(f, "%jd\n", (intmax_t)v);      // decimal, one per line
    fclose(f);
}

/* Compute the elimination tree for a quasidefinite matrix
 * in compressed sparse column form.
 */
QDLDL_int QDLDL_etree(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai, QDLDL_int* work,
                      QDLDL_int* Lnz, QDLDL_int* etree) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;
    QDLDL_int p = 0;
    QDLDL_int sumLnz = 0;

    for(i = 0; i < n; i++) {
        // Zero out Lnz and work.  Set all etree values to unknown
        work[i]  = 0;
        Lnz[i]   = 0;
        etree[i] = QDLDL_UNKNOWN;

        // Abort if A doesn't have at least one entry
        // one entry in every column
        if(Ap[i] == Ap[i + 1]) {
            return -1;
        }
    }

    for(j = 0; j < n; j++) {
        work[j] = j;

        for(p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p];

            // Abort if entries on lower triangle
            if(i > j) {
                return -1;
            };

            while(work[i] != j) {
                if(etree[i] == QDLDL_UNKNOWN) {
                    etree[i] = j;
                }
                Lnz[i]++; // Nonzeros in this column
                work[i] = j;
                i = etree[i];
            }
        }
    }

    // Compute the total nonzeros in L.  This much
    // space is required to store Li and Lx.  Return
    // error code -2 if the nonzero count will overflow
    // its unteger type.
    sumLnz = 0;

    for(i = 0; i < n; i++) {
        if(sumLnz > QDLDL_INT_MAX - Lnz[i]) {
            sumLnz = -2;
            break;
        } else {
            sumLnz += Lnz[i];
        }
    }

    return sumLnz;
}


QDLDL_int QDLDL_factor(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
                       const QDLDL_float* Ax, QDLDL_int* Lp, QDLDL_int* Li, QDLDL_float* Lx,
                       QDLDL_float* D, QDLDL_float* Dinv, const QDLDL_int* Lnz,
                       const QDLDL_int* etree, QDLDL_bool* bwork, QDLDL_int* iwork,
                       QDLDL_float* fwork) {
    QDLDL_int    i = 0;
    QDLDL_int    j = 0;
    QDLDL_int    k = 0;
    QDLDL_int    nnzY = 0;
    QDLDL_int    bidx = 0;
    QDLDL_int    cidx = 0;
    QDLDL_int    nextIdx = 0;
    QDLDL_int    nnzE = 0;
    QDLDL_int    tmpIdx = 0;
    QDLDL_int*   yIdx;
    QDLDL_int*   elimBuffer;
    QDLDL_int*   LNextSpaceInCol;
    QDLDL_float* yVals;
    QDLDL_float  yVals_cidx = 0.0;
    QDLDL_bool*  yMarkers;
    QDLDL_int    positiveValuesInD = 0;

    // printf("In QDLDL_factor\n");

    // printf("QDLDL_float = %zu bytes\n", sizeof(QDLDL_float));

    // Partition working memory into pieces
    yMarkers = bwork;
    yIdx = iwork;
    elimBuffer = iwork + n;
    LNextSpaceInCol = iwork + n * 2;
    yVals = fwork;


    Lp[0] = 0; // First column starts at index zero

    for(i = 0; i < n; i++) {
        // Compute L column indices
        Lp[i + 1] = Lp[i] + Lnz[i]; // cumsum, total at the end

        // Set all Yidx to be 'unused' initially
        // in each column of L, the next available space
        // to start is just the first space in the column
        yMarkers[i] = QDLDL_UNUSED;
        yVals[i] = 0.0;
        D[i] = 0.0;
        LNextSpaceInCol[i] = Lp[i];
    }

    // First element of the diagonal D.
    D[0] = Ax[0];

    if(D[0] == 0.0) {
        return -1;
    }

    if(D[0] > 0.0) {
        positiveValuesInD++;
    }
    // Dinv[0] = 1 / D[0];
    Dinv[0] = reciprocal_nr_fma(D[0]); // Use the NR FMA method for better precision

    // Start from 1 here. The upper LH corner is trivially 0
    // in L b/c we are only computing the subdiagonal elements
    for(k = 1; k < n; k++) {
        // NB : For each k, we compute a solution to
        // y = L(0:(k-1),0:k-1))\b, where b is the kth
        // column of A that sits above the diagonal.
        // The solution y is then the kth row of L,
        // with an implied '1' at the diagonal entry.

        // Number of nonzeros in this row of L
        nnzY = 0; // Number of elements in this row

        // This loop determines where nonzeros
        // will go in the kth row of L, but doesn't
        // compute the actual values
        tmpIdx = Ap[k + 1];

        for(i = Ap[k]; i < tmpIdx; i++) {
            bidx = Ai[i]; // We are working on this element of b

            // Initialize D[k] as the element of this column
            // corresponding to the diagonal place.  Don't use
            // this element as part of the elimination step
            // that computes the k^th row of L
            if(bidx == k) {
                D[k] = Ax[i];
                continue;
            }

            yVals[bidx] = Ax[i]; // Initialise y(bidx) = b(bidx)

            // Use the forward elimination tree to figure
            // out which elements must be eliminated after
            // this element of b
            nextIdx = bidx;

            if(yMarkers[nextIdx] == QDLDL_UNUSED) { // This y term not already visited

                yMarkers[nextIdx] = QDLDL_USED; // I touched this one
                elimBuffer[0] = nextIdx;        // It goes at the start of the current list
                nnzE = 1;                       // Length of unvisited elimination path from here

                nextIdx = etree[bidx];

                while(nextIdx != QDLDL_UNKNOWN && nextIdx < k) {
                    if(yMarkers[nextIdx] == QDLDL_USED)
                        break;

                    yMarkers[nextIdx] = QDLDL_USED; // I touched this one
                    elimBuffer[nnzE] = nextIdx;     // It goes in the current list
                    nnzE++;                         // The list is one longer than before
                    nextIdx = etree[nextIdx];       // One step further along tree

                }

                // Now I put the buffered elimination list into
                // my current ordering in reverse order
                while(nnzE) {
                    yIdx[nnzY++] = elimBuffer[--nnzE];
                }
            }
        }

        // This for loop places nonzeros values in the k^th row
        for(i = (nnzY - 1); i >= 0; i--) {
            //Which column are we working on?
            cidx = yIdx[i];

            // Loop along the elements in this
            // column of L and subtract to solve to y
            tmpIdx = LNextSpaceInCol[cidx];
            yVals_cidx = yVals[cidx];

            #ifdef OSQP_ROUNDING_MODE
            // Truncate version: mul/add with truncation (toward zero)
            int old_round_factor = fegetround();
            fesetround(OSQP_ROUNDING_MODE);

            for(j = Lp[cidx]; j < tmpIdx; j++) {
                // yVals[Li[j]] -= Lx[j] * yVals_cidx;
                volatile float prod = Lx[j] * yVals_cidx;
                yVals[Li[j]] = yVals[Li[j]] - prod;
            }

            Li[tmpIdx] = k;
            Lx[tmpIdx] = yVals_cidx * Dinv[cidx];

            // D[k] -= yVals_cidx * Lx[tmpIdx];
            volatile float prod_D = yVals_cidx * Lx[tmpIdx];
            D[k] = D[k] - prod_D;
            LNextSpaceInCol[cidx]++;

            fesetround(old_round_factor);
            #else
            for(j = Lp[cidx]; j < tmpIdx; j++) {

                yVals[Li[j]] -= Lx[j] * yVals_cidx;
            }

            // Now I have the cidx^th element of y = L\b.
            // so compute the corresponding element of
            // this row of L and put it into the right place
            Li[tmpIdx] = k;
            Lx[tmpIdx] = yVals_cidx * Dinv[cidx];

            // D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
            D[k] -= yVals_cidx * Lx[tmpIdx];
            LNextSpaceInCol[cidx]++;
            #endif

            // Reset the yvalues and indices back to zero and QDLDL_UNUSED
            // once I'm done with them
            yVals[cidx] = 0.0;
            yMarkers[cidx] = QDLDL_UNUSED;

        }

        // Maintain a count of the positive entries
        // in D.  If we hit a zero, we can't factor
        // this matrix, so abort
        if(D[k] == 0.0) {
            // printf("Zero diagonal entry found at k = %" PRId64 "\n", (int64_t)k);
            return -1;
        }

        if(D[k] > 0.0) {
            positiveValuesInD++;
        }

        // Compute the inverse of the diagonal
        // Dinv[k] = 1 / D[k];
        Dinv[k] = reciprocal_nr_fma(D[k]); // Use the NR FMA method for better precision

    }

    // dump_int_hex  ("Ap_skip.hex",    Ap,    n + 1);
    // dump_int_hex  ("Ai_skip.hex",    Ai,    Ap[n]);
    // dump_float_hex("Ax_skip.hex",    Ax,    Ap[n]);
    // append_dec_qint("Ap_n_skip.dec", Ap ? Ap[n] : 0);


    // dump_float_hex("D_skip.hex",     D,     n);
    // dump_float_hex("Dinv_skip.hex",  Dinv,  n);

    // dump_int_hex  ("Lnz_skip.hex",   Lnz,   n);
    // dump_int_hex  ("etree_skip.hex", etree, n);

    // printf("Factorization complete.  %" PRId64 " positive entries in D out of %" PRId64 " total.\n",
    //        (int64_t)positiveValuesInD, (int64_t)n);

    // ========================================
    // Factor 완료 후 L을 CSR로 변환 (Lsolve용 adder tree 지원)
    // L의 CSR은 L^T의 CSC와 동일 (Ltsolve에서도 활용)
    // ========================================
    convert_csc_to_csr(n, Lp, Li, Lx);

    return positiveValuesInD;
}


// ========================================================================
// Right-looking LDL^T factorization (general, same interface as QDLDL_factor)
// ========================================================================
// Uses a dense column representation for the lower-triangular part.
// For each column j (left to right):
//   1. Extract D[j] from the diagonal, compute Dinv[j]
//   2. Scale the sub-diagonal entries by Dinv[j] to get L[:,j]
//   3. Rank-1 update: for each k > j where L[k,j] != 0,
//      subtract L[k,j] * L[i,j] * D[j] from column k (rows i >= k)
// ========================================================================

// Position (0-178) → bank number (0-20)   [kept for hardware variant]
//   pos 0-79:   bank = pos/8        (banks 0-9, 8 words each)
//   pos 80-178: bank = 10+(pos-80)/9 (banks 10-20, 9 words each)
static int pos_to_bank(int pos) {
    if (pos < 80) return pos / 8;
    return 10 + (pos - 80) / 9;
}

// Negate float via sign-bit XOR (matches hardware: value ^ 32'h80000000)
static float negate_float_hw(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u ^= 0x80000000u;
    float r;
    memcpy(&r, &u, 4);
    return r;
}

QDLDL_int QDLDL_factor_right_looking(
    const QDLDL_int n,
    const QDLDL_int* Ap, const QDLDL_int* Ai, const QDLDL_float* Ax,
    QDLDL_int* Lp, QDLDL_int* Li, QDLDL_float* Lx,
    QDLDL_float* D, QDLDL_float* Dinv,
    const QDLDL_int* Lnz, const QDLDL_int* etree,
    QDLDL_bool* bwork, QDLDL_int* iwork, QDLDL_float* fwork)
{
    QDLDL_int positiveValuesInD = 0;
    QDLDL_int i, j, k, p;

    // D update debug log file (first KKT-sized factorization only)
    static int s_kkt_factor_count = 0;
    FILE* d_log_f = NULL;
    if (n > 480 && s_kkt_factor_count == 0) {
        system("mkdir -p ../result/asic_testcases/osqp/factorization");
        d_log_f = fopen("../result/asic_testcases/osqp/factorization/sample_0_metadata.csv", "w");
        if (d_log_f) {
            fprintf(d_log_f, "# D update log for rows 479, 480 (sample 0, n=%d)\n", (int)n);
            fprintf(d_log_f, "# j,k_row,neg_lkj,dense_jk,prod_d,D_after\n");
        }
        s_kkt_factor_count++;
    }

    // ================================================================
    // Step 1: Compute L structural pattern (Li, Lp) from etree
    //         (identical logic to QDLDL_factor's structural phase)
    // ================================================================
    {
        QDLDL_bool* yMarkers        = bwork;
        QDLDL_int*  yIdx            = iwork;
        QDLDL_int*  elimBuffer      = iwork + n;
        QDLDL_int*  LNextSpaceInCol = iwork + 2 * n;

        Lp[0] = 0;
        for (i = 0; i < n; i++) {
            Lp[i + 1] = Lp[i] + Lnz[i];
            yMarkers[i]        = QDLDL_UNUSED;
            LNextSpaceInCol[i] = Lp[i];
        }

        for (k = 1; k < n; k++) {
            QDLDL_int nnzY = 0;
            for (p = Ap[k]; p < Ap[k + 1]; p++) {
                QDLDL_int bidx = Ai[p];
                if (bidx == k) continue;

                QDLDL_int nxt = bidx;
                if (yMarkers[nxt] == QDLDL_UNUSED) {
                    yMarkers[nxt] = QDLDL_USED;
                    QDLDL_int nnzE = 1;
                    elimBuffer[0] = nxt;
                    nxt = etree[bidx];
                    while (nxt != QDLDL_UNKNOWN && nxt < k) {
                        if (yMarkers[nxt] == QDLDL_USED) break;
                        yMarkers[nxt] = QDLDL_USED;
                        elimBuffer[nnzE++] = nxt;
                        nxt = etree[nxt];
                    }
                    while (nnzE) yIdx[nnzY++] = elimBuffer[--nnzE];
                }
            }
            for (i = nnzY - 1; i >= 0; i--) {
                QDLDL_int cidx = yIdx[i];
                Li[LNextSpaceInCol[cidx]] = k;
                LNextSpaceInCol[cidx]++;
                yMarkers[cidx] = QDLDL_UNUSED;
            }
        }
    }

    // ================================================================
    // Step 2: Allocate dense column storage
    //   dense[j * n + i] holds the running value of column j, row i
    //   Only rows i > j are used (strictly lower triangular part).
    //   Diagonal values are stored separately in D[].
    // ================================================================
    QDLDL_float* dense = (QDLDL_float*)calloc((size_t)n * n, sizeof(QDLDL_float));
    if (!dense) return -1;

    // Load A (upper triangular CSC) into dense as full lower triangle
    // A is upper-tri CSC: for each column c, entries have row r <= c
    // We need the lower triangle: dense[c][r] for r > c, plus diagonal
    for (QDLDL_int c = 0; c < n; c++) {
        for (p = Ap[c]; p < Ap[c + 1]; p++) {
            QDLDL_int r = Ai[p];
            if (r == c) {
                D[c] = Ax[p];  // diagonal → D init
            } else {
                // A[r,c] with r < c (upper tri) → symmetric lower entry at col r, row c
                dense[r * n + c] = Ax[p];
            }
        }
    }

    // ================================================================
    // Step 3: Right-looking factorization
    //
    //   For each column j = 0 .. n-1:
    //     1. D[j] is already accumulated (from init + rank-1 updates)
    //     2. Dinv[j] = 1/D[j]
    //     3. Scale sub-diagonal dense[j][i] by Dinv[j] → L[i,j]
    //     4. Rank-1 update: for each k in L[:,j] (k > j):
    //          D[k]          -= L[k,j] * dense_orig[j][k]
    //          dense[k][i]   -= L[k,j] * dense_orig[j][i]   for i > k
    // ================================================================
#ifdef OSQP_ROUNDING_MODE
    int saved_round = fegetround();
    fesetround(OSQP_ROUNDING_MODE);
#endif

    // First column (j=0) diagonal is already in D[0] from init
    for (j = 0; j < n; j++) {
        // D[j] already holds the correct value (init for j=0, updated for j>0)
        QDLDL_float dj = D[j];

        if (dj == 0.0) {
            free(dense);
            if (d_log_f) fclose(d_log_f);
#ifdef OSQP_ROUNDING_MODE
            fesetround(saved_round);
#endif
            return -1;
        }
        if (dj > 0.0) positiveValuesInD++;

        QDLDL_float djinv = reciprocal_nr_fma(dj);
        Dinv[j] = djinv;

        // Scale sub-diagonal entries of column j to get L[:,j]
        // and perform rank-1 update on subsequent columns
        QDLDL_int col_start = Lp[j];
        QDLDL_int col_end   = Lp[j + 1];
        QDLDL_int nnz_col   = col_end - col_start;

        // Scale: L[row, j] = dense[j][row] * Dinv[j]
        for (p = col_start; p < col_end; p++) {
            QDLDL_int row = Li[p];
#ifdef OSQP_ROUNDING_MODE
            volatile float prod = (float)dense[j * n + row] * (float)djinv;
            Lx[p] = (QDLDL_float)prod;
#else
            Lx[p] = dense[j * n + row] * djinv;
#endif
        }

        // Rank-1 update: for each nonzero L[k,j] in column j
        //   D[k]        -= L[k,j] * dense[j][k]   (= L[k,j] * L[k,j] * D[j])
        //   dense[k][i] -= L[k,j] * dense[j][i]   for each i in L[:,j] with i >= k
        // Implementation: negate lkj via sign-bit flip (XOR 0x80000000),
        //   then use addition instead of subtraction.
        for (p = col_start; p < col_end; p++) {
            QDLDL_int k_row = Li[p];
            // Negate lkj by flipping sign bit
            QDLDL_float neg_lkj;
            {
                union { float f; uint32_t u; } _v;
                _v.f = (float)Lx[p];
                _v.u ^= 0x80000000u;
                neg_lkj = (QDLDL_float)_v.f;
            }
            QDLDL_float dense_jk = dense[j * n + k_row];  // unscaled value

            // Update diagonal D[k]: D[k] += (-lkj) * dense[j][k]
#ifdef OSQP_ROUNDING_MODE
            volatile float prod_d = (float)neg_lkj * (float)dense_jk;
            D[k_row] = D[k_row] + (QDLDL_float)prod_d;
            // Debug logging for D[479] and D[480] to file
            if (d_log_f && (k_row == 479 || k_row == 480)) {
                union { float f; uint32_t u; } _a, _b, _c, _d;
                _a.f = (float)neg_lkj;
                _b.f = (float)dense_jk;
                _c.f = (float)prod_d;
                _d.f = (float)D[k_row];
                fprintf(d_log_f, "%d,%d,%08x,%08x,%08x,%08x\n",
                        (int)j, (int)k_row, _a.u, _b.u, _c.u, _d.u);
            }
#else
            D[k_row] += neg_lkj * dense_jk;
#endif

            // Update sub-diagonal entries in column k_row
            QDLDL_int q;
            for (q = p + 1; q < col_end; q++) {
                QDLDL_int i_row = Li[q];
                // dense[k_row][i_row] += (-lkj) * dense[j][i_row]
#ifdef OSQP_ROUNDING_MODE
                volatile float prod_u = (float)neg_lkj * (float)dense[j * n + i_row];
                dense[k_row * n + i_row] = dense[k_row * n + i_row] + (QDLDL_float)prod_u;
#else
                dense[k_row * n + i_row] += neg_lkj * dense[j * n + i_row];
#endif
            }
        }
    }

#ifdef OSQP_ROUNDING_MODE
    fesetround(saved_round);
#endif

    // ================================================================
    // Step 4: Cleanup + CSR conversion for Lsolve/Ltsolve
    // ================================================================
    free(dense);

    if (d_log_f) {
        fclose(d_log_f);
        printf("[QDLDL] Saved D update log to factorization/sample_0_metadata.csv\n");
    }

    convert_csc_to_csr(n, Lp, Li, Lx);

    return positiveValuesInD;
}


// Solves (L+I)x = b
// CSR 포맷 사용 + Adder tree로 row별 연산
// (L+I)x = b => x_i = b_i - sum_j(L_ij * x_j) for j < i
void QDLDL_Lsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                  const QDLDL_float* Lx, QDLDL_float* x) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;

    // CSR이 초기화되지 않았으면 기존 CSC 방식 사용 (fallback)
    if (!g_L_csr_initialized || g_L_csr_n != n) {
        for(i = 0; i < n; i++) {
            QDLDL_float val = x[i];

            #ifdef OSQP_ROUNDING_MODE
            int old_round_Lsolve = fegetround();
            fesetround(OSQP_ROUNDING_MODE);

            for(j = Lp[i]; j < Lp[i + 1]; j++) {
                volatile float prod = Lx[j] * val;
                x[Li[j]] = x[Li[j]] - prod;
            }

            fesetround(old_round_Lsolve);
            #else

            for(j = Lp[i]; j < Lp[i + 1]; j++) {
                x[Li[j]] -= Lx[j] * val;
            }
            #endif
        }
        return;
    }

    // CSR 포맷 + Adder tree 사용
    // L은 strictly lower triangular이므로 row i의 열 인덱스는 모두 i보다 작음
    // 따라서 forward substitution: x[i] = b[i] - sum_{j<i} L[i,j] * x[j]

    #ifdef OSQP_ROUNDING_MODE
    int old_round_Lsolve = fegetround();
    fesetround(OSQP_ROUNDING_MODE);
    #endif

    for (i = 0; i < n; i++) {
        QDLDL_int row_start = g_L_csr_rowptr[i];
        QDLDL_int row_end = g_L_csr_rowptr[i + 1];
        QDLDL_int row_nnz = row_end - row_start;

        if (row_nnz == 0) {
            // 이 row에 비대각 요소 없음, x[i]는 b[i] 그대로
            continue;
        }

        // Stage 1: x[i]의 sign bit 반전 → adder tree 버퍼[0]에 넣고,
        //          L[i,j]*x[j] 곱셈 결과를 버퍼[1..row_nnz]에 넣음
        //          입력: { -x[i], L[i,j0]*x[j0], L[i,j1]*x[j1], ... }
        {
            union { float f; uint32_t u; } neg_xi;
            neg_xi.f = (float)x[i];
            neg_xi.u ^= 0x80000000u;  // sign bit 반전
            g_adder_tree_buffer[0] = (QDLDL_float)neg_xi.f;
        }

        for (j = 0; j < row_nnz; j++) {
            QDLDL_int col = g_L_csr_colind[row_start + j];
            QDLDL_float val = g_L_csr_values[row_start + j];
            QDLDL_float prod = val * x[col];
            #ifdef OSQP_ROUNDING_MODE
            volatile QDLDL_float truncated = prod;
            g_adder_tree_buffer[j + 1] = truncated;
            #else
            g_adder_tree_buffer[j + 1] = prod;
            #endif
        }

        // Stage 2: Adder tree로 합산 (row_nnz + 1개)
        //          = -x[i] + sum(L[i,j]*x[j]) = -(x[i] - sum(L[i,j]*x[j]))
        QDLDL_float neg_result = adder_tree_sum_qdldl(g_adder_tree_buffer, row_nnz + 1);

        // Stage 3: 결과의 sign bit 반전 → 최종 결과
        {
            union { float f; uint32_t u; } final_val;
            final_val.f = (float)neg_result;
            final_val.u ^= 0x80000000u;  // sign bit 반전
            #ifdef OSQP_ROUNDING_MODE
            volatile QDLDL_float truncated_result = (QDLDL_float)final_val.f;
            x[i] = truncated_result;
            #else
            x[i] = (QDLDL_float)final_val.f;
            #endif
        }
    }

    #ifdef OSQP_ROUNDING_MODE
    fesetround(old_round_Lsolve);
    #endif
}

// Solves (L+I)'x = b
// L의 CSR = L^T의 CSC 이므로, L^T의 column 순서로 backward substitution
// (L^T + I)x = b => x_i = b_i - sum_{j>i} L^T[i,j] * x[j]
//                       = b_i - sum_{j>i} L[j,i] * x[j]
// L의 CSR에서 row i는 L의 i번째 row = L^T의 i번째 column
void QDLDL_Ltsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                   const QDLDL_float* Lx, QDLDL_float* x) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;

    // CSR이 초기화되지 않았으면 기존 CSC 방식 사용 (fallback)
    if (!g_L_csr_initialized || g_L_csr_n != n) {
        for(i = n - 1; i >= 0; i--) {
            QDLDL_float val = x[i];

            #ifdef OSQP_ROUNDING_MODE
            int old_round_Ltsolve = fegetround();
            fesetround(OSQP_ROUNDING_MODE);

            for(j = Lp[i]; j < Lp[i + 1]; j++) {
                volatile float prod = Lx[j] * x[Li[j]];
                val = val - prod;
            }
            x[i] = val;

            fesetround(old_round_Ltsolve);
            #else

            for(j = Lp[i]; j < Lp[i + 1]; j++) {
                val -= Lx[j] * x[Li[j]];
            }
            x[i] = val;
            #endif
        }
        return;
    }

    // CSR 포맷 사용 (L의 CSR = L^T의 CSC)
    // L^T를 column 순서로 처리 = L의 row 순서로 처리
    // Backward substitution: i = n-1 부터 0까지
    // L^T[col i]의 비대각 요소들은 L[row j, col i] for j > i
    // 이는 L의 CSR에서 row j의 요소 중 colind == i인 것들

    // CSR에서 L^T의 backward substitution:
    // x[i]가 결정된 후, x[i]를 사용하여 x[col]들을 갱신
    // L의 CSR row i: L[i, col] for col < i
    // => L^T[col, i] for col < i
    // => x[col]의 계산에 L^T[col, i] * x[i]가 기여

    #ifdef OSQP_ROUNDING_MODE
    int old_round_Ltsolve = fegetround();
    fesetround(OSQP_ROUNDING_MODE);
    #endif

    // Backward: i = n-1 down to 0
    // x[i]가 결정되면, L의 CSR row i를 순회하며 x[col]들에서 빼줌
    for (i = n - 1; i >= 0; i--) {
        // x[i]는 현재 상태 그대로 (이전 iteration에서 이미 갱신됨)
        QDLDL_float val = x[i];

        // L의 CSR row i를 순회: L[i, col] for col < i
        // 이 값들은 L^T[col, i]이므로, x[col] -= L^T[col, i] * x[i]
        QDLDL_int row_start = g_L_csr_rowptr[i];
        QDLDL_int row_end = g_L_csr_rowptr[i + 1];

        for (j = row_start; j < row_end; j++) {
            QDLDL_int col = g_L_csr_colind[j];  // col < i
            QDLDL_float L_val = g_L_csr_values[j];  // L[i, col] = L^T[col, i]

            #ifdef OSQP_ROUNDING_MODE
            volatile QDLDL_float prod = L_val * val;
            x[col] = x[col] - prod;
            #else
            x[col] -= L_val * val;
            #endif
        }
    }

    #ifdef OSQP_ROUNDING_MODE
    fesetround(old_round_Ltsolve);
    #endif
}

// Solves Ax = b where A has given LDL factors
void QDLDL_solve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li, const QDLDL_float* Lx,
                 const QDLDL_float* Dinv, QDLDL_float* x) {
    QDLDL_int i = 0;

    // printf("In QDLDL_solve\n");

    QDLDL_Lsolve(n, Lp, Li, Lx, x);

    // printf("After Lsolve\n");

    #ifdef OSQP_ROUNDING_MODE
    int old_round_solve = fegetround();
    fesetround(OSQP_ROUNDING_MODE);

    for(i = 0; i < n; i++) {
        x[i] = x[i] * Dinv[i];
    }

    fesetround(old_round_solve);
    #else

    for(i = 0; i < n; i++) {
        x[i] *= Dinv[i];
    }
    #endif

    QDLDL_Ltsolve(n, Lp, Li, Lx, x);

    // printf("After Ltsolve\n");

    // dump_int_hex  ("Lp_skip.hex",    Lp,    n + 1);
    // dump_int_hex  ("Li_skip.hex",    Li,    Lp[n]);
    // dump_float_hex("Lx_skip.hex",    Lx,    Lp[n]);
    // append_dec_qint("../result/PKPT_test/Lp_n_P_ref.dec", Lp ? Lp[n] : 0);
}

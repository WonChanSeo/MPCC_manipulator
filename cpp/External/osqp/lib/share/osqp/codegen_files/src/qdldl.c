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

#ifdef QDLDL_USE_FLEXFLOAT
#include <flexfloat.h>
#endif

// FlexFloat precision configuration (should match OSQP settings)
#ifndef FF_exponent_bits
#define FF_exponent_bits 8
#endif
#ifndef FF_mantissa_bits
#define FF_mantissa_bits 23
#endif
#define QDLDL_FF_DESC ((flexfloat_desc_t){FF_exponent_bits, FF_mantissa_bits})

#define QDLDL_UNKNOWN (-1)
#define QDLDL_USED (1)
#define QDLDL_UNUSED (0)

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
#ifdef QDLDL_USE_FLEXFLOAT
    // FlexFloat enabled - use configured precision
    static char precision_str[32];
    snprintf(precision_str, sizeof(precision_str), "E%dM%d", FF_exponent_bits, FF_mantissa_bits);
    return precision_str;
#else
    // Standard precision based on QDLDL_float type
    if (sizeof(QDLDL_float) == 8) {
        return "double";  // E11M52 equivalent
    } else if (sizeof(QDLDL_float) == 4) {
        return "float";   // E8M23 equivalent
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
#ifdef QDLDL_USE_FLEXFLOAT
    fprintf(f, "flexfloat=ON\n");
    fprintf(f, "exponent_bits=%d\n", FF_exponent_bits);
    fprintf(f, "mantissa_bits=%d\n", FF_mantissa_bits);
#else
    fprintf(f, "flexfloat=OFF\n");
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

#ifdef QDLDL_USE_FLEXFLOAT
// FlexFloat version of reciprocal computation
float reciprocal_nr_fma(float d) {
    // Get initial approximation using magic number (standard float)
    float x_init = init_reciprocal(d);

    // Convert to FlexFloat
    flexfloat_t ff_d, ff_x, ff_t, ff_two, ff_zero, ff_neg_d;

    ff_init_float(&ff_d, d, QDLDL_FF_DESC);
    ff_init_float(&ff_x, x_init, QDLDL_FF_DESC);  // Use magic number approximation
    ff_init_float(&ff_t, 0.0f, QDLDL_FF_DESC);
    ff_init_float(&ff_two, 2.0f, QDLDL_FF_DESC);
    ff_init_float(&ff_zero, 0.0f, QDLDL_FF_DESC);

    // Negate d in FlexFloat
    float neg_d_val = -d;
    ff_init_float(&ff_neg_d, neg_d_val, QDLDL_FF_DESC);

    // Newton-Raphson iterations with FlexFloat arithmetic
    // Iteration 1: t = fma(-d, x, 2.0); x = fma(x, t, 0.0);
    ff_fma(&ff_t, &ff_neg_d, &ff_x, &ff_two);
    ff_fma(&ff_x, &ff_x, &ff_t, &ff_zero);

    // Iteration 2
    ff_fma(&ff_t, &ff_neg_d, &ff_x, &ff_two);
    ff_fma(&ff_x, &ff_x, &ff_t, &ff_zero);

    // Iteration 3 (may need more iterations for lower precision)
    ff_fma(&ff_t, &ff_neg_d, &ff_x, &ff_two);
    ff_fma(&ff_x, &ff_x, &ff_t, &ff_zero);

    // Additional iteration for lower precision (14-bit mantissa)
    ff_fma(&ff_t, &ff_neg_d, &ff_x, &ff_two);
    ff_fma(&ff_x, &ff_x, &ff_t, &ff_zero);

    return (float)ff_get_float(&ff_x);
}
#else
// Standard float version
float reciprocal_nr_fma(float d) {
    float x = init_reciprocal(d);

    float t = fmaf(-d, x, 2.0f);
    x = fmaf(x, t, 0.0f);

    t = fmaf(-d, x, 2.0f);
    x = fmaf(x, t, 0.0f);

    t = fmaf(-d, x, 2.0f);
    x = fmaf(x, t, 0.0f);

    return x;
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

            #ifdef QDLDL_USE_FLEXFLOAT
            for(j = Lp[cidx]; j < tmpIdx; j++) {

                // yVals[Li[j]] -= Lx[j] * yVals_cidx;
                flexfloat_t ff_yVal, ff_Lx, ff_yVals_cidx, ff_product, ff_result, ff_zero, ff_neg_one;
                ff_init_float(&ff_yVal, yVals[Li[j]], QDLDL_FF_DESC);
                ff_init_float(&ff_Lx, Lx[j], QDLDL_FF_DESC);
                ff_init_float(&ff_yVals_cidx, yVals_cidx, QDLDL_FF_DESC);
                ff_init_float(&ff_zero, 0.0f, QDLDL_FF_DESC);
                ff_init_float(&ff_neg_one, -1.0f, QDLDL_FF_DESC);
                ff_init_float(&ff_product, 0.0f, QDLDL_FF_DESC);
                ff_init_float(&ff_result, 0.0f, QDLDL_FF_DESC);
                ff_fma(&ff_product, &ff_Lx, &ff_yVals_cidx, &ff_zero);
                ff_fma(&ff_result, &ff_product, &ff_neg_one, &ff_yVal);
                yVals[Li[j]] = ff_get_float(&ff_result);
            }

            // Now I have the cidx^th element of y = L\b.
            // so compute the corresponding element of
            // this row of L and put it into the right place
            Li[tmpIdx] = k;
            // Lx[tmpIdx] = yVals_cidx * Dinv[cidx];
            flexfloat_t ff_yVals_cidx, ff_Dinv, ff_Lx, ff_zero2;
            ff_init_float(&ff_yVals_cidx, yVals_cidx, QDLDL_FF_DESC);
            ff_init_float(&ff_Dinv, Dinv[cidx], QDLDL_FF_DESC);
            ff_init_float(&ff_zero2, 0.0f, QDLDL_FF_DESC);
            ff_init_float(&ff_Lx, 0.0f, QDLDL_FF_DESC);
            ff_fma(&ff_Lx, &ff_yVals_cidx, &ff_Dinv, &ff_zero2);
            Lx[tmpIdx] = ff_get_float(&ff_Lx);

            // D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
            // D[k] -= yVals_cidx * Lx[tmpIdx];
            flexfloat_t ff_Dk, ff_yVals_cidx_sq, ff_temp, ff_Dk_new, ff_zero3, ff_zero4, ff_neg_one2;
            ff_init_float(&ff_Dk, D[k], QDLDL_FF_DESC);
            ff_init_float(&ff_yVals_cidx_sq, yVals_cidx, QDLDL_FF_DESC);
            ff_init_float(&ff_zero3, 0.0f, QDLDL_FF_DESC);
            ff_init_float(&ff_zero4, 0.0f, QDLDL_FF_DESC);
            ff_init_float(&ff_neg_one2, -1.0f, QDLDL_FF_DESC);
            ff_init_float(&ff_temp, 0.0f, QDLDL_FF_DESC);
            ff_init_float(&ff_Dk_new, 0.0f, QDLDL_FF_DESC);
            ff_fma(&ff_temp, &ff_yVals_cidx_sq, &ff_yVals_cidx_sq, &ff_zero3); // square
            ff_fma(&ff_temp, &ff_temp, &ff_Dinv, &ff_zero4); // * Dinv[cidx]
            ff_fma(&ff_Dk_new, &ff_temp, &ff_neg_one2, &ff_Dk);
            D[k] = ff_get_float(&ff_Dk_new);
            LNextSpaceInCol[cidx]++;
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
    

    return positiveValuesInD;
}

// Solves (L+I)x = b
void QDLDL_Lsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                  const QDLDL_float* Lx, QDLDL_float* x) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;

    for(i = 0; i < n; i++) {
        QDLDL_float val = x[i];

        #ifdef QDLDL_USE_FLEXFLOAT

        for(j = Lp[i]; j < Lp[i + 1]; j++)
        {
            flexfloat_t ff_Lx, ff_val, ff_result;

            ff_init_float(&ff_Lx, -Lx[j], QDLDL_FF_DESC);
            ff_init_float(&ff_val, val, QDLDL_FF_DESC);
            ff_init_float(&ff_result, x[Li[j]], QDLDL_FF_DESC);
            ff_fma(&ff_result, &ff_Lx, &ff_val, &ff_result);
            x[Li[j]] = ff_get_float(&ff_result);
        }
        #else

        for(j = Lp[i]; j < Lp[i + 1]; j++) {
            x[Li[j]] -= Lx[j] * val;
        }
        #endif
    }
}

// Solves (L+I)'x = b
void QDLDL_Ltsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                   const QDLDL_float* Lx, QDLDL_float* x) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;

    for(i = n - 1; i >= 0; i--) {
        QDLDL_float val = x[i];

        #ifdef QDLDL_USE_FLEXFLOAT

        for(j = Lp[i]; j < Lp[i + 1]; j++)
        {
            // val -= Lx[j] * x[Li[j]];
            flexfloat_t ff_Lx, ff_xLi, ff_result, ff_zero;
            
            ff_init_float(&ff_Lx, -Lx[j], QDLDL_FF_DESC);
            ff_init_float(&ff_xLi, x[Li[j]], QDLDL_FF_DESC);
            ff_init_float(&ff_zero, 0.0f, QDLDL_FF_DESC);
            ff_init_float(&ff_result, val, QDLDL_FF_DESC);
            ff_fma(&ff_result, &ff_Lx, &ff_xLi, &ff_result);
            val = ff_get_float(&ff_result);
        }
        x[i] = val;
        #else

        for(j = Lp[i]; j < Lp[i + 1]; j++) {
            val -= Lx[j] * x[Li[j]];
        }
        x[i] = val;
        #endif
    }
}

// Solves Ax = b where A has given LDL factors
void QDLDL_solve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li, const QDLDL_float* Lx,
                 const QDLDL_float* Dinv, QDLDL_float* x) {
    QDLDL_int i = 0;

    // printf("In QDLDL_solve\n");

    QDLDL_Lsolve(n, Lp, Li, Lx, x);

    // printf("After Lsolve\n");

    #ifdef QDLDL_USE_FLEXFLOAT

    for(i = 0; i < n; i++) {
        flexfloat_t xi, ff_Dinv, ff_result, ff_zero;
        ff_init_float(&xi, x[i], QDLDL_FF_DESC);
        ff_init_float(&ff_Dinv, Dinv[i], QDLDL_FF_DESC);
        ff_init_float(&ff_zero, 0.0f, QDLDL_FF_DESC);
        ff_init_float(&ff_result, 0.0f, QDLDL_FF_DESC);
        ff_fma(&ff_result, &xi, &ff_Dinv, &ff_zero);
        x[i] = ff_get_float(&ff_result);
    }
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

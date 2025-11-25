/*
 * QDLDL Precision Test Program
 *
 * Reads saved QDLDL samples and re-runs them with different precisions
 * to compare accuracy.
 *
 * Usage:
 *   ./qdldl_precision_test <sample_directory> <output_directory>
 *
 * Example:
 *   ./qdldl_precision_test result/qdldl_samples_double/sample_000000 result/qdldl_test_E8M20
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <errno.h>
#include "qdldl.h"

#ifdef QDLDL_USE_FLEXFLOAT
#include <flexfloat.h>
#ifndef FF_exponent_bits
#define FF_exponent_bits 8
#endif
#ifndef FF_mantissa_bits
#define FF_mantissa_bits 23
#endif
#define QDLDL_FF_DESC ((flexfloat_desc_t){FF_exponent_bits, FF_mantissa_bits})
#endif

// Read metadata from sample directory
typedef struct {
    QDLDL_int n;
    QDLDL_int nnz_A;
    QDLDL_int nnz_L;
    size_t sizeof_int;
    size_t sizeof_float;
    char precision[32];
} SampleMetadata;

// Helper to read int array from hex file
static int read_int_hex(const char* filepath, QDLDL_int* arr, QDLDL_int expected_len) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        perror(filepath);
        return -1;
    }

    for (QDLDL_int i = 0; i < expected_len; i++) {
        if (sizeof(QDLDL_int) == 8) {
            uint64_t val;
            if (fscanf(f, "%" SCNx64, &val) != 1) {
                fprintf(stderr, "Failed to read int at index %jd from %s\n", (intmax_t)i, filepath);
                fclose(f);
                return -1;
            }
            arr[i] = (QDLDL_int)val;
        } else {
            uint32_t val;
            if (fscanf(f, "%" SCNx32, &val) != 1) {
                fprintf(stderr, "Failed to read int at index %jd from %s\n", (intmax_t)i, filepath);
                fclose(f);
                return -1;
            }
            arr[i] = (QDLDL_int)val;
        }
    }

    fclose(f);
    return 0;
}

// Helper to read float array from hex file
static int read_float_hex(const char* filepath, QDLDL_float* arr, QDLDL_int expected_len) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        perror(filepath);
        return -1;
    }

    for (QDLDL_int i = 0; i < expected_len; i++) {
        if (sizeof(QDLDL_float) == 8) {
            uint64_t val;
            if (fscanf(f, "%" SCNx64, &val) != 1) {
                fprintf(stderr, "Failed to read float at index %jd from %s\n", (intmax_t)i, filepath);
                fclose(f);
                return -1;
            }
            memcpy(&arr[i], &val, 8);
        } else {
            uint32_t val;
            if (fscanf(f, "%" SCNx32, &val) != 1) {
                fprintf(stderr, "Failed to read float at index %jd from %s\n", (intmax_t)i, filepath);
                fclose(f);
                return -1;
            }
            memcpy(&arr[i], &val, 4);
        }
    }

    fclose(f);
    return 0;
}

// Helper to write float array in hex format
static void write_float_hex(const char* filepath, const QDLDL_float* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if (!f) {
        perror(filepath);
        return;
    }

    if (sizeof(QDLDL_float) == 8) {
        for (QDLDL_int i = 0; i < len; i++) {
            uint64_t u;
            memcpy(&u, &arr[i], 8);
            fprintf(f, "%016" PRIx64 "\n", u);
        }
    } else {
        for (QDLDL_int i = 0; i < len; i++) {
            uint32_t u;
            memcpy(&u, &arr[i], 4);
            fprintf(f, "%08" PRIx32 "\n", u);
        }
    }

    fclose(f);
}

// Read metadata from sample directory
static int read_metadata(const char* sample_dir, SampleMetadata* meta) {
    char filepath[1024];
    snprintf(filepath, sizeof(filepath), "%s/metadata.txt", sample_dir);

    FILE* f = fopen(filepath, "r");
    if (!f) {
        perror(filepath);
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "n=%jd", (intmax_t*)&meta->n) == 1) continue;
        if (sscanf(line, "nnz_A=%jd", (intmax_t*)&meta->nnz_A) == 1) continue;
        if (sscanf(line, "nnz_L=%jd", (intmax_t*)&meta->nnz_L) == 1) continue;
        if (sscanf(line, "sizeof_int=%zu", &meta->sizeof_int) == 1) continue;
        if (sscanf(line, "sizeof_float=%zu", &meta->sizeof_float) == 1) continue;
        if (sscanf(line, "precision=%31s", meta->precision) == 1) continue;
    }

    fclose(f);
    return 0;
}

// Get precision string based on current build configuration
static const char* get_current_precision_string(void) {
#ifdef QDLDL_USE_FLEXFLOAT
    static char precision_str[32];
    #ifndef FF_exponent_bits
        #define FF_exponent_bits 8
    #endif
    #ifndef FF_mantissa_bits
        #define FF_mantissa_bits 23
    #endif
    snprintf(precision_str, sizeof(precision_str), "E%dM%d", FF_exponent_bits, FF_mantissa_bits);
    return precision_str;
#else
    if (sizeof(QDLDL_float) == 8) {
        return "double";
    } else if (sizeof(QDLDL_float) == 4) {
        return "float";
    } else {
        return "unknown";
    }
#endif
}

// Check if FlexFloat is enabled
static int is_flexfloat_enabled(void) {
#ifdef QDLDL_USE_FLEXFLOAT
    return 1;
#else
    return 0;
#endif
}

// Get exponent bits
static int get_exponent_bits(void) {
#ifdef QDLDL_USE_FLEXFLOAT
    #ifndef FF_exponent_bits
        #define FF_exponent_bits 8
    #endif
    return FF_exponent_bits;
#else
    return 0;
#endif
}

// Get mantissa bits
static int get_mantissa_bits(void) {
#ifdef QDLDL_USE_FLEXFLOAT
    #ifndef FF_mantissa_bits
        #define FF_mantissa_bits 23
    #endif
    return FF_mantissa_bits;
#else
    return 0;
#endif
}

// Quantize float array to current precision (if FlexFloat enabled)
static void quantize_float_array(QDLDL_float* arr, QDLDL_int len) {
#ifdef QDLDL_USE_FLEXFLOAT
    flexfloat_t ff_val;
    for (QDLDL_int i = 0; i < len; i++) {
        // Convert to FlexFloat and back to quantize
        ff_init_float(&ff_val, (float)arr[i], QDLDL_FF_DESC);
        arr[i] = (QDLDL_float)ff_get_float(&ff_val);
    }
#else
    // No quantization needed if not using FlexFloat
    (void)arr;
    (void)len;
#endif
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <sample_directory> <output_directory>\n", argv[0]);
        fprintf(stderr, "Example: %s result/qdldl_samples_double/sample_000000 result/qdldl_test_E8M20\n", argv[0]);
        return 1;
    }

    const char* sample_dir = argv[1];
    const char* output_dir = argv[2];

    // Create output directory
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0755);
    }

    printf("=================================================\n");
    printf("QDLDL Precision Test\n");
    printf("=================================================\n");
    printf("Input sample:  %s\n", sample_dir);
    printf("Output dir:    %s\n", output_dir);
    printf("Current precision: %s\n", get_current_precision_string());
    printf("QDLDL_float size: %zu bytes\n", sizeof(QDLDL_float));
    printf("QDLDL_int size: %zu bytes\n", sizeof(QDLDL_int));
    printf("=================================================\n\n");

    // Read metadata
    SampleMetadata meta;
    if (read_metadata(sample_dir, &meta) != 0) {
        fprintf(stderr, "Failed to read metadata\n");
        return 1;
    }

    printf("Sample metadata:\n");
    printf("  n = %jd\n", (intmax_t)meta.n);
    printf("  nnz_A = %jd\n", (intmax_t)meta.nnz_A);
    printf("  nnz_L = %jd\n", (intmax_t)meta.nnz_L);
    printf("  Original precision: %s\n\n", meta.precision);

    // Allocate memory for input matrix A (CSC format)
    QDLDL_int* Ap = (QDLDL_int*)malloc(sizeof(QDLDL_int) * (meta.n + 1));
    QDLDL_int* Ai = (QDLDL_int*)malloc(sizeof(QDLDL_int) * meta.nnz_A);
    QDLDL_float* Ax = (QDLDL_float*)malloc(sizeof(QDLDL_float) * meta.nnz_A);

    // Allocate memory for L factor
    QDLDL_int* Lp = (QDLDL_int*)malloc(sizeof(QDLDL_int) * (meta.n + 1));
    QDLDL_int* Li = (QDLDL_int*)malloc(sizeof(QDLDL_int) * meta.nnz_L);
    QDLDL_float* Lx = (QDLDL_float*)malloc(sizeof(QDLDL_float) * meta.nnz_L);

    // Allocate memory for D, Dinv, etree, Lnz
    QDLDL_float* D = (QDLDL_float*)malloc(sizeof(QDLDL_float) * meta.n);
    QDLDL_float* Dinv = (QDLDL_float*)malloc(sizeof(QDLDL_float) * meta.n);
    QDLDL_int* etree = (QDLDL_int*)malloc(sizeof(QDLDL_int) * meta.n);
    QDLDL_int* Lnz = (QDLDL_int*)malloc(sizeof(QDLDL_int) * meta.n);

    // Allocate working memory
    QDLDL_int* iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int) * (3 * meta.n));
    QDLDL_bool* bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool) * meta.n);
    QDLDL_float* fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float) * meta.n);

    // Allocate memory for x (input and output)
    QDLDL_float* x_input = (QDLDL_float*)malloc(sizeof(QDLDL_float) * meta.n);
    QDLDL_float* x_output = (QDLDL_float*)malloc(sizeof(QDLDL_float) * meta.n);

    if (!Ap || !Ai || !Ax || !Lp || !Li || !Lx || !D || !Dinv || !etree || !Lnz ||
        !iwork || !bwork || !fwork || !x_input || !x_output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Read input data from sample directory
    char filepath[1024];

    printf("Reading input data...\n");
    snprintf(filepath, sizeof(filepath), "%s/Ap.hex", sample_dir);
    if (read_int_hex(filepath, Ap, meta.n + 1) != 0) return 1;

    snprintf(filepath, sizeof(filepath), "%s/Ai.hex", sample_dir);
    if (read_int_hex(filepath, Ai, meta.nnz_A) != 0) return 1;

    snprintf(filepath, sizeof(filepath), "%s/Ax.hex", sample_dir);
    if (read_float_hex(filepath, Ax, meta.nnz_A) != 0) return 1;

    snprintf(filepath, sizeof(filepath), "%s/x_input.hex", sample_dir);
    if (read_float_hex(filepath, x_input, meta.n) != 0) return 1;

    printf("Input data loaded successfully.\n\n");

    // Quantize input data to current precision (if FlexFloat enabled)
#ifdef QDLDL_USE_FLEXFLOAT
    printf("Quantizing input data to E%dM%d precision...\n", FF_exponent_bits, FF_mantissa_bits);
    quantize_float_array(Ax, meta.nnz_A);
    quantize_float_array(x_input, meta.n);
    printf("  Input data quantized.\n\n");
#endif

    // Run QDLDL factorization with current precision
    printf("Running QDLDL_etree...\n");
    QDLDL_int sum_Lnz = QDLDL_etree(meta.n, Ap, Ai, iwork, Lnz, etree);
    if (sum_Lnz < 0) {
        fprintf(stderr, "QDLDL_etree failed with error code %jd\n", (intmax_t)sum_Lnz);
        return 1;
    }
    printf("  sum_Lnz = %jd\n", (intmax_t)sum_Lnz);

    // Reallocate Li and Lx based on actual nnz
    free(Li);
    free(Lx);
    Li = (QDLDL_int*)malloc(sizeof(QDLDL_int) * sum_Lnz);
    Lx = (QDLDL_float*)malloc(sizeof(QDLDL_float) * sum_Lnz);
    if (!Li || !Lx) {
        fprintf(stderr, "Memory allocation failed for Li/Lx\n");
        return 1;
    }

    // Set Lp (column pointers for L)
    Lp[0] = 0;
    for (QDLDL_int i = 0; i < meta.n; i++) {
        Lp[i + 1] = Lp[i] + Lnz[i];
    }

    printf("Running QDLDL_factor...\n");
    QDLDL_int pos_D_count = QDLDL_factor(meta.n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);
    if (pos_D_count < 0) {
        fprintf(stderr, "QDLDL_factor failed with error code %jd\n", (intmax_t)pos_D_count);
        return 1;
    }
    printf("  Positive values in D: %jd / %jd\n", (intmax_t)pos_D_count, (intmax_t)meta.n);

    // Copy x_input to x_output for solve
    memcpy(x_output, x_input, sizeof(QDLDL_float) * meta.n);

    printf("Running QDLDL_solve...\n");
    QDLDL_solve(meta.n, Lp, Li, Lx, Dinv, x_output);
    printf("  Solve completed.\n\n");

    // Save results to output directory
    printf("Saving results to %s...\n", output_dir);

    // Save metadata
    snprintf(filepath, sizeof(filepath), "%s/metadata.txt", output_dir);
    FILE* f = fopen(filepath, "w");
    if (f) {
        fprintf(f, "n=%jd\n", (intmax_t)meta.n);
        fprintf(f, "nnz_A=%jd\n", (intmax_t)meta.nnz_A);
        fprintf(f, "nnz_L=%jd\n", (intmax_t)sum_Lnz);
        fprintf(f, "sizeof_int=%zu\n", sizeof(QDLDL_int));
        fprintf(f, "sizeof_float=%zu\n", sizeof(QDLDL_float));
        fprintf(f, "precision=%s\n", get_current_precision_string());
        fprintf(f, "input_precision=%s\n", meta.precision);

        if (is_flexfloat_enabled()) {
            fprintf(f, "flexfloat=ON\n");
            fprintf(f, "exponent_bits=%d\n", get_exponent_bits());
            fprintf(f, "mantissa_bits=%d\n", get_mantissa_bits());
        } else {
            fprintf(f, "flexfloat=OFF\n");
        }
        fclose(f);
    }

    // Save L factor structure (Lp, Li)
    snprintf(filepath, sizeof(filepath), "%s/Lp.hex", output_dir);
    {
        FILE* f = fopen(filepath, "w");
        if (f) {
            for (QDLDL_int i = 0; i <= meta.n; i++) {
                if (sizeof(QDLDL_int) == 8) {
                    fprintf(f, "%016" PRIx64 "\n", (uint64_t)Lp[i]);
                } else {
                    fprintf(f, "%08" PRIx32 "\n", (uint32_t)Lp[i]);
                }
            }
            fclose(f);
        }
    }

    snprintf(filepath, sizeof(filepath), "%s/Li.hex", output_dir);
    {
        FILE* f = fopen(filepath, "w");
        if (f) {
            for (QDLDL_int i = 0; i < sum_Lnz; i++) {
                if (sizeof(QDLDL_int) == 8) {
                    fprintf(f, "%016" PRIx64 "\n", (uint64_t)Li[i]);
                } else {
                    fprintf(f, "%08" PRIx32 "\n", (uint32_t)Li[i]);
                }
            }
            fclose(f);
        }
    }

    // Save L factor values
    snprintf(filepath, sizeof(filepath), "%s/Lx.hex", output_dir);
    write_float_hex(filepath, Lx, sum_Lnz);

    // Save D and Dinv
    snprintf(filepath, sizeof(filepath), "%s/D.hex", output_dir);
    write_float_hex(filepath, D, meta.n);

    snprintf(filepath, sizeof(filepath), "%s/Dinv.hex", output_dir);
    write_float_hex(filepath, Dinv, meta.n);

    // Save etree and Lnz
    snprintf(filepath, sizeof(filepath), "%s/etree.hex", output_dir);
    {
        FILE* f = fopen(filepath, "w");
        if (f) {
            for (QDLDL_int i = 0; i < meta.n; i++) {
                if (sizeof(QDLDL_int) == 8) {
                    fprintf(f, "%016" PRIx64 "\n", (uint64_t)etree[i]);
                } else {
                    fprintf(f, "%08" PRIx32 "\n", (uint32_t)etree[i]);
                }
            }
            fclose(f);
        }
    }

    snprintf(filepath, sizeof(filepath), "%s/Lnz.hex", output_dir);
    {
        FILE* f = fopen(filepath, "w");
        if (f) {
            for (QDLDL_int i = 0; i < meta.n; i++) {
                if (sizeof(QDLDL_int) == 8) {
                    fprintf(f, "%016" PRIx64 "\n", (uint64_t)Lnz[i]);
                } else {
                    fprintf(f, "%08" PRIx32 "\n", (uint32_t)Lnz[i]);
                }
            }
            fclose(f);
        }
    }

    // Save x_output (solution)
    snprintf(filepath, sizeof(filepath), "%s/x_output.hex", output_dir);
    write_float_hex(filepath, x_output, meta.n);

    printf("Results saved successfully.\n\n");

    printf("=================================================\n");
    printf("Test completed successfully!\n");
    printf("=================================================\n");

    // Cleanup
    free(Ap); free(Ai); free(Ax);
    free(Lp); free(Li); free(Lx);
    free(D); free(Dinv);
    free(etree); free(Lnz);
    free(iwork); free(bwork); free(fwork);
    free(x_input); free(x_output);

    return 0;
}

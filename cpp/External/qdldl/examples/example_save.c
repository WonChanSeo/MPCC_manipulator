/**
 * QDLDL Example with Matrix Saving
 *
 * This example demonstrates QDLDL factorization and saves all matrices
 * (A, L, D, etree) to hex files for precision analysis.
 *
 * Usage:
 *   ./qdldl_example_save [output_dir]
 *
 * If output_dir is not specified, defaults to "qdldl_output"
 */

#include "qdldl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#ifdef QDLDL_USE_FLEXFLOAT
#include "flexfloat.h"
#endif

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

void print_arrayi(const QDLDL_int* data, QDLDL_int n, char* varName);
void print_arrayf(const QDLDL_float* data, QDLDL_int n, char* varName);
void print_line(void);

// ========================================
// Matrix saving functions (hex format)
// ========================================
static void save_int_hex(const char* filepath, const QDLDL_int* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if(!f) {
        perror(filepath);
        return;
    }
    if(sizeof(QDLDL_int) == 8) {
        for(QDLDL_int i = 0; i < len; ++i)
            fprintf(f, "%016" PRIx64 "\n", (uint64_t) arr[i]);
    } else {
        for(QDLDL_int i = 0; i < len; ++i)
            fprintf(f, "%08" PRIx32 "\n", (uint32_t) arr[i]);
    }
    fclose(f);
}

static void save_float_hex(const char* filepath, const QDLDL_float* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if(!f) {
        perror(filepath);
        return;
    }

#ifdef QDLDL_USE_FLEXFLOAT
    // FlexFloat: save based on mantissa bits (E8M23 -> FP32, else FP64)
#if FF_mantissa_bits <= 23
    // Save as FP32 (convert double to float via bit truncation, no rounding)
    for(QDLDL_int i = 0; i < len; ++i) {
        uint64_t d_bits;
        memcpy(&d_bits, &arr[i], 8);

        uint32_t sign = (d_bits >> 63) & 0x1;
        int64_t d_exp = (d_bits >> 52) & 0x7FF;
        uint64_t d_mant = d_bits & 0x000FFFFFFFFFFFFFULL;

        uint32_t f_bits;

        if(d_exp == 0x7FF) {
            f_bits = (sign << 31) | 0x7F800000 | (uint32_t)(d_mant >> 29);
        } else if(d_exp == 0) {
            f_bits = sign << 31;
        } else {
            int64_t f_exp = d_exp - 1023 + 127;
            if(f_exp >= 255) {
                f_bits = (sign << 31) | 0x7F800000;
            } else if(f_exp <= 0) {
                f_bits = sign << 31;
            } else {
                uint32_t f_mant = (uint32_t)(d_mant >> 29);
                f_bits = (sign << 31) | ((uint32_t)f_exp << 23) | f_mant;
            }
        }
        fprintf(f, "%08" PRIx32 "\n", f_bits);
    }
#else
    for(QDLDL_int i = 0; i < len; ++i) {
        uint64_t u;
        memcpy(&u, &arr[i], 8);
        fprintf(f, "%016" PRIx64 "\n", u);
    }
#endif
#else
    // No FlexFloat: save based on actual sizeof
    if(sizeof(QDLDL_float) == 8) {
        for(QDLDL_int i = 0; i < len; ++i) {
            uint64_t u;
            memcpy(&u, &arr[i], 8);
            fprintf(f, "%016" PRIx64 "\n", u);
        }
    } else {
        for(QDLDL_int i = 0; i < len; ++i) {
            uint32_t u;
            memcpy(&u, &arr[i], 4);
            fprintf(f, "%08" PRIx32 "\n", u);
        }
    }
#endif

    fclose(f);
}

static void save_metadata(const char* filepath, QDLDL_int n, QDLDL_int nnz_A, QDLDL_int nnz_L) {
    FILE* f = fopen(filepath, "w");
    if(!f) {
        perror(filepath);
        return;
    }
    fprintf(f, "n=%jd\n", (intmax_t) n);
    fprintf(f, "nnz_A=%jd\n", (intmax_t) nnz_A);
    fprintf(f, "nnz_L=%jd\n", (intmax_t) nnz_L);
    fprintf(f, "sizeof_int=%zu\n", sizeof(QDLDL_int));
    fprintf(f, "sizeof_float_internal=%zu\n", sizeof(QDLDL_float));
#ifdef QDLDL_USE_FLEXFLOAT
    fprintf(f, "flexfloat=ON\n");
#ifdef FF_exponent_bits
    fprintf(f, "exponent_bits=%d\n", FF_exponent_bits);
#endif
#ifdef FF_mantissa_bits
    fprintf(f, "mantissa_bits=%d\n", FF_mantissa_bits);
// Saved size based on mantissa bits
#if FF_mantissa_bits <= 23
    fprintf(f, "sizeof_float_saved=4\n");
    fprintf(f, "hex_digits=8\n");
#else
    fprintf(f, "sizeof_float_saved=8\n");
    fprintf(f, "hex_digits=16\n");
#endif
#endif
#else
    fprintf(f, "flexfloat=OFF\n");
    if(sizeof(QDLDL_float) == 8) {
        fprintf(f, "precision=double\n");
        fprintf(f, "sizeof_float_saved=8\n");
        fprintf(f, "hex_digits=16\n");
    } else {
        fprintf(f, "precision=float\n");
        fprintf(f, "sizeof_float_saved=4\n");
        fprintf(f, "hex_digits=8\n");
    }
#endif
    fclose(f);
}

void save_all_matrices(const char* output_dir, QDLDL_int n, const QDLDL_int* Ap,
                       const QDLDL_int* Ai, const QDLDL_float* Ax_original,
                       const QDLDL_float* Ax_truncated, const QDLDL_int* Lp,
                       const QDLDL_int* Li, const QDLDL_float* Lx, const QDLDL_float* D,
                       const QDLDL_float* Dinv, const QDLDL_int* etree, const QDLDL_int* Lnz,
                       const QDLDL_float* b_vec, const QDLDL_float* x_vec) {
    char      filepath[512];
    QDLDL_int nnz_A = Ap[n];
    QDLDL_int nnz_L = Lp[n];

    mkdir(output_dir, 0755);

    printf("\n========================================\n");
    printf("[Saving matrices to %s]\n", output_dir);
    printf("========================================\n");

    // Save metadata
    snprintf(filepath, sizeof(filepath), "%s/metadata.txt", output_dir);
    save_metadata(filepath, n, nnz_A, nnz_L);
    printf("  Saved: metadata.txt\n");

    // Save A matrix (CSC format)
    snprintf(filepath, sizeof(filepath), "%s/Ap.hex", output_dir);
    save_int_hex(filepath, Ap, n + 1);
    snprintf(filepath, sizeof(filepath), "%s/Ai.hex", output_dir);
    save_int_hex(filepath, Ai, nnz_A);
    snprintf(filepath, sizeof(filepath), "%s/Ax_original.hex", output_dir);
    save_float_hex(filepath, Ax_original, nnz_A);
    snprintf(filepath, sizeof(filepath), "%s/Ax.hex", output_dir);
    save_float_hex(filepath, Ax_truncated, nnz_A);
    printf("  Saved: A matrix (Ap, Ai, Ax_original, Ax)\n");

    // Save L matrix (CSC format)
    snprintf(filepath, sizeof(filepath), "%s/Lp.hex", output_dir);
    save_int_hex(filepath, Lp, n + 1);
    snprintf(filepath, sizeof(filepath), "%s/Li.hex", output_dir);
    save_int_hex(filepath, Li, nnz_L);
    snprintf(filepath, sizeof(filepath), "%s/Lx.hex", output_dir);
    save_float_hex(filepath, Lx, nnz_L);
    printf("  Saved: L matrix (Lp, Li, Lx)\n");

    // Save D and Dinv
    snprintf(filepath, sizeof(filepath), "%s/D.hex", output_dir);
    save_float_hex(filepath, D, n);
    snprintf(filepath, sizeof(filepath), "%s/Dinv.hex", output_dir);
    save_float_hex(filepath, Dinv, n);
    printf("  Saved: D, Dinv\n");

    // Save etree and Lnz
    snprintf(filepath, sizeof(filepath), "%s/etree.hex", output_dir);
    save_int_hex(filepath, etree, n);
    snprintf(filepath, sizeof(filepath), "%s/Lnz.hex", output_dir);
    save_int_hex(filepath, Lnz, n);
    printf("  Saved: etree, Lnz\n");

    // Save b and x vectors
    snprintf(filepath, sizeof(filepath), "%s/b.hex", output_dir);
    save_float_hex(filepath, b_vec, n);
    snprintf(filepath, sizeof(filepath), "%s/x.hex", output_dir);
    save_float_hex(filepath, x_vec, n);
    printf("  Saved: b, x vectors\n");

    printf("========================================\n\n");
}

// ========================================
// FlexFloat truncation function
// ========================================
#ifdef QDLDL_USE_FLEXFLOAT
static void truncate_array_flexfloat(QDLDL_float* arr, QDLDL_int len) {
    flexfloat_desc_t desc = (flexfloat_desc_t){FF_exponent_bits, FF_mantissa_bits};
    flexfloat_t ff;

    printf("\n[Pre-truncating input array with FlexFloat E%dM%d]\n",
           FF_exponent_bits, FF_mantissa_bits);

    for(QDLDL_int i = 0; i < len; ++i) {
        // Initialize flexfloat from double - this applies truncation
        // Signature: ff_init_double(obj, value, desc)
        ff_init_double(&ff, (double)arr[i], desc);
        // Get the truncated value back
        arr[i] = (QDLDL_float)ff_get_double(&ff);
    }
}
#endif

// Test data
const QDLDL_int   An = 10;
const QDLDL_int   Ap[] = { 0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 17 };
const QDLDL_int   Ai[] = { 0, 1, 1, 2, 3, 4, 1, 5, 0, 6, 3, 7, 6, 8, 1, 2, 9 };
const QDLDL_float Ax[] = { 1.0,        0.460641,  -0.121189, 0.417928,  0.177828,   0.1,
                           -0.0290058, -1.0,      0.350321,  -0.441092, -0.0845395, -0.316228,
                           0.178663,   -0.299077, 0.182452,  -1.56506,  -0.1 };
const QDLDL_float b[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

int main(int argc, char* argv[]) {
    QDLDL_int i; // Counter

    // Output directory for matrix files (can be overridden via command line)
    const char* output_dir = "qdldl_output";
    if(argc > 1) {
        output_dir = argv[1];
    }

    // data for L and D factors
    QDLDL_int    Ln = An;
    QDLDL_int*   Lp;
    QDLDL_int*   Li;
    QDLDL_float* Lx;
    QDLDL_float* D;
    QDLDL_float* Dinv;

    // data for elim tree calculation
    QDLDL_int* etree;
    QDLDL_int* Lnz;
    QDLDL_int  sumLnz;

    // working data for factorisation
    QDLDL_int*   iwork;
    QDLDL_bool*  bwork;
    QDLDL_float* fwork;

    // Data for results of A\b
    QDLDL_float* x;

    // Mutable copy of Ax for pre-truncation
    QDLDL_float* Ax_truncated;

    // Print build configuration
    printf("\n========================================\n");
    printf("QDLDL Example - Build Configuration\n");
    printf("========================================\n");
    printf("  sizeof(QDLDL_int):   %zu bytes\n", sizeof(QDLDL_int));
    printf("  sizeof(QDLDL_float): %zu bytes\n", sizeof(QDLDL_float));
#ifdef QDLDL_USE_FLEXFLOAT
    printf("  FlexFloat:           ON\n");
#ifdef FF_exponent_bits
    printf("  Exponent bits:       %d\n", FF_exponent_bits);
#endif
#ifdef FF_mantissa_bits
    printf("  Mantissa bits:       %d\n", FF_mantissa_bits);
#endif
#else
    printf("  FlexFloat:           OFF\n");
    if(sizeof(QDLDL_float) == 8) {
        printf("  Precision:           double (E11M52)\n");
    } else {
        printf("  Precision:           float (E8M23)\n");
    }
#endif
    printf("========================================\n");

    /*--------------------------------
     * pre-factorisation memory allocations
     *---------------------------------*/

    // For the elimination tree
    etree = (QDLDL_int*) malloc(sizeof(QDLDL_int) * An);
    Lnz = (QDLDL_int*) malloc(sizeof(QDLDL_int) * An);

    // For the L factors
    Lp = (QDLDL_int*) malloc(sizeof(QDLDL_int) * (An + 1));
    D = (QDLDL_float*) malloc(sizeof(QDLDL_float) * An);
    Dinv = (QDLDL_float*) malloc(sizeof(QDLDL_float) * An);

    // Working memory
    iwork = (QDLDL_int*) malloc(sizeof(QDLDL_int) * (3 * An));
    bwork = (QDLDL_bool*) malloc(sizeof(QDLDL_bool) * An);
    fwork = (QDLDL_float*) malloc(sizeof(QDLDL_float) * An);

    // Allocate and copy Ax for potential truncation
    Ax_truncated = (QDLDL_float*) malloc(sizeof(QDLDL_float) * Ap[An]);
    for(i = 0; i < Ap[An]; i++) {
        Ax_truncated[i] = Ax[i];
    }

#ifdef QDLDL_USE_FLEXFLOAT
    // Pre-truncate input values through FlexFloat
    truncate_array_flexfloat(Ax_truncated, Ap[An]);
#endif

    /*--------------------------------
     * elimination tree calculation
     *---------------------------------*/
    sumLnz = QDLDL_etree(An, Ap, Ai, iwork, Lnz, etree);

    /*--------------------------------
     * LDL factorisation
     *---------------------------------*/
    Li = (QDLDL_int*) malloc(sizeof(QDLDL_int) * sumLnz);
    Lx = (QDLDL_float*) malloc(sizeof(QDLDL_float) * sumLnz);

    QDLDL_factor(An, Ap, Ai, Ax_truncated, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);

    /*--------------------------------
     * solve
     *---------------------------------*/
    x = (QDLDL_float*) malloc(sizeof(QDLDL_float) * An);

    for(i = 0; i < Ln; i++) {
        x[i] = b[i];
    }
    QDLDL_solve(Ln, Lp, Li, Lx, Dinv, x);

    /*--------------------------------
     * print factors and solution
     *---------------------------------*/
    printf("\n");
    printf("A (CSC format):\n");
    print_line();
    print_arrayi(Ap, An + 1, "A.p");
    print_arrayi(Ai, Ap[An], "A.i");
    print_arrayf(Ax_truncated, Ap[An], "A.x (truncated)");
    printf("\n\n");

    printf("elimination tree:\n");
    print_line();
    print_arrayi(etree, Ln, "etree");
    print_arrayi(Lnz, Ln, "Lnz");
    printf("\n\n");

    printf("L (CSC format):\n");
    print_line();
    print_arrayi(Lp, Ln + 1, "L.p");
    print_arrayi(Li, Lp[Ln], "L.i");
    print_arrayf(Lx, Lp[Ln], "L.x");
    printf("\n\n");

    printf("D:\n");
    print_line();
    print_arrayf(D, An, "diag(D)     ");
    print_arrayf(Dinv, An, "diag(D^{-1})");
    printf("\n\n");

    printf("solve results:\n");
    print_line();
    print_arrayf(b, An, "b");
    print_arrayf(x, An, "A\\b");
    printf("\n\n");

    /*--------------------------------
     * save matrices to files
     *---------------------------------*/
    save_all_matrices(output_dir, An, Ap, Ai, Ax, Ax_truncated, Lp, Li, Lx, D, Dinv, etree, Lnz, b, x);

    /*--------------------------------
     * clean up
     *---------------------------------*/
    free(Lp);
    free(Li);
    free(Lx);
    free(D);
    free(Dinv);
    free(etree);
    free(Lnz);
    free(iwork);
    free(bwork);
    free(fwork);
    free(Ax_truncated);
    free(x);

    return 0;
}

void print_line(void) {
    printf("--------------------------\n");
}

void print_arrayi(const QDLDL_int* data, QDLDL_int n, char* varName) {
    QDLDL_int i;
    printf("%s = [", varName);

    for(i = 0; i < n; i++) {
        printf("%i,", (int) data[i]);
    }
    printf("]\n");
}

void print_arrayf(const QDLDL_float* data, QDLDL_int n, char* varName) {
    QDLDL_int i;
    printf("%s = [", varName);

    for(i = 0; i < n; i++) {
        printf("%.3g,", data[i]);
    }
    printf("]\n");
}

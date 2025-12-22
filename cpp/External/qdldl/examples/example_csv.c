/**
 * QDLDL Example - CSV Matrix Input
 *
 * Reads a dense upper triangular matrix from CSV and performs LDL factorization.
 * Outputs Lp, Li, Lx to files.
 *
 * Usage:
 *   ./qdldl_csv <input.csv> [output_dir]
 */

#include "qdldl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

// ========================================
// Matrix saving functions (hex format)
// ========================================
static void save_int_hex(const char* filepath, const QDLDL_int* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if(!f) { perror(filepath); return; }
    if(sizeof(QDLDL_int) == 8) {
        for(QDLDL_int i = 0; i < len; ++i)
            fprintf(f, "%016" PRIx64 "\n", (uint64_t)arr[i]);
    } else {
        for(QDLDL_int i = 0; i < len; ++i)
            fprintf(f, "%08" PRIx32 "\n", (uint32_t)arr[i]);
    }
    fclose(f);
}

static void save_float_hex(const char* filepath, const QDLDL_float* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if(!f) { perror(filepath); return; }
    if(sizeof(QDLDL_float) == 8) {
        for(QDLDL_int i = 0; i < len; ++i) {
            uint64_t u; memcpy(&u, &arr[i], 8);
            fprintf(f, "%016" PRIx64 "\n", u);
        }
    } else {
        for(QDLDL_int i = 0; i < len; ++i) {
            uint32_t u; memcpy(&u, &arr[i], 4);
            fprintf(f, "%08" PRIx32 "\n", u);
        }
    }
    fclose(f);
}

// Save as decimal text (human readable)
static void save_int_dec(const char* filepath, const QDLDL_int* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if(!f) { perror(filepath); return; }
    for(QDLDL_int i = 0; i < len; ++i)
        fprintf(f, "%jd\n", (intmax_t)arr[i]);
    fclose(f);
}

// Save CSC row indices grouped by column (one line per column, comma separated)
static void save_csc_indices_by_col(const char* filepath, const QDLDL_int* p, const QDLDL_int* i, QDLDL_int n) {
    FILE* f = fopen(filepath, "w");
    if(!f) { perror(filepath); return; }

    for(QDLDL_int col = 0; col < n; col++) {
        QDLDL_int start = p[col];
        QDLDL_int end = p[col + 1];

        for(QDLDL_int idx = start; idx < end; idx++) {
            fprintf(f, "%jd", (intmax_t)i[idx]);
            if(idx < end - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// Save CSC values grouped by column (one line per column, comma separated)
static void save_csc_values_by_col(const char* filepath, const QDLDL_int* p, const QDLDL_float* x, QDLDL_int n) {
    FILE* f = fopen(filepath, "w");
    if(!f) { perror(filepath); return; }

    for(QDLDL_int col = 0; col < n; col++) {
        QDLDL_int start = p[col];
        QDLDL_int end = p[col + 1];

        for(QDLDL_int idx = start; idx < end; idx++) {
            fprintf(f, "%.17g", (double)x[idx]);
            if(idx < end - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

static void save_float_dec(const char* filepath, const QDLDL_float* arr, QDLDL_int len) {
    FILE* f = fopen(filepath, "w");
    if(!f) { perror(filepath); return; }
    for(QDLDL_int i = 0; i < len; ++i)
        fprintf(f, "%.17g\n", (double)arr[i]);
    fclose(f);
}

static void save_metadata(const char* filepath, QDLDL_int n, QDLDL_int nnz_A, QDLDL_int nnz_L) {
    FILE* f = fopen(filepath, "w");
    if(!f) { perror(filepath); return; }
    fprintf(f, "n=%jd\n", (intmax_t)n);
    fprintf(f, "nnz_A=%jd\n", (intmax_t)nnz_A);
    fprintf(f, "nnz_L=%jd\n", (intmax_t)nnz_L);
    fprintf(f, "sizeof_int=%zu\n", sizeof(QDLDL_int));
    fprintf(f, "sizeof_float=%zu\n", sizeof(QDLDL_float));
    fclose(f);
}

// ========================================
// CSV Reading and CSC Conversion
// ========================================

// Read CSV and convert to CSC upper triangular format
// Returns 0 on success, -1 on failure
int read_csv_to_csc_upper(const char* filename,
                          QDLDL_int* out_n,
                          QDLDL_int** out_Ap,
                          QDLDL_int** out_Ai,
                          QDLDL_float** out_Ax) {
    FILE* f = fopen(filename, "r");
    if(!f) {
        perror(filename);
        return -1;
    }

    // First pass: count rows and columns
    char line[1024*1024];  // Large buffer for wide matrices
    QDLDL_int n = 0;
    QDLDL_int ncols = 0;

    // Count rows
    while(fgets(line, sizeof(line), f)) {
        if(n == 0) {
            // Count columns from first row
            char* p = line;
            ncols = 1;
            while(*p) {
                if(*p == ',') ncols++;
                p++;
            }
        }
        n++;
    }

    printf("Matrix dimensions: %jd x %jd\n", (intmax_t)n, (intmax_t)ncols);

    if(n != ncols) {
        fprintf(stderr, "Error: Matrix must be square (got %jd x %jd)\n", (intmax_t)n, (intmax_t)ncols);
        fclose(f);
        return -1;
    }

    // Rewind and read matrix
    rewind(f);

    // Allocate dense matrix storage temporarily
    double* dense = (double*)calloc(n * n, sizeof(double));
    if(!dense) {
        fprintf(stderr, "Error: Cannot allocate memory for dense matrix\n");
        fclose(f);
        return -1;
    }

    QDLDL_int row = 0;
    while(fgets(line, sizeof(line), f) && row < n) {
        char* p = line;
        QDLDL_int col = 0;
        while(*p && col < n) {
            double val = strtod(p, &p);
            dense[row * n + col] = val;
            if(*p == ',') p++;
            col++;
        }
        row++;
    }
    fclose(f);

    // Count non-zeros in upper triangular part (including diagonal)
    QDLDL_int nnz = 0;
    for(QDLDL_int j = 0; j < n; j++) {
        for(QDLDL_int i = 0; i <= j; i++) {
            if(dense[i * n + j] != 0.0) {
                nnz++;
            }
        }
    }

    printf("Non-zeros in upper triangular: %jd\n", (intmax_t)nnz);

    // Allocate CSC arrays
    QDLDL_int* Ap = (QDLDL_int*)malloc((n + 1) * sizeof(QDLDL_int));
    QDLDL_int* Ai = (QDLDL_int*)malloc(nnz * sizeof(QDLDL_int));
    QDLDL_float* Ax = (QDLDL_float*)malloc(nnz * sizeof(QDLDL_float));

    if(!Ap || !Ai || !Ax) {
        fprintf(stderr, "Error: Cannot allocate CSC arrays\n");
        free(dense);
        free(Ap); free(Ai); free(Ax);
        return -1;
    }

    // Build CSC format (column by column, only upper triangular)
    QDLDL_int idx = 0;
    Ap[0] = 0;

    for(QDLDL_int j = 0; j < n; j++) {
        // For column j, store rows 0 to j (upper triangular)
        for(QDLDL_int i = 0; i <= j; i++) {
            double val = dense[i * n + j];
            if(val != 0.0) {
                Ai[idx] = i;
                Ax[idx] = (QDLDL_float)val;
                idx++;
            }
        }
        Ap[j + 1] = idx;
    }

    free(dense);

    *out_n = n;
    *out_Ap = Ap;
    *out_Ai = Ai;
    *out_Ax = Ax;

    return 0;
}

void print_arrayi(const QDLDL_int* data, QDLDL_int n, const char* varName) {
    printf("%s = [", varName);
    QDLDL_int limit = (n > 20) ? 20 : n;
    for(QDLDL_int i = 0; i < limit; i++) {
        printf("%jd", (intmax_t)data[i]);
        if(i < limit - 1) printf(", ");
    }
    if(n > 20) printf(", ... (%jd more)", (intmax_t)(n - 20));
    printf("]\n");
}

void print_arrayf(const QDLDL_float* data, QDLDL_int n, const char* varName) {
    printf("%s = [", varName);
    QDLDL_int limit = (n > 10) ? 10 : n;
    for(QDLDL_int i = 0; i < limit; i++) {
        printf("%.6g", (double)data[i]);
        if(i < limit - 1) printf(", ");
    }
    if(n > 10) printf(", ... (%jd more)", (intmax_t)(n - 10));
    printf("]\n");
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <input.csv> [output_dir]\n", argv[0]);
        return 1;
    }

    const char* input_csv = argv[1];
    const char* output_dir = (argc > 2) ? argv[2] : "qdldl_csv_output";

    printf("\n========================================\n");
    printf("QDLDL CSV Matrix Factorization\n");
    printf("========================================\n");
    printf("Input CSV: %s\n", input_csv);
    printf("Output dir: %s\n", output_dir);
    printf("sizeof(QDLDL_int): %zu\n", sizeof(QDLDL_int));
    printf("sizeof(QDLDL_float): %zu\n", sizeof(QDLDL_float));
    printf("========================================\n\n");

    // Read CSV to CSC
    QDLDL_int n;
    QDLDL_int* Ap;
    QDLDL_int* Ai;
    QDLDL_float* Ax;

    printf("Reading CSV file...\n");
    if(read_csv_to_csc_upper(input_csv, &n, &Ap, &Ai, &Ax) != 0) {
        return 1;
    }

    QDLDL_int nnz_A = Ap[n];
    printf("CSC conversion complete: n=%jd, nnz=%jd\n\n", (intmax_t)n, (intmax_t)nnz_A);

    // Allocate workspace for etree
    QDLDL_int* etree = (QDLDL_int*)malloc(n * sizeof(QDLDL_int));
    QDLDL_int* Lnz = (QDLDL_int*)malloc(n * sizeof(QDLDL_int));
    QDLDL_int* iwork = (QDLDL_int*)malloc(3 * n * sizeof(QDLDL_int));

    // Compute elimination tree
    printf("Computing elimination tree...\n");
    QDLDL_int sumLnz = QDLDL_etree(n, Ap, Ai, iwork, Lnz, etree);

    if(sumLnz < 0) {
        fprintf(stderr, "Error: QDLDL_etree failed with code %jd\n", (intmax_t)sumLnz);
        return 1;
    }
    printf("sumLnz = %jd (total non-zeros in L)\n\n", (intmax_t)sumLnz);

    // Allocate L factor storage
    QDLDL_int* Lp = (QDLDL_int*)malloc((n + 1) * sizeof(QDLDL_int));
    QDLDL_int* Li = (QDLDL_int*)malloc(sumLnz * sizeof(QDLDL_int));
    QDLDL_float* Lx = (QDLDL_float*)malloc(sumLnz * sizeof(QDLDL_float));
    QDLDL_float* D = (QDLDL_float*)malloc(n * sizeof(QDLDL_float));
    QDLDL_float* Dinv = (QDLDL_float*)malloc(n * sizeof(QDLDL_float));
    QDLDL_bool* bwork = (QDLDL_bool*)malloc(n * sizeof(QDLDL_bool));
    QDLDL_float* fwork = (QDLDL_float*)malloc(n * sizeof(QDLDL_float));

    // Perform LDL factorization
    printf("Performing LDL factorization...\n");
    QDLDL_int posD = QDLDL_factor(n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bwork, iwork, fwork);

    if(posD < 0) {
        fprintf(stderr, "Error: QDLDL_factor failed with code %jd\n", (intmax_t)posD);
        return 1;
    }
    printf("Factorization complete! Positive values in D: %jd / %jd\n\n", (intmax_t)posD, (intmax_t)n);

    QDLDL_int nnz_L = Lp[n];

    // Print summary
    printf("========================================\n");
    printf("Results Summary\n");
    printf("========================================\n");
    printf("Matrix size: %jd x %jd\n", (intmax_t)n, (intmax_t)n);
    printf("nnz(A upper): %jd\n", (intmax_t)nnz_A);
    printf("nnz(L): %jd\n", (intmax_t)nnz_L);
    printf("\n");

    print_arrayi(Ap, n + 1, "Ap");
    print_arrayi(Ai, (nnz_A > 50) ? 50 : nnz_A, "Ai");
    print_arrayf(Ax, (nnz_A > 20) ? 20 : nnz_A, "Ax");
    printf("\n");

    print_arrayi(Lp, n + 1, "Lp");
    print_arrayi(Li, (nnz_L > 50) ? 50 : nnz_L, "Li");
    print_arrayf(Lx, (nnz_L > 20) ? 20 : nnz_L, "Lx");
    printf("\n");

    print_arrayf(D, (n > 20) ? 20 : n, "D");
    printf("\n");

    // Create output directory and save files
    mkdir(output_dir, 0755);

    char filepath[512];

    // Save metadata
    snprintf(filepath, sizeof(filepath), "%s/metadata.txt", output_dir);
    save_metadata(filepath, n, nnz_A, nnz_L);

    // Save A matrix (CSC)
    snprintf(filepath, sizeof(filepath), "%s/Ap.hex", output_dir);
    save_int_hex(filepath, Ap, n + 1);
    snprintf(filepath, sizeof(filepath), "%s/Ai.hex", output_dir);
    save_int_hex(filepath, Ai, nnz_A);
    snprintf(filepath, sizeof(filepath), "%s/Ax.hex", output_dir);
    save_float_hex(filepath, Ax, nnz_A);

    // Save L matrix (CSC)
    snprintf(filepath, sizeof(filepath), "%s/Lp.hex", output_dir);
    save_int_hex(filepath, Lp, n + 1);
    snprintf(filepath, sizeof(filepath), "%s/Li.hex", output_dir);
    save_int_hex(filepath, Li, nnz_L);
    snprintf(filepath, sizeof(filepath), "%s/Lx.hex", output_dir);
    save_float_hex(filepath, Lx, nnz_L);

    // Save D and Dinv
    snprintf(filepath, sizeof(filepath), "%s/D.hex", output_dir);
    save_float_hex(filepath, D, n);
    snprintf(filepath, sizeof(filepath), "%s/Dinv.hex", output_dir);
    save_float_hex(filepath, Dinv, n);

    // Save etree and Lnz
    snprintf(filepath, sizeof(filepath), "%s/etree.hex", output_dir);
    save_int_hex(filepath, etree, n);
    snprintf(filepath, sizeof(filepath), "%s/Lnz.hex", output_dir);
    save_int_hex(filepath, Lnz, n);

    // Also save decimal versions for easier reading
    snprintf(filepath, sizeof(filepath), "%s/Lp.txt", output_dir);
    save_int_dec(filepath, Lp, n + 1);
    snprintf(filepath, sizeof(filepath), "%s/Li.txt", output_dir);
    save_int_dec(filepath, Li, nnz_L);
    snprintf(filepath, sizeof(filepath), "%s/Lx.txt", output_dir);
    save_float_dec(filepath, Lx, nnz_L);
    snprintf(filepath, sizeof(filepath), "%s/D.txt", output_dir);
    save_float_dec(filepath, D, n);

    // Save CSC data grouped by column (one line per column, comma separated)
    snprintf(filepath, sizeof(filepath), "%s/Li_by_col.txt", output_dir);
    save_csc_indices_by_col(filepath, Lp, Li, n);
    snprintf(filepath, sizeof(filepath), "%s/Lx_by_col.txt", output_dir);
    save_csc_values_by_col(filepath, Lp, Lx, n);
    snprintf(filepath, sizeof(filepath), "%s/Ai_by_col.txt", output_dir);
    save_csc_indices_by_col(filepath, Ap, Ai, n);
    snprintf(filepath, sizeof(filepath), "%s/Ax_by_col.txt", output_dir);
    save_csc_values_by_col(filepath, Ap, Ax, n);

    printf("========================================\n");
    printf("Files saved to: %s\n", output_dir);
    printf("========================================\n");
    printf("  - Ap.hex, Ai.hex, Ax.hex (input matrix A)\n");
    printf("  - Lp.hex, Li.hex, Lx.hex (L factor)\n");
    printf("  - Lp.txt, Li.txt, Lx.txt (decimal format)\n");
    printf("  - Li_by_col.txt, Lx_by_col.txt (grouped by column)\n");
    printf("  - Ai_by_col.txt, Ax_by_col.txt (grouped by column)\n");
    printf("  - D.hex, Dinv.hex, D.txt (diagonal)\n");
    printf("  - etree.hex, Lnz.hex\n");
    printf("  - metadata.txt\n");
    printf("========================================\n");

    // Cleanup
    free(Ap); free(Ai); free(Ax);
    free(Lp); free(Li); free(Lx);
    free(D); free(Dinv);
    free(etree); free(Lnz);
    free(iwork); free(bwork); free(fwork);

    return 0;
}

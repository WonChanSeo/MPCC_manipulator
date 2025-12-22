/*
 * FlexFloat FMA Test
 * Reads test vectors from testdata/ and compares ff_fma results with expected output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "flexfloat.h"

#define MAX_TEST_CASES 1024
#define HEX_LINE_LEN 16

// Test precision: E8M23 (standard float)
#define TEST_EXP_BITS 8
#define TEST_FRAC_BITS 23

typedef struct {
    uint32_t a[MAX_TEST_CASES];
    uint32_t b[MAX_TEST_CASES];
    uint32_t c[MAX_TEST_CASES];
    uint32_t expected[MAX_TEST_CASES];
    int count;
} TestData;

// Convert uint32_t to float (bit reinterpret)
static float u32_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

// Convert float to uint32_t (bit reinterpret)
static uint32_t float_to_u32(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(uint32_t));
    return u;
}

// Read hex file into array, return count
static int read_hex_file(const char* filename, uint32_t* arr, int max_count) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }

    char line[HEX_LINE_LEN];
    int count = 0;

    while (fgets(line, sizeof(line), f) && count < max_count) {
        // Skip empty lines
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }

        uint32_t val;
        if (sscanf(line, "%x", &val) == 1) {
            arr[count++] = val;
        }
    }

    fclose(f);
    return count;
}

// Load all test data
static int load_test_data(const char* testdata_dir, TestData* data) {
    char filepath[512];
    int count;

    // Load A
    snprintf(filepath, sizeof(filepath), "%s/fma_din_A.hex", testdata_dir);
    count = read_hex_file(filepath, data->a, MAX_TEST_CASES);
    if (count < 0) return -1;
    data->count = count;

    // Load B
    snprintf(filepath, sizeof(filepath), "%s/fma_din_B.hex", testdata_dir);
    count = read_hex_file(filepath, data->b, MAX_TEST_CASES);
    if (count < 0) return -1;
    if (count != data->count) {
        fprintf(stderr, "Error: Mismatched count in B (%d vs %d)\n", count, data->count);
        return -1;
    }

    // Load C
    snprintf(filepath, sizeof(filepath), "%s/fma_din_C.hex", testdata_dir);
    count = read_hex_file(filepath, data->c, MAX_TEST_CASES);
    if (count < 0) return -1;
    if (count != data->count) {
        fprintf(stderr, "Error: Mismatched count in C (%d vs %d)\n", count, data->count);
        return -1;
    }

    // Load expected result
    snprintf(filepath, sizeof(filepath), "%s/fma_result.hex", testdata_dir);
    count = read_hex_file(filepath, data->expected, MAX_TEST_CASES);
    if (count < 0) return -1;
    if (count != data->count) {
        fprintf(stderr, "Error: Mismatched count in result (%d vs %d)\n", count, data->count);
        return -1;
    }

    return 0;
}

// Check if two floats match (considering NaN)
static int floats_match(uint32_t actual, uint32_t expected) {
    // Exact bit match
    if (actual == expected) return 1;

    // Both NaN
    float f_actual = u32_to_float(actual);
    float f_expected = u32_to_float(expected);
    if (isnan(f_actual) && isnan(f_expected)) return 1;

    return 0;
}

// Run FMA test
static void run_fma_test(const TestData* data) {
    flexfloat_desc_t desc = {TEST_EXP_BITS, TEST_FRAC_BITS};

    int passed = 0;
    int failed = 0;

    printf("========================================\n");
    printf("FlexFloat FMA Test (E%dM%d)\n", TEST_EXP_BITS, TEST_FRAC_BITS);
    printf("========================================\n");
    printf("Total test cases: %d\n\n", data->count);

    for (int i = 0; i < data->count; i++) {
        // Convert input hex to float
        float fa = u32_to_float(data->a[i]);
        float fb = u32_to_float(data->b[i]);
        float fc = u32_to_float(data->c[i]);

        // Initialize flexfloat values
        flexfloat_t ff_a, ff_b, ff_c, ff_result;
        ff_init_float(&ff_a, fa, desc);
        ff_init_float(&ff_b, fb, desc);
        ff_init_float(&ff_c, fc, desc);
        ff_init(&ff_result, desc);

        // Perform FMA: result = a * b + c
        ff_fma(&ff_result, &ff_a, &ff_b, &ff_c);

        // Get result as float and convert to hex
        float f_result = ff_get_float(&ff_result);
        uint32_t actual_hex = float_to_u32(f_result);
        uint32_t expected_hex = data->expected[i];

        // Compare
        int match = floats_match(actual_hex, expected_hex);

        if (match) {
            passed++;
        } else {
            failed++;
            printf("[FAIL] Test %d:\n", i);
            printf("  A        = 0x%08X (%.8e)\n", data->a[i], fa);
            printf("  B        = 0x%08X (%.8e)\n", data->b[i], fb);
            printf("  C        = 0x%08X (%.8e)\n", data->c[i], fc);
            printf("  Expected = 0x%08X (%.8e)\n", expected_hex, u32_to_float(expected_hex));
            printf("  Actual   = 0x%08X (%.8e)\n", actual_hex, f_result);

            // Also show standard fmaf result for comparison
            float std_fma = fmaf(fa, fb, fc);
            printf("  std fmaf = 0x%08X (%.8e)\n", float_to_u32(std_fma), std_fma);
            printf("\n");
        }
    }

    printf("========================================\n");
    printf("Results: %d passed, %d failed out of %d\n", passed, failed, data->count);
    printf("========================================\n");

    if (failed == 0) {
        printf("\nAll tests PASSED!\n");
    } else {
        printf("\nSome tests FAILED.\n");
    }
}

int main(int argc, char* argv[]) {
    const char* testdata_dir = "testdata";

    if (argc > 1) {
        testdata_dir = argv[1];
    }

    printf("Loading test data from: %s\n\n", testdata_dir);

    TestData data;
    memset(&data, 0, sizeof(data));

    if (load_test_data(testdata_dir, &data) < 0) {
        fprintf(stderr, "Failed to load test data\n");
        return 1;
    }

    printf("Loaded %d test cases\n\n", data.count);

    run_fma_test(&data);

    return 0;
}

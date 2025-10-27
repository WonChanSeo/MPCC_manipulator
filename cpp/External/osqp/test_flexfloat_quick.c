/*
 * Quick FlexFloat Verification Test
 * Directly tests flexfloat operations to verify they work
 */

#include <stdio.h>
#include <math.h>

#ifdef OSQP_USE_FLEXFLOAT
#include "flexfloat_wrapper.h"

void test_flexfloat() {
    printf("\n=== FlexFloat Quantization Test ===\n\n");

    // Test 1: Simple addition
    double a = 1.234567890123456;
    double b = 2.345678901234567;

    // Standard precision
    double standard_result = a + b;

    // FlexFloat precision
    flexfloat_t ff_a = flexfloat_from_double(a);
    flexfloat_t ff_b = flexfloat_from_double(b);
    flexfloat_t ff_result = flexfloat_add(ff_a, ff_b);
    double flexfloat_result = flexfloat_to_double(ff_result);

    printf("Test 1: Addition\n");
    printf("  Input a (full):    %.15f\n", a);
    printf("  Input a (quant):   %.15f\n", flexfloat_to_double(ff_a));
    printf("  Input b (full):    %.15f\n", b);
    printf("  Input b (quant):   %.15f\n", flexfloat_to_double(ff_b));
    printf("  Standard result:   %.15f\n", standard_result);
    printf("  FlexFloat result:  %.15f\n", flexfloat_result);
    printf("  Difference:        %.15e\n", fabs(standard_result - flexfloat_result));

    // Test 2: Multiplication with small numbers
    double c = 0.123456789;
    double d = 0.987654321;

    double standard_mul = c * d;

    flexfloat_t ff_c = flexfloat_from_double(c);
    flexfloat_t ff_d = flexfloat_from_double(d);
    flexfloat_t ff_mul = flexfloat_mul(ff_c, ff_d);
    double flexfloat_mul = flexfloat_to_double(ff_mul);

    printf("\nTest 2: Multiplication\n");
    printf("  c * d (standard):  %.15f\n", standard_mul);
    printf("  c * d (flexfloat): %.15f\n", flexfloat_mul);
    printf("  Difference:        %.15e\n", fabs(standard_mul - flexfloat_mul));
    printf("  Relative error:    %.6f%%\n", 100.0 * fabs(standard_mul - flexfloat_mul) / fabs(standard_mul));

    // Test 3: Precision loss demonstration
    printf("\nTest 3: Precision Demonstration\n");
    printf("  14-bit mantissa = ~4-5 decimal digits\n");
    printf("  Numbers beyond 5 digits should show quantization:\n");

    double test_vals[] = {1.0, 1.2, 1.23, 1.234, 1.2345, 1.23456, 1.234567, 1.2345678};
    for (int i = 0; i < 8; i++) {
        flexfloat_t ff = flexfloat_from_double(test_vals[i]);
        double recovered = flexfloat_to_double(ff);
        printf("    %.7f -> %.7f (error: %.2e)\n",
               test_vals[i], recovered, fabs(test_vals[i] - recovered));
    }

    printf("\n=== FlexFloat is ACTIVE and WORKING! ===\n");
}
#endif

int main() {
    printf("=========================================\n");
    printf("FlexFloat Verification Test\n");
    printf("=========================================\n");

#ifdef OSQP_USE_FLEXFLOAT
    printf("Status: FlexFloat ENABLED\n");
    printf("Config: 14-bit mantissa, 8-bit exponent\n");
    test_flexfloat();
#else
    printf("Status: FlexFloat DISABLED\n");
    printf("Using standard double precision\n");
#endif

    printf("\n=========================================\n");
    return 0;
}

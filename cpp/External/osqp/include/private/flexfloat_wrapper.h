#ifndef FLEXFLOAT_WRAPPER_H
#define FLEXFLOAT_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <math.h>
#include <float.h>

/**
 * FlexFloat: Custom floating-point type with configurable precision
 * This implementation uses mantissa=14 bits, exponent=8 bits
 *
 * Format: [sign: 1 bit][exponent: 8 bits][mantissa: 14 bits] = 23 bits total
 * Represented internally as double for computation, with quantization applied
 */

/* FlexFloat configuration for OSQP */
#define FLEXFLOAT_MANTISSA_BITS 23  // float32와 동일 (full precision)
#define FLEXFLOAT_EXPONENT_BITS 8

/* Global flag to disable FlexFloat during scaling operations
 * 0 = FlexFloat enabled (ADMM iterations)
 * 1 = FlexFloat disabled (scaling operations)
 */
extern int g_disable_flexfloat_for_scaling;

/* Derived parameters */
#define FLEXFLOAT_EXPONENT_BIAS ((1 << (FLEXFLOAT_EXPONENT_BITS - 1)) - 1)  // 127
#define FLEXFLOAT_MAX_EXPONENT  ((1 << FLEXFLOAT_EXPONENT_BITS) - 1)        // 255
#define FLEXFLOAT_MANTISSA_MASK ((1ULL << FLEXFLOAT_MANTISSA_BITS) - 1)      // 0x3FFF

/* Range limits (same as IEEE-754 single precision with 8-bit exponent) */
#define FLEXFLOAT_MAX_VALUE  3.4028235e+38
#define FLEXFLOAT_MIN_VALUE  1.1754944e-38
#define FLEXFLOAT_EPSILON    6.1035156e-05  // 2^(-14) for 14-bit mantissa

/**
 * FlexFloat descriptor type (for Berkeley FlexFloat compatibility)
 * Describes the precision configuration: exponent bits and mantissa bits
 */
typedef struct {
    int exp_bits;   // Number of exponent bits
    int frac_bits;  // Number of mantissa/fraction bits
} flexfloat_desc_t;

/**
 * Internal representation of flexfloat number
 * Compatible with Berkeley FlexFloat API structure
 */
typedef struct {
    flexfloat_desc_t desc;  // Precision descriptor
    double value;           // Current quantized value
    uint32_t bits;          // [sign:1][exp:8][mantissa:20][unused:3] - internal representation
} flexfloat_t;

/**
 * Convert double to flexfloat with quantization
 * @param x Input double value
 * @return Quantized flexfloat representation
 */
static inline flexfloat_t flexfloat_from_double(double x) {
    flexfloat_t result;
    result.desc.exp_bits = FLEXFLOAT_EXPONENT_BITS;
    result.desc.frac_bits = FLEXFLOAT_MANTISSA_BITS;

    /* Bypass FlexFloat during scaling operations to preserve precision */
    if (g_disable_flexfloat_for_scaling) {
        result.value = x;
        result.bits = 0;
        return result;
    }

    /* Handle special cases */
    if (x == 0.0) {
        result.bits = (x < 0.0) ? 0x80000000 : 0x00000000;  // Preserve sign of zero
        result.value = x;
        return result;
    }
    if (isnan(x)) {
        result.bits = 0x7FC00000;  // NaN representation
        result.value = NAN;
        return result;
    }
    if (isinf(x)) {
        result.bits = (x < 0.0) ? 0xFF800000 : 0x7F800000;  // ±Infinity
        result.value = x;
        return result;
    }

    /* Extract sign */
    uint32_t sign = (x < 0.0) ? 1 : 0;
    double abs_x = fabs(x);

    /* Clamp to representable range */
    if (abs_x > FLEXFLOAT_MAX_VALUE) {
        result.bits = sign ? 0xFF800000 : 0x7F800000;  // Overflow to ±Infinity
        result.value = sign ? -INFINITY : INFINITY;
        return result;
    }
    if (abs_x < FLEXFLOAT_MIN_VALUE && abs_x != 0.0) {
        result.bits = sign << 31;  // Underflow to ±0
        result.value = sign ? -0.0 : 0.0;
        return result;
    }

    /* Extract exponent and mantissa using frexp */
    int exp_full;
    double mantissa_d = frexp(abs_x, &exp_full);  // x = mantissa_d * 2^exp_full, mantissa_d in [0.5, 1.0)

    /* Convert to biased exponent */
    int biased_exp = exp_full - 1 + FLEXFLOAT_EXPONENT_BIAS;  // -1 because frexp gives [0.5,1.0), we want [1.0,2.0)

    /* Handle subnormal numbers */
    if (biased_exp <= 0) {
        result.bits = sign << 31;  // Flush to zero
        result.value = sign ? -0.0 : 0.0;
        return result;
    }
    if (biased_exp >= FLEXFLOAT_MAX_EXPONENT) {
        result.bits = sign ? 0xFF800000 : 0x7F800000;  // Overflow to infinity
        result.value = sign ? -INFINITY : INFINITY;
        return result;
    }

    /* Quantize mantissa to FLEXFLOAT_MANTISSA_BITS */
    /* mantissa_d is in [0.5, 1.0), multiply by 2 to get [1.0, 2.0) with implicit leading 1 */
    mantissa_d = mantissa_d * 2.0 - 1.0;  // Remove implicit leading 1, now in [0.0, 1.0)

    /* Scale to n-bit integer and round */
    uint32_t mantissa_int = (uint32_t)(mantissa_d * (1 << FLEXFLOAT_MANTISSA_BITS) + 0.5);

    /* Handle rounding overflow */
    if (mantissa_int >= (1U << FLEXFLOAT_MANTISSA_BITS)) {
        mantissa_int = 0;
        biased_exp++;
        if (biased_exp >= FLEXFLOAT_MAX_EXPONENT) {
            result.bits = sign ? 0xFF800000 : 0x7F800000;  // Overflow to infinity
            result.value = sign ? -INFINITY : INFINITY;
            return result;
        }
    }

    /* Pack into 32-bit format: [sign:1][exp:8][mantissa:n][unused] */
    result.bits = (sign << 31) | ((biased_exp & 0xFF) << 23) | ((mantissa_int & ((1U << FLEXFLOAT_MANTISSA_BITS) - 1)) << (23 - FLEXFLOAT_MANTISSA_BITS));

    /* Reconstruct quantized value */
    int true_exp = biased_exp - FLEXFLOAT_EXPONENT_BIAS;
    double mantissa_reconstructed = 1.0 + (double)mantissa_int / (1 << FLEXFLOAT_MANTISSA_BITS);
    result.value = ldexp(mantissa_reconstructed, true_exp);
    if (sign) result.value = -result.value;

    return result;
}

/**
 * Convert flexfloat to double
 * @param ff Input flexfloat value
 * @return Reconstructed double value (stored in value field)
 */
static inline double flexfloat_to_double(flexfloat_t ff) {
    /* Simply return the quantized value */
    return ff.value;
}

/* Arithmetic operations with flexfloat quantization */

static inline flexfloat_t flexfloat_add(flexfloat_t a, flexfloat_t b) {
    double a_d = flexfloat_to_double(a);
    double b_d = flexfloat_to_double(b);
    double result_d = a_d + b_d;
    return flexfloat_from_double(result_d);
}

static inline flexfloat_t flexfloat_sub(flexfloat_t a, flexfloat_t b) {
    double a_d = flexfloat_to_double(a);
    double b_d = flexfloat_to_double(b);
    double result_d = a_d - b_d;
    return flexfloat_from_double(result_d);
}

static inline flexfloat_t flexfloat_mul(flexfloat_t a, flexfloat_t b) {
    double a_d = flexfloat_to_double(a);
    double b_d = flexfloat_to_double(b);
    double result_d = a_d * b_d;
    return flexfloat_from_double(result_d);
}

static inline flexfloat_t flexfloat_div(flexfloat_t a, flexfloat_t b) {
    double a_d = flexfloat_to_double(a);
    double b_d = flexfloat_to_double(b);
    double result_d = a_d / b_d;
    return flexfloat_from_double(result_d);
}

static inline flexfloat_t flexfloat_sqrt(flexfloat_t a) {
    double a_d = flexfloat_to_double(a);
    double result_d = sqrt(a_d);
    return flexfloat_from_double(result_d);
}

static inline flexfloat_t flexfloat_fabs(flexfloat_t a) {
    /* Just clear the sign bit */
    flexfloat_t result;
    result.bits = a.bits & 0x7FFFFFFF;
    return result;
}

/* Comparison operations */

static inline int flexfloat_lt(flexfloat_t a, flexfloat_t b) {
    return flexfloat_to_double(a) < flexfloat_to_double(b);
}

static inline int flexfloat_le(flexfloat_t a, flexfloat_t b) {
    return flexfloat_to_double(a) <= flexfloat_to_double(b);
}

static inline int flexfloat_gt(flexfloat_t a, flexfloat_t b) {
    return flexfloat_to_double(a) > flexfloat_to_double(b);
}

static inline int flexfloat_ge(flexfloat_t a, flexfloat_t b) {
    return flexfloat_to_double(a) >= flexfloat_to_double(b);
}

static inline int flexfloat_eq(flexfloat_t a, flexfloat_t b) {
    return a.bits == b.bits;  // Exact bit-level equality
}

/* Utility functions */

static inline flexfloat_t flexfloat_neg(flexfloat_t a) {
    flexfloat_t result;
    result.bits = a.bits ^ 0x80000000;  // Flip sign bit
    return result;
}

static inline flexfloat_t flexfloat_max(flexfloat_t a, flexfloat_t b) {
    return flexfloat_gt(a, b) ? a : b;
}

static inline flexfloat_t flexfloat_min(flexfloat_t a, flexfloat_t b) {
    return flexfloat_lt(a, b) ? a : b;
}

/* ============================================================================
 * Berkeley FlexFloat API compatibility layer for QDLDL
 * ============================================================================ */

/* Berkeley FlexFloat API: ff_* functions */
/* These are compatibility wrappers for QDLDL which expects Berkeley FlexFloat API */

static inline void flexfloat_sanitize(flexfloat_t *ff) {
    // Requantize the value to ensure it's within precision limits
    *ff = flexfloat_from_double(ff->value);
}

static inline void ff_init(flexfloat_t *result, double value) {
    *result = flexfloat_from_double(value);
}

static inline void ff_cast(flexfloat_t *result, const flexfloat_t *a, flexfloat_desc_t desc) {
    // For simplicity, just copy (we're using fixed precision anyway)
    (void)desc;  // Unused
    *result = *a;
}

static inline void ff_add(flexfloat_t *result, const flexfloat_t *a, const flexfloat_t *b) {
    *result = flexfloat_add(*a, *b);
}

static inline void ff_sub(flexfloat_t *result, const flexfloat_t *a, const flexfloat_t *b) {
    *result = flexfloat_sub(*a, *b);
}

static inline void ff_mul(flexfloat_t *result, const flexfloat_t *a, const flexfloat_t *b) {
    *result = flexfloat_mul(*a, *b);
}

static inline void ff_div(flexfloat_t *result, const flexfloat_t *a, const flexfloat_t *b) {
    *result = flexfloat_div(*a, *b);
}

static inline void ff_fma(flexfloat_t *result, const flexfloat_t *a, const flexfloat_t *b, const flexfloat_t *c) {
    // Fused multiply-add: result = a * b + c
    flexfloat_t temp = flexfloat_mul(*a, *b);
    *result = flexfloat_add(temp, *c);
}

static inline void ff_sqrt(flexfloat_t *result, const flexfloat_t *a) {
    *result = flexfloat_sqrt(*a);
}

static inline void ff_abs(flexfloat_t *result, const flexfloat_t *a) {
    *result = flexfloat_fabs(*a);
}

static inline void ff_neg(flexfloat_t *result, const flexfloat_t *a) {
    *result = flexfloat_neg(*a);
}

/* Comparison functions (return int: 1 for true, 0 for false) */
static inline int ff_eq(const flexfloat_t *a, const flexfloat_t *b) {
    return flexfloat_eq(*a, *b);
}

static inline int ff_lt(const flexfloat_t *a, const flexfloat_t *b) {
    return flexfloat_lt(*a, *b);
}

static inline int ff_le(const flexfloat_t *a, const flexfloat_t *b) {
    return flexfloat_le(*a, *b);
}

static inline int ff_gt(const flexfloat_t *a, const flexfloat_t *b) {
    return flexfloat_gt(*a, *b);
}

static inline int ff_ge(const flexfloat_t *a, const flexfloat_t *b) {
    return flexfloat_ge(*a, *b);
}

/* Conversion functions */
static inline double ff_get_double(const flexfloat_t *a) {
    return flexfloat_to_double(*a);
}

static inline float ff_get_float(const flexfloat_t *a) {
    return (float)flexfloat_to_double(*a);
}

#ifdef __cplusplus
}
#endif

#endif /* FLEXFLOAT_WRAPPER_H */

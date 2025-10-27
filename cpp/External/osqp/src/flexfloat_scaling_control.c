/*
 * FlexFloat Scaling Control
 *
 * This file defines a global flag to disable FlexFloat quantization
 * during scaling operations. Scaling involves very small values that
 * can underflow in reduced precision, so we bypass FlexFloat during
 * scaling to preserve numerical accuracy.
 */

#ifdef OSQP_USE_FLEXFLOAT

/* Global flag:
 * 0 = FlexFloat enabled (default for ADMM iterations)
 * 1 = FlexFloat disabled (for scaling operations)
 */
int g_disable_flexfloat_for_scaling = 0;

#endif

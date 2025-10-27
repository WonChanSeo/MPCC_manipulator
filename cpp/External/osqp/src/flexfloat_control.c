/**
 * FlexFloat control: Global flag to enable/disable quantization
 */

#ifdef OSQP_USE_FLEXFLOAT

/* Global flag: 0 = disabled (standard precision), 1 = enabled (FlexFloat quantization) */
int g_flexfloat_enabled = 0;

#endif /* OSQP_USE_FLEXFLOAT */

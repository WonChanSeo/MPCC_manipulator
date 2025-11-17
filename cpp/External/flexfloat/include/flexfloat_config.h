#ifndef FLEXFLOAT_CONFIG_H_INCLUDED
#define FLEXFLOAT_CONFIG_H_INCLUDED

/* Using double precision (FP64) as backend for OSQP stability */
#define FLEXFLOAT_ON_DOUBLE

/* Note: FP32 backend (FLEXFLOAT_ON_SINGLE) causes OSQP KKT matrix to be non-convex */
/* Only use FP32 if OSQP FlexFloat is disabled (USE_FLEXFLOAT=OFF) */
// #define FLEXFLOAT_ON_SINGLE

/* __float128 support available */
#define FLEXFLOAT_FLOAT128

#endif

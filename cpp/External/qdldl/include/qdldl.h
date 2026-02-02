/*
 * This file is part of QDLDL, a library for performing the LDL^T factorization
 * of a symmetric indefinite matrix.
 *
 * QDLDL is part of the OSQP project, and is available at https://github.com/osqp/qdldl.
 *
 * Copyright 2018, Paul Goulart, Bartolomeo Stellato, Goran Banjac, Ian McInerney, The OSQP developers
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
#ifndef QDLDL_H
#define QDLDL_H

// Include qdldl type options
#include "qdldl_types.h"
#include "qdldl_version.h"

// Define the function attributes that are needed to mark functions as being
// visible for linking in the shared library version of QDLDL
#if defined(_WIN32)
#if defined(BUILDING_QDLDL)
#define QDLDL_API_EXPORT __declspec(dllexport)
#else
#define QDLDL_API_EXPORT __declspec(dllimport)
#endif
#else
#if defined(BUILDING_QDLDL)
#define QDLDL_API_EXPORT __attribute__((visibility("default")))
#else
#define QDLDL_API_EXPORT
#endif
#endif

// Only define API export parts when using the shared library
#if defined(QDLDL_SHARED_LIB)
#define QDLDL_API QDLDL_API_EXPORT
#else
#define QDLDL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif // ifdef __cplusplus

/**
 * Compute the elimination tree for a quasidefinite matrix
 * in compressed sparse column form, where the input matrix is
 * assumed to contain data for the upper triangular part of A only,
 * and there are no duplicate indices.
 *
 * Returns an elimination tree for the factorization A = LDL^T and a
 * count of the nonzeros in each column of L that are strictly below the
 * diagonal.
 *
 * Does not use MALLOC.  It is assumed that the arrays work, Lnz, and
 * etree will be allocated with a number of elements equal to n.
 *
 * The data in (n,Ap,Ai) are from a square matrix A in CSC format, and
 * should include the upper triangular part of A only.
 *
 * This function is only intended for factorisation of QD matrices specified
 * by their upper triangular part.  An error is returned if any column has
 * data below the diagonal or s completely empty.
 *
 * For matrices with a non-empty column but a zero on the corresponding diagonal,
 * this function will *not* return an error, as it may still be possible to factor
 * such a matrix in LDL form.   No promises are made in this case though...
 *
 * @param  n      number of columns in CSC matrix A (assumed square)
 * @param  Ap     column pointers (size n+1) for columns of A
 * @param  Ai     row indices of A.  Has Ap[n] elements
 * @param  work   work vector (size n) (no meaning on return)
 * @param  Lnz    count of nonzeros in each column of L (size n) below diagonal
 * @param  etree  elimination tree (size n)
 * @return total  sum of Lnz (i.e. total nonzeros in L below diagonal).
 *                Returns -1 if the input is not triu or has an empty column.
 *                Returns -2 if the return value overflows QDLDL_int.
 *
 */
QDLDL_API QDLDL_int QDLDL_etree(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
                                QDLDL_int* work, QDLDL_int* Lnz, QDLDL_int* etree);


/**
 * Compute an LDL decomposition for a quasidefinite matrix
 * in compressed sparse column form, where the input matrix is
 * assumed to contain data for the upper triangular part of A only,
 * and there are no duplicate indices.
 *
 * Returns factors L, D and Dinv = 1./D.
 *
 * Does not use MALLOC.  It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx)  with sufficient space
 * allocated, with a number of nonzeros equal to the count given
 * as a return value by QDLDL_etree
 *
 * @param  n      number of columns in L and A (both square)
 * @param  Ap     column pointers (size n+1) for columns of A (not modified)
 * @param  Ai     row indices of A.  Has Ap[n] elements (not modified)
 * @param  Ax     data of A.  Has Ap[n] elements (not modified)
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  D      vectorized factor D.  Length is n
 * @param  Dinv   reciprocal of D.  Length is n
 * @param  Lnz    count of nonzeros in each column of L below diagonal,
 *                as given by QDLDL_etree (not modified)
 * @param  etree  elimination tree as as given by QDLDL_etree (not modified)
 * @param  bwork  working array of bools. Length is n
 * @param  iwork  working array of integers. Length is 3*n
 * @param  fwork  working array of floats. Length is n
 * @return        Returns a count of the number of positive elements
 *                in D.  Returns -1 and exits immediately if any element
 *                of D evaluates exactly to zero (matrix is not quasidefinite
 *                or otherwise LDL factorisable)
 *
 */
QDLDL_API QDLDL_int QDLDL_factor(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
                                 const QDLDL_float* Ax, QDLDL_int* Lp, QDLDL_int* Li,
                                 QDLDL_float* Lx, QDLDL_float* D, QDLDL_float* Dinv,
                                 const QDLDL_int* Lnz, const QDLDL_int* etree, QDLDL_bool* bwork,
                                 QDLDL_int* iwork, QDLDL_float* fwork);


/**
 * Right-looking LDL^T factorization (hardware bit-identical).
 *
 * Same inputs/outputs as QDLDL_factor, plus two hardware parameters.
 * Uses the same operation order as factorization_top.v:
 *   - 658×179 dense SRAM model (rows 479-657)
 *   - bank_sel masking per column
 *   - Separate mul + add (no FMA)
 *   - D[j<380] = -param_rho, D[j<479] = -param_rho_eq
 *
 * @param  param_rho     hardware rho parameter (FP32)
 * @param  param_rho_eq  hardware rho_eq parameter (FP32)
 * @return same as QDLDL_factor
 */
QDLDL_API QDLDL_int QDLDL_factor_right_looking(
    const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
    const QDLDL_float* Ax, QDLDL_int* Lp, QDLDL_int* Li,
    QDLDL_float* Lx, QDLDL_float* D, QDLDL_float* Dinv,
    const QDLDL_int* Lnz, const QDLDL_int* etree, QDLDL_bool* bwork,
    QDLDL_int* iwork, QDLDL_float* fwork,
    float param_rho, float param_rho_eq);


/**
  * Solves LDL'x = b
  *
  * It is assumed that L will be a compressed
  * sparse column matrix with data (n,Lp,Li,Lx).
  *
  * @param  n      number of columns in L
  * @param  Lp     column pointers (size n+1) for columns of L
  * @param  Li     row indices of L.  Has Lp[n] elements
  * @param  Lx     data of L.  Has Lp[n] elements
  * @param  Dinv   reciprocal of D.  Length is n
  * @param  x      initialized to b.  Equal to x on return
  *
  */
QDLDL_API void QDLDL_solve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                           const QDLDL_float* Lx, const QDLDL_float* Dinv, QDLDL_float* x);


/**
 * Solves (L+I)x = b
 *
 * It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx).
 *
 * @param  n      number of columns in L
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  x      initialized to b.  Equal to x on return
 *
 */
QDLDL_API void QDLDL_Lsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                            const QDLDL_float* Lx, QDLDL_float* x);


/**
 * Solves (L+I)'x = b
 *
 * It is assumed that L will be a compressed
 * sparse column matrix with data (n,Lp,Li,Lx).
 *
 * @param  n      number of columns in L
 * @param  Lp     column pointers (size n+1) for columns of L
 * @param  Li     row indices of L.  Has Lp[n] elements
 * @param  Lx     data of L.  Has Lp[n] elements
 * @param  x      initialized to b.  Equal to x on return
 *
 */
QDLDL_API void QDLDL_Ltsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                             const QDLDL_float* Lx, QDLDL_float* x);

// ========================================
// QDLDL Sample Data Logging API
// ========================================
// Enable sample logging by compiling with -DQDLDL_ENABLE_SAMPLE_LOGGING

#ifdef QDLDL_ENABLE_SAMPLE_LOGGING

/**
 * Initialize sample logging with output directory
 * @param output_dir  Directory where samples will be saved
 */
QDLDL_API void QDLDL_init_sample_logging(const char* output_dir);

/**
 * Reset the sample counter to 0
 */
QDLDL_API void QDLDL_reset_sample_counter(void);

/**
 * Get current sample count
 * @return Number of samples saved so far
 */
QDLDL_API int QDLDL_get_sample_count(void);

/**
 * Save a complete QDLDL sample (all inputs and outputs)
 * This saves: Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, etree, Lnz, x_input, x_output
 *
 * @param n          Matrix dimension
 * @param Ap         Column pointers for A (size n+1)
 * @param Ai         Row indices for A (size Ap[n])
 * @param Ax         Values for A (size Ap[n])
 * @param Lp         Column pointers for L (size n+1)
 * @param Li         Row indices for L (size Lp[n])
 * @param Lx         Values for L (size Lp[n])
 * @param D          Diagonal matrix D (size n)
 * @param Dinv       Inverse of D (size n)
 * @param etree      Elimination tree (size n)
 * @param Lnz        Nonzeros per column of L (size n)
 * @param x_input    Input b vector before solve (size n), can be NULL
 * @param x_output   Output x vector after solve (size n)
 */
QDLDL_API void QDLDL_save_sample(
    QDLDL_int n,
    const QDLDL_int* Ap, const QDLDL_int* Ai, const QDLDL_float* Ax,
    const QDLDL_int* Lp, const QDLDL_int* Li, const QDLDL_float* Lx,
    const QDLDL_float* D, const QDLDL_float* Dinv,
    const QDLDL_int* etree, const QDLDL_int* Lnz,
    const QDLDL_float* x_input, const QDLDL_float* x_output
);

/**
 * Store factor results for later save (called after QDLDL_factor)
 * The sample will be saved on the first subsequent solve call
 */
QDLDL_API void QDLDL_store_factor_for_sample(
    QDLDL_int n,
    const QDLDL_int* Ap, const QDLDL_int* Ai, const QDLDL_float* Ax,
    const QDLDL_int* Lp, const QDLDL_int* Li, const QDLDL_float* Lx,
    const QDLDL_float* D, const QDLDL_float* Dinv,
    const QDLDL_int* etree, const QDLDL_int* Lnz
);

/**
 * Check if there's a pending factor and save sample with solve results
 * Called automatically during solve to save factor + solve data together
 * Only saves once per factor call
 */
QDLDL_API void QDLDL_save_sample_on_solve(const QDLDL_float* x_input, const QDLDL_float* x_output);

#endif // QDLDL_ENABLE_SAMPLE_LOGGING

#ifdef __cplusplus
}
#endif // ifdef __cplusplus

#endif // ifndef QDLDL_H

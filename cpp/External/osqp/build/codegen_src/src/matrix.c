#include "osqp.h"
#include "lin_alg.h"
#include "algebra_impl.h"
#include "csc_math.h"
#include "csc_utils.h"
#include "printing.h"

#ifdef OSQP_USE_FLEXFLOAT
#include "flexfloat.h"
// FlexFloat precision bits are defined via compile-time flags:
// -DFF_exponent_bits and -DFF_mantissa_bits
#ifndef FF_exponent_bits
#define FF_exponent_bits 8
#endif
#ifndef FF_mantissa_bits
#define FF_mantissa_bits 23
#endif
#endif


#ifndef OSQP_EMBEDDED_MODE

/*  logical test functions ----------------------------------------------------*/

OSQPInt OSQPMatrix_is_eq(const OSQPMatrix* A,
                         const OSQPMatrix* B,
                         OSQPFloat         tol) {

  return (A->symmetry == B->symmetry &&
          csc_is_eq(A->csc, B->csc, tol) );
}


/*  Non-embeddable functions (using malloc) ----------------------------------*/

//Make a copy from a csc matrix.  Returns OSQP_NULL on failure
OSQPMatrix* OSQPMatrix_new_from_csc(const OSQPCscMatrix* A,
                                          OSQPInt        is_triu) {

  OSQPMatrix* out = c_malloc(sizeof(OSQPMatrix));
  if(!out) return OSQP_NULL;

  if(is_triu) out->symmetry = TRIU;
  else        out->symmetry = NONE;

  out->csc = csc_copy(A);

  if(!out->csc){
    c_free(out);
    return OSQP_NULL;
  }
  else{
    return out;
  }
}

OSQPCscMatrix* OSQPMatrix_get_csc(const OSQPMatrix* M) {return csc_copy(M->csc);}

// Make of a copy of a matrix
OSQPMatrix* OSQPMatrix_copy_new(const OSQPMatrix* A) {
    OSQPMatrix* out = c_malloc(sizeof(OSQPMatrix));
    if(!out) return OSQP_NULL;

    out->symmetry = A->symmetry;
    out->csc = csc_copy(A->csc);

    if(!out->csc){
        c_free(out);
        return OSQP_NULL;
    }
    else{
        return out;
    }
}

// Convert an upper triangular matrix into a fully populated matrix
OSQPMatrix* OSQPMatrix_triu_to_symm(const OSQPMatrix* A) {

    if (A->symmetry == TRIU) {
        OSQPMatrix* out = c_malloc(sizeof(OSQPMatrix));
        if(!out) return OSQP_NULL;

        out->symmetry = NONE;
        out->csc = triu_to_csc(A->csc);

        if (!out->csc) {
            c_free(out);
            return OSQP_NULL;
        } else{
            return out;
        }
    } else {
      c_eprint("input matrix not upper triangular");
      return OSQP_NULL;
    }
}

OSQPMatrix* OSQPMatrix_vstack(const OSQPMatrix* A,
                              const OSQPMatrix* B) {
    if ((A->symmetry == NONE) && (B->symmetry == NONE)) {
        OSQPMatrix* out = c_malloc(sizeof(OSQPMatrix));
        if(!out) return OSQP_NULL;

        out->symmetry = NONE;
        out->csc = vstack(A->csc, B->csc);

        if (!out->csc) {
            c_free(out);
            return OSQP_NULL;
        } else{
            return out;
        }
    } else {
        c_eprint("Can only vstack full matrices");
        return OSQP_NULL;
    }
}

#endif //OSQP_EMBEDDED_MODE

/*  direct data access functions ---------------------------------------------*/

void OSQPMatrix_update_values(OSQPMatrix*      M,
                              const OSQPFloat* Mx_new,
                              const OSQPInt*   Mx_new_idx,
                              OSQPInt          M_new_n) {
  csc_update_values(M->csc, Mx_new, Mx_new_idx, M_new_n);
}

/* Matrix dimensions and data access */
OSQPInt    OSQPMatrix_get_m(const OSQPMatrix* M)  {return M->csc->m;}
OSQPInt    OSQPMatrix_get_n(const OSQPMatrix* M)  {return M->csc->n;}
OSQPFloat* OSQPMatrix_get_x(const OSQPMatrix* M)  {return M->csc->x;}
OSQPInt*   OSQPMatrix_get_i(const OSQPMatrix* M)  {return M->csc->i;}
OSQPInt*   OSQPMatrix_get_p(const OSQPMatrix* M)  {return M->csc->p;}
OSQPInt    OSQPMatrix_get_nz(const OSQPMatrix* M) {return M->csc->p[M->csc->n];}

/* math functions ----------------------------------------------------------*/

//A = sc*A
void OSQPMatrix_mult_scalar(OSQPMatrix *A,
                            OSQPFloat   sc){
  csc_scale(A->csc,sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix*        A,
                           const OSQPVectorf* L) {
  csc_lmult_diag(A->csc, OSQPVectorf_data(L));
}

void OSQPMatrix_rmult_diag(OSQPMatrix* A,
                           const OSQPVectorf* R) {
  csc_rmult_diag(A->csc, R->values);
}

void OSQPMatrix_AtDA_extract_diag(const OSQPMatrix*  A,
                                  const OSQPVectorf* D,
                                        OSQPVectorf* d) {
    csc_AtDA_extract_diag(A->csc, OSQPVectorf_data(D), OSQPVectorf_data(d));
}

void OSQPMatrix_extract_diag(const OSQPMatrix*  A,
                                   OSQPVectorf* d) {
  csc_extract_diag(A->csc, OSQPVectorf_data(d));
}

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix*  A,
                     const OSQPVectorf* x,
                           OSQPVectorf* y,
                           OSQPFloat    alpha,
                           OSQPFloat    beta) {

  if(A->symmetry == NONE){
    //full matrix
    csc_Axpy(A->csc, x->values, y->values, alpha, beta);
  }
  else{
    //should be TRIU here, but not directly checked
    csc_Axpy_sym_triu(A->csc, x->values, y->values, alpha, beta);
  }
}

void OSQPMatrix_Atxpy(const OSQPMatrix*  A,
                      const OSQPVectorf* x,
                            OSQPVectorf* y,
                            OSQPFloat    alpha,
                            OSQPFloat    beta) {

   if(A->symmetry == NONE) csc_Atxpy(A->csc, x->values, y->values, alpha, beta);
   else            csc_Axpy_sym_triu(A->csc, x->values, y->values, alpha, beta);
}

// OSQPFloat OSQPMatrix_quad_form(const OSQPMatrix  *P,
//                              const OSQPVectorf *x) {

//    if (P->symmetry == TRIU) return csc_quad_form(P->csc, OSQPVectorf_data(x));
//    else {
//      c_eprint("quad_form matrix is not upper triangular");
//      return -1.0;
//    }
// }

#if OSQP_EMBEDDED_MODE != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E) {
   csc_col_norm_inf(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E) {
   if(M->symmetry == NONE) csc_row_norm_inf(M->csc, OSQPVectorf_data(E));
   else                    csc_row_norm_inf_sym_triu(M->csc, OSQPVectorf_data(E));
}

#endif // endef OSQP_EMBEDDED_MODE

#ifndef OSQP_EMBEDDED_MODE

void OSQPMatrix_free(OSQPMatrix* M){
  if (M) csc_spfree(M->csc);
  c_free(M);
}

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix*  A,
                                        const OSQPVectori* rows) {

  OSQPCscMatrix* M;
  OSQPMatrix*    out;


  if(A->symmetry == TRIU){
    c_eprint("row selection not implemented for partially filled matrices");
    return OSQP_NULL;
  }


  M = csc_submatrix_byrows(A->csc, rows->values);

  if(!M) return OSQP_NULL;

  out = c_malloc(sizeof(OSQPMatrix));

  if(!out){
    csc_spfree(M);
    return OSQP_NULL;
  }

  out->symmetry = NONE;
  out->csc      = M;

  return out;

}



#ifdef OSQP_USE_FLEXFLOAT
void OSQPMatrix_Axpy_FF(const OSQPMatrix*  A,
                     const OSQPVectorf* x,
                           OSQPVectorf* y,
                           OSQPFloat    alpha,
                           OSQPFloat    beta) {

  // y = alpha*A*x + beta*y
  OSQPInt i, j, k;
  OSQPCscMatrix* csc = A->csc;
  OSQPInt m = csc->m;
  OSQPInt n = csc->n;
  OSQPFloat* Ax = csc->x;
  OSQPInt* Ai = csc->i;
  OSQPInt* Ap = csc->p;

  flexfloat_t ff_yvi, ff_yvj, ff_Axk, ff_xvi, ff_xvj;

  flexfloat_t ff_alpha, ff_beta;
  ff_init_float(&ff_alpha, alpha, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_beta, beta, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  // First, scale existing y by beta
  for (i = 0; i < m; i++) {
    ff_init_float(&ff_yvi, y->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_mul(&ff_yvi, &ff_beta, &ff_yvi);
    // y->values[i] = beta * y->values[i];
    y->values[i] = ff_get_float(&ff_yvi);
  }

  // Compute y += alpha * A * x
  if (A->symmetry == NONE) {
    // General (non-symmetric) matrix
    for (j = 0; j < n; j++) {
      for (k = Ap[j]; k < Ap[j + 1]; k++) {
        i = Ai[k];
        // y->values[i] += alpha * Ax[k] * x->values[j];
        ff_init_float(&ff_yvi, y->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
        ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
        ff_init_float(&ff_xvj, x->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

        ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
        // ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
        // ff_add(&ff_yvi, &ff_Axk, &ff_yvi);
        ff_fma(&ff_yvi, &ff_xvj, &ff_Axk, &ff_yvi);  // ff_yvi += ff_Axk * ff_xvj

        y->values[i] = ff_get_float(&ff_yvi);
      }
    }
  }
  else {
    // Symmetric matrix (stored as upper triangular)
    for (j = 0; j < n; j++) {
      for (k = Ap[j]; k < Ap[j + 1]; k++) {
        i = Ai[k];
        if (i == j) {
          // Diagonal element
          ff_init_float(&ff_yvi, y->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvj, x->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
 
          // y->values[i] += alpha * Ax[k] * x->values[j];
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          // ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
          // ff_add(&ff_yvi, &ff_Axk, &ff_yvi);
          ff_fma(&ff_yvi, &ff_xvj, &ff_Axk, &ff_yvi);  // ff_yvi += ff_Axk * ff_xvj

          y->values[i] = ff_get_float(&ff_yvi);
        }
        else if (i < j) {
          // Upper triangular element (use symmetry)
          ff_init_float(&ff_yvi, y->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvj, x->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          // y->values[i] += alpha * Ax[k] * x->values[j];
          // y->values[j] += alpha * Ax[k] * x->values[i];
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          // ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
          // ff_add(&ff_yvi, &ff_Axk, &ff_yvi);
          ff_fma(&ff_yvi, &ff_xvj, &ff_Axk, &ff_yvi);  // ff_yvi += ff_Axk * ff_xvj
          y->values[i] = ff_get_float(&ff_yvi);

          // Now update y->values[j]
          ff_init_float(&ff_yvj, y->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvi, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);          
          // ff_mul(&ff_Axk, &ff_xvi, &ff_Axk);  // Use ff_xvi, not ff_xvj!
          // ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
          ff_fma(&ff_yvj, &ff_xvi, &ff_Axk, &ff_yvj);  // ff_yvj += ff_Axk * ff_xvi
          y->values[j] = ff_get_float(&ff_yvj);
        }
      }
    }
  }
}

void OSQPMatrix_Atxpy_FF(const OSQPMatrix*  A,
                      const OSQPVectorf* x,
                            OSQPVectorf* y,
                            OSQPFloat    alpha,
                            OSQPFloat    beta) {

  // y = alpha*A^T*x + beta*y
  OSQPInt i, j, k;
  OSQPCscMatrix* csc = A->csc;
  OSQPInt m = csc->m;
  OSQPInt n = csc->n;
  OSQPFloat* Ax = csc->x;
  OSQPInt* Ai = csc->i;
  OSQPInt* Ap = csc->p;

  flexfloat_t ff_yvi, ff_yvj, ff_Axk, ff_xvi, ff_xvj;
  flexfloat_t ff_alpha, ff_beta;
  ff_init_float(&ff_alpha, alpha, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_beta, beta, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  // First, scale existing y by beta
  for (j = 0; j < n; j++) {
    // y->values[j] = beta * y->values[j];
    ff_init_float(&ff_yvj, y->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_mul(&ff_yvj, &ff_beta, &ff_yvj);
    y->values[j] = ff_get_float(&ff_yvj);
  }

  // Compute y += alpha * A^T * x
  if (A->symmetry == NONE) {
    // General (non-symmetric) matrix
    // For A^T*x: column j of A becomes row j of A^T
    for (j = 0; j < n; j++) {
      for (k = Ap[j]; k < Ap[j + 1]; k++) {
        i = Ai[k];
        // y->values[j] += alpha * Ax[k] * x->values[i];
        ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
        ff_init_float(&ff_xvi, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
        ff_init_float(&ff_yvj, y->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
        ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
        // ff_mul(&ff_Axk, &ff_xvi, &ff_Axk);
        // ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
        ff_fma(&ff_yvj, &ff_xvi, &ff_Axk, &ff_yvj);  // ff_yvj += ff_Axk * ff_xvi
        y->values[j] = ff_get_float(&ff_yvj);  // Store the result!
      }
    }
  }
  else {
    // Symmetric matrix: A^T = A, so same as Axpy
    for (j = 0; j < n; j++) {
      for (k = Ap[j]; k < Ap[j + 1]; k++) {
        i = Ai[k];
        if (i == j) {
          // Diagonal element
          // y->values[j] += alpha * Ax[k] * x->values[i];

          ff_init_float(&ff_yvj, y->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvi, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          // ff_mul(&ff_Axk, &ff_xvi, &ff_Axk);
          // ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
          ff_fma(&ff_yvj, &ff_xvi, &ff_Axk, &ff_yvj);  // ff_yvj += ff_Axk * ff_xvi
          y->values[j] = ff_get_float(&ff_yvj);  // Store the result!
        }
        else if (i < j) {
          // Upper triangular element (use symmetry)
          // y->values[j] += alpha * Ax[k] * x->values[i];
          // y->values[i] += alpha * Ax[k] * x->values[j];

          // First update y->values[j]
          ff_init_float(&ff_yvj, y->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvi, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          // ff_mul(&ff_Axk, &ff_xvi, &ff_Axk);
          // ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
          ff_fma(&ff_yvj, &ff_xvi, &ff_Axk, &ff_yvj);  // ff_yvj += ff_Axk * ff_xvi
          y->values[j] = ff_get_float(&ff_yvj);

          // Now update y->values[i]
          ff_init_float(&ff_yvi, y->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvj, x->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          // ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
          // ff_add(&ff_yvi, &ff_Axk, &ff_yvi);
          ff_fma(&ff_yvi, &ff_xvj, &ff_Axk, &ff_yvi);  // ff_yvi += ff_Axk * ff_xvj
          y->values[i] = ff_get_float(&ff_yvi);
        }
      }
    }
  }
}

#endif /* OSQP_USE_FLEXFLOAT */

#endif /* OSQP_EMBEDDED_MODE != 1 */

#include "osqp.h"
#include "lin_alg.h"
#include "algebra_impl.h"
#include "csc_math.h"
#include "csc_utils.h"
#include "printing.h"

#include "blas_helpers.h"

#include "flexfloat.h"

#define FF_mantissa_bits 17;
#define FF_exponent_bits 8;
/*  logical test functions ----------------------------------------------------*/

OSQPInt OSQPMatrix_is_eq(const OSQPMatrix* A,
                         const OSQPMatrix* B,
                         OSQPFloat         tol) {

  return (A->symmetry == B->symmetry &&
          csc_is_eq(A->csc, B->csc, tol) );
}

/* Special routine to allocate a OSQPCscMatrix using the MKL memory routines
   instead of the normal memory routines */
static OSQPCscMatrix* mkl_csc_spalloc(OSQPInt m, OSQPInt n, OSQPInt nzmax, OSQPInt hasData) {
  OSQPCscMatrix* csc = c_calloc(1, sizeof(OSQPCscMatrix));

  if(!csc){
    return OSQP_NULL;
  }

  csc->m     = m;
  csc->n     = n;
  csc->nzmax = c_max(nzmax, 0);
  csc->nz    = -1;
  csc->p     = blas_malloc((n + 1) * sizeof(OSQPInt));
  csc->i     = hasData ? blas_malloc(nzmax * sizeof(OSQPInt)) : OSQP_NULL;
  csc->x     = hasData ? blas_malloc(nzmax * sizeof(OSQPFloat)) : OSQP_NULL;

  return csc;
}

static void mkl_csc_spfree(OSQPCscMatrix* M) {
  if(M) {
    if(M->p)
      blas_free(M->p);

    if(M->i)
      blas_free(M->i);

    if(M->x)
      blas_free(M->x);

    c_free(M);
  }
}

//Make a copy from a csc matrix.  Returns OSQP_NULL on failure
OSQPMatrix* OSQPMatrix_new_from_csc(const OSQPCscMatrix* A,
                                          OSQPInt        is_triu) {

  OSQPInt i       = 0;
  OSQPInt n       = A->n;   /* Number of columns */
  OSQPInt m       = A->m;   /* Number of rows */
  OSQPInt nzmax   = A->nzmax; /* Number of non-zeros */
  OSQPInt hasData = (A->x != OSQP_NULL); /* Input A matrix has data */

  MKL_INT retval = 0;

  OSQPMatrix* out = c_calloc(1, sizeof(OSQPMatrix));

  if (!out)
   return OSQP_NULL;

  if (is_triu)
    out->symmetry = TRIU;
  else
    out->symmetry = NONE;

  /* We specially allocate this matrix using the MKL memory routines,
     so it should NEVER touch the normal memory routines. */
  out->csc = mkl_csc_spalloc(m, n, nzmax, hasData);

  if(!out->csc){
    c_free(out);
    return OSQP_NULL;
  }

  for(i=0; i < n+1; i++) {
    out->csc->p[i] = A->p[i];
  }

  for(i=0; i < nzmax; i++) {
    out->csc->i[i] = A->i[i];
    out->csc->x[i] = A->x[i];
  }

  retval = spblas_create_csc(&out->mkl_mat,
                             SPARSE_INDEX_BASE_ZERO,
                             out->csc->m,      /* Number of rows */
                             out->csc->n,      /* Number of columns */
                             out->csc->p,      /* Array of column start indices (this will only look at the first n entries in p, skipping the last one) */
                             out->csc->p+1,    /* Array of column end indices (this will skip the first entry to only look at the last n) */
                             out->csc->i,      /* Array of row indices */
                             out->csc->x);     /* The actual data */

  /* We expect the SPARSE_STATUS_NOT_INITIALIZED return value if the matrix is either
     empty (no non-zero entries) or has zero rows/columns, so we treat it as a success
     as well so we still get an MKL matrix. */
  if (retval != SPARSE_STATUS_SUCCESS && retval != SPARSE_STATUS_NOT_INITIALIZED) {
    OSQPMatrix_free(out);
    return OSQP_NULL;
  }

  return out;
}

/*  direct data access functions ---------------------------------------------*/

void OSQPMatrix_update_values(OSQPMatrix*    M,
                            const OSQPFloat* Mx_new,
                            const OSQPInt*   Mx_new_idx,
                            OSQPInt          M_new_n) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_update_values(M->csc, Mx_new, Mx_new_idx, M_new_n);
}

/* Matrix dimensions and data access */
OSQPInt    OSQPMatrix_get_m(const OSQPMatrix* M)  {return M->csc->m;}
OSQPInt    OSQPMatrix_get_n(const OSQPMatrix* M)  {return M->csc->n;}
OSQPFloat* OSQPMatrix_get_x(const OSQPMatrix* M)  {return M->csc->x;}
OSQPInt*   OSQPMatrix_get_i(const OSQPMatrix* M)  {return M->csc->i;}
OSQPInt*   OSQPMatrix_get_p(const OSQPMatrix* M)  {return M->csc->p;}
OSQPInt    OSQPMatrix_get_nz(const OSQPMatrix* M) {return M->csc->p[M->csc->n];}

OSQPCscMatrix* OSQPMatrix_get_csc(const OSQPMatrix* M) {
  /* Values returned from the MKL object */
  sparse_index_base_t idx_method = 0;
  MKL_INT numrows = 0;
  MKL_INT numcols = 0;

  MKL_INT *p_start;
  MKL_INT *p_end;
  MKL_INT *row_idx;
  OSQPFloat *vals;

  /* Computed values */
  OSQPCscMatrix* B;
  OSQPInt i = 0;
  OSQPInt nnz = 0;

  spblas_export_csc(M->mkl_mat, &idx_method, &numrows, &numcols, &p_start, &p_end, &row_idx, &vals);

  /* Create the CSC using the returned data */
  nnz = p_end[numcols-1]+1;
  B = csc_spalloc(numcols, numrows, nnz, 1, 0);

  /* MKL doesn't give back the actual p we need, we need to take the last value from p_end and concatenate
     it onto the array returned in p_start */
  for (i = 0; i < numcols; i++) {
    B->p[i] = p_start[i];
  }
  B->p[numcols] = p_end[numcols-1] + 1;

  for (i=0; i < nnz; i++) {
    B->i[i] = row_idx[i];
    B->x[i] = vals[i];
  }

  return B;
}


/* math functions ----------------------------------------------------------*/

//A = sc*A
void OSQPMatrix_mult_scalar(OSQPMatrix* A,
                            OSQPFloat   sc) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_scale(A->csc, sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix*        A,
                           const OSQPVectorf* L) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_lmult_diag(A->csc, OSQPVectorf_data(L));
}

void OSQPMatrix_rmult_diag(OSQPMatrix*        A,
                           const OSQPVectorf* R) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_rmult_diag(A->csc, OSQPVectorf_data(R));
}

void OSQPMatrix_AtDA_extract_diag(const OSQPMatrix*  A,
                                  const OSQPVectorf* D,
                                        OSQPVectorf* d) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
    csc_AtDA_extract_diag(A->csc, OSQPVectorf_data(D), OSQPVectorf_data(d));
}

void OSQPMatrix_extract_diag(const OSQPMatrix*  A,
                                   OSQPVectorf* d) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_extract_diag(A->csc, OSQPVectorf_data(d));
}

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix*  A,
                     const OSQPVectorf* x,
                           OSQPVectorf* y,
                           OSQPFloat    alpha,
                           OSQPFloat    beta) {

  struct matrix_descr descr;

  if(A->symmetry == NONE){
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    /* These don't actually matter for this mode, but put them to a known value */
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }
  else{
    /* Assumed to be TRIU if not NONE */
    descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }

  spblas_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A->mkl_mat, descr, x->values, beta, y->values);
}

//y = alpha*A*x + beta*y
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
        ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
        ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
        ff_add(&ff_yvi, &ff_Axk, &ff_yvi);

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
          ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          ff_add(&ff_yvi, &ff_Axk, &ff_yvi);

          y->values[i] = ff_get_float(&ff_yvi);
        }
        else if (i < j) {
          // Upper triangular element (use symmetry)
          ff_init_float(&ff_yvi, y->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvj, x->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          // y->values[i] += alpha * Ax[k] * x->values[j];
          // y->values[j] += alpha * Ax[k] * x->values[i];

          ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          ff_add(&ff_yvi, &ff_Axk, &ff_yvi);
          y->values[i] = ff_get_float(&ff_yvi);

          // Now update y->values[j]
          ff_init_float(&ff_yvj, y->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvi, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
          y->values[j] = ff_get_float(&ff_yvj);
        }
      }
    }
  }
}

void OSQPMatrix_Atxpy(const OSQPMatrix*  A,
                      const OSQPVectorf* x,
                            OSQPVectorf* y,
                            OSQPFloat    alpha,
                            OSQPFloat    beta) {
  struct matrix_descr descr;

  if(A->symmetry == NONE){
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    /* These don't actually matter for this mode, but put them to a known value */
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }
  else{
    /* Assumed to be TRIU if not NONE */
    descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }

  spblas_mv(SPARSE_OPERATION_TRANSPOSE, alpha, A->mkl_mat, descr, x->values, beta, y->values);
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
        
        ff_mul(&ff_Axk, &ff_xvi, &ff_Axk);
        ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
        ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
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

          ff_mul(&ff_Axk, &ff_xvi, &ff_Axk);
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
        }
        else if (i < j) {
          // Upper triangular element (use symmetry)
          // y->values[j] += alpha * Ax[k] * x->values[i];
          // y->values[i] += alpha * Ax[k] * x->values[j];

          y->values[j] += alpha * Ax[k] * x->values[i];
          y->values[i] += alpha * Ax[k] * x->values[j];
          ff_init_float(&ff_yvj, y->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvi, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_mul(&ff_Axk, &ff_xvi, &ff_Axk);
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          ff_add(&ff_yvj, &ff_Axk, &ff_yvj);
          y->values[j] = ff_get_float(&ff_yvj);

          // Now update y->values[i]
          ff_init_float(&ff_yvi, y->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_Axk, Ax[k], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_init_float(&ff_xvj, x->values[j], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
          ff_mul(&ff_Axk, &ff_xvj, &ff_Axk);
          ff_mul(&ff_Axk, &ff_alpha, &ff_Axk);
          ff_add(&ff_yvi, &ff_Axk, &ff_yvi);
          y->values[i] = ff_get_float(&ff_yvi);
        }
      }
    }
  }
}

void OSQPMatrix_col_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
   csc_col_norm_inf(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
   if(M->symmetry == NONE) csc_row_norm_inf(M->csc, OSQPVectorf_data(E));
   else                    csc_row_norm_inf_sym_triu(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_free(OSQPMatrix* M) {
  if (M) {
    if(M->mkl_mat)
      mkl_sparse_destroy(M->mkl_mat);

    /* This CSC matrix is special, it wasn't created using the normal memory routines, it uses the
       blas-specific memory routines. Therefore it must be destroyed in a special way */
    mkl_csc_spfree(M->csc);
  };

  c_free(M);
}

static void int_vec_set_scalar(OSQPInt* a, OSQPInt sc, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++) a[i] = sc;
}

/* A modified version of the csc_submatrix_byrows that uses the MKL memory routines. */
static OSQPCscMatrix* mkl_submatrix_byrows(const OSQPCscMatrix* A,
                                          OSQPInt*       rows){

  OSQPInt        j;
  OSQPCscMatrix* R;
  OSQPInt        nzR = 0;
  OSQPInt        An = A->n;
  OSQPInt        Am = A->m;
  OSQPInt*       Ap = A->p;
  OSQPInt*       Ai = A->i;
  OSQPFloat*     Ax = A->x;
  OSQPInt*       Rp;
  OSQPInt*       Ri;
  OSQPFloat*     Rx;
  OSQPInt        Rm = 0;
  OSQPInt        ptr;
  OSQPInt*       rridx; //mapping from row indices to reduced row indices

  rridx = (OSQPInt*)c_malloc(Am * sizeof(OSQPInt));
  if(!rridx) return OSQP_NULL;

  //count the number of rows in the reduced
  //matrix, and build an index from the input
  //matrix rows to the reduced rows
  Rm    = 0;
  for(j = 0; j < Am; j++){
     if(rows[j]) rridx[j] = Rm++;
  }

  // Count number of nonzeros in Ared
  for (j = 0; j < Ap[An]; j++) {
    if(rows[A->i[j]]) nzR++;
  }

  // Form R = A(rows,:), where nrows = sum(rows != 0)
  R = mkl_csc_spalloc(Rm, An, nzR, 1);
  if (!R) return OSQP_NULL;

  // no active constraints
  if (Rm == 0) {
    int_vec_set_scalar(R->p, 0, An + 1);
  }

  else{
    nzR = 0; // reset counter
    Rp = R->p;
    Ri = R->i;
    Rx = R->x;

    for (j = 0; j < An; j++) { // Cycle over columns of A
      Rp[j] = nzR;
      for (ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
        // Cycle over elements in j-th column
        if (rows[A->i[ptr]]) {
          Ri[nzR] = rridx[Ai[ptr]];
          Rx[nzR] = Ax[ptr];
          nzR++;
    }}}
    // Update the last element in R->p
    Rp[An] = nzR;
  }

  c_free(rridx); //free internal work index

  return R;
}

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix*  A,
                                        const OSQPVectori* rows) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  OSQPCscMatrix* M;
  OSQPMatrix*    out;
  OSQPInt        retval = SPARSE_STATUS_SUCCESS;

  if(A->symmetry == TRIU){
    c_eprint("row selection not implemented for partially filled matrices");
    return OSQP_NULL;
  }

  M = mkl_submatrix_byrows(A->csc, rows->values);

  if(!M) return OSQP_NULL;

  out = c_calloc(1, sizeof(OSQPMatrix));

  if(!out){
    mkl_csc_spfree(M);
    return OSQP_NULL;
  }

  out->symmetry = NONE;
  out->csc      = M;

  if(!out->csc){
    c_free(out);
    return OSQP_NULL;
  }

  retval = spblas_create_csc(&out->mkl_mat,
                             SPARSE_INDEX_BASE_ZERO,
                             out->csc->m,      /* Number of rows */
                             out->csc->n,      /* Number of columns */
                             out->csc->p,      /* Array of column start indices (this will only look at the first n entries in p, skipping the last one) */
                             out->csc->p+1,    /* Array of column end indices (this will skip the first entry to only look at the last n) */
                             out->csc->i,      /* Array of row indices */
                             out->csc->x);     /* The actual data */

  /* We expect the SPARSE_STATUS_NOT_INITIALIZED return value if the matrix is either
     empty (no non-zero entries) or has zero rows/columns, so we treat it as a success
     as well so we still get an MKL matrix. */
  if (retval != SPARSE_STATUS_SUCCESS && retval != SPARSE_STATUS_NOT_INITIALIZED) {
    OSQPMatrix_free(out);
    return OSQP_NULL;
  }

  return out;
}

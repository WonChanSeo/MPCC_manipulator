#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_impl.h"

#ifdef OSQP_USE_FLEXFLOAT
#include "flexfloat.h"
#define FF_mantissa_bits 20
#define FF_exponent_bits 8
#endif

/* VECTOR FUNCTIONS ----------------------------------------------------------*/

#ifndef OSQP_EMBEDDED_MODE

OSQPInt OSQPVectorf_is_eq(const OSQPVectorf* A,
                          const OSQPVectorf* B,
                                OSQPFloat    tol) {

    if (A->length != B->length) return 0;
    OSQPInt i;
    OSQPInt retval = 1;
    for (i=0; i<A->length; i++) {
        if (c_absval(A->values[i] - B->values[i]) > tol) {
            retval = 0;
        }
    }
    return retval;
}

OSQPVectorf* OSQPVectorf_new(const OSQPFloat* a,
                                   OSQPInt    length) {

  OSQPVectorf* out = OSQPVectorf_malloc(length);
  if(!out) return OSQP_NULL;

  if (length > 0) {
    OSQPVectorf_from_raw(out, a);
  }
  return out;
}

OSQPVectori* OSQPVectori_new(const OSQPInt* a,
                                   OSQPInt  length) {

  OSQPVectori* out = OSQPVectori_malloc(length);
  if(!out) return OSQP_NULL;

  if (length > 0) {
    OSQPVectori_from_raw(out, a);
  }
  return out;
}

OSQPVectorf* OSQPVectorf_malloc(OSQPInt length) {

  OSQPVectorf *b = c_malloc(sizeof(OSQPVectorf));

  if (b) {
    b->length = length;
    if (length) {
      b->values = c_malloc(length * sizeof(OSQPFloat));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectori* OSQPVectori_malloc(OSQPInt length) {

  OSQPVectori *b = c_malloc(sizeof(OSQPVectori));

  if (b) {
    b->length = length;
    if (length) {
      b->values = c_malloc(length * sizeof(OSQPInt));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectorf* OSQPVectorf_calloc(OSQPInt length) {

  OSQPVectorf *b = c_malloc(sizeof(OSQPVectorf));

  if (b) {
    b->length = length;
    if (length) {
      b->values = c_calloc(length, sizeof(OSQPFloat));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectori* OSQPVectori_calloc(OSQPInt length) {

  OSQPVectori *b = c_malloc(sizeof(OSQPVectori));

  if (b) {
    b->length = length;
    if (length) {
      b->values = c_calloc(length, sizeof(OSQPInt));
      if (!(b->values)) {
        c_free(b);
        b = OSQP_NULL;
      }
    }
    else {
      b->values = OSQP_NULL;
    }
  }
  return b;
}

OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf* a) {

  OSQPVectorf* b = OSQPVectorf_malloc(a->length);
  if(b) OSQPVectorf_copy(b,a);
  return b;

}

// OSQPVectori* OSQPVectori_copy_new(const OSQPVectori *a){

//   OSQPVectori* b = OSQPVectori_malloc(a->length);
//   if(b) OSQPVectori_copy(b,a);
//   return b;
// }

void OSQPVectorf_free(OSQPVectorf* a) {
  if (a) {
      c_free(a->values);
  }
  c_free(a);
}

void OSQPVectori_free(OSQPVectori* a) {
  if (a) c_free(a->values);
  c_free(a);
}

void OSQPVectorf_subvector_assign(OSQPVectorf* A,
                                  OSQPFloat*   b,
                                  OSQPInt      start,
                                  OSQPInt      length,
                                  OSQPFloat    multiplier) {
  OSQPInt i;

  for (i = 0; i < length; i++) {
    A->values[start+i] = multiplier * b[i];
  }
}

void OSQPVectorf_subvector_assign_scalar(OSQPVectorf* A,
                                         OSQPFloat    sc,
                                         OSQPInt      start,
                                         OSQPInt      length) {
  OSQPInt i;

  for (i = 0; i < length; i++) {
    A->values[start+i] = sc;
  }
}

OSQPVectorf* OSQPVectorf_subvector_byrows(const OSQPVectorf* A,
                                          const OSQPVectori* rows) {
    OSQPInt i;

    OSQPInt rows_len = 0;
    for (i = 0; i < rows->length; i++)
        if (rows->values[i]) rows_len++;

    OSQPVectorf* out = OSQPVectorf_malloc(rows_len);
    if(!out) return OSQP_NULL;

    OSQPInt j = 0;
    for (i = 0; i < rows->length; i++) {
        if (rows->values[i]) {
            out->values[j] = A->values[i];
            j++;
        }
    }

    return out;
}

OSQPVectorf* OSQPVectorf_concat(const OSQPVectorf* A,
                                const OSQPVectorf* B) {

    OSQPVectorf* out = OSQPVectorf_malloc(A->length + B->length);
    if(!out) return OSQP_NULL;

    OSQPInt i, j;
    for (i = 0; i < A->length; i++)
        out->values[i] = A->values[i];
    for (j = 0; j < B->length; j++)
        out->values[j+i] = B->values[j];

    return out;
}

OSQPVectorf* OSQPVectorf_view(const OSQPVectorf* a,
                                    OSQPInt      head,
                                    OSQPInt      length) {

  OSQPVectorf* view = c_malloc(sizeof(OSQPVectorf));
  if (view) {
      OSQPVectorf_view_update(view, a, head, length);
  }
  return view;
}

void OSQPVectorf_view_update(OSQPVectorf*       a,
                             const OSQPVectorf* b,
                             OSQPInt            head,
                             OSQPInt            length) {
    a->length = length;
    a->values = b->values + head;
}

void OSQPVectorf_view_free(OSQPVectorf* a) {
  c_free(a);
}

OSQPFloat OSQPVectorf_norm_2(const OSQPVectorf* v) {
    OSQPInt i;
    OSQPInt length  = v->length;

    OSQPFloat* vv  = v->values;
    OSQPFloat  normval = 0.0;

    for (i = 0; i < length; i++) {
        normval += vv[i] * vv[i];
    }
    return c_sqrt(normval);
}

#endif /* ifndef OSQP_EMBEDDED_MODE */

OSQPInt OSQPVectorf_length(const OSQPVectorf* a) {return a->length;}
OSQPInt OSQPVectori_length(const OSQPVectori *a){return a->length;}

/* Pointer to vector data (floats) */
OSQPFloat* OSQPVectorf_data(const OSQPVectorf* a) {return a->values;}

// OSQPInt*   OSQPVectori_data(const OSQPVectori *a){return a->values;}

void OSQPVectorf_copy(OSQPVectorf*       b,
                      const OSQPVectorf* a) {
  OSQPVectorf_from_raw(b, a->values);
}

// void OSQPVectori_copy(OSQPVectori       *b,
//                       const OSQPVectori *a){
//   OSQPVectori_from_raw(b, a->values);
// }

void OSQPVectorf_from_raw(OSQPVectorf*     b,
                          const OSQPFloat* av) {
  OSQPInt    i;
  OSQPInt    length = b->length;
  OSQPFloat* bv  = b->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectori_from_raw(OSQPVectori*   b,
                          const OSQPInt* av) {
  OSQPInt  i;
  OSQPInt  length = b->length;
  OSQPInt* bv = b->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectorf_to_raw(OSQPFloat*         bv,
                        const OSQPVectorf* a) {
  OSQPInt    i;
  OSQPInt    length = a->length;
  OSQPFloat* av = a->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectori_to_raw(OSQPInt*           bv,
                        const OSQPVectori* a) {
  OSQPInt  i;
  OSQPInt  length = a->length;
  OSQPInt* av = a->values;

  for (i = 0; i < length; i++) {
    bv[i] = av[i];
  }
}

void OSQPVectorf_set_scalar(OSQPVectorf* a,
                            OSQPFloat    sc) {
  OSQPInt    i;
  OSQPInt    length = a->length;
  OSQPFloat* av  = a->values;

  for (i = 0; i < length; i++) {
    av[i] = sc;
  }
}


void OSQPVectorf_set_scalar_conditional(OSQPVectorf*       a,
                                        const OSQPVectori* test,
                                        OSQPFloat          sc_if_neg,
                                        OSQPFloat          sc_if_zero,
                                        OSQPFloat          sc_if_pos) {
  OSQPInt    i;
  OSQPInt    length = a->length;
  OSQPFloat* av     = a->values;
  OSQPInt*   testv  = test->values;

  for (i = 0; i < length; i++) {
      if (testv[i] == 0)      av[i] = sc_if_zero;
      else if (testv[i] > 0)  av[i] = sc_if_pos;
      else                    av[i] = sc_if_neg;
  }
}


void OSQPVectorf_round_to_zero(OSQPVectorf* a,
                               OSQPFloat    tol) {
  OSQPInt    i;
  OSQPInt    length = a->length;
  OSQPFloat* av     = a->values;

  for(i=0; i < length; i++) {
    if(c_absval(av[i]) < tol) {
      av[i] = (OSQPFloat)0.0;
    }
  }
}


void OSQPVectorf_mult_scalar(OSQPVectorf* a,
                             OSQPFloat    sc) {
  OSQPInt    i;
  OSQPInt    length = a->length;
  OSQPFloat* av = a->values;

  for (i = 0; i < length; i++) {
    av[i] *= sc;
  }
}

void OSQPVectorf_plus(OSQPVectorf*      x,
                     const OSQPVectorf* a,
                     const OSQPVectorf* b) {
  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat* xv = x->values;

  if (x == a){
    for (i = 0; i < length; i++) {
      xv[i] += bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] = av[i] + bv[i];
    }
  }
}

void OSQPVectorf_minus(OSQPVectorf*       x,
                       const OSQPVectorf* a,
                       const OSQPVectorf* b) {
  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat* xv = x->values;

  if (x == a) {
    for (i = 0; i < length; i++) {
      xv[i] -= bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] = av[i] - bv[i];
    }
  }
}

void OSQPVectorf_add_scaled(OSQPVectorf*       x,
                            OSQPFloat          sca,
                            const OSQPVectorf* a,
                            OSQPFloat          scb,
                            const OSQPVectorf* b) {
  OSQPInt i;
  OSQPInt length = x->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat* xv = x->values;

  /* shorter version when incrementing */
  if (x == a && sca == 1.){
    for (i = 0; i < length; i++) {
      xv[i] += scb * bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] = sca * av[i] + scb * bv[i];
    }
  }
}

void OSQPVectorf_add_scaled3(OSQPVectorf*       x,
                             OSQPFloat          sca,
                             const OSQPVectorf* a,
                             OSQPFloat          scb,
                             const OSQPVectorf* b,
                             OSQPFloat          scc,
                             const OSQPVectorf* c) {
  OSQPInt i;
  OSQPInt length = x->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat* cv = c->values;
  OSQPFloat* xv = x->values;

  /* shorter version when incrementing */
  if (x == a && sca == 1.){
    for (i = 0; i < length; i++) {
      xv[i] += scb * bv[i] + scc * cv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      xv[i] =  sca * av[i] + scb * bv[i] + scc * cv[i];
    }
  }
}


OSQPFloat OSQPVectorf_norm_inf(const OSQPVectorf* v) {

  OSQPInt i;
  OSQPInt length  = v->length;

  OSQPFloat  absval;
  OSQPFloat  normval = 0.0;
  OSQPFloat* vv      = v->values;

  for (i = 0; i < length; i++) {
    absval = c_absval(vv[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

// OSQPFloat OSQPVectorf_norm_1(const OSQPVectorf *v){

//   OSQPInt   i;
//   OSQPInt length  = v->length;
//   OSQPFloat*  vv  = v->values;
//   OSQPFloat normval = 0.0;

//   for (i = 0; i < length; i++) {
//     normval += c_absval(vv[i]);
//   }
//   return normval;
// }

OSQPFloat OSQPVectorf_scaled_norm_inf(const OSQPVectorf* S,
                                      const OSQPVectorf* v) {

  OSQPInt i;
  OSQPInt length = v->length;

  OSQPFloat* vv  = v->values;
  OSQPFloat* Sv  = S->values;
  OSQPFloat  absval;
  OSQPFloat  normval = 0.0;

  for (i = 0; i < length; i++) {
    absval = c_absval(Sv[i] * vv[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

// OSQPFloat OSQPVectorf_scaled_norm_1(const OSQPVectorf *S, const OSQPVectorf *v){

//   OSQPInt   i;
//   OSQPInt length  = v->length;
//   OSQPFloat*  vv  = v->values;
//   OSQPFloat*  Sv  = S->values;
//   OSQPFloat normval = 0.0;

//   for (i = 0; i < length; i++) {
//     normval += c_absval(Sv[i] * vv[i]);
//   }
//   return normval;
// }

OSQPFloat OSQPVectorf_norm_inf_diff(const OSQPVectorf* a,
                                    const OSQPVectorf* b) {
  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av   = a->values;
  OSQPFloat* bv   = b->values;
  OSQPFloat  absval;
  OSQPFloat  normDiff = 0.0;

  for (i = 0; i < length; i++) {
    absval = c_absval(av[i] - bv[i]);
    if (absval > normDiff) normDiff = absval;
  }
  return normDiff;
}

// OSQPFloat OSQPVectorf_norm_1_diff(const OSQPVectorf *a,
//                                 const OSQPVectorf *b){

//   OSQPInt   i;
//   OSQPInt   length = a->length;
//   OSQPFloat*  av   = a->values;
//   OSQPFloat*  bv   = b->values;
//   OSQPFloat normDiff = 0.0;

//   for (i = 0; i < length; i++) {
//     normDiff += c_absval(av[i] - bv[i]);
//   }
//   return normDiff;
// }

// OSQPFloat OSQPVectorf_sum(const OSQPVectorf *a){

//   OSQPInt   i;
//   OSQPInt   length = a->length;
//   OSQPFloat*  av   = a->values;
//   OSQPFloat val = 0.0;

//   for (i = 0; i < length; i++) {
//     val += av[i];
//   }

//   return val;
// }

OSQPFloat OSQPVectorf_dot_prod(const OSQPVectorf* a,
                               const OSQPVectorf* b) {

  OSQPInt   i;
  OSQPInt   length = a->length;

  OSQPFloat* av   = a->values;
  OSQPFloat* bv   = b->values;
  OSQPFloat dotprod = 0.0;

  for (i = 0; i < length; i++) {
    dotprod += av[i] * bv[i];
  }
  return dotprod;
}

OSQPFloat OSQPVectorf_dot_prod_signed(const OSQPVectorf* a,
                                      const OSQPVectorf* b,
                                            OSQPInt      sign) {

  OSQPInt   i;
  OSQPInt   length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat  dotprod = 0.0;

  if (sign == 1) {  /* dot with positive part of b */
    for (i = 0; i < length; i++) {
      dotprod += av[i] * c_max(bv[i], 0.);
    }
  }
  else if (sign == -1){  /* dot with negative part of b */
    for (i = 0; i < length; i++) {
      dotprod += av[i] * c_min(bv[i],0.);
    }
  }
  else{
    /* return the conventional dot product */
    dotprod = OSQPVectorf_dot_prod(a, b);
  }
  return dotprod;
}

void OSQPVectorf_ew_prod(OSQPVectorf*       c,
                         const OSQPVectorf* a,
                         const OSQPVectorf* b) {

  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat* cv = c->values;


  if (c == a) {
    for (i = 0; i < length; i++) {
      cv[i] *= bv[i];
    }
  }
  else {
    for (i = 0; i < length; i++) {
      cv[i] = av[i] * bv[i];
    }
  }
}

OSQPInt OSQPVectorf_all_leq(const OSQPVectorf* l,
                            const OSQPVectorf* u) {

  OSQPInt i;
  OSQPInt length = l->length;

  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  for (i = 0; i < length; i++) {
    if (lv[i] > uv[i]) return 0;
  }
  return 1;
}

void OSQPVectorf_ew_bound_vec(OSQPVectorf*       x,
                              const OSQPVectorf* z,
                              const OSQPVectorf* l,
                              const OSQPVectorf* u) {

  OSQPInt i;
  OSQPInt length = x->length;

  OSQPFloat* xv = x->values;
  OSQPFloat* zv = z->values;
  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  for (i = 0; i < length; i++) {
    xv[i] = c_min(c_max(zv[i], lv[i]), uv[i]);
  }
}

void OSQPVectorf_project_polar_reccone(OSQPVectorf*       y,
                                       const OSQPVectorf* l,
                                       const OSQPVectorf* u,
                                       OSQPFloat          infval) {

  OSQPInt i; // Index for loops
  OSQPInt length = y->length;

  OSQPFloat* yv = y->values;
  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  for (i = 0; i < length; i++) {
    if (uv[i]   > +infval) {       // Infinite upper bound
      if (lv[i] < -infval) {       // Infinite lower bound
        // Both bounds infinite
        yv[i] = 0.0;
      } else {
        // Only upper bound infinite
        yv[i] = c_min(yv[i], 0.0);
      }
    } else if (lv[i] < -infval) {  // Infinite lower bound
      // Only lower bound infinite
      yv[i] = c_max(yv[i], 0.0);
    }
  }
}

OSQPInt OSQPVectorf_in_reccone(const OSQPVectorf* y,
                               const OSQPVectorf* l,
                               const OSQPVectorf* u,
                                     OSQPFloat    infval,
                                     OSQPFloat    tol){

  OSQPInt i; // Index for loops
  OSQPInt length = y->length;

  OSQPFloat* yv = y->values;
  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  for (i = 0; i < length; i++) {
    if (((uv[i] < +infval) &&
         (yv[i] > +tol)) ||
        ((lv[i] > -infval) &&
         (yv[i] < -tol))) {
      // At least one condition not satisfied -> not dual infeasible
      return 0;
    }
  }
  return 1;
}


// void OSQPVectorf_permute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

//   OSQPInt j;
//   OSQPInt length = x->length;
//   OSQPFloat*  xv = x->values;
//   OSQPFloat*  bv = b->values;
//   OSQPInt*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[j] = bv[pv[j]];
//   }
// }

// void OSQPVectori_permute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

//   OSQPInt j;
//   OSQPInt length = x->length;
//   OSQPInt*    xv = x->values;
//   OSQPInt*    bv = b->values;
//   OSQPInt*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[j] = bv[pv[j]];
//   }
// }

// void OSQPVectorf_ipermute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p){

//   OSQPInt j;
//   OSQPInt length = x->length;
//   OSQPFloat*  xv = x->values;
//   OSQPFloat*  bv = b->values;
//   OSQPInt*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[pv[j]] = bv[j];
//   }
// }

// void OSQPVectori_ipermute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p){

//   OSQPInt j;
//   OSQPInt length = x->length;
//   OSQPInt*    xv = x->values;
//   OSQPInt*    bv = b->values;
//   OSQPInt*    pv = p->values;

//   for (j = 0; j < length; j++) {
//     xv[pv[j]] = bv[j];
//   }
// }



#if OSQP_EMBEDDED_MODE != 1

OSQPFloat OSQPVectorf_norm_1(const OSQPVectorf* a) {

  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av  = a->values;
  OSQPFloat  val = 0.0;

  if (length) {
    for (i = 0; i < length; i++) {
      val += c_absval(av[i]);
    }
  }

  return val;
}

void OSQPVectorf_ew_reciprocal(OSQPVectorf*       b,
                               const OSQPVectorf* a) {

  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;

  for (i = 0; i < length; i++) {
    bv[i] = (OSQPFloat)1.0 / av[i];
  }
}

void OSQPVectorf_ew_sqrt(OSQPVectorf* a){

  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;

  for (i = 0; i < length; i++) {
    av[i] = c_sqrt(av[i]);
  }
}

// void OSQPVectorf_ew_max(OSQPVectorf       *c,
//                         const OSQPVectorf *a,
//                         OSQPFloat            max_val){

//   OSQPInt i;
//   OSQPInt length = c->length;
//   OSQPFloat*  av = a->values;
//   OSQPFloat*  cv = c->values;

//   for (i = 0; i < length; i++) {
//     cv[i] = c_max(av[i], max_val);
//   }
// }

// void OSQPVectorf_ew_min(OSQPVectorf *c, const OSQPVectorf *a, OSQPFloat min_val){

//   OSQPInt i;
//   OSQPInt length = a->length;
//   OSQPFloat*  av = a->values;
//   OSQPFloat*  cv = c->values;

//   for (i = 0; i < length; i++) {
//     cv[i] = c_min(av[i], min_val);
//   }
// }

void OSQPVectorf_ew_max_vec(OSQPVectorf*       c,
                            const OSQPVectorf* a,
                            const OSQPVectorf* b) {
  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat* cv = c->values;

  for (i = 0; i < length; i++) {
    cv[i] = c_max(av[i], bv[i]);
  }
}

void OSQPVectorf_ew_min_vec(OSQPVectorf*        c,
                             const OSQPVectorf* a,
                             const OSQPVectorf* b) {
  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat* cv = c->values;

  for (i = 0; i < length; i++) {
    cv[i] = c_min(av[i], bv[i]);
  }
}

OSQPInt OSQPVectorf_ew_bounds_type(OSQPVectori*       iseq,
                                   const OSQPVectorf* l,
                                   const OSQPVectorf* u,
                                   OSQPFloat          tol,
                                   OSQPFloat          infval) {

  OSQPInt  i;
  OSQPInt  old_value;
  OSQPInt  has_changed = 0;
  OSQPInt  length = iseq->length;
  OSQPInt* iseqv  = iseq->values;

  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  for (i = 0; i < length; i++) {

    old_value = iseqv[i];

    if ((lv[i] < -infval) && (uv[i] > infval)) {
      // Loose bounds
      iseqv[i] = -1;
    } else if (uv[i] - lv[i] < tol) {
      // Equality constraints
      iseqv[i] = 1;
    } else {
      // Inequality constraints
      iseqv[i] = 0;
    }

    //has anything changed?
    has_changed = has_changed || (iseqv[i] != old_value);
  }

  return has_changed;

}

void OSQPVectorf_set_scalar_if_lt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval) {
  OSQPInt i;
  OSQPInt length = x->length;

  OSQPFloat* xv = x->values;
  OSQPFloat* zv = z->values;

  for (i = 0; i < length; i++) {
    xv[i] = zv[i] < testval ? newval : zv[i];
  }
}

void OSQPVectorf_set_scalar_if_gt(OSQPVectorf*       x,
                                  const OSQPVectorf* z,
                                  OSQPFloat          testval,
                                  OSQPFloat          newval) {
  OSQPInt i;
  OSQPInt length = x->length;

  OSQPFloat* xv = x->values;
  OSQPFloat* zv = z->values;

  for (i = 0; i < length; i++) {
    xv[i] = zv[i] > testval ? newval : zv[i];
  }
}

#endif /* OSQP_EMBEDDED_MODE != 1 */


#ifdef OSQP_USE_FLEXFLOAT
void OSQPVectorf_round_to_zero_FF(OSQPVectorf* a,
                               OSQPFloat    tol) {
  OSQPInt    i;
  OSQPInt    length = a->length;
  OSQPFloat* av     = a->values;

  flexfloat_t ff_avi_abs, ff_avi, ff_tol, ff_zero;
  ff_init_float(&ff_zero, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_avi_abs, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  for(i=0; i < length; i++) {
    ff_init_float(&ff_avi, av[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_tol, tol, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

    // Take absolute value using FlexFloat comparison: abs_val = (avi < 0) ? -avi : avi
    if (ff_lt(&ff_avi, &ff_zero)) {
      // Negate by subtracting from zero: -avi = 0 - avi
      ff_sub(&ff_avi_abs, &ff_zero, &ff_avi);
    } else {
      ff_avi_abs = ff_avi;
    }

    if(ff_lt(&ff_avi_abs, &ff_tol)) {
      av[i] = ff_get_float(&ff_zero);
    }
  }
}

void OSQPVectorf_mult_scalar_FF(OSQPVectorf *a,
                             OSQPFloat    sc) {
  OSQPInt i;
  OSQPInt length = a->length;
  OSQPFloat* av = a->values;

  flexfloat_t ff_av, ff_sc;

  ff_init_float(&ff_sc, sc, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  for (i = 0; i < length; i++) {
    ff_init_float(&ff_av, av[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_mul(&ff_av, &ff_sc, &ff_av);
    av[i] = ff_get_float(&ff_av);
  }
}

void OSQPVectorf_plus_FF(OSQPVectorf*      x,
                     const OSQPVectorf* a,
                     const OSQPVectorf* b) {

  // Compute x = a + b. If x == a, compute x += b.
  OSQPInt i;

  flexfloat_t ff_a, ff_b, ff_x;

  for (i = 0; i < a->length; i++) {
    ff_init_float(&ff_a, a->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_b, b->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_x, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

    ff_add(&ff_x, &ff_a, &ff_b);
    x->values[i] = ff_get_float(&ff_x);
  }
}

void OSQPVectorf_minus_FF(OSQPVectorf*       x,
                       const OSQPVectorf* a,
                       const OSQPVectorf* b) {

  // Compute x = a - b. If x == a, compute x -= b.
  OSQPInt i;

  flexfloat_t ff_a, ff_b, ff_x;

  for (i = 0; i < a->length; i++) {
    ff_init_float(&ff_a, a->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_b, b->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_x, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

    ff_sub(&ff_x, &ff_a, &ff_b);
    x->values[i] = ff_get_float(&ff_x);
  }
}

void OSQPVectorf_add_scaled_FF(OSQPVectorf*       x,
                            OSQPFloat          sca,
                            const OSQPVectorf* a,
                            OSQPFloat          scb,
                            const OSQPVectorf* b) {

  /* shorter version when incrementing */
  if (x == a && sca == 1.){
    OSQPInt i;
    flexfloat_t ff_scb, ff_bv, ff_xv, ff_tmp;
    ff_init_float(&ff_scb, scb, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_tmp, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    for (i = 0; i < x->length; i++) {
      ff_init_float(&ff_bv, b->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_xv, x->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_mul(&ff_tmp, &ff_scb, &ff_bv);
      ff_add(&ff_xv, &ff_xv, &ff_tmp);
      x->values[i] = ff_get_float(&ff_xv);
    }
  }
  else {
    OSQPInt i;
    OSQPInt length = x->length;

    flexfloat_t ff_av, ff_bv, ff_xv, ff_sca, ff_scb, ff_tmp1, ff_tmp2;

    ff_init_float(&ff_sca, sca, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_scb, scb, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_tmp1, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_tmp2, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

    OSQPFloat*  av = a->values;
    OSQPFloat*  bv = b->values;
    OSQPFloat*  xv = x->values;

    for (i = 0; i < length; i++) {
      ff_init_float(&ff_av, av[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_bv, bv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_xv, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

      // xv[i] = sca * av[i] + scb * bv[i];

      ff_mul(&ff_tmp1, &ff_sca, &ff_av);
      ff_mul(&ff_tmp2, &ff_scb, &ff_bv);
      ff_add(&ff_xv, &ff_tmp1, &ff_tmp2);

      xv[i] = ff_get_float(&ff_xv);
    }
  }

}

void OSQPVectorf_add_scaled3_FF(OSQPVectorf*       x,
                             OSQPFloat          sca,
                             const OSQPVectorf* a,
                             OSQPFloat          scb,
                             const OSQPVectorf* b,
                             OSQPFloat          scc,
                             const OSQPVectorf* c) {
  OSQPInt i;
  OSQPInt length = x->length;

  OSQPFloat*  av = a->values;
  OSQPFloat*  bv = b->values;
  OSQPFloat*  cv = c->values;
  OSQPFloat*  xv = x->values;

  flexfloat_t ff_av, ff_bv, ff_cv, ff_xv;
  flexfloat_t ff_sca, ff_scb, ff_scc;
  flexfloat_t ff_tmp1, ff_tmp2, ff_tmp3;

  ff_init_float(&ff_sca, sca, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_scb, scb, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_scc, scc, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_tmp1, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_tmp2, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_tmp3, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  /* shorter version when incrementing */
  if (x == a && sca == 1.){
    for (i = 0; i < length; i++) {
      ff_init_float(&ff_bv, bv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_cv, cv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_xv, xv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      
      // xv[i] += scb * bv[i] + scc * cv[i];
      ff_mul(&ff_tmp1, &ff_scb, &ff_bv);
      ff_mul(&ff_tmp2, &ff_scc, &ff_cv);
      ff_add(&ff_tmp3, &ff_tmp1, &ff_tmp2);
      ff_add(&ff_xv, &ff_xv, &ff_tmp3);

      xv[i] = ff_get_float(&ff_xv);
    }
  }
  else {
    for (i = 0; i < length; i++) {
      ff_init_float(&ff_av, av[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_bv, bv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_cv, cv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_xv, xv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

      // xv[i] =  sca * av[i] + scb * bv[i] + scc * cv[i];
      ff_mul(&ff_tmp1, &ff_sca, &ff_av);
      ff_mul(&ff_tmp2, &ff_scb, &ff_bv);
      ff_mul(&ff_tmp3, &ff_scc, &ff_cv);
      ff_add(&ff_tmp1, &ff_tmp1, &ff_tmp2);
      ff_add(&ff_xv, &ff_tmp1, &ff_tmp3);
      
      xv[i] = ff_get_float(&ff_xv);
    }
  }
}

OSQPFloat OSQPVectorf_norm_inf_FF(const OSQPVectorf* v) {

  OSQPFloat val = 0.0;
  OSQPInt i;

  flexfloat_t ff_val, ff_absval, ff_vi, ff_zero;

  if (v->length){
    // Initialize zero for comparison
    ff_init_float(&ff_zero, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_val, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_absval, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

    // Find the maximum absolute value
    for (i = 0; i < v->length; i++) {
      // Convert v->values[i] to FlexFloat first
      ff_init_float(&ff_vi, v->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

      // Take absolute value using FlexFloat comparison: abs_val = (vi < 0) ? -vi : vi
      if (ff_lt(&ff_vi, &ff_zero)) {
        // Negate by subtracting from zero: -vi = 0 - vi
        ff_sub(&ff_absval, &ff_zero, &ff_vi);
      } else {
        ff_absval = ff_vi;
      }

      // Update max if abs_val > val
      if (ff_gt(&ff_absval, &ff_val)) {
        ff_val = ff_absval;
      }
    }
  }

  return ff_get_float(&ff_val);
}

OSQPFloat OSQPVectorf_scaled_norm_inf_FF(const OSQPVectorf* S,
                                      const OSQPVectorf* v) {

  OSQPInt i;
  OSQPInt length  = v->length;

  OSQPFloat* vv  = v->values;
  OSQPFloat* Sv  = S->values;
  OSQPFloat  absval;
  OSQPFloat  normval = 0.0;

  flexfloat_t ff_absval, ff_normval, ff_tmp, ff_Svi, ff_vvi, ff_zero;

  ff_init_float(&ff_normval, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_tmp, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_zero, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_absval, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  // for (i = 0; i < length; i++) {
  //   absval = c_absval(Sv[i] * vv[i]);
  //   if (absval > normval) normval = absval;
  // }
  // return normval;

  for (i = 0; i < length; i++) {
    ff_init_float(&ff_Svi, Sv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_vvi, vv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_mul(&ff_tmp, &ff_Svi, &ff_vvi);

    // Take absolute value using FlexFloat comparison: abs_val = (tmp < 0) ? -tmp : tmp
    if (ff_lt(&ff_tmp, &ff_zero)) {
      // Negate by subtracting from zero: -tmp = 0 - tmp
      ff_sub(&ff_absval, &ff_zero, &ff_tmp);
    } else {
      ff_absval = ff_tmp;
    }

    if (ff_gt(&ff_absval, &ff_normval)) {
      ff_normval = ff_absval;
    }
  }
  return ff_get_float(&ff_normval);
}

OSQPFloat OSQPVectorf_dot_prod_FF(const OSQPVectorf* a,
                               const OSQPVectorf* b) {

  OSQPInt i;
  OSQPFloat dotprod = 0.0;

  flexfloat_t ff_a, ff_b, ff_dotprod, ff_tmp;

  ff_init_float(&ff_dotprod, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  for (i = 0; i < a->length; i++) {
    ff_init_float(&ff_a, a->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_b, b->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_tmp, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_mul(&ff_tmp, &ff_a, &ff_b);
    ff_add(&ff_dotprod, &ff_dotprod, &ff_tmp);
  }

  return ff_get_float(&ff_dotprod);
}

OSQPFloat OSQPVectorf_dot_prod_signed_FF(const OSQPVectorf* a,
                                      const OSQPVectorf* b,
                                      OSQPInt            sign) {

  OSQPInt i;
  OSQPInt length = a->length;

  OSQPFloat* av = a->values;
  OSQPFloat* bv = b->values;
  OSQPFloat  dotprod = 0.0;

  flexfloat_t ff_avi, ff_bvi, ff_tmp, ff_dotprod, ff_zero;
  ff_init_float(&ff_dotprod, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_zero, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_tmp, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  if (sign == 1) {  /* dot with positive part of b */
    for (i = 0; i < length; i++) {
      ff_init_float(&ff_avi, av[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_bvi, bv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      // dotprod += av[i] * c_max(bv[i], 0.);
      ff_max(&ff_bvi, &ff_bvi, &ff_zero);
      ff_mul(&ff_tmp, &ff_avi, &ff_bvi);
      ff_add(&ff_dotprod, &ff_dotprod, &ff_tmp);
      dotprod = ff_get_float(&ff_dotprod);
    }
  }
  else if (sign == -1){  /* dot with negative part of b */
    for (i = 0; i < length; i++) {
      ff_init_float(&ff_avi, av[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      ff_init_float(&ff_bvi, bv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
      // dotprod += av[i] * c_min(bv[i],0.);
      ff_min(&ff_bvi, &ff_bvi, &ff_zero);
      ff_mul(&ff_tmp, &ff_avi, &ff_bvi);
      ff_add(&ff_dotprod, &ff_dotprod, &ff_tmp);
      dotprod = ff_get_float(&ff_dotprod);
    }
  }
  else{
    /* return the conventional dot product */
    dotprod = OSQPVectorf_dot_prod_FF(a, b);
  }
  return dotprod;
}

void OSQPVectorf_ew_prod_FF(OSQPVectorf*       c,
                         const OSQPVectorf* a,
                         const OSQPVectorf* b) {
  OSQPInt i;

  flexfloat_t ff_av, ff_bv, ff_cv;

  for (i = 0; i < a->length; i++) {
    ff_init_float(&ff_av, a->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_bv, b->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_cv, c->values[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    
    ff_mul(&ff_cv, &ff_av, &ff_bv);

    c->values[i] = ff_get_float(&ff_cv);
  }
}

void OSQPVectorf_ew_bound_vec_FF(OSQPVectorf*       x,
                              const OSQPVectorf* z,
                              const OSQPVectorf* l,
                              const OSQPVectorf* u) {

  OSQPInt i;
  OSQPInt length = x->length;

  OSQPFloat* xv = x->values;
  OSQPFloat* zv = z->values;
  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  flexfloat_t ff_zv, ff_lv, ff_uv, ff_xv, ff_tmp;

  ff_init_float(&ff_tmp, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  for (i = 0; i < length; i++) {
    ff_init_float(&ff_zv, zv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_lv, lv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_uv, uv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_xv, xv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

    // xv[i] = c_min(c_max(zv[i], lv[i]), uv[i]);
    ff_max(&ff_tmp, &ff_zv, &ff_lv);
    ff_min(&ff_xv, &ff_tmp, &ff_uv);

    xv[i] = ff_get_float(&ff_xv);
  }
}

void OSQPVectorf_project_polar_reccone_FF(OSQPVectorf*       y,
                                       const OSQPVectorf* l,
                                       const OSQPVectorf* u,
                                       OSQPFloat          infval) {

  OSQPInt i; // Index for loops
  OSQPInt length = y->length;

  OSQPFloat* yv = y->values;
  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  flexfloat_t ff_uvi, ff_lvi, ff_yvi, ff_zero;
  flexfloat_t ff_lv, ff_uv, ff_infval_plus, ff_infval_minus;

  ff_init_float(&ff_zero, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_infval_plus, infval, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_infval_minus, -infval, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  for (i = 0; i < length; i++) {
    ff_init_float(&ff_uvi, uv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_lvi, lv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_yvi, yv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    
    if (ff_gt(&ff_uvi, &ff_infval_plus)) {       // Infinite upper bound
      if (ff_lt(&ff_lvi, &ff_infval_minus)) {       // Infinite lower bound
        // Both bounds infinite
        // yv[i] = 0.0;

        yv[i] = ff_get_float(&ff_zero);
      } else {
        // Only upper bound infinite
        // yv[i] = c_min(yv[i], 0.0);
        ff_min(&ff_yvi, &ff_yvi, &ff_zero);
        yv[i] = ff_get_float(&ff_yvi);
      }
    } else if (ff_lt(&ff_lvi, &ff_infval_minus)) {  // Infinite lower bound
      // Only lower bound infinite
      // yv[i] = c_max(yv[i], 0.0);
      ff_max(&ff_yvi, &ff_yvi, &ff_zero);
      yv[i] = ff_get_float(&ff_yvi);
    }
  }
}

OSQPFloat OSQPScalarf_prod_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b, ff_c;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_c, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  ff_mul(&ff_c, &ff_a, &ff_b);

  return ff_get_float(&ff_c);
}

OSQPFloat OSQPScalarf_div_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b, ff_c;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_c, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  ff_div(&ff_c, &ff_a, &ff_b);

  return ff_get_float(&ff_c);
}

OSQPFloat OSQPScalarf_add_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b, ff_c;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_c, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  ff_add(&ff_c, &ff_a, &ff_b);

  return ff_get_float(&ff_c);
}

OSQPFloat OSQPScalarf_minus_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b, ff_c;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_c, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  ff_sub(&ff_c, &ff_a, &ff_b);

  return ff_get_float(&ff_c);
}

OSQPFloat OSQPScalarf_max_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b, ff_c;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_c, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  ff_max(&ff_c, &ff_a, &ff_b);

  return ff_get_float(&ff_c);
}

OSQPFloat OSQPScalarf_min_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b, ff_c;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_c, 0.0, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  ff_min(&ff_c, &ff_a, &ff_b);

  return ff_get_float(&ff_c);
}

bool OSQPScalarf_gt_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  return ff_gt(&ff_a, &ff_b);
}

bool OSQPScalarf_ge_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  return ff_ge(&ff_a, &ff_b);
}

bool OSQPScalarf_lt_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  return ff_lt(&ff_a, &ff_b);
}

bool OSQPScalarf_le_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  return ff_le(&ff_a, &ff_b);
}

bool OSQPScalarf_eq_FF(
                         OSQPFloat a,
                         OSQPFloat b) {

  flexfloat_t ff_a, ff_b;

  ff_init_float(&ff_a, a, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_b, b, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  return ff_eq(&ff_a, &ff_b);
}

OSQPInt OSQPVectorf_in_reccone_FF(const OSQPVectorf* y,
                                  const OSQPVectorf* l,
                                  const OSQPVectorf* u,
                                        OSQPFloat    infval,
                                        OSQPFloat    tol){

  OSQPInt i;
  OSQPInt length = y->length;

  OSQPFloat* yv = y->values;
  OSQPFloat* lv = l->values;
  OSQPFloat* uv = u->values;

  flexfloat_t ff_uvi, ff_infval, ff_yvi, ff_tol, ff_lvi, ff_neg_infval, ff_neg_tol;

  ff_init_float(&ff_infval, infval, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_tol, tol, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_neg_infval, -infval, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
  ff_init_float(&ff_neg_tol, -tol, (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

  for (i = 0; i < length; i++) {
    ff_init_float(&ff_uvi, uv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_yvi, yv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});
    ff_init_float(&ff_lvi, lv[i], (flexfloat_desc_t) {FF_exponent_bits, FF_mantissa_bits});

    if ((ff_lt(&ff_uvi, &ff_infval) && ff_gt(&ff_yvi, &ff_tol)) ||
        (ff_gt(&ff_lvi, &ff_neg_infval) && ff_lt(&ff_yvi, &ff_neg_tol))) {
      return 0;
    }
  }
  return 1;
}

#endif /* OSQP_USE_FLEXFLOAT */


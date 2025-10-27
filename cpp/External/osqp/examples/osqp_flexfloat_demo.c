/*
 * OSQP FlexFloat Demo
 *
 * Simple QP problem to test FlexFloat integration:
 * minimize    0.5 * x'*P*x + q'*x
 * subject to  l <= Ax <= u
 *
 * where P = [4  1]    q = [1]    A = [1  1]    l = [1]    u = [1]
 *           [1  2]        [1]        [1  0]        [0]        [0.7]
 */

#include "osqp.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {

  printf("=================================================\n");
  printf("OSQP FlexFloat Demo\n");
#ifdef OSQP_USE_FLEXFLOAT
  printf("Running with FlexFloat (14-bit mantissa, 8-bit exponent)\n");
#else
  printf("Running with standard floating-point\n");
#endif
  printf("=================================================\n\n");

  /* Problem data */
  OSQPInt n = 2;  /* Number of variables */
  OSQPInt m = 2;  /* Number of constraints */

  /* Objective function: P and q */
  OSQPInt    P_nnz = 3;
  OSQPInt    P_i[] = {0, 0, 1};      /* Row indices */
  OSQPInt    P_p[] = {0, 2, 3};      /* Column pointers */
  OSQPFloat  P_x[] = {4.0, 1.0, 2.0}; /* Values */

  OSQPFloat q[] = {1.0, 1.0};

  /* Constraints: A, l, u */
  OSQPInt    A_nnz = 3;
  OSQPInt    A_i[] = {0, 1, 0};       /* Row indices */
  OSQPInt    A_p[] = {0, 2, 3};       /* Column pointers */
  OSQPFloat  A_x[] = {1.0, 1.0, 1.0}; /* Values */

  OSQPFloat l[] = {1.0, 0.0};
  OSQPFloat u[] = {1.0, 0.7};

  /* Create CSC matrices */
  OSQPCscMatrix P, A;
  OSQPCscMatrix_set_data(&P, n, n, P_nnz, P_x, P_i, P_p);
  OSQPCscMatrix_set_data(&A, m, n, A_nnz, A_x, A_i, A_p);

  /* Settings */
  OSQPSettings* settings = OSQPSettings_new();
  if (!settings) {
    printf("Error: Failed to allocate settings\n");
    return 1;
  }
  osqp_set_default_settings(settings);
  settings->verbose = 1;
  settings->max_iter = 100;
  settings->eps_abs = 1e-4;
  settings->eps_rel = 1e-4;

  /* Setup solver */
  OSQPSolver* solver = NULL;
  OSQPInt exitflag = osqp_setup(&solver, &P, q, &A, l, u, m, n, settings);

  if (exitflag) {
    printf("Error: osqp_setup failed with code %d\n", (int)exitflag);
    OSQPSettings_free(settings);
    return 1;
  }

  printf("\nSolver setup complete. Starting optimization...\n\n");

  /* Solve problem */
  exitflag = osqp_solve(solver);

  if (exitflag) {
    printf("Error: osqp_solve failed with code %d\n", (int)exitflag);
    osqp_cleanup(solver);
    OSQPSettings_free(settings);
    return 1;
  }

  /* Print results */
  printf("\n=================================================\n");
  printf("Results\n");
  printf("=================================================\n");
  printf("Status:              %s\n", solver->info->status);
  printf("Iterations:          %d\n", (int)solver->info->iter);
  printf("Objective value:     %.8f\n", solver->info->obj_val);
  printf("Primal residual:     %.8e\n", solver->info->prim_res);
  printf("Dual residual:       %.8e\n", solver->info->dual_res);
  printf("Solve time:          %.6f seconds\n", solver->info->solve_time);
  printf("\nOptimal solution:\n");
  printf("  x[0] = %.8f\n", solver->solution->x[0]);
  printf("  x[1] = %.8f\n", solver->solution->x[1]);
  printf("=================================================\n");

#ifdef OSQP_USE_FLEXFLOAT
  printf("\nNote: All computations were performed with FlexFloat quantization.\n");
  printf("Mantissa: 14 bits, Exponent: 8 bits\n");
  printf("Expected precision: ~4-5 decimal digits\n");
#endif

  /* Cleanup */
  osqp_cleanup(solver);
  OSQPSettings_free(settings);

  return 0;
}

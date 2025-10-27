/*
 * Simple FlexFloat test for OSQP
 * Tests a basic QP problem with FlexFloat enabled
 */

#include "osqp.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=================================================\n");
    printf("OSQP FlexFloat Test\n");
#ifdef OSQP_USE_FLEXFLOAT
    printf("FlexFloat: ENABLED (14-bit mantissa, 8-bit exponent)\n");
#else
    printf("FlexFloat: DISABLED (standard precision)\n");
#endif
    printf("=================================================\n\n");

    // Simple QP: min 0.5*x'*[4 1; 1 2]*x + [1 1]'*x
    //            s.t. x1 + x2 = 1, x2 >= 0.7

    OSQPInt n = 2;
    OSQPInt m = 2;

    // P matrix (upper triangular in CSC format)
    OSQPFloat P_x[] = {4.0, 1.0, 2.0};
    OSQPInt P_i[] = {0, 0, 1};
    OSQPInt P_p[] = {0, 2, 3};
    OSQPCscMatrix P;
    OSQPCscMatrix_set_data(&P, n, n, 3, P_x, P_i, P_p);

    // q vector
    OSQPFloat q[] = {1.0, 1.0};

    // A matrix
    OSQPFloat A_x[] = {1.0, 1.0, 1.0};
    OSQPInt A_i[] = {0, 1, 0};
    OSQPInt A_p[] = {0, 2, 3};
    OSQPCscMatrix A;
    OSQPCscMatrix_set_data(&A, m, n, 3, A_x, A_i, A_p);

    // Bounds
    OSQPFloat l[] = {1.0, 0.7};
    OSQPFloat u[] = {1.0, OSQP_INFTY};

    // Setup
    OSQPSolver* solver;
    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);
    settings->verbose = 1;
    settings->max_iter = 100;
    settings->eps_abs = 1e-4;
    settings->eps_rel = 1e-4;

    printf("Setting up solver...\n");
    OSQPInt exitflag = osqp_setup(&solver, &P, q, &A, l, u, m, n, settings);

    if (exitflag != 0) {
        printf("Setup failed with code %d\n", (int)exitflag);
        free(settings);
        return 1;
    }

    printf("Setup complete! Solving...\n\n");
    exitflag = osqp_solve(solver);

    if (exitflag != 0) {
        printf("Solve failed with code %d\n", (int)exitflag);
        osqp_cleanup(solver);
        free(settings);
        return 1;
    }

    printf("\n=================================================\n");
    printf("RESULTS\n");
    printf("=================================================\n");
    printf("Status:          %s\n", solver->info->status);
    printf("Iterations:      %d\n", (int)solver->info->iter);
    printf("Objective:       %.8f\n", solver->info->obj_val);
    printf("Primal residual: %.2e\n", solver->info->prim_res);
    printf("Dual residual:   %.2e\n", solver->info->dual_res);
    printf("Solve time:      %.6f sec\n", solver->info->solve_time);
    printf("\nSolution:\n");
    printf("  x[0] = %.8f\n", solver->solution->x[0]);
    printf("  x[1] = %.8f\n", solver->solution->x[1]);
    printf("=================================================\n");

#ifdef OSQP_USE_FLEXFLOAT
    printf("\nNote: All arithmetic operations used FlexFloat quantization.\n");
    printf("      Mantissa: 14 bits (~4-5 decimal digits precision)\n");
    printf("      Exponent: 8 bits (same range as float32)\n");
#endif

    osqp_cleanup(solver);
    free(settings);

    return 0;
}

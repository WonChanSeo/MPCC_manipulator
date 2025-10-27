/*
 * Simple QP Test - Guaranteed to converge
 * Problem: min 0.5*x^2 + 0.5*y^2
 *          s.t. x >= 1, y >= 1
 * Solution: x=1, y=1, obj=1
 */

#include "osqp.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("===========================================\n");
    printf("Simple QP Test - FlexFloat vs Standard\n");
#ifdef OSQP_USE_FLEXFLOAT
    printf("Mode: FlexFloat (14-bit mantissa)\n");
#else
    printf("Mode: Standard (52-bit mantissa)\n");
#endif
    printf("===========================================\n\n");

    // Problem: minimize 0.5*x^2 + 0.5*y^2 subject to x >= 1, y >= 1

    OSQPInt n = 2;  // 2 variables
    OSQPInt m = 2;  // 2 constraints

    // P = [1  0]  (Hessian = I)
    //     [0  1]
    OSQPFloat P_x[] = {1.0, 1.0};
    OSQPInt P_i[] = {0, 1};
    OSQPInt P_p[] = {0, 1, 2};
    OSQPCscMatrix P;
    OSQPCscMatrix_set_data(&P, n, n, 2, P_x, P_i, P_p);

    // q = [0, 0]
    OSQPFloat q[] = {0.0, 0.0};

    // A = [1  0]  (identity for simple bounds)
    //     [0  1]
    OSQPFloat A_x[] = {1.0, 1.0};
    OSQPInt A_i[] = {0, 1};
    OSQPInt A_p[] = {0, 1, 2};
    OSQPCscMatrix A;
    OSQPCscMatrix_set_data(&A, m, n, 2, A_x, A_i, A_p);

    // Bounds: x >= 1, y >= 1
    OSQPFloat l[] = {1.0, 1.0};
    OSQPFloat u[] = {OSQP_INFTY, OSQP_INFTY};

    // Settings
    OSQPSolver* solver;
    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);

    settings->verbose = 1;
    settings->max_iter = 100;
    settings->eps_abs = 1e-3;
    settings->eps_rel = 1e-3;
    settings->polishing = 0;
    settings->scaling = 0;  // Disable scaling for simpler test

    printf("Setting up solver...\n");
    OSQPInt exitflag = osqp_setup(&solver, &P, q, &A, l, u, m, n, settings);

    if (exitflag) {
        printf("Setup failed: %d\n", (int)exitflag);
        free(settings);
        return 1;
    }

    printf("Solving...\n\n");
    exitflag = osqp_solve(solver);

    if (exitflag) {
        printf("Solve failed: %d\n", (int)exitflag);
        osqp_cleanup(solver);
        free(settings);
        return 1;
    }

    // Results
    printf("\n===========================================\n");
    printf("RESULTS\n");
    printf("===========================================\n");
    printf("Status:       %s\n", solver->info->status);
    printf("Iterations:   %d\n", (int)solver->info->iter);
    printf("Solve time:   %.6f s\n", solver->info->solve_time);
    printf("\nObjective:    %.8f (expected: 1.0)\n", solver->info->obj_val);
    printf("Solution:\n");
    printf("  x = %.8f (expected: 1.0)\n", solver->solution->x[0]);
    printf("  y = %.8f (expected: 1.0)\n", solver->solution->x[1]);

    double err_x = solver->solution->x[0] - 1.0;
    double err_y = solver->solution->x[1] - 1.0;
    double err_obj = solver->info->obj_val - 1.0;

    printf("\nErrors:\n");
    printf("  Δx   = %.6e\n", err_x);
    printf("  Δy   = %.6e\n", err_y);
    printf("  Δobj = %.6e\n", err_obj);

    printf("\nResiduals:\n");
    printf("  Primal: %.6e\n", solver->info->prim_res);
    printf("  Dual:   %.6e\n", solver->info->dual_res);

#ifdef OSQP_USE_FLEXFLOAT
    printf("\n===========================================\n");
    printf("FlexFloat was applied to ALL computations!\n");
    printf("Every add/sub/mul in ADMM used 14-bit mantissa\n");
    printf("===========================================\n");
#endif

    osqp_cleanup(solver);
    free(settings);
    return 0;
}

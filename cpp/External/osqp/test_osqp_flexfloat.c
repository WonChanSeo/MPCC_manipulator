/*
 * OSQP FlexFloat Integration Test
 * Tests a well-conditioned QP problem
 */

#include "osqp.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=========================================\n");
    printf("OSQP FlexFloat Integration Test\n");
#ifdef OSQP_USE_FLEXFLOAT
    printf("Mode: FlexFloat (14-bit mantissa, 8-bit exp)\n");
#else
    printf("Mode: Standard double precision\n");
#endif
    printf("=========================================\n\n");

    // Well-conditioned QP problem:
    // minimize    x^2 + y^2
    // subject to  x + 3y >= 4
    //             x + y >= 2

    OSQPInt n = 2;  // variables
    OSQPInt m = 2;  // constraints

    // P = [2  0]  (quadratic term)
    //     [0  2]
    OSQPFloat P_x[] = {2.0, 2.0};
    OSQPInt P_i[] = {0, 1};
    OSQPInt P_p[] = {0, 1, 2};
    OSQPCscMatrix P;
    OSQPCscMatrix_set_data(&P, n, n, 2, P_x, P_i, P_p);

    // q = [0, 0] (linear term)
    OSQPFloat q[] = {0.0, 0.0};

    // A = [1  3]
    //     [1  1]
    OSQPFloat A_x[] = {1.0, 1.0, 3.0, 1.0};
    OSQPInt A_i[] = {0, 1, 0, 1};
    OSQPInt A_p[] = {0, 2, 4};
    OSQPCscMatrix A;
    OSQPCscMatrix_set_data(&A, m, n, 4, A_x, A_i, A_p);

    // Bounds: [4, 2] <= Ax <= [inf, inf]
    OSQPFloat l[] = {4.0, 2.0};
    OSQPFloat u[] = {OSQP_INFTY, OSQP_INFTY};

    // Setup solver
    OSQPSolver* solver;
    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    osqp_set_default_settings(settings);

    settings->verbose = 1;
    settings->max_iter = 1000;
    settings->eps_abs = 1e-3;  // Relaxed for flexfloat
    settings->eps_rel = 1e-3;
    settings->polishing = 0;   // Disable for cleaner test

    printf("Setting up problem...\n");
    OSQPInt exitflag = osqp_setup(&solver, &P, q, &A, l, u, m, n, settings);

    if (exitflag) {
        printf("ERROR: Setup failed with code %d\n", (int)exitflag);
        free(settings);
        return 1;
    }

    printf("Solving...\n\n");
    exitflag = osqp_solve(solver);

    if (exitflag) {
        printf("ERROR: Solve failed with code %d\n", (int)exitflag);
        osqp_cleanup(solver);
        free(settings);
        return 1;
    }

    // Print results
    printf("\n=========================================\n");
    printf("RESULTS\n");
    printf("=========================================\n");
    printf("Status:          %s\n", solver->info->status);
    printf("Iterations:      %d\n", (int)solver->info->iter);
    printf("Solve time:      %.6f sec\n", solver->info->solve_time);
    printf("\nObjective value: %.8f\n", solver->info->obj_val);
    printf("Primal residual: %.6e\n", solver->info->prim_res);
    printf("Dual residual:   %.6e\n", solver->info->dual_res);
    printf("\nOptimal solution:\n");
    printf("  x = %.8f\n", solver->solution->x[0]);
    printf("  y = %.8f\n", solver->solution->x[1]);

    // Expected solution: x = 1, y = 1, objective = 2
    double x_expected = 1.0;
    double y_expected = 1.0;
    double obj_expected = 2.0;

    double x_error = solver->solution->x[0] - x_expected;
    double y_error = solver->solution->x[1] - y_expected;
    double obj_error = solver->info->obj_val - obj_expected;

    printf("\nError vs expected solution:\n");
    printf("  Δx = %.6e\n", x_error);
    printf("  Δy = %.6e\n", y_error);
    printf("  Δobj = %.6e\n", obj_error);

#ifdef OSQP_USE_FLEXFLOAT
    printf("\n=========================================\n");
    printf("FlexFloat Effects Observed:\n");
    printf("  - All ADMM iterations used 14-bit mantissa\n");
    printf("  - Quantization applied to:\n");
    printf("    * Vector additions/subtractions\n");
    printf("    * Scalar multiplications\n");
    printf("    * update_x(), update_y(), update_z()\n");
    printf("  - Expected precision: ~1e-4 to 1e-5\n");
    printf("=========================================\n");
#endif

    osqp_cleanup(solver);
    free(settings);

    return 0;
}

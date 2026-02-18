// Copyright 2019 Alexander Liniger

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifndef MPCC_OSQP_INTERFACE_H
#define MPCC_OSQP_INTERFACE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cassert>

#include "OsqpEigen/OsqpEigen.h"

#include "Model/model.h"
#include "Cost/cost.h"
#include "Constraints/constraints.h"
#include "Constraints/bounds.h"
#include "solver_interface.h"
#include "Constraints/EnvCollision/EnvCollisionModel.h"

#include <config.h>
#include <iostream>
#include <memory>
#include <filesystem>
#include <cassert>
#include <vector>
#include <thread>

namespace mpcc{

/// @brief parameters for optimization, state and control input of MPC
/// @param xk  (State) state
/// @param uk  (Input) control input
struct OptVariables 
{
    State xk;
    Input uk;

    bool operator==(const OptVariables& other) const 
    {
        return xk == other.xk && uk == other.uk;
    }
    void setZero()
    {
        xk.setZero();
        uk.setZero();
    }
};

struct FilterData
{
    double obj;
    double gap_vio;
    // double gap_constr;
};

struct ComputeTime
{
    double set_env;
    double set_qp;
    double init_solver;  // initSolver time (legacy: scaling + permutation + factorization combined)
    double solve_qp;     // pure ADMM iteration time + initial factorization (for fair comparison)
    double get_alpha;
    double total;
    // Detailed timing breakdown (from osqp_setup)
    double scaling_time;       // Data scaling time
    double permutation_time;   // Permutation computation time
    double factorization_time; // Initial LDL factorization time
    // Adaptive rho information
    int rho_updates;           // Number of rho updates performed during solve
    void setZero(){
        set_env=0; set_qp=0; init_solver=0; solve_qp=0; get_alpha=0; total=0;
        scaling_time=0; permutation_time=0; factorization_time=0;
        rho_updates=0;
    }
};

class OsqpInterface : public SolverInterface {
public:
    OsqpInterface(double Ts,const PathToJson &path);
    OsqpInterface(double Ts,const PathToJson &path,const ParamValue &param_value);
    void setTrack(const ArcLengthSpline track);
    void setParam(const ParamValue &param_value);
    // void setEnvData(const std::vector<float> &voxel);
    void setEnvData(const Eigen::MatrixX3d &obs_positions, const double &obs_radius);
    void setCurrentInput(const Input &cutrent_input);
    void setInitialGuess(const std::vector<OptVariables> &initial_guess);
    bool solveOCP(std::vector<OptVariables> &opt_sol, Status *status, ComputeTime *mpc_time, int &iter_count, int &sqp_iter_count, int &total_iter_count, int solve_count = 0);
    // ▼▼▼▼▼▼▼▼▼▼▼ 여기에 getter 함수를 추가하세요 ▼▼▼▼▼▼▼▼▼▼▼
    const std::unique_ptr<EnvCollNNmodel>& getEnvColNN() const { return envcolNN_; }

    ~OsqpInterface(){ std::cout << "Deleting Osqp Interface" << std::endl;}

    

private:
    ArcLengthSpline track_;
    PathToJson path_;
    std::unique_ptr<RobotModel> robot_;
    std::unique_ptr<SelCollNNmodel> selcolNN_;
    std::unique_ptr<EnvCollNNmodel> envcolNN_;

    Cost cost_;
    Model model_;
    Constraints constraints_;
    Bounds bounds_;
    NormalizationParam normalization_param_;
    SQPParam sqp_param_;
    CostParam cost_param_;

    double Ts_;

    std::vector<RobotData> rb_;

    static const int N_var = (N+1)*NX + N*NU ;
    static const int N_eq = (N+1)*NX;
    static const int N_ineqb = N_var + N*NU; // add ddot joint constraint
    static const int N_ineqp = (N+1)*NPC; 
    static const int N_constr = N_eq + N_ineqb + N_ineqp;

    Input current_u_;
    std::vector<OptVariables> initial_guess_;
    Eigen::VectorXd initial_guess_vec_;
    Eigen::VectorXd lambda_;
    Eigen::VectorXd step_, step_prev_, step_lambda_;
    Eigen::VectorXd grad_L_;
    Eigen::VectorXd delta_grad_L_;

    Eigen::MatrixXd Hess_;
    Eigen::VectorXd grad_obj_;
    double obj_;
    Eigen::MatrixXd jac_constr_;
    Eigen::VectorXd constr_;
    Eigen::VectorXd l_, u_;

    double dual_step_norm_;
    double primal_step_norm_;

    // Timing for solveQP breakdown
    double last_init_solver_time_ = 0.0;  // initSolver time (scaling + permutation + factorization)
    double last_solve_time_ = 0.0;        // pure ADMM iteration time
    // Detailed timing breakdown (from osqp_setup)
    double last_scaling_time_ = 0.0;       // Data scaling time
    double last_permutation_time_ = 0.0;   // Permutation computation time
    double last_factorization_time_ = 0.0; // Initial LDL factorization time
    // Adaptive rho information
    int last_rho_updates_ = 0;             // Number of rho updates performed

    std::vector<FilterData> filter_data_list_;

    // OsqpEigen::Solver solver_;

    unsigned int sqp_iter_;
    int current_solve_count_;

    bool is_solved_ = false;
    OsqpEigen::Status qp_status_;

    void setMultiThread();
    void setCost(const std::vector<OptVariables> &initial_guess, 
                 double *obj, Eigen::VectorXd *grad_obj, Eigen::MatrixXd *hess_obj);
    void setDynamics(const std::vector<OptVariables> &initial_guess,
                     Eigen::MatrixXd *jac_constr_eq, Eigen::VectorXd *constr_eq, Eigen::VectorXd *l_eq, Eigen::VectorXd *u_eq);
    void setBounds(const std::vector<OptVariables> &initial_guess,
                   Eigen::MatrixXd *jac_constr_ineqb, Eigen::VectorXd *constr_ineqb, Eigen::VectorXd *l_ineqb, Eigen::VectorXd *u_ineqb);
    void setPolytopicConstraints(const std::vector<OptVariables> &initial_guess,
                                 Eigen::MatrixXd *jac_constr_ineqp, Eigen::VectorXd *constr_ineqp, Eigen::VectorXd *l_ineqp, Eigen::VectorXd *u_ineqp);
    void setConstraints(const std::vector<OptVariables> &initial_guess,
                        Eigen::MatrixXd *jac_constr, Eigen::VectorXd *constr, Eigen::VectorXd *l, Eigen::VectorXd *u);
    void setQP(const std::vector<OptVariables> &initial_guess,
               Eigen::MatrixXd *hess_obj, Eigen::VectorXd *grad_obj, double *obj, Eigen::MatrixXd *jac_constr, Eigen::VectorXd *constr, Eigen::VectorXd *l, Eigen::VectorXd *u);
    bool solveQP(const Eigen::MatrixXd &P, const Eigen::VectorXd &q, const Eigen::MatrixXd &A, const Eigen::VectorXd &l,const Eigen::VectorXd &u,
                 Eigen::VectorXd &step, Eigen::VectorXd &step_lambda, OsqpEigen::Status& qp_status, int &iter_count);
    bool SecondOrderCorrection(const std::vector<OptVariables> &initial_guess, const Eigen::MatrixXd &hess_obj, const Eigen::VectorXd &grad_obj, const Eigen::MatrixXd & jac_constr,
                               Eigen::VectorXd &step, Eigen::VectorXd &step_lambda, OsqpEigen::Status& qp_status);
    Eigen::MatrixXd BFGSUpdate(const Eigen::MatrixXd &Hess, const Eigen::VectorXd &step_prev, const Eigen::VectorXd &delta_grad_L);
    double meritLineSearch(const Eigen::VectorXd &step, const Eigen::MatrixXd &Hess, const Eigen::VectorXd &grad_obj, const double &obj, const Eigen::VectorXd &constr, const Eigen::VectorXd &l, const Eigen::VectorXd &u);
    double filterLineSearch(const std::vector<OptVariables> &initial_guess, const Eigen::VectorXd &step, 
                            std::vector<FilterData> &filter_data_list);
    bool isPosdef(const Eigen::MatrixXd& H);
    // custom function
    bool printAllEigenvalues(const Eigen::MatrixXd& H);
    template <typename Derived>
    void saveMatrixToFile(const Eigen::MatrixBase<Derived>& Matrix, const std::string& matrixName, const std::string& filename);
    void saveSparseMatrixToFile(const Eigen::SparseMatrix<float, 0, int>& Matrix, const std::string& matrixName, const std::string& filename);
    
    void saveAsicE2EInput(
        const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
        const Eigen::MatrixXd &A, const Eigen::VectorXd &l, const Eigen::VectorXd &u,
        const std::vector<OptVariables> &initial_guess, int sample_id);
    void saveAsicE2EOutput(
        const Eigen::VectorXd &step, const Eigen::VectorXd &step_lambda, int sample_id);
    bool isNan(const Eigen::MatrixXd& x);
    double constraint_norm(const Eigen::VectorXd &constr, const Eigen::VectorXd &l, const Eigen::VectorXd &u);
    std::vector<OptVariables> vectorToOptvar(const Eigen::VectorXd& opt_var_vec);
    Eigen::VectorXd OptvarToVector(const std::vector<OptVariables>& opt_var);
    Eigen::VectorXd deNormalizeStep(const Eigen::VectorXd& step);
    void printOptVar(std::vector<OptVariables> opt_var);

    // Diagonal scaling for ASIC implementation
    void computeDiagonalScaling(const Eigen::MatrixXd &P, const Eigen::MatrixXd &A,
                                Eigen::VectorXd &D, Eigen::VectorXd &E);
    void applyScaling(const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
                      const Eigen::MatrixXd &A, const Eigen::VectorXd &l, const Eigen::VectorXd &u,
                      const Eigen::VectorXd &D, const Eigen::VectorXd &E,
                      Eigen::MatrixXd &P_scaled, Eigen::VectorXd &q_scaled,
                      Eigen::MatrixXd &A_scaled, Eigen::VectorXd &l_scaled, Eigen::VectorXd &u_scaled);
    void unscaleSolution(const Eigen::VectorXd &x_scaled, const Eigen::VectorXd &lambda_scaled,
                         const Eigen::VectorXd &D, const Eigen::VectorXd &E,
                         Eigen::VectorXd &x, Eigen::VectorXd &lambda);
    bool solveQPWithScaling(const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
                            const Eigen::MatrixXd &A, const Eigen::VectorXd &l, const Eigen::VectorXd &u,
                            Eigen::VectorXd &step, Eigen::VectorXd &step_lambda,
                            OsqpEigen::Status &qp_status, int &iter_count);
};
}
#endif //MPCC_OSQP_INTERFACE_H
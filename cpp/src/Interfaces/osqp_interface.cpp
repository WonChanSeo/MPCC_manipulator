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

#include "Interfaces/osqp_interface.h"
#include "osqp_api_functions.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

#if defined(CONSTRAINTS_USE_TRUNCATE) || defined(CONSTRAINTS_USE_ROUND_TO_EVEN)
#include <cfenv>
#endif

#if defined(CONSTRAINTS_USE_TRUNCATE)
  #define CONSTRAINTS_ROUNDING_MODE FE_TOWARDZERO
#elif defined(CONSTRAINTS_USE_ROUND_TO_EVEN)
  #define CONSTRAINTS_ROUNDING_MODE FE_TONEAREST
#endif

// Enable this to save ASIC end-to-end test cases with EnvCollision parts zeroed out.
// Simulation runs normally (original data goes to solveQP). A separate copy with
// EnvCollision zeroed is saved to files. In the saved test case:
//   - A matrix: EnvCollision rows zeroed (ASIC fills from MLP)
//   - u vector: EnvCollision slots contain dq (ASIC reads dq, then overwrites with proper u)
//   - l vector: EnvCollision slots remain -INF (unchanged)
//   - P, q: unchanged (not affected by EnvCollision)
#define ASIC_ENVCOL_ZERO_MODE

#ifdef ASIC_ENVCOL_ZERO_MODE
#define ASIC_TESTCASE_SAVE_INTERVAL 50
static bool g_asic_dir_initialized = false;
static std::string resolve_asic_e2e_output_dir()
{
    const char* base_dir = std::getenv("ASIC_TESTCASE_BASE_DIR");
    if (base_dir && base_dir[0] != '\0') {
        std::string dir(base_dir);
        if (dir.back() != '/') dir += "/";
        return dir + "e2e";
    }
    return "/home/mms-wonchan/git/MPCC_manipulator/result/asic_testcases_runtime/e2e";
}
#endif

// FP32 hex helper for constraint logging
static std::string constraint_float_to_hex(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << bits;
    return ss.str();
}

// Global log file for solveQP failures
static std::ofstream g_solveQP_fail_log;

namespace mpcc{
OsqpInterface::OsqpInterface(double Ts,const PathToJson &path)
:cost_(path),
model_(Ts,path),
constraints_(Ts,path),
bounds_(BoundsParam(path.bounds_path),Param(path.param_path)),
normalization_param_(path.normalization_path),
sqp_param_(path.sqp_path),
cost_param_(path.cost_path),
path_(path),
Ts_(Ts)
{   
    setMultiThread();
    robot_ = std::make_unique<RobotModel>();

    selcolNN_ = std::make_unique<SelCollNNmodel>();
    Eigen::VectorXd sel_col_n_hidden(2);
    sel_col_n_hidden << 256, 64;
    selcolNN_->setNeuralNetwork(PANDA_DOF, 1, sel_col_n_hidden, true);

    envcolNN_ = std::make_unique<EnvCollNNmodel>();
    Eigen::VectorXd env_col_n_hidden(4);
    env_col_n_hidden << 256, 256, 256, 256;
    envcolNN_->setNeuralNetwork(PANDA_DOF+3, PANDA_NUM_LINKS, env_col_n_hidden, true);

    initial_guess_.resize(N+1);
    rb_.resize(N+1);
    for(size_t i=0; i<rb_.size(); i++) rb_[i].setZero();
}

OsqpInterface::OsqpInterface(double Ts,const PathToJson &path,const ParamValue &param_value)
:cost_(path, param_value),
model_(Ts,path),
constraints_(Ts,path,param_value),
bounds_(BoundsParam(path.bounds_path),Param(path.param_path,param_value.param)),
normalization_param_(path.normalization_path,param_value.normalization),
sqp_param_(path.sqp_path,param_value.sqp),
cost_param_(path.cost_path),
path_(path),
Ts_(Ts)
{   
    setMultiThread();
    robot_ = std::make_unique<RobotModel>();

    selcolNN_ = std::make_unique<SelCollNNmodel>();
    Eigen::VectorXd sel_col_n_hidden(2);
    sel_col_n_hidden << 256, 64;
    selcolNN_->setNeuralNetwork(PANDA_DOF, 1, sel_col_n_hidden, true);

    envcolNN_ = std::make_unique<EnvCollNNmodel>();
    Eigen::VectorXd env_col_n_hidden(4);
    env_col_n_hidden << 256, 256, 256, 256;
    envcolNN_->setNeuralNetwork(PANDA_DOF+3, PANDA_NUM_LINKS, env_col_n_hidden, true);

    initial_guess_.resize(N+1);
    rb_.resize(N+1);
    for(size_t i=0; i<rb_.size(); i++) rb_[i].setZero();
}

void OsqpInterface::setMultiThread()
{
    unsigned int cores = thread::hardware_concurrency();
    cout << "Available cores: " << cores << endl;

    cout << "Default threads: " << Eigen::nbThreads() << endl;
    Eigen::setNbThreads(cores > 1 ? cores - 1 : 1);
    Eigen::initParallel();
    cout << "Updated threads: " << Eigen::nbThreads() << endl;

    // Initialize solveQP failure log file
    if (!g_solveQP_fail_log.is_open()) {
        g_solveQP_fail_log.open("/home/mms-wonchan/git/MPCC_manipulator/result/solveQP_failures.txt", std::ios::out | std::ios::trunc);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "=== solveQP Failure Log ===" << std::endl;
            g_solveQP_fail_log << "Format: [FAIL TYPE] solve_count, sqp_iter, additional_info" << std::endl;
            g_solveQP_fail_log << "==========================================" << std::endl;
        }
    }
}

void OsqpInterface::setTrack(const ArcLengthSpline track)
{
    track_ = track;
}

void OsqpInterface::setParam(const ParamValue &param_value)
{
    cost_ = Cost(path_, param_value);
    constraints_ = Constraints(Ts_,path_,param_value);
    bounds_ = Bounds(BoundsParam(path_.bounds_path),Param(path_.param_path,param_value.param));
}

void OsqpInterface::setEnvData(const Eigen::MatrixX3d &obs_positions, const double &obs_radius)
{
    for(size_t i=0; i<=N; i++)
    {
        // printf("%d obs_positions: %f %f %f\n", i, obs_positions(0), obs_positions(1), obs_positions(2));

        rb_[i].updateEnv(obs_positions, obs_radius, envcolNN_);
        auto time = std::chrono::high_resolution_clock::now();
        // printf("Env data updated for step %zu at time: %lld ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count());
    }
}

void OsqpInterface::setInitialGuess(const std::vector<OptVariables> &initial_guess)
{
    for(size_t i=0; i<rb_.size(); i++) rb_[i].setZero();
    initial_guess_ = initial_guess;
    initial_guess_vec_.setZero(N_var);
    for(size_t i=0;i<=N; i++)
    {
        initial_guess_vec_.segment(NX*i, NX) = stateToVector(initial_guess[i].xk);
        if(i != N) initial_guess_vec_.segment(NX*(N+1) + NU*i, NU) = inputToVector(initial_guess[i].uk);

        rb_[i].update(stateToJointVector(initial_guess[i].xk), robot_, selcolNN_);
    }
}

void OsqpInterface::setCurrentInput(const Input &current_input)
{
    current_u_ = current_input;
}

void OsqpInterface::setCost(const std::vector<OptVariables> &initial_guess, 
                            double *obj, Eigen::VectorXd *grad_obj, Eigen::MatrixXd *hess_obj)
{
    if(obj) (*obj) = 0;
    if(grad_obj) grad_obj->setZero(N_var);
    if(hess_obj) hess_obj->setZero(N_var,N_var);
    for(size_t i=0; i<=N; i++)
    {
        assert(rb_[i].isUpdated() == true);

        double obj_k;
        CostGrad grad_cost_k;
        CostHess hess_cost_k;

        (&hess_cost_k)->setZero();

        // printf("Setting getCost...\n");
        if (&hess_cost_k) {
            // printf("Hessian is set\n");
            // // (0,0) 원소 출력
            // // printf("hess_obj(0,0) = %g\n", hess->f_xu);
            // printf("hess->f_xu: %f\n", (&hess_cost_k)->f_xu.norm());
        } else {
            // printf("Hessian is not set\n");
        }
        
        if(obj && !grad_obj && !hess_obj)
        {
            cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rb_[i],i, &obj_k, NULL,NULL);
            
        }
        else if(obj && grad_obj && !hess_obj)
        {
            cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rb_[i],i, &obj_k, &grad_cost_k,NULL);
        }
        else if(obj && grad_obj && hess_obj)
        {
            cost_.getCost(track_,initial_guess[i].xk,initial_guess[i].uk,rb_[i],i, &obj_k, &grad_cost_k,&hess_cost_k);
        }
        if(obj) (*obj) += obj_k;
        if(grad_obj) grad_obj->segment(NX*i, NX) = normalization_param_.T_x*grad_cost_k.f_x;
        if(hess_obj) hess_obj->block(NX*i,NX*i,NX,NX) = normalization_param_.T_x*hess_cost_k.f_xx*normalization_param_.T_x;
        if(i != N)
        {
            if(grad_obj) grad_obj->segment(NX*(N+1)+NU*i, NU) = normalization_param_.T_u*grad_cost_k.f_u;
            if(hess_obj) hess_obj->block(NX*(N+1)+NU*i,NX*(N+1)+NU*i,NU,NU) = normalization_param_.T_u*hess_cost_k.f_uu*normalization_param_.T_u;
            if(hess_obj) hess_obj->block(NX*i,NX*(N+1)+NU*i,NX,NU) = normalization_param_.T_x*hess_cost_k.f_xu*normalization_param_.T_u;
            if(hess_obj) hess_obj->block(NX*(N+1)+NU*i,NX*i,NU,NX) = (normalization_param_.T_x*hess_cost_k.f_xu*normalization_param_.T_u).transpose();
        }
        // for ddjoint cost
        if(i != N)
        {
            if(obj)
            {
                if(i != N-1) 
                {
                    (*obj) += cost_param_.r_ddq * (inputTodJointVector(initial_guess[i+1].uk) - inputTodJointVector(initial_guess[i].uk)).squaredNorm();
                }
            }
            if(grad_obj)
            {
                Eigen::VectorXd ddq_grad(PANDA_DOF);
                if(i == 0)
                {
                    ddq_grad = 2. * cost_param_.r_ddq * (inputTodJointVector(initial_guess[i].uk) - inputTodJointVector(initial_guess[i+1].uk));
                }
                else if(i == N-1)
                {
                    ddq_grad = 2. * cost_param_.r_ddq * (inputTodJointVector(initial_guess[i].uk) - inputTodJointVector(initial_guess[i-1].uk));
                }
                else
                {
                    ddq_grad = 2. * cost_param_.r_ddq * (2.*inputTodJointVector(initial_guess[i].uk) - inputTodJointVector(initial_guess[i+1].uk) - inputTodJointVector(initial_guess[i-1].uk));
                }
                grad_obj->segment(NX*(N+1)+NU*i, PANDA_DOF) += normalization_param_.T_u.block(si_index.dq1,si_index.dq1,PANDA_DOF,PANDA_DOF)*ddq_grad;
            }
            if(hess_obj)
            {
                Eigen::Matrix<double, PANDA_DOF, PANDA_DOF> ddq_hess_ii, ddq_hess_ij; // j=i+1
                if(i == 0)
                {
                    ddq_hess_ii = 2. * cost_param_.r_ddq * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF);
                    ddq_hess_ij = -2. * cost_param_.r_ddq * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF);
                }
                else if(i == N-1)
                {
                    ddq_hess_ii = 2. * cost_param_.r_ddq * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF);
                }
                else
                {
                    ddq_hess_ii = 4. * cost_param_.r_ddq * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF);
                    ddq_hess_ij = -2. * cost_param_.r_ddq * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF);
                }
                hess_obj->block(NX*(N+1)+NU*i,NX*(N+1)+NU*i,PANDA_DOF,PANDA_DOF) += normalization_param_.T_u.block(si_index.dq1,si_index.dq1,PANDA_DOF,PANDA_DOF)*ddq_hess_ii*normalization_param_.T_u.block(si_index.dq1,si_index.dq1,PANDA_DOF,PANDA_DOF);
                if(i != N-1)
                {
                    hess_obj->block(NX*(N+1)+NU*i,NX*(N+1)+NU*(i+1),PANDA_DOF,PANDA_DOF) += normalization_param_.T_u.block(si_index.dq1,si_index.dq1,PANDA_DOF,PANDA_DOF)*ddq_hess_ij*normalization_param_.T_u.block(si_index.dq1,si_index.dq1,PANDA_DOF,PANDA_DOF);
                    hess_obj->block(NX*(N+1)+NU*(i+1),NX*(N+1)+NU*i,PANDA_DOF,PANDA_DOF) += normalization_param_.T_u.block(si_index.dq1,si_index.dq1,PANDA_DOF,PANDA_DOF)*ddq_hess_ij*normalization_param_.T_u.block(si_index.dq1,si_index.dq1,PANDA_DOF,PANDA_DOF);
                }
            }
        }
    }
}

void OsqpInterface::setDynamics(const std::vector<OptVariables> &initial_guess,
                                Eigen::MatrixXd *jac_constr_eq, Eigen::VectorXd *constr_eq, Eigen::VectorXd *l_eq, Eigen::VectorXd *u_eq)
{
    if(jac_constr_eq) jac_constr_eq->setZero(N_eq,N_var);
    if(constr_eq) constr_eq->setZero(N_eq);
    if(l_eq) l_eq->setZero(N_eq);
    if(u_eq) u_eq->setZero(N_eq);

    for(size_t i=0;i<=N;i++)
    {
        if(i == 0)
        {
            if(jac_constr_eq) jac_constr_eq->block(0,0,NX,NX) = MatrixXd::Identity(NX,NX);
            if(constr_eq) constr_eq->segment(0,NX) = VectorXd::Zero(NX);
            if(l_eq) l_eq->segment(0,NX) = VectorXd::Zero(NX);
            if(u_eq) u_eq->segment(0,NX) = VectorXd::Zero(NX);
        }
        else
        {
            LinModelMatrix lin_model_prev = model_.getLinModel(initial_guess[i-1].xk,initial_guess[i-1].uk);
            if(jac_constr_eq)
            {
                jac_constr_eq->block(NX*i, NX*(i-1), NX, NX) = -normalization_param_.T_x_inv*lin_model_prev.A*normalization_param_.T_x;
                // std::cout << "lin_model_prev.A: \n" << lin_model_prev.A << std::endl;
                // std::cout << "normalization_param_.T_x: \n" << normalization_param_.T_x << std::endl;
                // std::cout << "normalization_param_.T_x_inv: \n" << normalization_param_.T_x_inv << std::endl;
                // std::cout << "Block being set at iteration " << i << ": \n" << -normalization_param_.T_x_inv*lin_model_prev.A*normalization_param_.T_x << std::endl;
                jac_constr_eq->block(NX*i, NX*i, NX, NX) = MatrixXd::Identity(NX,NX);
                jac_constr_eq->block(NX*i, NX*(N+1) + NU*(i-1), NX, NU) = -normalization_param_.T_x_inv*lin_model_prev.B*normalization_param_.T_u;
                // std::cout << "lin_model_prev.B: \n" << lin_model_prev.B << std::endl;
                // std::cout << "normalization_param_.T_u: \n" << normalization_param_.T_u << std::endl;
                // std::cout << "normalization_param_.T_x_inv: \n" << normalization_param_.T_x_inv << std::endl;
                // std::cout << "Block being set at iteration " << i << ": \n" << -normalization_param_.T_x_inv*lin_model_prev.B*normalization_param_.T_u << std::endl;

            }
            if(constr_eq) constr_eq->segment(NX*i, NX) = normalization_param_.T_x_inv * (stateToVector(initial_guess[i].xk) - (lin_model_prev.A*stateToVector(initial_guess[i-1].xk) + lin_model_prev.B*inputToVector(initial_guess[i-1].uk) +  lin_model_prev.g));
            if(l_eq) l_eq->segment(NX*i, NX) = VectorXd::Zero(NX);
            if(u_eq) u_eq->segment(NX*i, NX) = VectorXd::Zero(NX);
        }

        // if (jac_constr_eq) {
        //     std::cout << "jac_constr_eq matrix after iteration " << i << ": \n" << *jac_constr_eq << std::endl;
        //     std::cout << "jac_constr_eq matrix: \n" << *jac_constr_eq << std::endl;
        // }
    }
}

void OsqpInterface::setBounds(const std::vector<OptVariables> &initial_guess,
                              Eigen::MatrixXd *jac_constr_ineqb, Eigen::VectorXd *constr_ineqb, Eigen::VectorXd *l_ineqb, Eigen::VectorXd *u_ineqb)
{
    if(jac_constr_ineqb) jac_constr_ineqb->setZero(N_ineqb,N_var);
    if(constr_ineqb) constr_ineqb->setZero(N_ineqb);
    if(l_ineqb) l_ineqb->setZero(N_ineqb);
    if(u_ineqb) u_ineqb->setZero(N_ineqb);

    for(size_t i=0;i<=N;i++)
    {
        // for state bounds
        if(jac_constr_ineqb) jac_constr_ineqb->block(NX*i, NX*i, NX, NX) = Eigen::MatrixXd::Identity(NX, NX) * normalization_param_.T_x;
        if(constr_ineqb) constr_ineqb->segment(NX*i, NX) = stateToVector(initial_guess[i].xk);
        if(l_ineqb) l_ineqb->segment(NX*i, NX) = bounds_.getBoundsLX(initial_guess[i].xk);
        if(u_ineqb) u_ineqb->segment(NX*i, NX) = bounds_.getBoundsUX(initial_guess[i].xk,track_.getLength());

        if(i != N)
        {
            // for control input bounds
            if(jac_constr_ineqb) jac_constr_ineqb->block(NX*(N+1) + NU*i, NU*i, NU, NU) = Eigen::MatrixXd::Identity(NU, NU) * normalization_param_.T_u;
            if(constr_ineqb) constr_ineqb->segment(NX*(N+1) + NU*i, NU) =inputToVector(initial_guess[i].uk);
            if(l_ineqb) l_ineqb->segment(NX*(N+1) + NU*i, NU) = bounds_.getBoundsLU();
            if(u_ineqb) u_ineqb->segment(NX*(N+1) + NU*i, NU) = bounds_.getBoundsUU();

            // for ddjoint bounds
            if(i==0)
            {
                if(jac_constr_ineqb) jac_constr_ineqb->block(NX*(N+1) + NU*N + NU*i, NX*(N+1) + NU*i, PANDA_DOF, PANDA_DOF) = 1./Ts_ * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF) * normalization_param_.T_u.block(si_index.dq1, si_index.dq1, PANDA_DOF, PANDA_DOF);
                if(constr_ineqb) constr_ineqb->segment(NX*(N+1) + NU*N + NU*i, PANDA_DOF) = 1./Ts_ *  inputTodJointVector(initial_guess[i].uk);
                if(l_ineqb) l_ineqb->segment(NX*(N+1) + NU*N + NU*i, PANDA_DOF) = bounds_.getBoundsLddJoint() + 1./Ts_ * inputTodJointVector(current_u_);
                if(u_ineqb) u_ineqb->segment(NX*(N+1) + NU*N + NU*i, PANDA_DOF) = bounds_.getBoundsUddJoint() + 1./Ts_ * inputTodJointVector(current_u_);
            }
            else
            {
                if(jac_constr_ineqb)
                {
                    jac_constr_ineqb->block(NX*(N+1) + NU*N + NU*i, NX*(N+1) + NU*i,     PANDA_DOF, PANDA_DOF) =  1./Ts_ * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF) * normalization_param_.T_u.block(si_index.dq1, si_index.dq1, PANDA_DOF, PANDA_DOF);
                    jac_constr_ineqb->block(NX*(N+1) + NU*N + NU*i, NX*(N+1) + NU*(i-1), PANDA_DOF, PANDA_DOF) = -1./Ts_ * Eigen::MatrixXd::Identity(PANDA_DOF, PANDA_DOF) * normalization_param_.T_u.block(si_index.dq1, si_index.dq1, PANDA_DOF, PANDA_DOF);

                }
                if(constr_ineqb) constr_ineqb->segment(NX*(N+1) + NU*N + NU*i, PANDA_DOF) = 1./Ts_ * (inputTodJointVector(initial_guess[i].uk) - inputTodJointVector(initial_guess[i-1].uk));
                if(l_ineqb) l_ineqb->segment(NX*(N+1) + NU*N + NU*i, PANDA_DOF) = bounds_.getBoundsLddJoint();
                if(u_ineqb) u_ineqb->segment(NX*(N+1) + NU*N + NU*i, PANDA_DOF) = bounds_.getBoundsUddJoint();
            }
        }
    }
}

static std::string resolve_constraint_output_dir()
{
    const char* base_dir = std::getenv("ASIC_TESTCASE_BASE_DIR");
    if (base_dir && base_dir[0] != '\0') {
        std::string dir(base_dir);
        if (dir.back() != '/') dir += "/";
        return dir + "constraints/";
    }
    return "/home/mms-wonchan/git/MPCC_manipulator/result/asic_testcases_runtime/constraints/";
}

void OsqpInterface::setPolytopicConstraints(const std::vector<OptVariables> &initial_guess,
                                            Eigen::MatrixXd *jac_constr_ineqp, Eigen::VectorXd *constr_ineqp, Eigen::VectorXd *l_ineqp, Eigen::VectorXd *u_ineqp)
{
    if(jac_constr_ineqp) jac_constr_ineqp->setZero(N_ineqp,N_var);
    if(constr_ineqp) constr_ineqp->setZero(N_ineqp);
    if(l_ineqp) l_ineqp->setZero(N_ineqp);
    if(u_ineqp) u_ineqp->setZero(N_ineqp);

    // ===== Constraint logging 준비 =====
    static const int CONSTRAINT_SAVE_INTERVAL = 50;
    static const std::string g_constraint_output_dir = resolve_constraint_output_dir();
    static bool g_constraint_dir_initialized = false;
    bool should_log_constraints = (sqp_iter_ == 0 && current_solve_count_ >= 0
                                   && current_solve_count_ % CONSTRAINT_SAVE_INTERVAL == 0);

    struct EnvCollLogEntry {
        Eigen::VectorXd env_min_dist;       // MLP output (PANDA_NUM_LINKS)
        Eigen::MatrixXd d_env_min_dist;     // MLP Jacobian (PANDA_NUM_LINKS x PANDA_DOF)
        double obs_radius;
        Eigen::VectorXd constraint_c;       // constraint value (PANDA_NUM_LINKS)
        Eigen::MatrixXd jac_c_x;            // Jacobian wrt state - raw (PANDA_NUM_LINKS x NX)
        Eigen::MatrixXd jac_c_u;            // Jacobian wrt input - raw (PANDA_NUM_LINKS x NU)
        Eigen::MatrixXd jac_c_x_scaled;    // Jacobian wrt state * T_x (A matrix에 들어가는 값)
        Eigen::MatrixXd jac_c_u_scaled;    // Jacobian wrt input * T_u (A matrix에 들어가는 값)
    };
    std::vector<EnvCollLogEntry> constraint_log_entries;
    if (should_log_constraints) constraint_log_entries.resize(N + 1);

    for(size_t i=0;i<=N;i++)
    {
        assert(rb_[i].isUpdated() == true);

        ConstraintsInfo constr_info_k;
        ConstraintsJac jac_constr_k;
        if(constr_ineqp && !jac_constr_ineqp)
        {
            constraints_.getConstraints(initial_guess[i].xk,initial_guess[i].uk,rb_[i],i,&constr_info_k,NULL);
        }
        else if(constr_ineqp && jac_constr_ineqp)
        {
            constraints_.getConstraints(initial_guess[i].xk,initial_guess[i].uk,rb_[i],i,&constr_info_k,&jac_constr_k);
            // std:cout << "self collision constraint["<<i<<"]: " << constr_info_k.c_vec(si_index.con_selcol) << " < " << constr_info_k.c_uvec(si_index.con_selcol) <<std::endl;
        }
        if(jac_constr_ineqp)
        {
            // Non-EnvCollision rows (selcol, sing): double precision T_x/T_u multiplication
            const int n_non_envcol = si_index.con_envcol1;  // selcol + sing rows
            jac_constr_ineqp->block(NPC*i, NX*i, n_non_envcol, NX) =
                jac_constr_k.c_x.topRows(n_non_envcol) * normalization_param_.T_x;
            if(i!=N) jac_constr_ineqp->block(NPC*i, NX*(N+1) + NU*i, n_non_envcol, NU) =
                jac_constr_k.c_u.topRows(n_non_envcol) * normalization_param_.T_u;

            // EnvCollision rows: FP32 precision T_x/T_u multiplication, then zero-pad to double
            // jac_c_x EnvCollision values are already FP32 (cast to double in getEnvcollConstraint)
            // T_x/T_u are diagonal matrices; each element is a single FP32 multiply (no accumulation error)
            {
#ifdef CONSTRAINTS_ROUNDING_MODE
                int old_round = std::fegetround();
                std::fesetround(CONSTRAINTS_ROUNDING_MODE);
#endif
                Eigen::Matrix<float, PANDA_NUM_LINKS, NX> envcol_cx_f =
                    jac_constr_k.c_x.block(si_index.con_envcol1, 0, PANDA_NUM_LINKS, NX).cast<float>();
                Eigen::Matrix<float, NX, NX> T_x_f = normalization_param_.T_x.cast<float>();
                jac_constr_ineqp->block(NPC*i + si_index.con_envcol1, NX*i, PANDA_NUM_LINKS, NX) =
                    (envcol_cx_f * T_x_f).cast<double>();

                if(i!=N) {
                    Eigen::Matrix<float, PANDA_NUM_LINKS, NU> envcol_cu_f =
                        jac_constr_k.c_u.block(si_index.con_envcol1, 0, PANDA_NUM_LINKS, NU).cast<float>();
                    Eigen::Matrix<float, NU, NU> T_u_f = normalization_param_.T_u.cast<float>();
                    jac_constr_ineqp->block(NPC*i + si_index.con_envcol1, NX*(N+1) + NU*i, PANDA_NUM_LINKS, NU) =
                        (envcol_cu_f * T_u_f).cast<double>();
                }
#ifdef CONSTRAINTS_ROUNDING_MODE
                std::fesetround(old_round);
#endif
            }
        }
        if(constr_ineqp)
        {
            constr_ineqp->segment(NPC*i, NPC) = constr_info_k.c_vec;
        }
        if(l_ineqp)
        {
            l_ineqp->segment(NPC*i, NPC) = constr_info_k.c_lvec;
        }
        if(u_ineqp)
        {
            u_ineqp->segment(NPC*i, NPC) = constr_info_k.c_uvec;
        }

        // Accumulate constraint data for logging
        if (should_log_constraints) {
            auto& entry = constraint_log_entries[i];
            entry.env_min_dist = rb_[i].env_min_dist_;
            entry.d_env_min_dist = rb_[i].d_env_min_dist_;
            entry.obs_radius = rb_[i].obs_radius_;
            entry.constraint_c = constr_info_k.c_vec.segment(si_index.con_envcol1, PANDA_NUM_LINKS);
            if (jac_constr_ineqp) {
                entry.jac_c_x = jac_constr_k.c_x.block(si_index.con_envcol1, 0, PANDA_NUM_LINKS, NX);
                entry.jac_c_u = jac_constr_k.c_u.block(si_index.con_envcol1, 0, PANDA_NUM_LINKS, NU);
                // A matrix에 실제 들어가는 스케일링된 값 (FP32 precision, same rounding mode)
#ifdef CONSTRAINTS_ROUNDING_MODE
                int old_round_log = std::fegetround();
                std::fesetround(CONSTRAINTS_ROUNDING_MODE);
#endif
                entry.jac_c_x_scaled = (entry.jac_c_x.cast<float>() * normalization_param_.T_x.cast<float>()).cast<double>();
                if (i != N)
                    entry.jac_c_u_scaled = (entry.jac_c_u.cast<float>() * normalization_param_.T_u.cast<float>()).cast<double>();
#ifdef CONSTRAINTS_ROUNDING_MODE
                std::fesetround(old_round_log);
#endif
            }
        }
    }

    // ===== Constraint logging: 파일 저장 =====
    if (should_log_constraints) {
        try {
            if (!g_constraint_dir_initialized) {
                system(("mkdir -p " + g_constraint_output_dir).c_str());
                g_constraint_dir_initialized = true;
                std::cout << "[Constraint Logging] Starting constraint logging to: " << g_constraint_output_dir << std::endl;
            }

            std::string filename = g_constraint_output_dir + "constraints_sample_" + std::to_string(current_solve_count_) + ".csv";
            std::string filename_bits = g_constraint_output_dir + "constraints_sample_" + std::to_string(current_solve_count_) + "_bits.csv";
            std::ofstream file(filename);
            std::ofstream file_bits(filename_bits);
            if (!file.is_open() || !file_bits.is_open()) {
                std::cerr << "[Constraint Logging ERROR] Failed to open: " << filename << std::endl;
            } else {
                file << std::scientific << std::setprecision(8);

                // ===== 실수 값 파일 헤더 =====
                file << "# EnvColl Constraint Results - Solve Count " << current_solve_count_ << "\n";
                file << "# SQP Iteration: " << sqp_iter_ << "\n";
                file << "# Horizon Points: " << (N + 1) << "\n";
                file << "# PANDA_NUM_LINKS: " << PANDA_NUM_LINKS << "\n";
                file << "# PANDA_DOF: " << PANDA_DOF << "\n";
                file << "# NX: " << NX << ", NU: " << NU << "\n\n";

                // ===== FP32 비트 파일 헤더 =====
                file_bits << "# EnvColl Constraint Results (FP32 Bits) - Solve Count " << current_solve_count_ << "\n";
                file_bits << "# SQP Iteration: " << sqp_iter_ << "\n";
                file_bits << "# Horizon Points: " << (N + 1) << "\n";
                file_bits << "# PANDA_NUM_LINKS: " << PANDA_NUM_LINKS << "\n";
                file_bits << "# PANDA_DOF: " << PANDA_DOF << "\n";
                file_bits << "# NX: " << NX << ", NU: " << NU << "\n";
                file_bits << "# Format: 8-digit hexadecimal (32-bit IEEE 754)\n\n";

                for (size_t h = 0; h <= N; ++h) {
                    const auto& entry = constraint_log_entries[h];

                    // MLP output: env_min_dist (per-link minimum distance)
                    file << "# HORIZON_" << h << "_ENV_MIN_DIST - " << PANDA_NUM_LINKS << " values\n";
                    file_bits << "# HORIZON_" << h << "_ENV_MIN_DIST - " << PANDA_NUM_LINKS << " values\n";
                    for (int j = 0; j < PANDA_NUM_LINKS; ++j) {
                        file << entry.env_min_dist(j);
                        file_bits << constraint_float_to_hex(static_cast<float>(entry.env_min_dist(j)));
                        if (j < PANDA_NUM_LINKS - 1) { file << ","; file_bits << ","; }
                    }
                    file << "\n\n"; file_bits << "\n\n";

                    // MLP Jacobian: d_env_min_dist (PANDA_NUM_LINKS x PANDA_DOF)
                    file << "# HORIZON_" << h << "_D_ENV_MIN_DIST - " << PANDA_NUM_LINKS << " x " << PANDA_DOF << "\n";
                    file_bits << "# HORIZON_" << h << "_D_ENV_MIN_DIST - " << PANDA_NUM_LINKS << " x " << PANDA_DOF << "\n";
                    for (int r = 0; r < PANDA_NUM_LINKS; ++r) {
                        for (int c = 0; c < PANDA_DOF; ++c) {
                            file << entry.d_env_min_dist(r, c);
                            file_bits << constraint_float_to_hex(static_cast<float>(entry.d_env_min_dist(r, c)));
                            if (c < PANDA_DOF - 1) { file << ","; file_bits << ","; }
                        }
                        file << "\n"; file_bits << "\n";
                    }
                    file << "\n"; file_bits << "\n";

                    // Obstacle radius
                    file << "# HORIZON_" << h << "_OBS_RADIUS\n";
                    file_bits << "# HORIZON_" << h << "_OBS_RADIUS\n";
                    file << entry.obs_radius << "\n\n";
                    file_bits << constraint_float_to_hex(static_cast<float>(entry.obs_radius)) << "\n\n";

                    // Constraint value c = -d_min_dist * dq + RBF (PANDA_NUM_LINKS values)
                    file << "# HORIZON_" << h << "_CONSTRAINT_C - " << PANDA_NUM_LINKS << " values\n";
                    file_bits << "# HORIZON_" << h << "_CONSTRAINT_C - " << PANDA_NUM_LINKS << " values\n";
                    for (int j = 0; j < PANDA_NUM_LINKS; ++j) {
                        file << entry.constraint_c(j);
                        file_bits << constraint_float_to_hex(static_cast<float>(entry.constraint_c(j)));
                        if (j < PANDA_NUM_LINKS - 1) { file << ","; file_bits << ","; }
                    }
                    file << "\n\n"; file_bits << "\n\n";

                    // Constraint Jacobian wrt state (PANDA_NUM_LINKS x NX)
                    if (entry.jac_c_x.size() > 0) {
                        file << "# HORIZON_" << h << "_JAC_C_X - " << entry.jac_c_x.rows() << " x " << entry.jac_c_x.cols() << "\n";
                        file_bits << "# HORIZON_" << h << "_JAC_C_X - " << entry.jac_c_x.rows() << " x " << entry.jac_c_x.cols() << "\n";
                        for (int r = 0; r < entry.jac_c_x.rows(); ++r) {
                            for (int c = 0; c < entry.jac_c_x.cols(); ++c) {
                                file << entry.jac_c_x(r, c);
                                file_bits << constraint_float_to_hex(static_cast<float>(entry.jac_c_x(r, c)));
                                if (c < entry.jac_c_x.cols() - 1) { file << ","; file_bits << ","; }
                            }
                            file << "\n"; file_bits << "\n";
                        }
                        file << "\n"; file_bits << "\n";
                    }

                    // Constraint Jacobian wrt input (PANDA_NUM_LINKS x NU)
                    if (entry.jac_c_u.size() > 0) {
                        file << "# HORIZON_" << h << "_JAC_C_U - " << entry.jac_c_u.rows() << " x " << entry.jac_c_u.cols() << "\n";
                        file_bits << "# HORIZON_" << h << "_JAC_C_U - " << entry.jac_c_u.rows() << " x " << entry.jac_c_u.cols() << "\n";
                        for (int r = 0; r < entry.jac_c_u.rows(); ++r) {
                            for (int c = 0; c < entry.jac_c_u.cols(); ++c) {
                                file << entry.jac_c_u(r, c);
                                file_bits << constraint_float_to_hex(static_cast<float>(entry.jac_c_u(r, c)));
                                if (c < entry.jac_c_u.cols() - 1) { file << ","; file_bits << ","; }
                            }
                            file << "\n"; file_bits << "\n";
                        }
                        file << "\n"; file_bits << "\n";
                    }

                    // Scaled Jacobian wrt state: JAC_C_X * T_x (A matrix에 실제 들어가는 값)
                    if (entry.jac_c_x_scaled.size() > 0) {
                        file << "# HORIZON_" << h << "_JAC_C_X_SCALED - " << entry.jac_c_x_scaled.rows() << " x " << entry.jac_c_x_scaled.cols() << " (= JAC_C_X * T_x, A matrix)\n";
                        file_bits << "# HORIZON_" << h << "_JAC_C_X_SCALED - " << entry.jac_c_x_scaled.rows() << " x " << entry.jac_c_x_scaled.cols() << " (= JAC_C_X * T_x, A matrix)\n";
                        for (int r = 0; r < entry.jac_c_x_scaled.rows(); ++r) {
                            for (int c = 0; c < entry.jac_c_x_scaled.cols(); ++c) {
                                file << entry.jac_c_x_scaled(r, c);
                                file_bits << constraint_float_to_hex(static_cast<float>(entry.jac_c_x_scaled(r, c)));
                                if (c < entry.jac_c_x_scaled.cols() - 1) { file << ","; file_bits << ","; }
                            }
                            file << "\n"; file_bits << "\n";
                        }
                        file << "\n"; file_bits << "\n";
                    }

                    // Scaled Jacobian wrt input: JAC_C_U * T_u (A matrix에 실제 들어가는 값, k!=N만)
                    if (entry.jac_c_u_scaled.size() > 0) {
                        file << "# HORIZON_" << h << "_JAC_C_U_SCALED - " << entry.jac_c_u_scaled.rows() << " x " << entry.jac_c_u_scaled.cols() << " (= JAC_C_U * T_u, A matrix)\n";
                        file_bits << "# HORIZON_" << h << "_JAC_C_U_SCALED - " << entry.jac_c_u_scaled.rows() << " x " << entry.jac_c_u_scaled.cols() << " (= JAC_C_U * T_u, A matrix)\n";
                        for (int r = 0; r < entry.jac_c_u_scaled.rows(); ++r) {
                            for (int c = 0; c < entry.jac_c_u_scaled.cols(); ++c) {
                                file << entry.jac_c_u_scaled(r, c);
                                file_bits << constraint_float_to_hex(static_cast<float>(entry.jac_c_u_scaled(r, c)));
                                if (c < entry.jac_c_u_scaled.cols() - 1) { file << ","; file_bits << ","; }
                            }
                            file << "\n"; file_bits << "\n";
                        }
                        file << "\n"; file_bits << "\n";
                    }
                }
                file.close();
                file_bits.close();
            }
        } catch (const std::exception& e) {
            std::cerr << "[Constraint Logging ERROR] Exception: " << e.what() << std::endl;
        }
    }
}

void OsqpInterface::setConstraints(const std::vector<OptVariables> &initial_guess,
                                   Eigen::MatrixXd *jac_constr, Eigen::VectorXd *constr, Eigen::VectorXd *l, Eigen::VectorXd *u)
{
    Eigen::MatrixXd jac_constr_eq, jac_constr_ineqb, jac_constr_ineqp;
    Eigen::VectorXd constr_eq, constr_ineqb, constr_ineqp;
    Eigen::VectorXd l_eq, u_eq, l_ineqb, u_ineqb, l_ineqp, u_ineqp;

    if(constr && !jac_constr)
    {
        setDynamics(initial_guess, NULL, &constr_eq, &l_eq, &u_eq);
        setBounds(initial_guess, NULL, &constr_ineqb, &l_ineqb, &u_ineqb);
        setPolytopicConstraints(initial_guess, NULL, &constr_ineqp, &l_ineqp, &u_ineqp);
    }
    else if(constr && jac_constr)
    {
        setDynamics(initial_guess, &jac_constr_eq, &constr_eq, &l_eq, &u_eq);
        setBounds(initial_guess, &jac_constr_ineqb, &constr_ineqb, &l_ineqb, &u_ineqb);
        setPolytopicConstraints(initial_guess, &jac_constr_ineqp, &constr_ineqp, &l_ineqp, &u_ineqp);
    }
    if(jac_constr)
    {
        jac_constr->block(0, 0, N_eq, N_var) = jac_constr_eq;
        jac_constr->block(N_eq, 0, N_ineqb, N_var) = jac_constr_ineqb;
        jac_constr->block(N_eq+N_ineqb, 0, N_ineqp, N_var) = jac_constr_ineqp;
    }
    if(constr)
    {
        constr->segment(0, N_eq) = constr_eq;
        constr->segment(N_eq, N_ineqb) = constr_ineqb;
        constr->segment(N_eq+N_ineqb, N_ineqp) = constr_ineqp;
    }
    if(l)
    {
        l->segment(0, N_eq) = l_eq;
        l->segment(N_eq, N_ineqb) = l_ineqb;
        l->segment(N_eq+N_ineqb, N_ineqp) = l_ineqp;
    }
    if(u)
    {
        u->segment(0, N_eq) = u_eq;
        u->segment(N_eq, N_ineqb) = u_ineqb;
        u->segment(N_eq+N_ineqb, N_ineqp) = u_ineqp;
    }

    // ===== EnvCol Scaling Analysis ===== (DISABLED for performance measurement)
    // if(jac_constr)
    // {
    //     static int analysis_count = 0;
    //     if(analysis_count % 100 == 0)
    //     {
    //         // ... analysis code ...
    //     }
    //     analysis_count++;
    // }
}

void OsqpInterface::setQP(const std::vector<OptVariables> &initial_guess,
                          Eigen::MatrixXd *hess_obj, Eigen::VectorXd *grad_obj, double *obj, Eigen::MatrixXd *jac_constr, Eigen::VectorXd *constr, Eigen::VectorXd *l, Eigen::VectorXd *u)
{
    // printf("Setting QP problem...\n");
    // if (hess_obj) {
    //     printf("Hessian is set\n");
    //     // (0,0) 원소 출력
    //     printf("hess_obj(0,0) = %g\n", (*hess_obj)(0,0));
    // } else {
    //     printf("Hessian is not set\n");
    // }

    setCost(initial_guess, obj, grad_obj, hess_obj);
    setConstraints(initial_guess, jac_constr, constr, l, u);
}

#ifdef ASIC_ENVCOL_ZERO_MODE
// Helper: write a c_float vector in both float and hex format
static void asic_write_vector(FILE* f_float, FILE* f_bits, const char* name,
                               const Eigen::Matrix<c_float, Eigen::Dynamic, 1>& v) {
    auto to_hex = [](c_float val) -> uint32_t {
        float fv = (float)val; uint32_t bits;
        std::memcpy(&bits, &fv, sizeof(float)); return bits;
    };
    const int len = v.size();
    fprintf(f_float, "# %s - %d values\n", name, len);
    fprintf(f_bits,  "# %s (hex FP32) - %d values\n", name, len);
    for(int i = 0; i < len; i++) {
        fprintf(f_float, "%.8e", (double)v(i));
        fprintf(f_bits,  "%08x", to_hex(v(i)));
        if(i < len-1) { fprintf(f_float, ","); fprintf(f_bits, ","); }
    }
    fprintf(f_float, "\n\n"); fprintf(f_bits, "\n\n");
}

// Helper: write a sparse matrix CSC in both formats
static void asic_write_csc(FILE* f_float, FILE* f_bits, const char* name,
                            Eigen::SparseMatrix<c_float>& sp, int rows, int cols) {
    auto to_hex = [](c_float val) -> uint32_t {
        float fv = (float)val; uint32_t bits;
        std::memcpy(&bits, &fv, sizeof(float)); return bits;
    };
    sp.makeCompressed();
    long long nnz = sp.nonZeros();
    // float
    fprintf(f_float, "# %s (CSC) - %d x %d, nnz=%lld\n", name, rows, cols, nnz);
    fprintf(f_float, "# col_ptr\n");
    for(int j = 0; j <= cols; j++) { fprintf(f_float, "%d", (int)sp.outerIndexPtr()[j]); if(j < cols) fprintf(f_float, ","); }
    fprintf(f_float, "\n# row_idx\n");
    for(int k = 0; k < nnz; k++) { fprintf(f_float, "%d", (int)sp.innerIndexPtr()[k]); if(k < nnz-1) fprintf(f_float, ","); }
    fprintf(f_float, "\n# values\n");
    for(int k = 0; k < nnz; k++) { fprintf(f_float, "%.8e", (double)sp.valuePtr()[k]); if(k < nnz-1) fprintf(f_float, ","); }
    fprintf(f_float, "\n\n");
    // bits
    fprintf(f_bits, "# %s (CSC, hex FP32) - %d x %d, nnz=%lld\n", name, rows, cols, nnz);
    fprintf(f_bits, "# col_ptr\n");
    for(int j = 0; j <= cols; j++) { fprintf(f_bits, "%d", (int)sp.outerIndexPtr()[j]); if(j < cols) fprintf(f_bits, ","); }
    fprintf(f_bits, "\n# row_idx\n");
    for(int k = 0; k < nnz; k++) { fprintf(f_bits, "%d", (int)sp.innerIndexPtr()[k]); if(k < nnz-1) fprintf(f_bits, ","); }
    fprintf(f_bits, "\n# values (hex FP32)\n");
    for(int k = 0; k < nnz; k++) { fprintf(f_bits, "%08x", to_hex(sp.valuePtr()[k])); if(k < nnz-1) fprintf(f_bits, ","); }
    fprintf(f_bits, "\n\n");
}

// Helper: write a dense matrix in both formats
static void asic_write_dense(FILE* f_float, FILE* f_bits, const char* name,
                              const Eigen::MatrixXf& dense) {
    auto to_hex = [](float val) -> uint32_t {
        uint32_t bits; std::memcpy(&bits, &val, sizeof(float)); return bits;
    };
    const int rows = dense.rows(), cols = dense.cols();
    fprintf(f_float, "# %s (dense) - %d x %d\n", name, rows, cols);
    fprintf(f_bits,  "# %s (dense, hex FP32) - %d x %d\n", name, rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            fprintf(f_float, "%.8e", (double)dense(i,j));
            fprintf(f_bits,  "%08x", to_hex(dense(i,j)));
            if(j < cols-1) { fprintf(f_float, ","); fprintf(f_bits, ","); }
        }
        fprintf(f_float, "\n"); fprintf(f_bits, "\n");
    }
    fprintf(f_float, "\n"); fprintf(f_bits, "\n");
}

void OsqpInterface::saveAsicE2EInput(
    const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
    const Eigen::MatrixXd &A, const Eigen::VectorXd &l, const Eigen::VectorXd &u,
    const std::vector<OptVariables> &initial_guess, int sample_id)
{
    const std::string asic_dir = resolve_asic_e2e_output_dir();
    if (!g_asic_dir_initialized) {
        system(("mkdir -p " + asic_dir).c_str());
        g_asic_dir_initialized = true;
    }

    // Create modified copies (original data untouched)
    Eigen::MatrixXd A_mod = A;
    Eigen::VectorXd u_mod = u;

    for(int i = 0; i <= N; i++)
    {
        const int row_start = N_eq + N_ineqb + NPC * i + si_index.con_envcol1;
        A_mod.block(row_start, 0, PANDA_NUM_LINKS, N_var).setZero();
        u_mod.segment(row_start, PANDA_NUM_LINKS).setZero();
        if(i != N)
        {
            for(int j = 0; j < PANDA_DOF; j++)
                u_mod(row_start + j) = initial_guess[i].uk.get_dq(j);
        }
    }

    const int n = N_var, m = N_constr;

    // Reverse permute: v[i] -> v[n-1-i], M[i][j] -> M[rows-1-i][cols-1-j]
    // MLP inputs are NOT permuted; only OSQP matrices are.
    Eigen::MatrixXd P_rev(n, n), A_rev(m, n);
    Eigen::VectorXd q_rev(n), l_rev(m), u_rev(m);
    for(int i = 0; i < n; i++) {
        q_rev(i) = q(n - 1 - i);
        for(int j = 0; j < n; j++)
            P_rev(i, j) = P(n - 1 - i, n - 1 - j);
    }
    for(int i = 0; i < m; i++) {
        l_rev(i) = l(m - 1 - i);
        u_rev(i) = u_mod(m - 1 - i);
        for(int j = 0; j < n; j++)
            A_rev(i, j) = A_mod(m - 1 - i, n - 1 - j);
    }

    // Cast to c_float
    Eigen::SparseMatrix<c_float> P_sp = P_rev.cast<c_float>().sparseView();
    Eigen::SparseMatrix<c_float> A_sp = A_rev.cast<c_float>().sparseView();
    Eigen::Matrix<c_float, Eigen::Dynamic, 1> q_cf = q_rev.cast<c_float>();
    Eigen::Matrix<c_float, Eigen::Dynamic, 1> l_cf = l_rev.cast<c_float>();
    Eigen::Matrix<c_float, Eigen::Dynamic, 1> u_cf = u_rev.cast<c_float>();

    char path_f[512], path_b[512];
    snprintf(path_f, sizeof(path_f), "%s/sample_%d_e2e_input.csv", asic_dir.c_str(), sample_id);
    snprintf(path_b, sizeof(path_b), "%s/sample_%d_e2e_input_bits.csv", asic_dir.c_str(), sample_id);
    FILE* ff = fopen(path_f, "w");
    FILE* fb = fopen(path_b, "w");
    if(!ff || !fb) { if(ff) fclose(ff); if(fb) fclose(fb); return; }

    fprintf(ff, "# ASIC End-to-End Input - Sample %d\n", sample_id);
    fprintf(ff, "# n=%d, m=%d, N=%d\n\n", n, m, (int)N);
    fprintf(fb, "# ASIC End-to-End Input (hex FP32) - Sample %d\n", sample_id);
    fprintf(fb, "# n=%d, m=%d, N=%d\n\n", n, m, (int)N);

    // ===== 1. MLP NeRF inputs per horizon step =====
    for(int i = 0; i <= N; i++)
    {
        // Reconstruct MLP input: [q1..q7, obs_x, obs_y, obs_z]
        Eigen::VectorXf input_f = Eigen::VectorXf(PANDA_DOF + 3);
        input_f.head(PANDA_DOF) = rb_[i].q_.cast<float>();
        input_f.tail(3) = rb_[i].obs_position_.cast<float>();

        // NeRF encoding: [x, sin(x), cos(x)]
        Eigen::VectorXf nerf_f(3 * (PANDA_DOF + 3));
        nerf_f.segment(0 * (PANDA_DOF + 3), PANDA_DOF + 3) = input_f;
        nerf_f.segment(1 * (PANDA_DOF + 3), PANDA_DOF + 3) = input_f.array().sin();
        nerf_f.segment(2 * (PANDA_DOF + 3), PANDA_DOF + 3) = input_f.array().cos();

        char label[64];
        snprintf(label, sizeof(label), "HORIZON_%d_MLP_NERF_INPUT", i);
        Eigen::Matrix<c_float, Eigen::Dynamic, 1> nerf_cf = nerf_f.cast<c_float>();
        asic_write_vector(ff, fb, label, nerf_cf);
    }

    // ===== 2. OSQP matrices (EnvCol zeroed, dq in u) =====
    // P_csc
    asic_write_csc(ff, fb, "P_csc", P_sp, n, n);
    // A_csc
    asic_write_csc(ff, fb, "A_csc", A_sp, m, n);

    // P dense (symmetrized)
    Eigen::MatrixXf P_dense = Eigen::MatrixXf::Zero(n, n);
    for(int j = 0; j < n; j++)
        for(Eigen::SparseMatrix<c_float>::InnerIterator it(P_sp, j); it; ++it)
            P_dense(it.row(), j) = it.value();
    for(int i = 0; i < n; i++)
        for(int j = i+1; j < n; j++)
            if(P_dense(i,j) == 0.f && P_dense(j,i) != 0.f) P_dense(i,j) = P_dense(j,i);
    asic_write_dense(ff, fb, "P", P_dense);

    // A dense
    Eigen::MatrixXf A_dense = Eigen::MatrixXf::Zero(m, n);
    for(int j = 0; j < n; j++)
        for(Eigen::SparseMatrix<c_float>::InnerIterator it(A_sp, j); it; ++it)
            A_dense(it.row(), j) = it.value();
    asic_write_dense(ff, fb, "A", A_dense);

    asic_write_vector(ff, fb, "q", q_cf);
    asic_write_vector(ff, fb, "l", l_cf);
    asic_write_vector(ff, fb, "u", u_cf);

    // D_exp, E_exp (all zeros)
    fprintf(ff, "# D_exp - %d values\n", n);
    fprintf(fb, "# D_exp - %d values\n", n);
    for(int i = 0; i < n; i++) { fprintf(ff, "0"); fprintf(fb, "0"); if(i<n-1) { fprintf(ff, ","); fprintf(fb, ","); } }
    fprintf(ff, "\n\n"); fprintf(fb, "\n\n");
    fprintf(ff, "# E_exp - %d values\n", m);
    fprintf(fb, "# E_exp - %d values\n", m);
    for(int i = 0; i < m; i++) { fprintf(ff, "0"); fprintf(fb, "0"); if(i<m-1) { fprintf(ff, ","); fprintf(fb, ","); } }
    fprintf(ff, "\n\n"); fprintf(fb, "\n\n");

    // ===== 3. Parameters (obs_radius, tol_envcol) =====
    fprintf(ff, "# obs_radius - 1 value\n%.8e\n\n", rb_[0].obs_radius_);
    fprintf(fb, "# obs_radius (hex FP32) - 1 value\n");
    { float v = (float)rb_[0].obs_radius_; uint32_t b; std::memcpy(&b, &v, sizeof(float)); fprintf(fb, "%08x\n\n", b); }

    fclose(ff);
    fclose(fb);

    // ===== after_envcoll: original A, u with EnvCol applied, reverse permuted =====
    system(("mkdir -p " + asic_dir + "/after_envcoll").c_str());

    // Reverse permute the original (non-zeroed) A and u
    Eigen::VectorXd u_orig_rev(m);
    Eigen::MatrixXd A_orig_rev(m, n);
    for(int i = 0; i < m; i++) {
        u_orig_rev(i) = u(m - 1 - i);
        for(int j = 0; j < n; j++)
            A_orig_rev(i, j) = A(m - 1 - i, n - 1 - j);
    }

    Eigen::SparseMatrix<c_float> A_orig_sp = A_orig_rev.cast<c_float>().sparseView();
    Eigen::Matrix<c_float, Eigen::Dynamic, 1> u_orig_cf = u_orig_rev.cast<c_float>();

    char path_af[512], path_ab[512];
    snprintf(path_af, sizeof(path_af), "%s/after_envcoll/sample_%d_after_envcoll.csv", asic_dir.c_str(), sample_id);
    snprintf(path_ab, sizeof(path_ab), "%s/after_envcoll/sample_%d_after_envcoll_bits.csv", asic_dir.c_str(), sample_id);
    FILE* af = fopen(path_af, "w");
    FILE* ab = fopen(path_ab, "w");
    if(!af || !ab) { if(af) fclose(af); if(ab) fclose(ab); return; }

    fprintf(af, "# After EnvColl - reverse permuted (verification) - Sample %d\n", sample_id);
    fprintf(af, "# n=%d, m=%d\n\n", n, m);
    fprintf(ab, "# After EnvColl - reverse permuted (hex FP32) - Sample %d\n", sample_id);
    fprintf(ab, "# n=%d, m=%d\n\n", n, m);

    // P, q, l are identical to e2e_input (EnvCol doesn't affect them)
    asic_write_csc(af, ab, "P_csc", P_sp, n, n);
    asic_write_csc(af, ab, "A_csc", A_orig_sp, m, n);
    asic_write_dense(af, ab, "P", P_dense);

    Eigen::MatrixXf A_orig_dense = Eigen::MatrixXf::Zero(m, n);
    for(int j = 0; j < n; j++)
        for(Eigen::SparseMatrix<c_float>::InnerIterator it(A_orig_sp, j); it; ++it)
            A_orig_dense(it.row(), j) = it.value();
    asic_write_dense(af, ab, "A", A_orig_dense);

    asic_write_vector(af, ab, "q", q_cf);
    asic_write_vector(af, ab, "l", l_cf);
    asic_write_vector(af, ab, "u", u_orig_cf);

    fclose(af);
    fclose(ab);
}

void OsqpInterface::saveAsicE2EOutput(
    const Eigen::VectorXd &step, const Eigen::VectorXd &step_lambda, int sample_id)
{
    const std::string asic_dir = resolve_asic_e2e_output_dir();
    if (!g_asic_dir_initialized) {
        system(("mkdir -p " + asic_dir).c_str());
        g_asic_dir_initialized = true;
    }

    // Reverse permute: v[i] -> v[n-1-i] to match ASIC reversed index space
    const int n = step.size();
    const int m = step_lambda.size();
    Eigen::Matrix<c_float, Eigen::Dynamic, 1> x_cf(n), y_cf(m);
    for(int i = 0; i < n; i++) x_cf(i) = static_cast<c_float>(step(n - 1 - i));
    for(int i = 0; i < m; i++) y_cf(i) = static_cast<c_float>(step_lambda(m - 1 - i));

    char path_f[512], path_b[512];
    snprintf(path_f, sizeof(path_f), "%s/sample_%d_e2e_output.csv", asic_dir.c_str(), sample_id);
    snprintf(path_b, sizeof(path_b), "%s/sample_%d_e2e_output_bits.csv", asic_dir.c_str(), sample_id);
    FILE* ff = fopen(path_f, "w");
    FILE* fb = fopen(path_b, "w");
    if(!ff || !fb) { if(ff) fclose(ff); if(fb) fclose(fb); return; }

    fprintf(ff, "# ASIC End-to-End Golden Output - Sample %d\n", sample_id);
    fprintf(ff, "# x (primal) size=%d, y (dual) size=%d\n\n", (int)x_cf.size(), (int)y_cf.size());
    fprintf(fb, "# ASIC End-to-End Golden Output (hex FP32) - Sample %d\n", sample_id);
    fprintf(fb, "# x (primal) size=%d, y (dual) size=%d\n\n", (int)x_cf.size(), (int)y_cf.size());

    asic_write_vector(ff, fb, "x", x_cf);
    asic_write_vector(ff, fb, "y", y_cf);

    fclose(ff);
    fclose(fb);
}
#endif

bool OsqpInterface::solveOCP(std::vector<OptVariables> &opt_sol, Status *status, ComputeTime *mpc_time, int &iter_count, int &sqp_iter_count, int &total_iter_count, int solve_count)
{
    auto start_total = std::chrono::high_resolution_clock::now();

    // Store solve_count for use in solveQP
    current_solve_count_ = solve_count;

    // Initialize
    lambda_.setZero(N_constr);
    step_.setZero(N_var);
    step_prev_.setZero(N_var);
    step_lambda_.setZero(N_constr);

    grad_L_.setZero(N_var);
    delta_grad_L_.setZero(N_var);

    Hess_.setZero(N_var, N_var);
    grad_obj_.setZero(N_var);
    jac_constr_.setZero(N_constr, N_var);
    constr_.setZero(N_constr);
    l_.setZero(N_constr);
    u_.setZero(N_constr);

    filter_data_list_.clear();

    mpc_time->setZero();
    total_iter_count = 0;

    std::vector<OptVariables> zero_guess;
    zero_guess.resize(N+1);
    for(size_t i=0; i<=N; i++)
    {
        zero_guess[i].xk = initial_guess_[0].xk;
        zero_guess[i].uk.setZero();
    }

    // printf("sqp_param_.max_iter %d\n", sqp_param_.max_iter);

    // RTI (Real-Time Iteration) mode: single QP solve without convergence check
    if(sqp_param_.use_RTI)
    {
        sqp_iter_ = 0;

        auto start_set_qp = std::chrono::high_resolution_clock::now();

        // QP formulation
        setQP(initial_guess_, &Hess_, &grad_obj_, &obj_, &jac_constr_, &constr_, &l_, &u_);

        // Hessian check
        if (!isPosdef(Hess_))
        {
            std::cout << "[RTI] Hessian not positive definite\n";
            if (g_solveQP_fail_log.is_open()) {
                g_solveQP_fail_log << "[RTI_NON_PD_HESSIAN] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
                g_solveQP_fail_log.flush();
            }
            // Record timing even on failure
            auto end_set_qp_fail = std::chrono::high_resolution_clock::now();
            mpc_time->set_qp = std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp_fail - start_set_qp).count();
            auto end_total_fail = std::chrono::high_resolution_clock::now();
            mpc_time->total = std::chrono::duration_cast<std::chrono::duration<double>>(end_total_fail - start_total).count();
            (*status) = NON_PD_HESSIAN;
            return false;
        }
        if (isNan(Hess_))
        {
            std::cout << "[RTI] Hessian is NaN\n";
            if (g_solveQP_fail_log.is_open()) {
                g_solveQP_fail_log << "[RTI_NAN_HESSIAN] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
                g_solveQP_fail_log.flush();
            }
            // Record timing even on failure
            auto end_set_qp_fail = std::chrono::high_resolution_clock::now();
            mpc_time->set_qp = std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp_fail - start_set_qp).count();
            auto end_total_fail = std::chrono::high_resolution_clock::now();
            mpc_time->total = std::chrono::duration_cast<std::chrono::duration<double>>(end_total_fail - start_total).count();
            (*status) = NAN_HESSIAN;
            return false;
        }

        auto end_set_qp = std::chrono::high_resolution_clock::now();
        auto start_solve_qp = std::chrono::high_resolution_clock::now();

#ifdef ASIC_ENVCOL_ZERO_MODE
        if(current_solve_count_ >= 0 && current_solve_count_ % ASIC_TESTCASE_SAVE_INTERVAL == 0)
            saveAsicE2EInput(Hess_, grad_obj_, jac_constr_, l_-constr_, u_-constr_, initial_guess_, current_solve_count_);
#endif

        // Solve QP
        if(!solveQP(Hess_, grad_obj_, jac_constr_, l_-constr_, u_-constr_, step_, step_lambda_, qp_status_, iter_count))
        {
            total_iter_count += iter_count;
            printf("[RTI] QP solve FAILED : %d \n", qp_status_);
            if (g_solveQP_fail_log.is_open()) {
                g_solveQP_fail_log << "[RTI_solveQP_FAIL] " << current_solve_count_ << ", " << sqp_iter_
                                  << ", qp_status=" << static_cast<int>(qp_status_) << std::endl;
                g_solveQP_fail_log.flush();
            }
            // Record timing even on failure
            mpc_time->set_qp = std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp - start_set_qp).count();
            mpc_time->init_solver = last_init_solver_time_;
            mpc_time->solve_qp = last_solve_time_ + last_factorization_time_;  // ADMM + factorization
            mpc_time->scaling_time = last_scaling_time_;
            mpc_time->permutation_time = last_permutation_time_;
            mpc_time->factorization_time = last_factorization_time_;
            mpc_time->rho_updates = last_rho_updates_;
            auto end_total_fail = std::chrono::high_resolution_clock::now();
            mpc_time->total = std::chrono::duration_cast<std::chrono::duration<double>>(end_total_fail - start_total).count();
            switch (qp_status_)
            {
            case OsqpEigen::Status::DualInfeasibleInaccurate:
                (*status) = QP_DualInfeasibleInaccurate; break;
            case OsqpEigen::Status::PrimalInfeasibleInaccurate:
                (*status) = QP_PrimalInfeasibleInaccurate; break;
            case OsqpEigen::Status::SolvedInaccurate:
                (*status) = QP_SolvedInaccurate; break;
            case OsqpEigen::Status::MaxIterReached:
                (*status) = QP_MaxIterReached; break;
            case OsqpEigen::Status::PrimalInfeasible:
                (*status) = QP_PrimalInfeasible; break;
            case OsqpEigen::Status::DualInfeasible:
                (*status) = QP_DualInfeasible; break;
            case OsqpEigen::Status::Sigint:
                (*status) = Sigint; break;
            }
            return false;
        }

        total_iter_count += iter_count;

#ifdef ASIC_ENVCOL_ZERO_MODE
        if(current_solve_count_ >= 0 && current_solve_count_ % ASIC_TESTCASE_SAVE_INTERVAL == 0)
            saveAsicE2EOutput(step_, step_lambda_, current_solve_count_);
#endif

        auto end_solve_qp = std::chrono::high_resolution_clock::now();

        // Apply step with alpha=1 (no line search)
        initial_guess_vec_ += deNormalizeStep(step_);
        initial_guess_ = vectorToOptvar(initial_guess_vec_);

        // Update timing
        mpc_time->set_qp = std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp - start_set_qp).count();
        mpc_time->init_solver = last_init_solver_time_;  // initSolver time (scaling + permutation + factorization)
        mpc_time->solve_qp = last_solve_time_ + last_factorization_time_;  // ADMM + factorization time
        mpc_time->scaling_time = last_scaling_time_;
        mpc_time->permutation_time = last_permutation_time_;
        mpc_time->factorization_time = last_factorization_time_;
        mpc_time->rho_updates = last_rho_updates_;
        mpc_time->get_alpha = 0.0;  // No line search in RTI

        auto end_total = std::chrono::high_resolution_clock::now();
        mpc_time->total = std::chrono::duration_cast<std::chrono::duration<double>>(end_total - start_total).count();

        sqp_iter_count = 1;
        (*status) = SOLVED;
        opt_sol = initial_guess_;
        // printf("[RTI] SOLVED | init_solver: %.6f, solve_qp: %.6f\n", mpc_time->init_solver, mpc_time->solve_qp);
        return true;
    }

    // SQP itertion (non-RTI mode)
    for(sqp_iter_=0; sqp_iter_<sqp_param_.max_iter; sqp_iter_++)
    {
        // std::cout <<"sqp_iter_: " <<sqp_iter_<<std::endl;

        auto start_set_qp = std::chrono::high_resolution_clock::now();

        // QP formulation
        if(sqp_param_.use_BFGS)
        {
            // printf("Using BFGS update\n");
            if(sqp_iter_ == 0) setQP(initial_guess_,&Hess_, &grad_obj_, &obj_, &jac_constr_, &constr_, &l_, &u_);
            else setQP(initial_guess_,NULL, &grad_obj_, &obj_, &jac_constr_, &constr_, &l_, &u_);
        }
        else
        {
            // printf("Not using BFGS update\n");
            setQP(initial_guess_,&Hess_, &grad_obj_, &obj_, &jac_constr_, &constr_, &l_, &u_);
        }

        delta_grad_L_ = -grad_L_;
        grad_L_ = grad_obj_ + jac_constr_.transpose() * lambda_;
        delta_grad_L_ += grad_L_;  // delta_grad_L_ = grad_L - grad_L_prev

        // get Hessian
        if(sqp_param_.use_BFGS && sqp_iter_ != 0)  Hess_ = BFGSUpdate(Hess_, step_prev_, delta_grad_L_);
        
        // for (int i = 0; i < Hess_.rows(); ++i)
        // {
        //     std::cout << "Hess_(" << i << "," << i << ") = " << Hess_(i, i) << std::endl;
        // }

        // // Debug Hessian
        // printAllEigenvalues(Hess_);

        // if(sqp_iter_ == 0) {
        //     saveMatrixToFile(Hess_, "Hess_", "/home/mms-wonchan/Studies/OSQP/Precision/first_hessian.txt");
        // }
        
        if (!isPosdef(Hess_))
        {
            std::cout << "Hessian not positive definite\n";
            if (g_solveQP_fail_log.is_open()) {
                g_solveQP_fail_log << "[SQP_NON_PD_HESSIAN] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
                g_solveQP_fail_log.flush();
            }
            // Record timing even on failure
            auto end_set_qp_fail = std::chrono::high_resolution_clock::now();
            mpc_time->set_qp += std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp_fail - start_set_qp).count();
            // double tau = 1e-3;
            // Eigen::VectorXd v(N_var);
            // while (!isPosdef(Hess_))
            // {
            //     v.setConstant(tau);
            //     Hess_ += v.asDiagonal();
            //     tau *= 10;
            // }
            (*status) = NON_PD_HESSIAN;
            break;
        }
        if (isNan(Hess_))
        {
            std::cout << "Hessian is NaN\n";
            if (g_solveQP_fail_log.is_open()) {
                g_solveQP_fail_log << "[SQP_NAN_HESSIAN] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
                g_solveQP_fail_log.flush();
            }
            // Record timing even on failure
            auto end_set_qp_fail = std::chrono::high_resolution_clock::now();
            mpc_time->set_qp += std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp_fail - start_set_qp).count();
            (*status) = NAN_HESSIAN;
            break;
        }

        auto end_set_qp = std::chrono::high_resolution_clock::now();
        auto start_solve_qp = std::chrono::high_resolution_clock::now();

#ifdef ASIC_ENVCOL_ZERO_MODE
        if(sqp_iter_ == 0 && current_solve_count_ >= 0 && current_solve_count_ % ASIC_TESTCASE_SAVE_INTERVAL == 0)
            saveAsicE2EInput(Hess_, grad_obj_, jac_constr_, l_-constr_, u_-constr_, initial_guess_, current_solve_count_);
#endif

        // solve QP to get step_ and step_lambda_
        if(!solveQP(Hess_, grad_obj_, jac_constr_, l_-constr_, u_-constr_, step_, step_lambda_, qp_status_, iter_count))
        {
            total_iter_count += iter_count;  // Accumulate QP iterations
            switch (qp_status_)
            {
            case OsqpEigen::Status::DualInfeasibleInaccurate:
                (*status) = QP_DualInfeasibleInaccurate;
                break;
            case OsqpEigen::Status::PrimalInfeasibleInaccurate:
                (*status) = QP_PrimalInfeasibleInaccurate;
                break;
            case OsqpEigen::Status::SolvedInaccurate:
                (*status) = QP_SolvedInaccurate;
                break;
            case OsqpEigen::Status::MaxIterReached:
                (*status) = QP_MaxIterReached;
                break;
            case OsqpEigen::Status::PrimalInfeasible:
                (*status) = QP_PrimalInfeasible;
                break;
            case OsqpEigen::Status::DualInfeasible:
                (*status) = QP_DualInfeasible;
                break;
            case OsqpEigen::Status::Sigint:
                (*status) = Sigint;
                break;
            }
        }
        else
        {
            // Accumulate QP iterations for successful solve
            total_iter_count += iter_count;
#ifdef ASIC_ENVCOL_ZERO_MODE
            if(sqp_iter_ == 0 && current_solve_count_ >= 0 && current_solve_count_ % ASIC_TESTCASE_SAVE_INTERVAL == 0)
                saveAsicE2EOutput(step_, step_lambda_, current_solve_count_);
#endif
        }

        // printf("qp_status_: %d\n", qp_status_);
        // printf("*status: %d\n", (*status));

        if(sqp_param_.do_SOC)
        {
            if(!SecondOrderCorrection(initial_guess_,Hess_,grad_obj_,jac_constr_,step_,step_lambda_, qp_status_))
            {
                switch (qp_status_)
                {
                case OsqpEigen::Status::DualInfeasibleInaccurate:
                    (*status) = QP_DualInfeasibleInaccurate;
                    break;
                case OsqpEigen::Status::PrimalInfeasibleInaccurate:
                    (*status) = QP_PrimalInfeasibleInaccurate;
                    break;
                case OsqpEigen::Status::SolvedInaccurate:
                    (*status) = QP_SolvedInaccurate;
                    break;
                case OsqpEigen::Status::MaxIterReached:
                    (*status) = QP_MaxIterReached;
                    break;
                case OsqpEigen::Status::PrimalInfeasible:
                    (*status) = QP_PrimalInfeasible;
                    break;
                case OsqpEigen::Status::DualInfeasible:
                    (*status) = QP_DualInfeasible;
                    break;
                case OsqpEigen::Status::Sigint:
                    (*status) = Sigint;
                    break;
                }
            }
        }



        auto end_solve_qp = std::chrono::high_resolution_clock::now();
        auto start_get_alpha = std::chrono::high_resolution_clock::now();

        step_lambda_ -= lambda_;

        // double alpha = meritLineSearch(step_, Hess_, grad_obj_, obj_, constr_, l_, u_);
        double alpha = filterLineSearch(initial_guess_,step_,filter_data_list_);
        // std:cout << "\talpha: "<<alpha << std::endl;

        auto end_get_alpha = std::chrono::high_resolution_clock::now();

        // take step
        initial_guess_vec_ += alpha*deNormalizeStep(step_);
        lambda_ += alpha*step_lambda_;
        initial_guess_ = vectorToOptvar(initial_guess_vec_);
        // printOptVar( initial_guess_);

        // update step info
        step_prev_ = alpha * step_;
        primal_step_norm_ = alpha * step_.template lpNorm<Eigen::Infinity>();
        // // Fixed : step_이 정규화된 공간에서의 변화량이므로, 이를 비정규화된 공간으로 변환하여 노름을 계산해야 함.
        // const auto dx_denorm = deNormalizeStep(step_); // 적용과 동일한 공간
        // const double alpha_abs = std::abs(alpha);
        // primal_step_norm_ = (alpha_abs * dx_denorm).template lpNorm<Eigen::Infinity>();
        // primal_step_norm_ = (alpha * step_).norm();
        dual_step_norm_ = alpha * step_lambda_.template lpNorm<Eigen::Infinity>();
        // std::cout << "\tprimal_step_norm_: " << primal_step_norm_ << std::endl;
        // std::cout << "\tdual_step_norm_: " << dual_step_norm_ << std::endl;

        mpc_time->set_qp += std::chrono::duration_cast<std::chrono::duration<double>>(end_set_qp - start_set_qp).count();
        mpc_time->init_solver += last_init_solver_time_;  // initSolver time (scaling + permutation + factorization)
        mpc_time->solve_qp += last_solve_time_ + last_factorization_time_;  // ADMM + factorization time
        mpc_time->scaling_time += last_scaling_time_;
        mpc_time->permutation_time += last_permutation_time_;
        mpc_time->factorization_time += last_factorization_time_;
        mpc_time->rho_updates += last_rho_updates_;
        mpc_time->get_alpha += std::chrono::duration_cast<std::chrono::duration<double>>(end_get_alpha - start_get_alpha).count();

        // termination condition
        // TODO: critertion about constraint
        // if(primal_step_norm_ < sqp_param_.eps_prim && dual_step_norm_ < sqp_param_.eps_dual)
        if(primal_step_norm_ < sqp_param_.eps_prim)
        {
            (*status) = SOLVED;
            break;
        }
    }
    if(sqp_iter_ == sqp_param_.max_iter) (*status) = MAX_ITER_EXCEEDED;

    auto end_total = std::chrono::high_resolution_clock::now();
    mpc_time->total = std::chrono::duration_cast<std::chrono::duration<double>>(end_total - start_total).count();

    sqp_iter_count = sqp_iter_ + 1;
    printf("[SQP] sqp_iter_=%d, sqp_iter_count=%d, total_iter_count=%d\n", sqp_iter_, sqp_iter_count, total_iter_count);

    if((*status) == SOLVED)
    {
        opt_sol = initial_guess_;
        return true;
    }
    else
    {
        opt_sol = zero_guess;
        return false;
    }
}

// 행렬의 각 요소가 float로 변환 가능한지 검사하는 템플릿 함수
template <typename Derived>
void checkForFloatOverflow(const Eigen::MatrixBase<Derived>& M, const std::string &matrixName)
{
    const auto float_max = std::numeric_limits<float>::max();
    bool overflowFound = false;
    
    for (int i = 0; i < M.rows(); ++i)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            if (std::abs(M(i, j)) > float_max)
            {
                std::cout << "Warning: " << matrixName << "(" << i << ", " << j 
                          << ") = " << M(i, j) 
                          << " exceeds float representable range (" << float_max << ")." 
                          << std::endl;
                overflowFound = true;
            }
        }
    }
    
    if (!overflowFound)
    {
        std::cout << "No overflow detected in " << matrixName << "." << std::endl;
    }
}


bool OsqpInterface::solveQP(const Eigen::MatrixXd &P, const Eigen::VectorXd &q, const Eigen::MatrixXd &A, const Eigen::VectorXd &l,const Eigen::VectorXd &u,
                            Eigen::VectorXd &step, Eigen::VectorXd &step_lambda, OsqpEigen::Status &qp_status, int &iter_count)
{
    /*
    min   1/2 x' P x + q' x
     x

    subject to
    l <= A x <= u

    with :
    P sparse (n x n) positive definite
    q dense  (n x 1)
    A sparse (nc x n)
    l dense  (nc x 1)
    u dense  (nc x 1)
    */

    // saveMatrixToFile(l, "lower", "/home/mms-wonchan/Studies/OSQP/Precision/lower.txt");
    // saveMatrixToFile(u, "lower", "/home/mms-wonchan/Studies/OSQP/Precision/upper.txt");

   // Use c_float precision for OSQP (c_float = float when OSQP_USE_FLOAT=ON, double otherwise)
   Eigen::SparseMatrix<c_float> P_sp(N_var, N_var);
   Eigen::SparseMatrix<c_float> A_sp(N_constr, N_var);
   Eigen::Matrix<c_float, N_var,1> q_ds;
   Eigen::Matrix<c_float, N_constr,1> l_ds;
   Eigen::Matrix<c_float, N_constr,1> u_ds;
   P_sp = P.cast<c_float>().sparseView();
   A_sp = A.cast<c_float>().sparseView();
   q_ds = q.cast<c_float>();
   l_ds = l.cast<c_float>();
   u_ds = u.cast<c_float>();

//    saveSparseMatrixToFile(P_sp, "P_sp", "./P_sp.txt");
//    saveSparseMatrixToFile(A_sp, "A_sp", "./A_sp.txt");
//    saveMatrixToFile(q_ds, "q_ds", "./q_ds.txt");
//    saveMatrixToFile(l_ds, "l_ds", "./l_ds.txt");
//    saveMatrixToFile(u_ds, "u_ds", "./u_ds.txt");

    // saveMatrixToFile(l_ds, "l_ds", "/home/mms-wonchan/Studies/OSQP/Precision/l_ds.txt");
    // saveMatrixToFile(u_ds, "u_ds", "/home/mms-wonchan/Studies/OSQP/Precision/u_ds.txt");

    OsqpEigen::Solver solver_;
    // settings
    solver_.settings()->setWarmStart(false);
    solver_.settings()->setMaxIteration(250);
    solver_.settings()->getSettings()->eps_abs = 9.765625e-04f;    // 2^(-10), 0x3A800000, was 1e-3
    solver_.settings()->getSettings()->eps_rel = 1.220703125e-04f; // 2^(-13), 0x39000000, was 1e-4
    solver_.settings()->getSettings()->verbose = false;

    // set the initial data of the QP solver
    auto start_init = std::chrono::high_resolution_clock::now();
    solver_.data()->setNumberOfVariables(N_var);
    solver_.data()->setNumberOfConstraints(N_constr);
    if (!solver_.data()->setHessianMatrix(P_sp)) {
        printf("[solveQP FAIL] setHessianMatrix failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[setHessianMatrix] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }
    if (!solver_.data()->setGradient(q_ds)) {
        printf("[solveQP FAIL] setGradient failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[setGradient] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }
    if (!solver_.data()->setLinearConstraintsMatrix(A_sp)) {
        printf("[solveQP FAIL] setLinearConstraintsMatrix failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[setLinearConstraintsMatrix] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }
    if (!solver_.data()->setLowerBound(l_ds)) {
        printf("[solveQP FAIL] setLowerBound failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[setLowerBound] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }
    if (!solver_.data()->setUpperBound(u_ds)) {
        printf("[solveQP FAIL] setUpperBound failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[setUpperBound] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }

    // instantiate the solver
    if (!solver_.initSolver()) {
        printf("[solveQP FAIL] initSolver failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[initSolver] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }
    auto end_init = std::chrono::high_resolution_clock::now();
    last_init_solver_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(end_init - start_init).count();

    // Get detailed timing breakdown from OSQPInfo
    const OSQPInfo* osqp_info = solver_.getInfo();
    if (osqp_info != nullptr) {
        last_scaling_time_ = osqp_info->scaling_time;
        last_permutation_time_ = osqp_info->permutation_time;
        last_factorization_time_ = osqp_info->factorization_time;
    }
    // printf("After initSolver : %d\n", solver_.getStatus());

    // printf("Solver settings :\n");
    // printf("  eps_abs : %f\n", solver_.settings()->getSettings()->eps_abs);
    // printf("  eps_rel : %f\n", solver_.settings()->getSettings()->eps_rel);
    // printf("  time_limit : %f\n", solver_.settings()->getSettings()->time_limit);
    // printf("  verbose : %d\n", solver_.settings()->getSettings()->verbose);
    // printf("  max_iter : %d\n", solver_.settings()->getSettings()->max_iter);
    // printf("  polishing : %d\n", solver_.settings()->getSettings()->polishing);
    // printf("  polish_refine_iter : %d\n", solver_.settings()->getSettings()->polish_refine_iter);


    // Set iteration context for OSQP logging
    osqp_set_iteration_context(current_solve_count_, sqp_iter_);

    // solve the QP problem
    auto start_solve = std::chrono::high_resolution_clock::now();
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
        printf("[solveQP FAIL] solveProblem failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[solveProblem] " << current_solve_count_ << ", " << sqp_iter_ << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }
    auto end_solve = std::chrono::high_resolution_clock::now();
    last_solve_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(end_solve - start_solve).count();

    // Get rho_updates from OSQPInfo after solve
    const OSQPInfo* osqp_info_after_solve = solver_.getInfo();
    if (osqp_info_after_solve != nullptr) {
        last_rho_updates_ = osqp_info_after_solve->rho_updates;
    }

    // printf("After solveProblem : %d\n", solver_.getStatus());
    iter_count = solver_.getNumberOfIterations();
    qp_status = solver_.getStatus();
    // printf("qp_status solved inaccurate : %s\n", qp_status == OsqpEigen::Status::SolvedInaccurate ? "true" : "false");
    if (!(solver_.getStatus() == OsqpEigen::Status::Solved || solver_.getStatus() == OsqpEigen::Status::SolvedInaccurate)) {
        printf("[solveQP FAIL] solver status=%d (not Solved/SolvedInaccurate) at solve_count=%d, sqp_iter=%d\n",
               static_cast<int>(solver_.getStatus()), current_solve_count_, sqp_iter_);
        if (g_solveQP_fail_log.is_open()) {
            g_solveQP_fail_log << "[solverStatus] " << current_solve_count_ << ", " << sqp_iter_
                              << ", status=" << static_cast<int>(solver_.getStatus()) << std::endl;
            g_solveQP_fail_log.flush();
        }
        return false;
    }
    // if (!(solver_.getStatus() == OsqpEigen::Status::Solved)) return false;
    // if (!(solver_.getStatus() == OsqpEigen::Status::Solved || solver_.getStatus() == OsqpEigen::Status::SolvedInaccurate) && solver_.getStatus() != OsqpEigen::Status::TimeLimitReached) return false;
    // printf("After getStatus : %d\n", solver_.getStatus());

    // printf("solver_.getSolution_size : %ld", solver_.getSolution().rows());
    // printf("solver_.getDualSolution_size : %ld", solver_.getDualSolution().rows());
    // Get solution and cast to double (c_float may be float when OSQP_USE_FLOAT=ON)
    step = solver_.getSolution().cast<double>();
    step_lambda = solver_.getDualSolution().cast<double>();

    solver_.clearSolverVariables();
    solver_.clearSolver();

    // for(size_t i=0; i<rb_.size(); i++) rb_[i].setZero();

    return true; 
}

bool OsqpInterface::SecondOrderCorrection(const std::vector<OptVariables> &initial_guess, const Eigen::MatrixXd &hess_obj, const Eigen::VectorXd &grad_obj, const Eigen::MatrixXd & jac_constr,
                                          Eigen::VectorXd &step, Eigen::VectorXd &step_lambda, OsqpEigen::Status& qp_status)
{
    const std::vector<OptVariables> updates_initial_guess = vectorToOptvar(OptvarToVector(initial_guess) + step);

    Eigen::VectorXd constr_step;
    Eigen::VectorXd l_step, u_step;
    constr_step.setZero(N_constr);
    l_step.setZero(N_constr);
    u_step.setZero(N_constr);
    setConstraints(updates_initial_guess, NULL, &constr_step, &l_step, &u_step);

    // Constraints
    // from   l <= A.x + b <= u
    // to   l-b <= A.x     <= u-b
    Eigen::MatrixXd P = hess_obj;
    Eigen::VectorXd q = grad_obj;
    Eigen::MatrixXd A = jac_constr;
    Eigen::VectorXd d = constr_step - A*step;
    Eigen::VectorXd l = l_step - d;
    Eigen::VectorXd u = u_step - d;

    int i = 0;

    return solveQP(P, q, A, l, u, step, step_lambda, qp_status, i); //iter_count = 0
}

Eigen::MatrixXd OsqpInterface::BFGSUpdate(const Eigen::MatrixXd &Hess, const Eigen::VectorXd &step_prev, const Eigen::VectorXd &delta_grad_L)
{
    // Damped BFGS update
    // Implements "Procedure 18.2 Damped BFGS updating for SQP" form Numerical Optimization by Nocedal.
    double sy, sr, sBs;
    Eigen::VectorXd Bs, r;

    Bs = Hess * step_prev;
    sBs = step_prev.dot(Bs);
    sy = step_prev.dot(delta_grad_L);

    if (sy < 0.2 * sBs) 
    {
        // damped update to enforce positive definite B
        double theta;
        theta = 0.8 * sBs / (sBs - sy);
        r = theta * delta_grad_L + (1 - theta) * Bs;
        sr = theta * sy + (1 - theta) * sBs;
    } 
    else 
    {
        // unmodified BFGS
        r = delta_grad_L;
        sr = sy;
    }

    if (sr < std::numeric_limits<double>::epsilon()) 
    {
        return Hess;
    }

    return Hess - Bs * Bs.transpose() / sBs + r * r.transpose() / sr;
}

double OsqpInterface::meritLineSearch(const Eigen::VectorXd &step, const Eigen::MatrixXd &Hess, const Eigen::VectorXd &grad_obj, const double &obj, const Eigen::VectorXd &constr, const Eigen::VectorXd &l, const Eigen::VectorXd &u)
{
    double mu, phi_l1, Dp_phi_l1;
    const double tau = sqp_param_.line_search_tau; // line search step decrease, 0 < tau < settings.tau

    double constr_l1 = constraint_norm(constr, l, u);

    // get mu from merit function model using hessian of Lagrangian instead
    mu = (grad_obj.dot(step) + 0.5 * step.dot(Hess * step)) / ((1 - sqp_param_.line_search_rho) * constr_l1);

    phi_l1 = obj + mu * constr_l1;
    Dp_phi_l1 = grad_obj.dot(step) - mu * constr_l1;

    double updated_obj;
    Eigen::VectorXd updated_constr;
    Eigen::VectorXd updated_l, updated_u;

    double alpha = 1.0;
    for(size_t i=0; i<sqp_param_.line_search_max_iter; i++)
    {
        updated_constr.setZero(N_constr);
        updated_l.setZero(N_constr);
        updated_u.setZero(N_constr);

        Eigen::VectorXd updated_initial_guess_vec = initial_guess_vec_ + alpha * deNormalizeStep(step);
        std::vector<OptVariables> updated_initial_guess = vectorToOptvar(updated_initial_guess_vec);
        setQP(updated_initial_guess, NULL, NULL, &updated_obj, NULL, &updated_constr, &updated_l, &updated_u);

        double updated_phi_l1 = updated_obj + mu * constraint_norm(updated_constr, updated_l, updated_u);
        if (updated_phi_l1 <= phi_l1 + alpha * sqp_param_.line_search_eta * Dp_phi_l1) 
        {
            // accept step
            break;
        } 
        else 
        {
            alpha = tau * alpha;
        }
    }
    return alpha;
}

double OsqpInterface::filterLineSearch(const std::vector<OptVariables> &initial_guess, const Eigen::VectorXd &step, std::vector<FilterData> &filter_data_list)
{
    FilterData updated_filter_data;
    Eigen::VectorXd updated_constr, updated_l, updated_u;
    updated_constr.setZero(N_constr);
    updated_l.setZero(N_constr);
    updated_u.setZero(N_constr);

    bool is_alpha_accepted = true;

    double alpha = 1.0;
    for(size_t i=0; i<sqp_param_.line_search_max_iter; i++)
    {
        // get updated cost and constraint wrt x+delta_x
        const std::vector<OptVariables> updated_initial_guess = vectorToOptvar(OptvarToVector(initial_guess) + alpha*deNormalizeStep(step));
        setQP(updated_initial_guess, NULL, NULL, &updated_filter_data.obj, NULL, &updated_constr, &updated_l, &updated_u);
        updated_filter_data.gap_vio = constraint_norm(updated_constr, updated_l, updated_u);
        // filtering updated data
        for(size_t j=0; j<filter_data_list.size(); j++)
        {
            if(updated_filter_data.obj >= filter_data_list[j].obj && updated_filter_data.gap_vio >= filter_data_list[j].gap_vio)
            {
                is_alpha_accepted = false;
                break;
            }
        }
        if(is_alpha_accepted)
        {
            // update filter data list
            std::vector<FilterData> updated_filter_data_list;
            updated_filter_data_list.clear();
            for(size_t j=0; j<filter_data_list.size(); j++)
            {
                if(updated_filter_data.obj > filter_data_list[j].obj || updated_filter_data.gap_vio > filter_data_list[j].gap_vio)
                {
                    updated_filter_data_list.push_back(filter_data_list[j]);
                }
            }
            updated_filter_data_list.push_back(updated_filter_data);
            filter_data_list = updated_filter_data_list;
            break;
        }
        else
        {
            // decrease alpha
            alpha *= sqp_param_.line_search_tau;
        }
    }
    return alpha;
}

bool OsqpInterface::isPosdef(const Eigen::MatrixXd& H)
{
    Eigen::LLT<MatrixXd> llt(H);
    if (llt.info() == Eigen::NumericalIssue) {
        return false;
    }
    return true;
}

// custom function
bool OsqpInterface::printAllEigenvalues(const Eigen::MatrixXd& H)
{
    // SelfAdjointEigenSolver는 H가 대칭(자기수반)인 경우 사용하는 것이 적절합니다.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H);
    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "Eigen decomposition failed." << std::endl;
        return false;
    }
    
    // 모든 eigen value를 출력합니다.
    std::cerr << "All computed eigenvalues:" << std::endl;
    const Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
    for (int i = 0; i < eigenvalues.size(); ++i) {
        std::cerr << "Eigenvalue[" << i << "]: " << eigenvalues(i) << std::endl;
    }
    return true;
}

template <typename Derived>
void OsqpInterface::saveMatrixToFile(const Eigen::MatrixBase<Derived>& Matrix, const std::string& matrixName, const std::string& filename)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    // double 값의 모든 정보를 온전히 출력하기 위해 
    // std::scientific : 지수 표기법으로 출력하여 exponent도 명시적으로 출력
    // std::setprecision(std::numeric_limits<double>::max_digits10) : double를 완전하게 표현하는 데 필요한 자릿수 출력
    // outFile << std::scientific 
    //         << std::setprecision(std::numeric_limits<double>::max_digits10);

    
    for (int i = 0; i < Matrix.rows(); ++i)
    {
        for (int j = 0; j < Matrix.cols(); ++j)
        {
            // outFile << matrixName << "_(" << i << "," << j << ") = " 
            //         << std::hexfloat << Matrix(i, j) 
            //         << std::defaultfloat << "\t";

            outFile << matrixName << "_(" << i << "," << j << ") = " 
                    << std::scientific           // 과학적 표기 강제
                    << std::setprecision(4)      // 유효숫자 7자리
                    << Matrix(i, j)                     // "-3.0209997e-04"
                    << std::endl;
        }
        outFile << std::endl;
    }
    
    outFile.close();
}

void OsqpInterface::saveSparseMatrixToFile(const Eigen::SparseMatrix<float, 0, int>& Matrix, const std::string& matrixName, const std::string& filename)
{
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    // double 값의 모든 정보를 온전히 출력하기 위해 
    // std::scientific : 지수 표기법으로 출력하여 exponent도 명시적으로 출력
    // std::setprecision(std::numeric_limits<double>::max_digits10) : double를 완전하게 표현하는 데 필요한 자릿수 출력
    // outFile << std::scientific 
    //         << std::setprecision(std::numeric_limits<double>::max_digits10);

    
    // for (int i = 0; i < Matrix.rows(); ++i)
    // {
    //     for (int j = 0; j < Matrix.cols(); ++j)
    //     {
    //         // outFile << matrixName << "_(" << i << "," << j << ") = " 
    //         //         << std::hexfloat << Matrix(i, j) 
    //         //         << std::defaultfloat << "\t";

    //         outFile << matrixName << "_(" << i << "," << j << ") = " 
    //                 << std::scientific           // 과학적 표기 강제
    //                 << std::setprecision(4)      // 유효숫자 7자리
    //                 << Matrix(i, j)                     // "-3.0209997e-04"
    //                 << std::endl;
    //     }
    //     outFile << std::endl;
    // }

    outFile << "Sparse Matrix: " << matrixName << std::endl;
    outFile << Matrix << std::endl;
    
    outFile.close();
}

bool OsqpInterface::isNan(const Eigen::MatrixXd& x)
{
    return x.array().isNaN().any();
}

double OsqpInterface::constraint_norm(const Eigen::VectorXd &constr, const Eigen::VectorXd &l, const Eigen::VectorXd &u)
{
    double c_l1 = 0;

    // l <= c(x) <= u
    c_l1 += (l - constr).cwiseMax(0.0).sum();
    c_l1 += (constr - u).cwiseMax(0.0).sum();

    return c_l1;
}

std::vector<OptVariables> OsqpInterface::vectorToOptvar(const Eigen::VectorXd& opt_var_vec)
{
    std::vector<OptVariables> opt_var;
    opt_var.resize(N+1);
    for(size_t i=0;i<=N;i++)
    {
        opt_var[i].xk = vectorToState(opt_var_vec.segment(NX*i, NX));
        if(i!=N) opt_var[i].uk = vectorToInput(opt_var_vec.segment(NX*(N+1)+NU*i, NU));
    }
    return opt_var;
}

Eigen::VectorXd OsqpInterface::OptvarToVector(const std::vector<OptVariables>& opt_var)
{
    Eigen::VectorXd opt_var_vec;
    opt_var_vec.setZero(N_var);
    for(size_t i=0;i<=N;i++)
    {
        opt_var_vec.segment(NX*i,NX) = stateToVector(opt_var[i].xk);
        if(i!=N) opt_var_vec.segment(NX*(N+1)+NU*i, NU) = inputToVector(opt_var[i].uk);
    }
    return opt_var_vec;
}

Eigen::VectorXd OsqpInterface::deNormalizeStep(const Eigen::VectorXd& step)
{
    Eigen::VectorXd denormalized_step;
    denormalized_step.setZero(N_var);
    for(size_t i=0;i<=N;i++) 
    {
        denormalized_step.segment(NX*i, NX) = normalization_param_.T_x*step.segment(NX*i, NX);
        if(i!=N) denormalized_step.segment(NX*(N+1) + NU*i, NU) = normalization_param_.T_u*step.segment(NX*(N+1) + NU*i, NU);
    }
    return denormalized_step;
}

void OsqpInterface::printOptVar(std::vector<OptVariables> opt_var)
{
    for(size_t i=0;i<=N;i++)
    {
        std::cout << "State[" << i << "]: " << std::endl;;
        std::cout << "\tq    : " << stateToJointVector(opt_var[i].xk).transpose() << std::endl;
        std::cout << "\ts    : " << opt_var[i].xk.s << std::endl;
        std::cout << "\ts dot: " << opt_var[i].xk.vs << std::endl;
    }
    std::cout <<" \n";
    for(size_t i=0;i<N;i++)
    {
        std::cout << "Input[" << i << "]: " << std::endl;;
        std::cout << "\tqdot: " << inputTodJointVector(opt_var[i].uk).transpose() << std::endl;;
        std::cout << "\tdVs  : " << opt_var[i].uk.dVs << std::endl;;
    }
}

void OsqpInterface::computeDiagonalScaling(const Eigen::MatrixXd &P, const Eigen::MatrixXd &A,
                                           Eigen::VectorXd &D, Eigen::VectorXd &E)
{
    // Compute diagonal scaling factors using Jacobi scaling
    // D: variable scaling (n x 1), E: constraint scaling (m x 1)
    // D_i = 1 / sqrt(max(P_ii, max_j |A_ji|))
    // E_i = 1 / max_j |A_ij|

    const int n = P.rows();  // number of variables
    const int m = A.rows();  // number of constraints

    D.resize(n);
    E.resize(m);

    const double min_scaling = 1e-4;  // prevent division by very small numbers
    const double max_scaling = 1e4;   // prevent too large scaling factors

    // Compute D (variable scaling)
    for(int j = 0; j < n; j++)
    {
        double max_val = std::abs(P(j, j));
        for(int i = 0; i < m; i++)
        {
            max_val = std::max(max_val, std::abs(A(i, j)));
        }
        if(max_val < min_scaling)
        {
            D(j) = max_scaling;
        }
        else
        {
            D(j) = 1.0 / std::sqrt(max_val);
            D(j) = std::min(D(j), max_scaling);
        }
    }

    // Compute E (constraint scaling)
    for(int i = 0; i < m; i++)
    {
        double max_val = 0.0;
        for(int j = 0; j < n; j++)
        {
            max_val = std::max(max_val, std::abs(A(i, j)));
        }
        if(max_val < min_scaling)
        {
            E(i) = max_scaling;
        }
        else
        {
            E(i) = 1.0 / max_val;
            E(i) = std::min(E(i), max_scaling);
        }
    }
}

void OsqpInterface::applyScaling(const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
                                 const Eigen::MatrixXd &A, const Eigen::VectorXd &l, const Eigen::VectorXd &u,
                                 const Eigen::VectorXd &D, const Eigen::VectorXd &E,
                                 Eigen::MatrixXd &P_scaled, Eigen::VectorXd &q_scaled,
                                 Eigen::MatrixXd &A_scaled, Eigen::VectorXd &l_scaled, Eigen::VectorXd &u_scaled)
{
    // Apply scaling:
    // P_scaled = D * P * D
    // q_scaled = D * q
    // A_scaled = E * A * D
    // l_scaled = E * l
    // u_scaled = E * u

    const int n = P.rows();
    const int m = A.rows();

    // P_scaled = D * P * D (using diagonal matrix multiplication)
    P_scaled.resize(n, n);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            P_scaled(i, j) = D(i) * P(i, j) * D(j);
        }
    }

    // q_scaled = D * q
    q_scaled = D.asDiagonal() * q;

    // A_scaled = E * A * D
    A_scaled.resize(m, n);
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            A_scaled(i, j) = E(i) * A(i, j) * D(j);
        }
    }

    // l_scaled = E * l, u_scaled = E * u
    l_scaled = E.asDiagonal() * l;
    u_scaled = E.asDiagonal() * u;
}

void OsqpInterface::unscaleSolution(const Eigen::VectorXd &x_scaled, const Eigen::VectorXd &lambda_scaled,
                                    const Eigen::VectorXd &D, const Eigen::VectorXd &E,
                                    Eigen::VectorXd &x, Eigen::VectorXd &lambda)
{
    // Unscale solution:
    // x = D * x_scaled
    // lambda = E * lambda_scaled

    x = D.asDiagonal() * x_scaled;
    lambda = E.asDiagonal() * lambda_scaled;
}

bool OsqpInterface::solveQPWithScaling(const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
                                       const Eigen::MatrixXd &A, const Eigen::VectorXd &l, const Eigen::VectorXd &u,
                                       Eigen::VectorXd &step, Eigen::VectorXd &step_lambda,
                                       OsqpEigen::Status &qp_status, int &iter_count)
{
    // Compute scaling factors
    Eigen::VectorXd D, E;
    computeDiagonalScaling(P, A, D, E);

    // Apply scaling
    Eigen::MatrixXd P_scaled;
    Eigen::VectorXd q_scaled;
    Eigen::MatrixXd A_scaled;
    Eigen::VectorXd l_scaled, u_scaled;
    applyScaling(P, q, A, l, u, D, E, P_scaled, q_scaled, A_scaled, l_scaled, u_scaled);

    // Convert to c_float (OSQP precision)
    Eigen::SparseMatrix<c_float> P_sp = P_scaled.cast<c_float>().sparseView();
    Eigen::SparseMatrix<c_float> A_sp = A_scaled.cast<c_float>().sparseView();
    Eigen::Matrix<c_float, N_var, 1> q_ds = q_scaled.cast<c_float>();
    Eigen::Matrix<c_float, N_constr, 1> l_ds = l_scaled.cast<c_float>();
    Eigen::Matrix<c_float, N_constr, 1> u_ds = u_scaled.cast<c_float>();

    // Setup solver
    OsqpEigen::Solver solver_;
    solver_.settings()->setWarmStart(false);
    solver_.settings()->setMaxIteration(250);
    solver_.settings()->getSettings()->eps_abs = 9.765625e-04f;    // 2^(-10), 0x3A800000, was 1e-3
    solver_.settings()->getSettings()->eps_rel = 1.220703125e-04f; // 2^(-13), 0x39000000, was 1e-4
    solver_.settings()->getSettings()->verbose = false;

    auto start_init = std::chrono::high_resolution_clock::now();
    solver_.data()->setNumberOfVariables(N_var);
    solver_.data()->setNumberOfConstraints(N_constr);
    if (!solver_.data()->setHessianMatrix(P_sp)) {
        printf("[solveQPWithScaling FAIL] setHessianMatrix failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        return false;
    }
    if (!solver_.data()->setGradient(q_ds)) {
        printf("[solveQPWithScaling FAIL] setGradient failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        return false;
    }
    if (!solver_.data()->setLinearConstraintsMatrix(A_sp)) {
        printf("[solveQPWithScaling FAIL] setLinearConstraintsMatrix failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        return false;
    }
    if (!solver_.data()->setLowerBound(l_ds)) {
        printf("[solveQPWithScaling FAIL] setLowerBound failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        return false;
    }
    if (!solver_.data()->setUpperBound(u_ds)) {
        printf("[solveQPWithScaling FAIL] setUpperBound failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        return false;
    }

    if (!solver_.initSolver()) {
        printf("[solveQPWithScaling FAIL] initSolver failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        return false;
    }
    auto end_init = std::chrono::high_resolution_clock::now();
    last_init_solver_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(end_init - start_init).count();

    // Get detailed timing breakdown from OSQPInfo
    const OSQPInfo* osqp_info = solver_.getInfo();
    if (osqp_info != nullptr) {
        last_scaling_time_ = osqp_info->scaling_time;
        last_permutation_time_ = osqp_info->permutation_time;
        last_factorization_time_ = osqp_info->factorization_time;
    }

    // Set iteration context for OSQP logging
    osqp_set_iteration_context(current_solve_count_, sqp_iter_);

    // Solve
    auto start_solve = std::chrono::high_resolution_clock::now();
    if (solver_.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
        printf("[solveQPWithScaling FAIL] solveProblem failed at solve_count=%d, sqp_iter=%d\n", current_solve_count_, sqp_iter_);
        return false;
    }
    auto end_solve = std::chrono::high_resolution_clock::now();
    last_solve_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(end_solve - start_solve).count();

    // Get rho_updates from OSQPInfo after solve
    const OSQPInfo* osqp_info_after_solve = solver_.getInfo();
    if (osqp_info_after_solve != nullptr) {
        last_rho_updates_ = osqp_info_after_solve->rho_updates;
    }

    iter_count = solver_.getNumberOfIterations();
    qp_status = solver_.getStatus();

    if (!(solver_.getStatus() == OsqpEigen::Status::Solved ||
          solver_.getStatus() == OsqpEigen::Status::SolvedInaccurate)) {
        printf("[solveQPWithScaling FAIL] solver status=%d at solve_count=%d, sqp_iter=%d\n",
               static_cast<int>(solver_.getStatus()), current_solve_count_, sqp_iter_);
        return false;
    }

    // Get scaled solution
    Eigen::VectorXd step_scaled = solver_.getSolution().cast<double>();
    Eigen::VectorXd step_lambda_scaled = solver_.getDualSolution().cast<double>();

    // Unscale solution
    unscaleSolution(step_scaled, step_lambda_scaled, D, E, step, step_lambda);

    solver_.clearSolverVariables();
    solver_.clearSolver();

    return true;
}
}

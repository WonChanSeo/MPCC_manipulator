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

// Copyright ...
#include "MPC/mpc.h"
#include <algorithm>  // std::clamp
#include <cmath>      // std::abs, std::fabs
#include <Eigen/Core>
#include <cstdlib>    // std::getenv
#include <fstream>    // for logging
#include "osqp_api_functions.h"

// Global log file for generateNewInitialGuess calls
static std::ofstream g_initialGuess_log;

// Global log file for SQP status tracking
static std::ofstream g_sqp_status_log;

namespace mpcc {

MPC::MPC()
: Ts_(1.0)
{
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

MPC::MPC(double Ts,const PathToJson &path)
: Ts_(Ts),
  path_(path),
  valid_initial_guess_(false),
  num_valid_guess_failed_(0),
  solver_interface_(new OsqpInterface(Ts, path)),
  track_(path),
  param_(path.param_path),
  integrator_(Ts,path),
  robot_(new RobotModel()),
  solve_count_(0),
  max_iter_reached_count_(0),
  max_iter_solve_failed_count_(0)
{
    initial_guess_.resize(N+1);
    top3_total_iter_counts_.clear();
    top3_solve_nums_.clear();

    // Set OSQP log filepath from environment variable if available
    const char* osqp_log_path = std::getenv("OSQP_LOG_PATH");
    if (osqp_log_path) {
        osqp_set_log_filepath(osqp_log_path);
    }

    // Initialize SQP status log file
    if (!g_sqp_status_log.is_open()) {
        // Get log directory from environment variable, default to "result"
        const char* sim_name_env = std::getenv("SIM_NAME");
        std::string log_dir = "/home/mms-wonchan/git/MPCC_manipulator/result";
        if (sim_name_env) {
            log_dir += "/" + std::string(sim_name_env);
        }
        std::string log_path = log_dir + "/sqp_status_log.txt";

        g_sqp_status_log.open(log_path, std::ios::out | std::ios::trunc);
        if (g_sqp_status_log.is_open()) {
            g_sqp_status_log << "=== SQP Status Log (Non-SOLVED cases) ===" << std::endl;
            g_sqp_status_log << "Format: solve_count, sqp_iter_count, total_iter_count, status, status_name" << std::endl;
            g_sqp_status_log << "==========================================" << std::endl;
        }
    }
}

MPC::MPC(double Ts,const PathToJson &path,const ParamValue &param_value)
: Ts_(Ts),
  path_(path),
  valid_initial_guess_(false),
  num_valid_guess_failed_(0),
  solver_interface_(new OsqpInterface(Ts, path, param_value)),
  track_(path,param_value),
  param_(path.param_path, param_value.param),
  integrator_(Ts,path),
  robot_(new RobotModel()),
  solve_count_(0),
  max_iter_reached_count_(0),
  max_iter_solve_failed_count_(0)
{
    initial_guess_.resize(N+1);
    top3_total_iter_counts_.clear();
    top3_solve_nums_.clear();

    // Set OSQP log filepath from environment variable if available
    const char* osqp_log_path = std::getenv("OSQP_LOG_PATH");
    if (osqp_log_path) {
        osqp_set_log_filepath(osqp_log_path);
    }

    // Initialize SQP status log file
    if (!g_sqp_status_log.is_open()) {
        // Get log directory from environment variable, default to "result"
        const char* sim_name_env = std::getenv("SIM_NAME");
        std::string log_dir = "/home/mms-wonchan/git/MPCC_manipulator/result";
        if (sim_name_env) {
            log_dir += "/" + std::string(sim_name_env);
        }
        std::string log_path = log_dir + "/sqp_status_log.txt";

        g_sqp_status_log.open(log_path, std::ios::out | std::ios::trunc);
        if (g_sqp_status_log.is_open()) {
            g_sqp_status_log << "=== SQP Status Log (Non-SOLVED cases) ===" << std::endl;
            g_sqp_status_log << "Format: solve_count, sqp_iter_count, total_iter_count, status, status_name" << std::endl;
            g_sqp_status_log << "==========================================" << std::endl;
        }
    }
}

void MPC::updateInitialGuess(const State &x0)
{
    for(int i=1;i<N;i++) initial_guess_[i-1] = initial_guess_[i];

    initial_guess_[0].xk = x0;

    initial_guess_[N-1].xk = initial_guess_[N-2].xk;
    initial_guess_[N-1].uk = initial_guess_[N-2].uk;

    initial_guess_[N].xk = integrator_.RK4(initial_guess_[N-1].xk,initial_guess_[N-1].uk,Ts_);
    initial_guess_[N].uk.setZero();

    unwrapInitialGuess();
}

void MPC::unwrapInitialGuess()
{
    for(int i=0;i<=N;i++)
    {
        initial_guess_[i].xk.s = std::max(0., std::min(initial_guess_[i].xk.s, 1.));
    }
}

/* ===================== 보조 멤버 함수 ===================== */

Eigen::VectorXd MPC::stateToJointVector(const State& x) const {
    Eigen::VectorXd q(7);
    q << x.q1, x.q2, x.q3, x.q4, x.q5, x.q6, x.q7;
    return q;
}
Eigen::VectorXd MPC::inputToDqVector(const Input& u) const {
    Eigen::VectorXd dq(7);
    dq << u.dq1, u.dq2, u.dq3, u.dq4, u.dq5, u.dq6, u.dq7;
    return dq;
}

/* ===================== vs 정사영 함수 (track_/robot_ 사용) ===================== */

double MPC::project_vs_workspace(double s,
                                 const Eigen::Ref<const Eigen::VectorXd>& q,
                                 const Eigen::Ref<const Eigen::VectorXd>& dq) const
{
    // 경로 접선 a(s): ArcLengthSpline API는 getDerivative(s)
    Eigen::Vector3d a = track_.getDerivative(s);   // X'(s),Y'(s),Z'(s)

    // RobotModel::getJacobianv(q)는 non-const → const 우회
    auto* robot_nc = const_cast<RobotModel*>(robot_.get());
    const Eigen::MatrixXd& Jv = robot_nc->getJacobianv(q); // 3x7 선속도 자코비안

    Eigen::Vector3d xdot = Jv * dq;
    double denom = a.squaredNorm();
    if (denom <= 0.0) return 0.0;
    return a.dot(xdot) / denom;  // arc-length 매개화면 denom=1 → t^T xdot
}

double MPC::project_vs_joint(double /*s*/,
                             const Eigen::Ref<const Eigen::VectorXd>& /*dq*/) const
{
    // 현재 ArcLengthSpline에는 관절공간 경로 접선(dqref/ds) API가 없음.
    // 관절공간 경로를 쓰지 않는 구성이라면 사용되지 않음. 필요 시 track_에 dqref_ds(s) 추가 후 구현.
    return 0.0;
}

/* ===================== Cold start: 비상정지 기반 initial guess ===================== */
void MPC::generateNewInitialGuess(const State &x0)
{
    std::cout << "generate new initial guess!!" << std::endl;
    for(int i = 0; i <= N; i++)
    {
        initial_guess_[i].xk = x0;
        initial_guess_[i].uk.setZero();
    }
    unwrapInitialGuess();
    valid_initial_guess_ = true;
}

bool MPC::runMPC(MPCReturn &mpc_return, State &x0, Input &u0)
{
    Eigen::MatrixX3d dummy_position;
    double dummy_radius = 0.;
    return runMPC_(mpc_return, x0, u0, dummy_position, dummy_radius);
}

bool MPC::runMPC_(MPCReturn &mpc_return, State &x0, Input &u0, const Eigen::MatrixX3d &obs_positions, const double &obs_radius)
{
    auto start_mpc = std::chrono::high_resolution_clock::now();
    double last_s = x0.s;
    int iter_count = 0;
    int sqp_iter_count = 0;

    // 1. 현재 상태(x0) 보정 (EE pose로 경로 투영)
    Eigen::Matrix4d ee_pose = robot_->getEETransformation(stateToJointVector(x0));
    x0.s = track_.projectOnSpline(last_s, ee_pose);

    // 2. 경로 이탈 확인 → Cold start 플래그
    if(std::fabs(last_s - x0.s) > param_.max_dist_proj) 
    {
        valid_initial_guess_ = false;
        num_valid_guess_failed_++;
    }

    // 3. Warm/Cold start
    if(valid_initial_guess_)
        updateInitialGuess(x0);
    else {
        // Initialize log file if not open
        if (!g_initialGuess_log.is_open()) {
            const char* log_path = std::getenv("INITIAL_GUESS_LOG_PATH");
            std::string log_file = log_path ? log_path : "/home/mms-wonchan/git/MPCC_manipulator/result/initialGuess_calls.txt";
            g_initialGuess_log.open(log_file, std::ios::out | std::ios::trunc);
            if (g_initialGuess_log.is_open()) {
                g_initialGuess_log << "=== generateNewInitialGuess Call Log ===" << std::endl;
                g_initialGuess_log << "Format: [COLD_START] solve_count, reason" << std::endl;
                g_initialGuess_log << "=========================================" << std::endl;
            }
        }
        // Log the cold start call
        if (g_initialGuess_log.is_open()) {
            std::string reason = (num_valid_guess_failed_ > 0) ? "path_deviation" : "qp_failure";
            g_initialGuess_log << "[COLD_START] " << (solve_count_ + 1) << ", " << reason << std::endl;
            g_initialGuess_log.flush();
        }
        generateNewInitialGuess(x0);    // ★ 시그니처 수정
        // updateInitialGuess(x0);
    }

    // 4. 초기 추정치 전달
    solver_interface_->setCurrentInput(u0);
    solver_interface_->setInitialGuess(initial_guess_);

    // 5. 환경 데이터 설정
    auto start_env = std::chrono::high_resolution_clock::now();
    solver_interface_->setEnvData(obs_positions, obs_radius);
    auto end_env = std::chrono::high_resolution_clock::now();

    // 6. 장애물 스위치 감지 → 다음 스텝 Cold start
    if (solver_interface_->getEnvColNN()->obstacle_switched) {
        std::cout << "[MPC] Obstacle switch detected! Forcing a new initial guess for the next step." << std::endl;
        valid_initial_guess_ = false;
    }

    printf("BEFORE initial_guess_[0].uk: [%f, %f, %f, %f, %f, %f, %f, %f]\n",
           initial_guess_[0].uk.dq1, initial_guess_[0].uk.dq2, initial_guess_[0].uk.dq3,
           initial_guess_[0].uk.dq4, initial_guess_[0].uk.dq5, initial_guess_[0].uk.dq6,
           initial_guess_[0].uk.dq7, initial_guess_[0].uk.dVs);

    // 7. QP 풀이
    Status sqp_status;
    ComputeTime time_nmpc;
    int total_iter_count = 0;
    solver_interface_->solveOCP(initial_guess_, &sqp_status, &time_nmpc, iter_count, sqp_iter_count, total_iter_count, solve_count_ + 1);
   
    // Track max_iter reached cases
    bool is_max_iter_reached = (sqp_status == MAX_ITER_EXCEEDED || sqp_status == QP_MaxIterReached);
    if (is_max_iter_reached) {
        max_iter_reached_count_++;
    }

    // Helper lambda to get status name
    auto getStatusName = [](Status status) -> std::string {
        switch(status) {
            case SOLVED: return "SOLVED";
            case MAX_ITER_EXCEEDED: return "MAX_ITER_EXCEEDED";
            case QP_MaxIterReached: return "QP_MaxIterReached";
            case QP_SolvedInaccurate: return "QP_SolvedInaccurate";
            case QP_DualInfeasible: return "QP_DualInfeasible";
            case QP_DualInfeasibleInaccurate: return "QP_DualInfeasibleInaccurate";
            case QP_PrimalInfeasible: return "QP_PrimalInfeasible";
            case QP_PrimalInfeasibleInaccurate: return "QP_PrimalInfeasibleInaccurate";
            case NAN_HESSIAN: return "NAN_HESSIAN";
            case NON_PD_HESSIAN: return "NON_PD_HESSIAN";
            default: return "UNKNOWN";
        }
    };

    // Log non-SOLVED status
    if(sqp_status != SOLVED && g_sqp_status_log.is_open()) {
        g_sqp_status_log << solve_count_ << ", " << sqp_iter_count << ", " << total_iter_count
                         << ", " << static_cast<int>(sqp_status) << ", " << getStatusName(sqp_status) << std::endl;
        g_sqp_status_log.flush();
    }

    if(sqp_status == SOLVED)
    {
        valid_initial_guess_ = true;
        num_valid_guess_failed_ = 0;
    }
    else if(sqp_status == MAX_ITER_EXCEEDED)
    {
        // MAX_ITER_EXCEEDED: Immediately trigger cold start with generateNewInitialGuess
        std::cout << "===================================================" << std::endl;
        std::cout << "============== SQP Max Iter reached ===============" << std::endl;
        std::cout << "========== Triggering Cold Start Immediately ======" << std::endl;
        std::cout << "===================================================" << std::endl;

        // Immediately generate new initial guess (cold start)
        generateNewInitialGuess(x0);
        num_valid_guess_failed_ = 0;  // Reset counter after cold start
    }
    else
    {
        // Critical failure - stop simulation
        std::cout << "===================================================" << std::endl;
        std::cout << "========== CRITICAL FAILURE - STOPPING ============" << std::endl;
        std::cout << "================ QP did not solved ================" << std::endl;
        switch (sqp_status)
        {
        case MAX_ITER_EXCEEDED:
            std::cout << "============== SQP Max Iter reached ==============="<< std::endl;
            break;
        case QP_DualInfeasible:
            std::cout << "================= Dual Infeasible ================="<< std::endl;
            break;
        case QP_DualInfeasibleInaccurate:
            std::cout << "============ Dual Infeasible Inaccurate ============"<< std::endl;
            break;
        case QP_PrimalInfeasible:
            std::cout << "================= Primal Infeasible ================"<< std::endl;
            break;
        case QP_PrimalInfeasibleInaccurate:
            std::cout << "=========== Primal Infeasible Inaccurate ============"<< std::endl;
            break;
        case NAN_HESSIAN:
            std::cout << "==================== Nan Hessian ===================="<< std::endl;
            break;
        case NON_PD_HESSIAN:
            std::cout << "========== Not Possitive Definite Hessian ==========="<< std::endl;
            break;
        }
        std::cout << "============ Simulation will be terminated =========" << std::endl;
        std::cout << "===================================================" << std::endl;
        valid_initial_guess_ = false;
        num_valid_guess_failed_++;
    }

    printf("AFTER initial_guess_[0].uk: [%f, %f, %f, %f, %f, %f, %f, %f]\n",
           initial_guess_[0].uk.dq1, initial_guess_[0].uk.dq2, initial_guess_[0].uk.dq3,
           initial_guess_[0].uk.dq4, initial_guess_[0].uk.dq5, initial_guess_[0].uk.dq6,
           initial_guess_[0].uk.dq7, initial_guess_[0].uk.dVs);

    mpc_return = {initial_guess_[0].uk, initial_guess_, time_nmpc};

    mpc_return.compute_time = time_nmpc;
    auto end_mpc = std::chrono::high_resolution_clock::now();
    mpc_return.compute_time.total   = std::chrono::duration_cast<std::chrono::duration<double>>(end_mpc - start_mpc).count();
    mpc_return.compute_time.set_env = std::chrono::duration_cast<std::chrono::duration<double>>(end_env - start_env).count();
    mpc_return.iter_count = iter_count;
    mpc_return.sqp_iter_count = sqp_iter_count;
    mpc_return.total_iter_count = total_iter_count;

    // Increment solve counter
    solve_count_++;
    mpc_return.solve_count = solve_count_;

    // Update top 3 total_iter_count tracking
    bool found = false;
    for (size_t i = 0; i < top3_total_iter_counts_.size(); i++) {
        if (total_iter_count == top3_total_iter_counts_[i]) {
            // Same value found, add to existing list
            top3_solve_nums_[i].push_back(solve_count_);
            found = true;
            break;
        }
    }

    if (!found) {
        // New value, need to insert into top 3
        if (top3_total_iter_counts_.size() < 3) {
            // Less than 3 entries, just add
            top3_total_iter_counts_.push_back(total_iter_count);
            top3_solve_nums_.push_back(std::vector<int>{solve_count_});
        } else {
            // Check if this value belongs in top 3
            int min_idx = -1;
            int min_val = top3_total_iter_counts_[0];
            for (size_t i = 0; i < 3; i++) {
                if (top3_total_iter_counts_[i] < min_val) {
                    min_val = top3_total_iter_counts_[i];
                    min_idx = i;
                }
            }

            if (total_iter_count > min_val) {
                // Replace the minimum
                top3_total_iter_counts_[min_idx] = total_iter_count;
                top3_solve_nums_[min_idx].clear();
                top3_solve_nums_[min_idx].push_back(solve_count_);
            }
        }

        // Sort in descending order (bubble sort is fine for 3 elements)
        for (int i = 0; i < (int)top3_total_iter_counts_.size() - 1; i++) {
            for (int j = i + 1; j < (int)top3_total_iter_counts_.size(); j++) {
                if (top3_total_iter_counts_[i] < top3_total_iter_counts_[j]) {
                    std::swap(top3_total_iter_counts_[i], top3_total_iter_counts_[j]);
                    std::swap(top3_solve_nums_[i], top3_solve_nums_[j]);
                }
            }
        }
    }

    mpc_return.top3_total_iter_counts = top3_total_iter_counts_;
    mpc_return.top3_solve_nums = top3_solve_nums_;
    mpc_return.max_iter_reached_count = max_iter_reached_count_;
    mpc_return.max_iter_solve_failed_count = max_iter_solve_failed_count_;

    // RTI mode (max_iter=1) success criteria:
    // - SOLVED: convergence achieved
    // - MAX_ITER_EXCEEDED: SQP max iter (normal in RTI mode)
    // - QP partially solved: acceptable in RTI mode
    // Critical failures (Infeasible, NAN, etc.) will stop simulation

    bool qp_acceptable = (sqp_status == SOLVED) ||
                         (sqp_status == MAX_ITER_EXCEEDED) ||
                         (sqp_status == QP_MaxIterReached) ||
                         (sqp_status == QP_SolvedInaccurate);

    if(qp_acceptable)
    {
        return true;
    }
    else {
        // Critical failure - terminate simulation
        printf("MPC CRITICAL FAILURE: status=%d, num_valid_guess_failed=%d\n", sqp_status, num_valid_guess_failed_);
        printf("Simulation terminated due to unrecoverable solver failure.\n");
        return false;
    }
}

void MPC::setTrack(const Eigen::VectorXd &X, const Eigen::VectorXd &Y,const Eigen::VectorXd &Z,const std::vector<Eigen::Matrix3d> &R)
{
    track_.gen6DSpline(X,Y,Z,R);
    solver_interface_->setTrack(track_);
    valid_initial_guess_ = false;
}

double MPC::getTrackLength()
{
    return track_.getLength();
}

void MPC::setParam(const ParamValue &param_value)
{
    param_ = Param(path_.param_path, param_value.param);
    bounds_param_ = BoundsParam(path_.bounds_path, param_value.bounds);
    solver_interface_->setParam(param_value);
    printParamValue(param_value);
}

void MPC::printParamValue(const ParamValue& param_value) 
{
    std::cout << "param values:" << std::endl;
    for (const auto& item : param_value.param) std::cout << "\t" << item.first << ": " << item.second << std::endl;

    std::cout << "cost values:" << std::endl;
    for (const auto& item : param_value.cost) std::cout << "\t" << item.first << ": " << item.second << std::endl;

    std::cout << "bounds values:" << std::endl;
    for (const auto& item : param_value.bounds) std::cout << "\t" << item.first << ": " << item.second << std::endl;

    std::cout << "normalization values:" << std::endl;
    for (const auto& item : param_value.normalization) std::cout << "\t" << item.first << ": " << item.second << std::endl;

    std::cout << "sqp values:" << std::endl;
    for (const auto& item : param_value.sqp) std::cout << "\t" << item.first << ": " << item.second << std::endl;
}

} // namespace mpcc

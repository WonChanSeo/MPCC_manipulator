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
  solve_count_(0)
{
    initial_guess_.resize(N+1);
    top3_total_iter_counts_.clear();
    top3_solve_nums_.clear();

    // Set OSQP log filepath from environment variable if available
    const char* osqp_log_path = std::getenv("OSQP_LOG_PATH");
    if (osqp_log_path) {
        osqp_set_log_filepath(osqp_log_path);
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
  solve_count_(0)
{
    initial_guess_.resize(N+1);
    top3_total_iter_counts_.clear();
    top3_solve_nums_.clear();

    // Set OSQP log filepath from environment variable if available
    const char* osqp_log_path = std::getenv("OSQP_LOG_PATH");
    if (osqp_log_path) {
        osqp_set_log_filepath(osqp_log_path);
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
    std::cout << "====================================================================" << std::endl;
    std::cout << "[MPC] Generating a new initial guess based on maximum deceleration." << std::endl;

    initial_guess_[0] = initial_guess_[1];
    initial_guess_[0].xk = x0;

    const double dVs_min = bounds_param_.get_dVs("l"); // 보통 음수(최대 감속)
    const double dVs_max = bounds_param_.get_dVs("u"); // 보통 양수(최대 가속)

    // ★ off-by-one: next index(i+1) 접근하므로 i<N
    for (int i = 0; i < N; ++i)
    {
        State& cur_x = initial_guess_[i].xk;
        Input& cur_u = initial_guess_[i].uk;
        Input& nxt_u = initial_guess_[i + 1].uk;
        State& nxt_x = initial_guess_[i + 1].xk;

        // 1) 관절 감속 명령 (한 스텝 0-크로싱 방지)
        std::array<double, 7> ddq_array{};
        for (int j = 0; j < 7; ++j)
        {
            const double v = cur_u.get_dq(j);
            if      (v >  1e-4) ddq_array[j] = bounds_param_.get_ddq(j, "l"); // 음의 가속(감속)
            else if (v < -1e-4) ddq_array[j] = bounds_param_.get_ddq(j, "u"); // 양의 가속(감속)
            else                ddq_array[j] = 0.0;
        }

        // 2) vs를 dq에서 정사영으로 동기화
        Eigen::VectorXd q_now  = stateToJointVector(cur_x);
        Eigen::VectorXd dq_now = inputToDqVector(cur_u);

        // double vs_now = project_vs_workspace(cur_x.s, q_now, dq_now);
        // cur_x.vs = vs_now;

        // 3) 1-스텝 예측 vs_next → dVs = (vs_next - vs_now)/Ts_
        Eigen::VectorXd dq_next = dq_now;
        for (int j = 0; j < 7; ++j) {
            dq_next[j] = dq_now[j] + ddq_array[j] * Ts_;
            if (dq_now[j] * dq_next[j] < 0.0) dq_next[j] = 0.0; // 0-크로싱 방지
        }

        Eigen::Map<Eigen::VectorXd> ddq_vec(ddq_array.data(), 7);
        Eigen::VectorXd q_next_pred = q_now + dq_now * Ts_ + 0.5 * ddq_vec * (Ts_ * Ts_);
        double s_next_pred = cur_x.s + cur_x.vs * Ts_;

        // 예측 기하 사용(보수적으로 현재 기하 사용해도 됨)
        double vs_next = project_vs_workspace(s_next_pred, q_next_pred, dq_next);

        double dVs = (vs_next - cur_x.vs) / Ts_;
        if (cur_x.vs * vs_next < 0.0) {          // vs 부호 반전 시 정확히 0으로 스냅
            dVs = -cur_x.vs / Ts_;
            vs_next = 0.0;
        }
        dVs = std::clamp(dVs, dVs_min, dVs_max);

        // 4) 현재 스텝 입력에 dVs 반영 → 적분
        cur_u.set_dVs(dVs);
        nxt_x = integrator_.RK4(cur_x, cur_u, Ts_);

        // 5) 다음 스텝 dq 초기화(예측값 기록)
        for (int j = 0; j < 7; ++j) {
            nxt_u.set_dq(j, dq_next[j]);
        }

        printf("----------------------------------------------------\n");
        printf("Step %d - Current Input: [%f, %f, %f, %f, %f, %f, %f, %f], ddq: [%f, %f, %f, %f, %f, %f, %f], dVs: [%f]\n", i, cur_u.dq1, cur_u.dq2, cur_u.dq3, cur_u.dq4, cur_u.dq5, cur_u.dq6, cur_u.dq7, cur_u.dVs, ddq_array[0], ddq_array[1], ddq_array[2], ddq_array[3], ddq_array[4], ddq_array[5], ddq_array[6], dVs);
        printf("Step %d - Current State: [q: %f, %f %f %f %f %f %f, s: %f, vs: %f]\n", i, cur_x.q1, cur_x.q2, cur_x.q3, cur_x.q4, cur_x.q5, cur_x.q6, cur_x.q7, cur_x.s, cur_x.vs);
        printf("Next State: [q: %f, %f %f %f %f %f %f, s: %f, vs: %f]\n", nxt_x.q1, nxt_x.q2, nxt_x.q3, nxt_x.q4, nxt_x.q5, nxt_x.q6, nxt_x.q7, nxt_x.s, nxt_x.vs);
        printf("----------------------------------------------------\n");
    }

    // 마지막 입력은 안전하게 0
    initial_guess_[N].uk.setZero();

    unwrapInitialGuess();

    std::cout << "====================================================================" << std::endl;
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
            g_initialGuess_log.open("/home/mms-wonchan/git/MPCC_manipulator/result/initialGuess_calls.txt", std::ios::out | std::ios::trunc);
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
   
    if(sqp_status == SOLVED)
    {
        valid_initial_guess_ = true;
        num_valid_guess_failed_ = 0;
    }
    else
    {
        std::cout << "===================================================" << std::endl;
        std::cout << "================ QP did not solved ================" << std::endl;
        switch (sqp_status)
        {
        case MAX_ITER_EXCEEDED:
            std::cout << "============== SQP Max Iter reached ==============="<< std::endl; break;
        case QP_DualInfeasible:
            std::cout << "================= Dual Infeasible ================="<< std::endl; break;
        case QP_DualInfeasibleInaccurate:
            std::cout << "============ Dual Infeasible Inaccurate ============"<< std::endl; break;
        case QP_MaxIterReached:
            std::cout << "================ Max Iter reached =================="<< std::endl; break;
        case QP_PrimalInfeasible:
            std::cout << "================= Primal Infeasible ================"<< std::endl; break;
        case QP_PrimalInfeasibleInaccurate:
            std::cout << "=========== Primal Infeasible Inaccurate ============"<< std::endl; break;
        case QP_SolvedInaccurate:
            std::cout << "================= Solved Inaccurate ================="<< std::endl; break;
        case NAN_HESSIAN:
            std::cout << "==================== Nan Hessian ===================="<< std::endl; break;
        case NON_PD_HESSIAN:
            std::cout << "========== Not Possitive Definite Hessian ==========="<< std::endl; break;
        }
        std::cout << "===================================================" << std::endl;
        valid_initial_guess_ = false;
        mpc_return.mpc_horizon = initial_guess_;
    }

    printf("AFTER initial_guess_[0].uk: [%f, %f, %f, %f, %f, %f, %f, %f]\n",
           initial_guess_[0].uk.dq1, initial_guess_[0].uk.dq2, initial_guess_[0].uk.dq3,
           initial_guess_[0].uk.dq4, initial_guess_[0].uk.dq5, initial_guess_[0].uk.dq6,
           initial_guess_[0].uk.dq7, initial_guess_[0].uk.dVs);

    mpc_return = {initial_guess_[0].uk,initial_guess_,time_nmpc};
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

    if(sqp_status == SOLVED || 
       ((sqp_status == MAX_ITER_EXCEEDED || sqp_status == QP_MaxIterReached) && num_valid_guess_failed_ < 5))
    {
        return true;
    }
    else {
        printf("MPC did not solve, status: %d, num_valid_guess_failed: %d\n", sqp_status, num_valid_guess_failed_);
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

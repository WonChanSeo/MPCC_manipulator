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
  robot_(new RobotModel())
{
    initial_guess_.resize(N+1);
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
  robot_(new RobotModel())
{
    initial_guess_.resize(N+1);
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

        double vs_now = project_vs_workspace(cur_x.s, q_now, dq_now);
        cur_x.vs = vs_now;

        // 3) 1-스텝 예측 vs_next → dVs = (vs_next - vs_now)/Ts_
        Eigen::VectorXd dq_next = dq_now;
        for (int j = 0; j < 7; ++j) {
            dq_next[j] = dq_now[j] + ddq_array[j] * Ts_;
            if (dq_now[j] * dq_next[j] < 0.0) dq_next[j] = 0.0; // 0-크로싱 방지
        }

        Eigen::Map<Eigen::VectorXd> ddq_vec(ddq_array.data(), 7);
        Eigen::VectorXd q_next_pred = q_now + dq_now * Ts_ + 0.5 * ddq_vec * (Ts_ * Ts_);
        double s_next_pred = cur_x.s + vs_now * Ts_;

        // 예측 기하 사용(보수적으로 현재 기하 사용해도 됨)
        double vs_next = project_vs_workspace(s_next_pred, q_next_pred, dq_next);

        double dVs = (vs_next - vs_now) / Ts_;
        if (vs_now * vs_next < 0.0) {          // vs 부호 반전 시 정확히 0으로 스냅
            dVs = -vs_now / Ts_;
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
    solver_interface_->solveOCP(initial_guess_, &sqp_status, &time_nmpc, iter_count);
   
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

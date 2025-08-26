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

#include "MPC/mpc.h"

namespace mpcc{
MPC::MPC()
:Ts_(1.0)
{
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

MPC::MPC(double Ts,const PathToJson &path)
:Ts_(Ts),
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
:Ts_(Ts),
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

void MPC::generateNewInitialGuess(const State &x0)
{
    std::cout<< "generate new initial guess!!"<<std::endl;
    for(int i = 0;i<=N;i++)
    {
        initial_guess_[i].xk = x0;
        initial_guess_[i].uk.setZero();
    }
    unwrapInitialGuess();
    valid_initial_guess_ = true;
}

void MPC::generateGuessWithPreviousInput(const State &x0, const Input &u0)
{
    std::cout << "[MPC] Generating fallback initial guess using previous input." << std::endl;

    // 1. 예측 호라이즌의 시작점은 현재 상태(x0)로 설정합니다.
    initial_guess_[0].xk = x0;

    // 2. 예측 호라이즌 전체에 걸쳐 u0를 적용하며 다음 상태를 순차적으로 계산합니다.
    for (int i = 0; i < N; ++i)
    {
        // 제어 입력은 u0로 고정합니다.
        initial_guess_[i].uk = u0;

        // 현재 스텝(i)의 상태와 제어 입력을 사용해 다음 스텝(i+1)의 상태를 계산합니다.
        initial_guess_[i+1].xk = integrator_.RK4(initial_guess_[i].xk, initial_guess_[i].uk, Ts_);
    }

    // 3. 마지막 제어 입력은 0으로 설정합니다.
    initial_guess_[N].uk.setZero();

    // 4. 계산된 경로 변수(s)가 [0, 1] 범위를 벗어나지 않도록 보정합니다.
    unwrapInitialGuess();

    // 5. 새로운 추정치가 생성되었으므로 플래그를 true로 설정합니다.
    valid_initial_guess_ = true;
}

bool MPC::runMPC(MPCReturn &mpc_return, State &x0, Input &u0)
{
    Eigen::MatrixX3d dummy_position;
    // dummy_position << 3,3,3;
    double dummy_radius = 0.;
    return runMPC_(mpc_return, x0, u0, dummy_position, dummy_radius);
}

bool MPC::runMPC_(MPCReturn &mpc_return, State &x0, Input &u0, const Eigen::MatrixX3d &obs_positions, const double &obs_radius)
{
    auto start_mpc = std::chrono::high_resolution_clock::now();
    double last_s = x0.s;
    int iter_count = 0;

    // 1. 현재 상태(x0) 보정 (기존과 동일)
    Eigen::Matrix4d ee_pose = robot_->getEETransformation(stateToJointVector(x0));
    x0.s = track_.projectOnSpline(last_s, ee_pose);

    // ▼▼▼▼▼▼▼▼▼▼▼▼▼ (핵심 순서 변경) ▼▼▼▼▼▼▼▼▼▼▼▼▼
    // 2. 경로 이탈 및 장애물 변경 여부 확인 -> valid_initial_guess_ 플래그 설정
    if(fabs(last_s - x0.s) > param_.max_dist_proj) 
    {
        valid_initial_guess_ = false;
        num_valid_guess_failed_++;
    }

    // ▼▼▼▼▼ [변경] 초기 추정치 생성 로직 수정 ▼▼▼▼▼
    // 3. 플래그 상태에 따라 초기 추정치 전체 시퀀스를 생성/업데이트
    if (valid_initial_guess_) 
    {
        updateInitialGuess(x0); // Warm Start (성공적인 이전 스텝)
    }
    else if (max_iter_failure_)
    {
        // 이전 스텝에서 최대 반복 횟수 초과 실패가 있었을 경우,
        // 이전 제어 입력을 기반으로 한 폴백 추정치를 생성합니다.
        generateGuessWithPreviousInput(x0, u0);
    }
    else 
    {
        // 그 외의 실패(경로 이탈 등)의 경우, 완전히 새로운 추정치를 생성합니다.
        generateNewInitialGuess(x0); // Cold Start
    }
    // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


    // 4. 최신 initial_guess_를 솔버에 전달
    solver_interface_->setInitialGuess(initial_guess_);
    solver_interface_->setCurrentInput(u0);

    // 5. 이제 최신 상태가 반영된 rb_를 사용하여 환경 데이터 설정
    auto start_env = std::chrono::high_resolution_clock::now();
    solver_interface_->setEnvData(obs_positions, obs_radius);
    auto end_env = std::chrono::high_resolution_clock::now();

    // 6. 가장 위험한 장애물이 바뀌었는지 확인하고, 바뀌었다면 다음 스텝을 위해 Cold Start를 강제
    if (solver_interface_->getEnvColNN()->obstacle_switched) {
        std::cout << "[MPC] Obstacle switch detected! Forcing a new initial guess for the next step." << std::endl;
        valid_initial_guess_ = false;
    }
    // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    // 7. QP 문제 풀이
    Status sqp_status;
    ComputeTime time_nmpc;
    solver_interface_->solveOCP(initial_guess_, &sqp_status, &time_nmpc, iter_count);

    // printf("sqp_status: %d\n", sqp_status);
   
    if(sqp_status == SOLVED)
    {
        valid_initial_guess_ = true;
        num_valid_guess_failed_ = 0;

        // ▼▼▼▼▼ 성공했을 때만 최적의 제어 입력을 반환값에 할당합니다. ▼▼▼▼▼
        mpc_return.u0 = initial_guess_[0].uk;
        mpc_return.mpc_horizon = initial_guess_;
    }
    else
    {
        std::cout << "===================================================" << std::endl;
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
        case QP_MaxIterReached:
            std::cout << "================ Max Iter reached =================="<< std::endl;
            break;
        case QP_PrimalInfeasible:
            std::cout << "================= Primal Infeasible ================"<< std::endl;
            break;
        case QP_PrimalInfeasibleInaccurate:
            std::cout << "=========== Primal Infeasible Inaccurate ============"<< std::endl;
            break;
        case QP_SolvedInaccurate:
            std::cout << "================= Solved Inaccurate ================="<< std::endl;
            break;
        case NAN_HESSIAN:
            std::cout << "==================== Nan Hessian ===================="<< std::endl;
            break;
        case NON_PD_HESSIAN:
            std::cout << "========== Not Possitive Definite Hessian ==========="<< std::endl;
            break;
        }
        std::cout << "===================================================" << std::endl;
        
        valid_initial_guess_ = false;

        // ▼▼▼▼▼ [변경] 실패 유형에 따라 플래그 설정 ▼▼▼▼▼
        if (sqp_status == MAX_ITER_EXCEEDED || sqp_status == QP_MaxIterReached)
        {
            max_iter_failure_ = true;
        }
        else
        {
            max_iter_failure_ = false;
        }

        // ... (실패 시 반환값 설정 로직은 이전 답변과 동일하게 유지) ...
        mpc_return.u0 = u0;
        for (auto& guess : initial_guess_) {
            guess.uk = u0;
        }
        mpc_return.mpc_horizon = initial_guess_;
    }

    // 공통적으로 적용되는 반환값들을 설정합니다.
    mpc_return.compute_time = time_nmpc;
    auto end_mpc = std::chrono::high_resolution_clock::now();
    mpc_return.compute_time.total = std::chrono::duration_cast<std::chrono::duration<double>>(end_mpc - start_mpc).count();
    mpc_return.compute_time.set_env = std::chrono::duration_cast<std::chrono::duration<double>>(end_env - start_env).count();
    mpc_return.iter_count = iter_count;

    // 함수 반환 조건은 그대로 유지 (MAX_ITER_EXCEEDED가 5회 미만일 때는 일단 계속 시도)
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


}
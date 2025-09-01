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

// void MPC::generateNewInitialGuess(const State &x0)
// {
//     std::cout<< "generate new initial guess!!"<<std::endl;
//     for(int i = 0;i<=N;i++)
//     {
//         initial_guess_[i].xk = x0;
//         initial_guess_[i].uk.setZero();
//     }
//     unwrapInitialGuess();
//     valid_initial_guess_ = true;
// }


/*
* Braking QP와 같은 폴백 솔버의 성능을 향상시킵니다.
* @param x0 로봇의 현재 상태 (위치 및 속도 포함)
*/
void MPC::generateNewInitialGuess(const State &x0)
{
    std::cout << "====================================================================" << std::endl;
   std::cout << "[MPC] Generating a new initial guess based on maximum deceleration." << std::endl;

   // 1. 예측 호라이즌의 시작점은 현재 상태(x0)로 설정
   initial_guess_[0].xk = x0;

   // 2. 예측 호라이즌 전체에 걸쳐 순차적으로 시뮬레이션
   for (int i = 0; i < N; ++i)
   {
       State& current_state = initial_guess_[i].xk;
       Input& current_input = initial_guess_[i].uk;
       Input& next_input = initial_guess_[i + 1].uk;
        State& next_state = initial_guess_[i + 1].xk;

       std::array<double, 7> ddq_array = {0, 0, 0, 0, 0, 0, 0};

       // 현재 스텝의 속도를 기반으로 최대 감속 제어 입력(가속도) 결정
       for (int j = 0; j < 7; ++j)
       {
           double current_dq = current_input.get_dq(j);

           // 속도가 임계값보다 크면 (움직이고 있으면)
           if (current_dq > 1e-4) // 양의 방향으로 움직일 때
           {
                printf("bounds_param_.get_ddq(%d, \"l\") = %f\n", j, bounds_param_.get_ddq(j, "l"));
               ddq_array[j] = bounds_param_.get_ddq(j, "l"); // 최대 음의 가속도 적용
           }
           else if (current_dq < -1e-4) // 음의 방향으로 움직일 때
           {
                printf("bounds_param_.get_ddq(%d, \"u\") = %f\n", j, bounds_param_.get_ddq(j, "u"));
                ddq_array[j] = bounds_param_.get_ddq(j, "u"); // 최대 양의 가속도 적용
           }
           else // 거의 멈춰있을 때
           {
                ddq_array[j] = 0.0;
           }
       }

       printf("Step %d - Current Input: [%f, %f, %f, %f, %f, %f, %f], ddq: [%f, %f, %f, %f, %f, %f, %f]\n", 
              i, current_input.dq1, current_input.dq2, current_input.dq3, current_input.dq4, current_input.dq5, current_input.dq6, current_input.dq7,
              ddq_array[0], ddq_array[1], ddq_array[2], ddq_array[3], ddq_array[4], ddq_array[5], ddq_array[6]);

       // 3. 현재 상태와 계산된 제어 입력을 사용하여 다음 상태를 계산 (적분)
       next_state = integrator_.RK4(current_state, current_input, Ts_);

       // 4. 속도가 0을 지나 부호가 바뀌었는지 확인하고, 바뀌었다면 0으로 고정 (Overshoot 방지)
       for (int j = 0; j < 7; ++j)
       {
           // 이전 속도와 현재 속도의 부호가 다르면 정지한 것으로 간주
           next_input.set_dq(j, current_input.get_dq(j) + ddq_array[j] * Ts_);

           if (current_input.get_dq(j) * next_input.get_dq(j) < 0.0)
           {
                next_input.set_dq(j, 0.0); // 속도를 0으로 설정
           }
       }
   }

//    // 5. 마지막 제어 입력은 0으로 설정
//    initial_guess_[N].uk.setZero();

//    unwrapInitialGuess();

    std::cout << "====================================================================" << std::endl;
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

    // 3. 플래그 상태에 따라 초기 추정치 전체 시퀀스를 먼저 생성/업데이트
    if(valid_initial_guess_) 
        updateInitialGuess(x0); // Warm Start
    else {
        generateNewInitialGuess(x0); // Cold Start
        updateInitialGuess(x0);
    }

    // 4. 최신 initial_guess_를 솔버에 전달
    solver_interface_->setCurrentInput(u0);
    solver_interface_->setInitialGuess(initial_guess_);


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

    printf("BEFORE initial_guess_[0].uk: [%f, %f, %f, %f, %f, %f, %f, %f]\n", initial_guess_[0].uk.dq1, initial_guess_[0].uk.dq2, initial_guess_[0].uk.dq3, initial_guess_[0].uk.dq4, initial_guess_[0].uk.dq5, initial_guess_[0].uk.dq6, initial_guess_[0].uk.dq7, initial_guess_[0].uk.dVs);

    // 7. QP 문제 풀이
    Status sqp_status;
    ComputeTime time_nmpc;
    solver_interface_->solveOCP(initial_guess_, &sqp_status, &time_nmpc, iter_count);

    // printf("sqp_status: %d\n", sqp_status);
   
    if(sqp_status == SOLVED)
    {
        valid_initial_guess_ = true;
        num_valid_guess_failed_ = 0;

        // // ▼▼▼▼▼ 성공했을 때만 최적의 제어 입력을 반환값에 할당합니다. ▼▼▼▼▼
        // mpc_return.u0 = initial_guess_[0].uk;
        // mpc_return.mpc_horizon = initial_guess_;
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
        // num_valid_guess_failed_++;

        // ▼▼▼▼▼ 실패했을 때는 '정지' 명령(0)을 반환값에 할당합니다. ▼▼▼▼▼
        // mpc_return.u0.setZero();
        // 디버깅 및 시각화를 위해, 솔버가 마지막으로 계산한 예측 경로는 그대로 유지합니다.
        mpc_return.mpc_horizon = initial_guess_;
    }

    // 공통적으로 적용되는 반환값들을 설정합니다.
    printf("AFTER initial_guess_[0].uk: [%f, %f, %f, %f, %f, %f, %f, %f]\n", initial_guess_[0].uk.dq1, initial_guess_[0].uk.dq2, initial_guess_[0].uk.dq3, initial_guess_[0].uk.dq4, initial_guess_[0].uk.dq5, initial_guess_[0].uk.dq6, initial_guess_[0].uk.dq7, initial_guess_[0].uk.dVs);
    mpc_return = {initial_guess_[0].uk,initial_guess_,time_nmpc};
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


}
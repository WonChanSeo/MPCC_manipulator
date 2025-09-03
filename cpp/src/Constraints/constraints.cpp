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

#include "Constraints/constraints.h"
namespace mpcc{
Constraints::Constraints()
{   
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

Constraints::Constraints(double Ts,const PathToJson &path) 
:param_(Param(path.param_path))
{
}

Constraints::Constraints(double Ts,const PathToJson &path,const ParamValue &param_value)
:param_(Param(path.param_path,param_value.param))
{
}

// CORDIC 알고리즘으로 자연로그(ln)를 계산하는 함수
// z의 자연로그, 즉 ln(z)를 계산합니다.
double cordic_log(double z, int iterations = 48) {
    // 로그는 양수에 대해서만 정의됩니다.
    if (z <= 0) {
        return NAN; // Not-a-Number 반환
    }

    // 미리 계산된 atanh(2^-i) 값 테이블
    // 실제 하드웨어에서는 이 값들이 ROM에 저장됩니다.
    static std::vector<double> atanh_lookup;
    if (atanh_lookup.empty()) {
        atanh_lookup.resize(iterations);
        for (int i = 0; i < iterations; ++i) {
            atanh_lookup[i] = atanh(pow(2.0, -(i + 1)));
        }
    }

    // 1. 초기값 설정 (쌍곡선 벡터링 모드)
    // ln(z)를 계산하기 위해 초기 벡터를 (z+1, z-1)로 설정합니다.
    double x = z + 1.0;
    double y = z - 1.0;
    double acc_z = 0.0; // 누적될 결과값

    // 2. CORDIC 반복 계산
    for (int i = 0; i < iterations; ++i) {
        double current_x = x;
        int d = (y > 0) ? -1 : 1; // y를 0으로 만들기 위한 회전 방향 결정

        // x, y 값 업데이트 (곱셈 대신 시프트 연산 시뮬레이션)
        x = current_x + d * y * pow(2.0, -(i + 1));
        y = y + d * current_x * pow(2.0, -(i + 1));

        // z 값(각도) 업데이트
        acc_z = acc_z - d * atanh_lookup[i];

        // 쌍곡선 모드는 수렴을 위해 특정 단계(i=4, 13, 40...)를 반복해야 합니다.
        // 여기서는 간단한 구현을 위해 3k+1 규칙을 적용합니다. (i=3, 6, 9...)
        if ((i + 1) % 3 == 1 && i > 0) {
             current_x = x;
             d = (y > 0) ? -1 : 1;
             x = current_x + d * y * pow(2.0, -(i + 1));
             y = y + d * current_x * pow(2.0, -(i + 1));
             acc_z = acc_z - d * atanh_lookup[i];
        }
    }

    // 3. 최종 결과 반환
    // 누적된 acc_z 값에 2를 곱하면 ln(z)의 근사값이 됩니다.
    return 2.0 * acc_z;
}

// MAC 연산을 시뮬레이션하여 log2(y)를 계산하는 함수
double mac_log2(double y) {
    // 로그는 양수에 대해서만 정의됩니다.
    if (y <= 0) {
        return NAN; // Not-a-Number 반환
    }

    // 1. 계수 (Coefficients)
    // 미리 계산된 다항식 계수들입니다.
    // log2(1+f)를 f가 [0, 1) 범위에 있을 때 근사하는 다항식의 계수입니다.
    // (6차 다항식의 예시, 정밀도를 높이려면 차수를 높여야 함)
    static const std::vector<double> coeffs = {
        1.44266956, -0.72011235, 0.47289345,
       -0.34298104, 0.25146527, -0.180459, 0.126333
    };
    // P(f) = c0*f^1 + c1*f^2 + ... -> log2(1+f)를 근사
    // 호너 방법을 위해 계수 순서를 역으로 사용합니다.
    static const std::vector<double> horner_coeffs = {
        0.126333, -0.180459, 0.25146527, -0.34298104,
        0.47289345, -0.72011235, 1.44266956
    };


    // 2. 범위 축소 (Range Reduction)
    int e;
    // y = m * 2^e 형태로 분해. 여기서 m은 [0.5, 1.0) 범위.
    double m = frexp(y, &e);
    // 다항식이 최적화된 [1.0, 2.0) 범위로 m을 조정
    double m_prime = m * 2.0;
    int e_prime = e - 1;

    // 다항식의 입력으로 사용할 f를 계산 (f는 [0, 1) 범위)
    double f = m_prime - 1.0;

    // 3. 호너 방법 (Horner's Method) - MAC 연산 시뮬레이션
    // log2(m') = log2(1+f) 값을 계산
    double log_m = horner_coeffs[0];
    for (size_t i = 1; i < horner_coeffs.size(); ++i) {
        // 이 부분이 MAC 연산에 해당합니다: result = (result * input) + coeff
        log_m = log_m * f + horner_coeffs[i];
    }
    // 최종 다항식 결과
    log_m = log_m * f;


    // 4. 최종 결과 조합
    // log2(y) = log2(m') + e'
    return log_m + e_prime;
}

// mac_log2 결과를 이용해 자연로그(ln)를 계산하는 최종 함수
double mac_ln(double y) {
    // ln(2)는 상수입니다.
    const double LN2 = 0.6931471805599453;

    // 1. 먼저 밑이 2인 로그를 계산합니다.
    double log2_y = mac_log2(y);

    // 2. 밑 변환 공식을 적용해 자연로그로 변환합니다.
    return log2_y * LN2;
}

double getRBF(const double &delta, const double &h)
{
    // Grandia, Ruben, et al. 
    // "Feedback mpc for torque-controlled legged robots." 
    // 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019.
    double result;
    if (h >= delta) result = -cordic_log(h+1);
    else            result = -cordic_log(delta+1) - 1/(delta+1) * (h-delta) + 1/(2*pow(delta+1,2)) * pow(h-delta,2);
    return result;
}

Eigen::VectorXd getRBF(const Eigen::VectorXd& delta, const Eigen::VectorXd &h)
{
    Eigen::VectorXd result(h.size());
    for(size_t i=0; i<result.size(); i++) result(i) = getRBF(delta(i), h(i));
    return result;
}

double getDRBF(const double &delta, const double &h)
{
    // Grandia, Ruben, et al. 
    // "Feedback mpc for torque-controlled legged robots." 
    // 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019.
    double result;
    if (h >= delta) result = -1/(h+1);
    else            result = -1/(delta+1) + 1/(pow(delta+1,2)) * (h-delta);
    return result;
}

Eigen::VectorXd getDRBF(const Eigen::VectorXd &delta, const Eigen::VectorXd &h)
{
    Eigen::VectorXd result(h.size());
    for(size_t i=0; i<result.size(); i++) result(i) = getDRBF(delta(i), h(i));
    return result;
}

void Constraints::getSelcollConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                                       OneDConstraintInfo *constraint, OneDConstraintsJac* Jac)
{
    // compute self-collision constraints
    // -∇_q Γ(q)^T * q_dot + RBF(Γ(q) - r) <= 0, where r is buffer
    const JointVector q = stateToJointVector(x);
    const dJointVector dq = inputTodJointVector(u);

    // compute minimum distance between each links and its derivative
    double min_dist = 0.01*rb.sel_min_dist_; // unit: [cm] -> [m]
    Eigen::VectorXd d_min_dist = 0.01*rb.d_sel_min_dist_; // unit: [cm]->[m]

    // compute RBF value of minimum distance and its derivative
    double r = param_.tol_selcol*0.01; // buffer [cm] -> [m]
    double delta = -0.5; // switching point of RBF
    double RBF = getRBF(delta, min_dist - r);

    if(constraint)
    {
        constraint->setZero();
        if(k != N)
        {
            constraint->c_l = -INF;
            constraint->c_u = 0.0;
            constraint->c = -d_min_dist.dot(dq) + RBF;
        }
    }
    if(Jac)
    {
        Jac->setZero();
        if(k != N)
        {
            double d_RBF = getDRBF(delta, min_dist - r);
            Jac->c_x_i.block(0,si_index.q1,1,PANDA_DOF) = (d_RBF*d_min_dist).transpose();
            Jac->c_u_i.block(0,si_index.dq1,1,PANDA_DOF) = -d_min_dist.transpose();
        }
    }
    return;
}

void Constraints::getSingularConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                                        OneDConstraintInfo *constraint, OneDConstraintsJac* Jac)
{
    // compute singularity constraints
    // -∇_q μ(q)^T * q_dot + RBF(μ(q) - ɛ) <= 0, where ɛ is buffer
    const dJointVector dq = inputTodJointVector(u);

    //  compute manipulability and its derivative
    double manipulability = rb.manipul_; 
    Eigen::VectorXd d_manipulability = rb.d_manipul_;

    // compute RBF value of manipulability and its derivative
    double eps = param_.tol_sing;    // buffer
    double delta = -0.5;  // switching point of RBF
    double RBF = getRBF(delta, manipulability - eps);

    if(constraint)
    {
        constraint->setZero();
        if(k!=N)
        {
            constraint->c_l = -INF;
            constraint->c_u = 0.0;
            constraint->c = -d_manipulability.dot(dq) + RBF;
        }
    }
    if(Jac)
    {
        Jac->setZero();
        if(k!=N)
        {
            double d_RBF = getDRBF(delta, manipulability - eps);
            Jac->c_x_i.block(0,si_index.q1,1,PANDA_DOF) = (d_RBF*d_manipulability).transpose();
            Jac->c_u_i.block(0,si_index.dq1,1,PANDA_DOF) = -d_manipulability.transpose();
        }
    }
    return;
}

void Constraints::getEnvcollConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                                       XDConstraintInfo *constraint, XDConstraintsJac* Jac)
{
    // compute environment-collision constraints
    // -∇_q Γ(q)^T * q_dot + RBF(Γ(q) - r - ɛ) <= 0, where r is radius of obstacle and ɛ is buffer
    const JointVector q = stateToJointVector(x);
    const dJointVector dq = inputTodJointVector(u);

    // compute minimum distance between each links and its derivative
    Eigen::Matrix<double, PANDA_NUM_LINKS, 1> min_dist = 0.01*(rb.env_min_dist_ - Eigen::Matrix<double, PANDA_NUM_LINKS, 1>::Constant(rb.obs_radius_*1.2)); // unit: [cm]->[m]
    Eigen::Matrix<double, PANDA_NUM_LINKS, PANDA_DOF> d_min_dist = 0.01*rb.d_env_min_dist_; // unit: [cm]->[m]

    // compute RBF value of minimum distance and its derivative
    double r = 0.01*param_.tol_envcol; //  [cm]->[m]
    double delta = -0.5; // switching point of RBF
    Eigen::Matrix<double, PANDA_NUM_LINKS, 1> RBF = getRBF(Eigen::Matrix<double, PANDA_NUM_LINKS, 1>::Constant(delta), 
                                                           min_dist - Eigen::Matrix<double, PANDA_NUM_LINKS, 1>::Constant(r));
    

    if(constraint)
    {
        constraint->setZero(PANDA_NUM_LINKS);
        if(k != N)
        {
            constraint->c_l = Eigen::Matrix<double, PANDA_NUM_LINKS, 1>::Constant(-INF);
            constraint->c_u = Eigen::Matrix<double, PANDA_NUM_LINKS, 1>::Zero();
            constraint->c = -d_min_dist*dq + RBF;
        }
    }
    if(Jac)
    {
        Jac->setZero(PANDA_NUM_LINKS);
        if(k != N)
        {
            Eigen::Matrix<double, PANDA_NUM_LINKS, 1> d_RBF = getDRBF(Eigen::Matrix<double, PANDA_NUM_LINKS, 1>::Constant(delta), 
                                                                      min_dist - Eigen::Matrix<double, PANDA_NUM_LINKS, 1>::Constant(r));
            Jac->c_x_i.block(0,si_index.q1,PANDA_NUM_LINKS,PANDA_DOF) = d_RBF.asDiagonal()*d_min_dist;
            Jac->c_u_i.block(0,si_index.dq1,PANDA_NUM_LINKS,PANDA_DOF) = -d_min_dist;
        }
    }
    return;
}

void Constraints::getConstraints(const State &x,const Input &u,const RobotData &rb,int k,
                                 ConstraintsInfo *constraint, ConstraintsJac* Jac)
{
    // compute all the polytopic state constraints
    // compute the three constraints
    OneDConstraintInfo constraint_selcol, constraint_sing;
    OneDConstraintsJac jac_selcol, jac_sing;
    XDConstraintInfo constraint_envcol;
    XDConstraintsJac jac_envcol;

    if(Jac)
    {
        getSelcollConstraint(x, u, rb, k, &constraint_selcol, &jac_selcol);
        getSingularConstraint(x, u, rb, k, &constraint_sing, &jac_sing);
        getEnvcollConstraint(x, u, rb, k, &constraint_envcol, &jac_envcol);
    }
    else
    {
        getSelcollConstraint(x, u, rb, k, &constraint_selcol, NULL);
        getSingularConstraint(x, u, rb, k, &constraint_sing, NULL);
        getEnvcollConstraint(x, u, rb, k, &constraint_envcol, NULL);
    }

    if(constraint)
    {
        constraint->setZero();
        constraint->c_vec(si_index.con_selcol) = constraint_selcol.c;
        constraint->c_vec(si_index.con_sing) = constraint_sing.c;
        constraint->c_vec.segment(si_index.con_envcol1, PANDA_NUM_LINKS) = constraint_envcol.c;

        constraint->c_lvec(si_index.con_selcol) = constraint_selcol.c_l;
        constraint->c_lvec(si_index.con_sing) = constraint_sing.c_l;
        constraint->c_lvec.segment(si_index.con_envcol1, PANDA_NUM_LINKS) = constraint_envcol.c_l;

        constraint->c_uvec(si_index.con_selcol) = constraint_selcol.c_u;
        constraint->c_uvec(si_index.con_sing) = constraint_sing.c_u;
        constraint->c_uvec.segment(si_index.con_envcol1, PANDA_NUM_LINKS) = constraint_envcol.c_u;
    }
    
    if(Jac)
    {
        Jac->setZero();
        Jac->c_x.row(si_index.con_selcol) = jac_selcol.c_x_i;
        Jac->c_x.row(si_index.con_sing) = jac_sing.c_x_i;
        Jac->c_x.block(si_index.con_envcol1, 0, PANDA_NUM_LINKS, NX) = jac_envcol.c_x_i;

        Jac->c_u.row(si_index.con_selcol) = jac_selcol.c_u_i;
        Jac->c_u.row(si_index.con_sing) = jac_sing.c_u_i;
        Jac->c_u.block(si_index.con_envcol1, 0, PANDA_NUM_LINKS, NU) = jac_envcol.c_u_i;
    }
    return;
}
}
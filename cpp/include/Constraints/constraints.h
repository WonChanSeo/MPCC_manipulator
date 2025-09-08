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

#ifndef MPCC_CONSTRAINTS_H
#define MPCC_CONSTRAINTS_H

#include "config.h"
#include "Spline/arc_length_spline.h"
#include "Model/model.h"
#include "Model/robot_data.h"
namespace mpcc{

/// @brief Constraint Information
/// @param c_vec (Eigen::Matrix<double,NPC,1>) c(x,u) value; l < c(x,u) < u
/// @param c_lvec (Eigen::Matrix<double,NPC,1>) lower value
/// @param c_uvec (Eigen::Matrix<double,NPC,1>) upper value
struct ConstraintsInfo{
    Eigen::Matrix<double,NPC,1> c_vec;
    Eigen::Matrix<double,NPC,1> c_lvec;
    Eigen::Matrix<double,NPC,1> c_uvec;

    void setZero()
    {
        c_vec.setZero();
        c_lvec.setZero();
        c_uvec.setZero();
    }
};

/// @brief 1-D Constraint Information
/// @param c (double) c(x,u) value
/// @param c_l (double) lower value
/// @param c_u (double) upper value
struct OneDConstraintInfo{
    double c;
    double c_l;
    double c_u;

    void setZero()
    {
        c = 0.0;
        c_l = 0.0;
        c_u = 0.0;
    }
};

/// @brief X-D Constraint Information
/// @param c (Eigen::VectorXd) c(x,u) value
/// @param c_l (Eigen::VectorXd) lower value
/// @param c_u (Eigen::VectorXd) upper value
struct XDConstraintInfo{
    Eigen::VectorXd c;
    Eigen::VectorXd c_l;
    Eigen::VectorXd c_u;

    void setZero(const int &size)
    {
        c.setZero(size);
        c_l.setZero(size);
        c_u.setZero(size);
    }
};

/// @brief Jacobian of Constraint
/// @param c_x_i (Eigen::Matrix<double,NPC,NX>) jacobian of c(x,u) wrt state
/// @param c_u_i (Eigen::Matrix<double,NPC,NU>) jacobian of c(x,u) wrt input
struct ConstraintsJac{
    C_MPC c_x;
    D_MPC c_u;

    void setZero()
    {
        c_x.setZero();
        c_u.setZero();
    }
};

/// @brief 1-D Jacobian of Constraint
/// @param c_x_i (Eigen::Matrix<double,1,NX>) jacobian of c(x,u) wrt state
/// @param c_u_i (Eigen::Matrix<double,1,NU>) jacobian of c(x,u) wrt input
struct OneDConstraintsJac{
    C_i_MPC c_x_i;
    D_i_MPC c_u_i;

    void setZero()
    {
        c_x_i.setZero();
        c_u_i.setZero();
    }
};

/// @brief X-D Jacobian of Constraint
/// @param c_x_i (Eigen::MatrixXd) jacobian of c(x,u) wrt state
/// @param c_u_i (Eigen::MatrixXd) jacobian of c(x,u) wrt input
struct XDConstraintsJac{
    Eigen::MatrixXd c_x_i;
    Eigen::MatrixXd c_u_i;

    void setZero(const int &size)
    {
        c_x_i.setZero(size, NX);
        c_u_i.setZero(size, NU);
    }
};

/// @brief compute Relaxed Barrier Function of h
/// @param delta (double) switching point from logarithm to quadratic function
/// @param h (double) input value
/// @return (double) RBF value
float getRBF(const float &delta, const float &h);

/// @brief compute Relaxed Barrier Function of h
/// @param delta (Eigen::VectorXd) switching point from logarithm to quadratic function
/// @param h (Eigen::VectorXd) input value
/// @return (Eigen::VectorXd) RBF value
Eigen::VectorXf getRBF(const Eigen::VectorXf& delta, const Eigen::VectorXf &h);

/// @brief compute derivative ofRelaxed Barrier Function wrt h
/// @param delta (float) switching point from logarithm to quadratic function
/// @param h (float) input value
/// @return (float) derivation RBF value
float getDRBF(const float &delta, const float &h);

/// @brief compute derivative ofRelaxed Barrier Function wrt h
/// @param delta (Eigen::VectorXf) switching point from logarithm to quadratic function
/// @param h (Eigen::VectorXf) input value
/// @return (Eigen::VectorXf) derivation RBF value
Eigen::VectorXf getDRBF(const Eigen::VectorXf &delta, const Eigen::VectorXf &h);

class Constraints {
public:
    Constraints();
    Constraints(float Ts,const PathToJson &path);
    Constraints(float Ts,const PathToJson &path,const ParamValue &param_value);
    
    /// @brief compute all the polytopic state constraints given current state
    /// @param x (State) current state
    /// @param u (ControlInput) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param constraint (*ConstraintsInfo) constraint information(c,l,u) wrt state and input 
    /// @param Jac (*ConstraintsJac) jacobian of constraint value wrt state and input 
    void getConstraints(const State &x,const Input &u,const RobotData &rb,int k,
                        ConstraintsInfo *constraint, ConstraintsJac* Jac);

private:

    /// @brief compute self-collision inequality constraint given current state and input
    /// @param x (State) current state
    /// @param u (ControlInput) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param constraint (*OneDConstraintInfo) constraint information(c,l,u) wrt state and input 
    /// @param Jac (*OneDConstraintsJac) jacobian of constraint value wrt state and input 
    void getSelcollConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                              OneDConstraintInfo *constraint, OneDConstraintsJac* Jac);

    /// @brief compute Singularity inequality constraint given current state and input
    /// @param x (State) current state
    /// @param u (ControlInput) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param constraint (*OneDConstraintInfo) constraint information(c,l,u) wrt state and input 
    /// @param Jac (*OneDConstraintsJac) jacobian of constraint value wrt state and input 
    void getSingularConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                               OneDConstraintInfo *constraint, OneDConstraintsJac* Jac);

    /// @brief compute environment-collision inequality constraint given current state and input
    /// @param x (State) current state
    /// @param u (ControlInput) current control input
    /// @param rb (RobotData) kinemetic information (ex. EE-pose, Jacobian, ...) wrt current state
    /// @param k (int) receding horizon index
    /// @param constraint (*XDConstraintInfo) constraint information(c,l,u) wrt state and input 
    /// @param Jac (*XDConstraintsJac) jacobian of constraint value wrt state and input 
    void getEnvcollConstraint(const State &x,const Input &u,const RobotData &rb,int k,
                              XDConstraintInfo *constraint, XDConstraintsJac* Jac);


    Param param_;
};
}

#endif //MPCC_CONSTRAINTS_H

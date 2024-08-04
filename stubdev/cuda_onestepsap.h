#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

#ifndef dt
#define dt 0.01
#endif

#ifndef gravity
#define gravity -9.81
#endif

// define a constraint data structure
struct ConstraintData {
  Eigen::MatrixXf J;
  std::vector<Eigen::MatrixXf> G;

  Eigen::VectorXf phi0;
  Eigen::VectorXf contact_stiffness;
  Eigen::VectorXf contact_damping;

  int num_active_contacts;
};

struct SAPCPUData {
  Eigen::MatrixXf dynamics_matrix;  // Dynamics matrix A
  Eigen::MatrixXf v_star;           // Free motion velocity v*
  Eigen::MatrixXf v_guess;
  ConstraintData constraint_data;
  std::vector<Eigen::Vector3f> gamma;  // impulse data vector
  std::vector<Eigen::Vector3f> R;      // regularization matrix vector

  int num_contacts;
  int num_velocities;
  int num_problems;
};

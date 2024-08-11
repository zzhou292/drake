#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

float MatrixSolve(std::vector<Eigen::MatrixXf>& M,
                  std::vector<Eigen::MatrixXf>& b,
                  std::vector<Eigen::MatrixXf>& x);

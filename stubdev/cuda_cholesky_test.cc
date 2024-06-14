#include "stubdev/cuda_cholesky.h"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, Cholesky) {
  const int N = 50;
  int num_problems = 5;
  std::vector<Eigen::MatrixXd> M;
  std::vector<Eigen::VectorXd> b;
  std::vector<Eigen::VectorXd> x;
  for (int i = 0; i < num_problems; ++i) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    M.push_back(A.transpose() * A);
    x.push_back(Eigen::VectorXd::Random(N));
    b.push_back(M[i] * x[i]);
  }

  MatrixSolve(M, b, x);

  for (int i = 0; i < num_problems; ++i) {
    Eigen::VectorXd error = M[i] * x[i] - b[i];
    EXPECT_LT(error.norm(), 1e-10);
  }
}

}  // namespace
}  // namespace drake

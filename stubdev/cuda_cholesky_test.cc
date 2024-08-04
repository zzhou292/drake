#include "stubdev/cuda_cholesky.h"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, Cholesky) {
  const int N = 66;
  int num_problems = 1000;
  std::vector<Eigen::MatrixXf> M;
  std::vector<Eigen::MatrixXf> b;
  std::vector<Eigen::MatrixXf> x;
  for (int i = 0; i < num_problems; ++i) {
    Eigen::MatrixXf A;
    A = Eigen::MatrixXf::Random(N, N);
    M.push_back(A.transpose() * A);
    x.push_back(Eigen::MatrixXf::Random(N, 1));
    b.push_back(M[i] * x[i]);
  }

  MatrixSolve(M, b, x);

  for (int i = 0; i < num_problems; ++i) {
    std::cout << x[i](0, 0) << std::endl;
    Eigen::MatrixXf error = M[i] * x[i] - b[i];

    for (int j = 0; j < N; ++j) {
      std::cout << error(j, 0) << std::endl;
    }
  }
}

}  // namespace
}  // namespace drake

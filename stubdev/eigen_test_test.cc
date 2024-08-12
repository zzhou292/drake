#include "stubdev/eigen_test.h"

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, Cholesky) {
  // EigenTest();
  EigenRowTest();
  double sum = 0.f;
  for (int i = 0; i < 100; i++) {
    sum += EigenRowTest();
  }
  std::cout << "Eigen Row avg_time:" << sum / 100 << std::endl;

  EigenTest();
  sum = 0.f;
  for (int i = 0; i < 100; i++) {
    sum += EigenTest();
  }
  std::cout << "Eigen No Row avg_time:" << sum / 100 << std::endl;
}

}  // namespace
}  // namespace drake

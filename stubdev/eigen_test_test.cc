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
  float sum = 0.f;
  for (int i = 0; i < 50; i++) {
    // sum += EigenTest();
    sum += EigenRowTest();
  }
  std::cout << "avg_time:" << sum / 50 << std::endl;
}

}  // namespace
}  // namespace drake

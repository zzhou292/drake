#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>

// Kernel function to perform matrix-vector multiplication
__global__ void matVecMultiply(float* matrix_ptr, float* vector_ptr,
                               float* result_ptr, int N) {
  Eigen::Map<Eigen::MatrixXf> matrix(matrix_ptr, N, N);
  Eigen::Map<Eigen::MatrixXf> vector(vector_ptr, N, 1);
  Eigen::Map<Eigen::MatrixXf> result(result_ptr, N, 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    float sum = 0.0;
    for (int j = 0; j < N; ++j) {
      sum += matrix(idx, j) * vector(j, 0);
    }
    result(idx, 0) = sum;
  }
}

__global__ void matVecMultiplyRow(float* matrix_ptr, float* vector_ptr,
                                  float* result_ptr, int N) {
  Eigen::Map<Eigen::MatrixXf> matrix(matrix_ptr, N, N);
  Eigen::Map<Eigen::MatrixXf> vector(vector_ptr, N, 1);
  Eigen::Map<Eigen::MatrixXf> result(result_ptr, N, 1);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    float sum = 0.0;
    sum += (matrix.row(idx) * vector)(0, 0);
    result(idx, 0) = sum;
  }
}
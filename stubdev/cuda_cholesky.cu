#include <stdio.h>

#include <iostream>

#include "cuda_cholesky.cuh"
#include "cuda_cholesky.h"
// CUDA error handeling
// =====================
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void CholeskySolveKernel(float* M, float* L, float* b, float* x,
                                    float* y, size_t num_problems, size_t n) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (equ_idx >= num_problems) {
    return;
  }

  Eigen::Map<Eigen::MatrixXf> d_M(M + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXf> d_L(L + equ_idx * n * n, n, n);
  Eigen::Map<Eigen::MatrixXf> d_b(b + equ_idx * n, n, 1);
  Eigen::Map<Eigen::MatrixXf> d_x(x + equ_idx * n, n, 1);
  Eigen::Map<Eigen::MatrixXf> d_y(y + equ_idx * n, n, 1);

  int num_stride = (n + 31) / 32;

  CholeskyFactorizationFunc(d_M, d_L, equ_idx, thread_idx, n, num_stride);
  __syncwarp();
  CholeskySolveForwardFunc(d_L, d_b, d_y, equ_idx, thread_idx, n, num_stride);
  __syncwarp();
  CholeskySolveBackwardFunc(d_L, d_y, d_x, equ_idx, thread_idx, n, num_stride);
  __syncwarp();
}

// Main solve function - including memory allocation, copy, and kernel calls
float MatrixSolve(std::vector<Eigen::MatrixXf>& M,
                  std::vector<Eigen::MatrixXf>& b,
                  std::vector<Eigen::MatrixXf>& x) {
  const int num_problems = M.size();
  const int n = b[0].size();

  float* x_result = new float[num_problems * n];

  // Allocate device arrays
  float *d_M, *d_b, *d_y, *d_x, *d_L;
  HANDLE_ERROR(cudaMalloc((void**)&d_M, sizeof(float) * num_problems * n * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_L, sizeof(float) * num_problems * n * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(float) * num_problems * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_x, sizeof(float) * num_problems * n));
  HANDLE_ERROR(cudaMalloc((void**)&d_y, sizeof(float) * num_problems * n));

  // Set d_L and d_x to be 0
  HANDLE_ERROR(cudaMemset(d_L, 0, sizeof(float) * num_problems * n * n));
  HANDLE_ERROR(cudaMemset(d_x, 0, sizeof(float) * num_problems * n));
  HANDLE_ERROR(cudaMemset(d_y, 0, sizeof(float) * num_problems * n));

  // Copy to device
  for (int i = 0; i < num_problems; ++i) {
    HANDLE_ERROR(cudaMemcpy(d_M + i * n * n, M[i].data(), sizeof(float) * n * n,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b + i * n, b[i].data(), sizeof(float) * n,
                            cudaMemcpyHostToDevice));
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Matrix Cholesky factorization
  CholeskySolveKernel<<<num_problems, 32>>>(d_M, d_L, d_b, d_x, d_y,
                                            num_problems, n);

  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time for Cholesky Solve: " << milliseconds << " ms\n";

  // Copy to host
  HANDLE_ERROR(cudaMemcpy(x_result, d_x, sizeof(float) * num_problems * n,
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_problems; ++i) {
    Eigen::Map<Eigen::MatrixXf> x_result_i(x_result + i * n, n, 1);
    x[i] = x_result_i;
  }

  return 0;
}

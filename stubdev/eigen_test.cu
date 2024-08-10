#include <stdio.h>

#include <iostream>

#include "eigen_test.cuh"
#include "eigen_test.h"
#include <cuda_runtime.h>
// CUDA error handeling
// =====================
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// Main solve function - including memory allocation, copy, and kernel calls
float EigenTest() {
  const int N = 40960;
  const int sizeMatrix = N * N * sizeof(float);
  const int sizeVector = N * sizeof(float);

  // Allocate host memory
  float* h_matrix = new float[N * N];
  float* h_vector = new float[N];
  float* h_result = new float[N];

  // Initialize host memory
  for (int i = 0; i < N; ++i) {
    h_vector[i] = 1.0f;  // example values for the vector
    for (int j = 0; j < N; ++j) {
      h_matrix[i * N + j] = 1.0f;  // example values for the matrix
    }
  }

  // Allocate device memory
  float* d_matrix;
  float* d_vector;
  float* d_result;
  cudaMalloc((void**)&d_matrix, sizeMatrix);
  cudaMalloc((void**)&d_vector, sizeVector);
  cudaMalloc((void**)&d_result, sizeVector);

  // Copy data from host to device
  cudaMemcpy(d_matrix, h_matrix, sizeMatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector, h_vector, sizeVector, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  size_t threads_per_block = 64;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record the start event
  cudaEventRecord(start, 0);

  // Launch the kernel
  matVecMultiply<<<number_of_blocks, threads_per_block>>>(d_matrix, d_vector,
                                                          d_result, N);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Record the stop event
  cudaEventRecord(stop, 0);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate the elapsed time
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Destroy the events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy result from device to host
  cudaMemcpy(h_result, d_result, sizeVector, cudaMemcpyDeviceToHost);

  // Print some of the results
  //   for (int i = 0; i < 5; ++i) {
  //     std::cout << "Result[" << i << "] = " << h_result[i] << std::endl;
  //     std::cout << "Result[" << N - 1 - i << "] = " << h_result[i] <<
  //     std::endl;
  //   }

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_vector);
  cudaFree(d_result);

  // Free host memory
  free(h_matrix);
  free(h_vector);
  free(h_result);

  return elapsedTime;
}

// Main solve function - including memory allocation, copy, and kernel calls
float EigenRowTest() {
  const int N = 40960;
  const int sizeMatrix = N * N * sizeof(float);
  const int sizeVector = N * sizeof(float);

  // Allocate host memory
  float* h_matrix = new float[N * N];
  float* h_vector = new float[N];
  float* h_result = new float[N];

  // Initialize host memory
  for (int i = 0; i < N; ++i) {
    h_vector[i] = 1.0f;  // example values for the vector
    for (int j = 0; j < N; ++j) {
      h_matrix[i * N + j] = 1.0f;  // example values for the matrix
    }
  }

  // Allocate device memory
  float* d_matrix;
  float* d_vector;
  float* d_result;
  cudaMalloc((void**)&d_matrix, sizeMatrix);
  cudaMalloc((void**)&d_vector, sizeVector);
  cudaMalloc((void**)&d_result, sizeVector);

  // Copy data from host to device
  cudaMemcpy(d_matrix, h_matrix, sizeMatrix, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector, h_vector, sizeVector, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  size_t threads_per_block = 32;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record the start event
  cudaEventRecord(start, 0);

  // Launch the kernel
  matVecMultiplyRow<<<number_of_blocks, threads_per_block>>>(d_matrix, d_vector,
                                                             d_result, N);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Record the stop event
  cudaEventRecord(stop, 0);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate the elapsed time
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  // Destroy the events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy result from device to host
  cudaMemcpy(h_result, d_result, sizeVector, cudaMemcpyDeviceToHost);

  // Print some of the results
  //   for (int i = 0; i < 5; ++i) {
  //     std::cout << "Result[" << i << "] = " << h_result[i] << std::endl;
  //     std::cout << "Result[" << N - 1 - i << "] = " << h_result[i] <<
  //     std::endl;
  //   }

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_vector);
  cudaFree(d_result);

  // Free host memory
  free(h_matrix);
  free(h_vector);
  free(h_result);

  return elapsedTime;
}

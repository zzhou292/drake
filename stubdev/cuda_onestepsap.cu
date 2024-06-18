
#include <iostream>

#include "cuda_cholesky.cuh"
#include "cuda_onestepsap.cuh"
#include <cuda_runtime.h>

// ========================================================================
// OneStepSapGPU Kernels and Functions with new data struct
// ========================================================================

// ========================================================================
// Device Functions
// ========================================================================
// Device function to perform SAPXY operation
// Matrix A, B and C are represented by Eigen::MatrixXd
// C = alpha * A + B
__device__ void SAXPY(double alpha, const Eigen::Map<Eigen::MatrixXd> A,
                      const Eigen::Map<Eigen::MatrixXd> B,
                      Eigen::Map<Eigen::MatrixXd> C) {
  int thread_idx = threadIdx.x;
  int row = A.rows();
  int col = A.cols();

  int num_strides = (A.rows() + 31) / 32;

  for (int i = 0; i < num_strides; i++) {
    int cur_idx = i * 32 + thread_idx;
    if (cur_idx >= row * col) continue;
    int cur_col = cur_idx / row;
    int cur_row = cur_idx % row;

    if (cur_row < row && cur_col < col) {
      C(cur_row, cur_col) = alpha * A(cur_row, cur_col) + B(cur_row, cur_col);
    }
  }
}

// Device function to calculate alpha*(A*B) = C
// A and B are const inputs, C is mutable.
__device__ void MMultiply(double alpha, const Eigen::Map<Eigen::MatrixXd> A,
                          const Eigen::Map<Eigen::MatrixXd> B,
                          Eigen::Map<Eigen::MatrixXd> C, double* sums) {
  int A_row = A.rows();
  int A_col = A.cols();
  int B_col = B.cols();
  int stride = (A_row + 31) / 32;
  int thread_idx = threadIdx.x;

  for (int k = 0; k < B_col; k++) {
    for (int j = 0; j < A_col; j++) {
      for (int i = 0; i < stride; i++) {
        int row = i * 32 + thread_idx;
        int col = j;
        if (row < A_row) {
          if (j == 0) {
            sums[row] = 0.0;
          }

          sums[row] += A(row, col) * B(col, k);

          if (col == A_col - 1) {
            C(row, k) = alpha * sums[row];
          }
        }
      }
    }
  }
}

// Sets lambda_r = 0.5 * gamma.transpose() * R * gamma by modifying `data`
__device__ void CalcRegularizationCost(SAPGPUData* data) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    sum += 0.5 * data->gamma(i).dot(data->R(i).cwiseProduct(data->gamma(i)));
  }
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
  if (threadIdx.x == 0) {
    data->regularizer_cost()(0, 0) = sum;
  }
}

// Contruct Hessian, H = dynamics_matrix + J * G * J^T
__device__ void CalculateHessian(SAPGPUData* data) {
  int num_stride = ((data->NumVelocities() * data->NumVelocities()) + 31) / 32;
  for (int i = 0; i < num_stride; i++) {
    int cur_idx = i * 32 + threadIdx.x;
    if (cur_idx >= data->NumVelocities() * data->NumVelocities()) return;
    int cur_col = cur_idx / (data->NumVelocities());
    int cur_row = cur_idx % (data->NumVelocities());

    if (cur_row < data->NumVelocities() && cur_col < data->NumVelocities()) {
      data->H()(cur_row, cur_col) =
          data->dynamics_matrix()(cur_row, cur_col) +
          data->J().col(cur_row).dot(data->G_J().col(cur_col));
    }
  }
}

__device__ void CalcMomentumCost(SAPGPUData* data, double* sums) {
  // Calculate velocity gain
  SAXPY(-1.0, data->v_star(), data->v_guess(), data->velocity_gain());

  // Calculate momentum gain
  MMultiply(1.0, data->dynamics_matrix(), data->velocity_gain(),
            data->momentum_gain(), sums);

  // Calculate momentum cost
  MMultiply(0.5, data->velocity_gain_transpose(), data->momentum_gain(),
            data->momentum_cost(), sums);
}

// ========================================================================
// Kernels
// ========================================================================

// Kernel function serving as a wrapper
__global__ void CalcMomentumCostKernel(SAPGPUData* data) {
  extern __shared__ double sums[];

  CalcMomentumCost(data, sums);
}

__global__ void CalcRegularizerCostKernel(SAPGPUData* data) {
  extern __shared__ double sums[];
  int equ_idx = blockIdx.x;
  int num_problems = data->NumProblems();
  int num_contacts = data->NumContacts();

  // Calculate regularization cost
  CalcRegularizationCost(data);
}

__global__ void CalcHessianKernel(SAPGPUData* data) {
  extern __shared__ double sums[];
  int equ_idx = blockIdx.x;
  int num_problems = data->NumProblems();
  int num_contacts = data->NumContacts();

  // Calculate G*J
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    int J_row = i * 3;
    // do a simple matrix multiplication of 3x3 multiplied by 3 by
    // 3*num_velocities
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < data->NumVelocities(); b++) {
        data->G_J()(J_row + a, b) = 0;
        for (int c = 0; c < 3; c++) {
          data->G_J()(J_row + a, b) +=
              data->G(i)(a, c) * data->J()(J_row + c, b);
        }
      }
    }
  }

  // Calculate data->H() = J_transpose * (G*J)
  CalculateHessian(data);
}

// Calculate -grad
__global__ void CalcNegGradKernel(SAPGPUData* data) {
  extern __shared__ double sums[];
  int equ_idx = blockIdx.x;
  int num_problems = data->NumProblems();
  int num_contacts = data->NumContacts();

  for (int i = threadIdx.x; i < data->NumVelocities(); i += blockDim.x) {
    double sum = 0.0;
    for (int j = 0; j < 3 * data->NumContacts(); j++) {
      sum += data->J()(j, i) * data->gamma_full()(j);
    }
    data->neg_grad()(i, 0) = -(data->momentum_gain()(i, 0) - sum);
  }
}

// Solve for Hx = -grad
__global__ void SAPCholeskySolveKernel(SAPGPUData* data) {
  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  int num_stride = (data->NumVelocities() + 31) / 32;

  Eigen::Map<Eigen::MatrixXd> d_M = data->H();
  Eigen::Map<Eigen::MatrixXd> d_L = data->chol_L();
  Eigen::Map<Eigen::VectorXd> d_b = data->neg_grad();
  Eigen::Map<Eigen::VectorXd> d_x = data->chol_x();
  Eigen::Map<Eigen::VectorXd> d_y = data->chol_y();

  CholeskyFactorizationFunc(d_M, d_L, equ_idx, thread_idx,
                            data->NumVelocities(), num_stride);
  __syncwarp();
  CholeskySolveForwardFunc(d_L, d_b, d_y, equ_idx, thread_idx,
                           data->NumVelocities(), num_stride);
  __syncwarp();
  CholeskySolveBackwardFunc(d_L, d_y, d_x, equ_idx, thread_idx,
                            data->NumVelocities(), num_stride);
  __syncwarp();
}

// ==========================================================================
// Scratch space starting Jun 17
// Note that eval cost will overwrite all cost-related values in global memory
__device__ void SAPLineSearchEvalCost(SAPGPUData* data, double alpha, double* f,
                                      double* sums,
                                      Eigen::Map<Eigen::MatrixXd> dv_alpha,
                                      Eigen::Map<Eigen::MatrixXd> v_alpha) {
  Eigen::Map<Eigen::MatrixXd> x_dir(data->chol_x().data(),
                                    data->NumVelocities(), 1);
  SAXPY(alpha, x_dir, data->v_guess(), v_alpha);

  if (threadIdx.x < data->NumVelocities()) {
    data->v_guess()(threadIdx.x, 0) = v_alpha(threadIdx.x, 0);
  }
  __syncwarp();

  CalcMomentumCost(data, sums);

  __syncwarp();

  CalcRegularizationCost(data);

  __syncwarp();

  if (threadIdx.x == 0) {
    *f = data->momentum_cost()(0, 0) + data->regularizer_cost()(0, 0);
  }

  __syncwarp();
}

__device__ void SAPLineSearchEvalDer(SAPGPUData* data, double alpha, double* d,
                                     double* sums,
                                     Eigen::Map<Eigen::MatrixXd> dv_alpha,
                                     Eigen::Map<Eigen::MatrixXd> v_alpha,
                                     double* dmid_1, double* dmid_2) {
  double res = 0.0;
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    res += 0.5 * data->momentum_gain()(i, 0) * dv_alpha(i, 0);
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  if (threadIdx.x == 0) {
    *dmid_1 = res;
  }

  __syncwarp();

  // TODO: calculate dmid_2
}

__device__ void SAPLineSearchEval2Der(SAPGPUData* data, double alpha,
                                      double* d2, double* sums,
                                      Eigen::Map<Eigen::MatrixXd> dv_alpha,
                                      Eigen::Map<Eigen::MatrixXd> v_alpha,
                                      double* ddmid_1, double* ddmid_2) {
  double res = 0.0;

  // calculate ddmid_1 = dv_alpha.transpose() * momentum_gain
  for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
    res += dv_alpha(i, 0) * data->momentum_gain()(i, 0);
  }
  res += __shfl_down_sync(0xFFFFFFFF, res, 16);
  res += __shfl_down_sync(0xFFFFFFFF, res, 8);
  res += __shfl_down_sync(0xFFFFFFFF, res, 4);
  res += __shfl_down_sync(0xFFFFFFFF, res, 2);
  res += __shfl_down_sync(0xFFFFFFFF, res, 1);

  if (threadIdx.x == 0) {
    *ddmid_1 = res;
  }

  __syncwarp();

  // TODO: calculate ddmid_2
}

__global__ void SAPLineSearchKernel(SAPGPUData* data) {
  // scratch space for each block (each problem)
  // 0 - l_alpha
  // 1 - r_alpha
  // 2 - mid_alpha
  // 3 - fl
  // 4 - fr
  // 5 - fmid
  // 6 - dmid
  // 7 - d2mid
  // 8 - dmid_1 (momentum_gain.transpose() * dv(alpha))
  // 9 - dmid_2 ((J*dv(alpha)).transpose() * gamma(alpha))
  // 10 - ddmid_1 (dv_alpha.transpose() * momentum_gain)
  // 11 - ddmid_2 (J*dv_alpha).transpose() * G * (J*dv_alpha)
  // [10, (num_velocities + 10)) - dv(alpha)
  // [(num_velocities + 10), (2*num_velocities + 10)) - v(alpha)

  extern __shared__ double buff[];
  size_t buff_arr_size = 2;
  size_t buff_arr_offset = 12;

  int equ_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  if (thread_idx == 0) {
    buff[0] = data->l_alpha();
    buff[1] = data->r_alpha();
    buff[2] = (buff[0] + buff[1]) / 2.0;
  }

  double* sums =
      buff + (buff_arr_size * data->NumVelocities() + buff_arr_offset);

  Eigen::Map<Eigen::MatrixXd> dv_alpha(
      buff + buff_arr_offset, data->NumVelocities(),
      1);  // scratch space, needs to be calculated on the fly
  Eigen::Map<Eigen::MatrixXd> v_alpha(
      buff + buff_arr_offset + data->NumVelocities(), data->NumVelocities(), 1);
  double* dmid_1 = buff + 8;
  double* dmid_2 = buff + 9;
  double* ddmid_1 = buff + 10;
  double* ddmid_2 = buff + 11;

  // TODO: Update G and gamma

  // evaluate the cost function at l_alpha and r_alpha
  SAPLineSearchEvalCost(data, buff[0], &buff[3], sums, dv_alpha, v_alpha);
  SAPLineSearchEvalCost(data, buff[1], &buff[4], sums, dv_alpha, v_alpha);

  // we evaluate fmid the last as cache will be left in the global memory
  SAPLineSearchEvalCost(data, buff[2], &buff[5], sums, dv_alpha, v_alpha);

  // evaluate the first derivative of mid_alpha
  SAPLineSearchEvalDer(data, buff[2], &buff[6], sums, dv_alpha, v_alpha, dmid_1,
                       dmid_2);

  // evaluate the second derivative of mid_alpha
  SAPLineSearchEval2Der(data, buff[2], &buff[7], sums, dv_alpha, v_alpha,
                        ddmid_1, ddmid_2);
}

// ==========================================================================
// Driver function to involke
void TestOneStepSapGPU(std::vector<SAPCPUData>& sap_cpu_data,
                       std::vector<double>& momentum_cost,
                       std::vector<double>& regularizer_cost,
                       std::vector<Eigen::MatrixXd>& hessian,
                       std::vector<Eigen::MatrixXd>& neg_grad,
                       std::vector<Eigen::VectorXd>& chol_x, int num_velocities,
                       int num_contacts, int num_problems) {
  std::cout << "TestOneStepSapGPU with GPU called with " << num_problems
            << " problems" << std::endl;
  SAPGPUData sap_gpu_data;
  sap_gpu_data.MakeSAPGPUData(sap_cpu_data);

  // copy SAPGPUData to GPU
  SAPGPUData* d_sap_gpu_data;
  HANDLE_ERROR(cudaMalloc(&d_sap_gpu_data, sizeof(SAPGPUData)));
  HANDLE_ERROR(cudaMemcpy(d_sap_gpu_data, &sap_gpu_data, sizeof(SAPGPUData),
                          cudaMemcpyHostToDevice));

  int threadsPerBlock = 32;

  // Evaluate Cost
  CalcMomentumCostKernel<<<num_problems, threadsPerBlock,
                           2048 * sizeof(double)>>>(d_sap_gpu_data);

  CalcRegularizerCostKernel<<<num_problems, threadsPerBlock,
                              2048 * sizeof(double)>>>(d_sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Assemble Hessian
  // Calculate G*J
  CalcHessianKernel<<<num_problems, threadsPerBlock, 2048 * sizeof(double)>>>(
      d_sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Assemble -grad
  CalcNegGradKernel<<<num_problems, threadsPerBlock, 2048 * sizeof(double)>>>(
      d_sap_gpu_data);

  HANDLE_ERROR(cudaDeviceSynchronize());

  // Cholesky Solve
  SAPCholeskySolveKernel<<<num_problems, threadsPerBlock>>>(d_sap_gpu_data);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Transfer back to CPU for gtest validation
  sap_gpu_data.RetriveMomentumCostToCPU(momentum_cost);
  sap_gpu_data.RetriveRegularizerCostToCPU(regularizer_cost);
  sap_gpu_data.RetriveHessianToCPU(hessian);
  sap_gpu_data.RetriveNegGradToCPU(neg_grad);
  sap_gpu_data.RetriveCholXToCPU(chol_x);

  // Line search
  // call search kernel - find mid, eval der, 2der, control logic
  // the final goal is to return an updated x
  SAPLineSearchKernel<<<num_problems, threadsPerBlock, 8192 * sizeof(double)>>>(
      d_sap_gpu_data);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

// ===========================================================================

// ===========================================================================
// Joe's Notes
// ===========================================================================
// Sets vc = J * v by modifying `data` __device__ void CalcConstraintVelocity(
//                   SAPGPUData * data) {
//   // vc = J*v
//   for (int i = threadIdx.x; i < data->NumContacts(); i += blockDim.x) {
//     data->v_vc(i) = data->v_J(i) * data->v_guess();
//   }
// }
// ===========================================================================
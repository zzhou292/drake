#pragma once

#include "cuda_onestepsap.h"

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
//
// define a SAP data strucutre
struct SAPGPUData {
  // Mutable get functions
  __device__ Eigen::Map<Eigen::MatrixXd> dynamics_matrix() {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        A_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  // Const get functions
  __device__ const Eigen::Map<Eigen::MatrixXd> dynamics_matrix() const {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        A_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_guess() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_guess_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_guess() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_guess_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> v_star() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_star_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> v_star() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        v_star_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> velocity_gain() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> velocity_gain() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> velocity_gain_transpose() {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> velocity_gain_transpose() const {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> momentum_gain() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> momentum_gain() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> momentum_gain_transpose() {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> momentum_gain_transpose() const {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> J() {
    int row_size = 3 * num_contacts;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> J() const {
    int row_size = 3 * num_contacts;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> G(int constraint_index) {
    int row_size = 3;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        G_global + (blockIdx.x * num_contacts + constraint_index) * row_size *
                       col_size,
        row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> G(int constraint_index) const {
    int row_size = 3;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXd>(
        G_global + (blockIdx.x * num_contacts + constraint_index) * row_size *
                       col_size,
        row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::Vector3d> gamma(int constraint_index) {
    return Eigen::Map<Eigen::Vector3d>(
        gamma_global + (blockIdx.x * num_contacts + constraint_index) * 3, 3,
        1);
  }

  __device__ Eigen::Map<Eigen::Vector3d> gamma_full() const {
    return Eigen::Map<Eigen::Vector3d>(
        gamma_global + blockIdx.x * num_contacts * 3, 3 * num_contacts, 1);
  }

  __device__ Eigen::Map<Eigen::Vector3d> R(int constraint_index) {
    return Eigen::Map<Eigen::Vector3d>(
        R_global + (blockIdx.x * num_contacts + constraint_index) * 3, 3, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> momentum_cost() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        momentum_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> momentum_cost() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        momentum_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> regularizer_cost() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        regularizer_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> regularizer_cost() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXd>(
        regularizer_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> G_J() {
    int row_size = num_contacts * 3;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        G_J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> G_J() const {
    int row_size = num_contacts * 3;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        G_J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> H() {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        H_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> H() const {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        H_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> neg_grad() {
    return Eigen::Map<Eigen::MatrixXd>(
        neg_grad_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  // Cholesky solve related functions
  __device__ Eigen::Map<Eigen::MatrixXd> chol_L() {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        chol_L_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> chol_L() const {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXd>(
        chol_L_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> chol_y() {
    return Eigen::Map<Eigen::MatrixXd>(
        chol_y_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> chol_y() const {
    return Eigen::Map<Eigen::MatrixXd>(
        chol_y_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXd> chol_x() {
    return Eigen::Map<Eigen::MatrixXd>(
        chol_x_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ const Eigen::Map<Eigen::MatrixXd> chol_x() const {
    return Eigen::Map<Eigen::MatrixXd>(
        chol_x_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ double l_alpha() { return l_alpha_global[blockIdx.x]; }

  __device__ const double l_alpha() const { return l_alpha_global[blockIdx.x]; }

  __device__ double r_alpha() { return r_alpha_global[blockIdx.x]; }

  __device__ const double r_alpha() const { return r_alpha_global[blockIdx.x]; }

  __host__ __device__ const int NumVelocities() const { return num_velocities; }
  __host__ __device__ const int NumContacts() const { return num_contacts; }
  __host__ __device__ const int NumProblems() const { return num_problems; }

  // Retrival functions - copy Momentum cost data back to CPU
  void RetriveMomentumCostToCPU(std::vector<double>& momentum_cost) {
    momentum_cost.resize(num_problems);
    cudaMemcpy(momentum_cost.data(), momentum_cost_global,
               num_problems * sizeof(double), cudaMemcpyDeviceToHost);
  }

  // Retrival functions - copy Regularizer cost data back to CPU
  void RetriveRegularizerCostToCPU(std::vector<double>& regularizer_cost) {
    regularizer_cost.resize(num_problems);
    cudaMemcpy(regularizer_cost.data(), regularizer_cost_global,
               num_problems * sizeof(double), cudaMemcpyDeviceToHost);
  }

  // Retrival function - copy Hessian data back to CPU
  void RetriveHessianToCPU(std::vector<Eigen::MatrixXd>& hessian) {
    hessian.resize(num_problems);
    for (int i = 0; i < num_problems; i++) {
      hessian[i].resize(num_velocities, num_velocities);
      cudaMemcpy(hessian[i].data(),
                 H_global + i * num_velocities * num_velocities,
                 num_velocities * num_velocities * sizeof(double),
                 cudaMemcpyDeviceToHost);
    }
  }

  // Retrival function - copy Cholesky x data back to CPU
  void RetriveCholXToCPU(std::vector<Eigen::MatrixXd>& chol_x) {
    chol_x.resize(num_problems);
    for (int i = 0; i < num_problems; i++) {
      chol_x[i].resize(num_velocities, 1);
      cudaMemcpy(chol_x[i].data(), chol_x_global + i * num_velocities,
                 num_velocities * sizeof(double), cudaMemcpyDeviceToHost);
    }
  }

  // Retrival function - copy negative gradient data back to CPU
  void RetriveNegGradToCPU(std::vector<Eigen::MatrixXd>& neg_grad) {
    neg_grad.resize(num_problems);
    for (int i = 0; i < num_problems; i++) {
      neg_grad[i].resize(num_velocities, 1);
      cudaMemcpy(neg_grad[i].data(), neg_grad_global + i * num_velocities,
                 num_velocities * sizeof(double), cudaMemcpyDeviceToHost);
    }
  }

  void MakeSAPGPUData(std::vector<SAPCPUData> data) {
    this->num_contacts = data[0].num_contacts;
    this->num_velocities = data[0].num_velocities;
    this->num_problems = data.size();

    // Malloc for all pointers
    HANDLE_ERROR(cudaMalloc(&A_global, num_problems * num_velocities *
                                           num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&v_guess_global,
                            num_problems * num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&v_star_global,
                            num_problems * num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&delta_v_global,
                            num_problems * num_velocities * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&J_global, num_problems * 3 * num_contacts *
                                           num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(
        &G_global, num_problems * num_contacts * 3 * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&gamma_global,
                            num_problems * num_contacts * 3 * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&R_global,
                            num_problems * num_contacts * 3 * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&delta_p_global,
                            num_problems * num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&momentum_cost_global,
                            num_problems * sizeof(double)));  // 1D vector

    HANDLE_ERROR(
        cudaMalloc(&regularizer_cost_global, num_problems * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&G_J_global, num_problems * 3 * num_contacts *
                                             num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&H_global, num_problems * num_velocities *
                                           num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&neg_grad_global,
                            num_problems * num_velocities * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(
        &chol_L_global,
        num_problems * num_velocities * num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&chol_y_global,
                            num_problems * num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&chol_x_global,
                            num_problems * num_velocities * sizeof(double)));

    HANDLE_ERROR(cudaMalloc(&dl_eval_global, num_problems * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&dll_eval_global, num_problems * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&l_alpha_global, num_problems * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&r_alpha_global, num_problems * sizeof(double)));

    // Set data to initialized value using cudaMemset
    HANDLE_ERROR(cudaMemset(
        chol_L_global, 0,
        num_problems * num_velocities * num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMemset(chol_y_global, 0,
                            num_problems * num_velocities * sizeof(double)));
    HANDLE_ERROR(cudaMemset(chol_x_global, 0,
                            num_problems * num_velocities * sizeof(double)));

    // Initialize line search parameters, reconsider the necessity
    HANDLE_ERROR(cudaMemset(dl_eval_global, 0, num_problems * sizeof(double)));
    HANDLE_ERROR(cudaMemset(dll_eval_global, 0, num_problems * sizeof(double)));
    HANDLE_ERROR(cudaMemset(l_alpha_global, 0, num_problems * sizeof(double)));
    HANDLE_ERROR(cudaMemset(r_alpha_global, 0, num_problems * sizeof(double)));

    // Copy data to GPU
    for (int i = 0; i < num_problems; i++) {
      HANDLE_ERROR(cudaMemcpy(A_global + i * num_velocities * num_velocities,
                              data[i].dynamics_matrix.data(),
                              num_velocities * num_velocities * sizeof(double),
                              cudaMemcpyHostToDevice));
      HANDLE_ERROR(
          cudaMemcpy(v_star_global + i * num_velocities, data[i].v_star.data(),
                     num_velocities * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(
          v_guess_global + i * num_velocities, data[i].v_guess.data(),
          num_velocities * sizeof(double), cudaMemcpyHostToDevice));
      HANDLE_ERROR(
          cudaMemcpy(J_global + i * 3 * num_contacts * num_velocities,
                     data[i].constraint_data.J.data(),
                     3 * num_contacts * num_velocities * sizeof(double),
                     cudaMemcpyHostToDevice));

      for (int j = 0; j < num_contacts; j++) {
        HANDLE_ERROR(cudaMemcpy(gamma_global + i * num_contacts * 3 + j * 3,
                                data[i].gamma[j].data(), 3 * sizeof(double),
                                cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(R_global + i * num_contacts * 3 + j * 3,
                                data[i].R[j].data(), 3 * sizeof(double),
                                cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(G_global + i * num_contacts * 3 * 3 + j * 3 * 3,
                                data[i].constraint_data.G[j].data(),
                                3 * 3 * sizeof(double),
                                cudaMemcpyHostToDevice));
      }
    }
  }

 private:
  double* A_global;        // Global memory dynamics matrix A for all sims
  double* v_star_global;   // Global memory free motion generalized velocity v*.
  double* v_guess_global;  // Global memory v_guess for all sims
  double* J_global;        // Global memory J matrix for all sims
  double* G_global;        // Global memory G matrix for all sims
  double* gamma_global;    // Global memory v_gamma for all sims
  double* R_global;        // Global memory v_R for all sims

  double* delta_v_global;  // Global memory velocity gain = v - v*
  double* delta_p_global;  // Global memory momentum gain = A * (v - v*)

  double* momentum_cost_global;  // Global memory momentum_cost for all sims
  double*
      regularizer_cost_global;  // Global memory regularizer cost for all sims

  double* G_J_global;       // Global memory to hold G*J
  double* H_global;         // Global memory to hold Hessian
  double* neg_grad_global;  // Global memory to hold negative gradient

  // Chlosky solve related variables
  double*
      chol_L_global;  // Global memory to hold factorized L matrix in cholesky
  double* chol_y_global;  // Global memory to hold y in cholesky
  double* chol_x_global;  // Global memory to hold x in cholesky

  int num_contacts;    // Number of contacts
  int num_problems;    // Number of problems
  int num_velocities;  // Number of velocities

  // Line search related variables
  double* dl_eval_global;   // Global memory to evaluate dl/dalpha, this also
                            // serves as a scratch space during the line search
  double* dll_eval_global;  // Global memory to evaluate dl2/dalpha2, this also
                            // serves as a scratch space during the line search
  double*
      l_alpha_global;  // Global memory to hold left alpha for line search this
                       // also serves as a scratch space during the line search
  double*
      r_alpha_global;  // Global memory to hold right alpha for line search this
                       // also serves as a scratch space during the line search
};

// ===========================================================================
// Joe's Notes
// ===========================================================================
//   __device__ Eigen::Map<Eigen::MatrixXd> J(int constraint_index) {
//     J().block(3 * constraint_index, 0, 3, );
//   }

#pragma once

#include "cuda_gpu_collision.cuh"
#include "cuda_onestepsap.h"

#ifndef HANDLE_ERROR_MACRO
#define HANDLE_ERROR_MACRO
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif

//
// define a SAP data strucutre
struct SAPGPUData {
#if defined(__CUDACC__)

  // Mutable get functions
  __device__ Eigen::Map<Eigen::MatrixXf> dynamics_matrix() {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        A_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  // Const get functions
  __device__ const Eigen::Map<Eigen::MatrixXf> dynamics_matrix() const {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        A_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> v_guess() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_guess_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> v_guess() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_guess_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> v_star() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_star_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> v_star() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_star_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> velocity_gain() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> velocity_gain() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> velocity_gain_transpose() {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> velocity_gain_transpose() const {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_v_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> momentum_gain() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> momentum_gain() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> momentum_gain_transpose() {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> momentum_gain_transpose() const {
    int row_size = 1;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_p_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> J() {
    int row_size = 3 * num_contacts;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> J() const {
    int row_size = 3 * num_contacts;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> G(int constraint_index) {
    int row_size = 3;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXf>(
        G_global + (blockIdx.x * num_contacts + constraint_index) * row_size *
                       col_size,
        row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> G(int constraint_index) const {
    int row_size = 3;
    int col_size = 3;
    return Eigen::Map<Eigen::MatrixXf>(
        G_global + (blockIdx.x * num_contacts + constraint_index) * row_size *
                       col_size,
        row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::Vector3f> gamma(int constraint_index) {
    return Eigen::Map<Eigen::Vector3f>(
        gamma_global + (blockIdx.x * num_contacts + constraint_index) * 3, 3,
        1);
  }

  __device__ Eigen::Map<Eigen::Vector3f> gamma_full() const {
    return Eigen::Map<Eigen::Vector3f>(
        gamma_global + blockIdx.x * num_contacts * 3, 3 * num_contacts, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> momentum_cost() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        momentum_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> momentum_cost() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        momentum_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> constraint_cost() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        constraint_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> constraint_cost() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        constraint_cost_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> dl_dalpha0() {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        dl_dalpha0_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> dl_dalpha0() const {
    int row_size = 1;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        dl_dalpha0_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> G_J() {
    int row_size = num_contacts * 3;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        G_J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> G_J() const {
    int row_size = num_contacts * 3;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        G_J_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> H() {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        H_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> H() const {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        H_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> neg_grad() {
    return Eigen::Map<Eigen::MatrixXf>(
        neg_grad_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  // Cholesky solve related functions
  __device__ Eigen::Map<Eigen::MatrixXf> chol_L() {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        chol_L_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> chol_L() const {
    int row_size = num_velocities;
    int col_size = num_velocities;
    return Eigen::Map<Eigen::MatrixXf>(
        chol_L_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> chol_y() {
    return Eigen::Map<Eigen::MatrixXf>(
        chol_y_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> chol_y() const {
    return Eigen::Map<Eigen::MatrixXf>(
        chol_y_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> chol_x() {
    return Eigen::Map<Eigen::MatrixXf>(
        chol_x_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> chol_x() const {
    return Eigen::Map<Eigen::MatrixXf>(
        chol_x_global + blockIdx.x * num_velocities, num_velocities, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> phi0(int constraint_index) {
    return Eigen::Map<Eigen::MatrixXf>(
        phi0_global + blockIdx.x * num_contacts + constraint_index, 1, 1);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> phi0(
      int constraint_index) const {
    return Eigen::Map<Eigen::MatrixXf>(
        phi0_global + blockIdx.x * num_contacts + constraint_index, 1, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> contact_stiffness(
      int constraint_index) {
    return Eigen::Map<Eigen::MatrixXf>(
        contact_stiffness_global + blockIdx.x * num_contacts + constraint_index,
        1, 1);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> contact_stiffness(
      int constraint_index) const {
    return Eigen::Map<Eigen::MatrixXf>(
        contact_stiffness_global + blockIdx.x * num_contacts + constraint_index,
        1, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> contact_damping(int constraint_index) {
    return Eigen::Map<Eigen::MatrixXf>(
        contact_damping_global + blockIdx.x * num_contacts + constraint_index,
        1, 1);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> contact_damping(
      int constraint_index) const {
    return Eigen::Map<Eigen::MatrixXf>(
        contact_damping_global + blockIdx.x * num_contacts + constraint_index,
        1, 1);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> v_guess_prev() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_guess_prev_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> v_guess_prev() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_guess_prev_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> delta_p_chol() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_p_chol_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> delta_p_chol() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_p_chol_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> delta_v_c() {
    int row_size = 3 * num_contacts;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_v_c_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> delta_v_c() const {
    int row_size = 3 * num_contacts;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        delta_v_c_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> v_alpha() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_alpha_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> v_alpha() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_alpha_global + blockIdx.x * row_size * col_size, row_size, col_size);
  }

  __device__ Eigen::Map<Eigen::MatrixXf> v_guess_prev_newton() {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_guess_prev_newton_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ const Eigen::Map<Eigen::MatrixXf> v_guess_prev_newton() const {
    int row_size = num_velocities;
    int col_size = 1;
    return Eigen::Map<Eigen::MatrixXf>(
        v_guess_prev_newton_global + blockIdx.x * row_size * col_size, row_size,
        col_size);
  }

  __device__ int& num_active_contacts() {
    return num_active_contacts_global[blockIdx.x];
  }

  __device__ CollisionGPUData* get_collision_gpu_data() {
    return d_collision_gpu_data;
  };

  __host__ __device__ const int NumVelocities() const { return num_velocities; }
  __host__ __device__ const int NumContacts() const { return num_contacts; }
  __host__ __device__ const int NumProblems() const { return num_problems; }

#endif

  // Retrival functions - copy Momentum cost data back to CPU
  void RetriveMomentumCostToCPU(std::vector<float>& momentum_cost) {
    momentum_cost.resize(num_problems);
    cudaMemcpy(momentum_cost.data(), momentum_cost_global,
               num_problems * sizeof(float), cudaMemcpyDeviceToHost);
  }

  // Retrival functions - copy Regularizer cost data back to CPU
  void RetriveConstraintCostToCPU(std::vector<float>& constraint_cost) {
    constraint_cost.resize(num_problems);
    cudaMemcpy(constraint_cost.data(), constraint_cost_global,
               num_problems * sizeof(float), cudaMemcpyDeviceToHost);
  }

  // Retrival function - copy Hessian data back to CPU
  void RetriveHessianToCPU(std::vector<Eigen::MatrixXf>& hessian) {
    hessian.resize(num_problems);
    for (int i = 0; i < num_problems; i++) {
      hessian[i].resize(num_velocities, num_velocities);
      cudaMemcpy(hessian[i].data(),
                 H_global + i * num_velocities * num_velocities,
                 num_velocities * num_velocities * sizeof(float),
                 cudaMemcpyDeviceToHost);
    }
  }

  // Retrival function - copy Cholesky x data back to CPU
  void RetriveCholXToCPU(std::vector<Eigen::MatrixXf>& chol_x) {
    chol_x.resize(num_problems);
    for (int i = 0; i < num_problems; i++) {
      chol_x[i].resize(num_velocities, 1);
      cudaMemcpy(chol_x[i].data(), chol_x_global + i * num_velocities,
                 num_velocities * sizeof(float), cudaMemcpyDeviceToHost);
    }
  }

  // Retrival function - copy negative gradient data back to CPU
  void RetriveNegGradToCPU(std::vector<Eigen::MatrixXf>& neg_grad) {
    neg_grad.resize(num_problems);
    for (int i = 0; i < num_problems; i++) {
      neg_grad[i].resize(num_velocities, 1);
      cudaMemcpy(neg_grad[i].data(), neg_grad_global + i * num_velocities,
                 num_velocities * sizeof(float), cudaMemcpyDeviceToHost);
    }
  }

  void RetriveVGuessToCPU(std::vector<Eigen::MatrixXf>& v_solved) {
    v_solved.resize(num_problems);
    for (int i = 0; i < num_problems; i++) {
      v_solved[i].resize(num_velocities, 1);
      cudaMemcpy(v_solved[i].data(), v_guess_global + i * num_velocities,
                 num_velocities * sizeof(float), cudaMemcpyDeviceToHost);
    }
  }

  void RetriveNumActiveContactToCPU(std::vector<int>& num_active_contacts) {
    num_active_contacts.resize(num_problems);
    cudaMemcpy(num_active_contacts.data(), num_active_contacts_global,
               num_problems * sizeof(int), cudaMemcpyDeviceToHost);
  }

  void Initialize(int in_num_contacts, int in_num_velocities,
                  int in_num_problems, CollisionGPUData* gpu_collision_data) {
    num_contacts = in_num_contacts;
    num_velocities = in_num_velocities;
    num_problems = in_num_problems;

    HANDLE_ERROR(cudaMalloc(&delta_v_global,
                            num_problems * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(
        &G_global, num_problems * num_contacts * 3 * 3 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&gamma_global,
                            num_problems * num_contacts * 3 * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&delta_p_global,
                            num_problems * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&momentum_cost_global,
                            num_problems * sizeof(float)));  // 1D vector

    HANDLE_ERROR(
        cudaMalloc(&constraint_cost_global, num_problems * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&G_J_global, num_problems * 3 * num_contacts *
                                             num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&H_global, num_problems * num_velocities *
                                           num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&neg_grad_global,
                            num_problems * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&dl_dalpha0_global, num_problems * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(
        &chol_L_global,
        num_problems * num_velocities * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&chol_y_global,
                            num_problems * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&chol_x_global,
                            num_problems * num_velocities * sizeof(float)));

    HANDLE_ERROR(cudaMalloc(&v_guess_prev_global,
                            num_problems * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&delta_p_chol_global,
                            num_problems * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&delta_v_c_global,
                            num_problems * 3 * num_contacts * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&v_alpha_global,
                            num_problems * num_velocities * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&v_guess_prev_newton_global,
                            num_problems * num_velocities * sizeof(float)));

    // retrieve data from the gpu_collision_data
    A_global = gpu_collision_data->GetDynamicMatrixPtr();
    v_star_global = gpu_collision_data->GetVStarPtr();
    v_guess_global = gpu_collision_data->GetVelocityVectorPtr();
    J_global = gpu_collision_data->GetJacobianPtr();
    phi0_global = gpu_collision_data->GetPhi0Ptr();
    contact_stiffness_global = gpu_collision_data->GetContactStiffnessPtr();
    contact_damping_global = gpu_collision_data->GetContactDampingPtr();
    num_active_contacts_global = gpu_collision_data->GetNumCollisionsPtr();

    d_collision_gpu_data = gpu_collision_data->GetCollisionGPUDataPtr();

    // copy struct to device
    HANDLE_ERROR(cudaMalloc(&d_sap_gpu_data_solve, sizeof(SAPGPUData)));
    HANDLE_ERROR(cudaMemcpy(d_sap_gpu_data_solve, this, sizeof(SAPGPUData),
                            cudaMemcpyHostToDevice));
  }

  // Free memory
  void Destroy() {
    // HANDLE_ERROR(cudaFree(A_global));
    // HANDLE_ERROR(cudaFree(v_star_global));
    // HANDLE_ERROR(cudaFree(v_guess_global));
    // HANDLE_ERROR(cudaFree(J_global));
    HANDLE_ERROR(cudaFree(G_global));
    HANDLE_ERROR(cudaFree(gamma_global));
    HANDLE_ERROR(cudaFree(delta_v_global));
    HANDLE_ERROR(cudaFree(delta_p_global));
    HANDLE_ERROR(cudaFree(momentum_cost_global));
    HANDLE_ERROR(cudaFree(constraint_cost_global));
    HANDLE_ERROR(cudaFree(dl_dalpha0_global));
    HANDLE_ERROR(cudaFree(G_J_global));
    HANDLE_ERROR(cudaFree(H_global));
    HANDLE_ERROR(cudaFree(neg_grad_global));
    // HANDLE_ERROR(cudaFree(phi0_global));
    // HANDLE_ERROR(cudaFree(contact_stiffness_global));
    // HANDLE_ERROR(cudaFree(contact_damping_global));
    // HANDLE_ERROR(cudaFree(num_active_contacts_global));
    HANDLE_ERROR(cudaFree(chol_L_global));
    HANDLE_ERROR(cudaFree(chol_y_global));
    HANDLE_ERROR(cudaFree(chol_x_global));
    // HANDLE_ERROR(cudaFree(dl_eval_global));
    // HANDLE_ERROR(cudaFree(dll_eval_global));
    // HANDLE_ERROR(cudaFree(l_alpha_global));
    // HANDLE_ERROR(cudaFree(r_alpha_global));
    HANDLE_ERROR(cudaFree(v_guess_prev_global));
    HANDLE_ERROR(cudaFree(delta_p_chol_global));
    HANDLE_ERROR(cudaFree(delta_v_c_global));
    HANDLE_ERROR(cudaFree(v_alpha_global));
    HANDLE_ERROR(cudaFree(v_guess_prev_newton_global));
  }

  void TestOneStepSapGPU(int num_steps = 1);

 private:
  float* A_global;        // Global memory dynamics matrix A for all sims
  float* v_star_global;   // Global memory free motion generalized velocity v*.
  float* v_guess_global;  // Global memory v_guess for all sims
  float* J_global;        // Global memory J matrix for all sims
  float* G_global;        // Global memory G matrix for all sims
  float* gamma_global;    // Global memory v_gamma for all sims

  float* delta_v_global;  // Global memory velocity gain = v - v*
  float* delta_p_global;  // Global memory momentum gain = A * (v - v*)

  float* momentum_cost_global;    // Global memory momentum_cost for all sims
  float* constraint_cost_global;  // Global memory regularizer cost for all sims
  float* dl_dalpha0_global;  // Global memory dℓ/dα(α = 0) = ∇ᵥℓ(α = 0)⋅Δv.

  float* G_J_global;       // Global memory to hold G*J
  float* H_global;         // Global memory to hold Hessian
  float* neg_grad_global;  // Global memory to hold negative gradient

  float* phi0_global;  // Global memory to hold phi0 - collision penetration
                       // distance reported by the geometry engine
  float* contact_stiffness_global;  // (harmonic mean of stiffness between two
                                    // materials) contact stiffness reported by
                                    // the geometry engine
  float* contact_damping_global;    // (harmonic mean of damping between two
                                  // materials) contact damping reported by the
                                  // geomtry engine
  int* num_active_contacts_global;  // Global memory to hold number of active
                                    // contacts for each problem, one int per
                                    // problem

  // Newton outer loop related parameters
  float* v_guess_prev_newton_global;  // Global memory to hold v_guess_prev in
                                      // newton step for all sims

  // Line search related parameters
  float*
      v_guess_prev_global;  // Global memory to hold v_guess_prev for all sims
  float* delta_p_chol_global;  // Momentum gain at alpha = 1
  float* delta_v_c_global;     // Global memory to hold contact velocity vector
  float* v_alpha_global;       // Global memory to hold v_alpha for all sims

  // Chlosky solve related variables
  float*
      chol_L_global;  // Global memory to hold factorized L matrix in cholesky
  float* chol_y_global;  // Global memory to hold y in cholesky
  float* chol_x_global;  // Global memory to hold x in cholesky

  int num_contacts;    // Number of contacts
  int num_problems;    // Number of problems
  int num_velocities;  // Number of velocities

  SAPGPUData* d_sap_gpu_data_solve;        // Storing GPU copy of SAPGPUData
  CollisionGPUData* d_collision_gpu_data;  // Storing GPU copy of SAPGPUData
};

// ===========================================================================
// Joe's Notes
// ===========================================================================
//   __device__ Eigen::Map<Eigen::MatrixXf> J(int constraint_index) {
//     J().block(3 * constraint_index, 0, 3, );
//   }

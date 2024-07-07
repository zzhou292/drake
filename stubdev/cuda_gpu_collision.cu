#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.h"

// CUDA error handeling
// =====================
static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
// =====================

// Device function to check Sphere-Sphere collision
__device__ CollisionData CheckSphereCollision(const Sphere& a,
                                              const Sphere& b) {
  CollisionData data = {
      false, {0, 0, 0}, {0, 0, 0}, 0, Eigen::Matrix3d::Zero()};

  Eigen::Vector3d dist = a.center - b.center;
  double distSquared =
      dist(0) * dist(0) + dist(1) * dist(1) + dist(2) * dist(2);
  double distLength = sqrt(distSquared);
  double radiusSum = a.radius + b.radius;

  dist.normalize();

  if (distSquared <= (radiusSum * radiusSum)) {
    data.isColliding = true;
    // Calculate collision normal
    data.nhat_BA_W = dist;
    // Normalize the collision normal

    data.nhat_BA_W.normalize();
    // Calculate collision points
    Eigen::Vector3d midpoint;
    midpoint(0) = (a.center(0) + b.center(0)) / 2;
    midpoint(1) = (a.center(1) + b.center(1)) / 2;
    midpoint(2) = (a.center(2) + b.center(2)) / 2;

    data.phi0 = -(distLength - radiusSum);  // sign convention

    data.p_WC = midpoint;

    // Get collision frame matrix
    // Random vector v is default to {1.0, 1.0, 1.0}
    Eigen::Vector3d v(1.0, 1.0, 1.0);
    v.normalize();

    double y_hat_temp = v.dot(data.nhat_BA_W);
    Eigen::Vector3d y_hat = v - y_hat_temp * data.nhat_BA_W;
    y_hat.normalize();
    Eigen::Vector3d x_hat = y_hat.cross(data.nhat_BA_W);

    data.R(0, 0) = x_hat(0);           // x of x-axis
    data.R(0, 1) = x_hat(1);           // y of x-axis
    data.R(0, 2) = x_hat(2);           // z of x-axis
    data.R(1, 0) = y_hat(0);           // x of y-axis
    data.R(1, 1) = y_hat(1);           // y of y-axis
    data.R(1, 2) = y_hat(2);           // z of y-axis
    data.R(2, 0) = data.nhat_BA_W(0);  // x of z-axis
    data.R(2, 1) = data.nhat_BA_W(1);  // y of z-axis
    data.R(2, 2) = data.nhat_BA_W(2);  // z of z-axis

    data.vn = -(a.velocity - b.velocity)
                   .dot(data.nhat_BA_W);  // negative for departing, positive
                                          // for approaching
  } else {
    data.isColliding = false;
  }

  return data;
}

__device__ void CalculateIndividualGamma(CollisionData& data, const Sphere& a,
                                         const Sphere& b) {
  // calculate gamma in the local frame
  double stiffness_avg = (a.stiffness + b.stiffness) / 2;
  double damping_avg = (a.damping + b.damping) / 2;

  double gamma_z =
      (stiffness_avg * (data.phi0)) * (fmax(0, 1 + damping_avg * data.vn));
  Eigen::Vector3d gamma_temp(0.0, 0.0, gamma_z);
  data.gamma = gamma_temp;

  // calculate gamma in the global frame
  data.gamma_W = data.R.transpose() * data.gamma;
}

// Kernel to detect collisions between Spheres
__global__ void DetectSphereCollisions(const Sphere* spheres, int numProblems,
                                       int numSpheres,
                                       CollisionData* collisionMatrix) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      for (int k = j + 1; k < numSpheres; k++) {
        collisionMatrix[(p_idx * numSpheres * numSpheres) + j * numSpheres +
                        k] =
            CheckSphereCollision(spheres[p_idx * numSpheres + j],
                                 spheres[p_idx * numSpheres + k]);
      }
    }
  }
  __syncwarp();

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      for (int k = j + 1; k < numSpheres; k++) {
        CalculateIndividualGamma(
            collisionMatrix[(p_idx * numSpheres * numSpheres) + j * numSpheres +
                            k],
            spheres[p_idx * numSpheres + j], spheres[p_idx * numSpheres + k]);
      }
    }
  }

  __syncwarp();
}

// d_spheres, numProblems, numSpheres, d_collisionMatrixSpheres, d_jacobian,
//      d_gamma, d_num_collisions

// Kernel to detect collisions between Spheres
__global__ void ConstructJacobianGamma(const Sphere* spheres, int numProblems,
                                       int numSpheres,
                                       CollisionData* collisionMatrix,
                                       double* d_jacobian, double* d_gamma,
                                       int* d_num_collisions) {
  int idx = threadIdx.x;
  int p_idx = blockIdx.x;

  Eigen::Map<Eigen::VectorXd> full_gamma(
      d_gamma + blockIdx.x * numSpheres * numSpheres * 3,
      numSpheres * numSpheres * 3, 1);
  Eigen::Map<Eigen::MatrixXd> full_jacobian(
      d_jacobian +
          blockIdx.x * (numSpheres * 3) * (numSpheres * numSpheres * 3),
      numSpheres * 3, numSpheres * numSpheres * 3);

  for (int j = idx; j < numSpheres; j += blockDim.x) {
    if (j < numSpheres) {
      for (int k = j + 1; k < numSpheres; k++) {
        if (collisionMatrix[(p_idx * numSpheres * numSpheres) + j * numSpheres +
                            k]
                .isColliding) {
          int collision_idx = atomicAdd(&d_num_collisions[p_idx], 1);

          // construct full gamma vector
          full_gamma(collision_idx * 3 + 0) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .gamma_W(0);
          full_gamma(collision_idx * 3 + 1) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .gamma_W(1);
          full_gamma(collision_idx * 3 + 2) =
              collisionMatrix[(p_idx * numSpheres * numSpheres) +
                              j * numSpheres + k]
                  .gamma_W(2);

          // construct Jacobian matrix
          full_jacobian.block<3, 3>(j * 3, collision_idx * 3) =
              Eigen::MatrixXd::Identity(3, 3);
          full_jacobian.block<3, 3>(k * 3, collision_idx * 3) =
              -Eigen::MatrixXd::Identity(3, 3);
        }
      }
    }
  }
  __syncwarp();
}

void CollisionEngine(Sphere* h_spheres, const int numProblems,
                     const int numSpheres,
                     CollisionData* h_collisionMatrixSpheres,
                     double* h_jacobian, double* h_gamma,
                     int* h_num_collisions) {
  // Device memory allocations
  Sphere* d_spheres;
  CollisionData* d_collisionMatrixSpheres;
  double* d_jacobian;
  int* d_num_collisions;
  double* d_gamma;

  HANDLE_ERROR(cudaMalloc((void**)&d_spheres,
                          numProblems * numSpheres * sizeof(Sphere)));
  HANDLE_ERROR(cudaMalloc(
      (void**)&d_collisionMatrixSpheres,
      numProblems * numSpheres * numSpheres * sizeof(CollisionData)));
  HANDLE_ERROR(cudaMalloc((void**)&d_jacobian,
                          numProblems * sizeof(double) * (numSpheres * 3) *
                              numSpheres * numSpheres * 3));
  HANDLE_ERROR(cudaMalloc((void**)&d_gamma, numProblems * sizeof(double) *
                                                numSpheres * numSpheres * 3));
  HANDLE_ERROR(
      cudaMalloc((void**)&d_num_collisions, numProblems * sizeof(int)));

  // Copy data to device
  HANDLE_ERROR(cudaMemcpy(d_spheres, h_spheres,
                          numProblems * numSpheres * sizeof(Sphere),
                          cudaMemcpyHostToDevice));

  // Set jacobian, num_collisions, full gamma vector to zero
  HANDLE_ERROR(cudaMemset(d_jacobian, 0,
                          numProblems * sizeof(double) * (numSpheres * 3) *
                              numSpheres * numSpheres * 3));
  HANDLE_ERROR(cudaMemset(d_num_collisions, 0, numProblems * sizeof(int)));
  HANDLE_ERROR(cudaMemset(
      d_gamma, 0, numProblems * sizeof(double) * numSpheres * numSpheres * 3));

  // Kernel launches
  int threadsPerBlock = 32;
  int blocksPerGridSpheres = numProblems;
  DetectSphereCollisions<<<blocksPerGridSpheres, threadsPerBlock>>>(
      d_spheres, numProblems, numSpheres, d_collisionMatrixSpheres);
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Construct Jacobian matrix and Gamma vector
  ConstructJacobianGamma<<<blocksPerGridSpheres, threadsPerBlock>>>(
      d_spheres, numProblems, numSpheres, d_collisionMatrixSpheres, d_jacobian,
      d_gamma, d_num_collisions);

  // Copy results back to host
  HANDLE_ERROR(
      cudaMemcpy(h_collisionMatrixSpheres, d_collisionMatrixSpheres,
                 numProblems * numSpheres * numSpheres * sizeof(CollisionData),
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_jacobian, d_jacobian,
                          numProblems * sizeof(double) * (numSpheres * 3) *
                              numSpheres * numSpheres * 3,
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(
      cudaMemcpy(h_gamma, d_gamma,
                 numProblems * sizeof(double) * numSpheres * numSpheres * 3,
                 cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_num_collisions, d_num_collisions,
                          numProblems * sizeof(int), cudaMemcpyDeviceToHost));

  // Free device memory
  cudaFree(d_spheres);
  cudaFree(d_collisionMatrixSpheres);
  cudaFree(d_jacobian);
  cudaFree(d_num_collisions);
  cudaFree(d_gamma);
}

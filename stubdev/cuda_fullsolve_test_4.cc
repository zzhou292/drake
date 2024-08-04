#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "cuda_fullsolve.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

GTEST_TEST(KernelTest, FullSolveTest) {
  int numSpheres = 4;
  int numPlanes = 4;
  int numProblems = 15000;
  int numContacts = numSpheres * numSpheres;

  // initialize the problem input spheres_vec
  Sphere* h_spheres = new Sphere[numProblems * numSpheres];
  for (int i = 0; i < numProblems; i++) {
    for (int j = 0; j < numSpheres; j++) {
      Eigen::Vector3f p;

      if (j == 0) {
        p << 0.0, 0.0, 0.0;
      } else if (j == 1) {
        p << -0.03, 0.05196152422706632, 0.0;
      } else if (j == 2) {
        p << 0.03, 0.05196152422706632, 0.0;
      } else if (j == 3) {
        // TODO: randomize the last sphere
        // x between -1.5 to 1.5
        // y between -1.0 and -1.5
        // cur ball position
        // int col = i % 20;
        // p << -2.4 + static_cast<float>(col) * (4.8 / 20.0), -1.7, 0.0;

        // float random_angle =
        //     static_cast<float>(rand()) / RAND_MAX * 2.0 * M_PI;
        // p << 0.0 + 0.05 * cos(random_angle), -0.10 + 0.03 *
        // sin(random_angle),
        //     0.0;

        p << 0.0, -2.0, 0.0;
      }

      h_spheres[i * numSpheres + j].center = p;

      h_spheres[i * numSpheres + j].velocity = Eigen::Vector3f::Zero();

      h_spheres[i * numSpheres + j].mass = 0.17;

      if (j == 3) [[unlikely]] {
        Eigen::Vector3f random_target(0.0, 0.0, 0.0);
        Eigen::Vector3f direction = random_target - p;
        direction.normalize();
        // scale up the velocity to 8.0 to 20.0, random
        // h_spheres[i * numSpheres + j].velocity =
        //     direction * 1.2 +
        //     static_cast<float>(rand()) / RAND_MAX * 0.5 * direction;
        h_spheres[i * numSpheres + j].velocity = Eigen::Vector3f(0.0, 1.2, 0.0);

        h_spheres[i * numSpheres + j].mass = 0.17;
      }

      h_spheres[i * numSpheres + j].radius = 0.03;

      // initialize material properties
      h_spheres[i * numSpheres + j].stiffness = 10000.0;
      h_spheres[i * numSpheres + j].damping = 1e-20;
    }
  }

  Plane* h_planes = new Plane[numProblems * numPlanes];
  for (int i = 0; i < numProblems; i++) {
    for (int j = 0; j < numPlanes; j++) {
      if (j == 0) {
        h_planes[i * numPlanes + j].p1 << -0.1, 1.0, 0.0;  // -0.25 | -0.1
        h_planes[i * numPlanes + j].p2 << -0.1, 0.0, 0.0;
        h_planes[i * numPlanes + j].n << 1.0, 0.0, 0.0;
      } else if (j == 1) {
        h_planes[i * numPlanes + j].p1 << 0.0, 0.09, 0.0;
        h_planes[i * numPlanes + j].p2 << 1.0, 0.09, 0.0;  // 0.35 | 0.09
        h_planes[i * numPlanes + j].n << 0.0, -1.0, 0.0;
      } else if (j == 2) {
        h_planes[i * numPlanes + j].p1 << 0.1, 1.0, 0.0;
        h_planes[i * numPlanes + j].p2 << 0.1, 1.0, 0.0;
        h_planes[i * numPlanes + j].n << -1.0, 0.0, 0.0;
      } else if (j == 3) {
        h_planes[i * numPlanes + j].p1 << 0.0, -0.17, 0.0;
        h_planes[i * numPlanes + j].p2 << 1.0, -0.17, 0.0;
        h_planes[i * numPlanes + j].n << 0.0, 1.0, 0.0;
      }

      h_planes[i * numPlanes + j].stiffness = 10000.0;
      h_planes[i * numPlanes + j].damping = 1e-20;
    }
  }

  FullSolveSAP solver;

  solver.init(h_spheres, h_planes, numProblems, numSpheres, numPlanes,
              numContacts, false);

  // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 1; i++) {
    std::cout << "Step " << i << std::endl;
    solver.step(800);
  }

  // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<float, std::milli> duration = end - start;
  std::cout << "Time taken for 400 iterations: " << duration.count()
            << "mili seconds" << std::endl;

  solver.destroy();
}

// ===================================================
// END OF ACTUAL SAP SOLVER FUNCTION CALLS
// ===================================================

}  // namespace
}  // namespace drake

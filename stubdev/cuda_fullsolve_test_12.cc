#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "cuda_fullsolve.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace drake {
namespace {

double run(int numProblems) {
  int numSpheres = 12;
  int numPlanes = 4;
  // int numProblems = 15000;
  int numContacts = numSpheres * numSpheres;

  // initialize the problem input spheres_vec
  Sphere* h_spheres = new Sphere[numProblems * numSpheres];
  for (int i = 0; i < numProblems; i++) {
    for (int j = 0; j < numSpheres; j++) {
      Eigen::Vector3d p;
      // Case 1 - 22 Spheres in the environment
      if (j == 0) {
        p << 0.0, 0.0, 0.0;
      } else if (j == 1) {
        p << -0.03, 0.05196152422706632, 0.0;
      } else if (j == 2) {
        p << 0.03, 0.05196152422706632, 0.0;
      } else if (j == 3) {
        p << -0.06, 0.10392304845413263, 0.0;
      } else if (j == 4) {
        p << 0.0, 0.10392304845413263, 0.0;
      } else if (j == 5) {
        p << 0.06, 0.10392304845413263, 0.0;
      } else if (j == 6) {
        p << -0.09, 0.15588457268119896, 0.0;
      } else if (j == 7) {
        p << -0.03, 0.15588457268119896, 0.0;
      } else if (j == 8) {
        p << 0.03, 0.15588457268119896, 0.0;
      } else if (j == 9) {
        p << 0.09, 0.15588457268119896, 0.0;
      } else if (j == 10) {
        p << -0.12, 0.2078460969082653, 0.0;
      } else if (j == 11) {
        // TODO: randomize the last sphere
        // x between -1.5 to 1.5
        // y between -1.0 and -1.5
        // cur ball position
        // int col = i % 20;
        // p << -2.4 + static_cast<double>(col) * (4.8 / 20.0), -1.7, 0.0;

        double random_angle =
            static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
        p << 0.0 + 0.15 * cos(random_angle), -0.12 + 0.03 * sin(random_angle),
            0.0;

        // p << 0.0, -2.0, 0.0;
      }

      h_spheres[i * numSpheres + j].center = p;

      h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d::Zero();

      h_spheres[i * numSpheres + j].mass = 0.17;

      if (j == 11) [[unlikely]] {
        // a random aiming point, from (0,0.25) to (0.0,3.5)
        Eigen::Vector3d random_target(
            0.0, 0.03 + static_cast<double>(rand()) / RAND_MAX * 0.15, 0.0);
        Eigen::Vector3d direction = random_target - p;
        direction.normalize();
        // scale up the velocity to 8.0 to 20.0,
        h_spheres[i * numSpheres + j].velocity =
            direction * 1.2 +
            static_cast<double>(rand()) / RAND_MAX * 0.5 * direction;
        // h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d(0.0, 1.2,
        // 0.0);

        h_spheres[i * numSpheres + j].mass = 0.17;
      }

      h_spheres[i * numSpheres + j].radius = 0.03;

      // initialize material properties
      h_spheres[i * numSpheres + j].stiffness = 10000.0;
      h_spheres[i * numSpheres + j].damping = 1e-10;  //

      // if (j == 0) {
      //   p << 0.0, 0.0, 0.0;
      // } else if (j == 1) {
      //   p << -0.03, 0.05196152422706632, 0.0;
      // } else if (j == 2) {
      //   p << 0.03, 0.05196152422706632, 0.0;
      // } else if (j == 3) {
      //   // TODO: randomize the last sphere
      //   // x between -1.5 to 1.5
      //   // y between -1.0 and -1.5
      //   // cur ball position
      //   // int col = i % 20;
      //   // p << -2.4 + static_cast<double>(col) * (4.8 / 20.0), -1.7, 0.0;

      //   double random_angle =
      //       static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
      //   p << 0.0 + 0.05 * cos(random_angle), -0.12 + 0.03 *
      //   sin(random_angle),
      //       0.0;

      //   // p << 0.0, -2.0, 0.0;
      // }

      // h_spheres[i * numSpheres + j].center = p;

      // h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d::Zero();

      // h_spheres[i * numSpheres + j].mass = 0.17;

      // if (j == 3) [[unlikely]] {
      //   Eigen::Vector3d random_target(0.0, 0.0, 0.0);
      //   Eigen::Vector3d direction = random_target - p;
      //   direction.normalize();
      //   // scale up the velocity to 8.0 to 20.0, random
      //   h_spheres[i * numSpheres + j].velocity =
      //       direction * 1.2 +
      //       static_cast<double>(rand()) / RAND_MAX * 0.5 * direction;
      //   // h_spheres[i * numSpheres + j].velocity = Eigen::Vector3d(0.0, 5.0,
      //   // 0.0);

      //   h_spheres[i * numSpheres + j].mass = 0.17;
      // }

      // h_spheres[i * numSpheres + j].radius = 0.03;

      // // initialize material properties
      // h_spheres[i * numSpheres + j].stiffness = 10000.0;
      // h_spheres[i * numSpheres + j].damping = 1e-20;
    }
  }

  Plane* h_planes = new Plane[numProblems * numPlanes];
  for (int i = 0; i < numProblems; i++) {
    for (int j = 0; j < numPlanes; j++) {
      if (j == 0) {
        h_planes[i * numPlanes + j].p1 << -0.25, 1.0, 0.0;  // -0.25 | -0.1
        h_planes[i * numPlanes + j].p2 << -0.25, 0.0, 0.0;
        h_planes[i * numPlanes + j].n << 1.0, 0.0, 0.0;
      } else if (j == 1) {
        h_planes[i * numPlanes + j].p1 << 0.0, 0.35, 0.0;
        h_planes[i * numPlanes + j].p2 << 1.0, 0.35, 0.0;  // 0.35 | 0.09
        h_planes[i * numPlanes + j].n << 0.0, -1.0, 0.0;
      } else if (j == 2) {
        h_planes[i * numPlanes + j].p1 << 0.25, 1.0, 0.0;
        h_planes[i * numPlanes + j].p2 << 0.25, 1.0, 0.0;
        h_planes[i * numPlanes + j].n << -1.0, 0.0, 0.0;
      } else if (j == 3) {
        h_planes[i * numPlanes + j].p1 << 0.0, -0.2, 0.0;
        h_planes[i * numPlanes + j].p2 << 1.0, -0.2, 0.0;
        h_planes[i * numPlanes + j].n << 0.0, 1.0, 0.0;
      }

      h_planes[i * numPlanes + j].stiffness = 10000.0;
      h_planes[i * numPlanes + j].damping = 1e-10;
    }
  }

  FullSolveSAP solver;

  solver.init(h_spheres, h_planes, numProblems, numSpheres, numPlanes,
              numContacts, false);

  // Record the start time
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 1; i++) {
    // std::cout << "Step " << i << std::endl;
    solver.step(800);
  }

  // Record the end time
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate the duration
  std::chrono::duration<double, std::milli> duration = end - start;

  solver.destroy();

  return duration.count();
}

GTEST_TEST(KernelTest, FullSolveTest) {
  double sum = 0.0;
  double min_val = std::numeric_limits<double>::max();
  double max_val = std::numeric_limits<double>::min();

  for (int i = 0; i < 10; i++) {
    double timing = run(1);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "1 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(10);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "10 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(50);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "50 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(100);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "100 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(200);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "200 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(300);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "300 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(400);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "400 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(500);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "500 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(600);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "600 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(700);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "700 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(800);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "800 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(900);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "900 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(1000);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "1000 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(2000);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "2000 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(5000);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "5000 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(10000);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "10000 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;

  sum = 0.0;
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::min();
  for (int i = 0; i < 10; i++) {
    double timing = run(15000);
    sum += timing;
    min_val = std::min(timing, min_val);
    max_val = std::max(timing, max_val);
  }
  std::cout << "15000 Average: " << sum / 10.0 << "  min: " << min_val
            << "  max: " << max_val << std::endl;
}

// ===================================================
// END OF ACTUAL SAP SOLVER FUNCTION CALLS
// ===================================================

}  // namespace
}  // namespace drake

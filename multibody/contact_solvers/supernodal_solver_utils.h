#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/contact_solvers/supernodal_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

std::unique_ptr<BlockSparseSymmetricMatrix> ConstructHessian(
    std::vector<Eigen::MatrixXd> mass_matrices,
    std::vector<BlockTriplet> jacobian_blocks,
    std::vector<std::vector<int>> row_to_triplet_index);

bool AssembleHessian(std::vector<Eigen::MatrixXd> mass_matrices,
                     std::vector<BlockTriplet> jacobian_blocks,
                     const std::vector<Eigen::MatrixXd>& weight_matrix,
                     std::vector<std::vector<int>> row_to_triplet_index,
                     BlockSparseSymmetricMatrix* H);

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
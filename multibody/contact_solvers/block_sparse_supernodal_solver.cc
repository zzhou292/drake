#include "drake/multibody/contact_solvers/block_sparse_supernodal_solver.h"

#include <utility>

#include "drake/multibody/contact_solvers/supernodal_solver_utils.h"

using Eigen::MatrixXd;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

BlockSparseSuperNodalSolver::BlockSparseSuperNodalSolver(
    int num_jacobian_row_blocks, std::vector<BlockTriplet> jacobian_blocks,
    std::vector<Eigen::MatrixXd> mass_matrices)
    : jacobian_blocks_(std::move(jacobian_blocks)),
      mass_matrices_(std::move(mass_matrices)) {
  const std::vector<int> jacobian_column_block_size =
      GetJacobianBlockSizesVerifyTriplets(jacobian_blocks_);
  /* Throw an exception if verification fails. */
  if (!MassMatrixPartitionEqualsJacobianPartition(jacobian_column_block_size,
                                                  mass_matrices_)) {
    throw std::runtime_error(
        "Mass matrices and constraint Jacobians are incompatible.");
  }
  row_to_triplet_index_ =
      GetRowToTripletMapping(num_jacobian_row_blocks, jacobian_blocks_);

  H_ =
      ConstructHessian(mass_matrices_, jacobian_blocks_, row_to_triplet_index_);
  /* The solver analyzes the sparsity pattern of the H_ (currently a zero
   matrix) so that subsequent updates to the matrix can use UpdateMatrix()
   that doesn't perform symbolic factorization and allocation. */
  solver_.SetMatrix(*H_);
}

BlockSparseSuperNodalSolver::~BlockSparseSuperNodalSolver() = default;

bool BlockSparseSuperNodalSolver::DoSetWeightMatrix(
    const std::vector<Eigen::MatrixXd>& weight_matrix) {
  if (!AssembleHessian(mass_matrices_, jacobian_blocks_, weight_matrix,
                       row_to_triplet_index_, H_.get())) {
    return false;
  }
  solver_.UpdateMatrix(*H_);
  return true;
}

bool BlockSparseSuperNodalSolver::DoFactor() {
  return solver_.Factor();
}

void BlockSparseSuperNodalSolver::DoSolveInPlace(Eigen::VectorXd* b) const {
  solver_.SolveInPlace(b);
}

Eigen::MatrixXd BlockSparseSuperNodalSolver::DoMakeFullMatrix() const {
  return H_->MakeDenseMatrix();
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

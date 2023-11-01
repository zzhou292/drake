#include "drake/multibody/contact_solvers/cuda_supernodal_solver.h"

#include <utility>

#include "drake/multibody/contact_solvers/cuda_solvers/cuda_sparse_cholesky_solver.h"
#include "drake/multibody/contact_solvers/supernodal_solver_utils.h"

using Eigen::MatrixXd;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

CUDASuperNodalSolver::CUDASuperNodalSolver(
    int num_jacobian_row_blocks, std::vector<BlockTriplet> jacobian_blocks,
    std::vector<Eigen::MatrixXd> mass_matrices,
    CUDASuperNodalSolverType solver_type)
    : jacobian_blocks_(std::move(jacobian_blocks)),
      mass_matrices_(std::move(mass_matrices)),
      solver_type_(solver_type) {
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
  H_csr_ = std::make_unique<CSRMatrix>(*H_);

  // Instantiate CUDA linear solver.
  switch (solver_type_) {
    case CUDASuperNodalSolverType::kCholesky:
      solver_ = std::make_unique<CUDASparseCholeskySolver>();
      break;
    default:
      throw std::logic_error("Unsupported CUDASuperNodalSolverType.");
  }

  solver_->Init();
  solver_->SetMatrix(*H_csr_);
}

CUDASuperNodalSolver::~CUDASuperNodalSolver() = default;

bool CUDASuperNodalSolver::DoSetWeightMatrix(
    const std::vector<Eigen::MatrixXd>& weight_matrix) {
  // Assemble the block sparse matrix and reset the values in the CSR matrix.
  if (!AssembleHessian(mass_matrices_, jacobian_blocks_, weight_matrix,
                       row_to_triplet_index_, H_.get())) {
    return false;
  }
  H_csr_->SetValuesFrom(*H_);
  solver_->ResetMatrixValues(*H_csr_);
  return true;
}

bool CUDASuperNodalSolver::DoFactor() {
  return solver_->Factor();
}

void CUDASuperNodalSolver::DoSolveInPlace(Eigen::VectorXd* b) const {
  solver_->SolveInPlace(b);
}

Eigen::MatrixXd CUDASuperNodalSolver::DoMakeFullMatrix() const {
  return H_csr_->MakeDenseMatrix();
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

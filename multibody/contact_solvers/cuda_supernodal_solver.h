#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/contact_solvers/cuda_solvers/cuda_sparse_linear_solver.h"
#include "drake/multibody/contact_solvers/csr_matrix.h"
#include "drake/multibody/contact_solvers/supernodal_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

enum class CUDASuperNodalSolverType {
  kCholesky = 0, // Use a sparse cholesky solver.
};

/* Sparse CUDA based supernodal solver. */
class CUDASuperNodalSolver final : public SuperNodalSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CUDASuperNodalSolver)

  /* Constructs a CUDASuperNodalSolver.
   @param[in] num_jacobian_row_blocks
     Number of row blocks in the matrix J.
   @param[in] jacobian_blocks
     A vector of triplets (p, t, Jₚₜ) specifying the non-zero blocks of the
     Jacobian matrix.  The number of block columns nₜ is inferred
     from the largest column index t in the vector of triplets (p, t, Jₚₜ).
     An exception is thrown if any of the following conditions fail:
       1) There is at least one triplet  (p, t, Jₚₜ) with column index t for
       each t ∈ [0, nₜ).
       2) There is at most two triplets (p, t, Jₚₜ) with the same row
       index p.
   @param[in] mass_matrices
     Specifies a block-diagonal mass matrix M of size nᵥ x nᵥ. The block
     columns of the mass matrix and the block columns of the Jacobian J both
     induce a partition of the set {0, 1, ..., nᵥ - 1}, where nᵥ denotes the
     number of scalar variables. These two partitions must be the same,
     otherwise an exception is thrown.
  @param[in] solver_type
     Specifies the type of CUDA linear solver to use. */
  CUDASuperNodalSolver(int num_jacobian_row_blocks,
                       std::vector<BlockTriplet> jacobian_blocks,
                       std::vector<Eigen::MatrixXd> mass_matrices,
                       CUDASuperNodalSolverType solver_type =
                           CUDASuperNodalSolverType::kCholesky);

  ~CUDASuperNodalSolver() final;

 private:
  /* NVI implementations. */
  bool DoSetWeightMatrix(
      const std::vector<Eigen::MatrixXd>& block_diagonal_G) final;
  Eigen::MatrixXd DoMakeFullMatrix() const final;
  bool DoFactor() final;
  void DoSolveInPlace(Eigen::VectorXd* b) const final;

  int DoGetSize() const final {
    DRAKE_DEMAND(H_ != nullptr);
    return H_->cols();
  }

  /* Matrix H in the sparse system H⋅x = b where H = M + Jᵀ⋅G⋅J. */
  std::unique_ptr<BlockSparseSymmetricMatrix> H_;
  /* CSR representation of H. */
  std::unique_ptr<CSRMatrix> H_csr_;
  /* The i-th entry contains the indices into `jacobian_blocks_` for the
   jacobian blocks associated with the i-th block row of the full jacobian
   matrix. Each entry contains at least one and at most two entries. */
  std::vector<std::vector<int>> row_to_triplet_index_;
  /* Block indices and non-zero entries of the block sparse matrix J. */
  std::vector<BlockTriplet> jacobian_blocks_;
  /* Diagonal blocks of the block diagonal matrix M. */
  std::vector<Eigen::MatrixXd> mass_matrices_;

  std::unique_ptr<CUDASparseLinearSolver> solver_;
  CUDASuperNodalSolverType solver_type_;
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

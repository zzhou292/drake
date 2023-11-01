#include "drake/multibody/contact_solvers/supernodal_solver_utils.h"

using Eigen::MatrixXd;


namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

std::unique_ptr<BlockSparseSymmetricMatrix> ConstructHessian(
    std::vector<Eigen::MatrixXd> mass_matrices,
    std::vector<BlockTriplet> jacobian_blocks,
    std::vector<std::vector<int>> row_to_triplet_index) {
  const int num_nodes = mass_matrices.size();
  std::vector<int> block_sizes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    block_sizes[i] = mass_matrices[i].rows();
  }
  /* Build diagonal entry in sparsity pattern. */
  std::vector<std::vector<int>> sparsity(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    sparsity[i].emplace_back(i);
  }
  /* Build off-diagonal entry in sparsity pattern. */
  const int num_constraints = row_to_triplet_index.size();
  for (int r = 0; r < num_constraints; ++r) {
    const std::vector<int>& triplet_indices = row_to_triplet_index[r];
    DRAKE_DEMAND(triplet_indices.size() <= 2);
    if (triplet_indices.size() == 2) {
      const int j = jacobian_blocks[triplet_indices[0]].col;
      const int i = jacobian_blocks[triplet_indices[1]].col;
      DRAKE_DEMAND(j < i);
      sparsity[j].emplace_back(i);
    }
  }
  return std::make_unique<BlockSparseSymmetricMatrix>(
      BlockSparsityPattern(std::move(block_sizes), std::move(sparsity)));
}

bool AssembleHessian(std::vector<Eigen::MatrixXd> mass_matrices,
                     std::vector<BlockTriplet> jacobian_blocks,
                     const std::vector<Eigen::MatrixXd>& weight_matrix,
                     std::vector<std::vector<int>> row_to_triplet_index,
                     BlockSparseSymmetricMatrix* H) {
  H->SetZero();
  /* Add mass matrices. */
  const int block_cols = mass_matrices.size();
  for (int i = 0; i < block_cols; ++i) {
    H->SetBlock(i, i, mass_matrices[i]);
  }
  /* Add in JᵀGJ terms. */
  const int num_constraints = row_to_triplet_index.size();
  DRAKE_THROW_UNLESS(ssize(weight_matrix) >= num_constraints);
  // TODO(xuchenhan-tri): Getting the starting indices of G blocks as well as
  // checking partitions of G refines partitions of block rows of J should
  // happen in the base class.
  /* Recall that the partition of the weight matrix G is a refinement on the
   partition of the block rows of J. Here we use `weight_start` and
   `weight_end` to track the indices into `weight_matrix` that corresponds to
   the k-th block row of J. */
  int weight_start = 0;
  int weight_end = 0;
  for (int k = 0; k < num_constraints; ++k) {
    const std::vector<int>& triplet_indices = row_to_triplet_index[k];
    const int num_constraint_equations =
        jacobian_blocks[triplet_indices[0]].value.rows();
    int G_rows = 0;
    while (G_rows < num_constraint_equations &&
           weight_end < ssize(weight_matrix)) {
      G_rows += weight_matrix[weight_end++].rows();
    }
    if (G_rows != num_constraint_equations) {
      return false;
    }

    if (triplet_indices.size() == 1) {
      const MatrixBlock<double>& J = jacobian_blocks[triplet_indices[0]].value;
      const MatrixBlock<double> GJ = J.LeftMultiplyByBlockDiagonal(
          weight_matrix, weight_start, weight_end - 1);
      MatrixXd JTGJ = MatrixXd::Zero(J.cols(), J.cols());
      // TODO(xuchenhan-tri): Consider adding a more specialized routine for
      // computing JᵢᵀGJⱼ to further exploit sparsity. */
      J.TransposeAndMultiplyAndAddTo(GJ, &JTGJ);
      const int c = jacobian_blocks[triplet_indices[0]].col;
      H->AddToBlock(c, c, std::move(JTGJ));
    } else {
      DRAKE_DEMAND(triplet_indices.size() == 2);
      const int j = jacobian_blocks[triplet_indices[0]].col;
      const int i = jacobian_blocks[triplet_indices[1]].col;
      DRAKE_DEMAND(j < i);
      const MatrixBlock<double>& Jj = jacobian_blocks[triplet_indices[0]].value;
      const MatrixBlock<double>& Ji = jacobian_blocks[triplet_indices[1]].value;

      // TODO(xuchenhan-tri): Consider adding a more specialized routine for
      // computing JᵀGJ to further exploit sparsity. */
      const MatrixBlock<double> GJj = Jj.LeftMultiplyByBlockDiagonal(
          weight_matrix, weight_start, weight_end - 1);
      const MatrixBlock<double> GJi = Ji.LeftMultiplyByBlockDiagonal(
          weight_matrix, weight_start, weight_end - 1);

      MatrixXd JiTGJi = MatrixXd::Zero(Ji.cols(), Ji.cols());
      MatrixXd JiTGJj = MatrixXd::Zero(Ji.cols(), Jj.cols());
      MatrixXd JjTGJj = MatrixXd::Zero(Jj.cols(), Jj.cols());

      Ji.TransposeAndMultiplyAndAddTo(GJi, &JiTGJi);
      Ji.TransposeAndMultiplyAndAddTo(GJj, &JiTGJj);
      Jj.TransposeAndMultiplyAndAddTo(GJj, &JjTGJj);

      H->AddToBlock(i, i, JiTGJi);
      H->AddToBlock(i, j, JiTGJj);
      H->AddToBlock(j, j, JjTGJj);
    }
    weight_start = weight_end;
  }
  return true;
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
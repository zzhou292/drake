#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

/* A sparse matrix data structure stored in CSR format. */
class CSRMatrix {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(CSRMatrix);

  CSRMatrix(const std::vector<double>& values,
            const std::vector<int>& row_indices,
            const std::vector<int>& col_indices, int rows, int cols)
      : values_(values),
        row_indices_(row_indices),
        col_indices_(col_indices),
        rows_(rows),
        cols_(cols) {}

  explicit CSRMatrix(const MatrixX<double>& M);
  explicit CSRMatrix(const BlockSparseSymmetricMatrix& M);

  void SetValuesFrom(const MatrixX<double>& M);
  void SetValuesFrom(const BlockSparseSymmetricMatrix& M);

  bool HasSameSparsityStructure(const MatrixX<double>& M);
  bool HasSameSparsityStructure(const BlockSparseSymmetricMatrix& M);

  MatrixX<double> MakeDenseMatrix() const;

  const std::vector<double>& values() const { return values_; }
  const std::vector<int>& row_indices() const { return row_indices_; }
  const std::vector<int>& col_indices() const { return col_indices_; }

  int rows() const { return rows_; }
  int cols() const { return cols_; }

 private:
  std::vector<double> values_;
  std::vector<int> row_indices_;
  std::vector<int> col_indices_;
  int rows_;
  int cols_;
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
#include "drake/multibody/contact_solvers/csr_matrix.h"
namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

CSRMatrix::CSRMatrix(const MatrixX<double>& M) {
  rows_ = M.rows();
  cols_ = M.cols();

  int row_offset = 0;
  row_indices_.push_back(row_offset);
  for (int row = 0; row < M.rows(); ++row) {
    int nnz_columns = 0;
    for (int col = 0; col < M.cols(); ++col) {
      if (M(row, col) != 0) {
        values_.push_back(M(row, col));
        col_indices_.push_back(col);
        ++nnz_columns;
      }
    }
    row_offset += nnz_columns;
    row_indices_.push_back(row_offset);
  }
}

CSRMatrix::CSRMatrix(const BlockSparseSymmetricMatrix& M) {
  rows_ = M.rows();
  cols_ = M.cols();

  int row_offset = 0;
  row_indices_.push_back(row_offset);
  for (int i = 0; i < M.block_rows(); ++i) {
    const int rows = M.diagonal_block(i).rows();
    for (int row = 0; row < rows; ++row) {
      int nnz_columns = 0;
      for (int j = 0; j < M.block_cols(); ++j) {
        if (M.HasBlock(i, j)) {
          const MatrixX<double>& block =
              (i < j ? M.block(j, i).transpose() : M.block(i, j));
          nnz_columns += block.cols();
          const int starting_col = M.starting_cols()[j];
          for (int col = 0; col < block.cols(); ++col) {
            col_indices_.push_back(starting_col + col);
            values_.push_back(block(row, col));
          }
        }
      }
      row_offset += nnz_columns;
      row_indices_.push_back(row_offset);
    }
  }
}

  void CSRMatrix::SetValuesFrom(const MatrixX<double>& M) {
    DRAKE_DEMAND(HasSameSparsityStructure(M));
    int row_offset = 0;
    for (int row = 0; row < M.rows(); ++row) {
      int nnz_columns = 0;
      for (int col = 0; col < M.cols(); ++col) {
        if (M(row, col) != 0) {
          values_[row_offset + nnz_columns] = M(row, col);
          ++nnz_columns;
        }
      }
      row_offset += nnz_columns;
    }
  }

  void CSRMatrix::SetValuesFrom(const BlockSparseSymmetricMatrix& M) {
    DRAKE_DEMAND(HasSameSparsityStructure(M));
    int row_offset = 0;
    for (int i = 0; i < M.block_rows(); ++i) {
      const int rows = M.diagonal_block(i).rows();
      for (int row = 0; row < rows; ++row) {
        int nnz_columns = 0;
        for (int j = 0; j < M.block_cols(); ++j) {
          if (M.HasBlock(i, j)) {
            const MatrixX<double>& block =
                (i < j ? M.block(j, i).transpose() : M.block(i, j));
            for (int col = 0; col < block.cols(); ++col) {
              values_[row_offset + nnz_columns + col] = block(row, col);
            }
            nnz_columns += block.cols();
          }
        }
        row_offset += nnz_columns;
      }
    }
  }

  bool CSRMatrix::HasSameSparsityStructure(const MatrixX<double>& M) {
    CSRMatrix A(M);
    return A.col_indices() == col_indices() && A.row_indices() == row_indices();
  }

  bool CSRMatrix::HasSameSparsityStructure(const BlockSparseSymmetricMatrix& M) {
    CSRMatrix A(M);
    return A.col_indices() == col_indices() && A.row_indices() == row_indices();
  }

MatrixX<double> CSRMatrix::MakeDenseMatrix() const {
  MatrixX<double> M = MatrixX<double>::Zero(rows_, cols_);
  for (int row = 0; row < rows_; ++row) {
    for (int flat = row_indices_[row]; flat < row_indices_[row + 1]; ++flat) {
      M(row, col_indices_[flat]) = values_[flat];
    }
  }
  return M;
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
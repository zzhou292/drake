#include "drake/multibody/contact_solvers/csr_matrix.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;

// Simple 4x4 Matrix and its CSR representation.
// clang-format off
const MatrixXd M =
    (MatrixXd(4, 4) << 5, 0, 0, 0,
                       0, 8, 0, 6,
                       0, 0, 3, 0,
                       0, 6, 0, 2).finished();
// clang-format on
const std::vector<double> M_values{5, 8, 6, 3, 6, 2};
const std::vector<int> M_row_indices{0, 1, 3, 4, 6};
const std::vector<int> M_col_indices{0, 1, 3, 2, 1, 3};

// 4x4 Matrix with the same sparsity structure as M but different values.
// clang-format off
const MatrixXd M_neg =
    (MatrixXd(4, 4) << -5,  0,  0,  0,
                        0, -8,  0, -6,
                        0,  0, -3,  0,
                        0, -6,  0, -2).finished();
// clang-format on
const std::vector<double> M_neg_values{-5, -8, -6, -3, -6, -2};

// Block-sparse symmetric version of M.
BlockSparseSymmetricMatrix MakeSymmetricM() {
  std::vector<int> diag{1, 1, 1, 1};
  std::vector<std::vector<int>> sparsity;
  sparsity.push_back(std::vector<int>{0});
  sparsity.push_back(std::vector<int>{1, 3});
  sparsity.push_back(std::vector<int>{2});
  sparsity.push_back(std::vector<int>{3});
  BlockSparsityPattern pattern(diag, sparsity);
  BlockSparseSymmetricMatrix M_blocks(pattern);
  const MatrixXd M00 = (MatrixXd(1, 1) << 5).finished();
  const MatrixXd M11 = (MatrixXd(1, 1) << 8).finished();
  const MatrixXd M22 = (MatrixXd(1, 1) << 3).finished();
  const MatrixXd M31 = (MatrixXd(1, 1) << 6).finished();
  const MatrixXd M33 = (MatrixXd(1, 1) << 2).finished();
  M_blocks.SetBlock(0, 0, M00);
  M_blocks.SetBlock(1, 1, M11);
  M_blocks.SetBlock(2, 2, M22);
  M_blocks.SetBlock(3, 1, M31);
  M_blocks.SetBlock(3, 3, M33);
  return M_blocks;
}

// Block-sparse symmetric version of M.
BlockSparseSymmetricMatrix MakeSymmetricM_neg() {
  std::vector<int> diag{1, 1, 1, 1};
  std::vector<std::vector<int>> sparsity;
  sparsity.push_back(std::vector<int>{0});
  sparsity.push_back(std::vector<int>{1, 3});
  sparsity.push_back(std::vector<int>{2});
  sparsity.push_back(std::vector<int>{3});
  BlockSparsityPattern pattern(diag, sparsity);
  BlockSparseSymmetricMatrix M_blocks(pattern);
  const MatrixXd M00 = (MatrixXd(1, 1) << -5).finished();
  const MatrixXd M11 = (MatrixXd(1, 1) << -8).finished();
  const MatrixXd M22 = (MatrixXd(1, 1) << -3).finished();
  const MatrixXd M31 = (MatrixXd(1, 1) << -6).finished();
  const MatrixXd M33 = (MatrixXd(1, 1) << -2).finished();
  M_blocks.SetBlock(0, 0, M00);
  M_blocks.SetBlock(1, 1, M11);
  M_blocks.SetBlock(2, 2, M22);
  M_blocks.SetBlock(3, 1, M31);
  M_blocks.SetBlock(3, 3, M33);
  return M_blocks;
}

// Block-sparse 9x9 Matrix and its CSR representation.
// clang-format off
const Matrix2d A00 = (Eigen::Matrix2d() << 11, 12,
                                           12, 15).finished();
const Matrix3d A11 = (Eigen::Matrix3d() << 1, 2, 3,
                                           2, 5, 6,
                                           3, 6, 9).finished();
const Matrix4d A22 = (Eigen::Matrix4d() << 11, 12, 13, 23,
                                           12, 15, 16, 26,
                                           13, 16, 19, 19,
                                           23, 26, 19, 23).finished();
const Eigen::Matrix<double, 4, 3> A21 =
    (Eigen::Matrix<double, 4, 3>() << 11, 22, 33,
                                      33, 44, 55,
                                      66, 11, 88,
                                      22, 56, 78).finished();
const std::vector<double> A_values{11, 12,
                                   12, 15,
                                           1,  2,  3, 11, 33, 66, 22,
                                           2,  5,  6, 22, 44, 11, 56,
                                           3,  6,  9, 33, 55, 88, 78,
                                          11, 22, 33, 11, 12, 13, 23,
                                          33, 44, 55, 12, 15, 16, 26,
                                          66, 11, 88, 13, 16, 19, 19,
                                          22, 56, 78, 23, 26, 19, 23};
const std::vector<int> A_col_indices{0, 1,
                                     0, 1,
                                           2, 3, 4, 5, 6, 7, 8,
                                           2, 3, 4, 5, 6, 7, 8,
                                           2, 3, 4, 5, 6, 7, 8,
                                           2, 3, 4, 5, 6, 7, 8,
                                           2, 3, 4, 5, 6, 7, 8,
                                           2, 3, 4, 5, 6, 7, 8,
                                           2, 3, 4, 5, 6, 7, 8};
const std::vector<int> A_row_indices{0, 2, 4, 11, 18, 25, 32, 39, 46, 53};
// clang-format on

/* Makes a dense matrix
   A =   A00 |  0  |  0
        -----------------
          0  | A11 | A12
        -----------------
          0  | A21 | A22
 where A12 = A21.transpose().
*/
MatrixXd MakeDenseA() {
  MatrixXd A = MatrixXd::Zero(9, 9);
  A.block<2, 2>(0, 0) = A00;
  A.block<3, 3>(2, 2) = A11;
  A.block<4, 3>(5, 2) = A21;
  A.block<3, 4>(2, 5) = A21.transpose();
  A.block<4, 4>(5, 5) = A22;
  return A;
}

/* Makes a block sparse symmetric matrix
   A =   A00 |  0  |  0
        -----------------
          0  | A11 | A12
        -----------------
          0  | A21 | A22
 where A21 = A12.transpose(). */
BlockSparseSymmetricMatrix MakeSymmetricA() {
  std::vector<int> diag{2, 3, 4};
  std::vector<std::vector<int>> sparsity;
  sparsity.push_back(std::vector<int>{0});
  sparsity.push_back(std::vector<int>{1, 2});
  sparsity.push_back(std::vector<int>{2});
  BlockSparsityPattern pattern(diag, sparsity);
  BlockSparseSymmetricMatrix A_blocks(pattern);
  A_blocks.SetBlock(0, 0, A00);
  A_blocks.SetBlock(1, 1, A11);
  A_blocks.SetBlock(2, 1, A21);
  A_blocks.SetBlock(2, 2, A22);
  return A_blocks;
}

void TestCSRMatrixIsM(const CSRMatrix& A) {
  EXPECT_EQ(A.rows(), M.rows());
  EXPECT_EQ(A.cols(), M.cols());
  EXPECT_THAT(A.values(), testing::ElementsAreArray(M_values));
  EXPECT_THAT(A.row_indices(), testing::ElementsAreArray(M_row_indices));
  EXPECT_THAT(A.col_indices(), testing::ElementsAreArray(M_col_indices));
  EXPECT_TRUE(CompareMatrices(A.MakeDenseMatrix(), M));
}

void TestCSRMatrixIsM_neg(const CSRMatrix& A) {
  EXPECT_EQ(A.rows(), M.rows());
  EXPECT_EQ(A.cols(), M.cols());
  EXPECT_THAT(A.values(), testing::ElementsAreArray(M_neg_values));
  EXPECT_THAT(A.row_indices(), testing::ElementsAreArray(M_row_indices));
  EXPECT_THAT(A.col_indices(), testing::ElementsAreArray(M_col_indices));
  EXPECT_TRUE(CompareMatrices(A.MakeDenseMatrix(), M_neg));
}

void TestCSRMatrixIsA(const CSRMatrix& B) {
  const MatrixXd A = MakeDenseA();
  EXPECT_EQ(B.rows(), A.rows());
  EXPECT_EQ(B.cols(), A.cols());
  EXPECT_THAT(B.values(), testing::ElementsAreArray(A_values));
  EXPECT_THAT(B.row_indices(), testing::ElementsAreArray(A_row_indices));
  EXPECT_THAT(B.col_indices(), testing::ElementsAreArray(A_col_indices));
  EXPECT_TRUE(CompareMatrices(B.MakeDenseMatrix(), A));
}

GTEST_TEST(CSRMatrix, Construction) {
  CSRMatrix A(M_values, M_row_indices, M_col_indices, 4, 4);
  TestCSRMatrixIsM(A);
  CSRMatrix B(A_values, A_row_indices, A_col_indices, 9, 9);
  TestCSRMatrixIsA(B);
}

GTEST_TEST(CSRMatrix, MatrixXConstruction) {
  CSRMatrix A(M);
  TestCSRMatrixIsM(A);
  CSRMatrix B(MakeDenseA());
  TestCSRMatrixIsA(B);
}

GTEST_TEST(CSRMatrix, BlockSparseSymmetricMatrixConstruction) {
  CSRMatrix A(MakeSymmetricM());
  TestCSRMatrixIsM(A);
  CSRMatrix B(MakeSymmetricA());
  TestCSRMatrixIsA(B);
}

GTEST_TEST(CSRMatrix, SetValuesFrom) {
  BlockSparseSymmetricMatrix M_sparse = MakeSymmetricM();
  BlockSparseSymmetricMatrix M_neg_sparse = MakeSymmetricM_neg();

  CSRMatrix A(M);
  CSRMatrix B(M_sparse);

  EXPECT_TRUE(A.HasSameSparsityStructure(M));
  EXPECT_TRUE(A.HasSameSparsityStructure(M_neg));
  EXPECT_TRUE(A.HasSameSparsityStructure(M_sparse));
  EXPECT_TRUE(A.HasSameSparsityStructure(M_neg_sparse));
  EXPECT_TRUE(B.HasSameSparsityStructure(M));
  EXPECT_TRUE(B.HasSameSparsityStructure(M_neg));
  EXPECT_TRUE(B.HasSameSparsityStructure(M_sparse));
  EXPECT_TRUE(B.HasSameSparsityStructure(M_neg_sparse));

  EXPECT_FALSE(A.HasSameSparsityStructure(MatrixXd::Ones(4, 4)));
  EXPECT_FALSE(B.HasSameSparsityStructure(MatrixXd::Ones(4, 4)));

  A.SetValuesFrom(M_neg);
  B.SetValuesFrom(M_neg_sparse);

  TestCSRMatrixIsM_neg(A);
  TestCSRMatrixIsM_neg(B);
}

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
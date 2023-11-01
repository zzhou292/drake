#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/csr_matrix.h"
#include "drake/multibody/contact_solvers/cuda_solvers/cuda_sparse_linear_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

class CUDASparseCholeskySolver : public CUDASparseLinearSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CUDASparseCholeskySolver);

  CUDASparseCholeskySolver() {}
  ~CUDASparseCholeskySolver() final;

  void Init() final;
  void SetMatrix(const CSRMatrix& M) final;
  void ResetMatrixValues(const CSRMatrix& M) final;
  bool Factor() final;
  void SolveInPlace(Eigen::VectorXd* b) const final;
private:
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
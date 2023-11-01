#pragma once

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/csr_matrix.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

class CUDASparseLinearSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CUDASparseLinearSolver);

  CUDASparseLinearSolver() {}
  virtual ~CUDASparseLinearSolver() = default;

  virtual void Init() = 0;
  virtual void SetMatrix(const CSRMatrix& M) = 0;
  virtual void ResetMatrixValues(const CSRMatrix& M) = 0;
  virtual bool Factor() = 0;
  virtual void SolveInPlace(Eigen::VectorXd* b) const = 0;
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
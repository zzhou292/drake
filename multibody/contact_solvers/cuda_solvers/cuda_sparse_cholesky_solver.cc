#include "drake/multibody/contact_solvers/cuda_solvers/cuda_sparse_cholesky_solver.h"

#include <vector>

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

CUDASparseCholeskySolver::~CUDASparseCholeskySolver() {}
void CUDASparseCholeskySolver::Init() {}
void CUDASparseCholeskySolver::SetMatrix(const CSRMatrix&) {}
void CUDASparseCholeskySolver::ResetMatrixValues(const CSRMatrix&) {}
bool CUDASparseCholeskySolver::Factor() { return false; }
void CUDASparseCholeskySolver::SolveInPlace(Eigen::VectorXd*) const {}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
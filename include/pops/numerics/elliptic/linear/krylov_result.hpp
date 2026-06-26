#pragma once

/// @file
/// @brief KrylovResult -- the shared result type of every Krylov solve in
///        `include/pops/numerics/elliptic/linear`.
///
/// One definition shared by krylov_solver.hpp (the GeometricMG-coupled BiCGStab, TensorKrylovSolver)
/// and generic_krylov.hpp (the matrix-free richardson/cg/bicgstab loops), so the two never carry
/// hand-synchronised copies (a cross-TU ODR hazard if they ever drift).

#include <pops/core/foundation/types.hpp>

namespace pops {

/// Outcome of a Krylov solve: iterations performed, final relative residual, convergence flag.
struct KrylovResult {
  int iters = 0;           ///< number of iterations performed
  Real rel_residual = 0;   ///< ||r_final|| / ||b|| (global L2 norm; base 1 when ||b|| == 0)
  bool converged = false;  ///< true if ||r|| <= rel_tol * ||b|| was reached
};

}  // namespace pops

#pragma once

/// @file
/// @brief DIRECT EllipticSolver backends via spectral FFT (periodic BCs), MultiFab wrappers around
///        PoissonFFT: PoissonFFTSolver (single-rank, single box) and DistributedFFTSolver (distributed, slabs).
///
/// Layer: `include/adc/numerics/elliptic`.
/// Role: solve the SAME discrete 5-point Laplacian as GeometricMG (same eigenvalues) but in ONE
/// transform instead of iterating -- far cheaper when the elliptic part dominates the run (cf. PERFORMANCE.md).
/// Same constructor signature and interface as GeometricMG, so they are interchangeable as the Coupler's
/// Elliptic parameter. Both model the EllipticSolver concept (static_assert at the bottom).
/// Contract: PoissonFFTSolver is single-rank/single box (the default case of the Coupler and the facades);
/// DistributedFFTSolver is the distributed counterpart (1 slab per rank, internal MPI transpose of PoissonFFT).
///
/// Invariants:
/// - PoissonFFTSolver throws (HARD guard, active in Release) if n_ranks() > 1 or ba.size() != 1:
///   otherwise a rank with no local box would dereference fab(0) -> SIGSEGV;
/// - solve() writes phi with zero mean (mode k=0 set to zero) and fills phi's periodic ghosts
///   (fill_boundary) so the aux derivation (centered grad phi) reads up-to-date ghosts;
/// - residual() of DistributedFFTSolver is COLLECTIVE (all_reduce_max) and first fills the
///   inter-slab halos (fill_boundary MPI).

#include <adc/numerics/elliptic/elliptic_solver.hpp>
#include <adc/numerics/elliptic/poisson_fft.hpp>
#include <adc/numerics/elliptic/poisson_operator.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cassert>
#include <functional>
#include <stdexcept>
#include <vector>

namespace adc {

class PoissonFFTSolver {
 public:
  /// @p spectral: Laplacian symbol (false = discrete 5-point stencil, bit-identical default;
  /// true = continuous symbol -(kx^2+ky^2), faithful to spectral references -- cf. PoissonFFT).
  PoissonFFTSolver(const Geometry& geom, const BoxArray& ba,
                   const BCRec& = BCRec{}, std::function<bool(Real, Real)> = {},
                   bool spectral = false)
      : geom_(geom),
        dm_(ba.size(), n_ranks()),
        phi_(ba, dm_, 1, 1),
        rhs_(ba, dm_, 1, 0),
        res_(ba, dm_, 1, 0),
        fft_(geom.domain.nx(), geom.domain.ny(), geom.xhi - geom.xlo,
             geom.yhi - geom.ylo, spectral) {
    // HARD guard (active in Release, NDEBUG does NOT remove it): this direct solver is single-rank /
    // single box. Under a system DistributionMapping with n_ranks()>1, some ranks have no local
    // box (local_size()==0) and solve() would dereference a non-existent fab(0) -> SIGSEGV. The old
    // assert vanished in Release and the protection silently disappeared. We throw on ALL ranks
    // (each one constructs the object), so no deadlock. For the distributed periodic case:
    // DistributedFFTSolver (slabs, MPI_Alltoall).
    if (n_ranks() != 1)
      throw std::runtime_error(
          "fft solver unsupported in MPI (n_ranks>1): use geometric_mg or the distributed fft "
          "solver (DistributedFFTSolver)");
    if (ba.size() != 1)
      throw std::runtime_error(
          "PoissonFFTSolver: single box required (ba.size()==1); for a distributed multi-box "
          "domain, use DistributedFFTSolver or geometric_mg");
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const Geometry& geom() const { return geom_; }

  // Solve lap(phi) = rhs in place (direct, one forward FFT + inverse).
  void solve() {
    const int Nx = geom_.domain.nx(), Ny = geom_.domain.ny();
    const ConstArray4 f = rhs_.fab(0).const_array();
    const Box2D v = rhs_.box(0);
    std::vector<double> rho(static_cast<std::size_t>(Nx) * Ny), phil;
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i)
        rho[static_cast<std::size_t>(j) * Nx + i] = f(v.lo[0] + i, v.lo[1] + j);
    fft_.solve(rho, phil);  // mode k=0 (constant) set to zero -> phi with zero mean
    Array4 p = phi_.fab(0).array();
    for (int j = 0; j < Ny; ++j)
      for (int i = 0; i < Nx; ++i)
        p(v.lo[0] + i, v.lo[1] + j) = phil[static_cast<std::size_t>(j) * Nx + i];
    // phi GHOSTS: the aux derivation (centered grad phi, solve_fields) reads the i+-1/j+-1
    // neighbors, hence the domain-boundary ghosts. GeometricMG fills them while smoothing; this
    // direct solver writes ONLY the valid cells -> without this exchange, the boundary gradient
    // would read stale ghosts (latent bug exposed by an electric source wired onto 'fft':
    // wrong Ex on the boundary ring). Pure periodic by construction (ctor guards).
    fill_boundary(phi_, geom_.domain, Periodicity{true, true});
  }

  // Discrete residual ||lap(phi) - rhs|| (~ round-off for this direct solver).
  Real residual() {
    BCRec bc;  // periodic
    poisson_residual(phi_, rhs_, geom_, bc, res_);
    return norm_inf(res_);
  }

 private:
  Geometry geom_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_, res_;
  PoissonFFT fft_;
};

static_assert(EllipticSolver<PoissonFFTSolver>,
              "PoissonFFTSolver must model EllipticSolver");

/// DIRECT periodic Poisson solver (spectral FFT) DISTRIBUTED, models EllipticSolver.
///
/// Role: distributed variant of PoissonFFTSolver. The periodic domain is split into SLABS (slabs,
/// 1 box per rank -- the FFT solver's native layout); PoissonFFT does the parallel transpose
/// (MPI_Alltoall) internally. It is a STANDALONE EllipticSolver, usable as
/// Coupler<Model, DistributedFFTSolver>, instead of locking the distributed FFT into SpectralCoupler.
/// Usage: replaces GeometricMG/PoissonFFTSolver when the periodic Poisson must run on n_ranks() > 1.
///
/// Preconditions:
/// - geom.domain.ny() % n_ranks() == 0: Ny must be divisible by the number of ranks (split into
///   equal slabs; enforced by an assert in the constructor);
/// - geom.domain.nx() and geom.domain.ny() powers of 2: otherwise PoissonFFT falls back on the
///   direct O(n^2) DFT per slab (correct but quadratic), and the slab transpose still requires
///   divisibility by np;
/// - periodic BCs on both axes (the passed BCRec is ignored: periodic by construction).
///
/// Invariants / constraints:
/// - solve() writes phi with zero mean (mode k=0 set to zero), one direct pass (no tolerance);
/// - residual() is COLLECTIVE: it first fills the inter-slab halos (fill_boundary MPI) then
///   reduces the residual norm over all ranks (all_reduce_max);
/// - in serial (n_ranks() == 1) a single slab covers the domain -> identical to PoissonFFTSolver.
class DistributedFFTSolver {
 public:
  DistributedFFTSolver(const Geometry& geom, const BCRec& = BCRec{},
                       std::function<bool(Real, Real)> = {})
      : geom_(geom),
        Nx_(geom.domain.nx()),
        nyl_(geom.domain.ny() / n_ranks()),
        fft_(geom.domain.nx(), geom.domain.ny(), geom.xhi - geom.xlo,
             geom.yhi - geom.ylo) {
    const int np = n_ranks();
    // HARD guard (active in Release): nyl_ = Ny / np is already computed in the init list.
    // If Ny is not divisible by np, the slabs would be mis-sized and solve() would read out
    // of bounds. The old assert vanished under NDEBUG. Aligned with PoissonFFTSolver (throw).
    if (geom.domain.ny() % np != 0)
      throw std::runtime_error(
          "DistributedFFTSolver: Ny must be divisible by n_ranks()");
    std::vector<Box2D> slabs;
    for (int r = 0; r < np; ++r)
      slabs.push_back(Box2D{{0, r * nyl_}, {Nx_ - 1, (r + 1) * nyl_ - 1}});
    BoxArray ba(std::move(slabs));
    dm_ = DistributionMapping(np, np);  // slab r -> rank r
    phi_ = MultiFab(ba, dm_, 1, 1);
    rhs_ = MultiFab(ba, dm_, 1, 0);
    res_ = MultiFab(ba, dm_, 1, 0);
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const Geometry& geom() const { return geom_; }

  // Solve lap(phi) = rhs in place: local slab -> flat array -> PoissonFFT (internal MPI transpose)
  // -> rewrite the local slab. Mode k=0 set to zero (phi with zero mean).
  void solve() {
    const ConstArray4 f = rhs_.fab(0).const_array();
    const Box2D v = rhs_.box(0);  // local slab [0..Nx-1] x [y0..y0+nyl-1]
    std::vector<double> rho(static_cast<std::size_t>(nyl_) * Nx_), phil;
    for (int jl = 0; jl < nyl_; ++jl)
      for (int i = 0; i < Nx_; ++i)
        rho[static_cast<std::size_t>(jl) * Nx_ + i] = f(v.lo[0] + i, v.lo[1] + jl);
    fft_.solve(rho, phil);
    Array4 p = phi_.fab(0).array();
    for (int jl = 0; jl < nyl_; ++jl)
      for (int i = 0; i < Nx_; ++i)
        p(v.lo[0] + i, v.lo[1] + jl) = phil[static_cast<std::size_t>(jl) * Nx_ + i];
  }

  // Discrete residual ||lap(phi) - rhs|| reduced over all ranks (~round-off: direct solve exact).
  Real residual() {
    BCRec bc;  // periodic
    fill_boundary(phi_, geom_.domain, Periodicity{true, true});  // inter-slab halos (MPI)
    poisson_residual(phi_, rhs_, geom_, bc, res_);
    return all_reduce_max(norm_inf(res_));
  }

 private:
  Geometry geom_;
  int Nx_, nyl_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_, res_;
  PoissonFFT fft_;
};

static_assert(EllipticSolver<DistributedFFTSolver>,
              "DistributedFFTSolver must model EllipticSolver");

}  // namespace adc

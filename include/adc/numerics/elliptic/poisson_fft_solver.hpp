#pragma once

/// @file
/// @brief DIRECT EllipticSolver backends via spectral FFT (periodic BCs), MultiFab wrappers around
///        PoissonFFT: PoissonFFTSolver (single-rank, single box), DistributedFFTSolver (distributed,
///        slabs) and RemappedFFTSolver (System single-box layout outward, slabs inside).
///
/// Layer: `include/adc/numerics/elliptic`.
/// Role: solve the SAME discrete 5-point Laplacian as GeometricMG (same eigenvalues) but in ONE
/// transform instead of iterating -- far cheaper when the elliptic part dominates the run (cf. PERFORMANCE.md).
/// Same constructor signature and interface as GeometricMG, so they are interchangeable as the Coupler's
/// Elliptic parameter. All three model the EllipticSolver concept (static_assert at the bottom).
/// Contract: PoissonFFTSolver is single-rank/single box (the default case of the Coupler and the facades);
/// DistributedFFTSolver is the distributed counterpart (1 slab per rank, internal MPI transpose of PoissonFFT);
/// RemappedFFTSolver exposes the System's single round-robin box (rhs()/phi() on ba/dm = same layout as
/// the aux) and hides a box<->slab scatter/gather around PoissonFFT inside solve(), so the System field
/// solver stays untouched under MPI (ADC-287).
///
/// Invariants:
/// - PoissonFFTSolver throws (HARD guard, active in Release) if n_ranks() > 1 or ba.size() != 1:
///   otherwise a rank with no local box would dereference fab(0) -> SIGSEGV;
/// - solve() writes phi with zero mean (mode k=0 set to zero) and fills phi's periodic ghosts
///   (fill_boundary) so the aux derivation (centered grad phi) reads up-to-date ghosts;
/// - residual() of DistributedFFTSolver and RemappedFFTSolver is COLLECTIVE (all_reduce_max) and
///   first fills the inter-box/-slab halos (fill_boundary).

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
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

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

/// DIRECT periodic Poisson solver (spectral FFT) under MPI, presenting the SYSTEM LAYOUT, models
/// EllipticSolver (ADC-287).
///
/// Role: let System.set_poisson(..., "fft"|"fft_spectral") run on n_ranks() > 1 WITHOUT touching the
/// system field solver. Unlike DistributedFFTSolver (slabs, DistributionMapping(np, np)), this solver
/// allocates rhs()/phi() on the System's own BoxArray/DistributionMapping (the single round-robin box,
/// DistributionMapping(1, np) -> box on one owner rank). The whole box<->slab transpose is HIDDEN inside
/// solve(): the System path (RHS assembly via add_poisson_rhs, the phi->aux derivation loop aligned on
/// fab(li), fill_boundary, the device_fence ordering) keeps reading rhs()/phi() on the SAME layout as the
/// aux, so system_field_solver.hpp needs no change beyond selecting this type.
///
/// Remap mechanics (the box lives ENTIRELY on the owner rank, so this is a scatter/gather, NOT a
/// symmetric alltoall): solve() (1) the owner packs its full Nx*Ny rhs fab into per-slab contiguous
/// buffers (slab r = global rows [r*nyl, (r+1)*nyl), full x); (2) MPI_Scatter -> each rank gets nyl*Nx
/// row-major rho_local; (3) every rank calls the proven PoissonFFT::solve (its internal MPI_Alltoall does
/// the y<->x transpose); (4) MPI_Gather phi_local -> owner; (5) owner unpacks into phi's valid cells;
/// (6) owner fill_boundary fills the periodic ghosts (a pure intra-box wrap since only the owner has a
/// fab -> no inter-rank messages, deadlock-free) so the centered grad phi reads correct halos.
///
/// Preconditions / constraints:
/// - the System box: ba.size() == 1 (the round-robin single box); rejected otherwise (collective throw);
/// - geom.domain.ny() % n_ranks() == 0 (equal slabs for the internal transpose); collective throw,
///   no deadlock since ensure_elliptic is called collectively and every rank constructs the object;
/// - periodic BCs, constant coefficient (the BCRec and wall predicate are ignored; the System guards
///   wall / variable-eps / anisotropic-eps / kappa BEFORE selecting this solver);
/// - Nx, Ny powers of two for the fast radix-2 path; otherwise PoissonFFT's correct O(n^2) DFT fallback.
///
/// Conservation / gauge: the scatter/gather is a pure permutation of doubles (no FP arithmetic, exact),
/// so the charge integral is preserved bit-exactly through the remap; PoissonFFT zeroes the k=0 mode, so
/// phi has zero mean. The only FP difference vs the single-rank PoissonFFTSolver is PoissonFFT's internal
/// alltoall reassociation, already characterised at machine zero by test_mpi_fft_distributed (np=1/2/4).
///
/// NOTE: STRUCTURAL change pending the ADC-273 design vote (new EllipticSolver type in the System ell_
/// variant, new MPI collective pattern). Behind the existing "fft"/"fft_spectral" tokens; geometric_mg
/// stays the MPI default and the only option for walls / variable-eps / kappa.
class RemappedFFTSolver {
 public:
  /// @p spectral: Laplacian symbol forwarded to PoissonFFT (false = discrete 5-point stencil,
  /// bit-identical default; true = continuous symbol -(kx^2+ky^2)). Same constructor shape as
  /// PoissonFFTSolver (BCRec and wall predicate accepted but ignored: periodic constant coefficient).
  RemappedFFTSolver(const Geometry& geom, const BoxArray& ba,
                    const BCRec& = BCRec{}, std::function<bool(Real, Real)> = {},
                    bool spectral = false)
      : geom_(geom),
        dm_(ba.size(), n_ranks()),  // SAME layout System uses: box i -> rank i % np
        phi_(ba, dm_, 1, 1),
        rhs_(ba, dm_, 1, 0),
        res_(ba, dm_, 1, 0),
        Nx_(geom.domain.nx()),
        Ny_(geom.domain.ny()),
        nyl_(geom.domain.ny() / n_ranks()),
        owner_rank_(ba.size() > 0 ? dm_[0] : 0),
        fft_(geom.domain.nx(), geom.domain.ny(), geom.xhi - geom.xlo,
             geom.yhi - geom.ylo, spectral) {
    // CONTRACT (ADC-273 vote on ADC-287): this solver is valid ONLY for System's round-robin
    // single-box Cartesian layout, where box 0 is owned by rank dm_[0] (= owner_rank_) and every other
    // rank holds an empty fab. It presents that layout outward (rhs()/phi() on ba/dm) and hides the
    // box<->slab transpose inside solve(). For a genuinely slab-distributed domain use
    // DistributedFFTSolver; for wall/non-constant-coefficient cases use geometric_mg.
    // COLLECTIVE hard guards (throw on ALL ranks -> no deadlock; ensure_elliptic is collective and
    // every rank constructs this object). Ny must split into np equal slabs for the internal transpose.
    if (ba.size() != 1) {
      throw std::runtime_error(
          "RemappedFFTSolver: System single-box layout required (ba.size()==1); for a slab-distributed "
          "domain use DistributedFFTSolver or geometric_mg");
    }
    if (geom.domain.ny() % n_ranks() != 0) {
      throw std::runtime_error(
          "RemappedFFTSolver: Ny must be divisible by n_ranks() for the slab FFT");
    }
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  const Geometry& geom() const { return geom_; }

  // Solve lap(phi) = rhs in place. THREE MPI collectives, in order: (1) MPI_Scatter the owner's box-major
  // rhs into per-rank nyl_ x Nx_ slabs; (2) PoissonFFT::solve runs its internal MPI_Alltoall (y<->x
  // transpose) + FFT on each slab; (3) MPI_Gather the phi slabs back onto the owner. Then the owner fills
  // periodic ghosts on the System layout (intra-box wrap, no messages -> deadlock-free). The caller's
  // device_fence() after ell_solve() (system_field_solver.hpp) covers the GPU read of the centered grad
  // phi, identical to the PoissonFFTSolver path. Mode k=0 set to zero (phi with zero mean).
  void solve() {
    // box-major rhs lives ENTIRELY on owner_rank_; each rank ends up with one nyl_ x Nx_ slab.
    std::vector<double> rho_local(static_cast<std::size_t>(nyl_) * Nx_), phi_local;
#ifdef ADC_HAS_MPI
    std::vector<double> sendbuf;  // only filled on the owner
    if (my_rank() == owner_rank_) {
      const ConstArray4 f = rhs_.fab(0).const_array();
      const Box2D v = rhs_.box(0);  // valid box = whole domain [0..Nx-1] x [0..Ny-1]
      sendbuf.resize(static_cast<std::size_t>(Nx_) * Ny_);
      // pack per destination rank r: rows [r*nyl_, (r+1)*nyl_) contiguous, row-major full x.
      for (int r = 0; r < n_ranks(); ++r)
        for (int jl = 0; jl < nyl_; ++jl)
          for (int i = 0; i < Nx_; ++i)
            sendbuf[(static_cast<std::size_t>(r) * nyl_ + jl) * Nx_ + i] =
                f(v.lo[0] + i, v.lo[1] + r * nyl_ + jl);
    }
    const int cnt = nyl_ * Nx_;
    MPI_Scatter(my_rank() == owner_rank_ ? sendbuf.data() : nullptr, cnt, MPI_DOUBLE,
                rho_local.data(), cnt, MPI_DOUBLE, owner_rank_, MPI_COMM_WORLD);
#else
    {
      const ConstArray4 f = rhs_.fab(0).const_array();
      const Box2D v = rhs_.box(0);
      for (int j = 0; j < Ny_; ++j)
        for (int i = 0; i < Nx_; ++i)
          rho_local[static_cast<std::size_t>(j) * Nx_ + i] = f(v.lo[0] + i, v.lo[1] + j);
    }
#endif
    fft_.solve(rho_local, phi_local);  // PROVEN slab FFT (internal MPI_Alltoall), zero-mean phi
#ifdef ADC_HAS_MPI
    std::vector<double> recvbuf;
    if (my_rank() == owner_rank_) recvbuf.resize(static_cast<std::size_t>(Nx_) * Ny_);
    MPI_Gather(phi_local.data(), cnt, MPI_DOUBLE,
               my_rank() == owner_rank_ ? recvbuf.data() : nullptr, cnt, MPI_DOUBLE,
               owner_rank_, MPI_COMM_WORLD);
    if (my_rank() == owner_rank_) {
      Array4 p = phi_.fab(0).array();
      const Box2D v = phi_.box(0);
      for (int r = 0; r < n_ranks(); ++r)
        for (int jl = 0; jl < nyl_; ++jl)
          for (int i = 0; i < Nx_; ++i)
            p(v.lo[0] + i, v.lo[1] + r * nyl_ + jl) =
                recvbuf[(static_cast<std::size_t>(r) * nyl_ + jl) * Nx_ + i];
    }
#else
    {
      Array4 p = phi_.fab(0).array();
      const Box2D v = phi_.box(0);
      for (int j = 0; j < Ny_; ++j)
        for (int i = 0; i < Nx_; ++i)
          p(v.lo[0] + i, v.lo[1] + j) = phi_local[static_cast<std::size_t>(j) * Nx_ + i];
    }
#endif
    // Periodic ghosts on the System layout: only the owner has a fab -> pure intra-box wrap (self-copy),
    // no inter-rank messages, deadlock-free. SAME role as in PoissonFFTSolver::solve() (the centered grad
    // phi of the aux derivation reads the i+-1 / j+-1 ghosts).
    fill_boundary(phi_, geom_.domain, Periodicity{true, true});
    // PR #254 managed-buffer / device-ordering discipline: owner-only device_fence so the host gather +
    // ghost wrap are settled before phi() is read on the device. The caller's device_fence() after
    // ell_solve() (system_field_solver.hpp) already brackets the grad-phi read, so this is belt-and-
    // suspenders, but it self-documents the ordering at the solver seam. Gated on the owner because only
    // it holds a fab (no-op on the empty ranks); no-op on the CPU/Serial backend.
    if (my_rank() == owner_rank_) device_fence();
  }

  // Discrete residual ||lap(phi) - rhs|| reduced over all ranks (~round-off: direct solve exact). The
  // box is owner-only, so poisson_residual runs on the owner's fab and is 0 elsewhere; the all_reduce_max
  // is COLLECTIVE (every rank participates).
  Real residual() {
    BCRec bc;  // periodic
    poisson_residual(phi_, rhs_, geom_, bc, res_);
    return all_reduce_max(norm_inf(res_));
  }

 private:
  Geometry geom_;
  DistributionMapping dm_;
  MultiFab phi_, rhs_, res_;
  int Nx_, Ny_, nyl_, owner_rank_;
  PoissonFFT fft_;
};

static_assert(EllipticSolver<RemappedFFTSolver>,
              "RemappedFFTSolver must model EllipticSolver");

}  // namespace adc

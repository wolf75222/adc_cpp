/// @file
/// @brief SpectralCoupler: DISTRIBUTED periodic E x B coupled stepper (banded FFT). DEPRECATED.
///
/// DEPRECATED: no #include in the core, the tests or the Python bindings. The role is taken over
/// by Coupler<Model, DistributedFFTSolver> (the distributed FFT solver became a standalone
/// EllipticSolver, cf. poisson_fft_solver.hpp); to remove after migration. Encapsulates the loop
/// hand-rewritten in each example: rho = elliptic_rhs(U) -> distributed spectral Poisson (FFT, bands) ->
/// aux = (phi, grad phi) with halos -> advance (Rusanov, E x B drift). BANDED decomposition (1
/// box per rank); Nx, Ny powers of 2 divisible by n_ranks(). Generic over the model.

#pragma once

#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>  // DistributedFFTSolver (wraps PoissonFFT)
#include <pops/numerics/time/reference/amr_reflux.hpp>              // advance_fab_1c, xface_box, yface_box
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>  // for_each_cell_reduce_sum, POPS_HD, device_fence
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace pops {

/// Periodic E x B coupled stepper, BANDED-distributed (FFT). DEPRECATED (cf. @file). @tparam Model:
/// E x B drift model (alpha, n_i0, B0, E x B flux). PRECONDITION: Nx, Ny powers of 2,
/// divisible by n_ranks().
template <class Model>
class SpectralCoupler {
 public:
  /// Builds the coupler on a domain [0, Lx] x [0, Ly] discretized Nx x Ny, split into bands
  /// (1 box per rank). Allocates the state U_ (1 component) and the aux Uaux_ (3 components), 1 ghost.
  SpectralCoupler(const Model& model, int Nx, int Ny, double Lx, double Ly)
      : model_(model),
        Nx_(Nx),
        Ny_(Ny),
        dx_(Lx / Nx),
        dy_(Ly / Ny),
        dom_(Box2D::from_extents(Nx, Ny)),
        np_(n_ranks()),
        me_(my_rank()),
        nyl_(Ny / np_),
        y0_(me_ * nyl_),
        fft_(Geometry{Box2D::from_extents(Nx, Ny), 0.0, Lx, 0.0, Ly}) {
    std::vector<Box2D> slabs;
    for (int r = 0; r < np_; ++r)
      slabs.push_back(Box2D{{0, r * nyl_}, {Nx - 1, (r + 1) * nyl_ - 1}});
    ba_ = BoxArray(std::move(slabs));
    dm_ = DistributionMapping(np_, np_);  // box r -> rank r
    U_ = MultiFab(ba_, dm_, 1, 1);
    Uaux_ = MultiFab(ba_, dm_, 3, 1);
  }

  // --- access to the state (the local band of this rank) ---
  MultiFab& state() { return U_; }
  const MultiFab& state() const { return U_; }
  Fab2D& local() { return U_.fab(0); }  // the fab of the local band
  const Box2D& domain() const { return dom_; }
  int y_begin() const { return y0_; }
  int ny_local() const { return nyl_; }
  int nx() const { return Nx_; }
  double dx() const { return dx_; }
  double dy() const { return dy_; }

  /// Distributed spectral Poisson + aux = (phi, grad phi) + halos, WITHOUT advancing in time (to fix dt
  /// or diagnose). Uaux_ up to date on return.
  void solve_aux() {
    // rho into the rhs of the distributed FFT solver (same banded split as U_, so
    // bit-identical to the old inline path that called PoissonFFT directly).
    const ConstArray4 u = U_.fab(0).const_array();
    Array4 r = fft_.rhs().fab(0).array();
    // f = model.elliptic_rhs(U): we call the PhysicalModel (like the normal Coupler)
    // instead of hard-coding the background coupling alpha*(u - n0) in the core.
    // For this coupling it is exactly the same expression -> bit-identical.
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i)
        r(i, j) = model_.elliptic_rhs(typename Model::State{u(i, j)});
    fft_.solve();
    fill_boundary(fft_.phi(), dom_, Periodicity{true, true});
    const ConstArray4 pc = fft_.phi().fab(0).const_array();
    Array4 a = Uaux_.fab(0).array();
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i) {
        a(i, j, 0) = pc(i, j);
        a(i, j, 1) = (pc(i + 1, j) - pc(i - 1, j)) / (2 * dx_);
        a(i, j, 2) = (pc(i, j + 1) - pc(i, j - 1)) / (2 * dy_);
      }
    fill_boundary(Uaux_, dom_, Periodicity{true, true});
  }

  /// One full coupled step: solve_aux then advance (Rusanov, E x B drift) of the local band over dt.
  void step(double dt) {
    solve_aux();
    fill_boundary(U_, dom_, Periodicity{true, true});
    Fab2D fx(xface_box(U_.box(0)), 1, 0), fy(yface_box(U_.box(0)), 1, 0);
    advance_fab_1c(model_, U_.fab(0), Uaux_.fab(0), dx_, dy_, dt, fx, fy);
  }

  // max wave speed (all-reduce), for the CFL. GENERALIZED (milestone 4.3): via
  // model.max_wave_speed instead of the /B0 drift hard-coded -> the spectral coupler
  // is no longer tied to a particular drift flux. NB: for the E x B drift it is
  // max(|gx|,|gy|)/B0 (per direction) instead of hypot(gx,gy)/B0; the CFL dt thus changes slightly (to re-validate
  // on the physics side, this is NOT bit-identical to the old diagnostic).
  double max_drift_speed() const {
    device_fence();  // GPU: barrier before host read after advance_fab_1c (device)
    const ConstArray4 u = U_.fab(0).const_array();
    const ConstArray4 a = Uaux_.fab(0).const_array();
    const Model m = model_;
    double v = 0;
    for (int j = y0_; j < y0_ + nyl_; ++j)
      for (int i = 0; i < Nx_; ++i) {
        typename Model::State s{};
        s[0] = u(i, j, 0);
        const typename Model::Aux ax{a(i, j, 0), a(i, j, 1), a(i, j, 2)};
        const Real wx = m.max_wave_speed(s, ax, 0);
        const Real wy = m.max_wave_speed(s, ax, 1);
        v = std::max(v, static_cast<double>(wx > wy ? wx : wy));
      }
    return std::max(all_reduce_max(v), 1e-12);
  }

  // total mass (all-reduce).
  double mass() const {
    // reducer seam (local band = 1 fab); Kokkos::Sum reassociates the sum per tile
    // (deterministic/idempotent but not bit-identical to a lexicographic sum);
    // parallel_reduce absorbs the barrier, no more device_fence at the head.
    const ConstArray4 u = U_.fab(0).const_array();
    const Real s =
        for_each_cell_reduce_sum(U_.box(0), [u] POPS_HD(int i, int j) { return u(i, j); });
    return all_reduce_sum(static_cast<double>(s)) * dx_ * dy_;
  }

 private:
  Model model_;
  int Nx_, Ny_;
  double dx_, dy_;
  Box2D dom_;
  int np_, me_, nyl_, y0_;
  BoxArray ba_;
  DistributionMapping dm_;
  MultiFab U_, Uaux_;
  DistributedFFTSolver fft_;
};

}  // namespace pops

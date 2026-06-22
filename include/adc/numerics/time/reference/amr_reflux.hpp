#pragma once

#include <adc/core/state/state.hpp>
#include <adc/amr/hierarchy/refinement_ratio.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/numerics/spatial_operator.hpp>

#include <vector>

/// @file
/// @brief Reference conservation-critical brick (Fab2D, 1 component): 2-level reflux a la AMReX.
///        Provides the 1-component Rusanov fluxes (compute_fluxes_1c), the explicit Euler advance
///        (advance_fab_1c), the periodic ghosts (fill_periodic_fab) and coarse-fine space-time
///        ghosts (fill_fine_ghosts_t), average_down_fab, and the amr_step_2level step.
///
/// Layer: `include/adc/numerics/time`.
/// Role: minimal and testable version of the AMR reflux. The numerical flux at the coarse-fine
///        interface is inconsistent between the coarse and fine grids; the reflux corrects the
///        adjacent coarse cells (FluxRegister) to restore exact conservation. This is the mono-box
///        ground-truth of the Fab2D -> MF -> MP parity chain.
///
/// Invariants:
/// - minimal case: 1 component, explicit Euler, ratio 2, Berger-Oliger subcycling (r=2 substeps
///   of dt/2, fine ghosts interpolated in time), one rectangular fine box strictly interior to the
///   periodic coarse domain;
/// - the composite Poisson coupling (FAC) is handled elsewhere (elliptic/composite_fac_poisson);
/// - device kernels = NAMED functors (RusanovFaceXKernel, AdvanceFab1cKernel, ...) and not
///   ADC_HD lambdas: first instantiation from an external loader TU would trip nvcc; body strictly
///   identical to the previous lambdas -> bit-identical CPU and device.

namespace adc {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

// xface_box / yface_box: provided by numerics/spatial_operator.hpp (included above),
// same face-box conventions. We do not redefine them here.

namespace detail {

// NAMED FUNCTORS (and not ADC_HD lambdas) for the AMR reflux device kernels. Same reasons as the
// rest of the mesh/time path (AmrSspRhsKernel, fill_boundary recipe): these kernels are first-
// instantiated from an external loader TU, where an extended lambda trips the device kernel emission
// under nvcc. Body strictly identical to the previous lambdas -> bit-identical CPU and device.

/// Rusanov flux at the left face (x axis) of cell (i,j).
template <class Model>
struct RusanovFaceXKernel {
  Model m;
  ConstArray4 u, ax;
  Array4 F;
  ADC_HD void operator()(int i, int j) const {
    typename Model::State UL{}, UR{};
    UL[0] = u(i - 1, j);
    UR[0] = u(i, j);
    F(i, j) = rusanov_flux(m, UL, load_aux<aux_comps<Model>()>(ax, i - 1, j), UR,
                           load_aux<aux_comps<Model>()>(ax, i, j), 0)[0];
  }
};

/// Rusanov flux at the bottom face (y axis) of cell (i,j).
template <class Model>
struct RusanovFaceYKernel {
  Model m;
  ConstArray4 u, ax;
  Array4 F;
  ADC_HD void operator()(int i, int j) const {
    typename Model::State UL{}, UR{};
    UL[0] = u(i, j - 1);
    UR[0] = u(i, j);
    F(i, j) = rusanov_flux(m, UL, load_aux<aux_comps<Model>()>(ax, i, j - 1), UR,
                           load_aux<aux_comps<Model>()>(ax, i, j), 1)[0];
  }
};

/// Explicit Euler, 1 component: U -= dt div(F) on cell (i,j).
struct AdvanceFab1cKernel {
  Array4 uu;
  ConstArray4 FX, FY;
  double dx, dy, dt;
  ADC_HD void operator()(int i, int j) const {
    uu(i, j) -= dt * ((FX(i + 1, j) - FX(i, j)) / dx + (FY(i, j + 1) - FY(i, j)) / dy);
  }
};

}  // namespace detail

// First-order Rusanov flux, 1 component, aux variable in space (Fab2D with
// 3 components [phi, gx, gy], ghosts filled), on a Fab2D.
// fx(i,j) = flux at the left face of cell i; fy(i,j) = bottom face of j.
template <class Model>
void compute_fluxes_1c(const Model& m, const Fab2D& U, const Fab2D& aux, Fab2D& fx, Fab2D& fy) {
  const ConstArray4 u = U.const_array();
  const ConstArray4 ax = aux.const_array();
  {
    Array4 F = fx.array();
    for_each_cell(fx.box(), detail::RusanovFaceXKernel<Model>{m, u, ax, F});
  }
  {
    Array4 F = fy.array();
    for_each_cell(fy.box(), detail::RusanovFaceYKernel<Model>{m, u, ax, F});
  }
}

// Explicit Euler: U -= dt div(F). The ghosts of U must be filled.
template <class Model>
void advance_fab_1c(const Model& m, Fab2D& U, const Fab2D& aux, double dx, double dy, double dt,
                    Fab2D& fx, Fab2D& fy) {
  compute_fluxes_1c(m, U, aux, fx, fy);
  Array4 uu = U.array();
  const ConstArray4 FX = fx.const_array();
  const ConstArray4 FY = fy.const_array();
  for_each_cell(U.box(), detail::AdvanceFab1cKernel{uu, FX, FY, dx, dy, dt});
}

// Periodic ghosts for a single Fab2D covering the domain.
inline void fill_periodic_fab(Fab2D& U, const Box2D& dom) {
  const int ng = U.n_ghost();
  const int nx = dom.nx(), ny = dom.ny();
  Array4 a = U.array();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int g = 1; g <= ng; ++g) {
      a(dom.lo[0] - g, j) = a(dom.hi[0] - g + 1, j);
      a(dom.hi[0] + g, j) = a(dom.lo[0] + g - 1, j);
    }
  for (int i = dom.lo[0] - ng; i <= dom.hi[0] + ng; ++i)
    for (int g = 1; g <= ng; ++g) {
      a(i, dom.lo[1] - g) = a(i, dom.hi[1] - g + 1);
      a(i, dom.hi[1] + g) = a(i, dom.lo[1] + g - 1);
    }
}

// Fine ghosts by injection from the coarse (ratio 2), interpolated in time
// between the old coarse state (frac=0) and the new one (frac=1): space-time
// FillPatch for the Berger-Oliger subcycling.
inline void fill_fine_ghosts_t(Fab2D& Uf, const Fab2D& Uco, const Fab2D& Ucn, double frac) {
  const ConstArray4 co = Uco.const_array();
  const ConstArray4 cn = Ucn.const_array();
  Array4 f = Uf.array();
  const Box2D g = Uf.grown_box();
  const Box2D v = Uf.box();
  auto coarsen = [](int x) { return (x >= 0) ? x / 2 : -((-x + 1) / 2); };
  for (int j = g.lo[1]; j <= g.hi[1]; ++j)
    for (int i = g.lo[0]; i <= g.hi[0]; ++i)
      if (!v.contains(i, j)) {
        const int ci = coarsen(i), cj = coarsen(j);
        f(i, j) = (1 - frac) * co(ci, cj) + frac * cn(ci, cj);
      }
}

// Fine -> coarse average over the covered region (ratio 2).
inline void average_down_fab(const Fab2D& Uf, Fab2D& Uc, int CI0, int CI1, int CJ0, int CJ1) {
  const ConstArray4 f = Uf.const_array();
  Array4 c = Uc.array();
  for (int J = CJ0; J <= CJ1; ++J)
    for (int I = CI0; I <= CI1; ++I)
      c(I, J) = 0.25 * (f(2 * I, 2 * J) + f(2 * I + 1, 2 * J) + f(2 * I, 2 * J + 1) +
                        f(2 * I + 1, 2 * J + 1));
}

// One conservative 2-level step with Berger-Oliger subcycling (the fine does
// r=2 substeps of dt/2) and reflux. Fine region = coarse cells
// [CI0..CI1] x [CJ0..CJ1] (strictly interior), refined into
// [2CI0..2CI1+1] x [...]. dom = periodic coarse domain.
//
// Flux register: we accumulate the coarse flux (x dt) and the sum of the fine
// fluxes (x dt/2 per substep, spatially averaged) at the 4 faces of the fine
// region, then we correct the adjacent coarse cells by their difference.
template <class Model>
void amr_step_2level(const Model& m, Fab2D& Uc, const Box2D& dom, double dxc, double dyc, Fab2D& Uf,
                     int CI0, int CI1, int CJ0, int CJ1, const Fab2D& auxc, const Fab2D& auxf,
                     double dt) {
  const int r = kAmrRefRatio;
  const double dxf = dxc / kAmrRefRatio, dyf = dyc / kAmrRefRatio, dtf = dt / r;
  const int nJ = CJ1 - CJ0 + 1, nI = CI1 - CI0 + 1;

  const Fab2D Uc_old = Uc;  // coarse state at time t (for temporal interp)

  // --- coarse fluxes (before update) at the 4 faces of the fine region ---
  fill_periodic_fab(Uc, dom);
  Fab2D fxc(xface_box(Uc.box()), 1, 0), fyc(yface_box(Uc.box()), 1, 0);
  compute_fluxes_1c(m, Uc, auxc, fxc, fyc);
  const ConstArray4 FXc = fxc.const_array();
  const ConstArray4 FYc = fyc.const_array();
  std::vector<double> cL(nJ), cR(nJ), cB(nI), cT(nI);
  for (int J = CJ0; J <= CJ1; ++J) {
    cL[J - CJ0] = FXc(CI0, J);
    cR[J - CJ0] = FXc(CI1 + 1, J);
  }
  for (int I = CI0; I <= CI1; ++I) {
    cB[I - CI0] = FYc(I, CJ0);
    cT[I - CI0] = FYc(I, CJ1 + 1);
  }

  advance_fab_1c(m, Uc, auxc, dxc, dyc, dt, fxc, fyc);  // Uc becomes the "t+dt" state

  // --- fine subcycling: r substeps, accumulation of fine fluxes (x dtf) ---
  std::vector<double> fL(nJ, 0), fR(nJ, 0), fB(nI, 0), fT(nI, 0);
  Fab2D fxf(xface_box(Uf.box()), 1, 0), fyf(yface_box(Uf.box()), 1, 0);
  for (int s = 0; s < r; ++s) {
    fill_fine_ghosts_t(Uf, Uc_old, Uc, double(s) / r);  // time-interpolated BC
    compute_fluxes_1c(m, Uf, auxf, fxf, fyf);
    const ConstArray4 FXf = fxf.const_array();
    const ConstArray4 FYf = fyf.const_array();
    for (int J = CJ0; J <= CJ1; ++J) {
      fL[J - CJ0] += 0.5 * (FXf(2 * CI0, 2 * J) + FXf(2 * CI0, 2 * J + 1)) * dtf;
      fR[J - CJ0] += 0.5 * (FXf(2 * CI1 + 2, 2 * J) + FXf(2 * CI1 + 2, 2 * J + 1)) * dtf;
    }
    for (int I = CI0; I <= CI1; ++I) {
      fB[I - CI0] += 0.5 * (FYf(2 * I, 2 * CJ0) + FYf(2 * I + 1, 2 * CJ0)) * dtf;
      fT[I - CI0] += 0.5 * (FYf(2 * I, 2 * CJ1 + 2) + FYf(2 * I + 1, 2 * CJ1 + 2)) * dtf;
    }
    advance_fab_1c(m, Uf, auxf, dxf, dyf, dtf, fxf, fyf);
  }

  average_down_fab(Uf, Uc, CI0, CI1, CJ0, CJ1);  // sync of the covered cells

  // --- reflux: coarse flux (x dt) replaced by sum of fine fluxes (x dtf) ---
  Array4 c = Uc.array();
  for (int J = CJ0; J <= CJ1; ++J) {
    c(CI0 - 1, J) -= (fL[J - CJ0] - cL[J - CJ0] * dt) / dxc;
    c(CI1 + 1, J) += (fR[J - CJ0] - cR[J - CJ0] * dt) / dxc;
  }
  for (int I = CI0; I <= CI1; ++I) {
    c(I, CJ0 - 1) -= (fB[I - CI0] - cB[I - CI0] * dt) / dyc;
    c(I, CJ1 + 1) += (fT[I - CI0] - cT[I - CI0] * dt) / dyc;
  }
}

}  // namespace adc

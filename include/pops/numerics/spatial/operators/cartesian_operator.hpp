/// @file
/// @brief Cartesian residual R = -div Fhat + S over the cells of a level (method of lines).
///
/// CONTRACT: the "PDE -> ODE system" arrow. The time integrator (time/) only knows R; it is
/// unaware of the geometry and the reconstruction scheme.
///   - assemble_rhs<Limiter,NumericalFlux>: main entry point; residual + optional Fickian term.
///   - assemble_rhs_hll_cached<Limiter>: OPT-IN HLL path with the per-cell wave-speed cache
///     (wave_speed.hpp); BIT-IDENTICAL to assemble_rhs<NoSlope, HLLFlux> with NoSlope.
///
/// Reconstruction (reconstruct_pp) and the structural ghost guard come from face_flux.hpp; the
/// positivity role from positivity.hpp; the cache fill from wave_speed.hpp.

#pragma once

#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial/primitives/face_flux.hpp>     // reconstruct_pp, require_reconstruction_ghosts
#include <pops/numerics/spatial/primitives/positivity.hpp>    // detail::positivity_comp
#include <pops/numerics/spatial/primitives/state_access.hpp>  // load_state, load_aux, DiffusiveModel
#include <pops/numerics/spatial/primitives/wave_speed.hpp>    // fill_wave_speed_cache

namespace pops {

namespace detail {
/// AssembleRhsKernel<Limiter,NumericalFlux,Model>: device kernel of the central residual of
/// assemble_rhs.
///
/// Computes R(i,j) = S - (Fxp-Fxm)/dx - (Fyp-Fym)/dy (+ Fickian term if DiffusiveModel).
/// Named functor: key point of the AOT native parity (add_compiled_model via external TU).
/// Body bit-identical to the former lambda. POPS_HD.
//
// nvcc does not reliably emit the device kernel of a Model-template extended lambda first
// instantiated from an EXTERNAL TU through the std::function / host-lambda nesting of block_builder:
// the test passes on Serial and under compute-sanitizer but segfaults at runtime on Cuda (Heisenbug).
// A device-callable class does not have these instantiation-context restrictions. Body IDENTICAL to
// the former lambda -> residual BIT-IDENTICAL to add_block on CPU (and, targeted, on device).
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  POPS_HD void operator()(int i, int j) const {
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // x faces: reconstruction of the states on either side of each face
    const auto Lxm =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp =
        reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp =
        reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
    const auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);

    // y faces
    const auto Lym =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp =
        reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp =
        reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Fym = nflux(model, Lym, Aym, Rym, Ac, 1);
    const auto Fyp = nflux(model, Lyp, Ac, Ryp, Ayp, 1);

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;

    // Parabolic (Fickian) term: +nu Lap(U), 5-point centered differences.
    // Guarded by DiffusiveModel: no effect (nor codegen) for a non-diffusive model.
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      const Real idx2 = Real(1) / (dx * dx), idy2 = Real(1) / (dy * dy);
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) += nu * ((u(i + 1, j, c) - 2 * u(i, j, c) + u(i - 1, j, c)) * idx2 +
                            (u(i, j + 1, c) - 2 * u(i, j, c) + u(i, j - 1, c)) * idy2);
    }
  }
};
}  // namespace detail

/// assemble_rhs<Limiter,NumericalFlux>: residual R = -div Fhat + S over all boxes.
///
/// Main entry point of the Cartesian spatial operator. The limiter (reconstruction) AND the
/// numerical flux are template parameters chosen at compile time (default: NoSlope + RusanovFlux).
/// recon_prim = true enables reconstruction in primitive variables if the model exposes
/// HasPrimitiveVars. For the diffusive term, see DiffusiveModel.
/// INVARIANT: the operator does not modify U, aux -- it only writes R. No ghost fill.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs(const Model& model, const MultiFab& U, const MultiFab& aux, const Geometry& geom,
                  MultiFab& R, bool recon_prim = false, Real pos_floor = Real(0)) {
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>{
                         model, u, ax, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

namespace detail {
/// AssembleRhsHllCachedKernel: kernel of the residual R = -div Fhat + S for the HLL flux with wave
/// speeds PRE-COMPUTED per cell (scratch @c ws, 4 components). Reconstruction and numerical flux
/// IDENTICAL to AssembleRhsKernel<.., HLLFlux>; only the source of the signal speeds changes: instead
/// of hll_speeds (per-face recall of model.wave_speeds), each face bounds sL/sR by min/max of the
/// cached speeds of its two adjacent cells -- exactly the Davis estimates of the per-face path when the
/// reconstructed states equal the cell values (NoSlope). Named functor (device-clean cross-TU). POPS_HD.
template <class Limiter, class Model>
struct AssembleRhsHllCachedKernel {
  Model model;
  ConstArray4 u, ax, ws;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< Density role component (resolved by the host caller)
  POPS_HD void operator()(int i, int j) const {
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // x faces: reconstruction of the states on both sides of each face
    const auto Lxm =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp =
        reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp =
        reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    // face signal speeds = union (min of the lo, max of the hi) of the two adjacent cells: the LEFT
    // cell is always the lower-index neighbor (same operands/order as hll_speeds).
    const Real sLxm = ws(i - 1, j, 0) < ws(i, j, 0) ? ws(i - 1, j, 0) : ws(i, j, 0);
    const Real sRxm = ws(i - 1, j, 1) > ws(i, j, 1) ? ws(i - 1, j, 1) : ws(i, j, 1);
    const Real sLxp = ws(i, j, 0) < ws(i + 1, j, 0) ? ws(i, j, 0) : ws(i + 1, j, 0);
    const Real sRxp = ws(i, j, 1) > ws(i + 1, j, 1) ? ws(i, j, 1) : ws(i + 1, j, 1);
    const auto Fxm = hll_flux_with_speeds(model, Lxm, Axm, Rxm, Ac, 0, sLxm, sRxm);
    const auto Fxp = hll_flux_with_speeds(model, Lxp, Ac, Rxp, Axp, 0, sLxp, sRxp);

    // y faces (components 2 = lo_y, 3 = hi_y of the scratch)
    const auto Lym =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp =
        reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp =
        reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const Real sLym = ws(i, j - 1, 2) < ws(i, j, 2) ? ws(i, j - 1, 2) : ws(i, j, 2);
    const Real sRym = ws(i, j - 1, 3) > ws(i, j, 3) ? ws(i, j - 1, 3) : ws(i, j, 3);
    const Real sLyp = ws(i, j, 2) < ws(i, j + 1, 2) ? ws(i, j, 2) : ws(i, j + 1, 2);
    const Real sRyp = ws(i, j, 3) > ws(i, j + 1, 3) ? ws(i, j, 3) : ws(i, j + 1, 3);
    const auto Fym = hll_flux_with_speeds(model, Lym, Aym, Rym, Ac, 1, sLym, sRym);
    const auto Fyp = hll_flux_with_speeds(model, Lyp, Ac, Ryp, Ayp, 1, sLyp, sRyp);

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;

    // Parabolic (Fickian) term: identical to AssembleRhsKernel, guarded by DiffusiveModel.
    if constexpr (DiffusiveModel<Model>) {
      const Real nu = model.diffusivity();
      const Real idx2 = Real(1) / (dx * dx), idy2 = Real(1) / (dy * dy);
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) += nu * ((u(i + 1, j, c) - 2 * u(i, j, c) + u(i - 1, j, c)) * idx2 +
                            (u(i, j + 1, c) - 2 * u(i, j, c) + u(i, j - 1, c)) * idy2);
    }
  }
};
}  // namespace detail

/// assemble_rhs_hll_cached<Limiter>: residual R = -div Fhat + S at the HLL flux, wave speeds
/// PRE-COMPUTED per cell (OPT-IN). Two passes: (1) fill_wave_speed_cache fills @p cache on
/// grow(valid, 1); (2) the kernel reads the scratch and bounds each face by min/max of the two cells.
/// @p cache must have the layout of @p U, 4 components, >= 1 ghost (re-allocated by the caller).
/// With NoSlope: BIT-IDENTICAL to assemble_rhs<NoSlope, HLLFlux> (cf. the section header). The model
/// MUST expose wave_speeds (guaranteed by the HLL dispatch).
template <class Limiter = NoSlope, class Model>
void assemble_rhs_hll_cached(const Model& model, const MultiFab& U, const MultiFab& aux,
                             const Geometry& geom, MultiFab& R, MultiFab& cache,
                             bool recon_prim = false, Real pos_floor = Real(0)) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  fill_wave_speed_cache(model, U, aux, cache);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 ws = cache.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsHllCachedKernel<Limiter, Model>{
                         model, u, ax, ws, r, dx, dy, lim, recon_prim, pos_floor, pos_comp});
  }
}

}  // namespace pops

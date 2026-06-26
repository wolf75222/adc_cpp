/// @file
/// @brief Domain-mask-aware Cartesian residual (conservative active sub-domain, OPT-IN).
///
/// CONTRACT: the mask-aware variant of assemble_rhs. SEPARATE entry point: the default path
/// (System::step) stays strictly bit-identical as long as it does not call this overload.
///   - assemble_rhs_masked<Limiter,NumericalFlux>: residual restricted to a 0/1 cell-centered mask.
///
/// Convention: mask(i,j) >= 0.5 -> ACTIVE. A face is OPEN only if BOTH adjacent cells are active;
/// otherwise the normal flux is set to ZERO (FV wall), so the mass over the active sub-domain is
/// conserved to machine precision. Reconstruction and the positivity role come from face_flux.hpp /
/// positivity.hpp.

#pragma once

#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial/primitives/face_flux.hpp>     // reconstruct_pp, require_reconstruction_ghosts
#include <pops/numerics/spatial/primitives/positivity.hpp>    // detail::positivity_comp
#include <pops/numerics/spatial/primitives/state_access.hpp>  // load_state, load_aux

namespace pops {

// ============================================================================
// DOMAIN MASK (T2 effort, conservative, OPT-IN -- default path untouched)
// ============================================================================
// The mask makes the FV transport aware of an ACTIVE sub-domain (e.g. a bounded disk-shaped region).
// Convention: mask(i, j) >= 0.5 -> ACTIVE cell, otherwise INACTIVE. A face is OPEN (normal flux
// computed) if BOTH adjacent cells are active; it is CLOSED (normal flux set to ZERO) if at least
// one is inactive. Zeroing the normal flux at active/inactive faces makes the step CONSERVATIVE
// over the active sub-domain: no mass crosses the boundary, so the total mass over the active cells
// is conserved to machine precision (telescoping internal fluxes, zero boundary fluxes). This is the
// FV counterpart of the conducting wall (which only acts on the elliptic part).
//
// The residual is written ONLY on the active cells; an inactive cell keeps its residual at 0
// (the caller does not advance it). This header does NOT wire this path into System::step: it
// provides the mask-aware brick, exercised directly by the tests and, eventually, behind the
// active-sub-domain opt-in.

namespace detail {
/// Activity indicator of a cell from a 0/1 cell-centered mask (>= 0.5 -> active).
POPS_HD inline bool mask_active(const ConstArray4& mask, int i, int j) {
  return mask(i, j, 0) >= Real(0.5);
}

/// AssembleRhsMaskedKernel: variant of AssembleRhsKernel AWARE of a domain mask.
///
/// Inactive cell -> residual 0 (not advanced by the caller). Active cell -> R = -div Fhat + S,
/// BUT the normal flux of a face whose neighbor cell is INACTIVE is set to ZERO (FV wall:
/// zero normal flux at the active/inactive boundary) -> mass conservation over the active
/// sub-domain. Named functor (same device contract as AssembleRhsKernel). POPS_HD.
///
/// NB: without a diffusive term (transport-only models); a DiffusiveModel keeps
/// its UNmasked Laplacian here (separate refinement -- the conservative mask targets the hyperbolic
/// flux only).
template <class Limiter, class NumericalFlux, class Model>
struct AssembleRhsMaskedKernel {
  Model model;
  ConstArray4 u, ax, mask;
  Array4 r;
  Real dx, dy;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  POPS_HD void operator()(int i, int j) const {
    if (!mask_active(mask, i,
                     j)) {  // cell outside the active sub-domain: zero residual, not advanced
      for (int c = 0; c < Model::n_vars; ++c)
        r(i, j, c) = Real(0);
      return;
    }
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const Aux Axm = load_aux<aux_comps<Model>()>(ax, i - 1, j);
    const Aux Axp = load_aux<aux_comps<Model>()>(ax, i + 1, j);
    const Aux Aym = load_aux<aux_comps<Model>()>(ax, i, j - 1);
    const Aux Ayp = load_aux<aux_comps<Model>()>(ax, i, j + 1);

    // x faces: reconstruction on either side, numerical flux, THEN mask gate (closed face
    // -> zero normal flux) -- an inactive neighbor cell closes the face between it and (i, j).
    const auto Lxm =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxm =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lxp =
        reconstruct_pp<Model>(model, u, i, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rxp =
        reconstruct_pp<Model>(model, u, i + 1, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    auto Fxm = nflux(model, Lxm, Axm, Rxm, Ac, 0);
    auto Fxp = nflux(model, Lxp, Ac, Rxp, Axp, 0);
    if (!mask_active(mask, i - 1, j))
      Fxm = typename Model::State{};
    if (!mask_active(mask, i + 1, j))
      Fxp = typename Model::State{};

    // y faces
    const auto Lym =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rym =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto Lyp =
        reconstruct_pp<Model>(model, u, i, j, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Ryp =
        reconstruct_pp<Model>(model, u, i, j + 1, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    auto Fym = nflux(model, Lym, Aym, Rym, Ac, 1);
    auto Fyp = nflux(model, Lyp, Ac, Ryp, Ayp, 1);
    if (!mask_active(mask, i, j - 1))
      Fym = typename Model::State{};
    if (!mask_active(mask, i, j + 1))
      Fyp = typename Model::State{};

    const auto S = model.source(load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c)
      r(i, j, c) = S[c] - (Fxp[c] - Fxm[c]) / dx - (Fyp[c] - Fym[c]) / dy;
  }
};
}  // namespace detail

/// assemble_rhs_masked<Limiter,NumericalFlux>: residual R = -div Fhat + S RESTRICTED to a 0/1
/// cell-centered domain mask (OPT-IN, T2 effort). On an inactive cell R = 0 (not advanced); on an
/// active cell, the normal flux of a face whose neighbor is inactive is set to zero (FV wall).
/// Result: the mass over the active sub-domain is CONSERVED to machine precision (no flux crosses
/// the boundary) -- property validated by the active-sub-domain mass-conservation test.
///
/// @p mask must have the SAME layout as @p U (same BoxArray / DistributionMapping) and carry at
/// least 1 ghost (reading the neighbors i-1/i+1/j-1/j+1 up to the edge). This entry point is
/// SEPARATE from assemble_rhs: the default path (System::step) stays strictly bit-identical as long
/// as it does NOT call this overload.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs_masked(const Model& model, const MultiFab& U, const MultiFab& aux,
                         const MultiFab& mask, const Geometry& geom, MultiFab& R,
                         bool recon_prim = false, Real pos_floor = Real(0)) {
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 mk = mask.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::AssembleRhsMaskedKernel<Limiter, NumericalFlux, Model>{
                         model, u, ax, mk, r, dx, dy, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
}

}  // namespace pops

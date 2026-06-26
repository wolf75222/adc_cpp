#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>  // PolarGeometry
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial_operator.hpp>  // reconstruct<>, load_state/load_aux (REUSED verbatim)

#include <concepts>
#include <utility>
#include <vector>

/// @file
/// @brief Additive POLAR spatial operator: R = -div_polar F + S on an annular grid (r, theta).
///
/// "Annular polar grid" work, Phase 1 (TRANSPORT only). This is a SEPARATE assemble_rhs from the
/// Cartesian history (pops/numerics/spatial_operator.hpp): that one stays STRICTLY UNTOUCHED, so a
/// run on a Cartesian mesh is bit-identical. The polar path is PURELY ADDITIVE, opt-in (the caller
/// chooses assemble_rhs_polar with a PolarGeometry).
///
/// AXIS CONVENTION (cf. PolarGeometry): index direction 0 = RADIAL (r), index direction 1 =
/// AZIMUTHAL (theta). Domain r in [r_min, r_max] (physical BC in r), theta in [0, 2pi) (periodic).
///
/// DIVERGENCE IN POLAR COORDINATES (conservative form, finite volumes):
///   div F = (1/r) d_r(r F_r) + (1/r) d_theta(F_theta)
/// Per-cell discretization (i, j), with r_i = r_cell(i), r_{i+/-1/2} = r_face(i+1)/r_face(i),
/// dr = PolarGeometry::dr(), dtheta = PolarGeometry::dtheta():
///   -div = -(1/r_i) (r_{i+1/2} Fr_{i+1/2} - r_{i-1/2} Fr_{i-1/2}) / dr
///          -(1/r_i) (Ftheta_{j+1/2} - Ftheta_{j-1/2}) / dtheta
/// Fr, Ftheta are the PHYSICAL numerical fluxes at the faces (the model returns the physical
/// component in the local basis; cf. ExBVelocityPolar). The metric factor r at radial faces (and
/// 1/r in cell) is carried HERE. CONSERVATION: the mass Sum_ij n_ij r_i dr dtheta sees the radial
/// term telescope (the weight r_{i+1/2} of a face is shared by the two neighboring cells) and the
/// azimuthal term telescope exactly (periodic). Only the radial fluxes at the physical boundaries
/// r_min / r_max count: a wall (zero radial flux) conserves mass to machine precision.
///
/// NAMED FUNCTORS (not extended lambdas), like spatial_operator.hpp (#64/#97): robust device
/// emission if the Model-template kernel is instantiated cross-TU. The RECONSTRUCTION (weno5z,
/// minmod via reconstruct<>) and the numerical FLUX (RusanovFlux...) are REUSED verbatim from the
/// Cartesian operator: direction-generic (dir 0/1), they apply as-is to the (r, theta) axes.

namespace pops {

namespace detail {

// The source term is OPTIONAL in polar transport (Phase 1): a pure transport brick
// (ExBVelocityPolar) does not expose source(), only a composite model (CompositeModel) does. We
// detect it and fall back to 0 otherwise -> the polar operator works just as well on the standalone
// brick as on a composite model, without requiring source() (the Cartesian path, for its part,
// assumes a composite model).
/// PolarHasSource<M>: internal concept -- true if M exposes source(U, aux) -> State.
///
/// Used to route polar_source: falls back to the zero state if the pure brick (ExBVelocityPolar)
/// does not expose source(), without requiring the contract from all hyperbolic bricks.
template <class M>
concept PolarHasSource = requires(const M m, const typename M::State u, const Aux a) {
  { m.source(u, a) } -> std::convertible_to<typename M::State>;
};

/// polar_source<Model>: returns m.source(u, a) if PolarHasSource<Model>, otherwise the zero state.
///
/// if constexpr guard: zero extra codegen for bricks without a source. POPS_HD.
template <class Model>
POPS_HD inline typename Model::State polar_source(const Model& m, const typename Model::State& u,
                                                 const Aux& a) {
  if constexpr (PolarHasSource<Model>)
    return m.source(u, a);
  else
    return typename Model::State{};
}

/// PolarHasGeomSource<M>: internal concept -- true if M exposes polar_geom_source(u, r) -> State.
///
/// The GEOMETRIC curvature term (centrifugal -rho v_theta^2/r etc.) only makes sense in polar
/// metric and depends ONLY on the state and the cell radius (no aux). A SCALAR transport brick
/// (ExBVelocityPolar) does not expose it -> we fall back to 0 (bit-identical to the historical
/// polar ExB transport). A polar FLUID brick (IsothermalFluxPolar) exposes it -> it is added in
/// cell by PolarAssembleRhsKernel (which alone knows r_cell(i)).
template <class M>
concept PolarHasGeomSource = requires(const M m, const typename M::State u, Real r) {
  { m.polar_geom_source(u, r) } -> std::convertible_to<typename M::State>;
};

/// polar_geom_source<Model>: returns m.polar_geom_source(u, r) if PolarHasGeomSource<Model>,
/// otherwise the zero state. if constexpr guard: zero extra codegen for scalar bricks (the polar
/// ExB path stays strictly bit-identical). POPS_HD. r > 0 (annulus) enforced upstream.
template <class Model>
POPS_HD inline typename Model::State polar_geom_source(const Model& m,
                                                      const typename Model::State& u, Real r) {
  if constexpr (PolarHasGeomSource<Model>)
    return m.polar_geom_source(u, r);
  else
    return typename Model::State{};
}

// RADIAL FACE FLUX kernel (dir 0): numerical flux at the radial face i (between i-1 and i), ALREADY
// weighted by the face radius r_face(i). We store r_{i-1/2} Fr_{i-1/2} so the divergence is a
// simple difference. NAMED FUNCTOR (device-clean cross-TU).
/// PolarFaceFluxRKernel: device kernel of the flux at the radial face i (weighting by r_face(i)).
///
/// Stores r_face(i) * Fr at the face between i-1 and i, so the discrete divergence is a simple
/// difference (cf. formula in @file). If wall_radial == true, forces the flux to zero at the
/// physical boundary faces (no-penetration wall, mass conservation to machine precision).
/// Named functor, device-clean cross-TU. POPS_HD.
template <class Limiter, class NumericalFlux, class Model>
struct PolarFaceFluxRKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fr;       // output: r_face(i) * Fr at the radial face i (ncomp components)
  Real r_min, dr;  // radial geometry (r_face(i) = r_min + i*dr)
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  // Optional RADIAL WALL (no-penetration). wall_radial == false (default): no effect, boundary flux
  // computed like the interior (BIT-IDENTICAL to the history: MMS, azimuthal conservation). true:
  // the radial flux at BOTH physical boundary faces (i = i_lo_face = lo, i = i_hi_face = hi+1) is
  // forced to ZERO -> the radial term telescopes EXACTLY (each interior face is shared, the
  // boundaries no longer count) -> mass Sum n r dr dtheta conserved to machine precision, whatever
  // v_r (solid wall).
  bool wall_radial;
  int i_lo_face,
      i_hi_face;  // FACE indices of physical boundaries (lo and hi+1); ignored if !wall_radial
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  POPS_HD void operator()(int i, int j) const {
    const Real rf = r_min + i * dr;  // r_face(i) (positive on the annulus: r_min >= 0, i >= 0)
    if (wall_radial && (i == i_lo_face || i == i_hi_face)) {
      for (int c = 0; c < Model::n_vars; ++c)
        fr(i, j, c) = Real(0);  // wall: zero radial flux
      return;
    }
    // Reconstructed states on either side of the radial face i (REUSES Cartesian reconstruct_pp<>,
    // dir == 0). L = extrapolation from cell i-1 toward its + face; R = from cell i toward its
    // - face.
    const auto L =
        reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr =
        reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i - 1, j), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 0);
    for (int c = 0; c < Model::n_vars; ++c)
      fr(i, j, c) = rf * F[c];
  }
};

// AZIMUTHAL FACE FLUX kernel (dir 1): numerical flux at the theta face j (between j-1 and j). NO
// weighting by r (the azimuthal metric 1/r is applied in cell, cf. PolarAssembleRhsKernel).
/// PolarFaceFluxThetaKernel: device kernel of the flux at the azimuthal face j.
///
/// Computes Ftheta at the face between j-1 and j. No weighting by r here: the 1/r factor is applied
/// in cell in PolarAssembleRhsKernel. Named functor. POPS_HD.
template <class Limiter, class NumericalFlux, class Model>
struct PolarFaceFluxThetaKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 ft;  // output: Ftheta at the azimuthal face j (ncomp components)
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  POPS_HD void operator()(int i, int j) const {
    const auto L =
        reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr =
        reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i, j - 1), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 1);
    for (int c = 0; c < Model::n_vars; ++c)
      ft(i, j, c) = F[c];
  }
};

// Polar residual assembly kernel in cell (i, j):
//   R = S - (1/r_i) (Fr_weighted_{i+1} - Fr_weighted_i) / dr - (1/r_i) (Ftheta_{j+1} - Ftheta_j) / dtheta
// Fr_weighted = r_face * Fr (already produced by PolarFaceFluxRKernel). NAMED FUNCTOR.
/// PolarAssembleRhsKernel: device kernel of the polar residual in cell (i,j).
///
/// R = S - (1/r_i) [ (r Fr)_{i+1} - (r Fr)_i ] / dr - (1/r_i) [ Ftheta_{j+1} - Ftheta_j ] / dtheta.
/// fr already contains r_face * Fr (produced by PolarFaceFluxRKernel). Named functor. POPS_HD.
template <class Model>
struct PolarAssembleRhsKernel {
  Model model;
  ConstArray4 u, ax, fr, ft;  // state, aux, r-weighted radial flux, azimuthal flux
  Array4 r;                   // output: residual
  Real r_min, dr, dtheta;
  POPS_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;  // r_cell(i)
    const Real inv_r = Real(1) / ri;
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const auto Us = load_state<Model>(u, i, j);
    const auto S = polar_source<Model>(model, Us, Ac);
    // GEOMETRIC curvature term (centrifugal + cross curvature): not captured by the conservative
    // divergence in the rotating local basis (e_r, e_theta). Zero for scalar bricks (polar ExB =
    // bit-identical); non-zero for a polar fluid (IsothermalFluxPolar). See PolarHasGeomSource /
    // IsothermalFluxPolar for the derivation. r_cell(i) > 0 (annulus).
    const auto Sg = polar_geom_source<Model>(model, Us, ri);
    for (int c = 0; c < Model::n_vars; ++c) {
      const Real div_r = (fr(i + 1, j, c) - fr(i, j, c)) / dr;      // d_r(r Fr) discrete
      const Real div_t = (ft(i, j + 1, c) - ft(i, j, c)) / dtheta;  // d_theta(Ftheta) discrete
      r(i, j, c) = S[c] + Sg[c] - inv_r * (div_r + div_t);
    }
  }
};

// Radial / azimuthal FACE boxes (cf. Cartesian xface_box/yface_box): radial faces = nr+1 x ntheta,
// azimuthal faces = nr x ntheta+1. Reuses the Cartesian helpers (same face index conventions) to
// avoid duplication.
}  // namespace detail

/// assemble_rhs_polar<Limiter, NumericalFlux>: R = -div_polar F* + S on a PolarGeometry. The limiter
/// (reconstruction) and the numerical flux are template parameters, like the Cartesian operator.
/// First computes the FACE fluxes (r-weighted radial, azimuthal) into temporary MultiFabs, then
/// differences. All kernels are device-callable (named functors).
///
/// BOUNDARY CONDITIONS: theta PERIODIC (the caller fills the azimuthal ghosts via periodic
/// fill_boundary). r PHYSICAL: the caller fills the radial ghosts (wall / outflow). The radial
/// fluxes at the r_min (i = lo) and r_max (i = hi+1) faces are computed from the ghost states (free
/// outflow), EXCEPT if @p wall_radial == true: then the radial flux at both physical boundary faces
/// is forced to ZERO (SOLID no-penetration WALL), which makes the mass Sum n r dr dtheta conserved
/// TO MACHINE precision whatever v_r (the radial term telescopes exactly). @p wall_radial == false
/// (default) reproduces EXACTLY the history (MMS, azimuthal conservation -- cf.
/// test_polar_transport_mms).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs_polar(const Model& model, const MultiFab& U, const MultiFab& aux,
                        const PolarGeometry& geom, MultiFab& R, bool recon_prim = false,
                        bool wall_radial = false, Real pos_floor = Real(0)) {
  // STATE-GHOST WIDTH: exactly Limiter::n_ghost, like the Cartesian operator. The polar face kernels
  // (PolarFaceFluxRKernel / PolarFaceFluxThetaKernel) reuse reconstruct_pp<> VERBATIM at the SAME
  // i-1/i (radial) and j-1/j (azimuthal) offsets over the SAME face boxes (xface_box/yface_box, up
  // to hi+1) as compute_face_fluxes -> identical reconstruction stencil (i+-Limiter::n_ghost at the
  // edge of the valid box). The polar metric (r_face = r_min + i*dr, r_cell, 1/r) is computed from
  // INDICES, never read from U, so it adds NO state-ghost width; aux is read at i+-1 only (1 ghost,
  // narrower). HOST-only guard, BEFORE the pass-1/pass-2 loops -- never inside a kernel.
  detail::require_reconstruction_ghosts<Limiter>(U);  // state ghosts >= stencil (otherwise OOB)
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  const Real r_min = geom.r_min, dr = geom.dr(), dtheta = geom.dtheta();
  // Physical radial boundary faces (wall): r_min at the lo face of the index domain, r_max at the
  // hi+1 face. geom.domain is the GLOBAL domain (System's PolarGeometry covers the whole annulus,
  // mono-box).
  const int i_lo_face = geom.domain.lo[0];
  const int i_hi_face = geom.domain.hi[0] + 1;
  const Limiter lim{};
  const NumericalFlux nflux{};
  // FACE BoxArrays (cf. Cartesian compute_face_fluxes): radial faces = surroundingNodes in x
  // (xface_box), azimuthal faces in y (yface_box). Same face index conventions -> fr(i, .) is the
  // radial face between i-1 and i, ft(., j) the azimuthal face between j-1 and j.
  std::vector<Box2D> rfaces, tfaces;
  rfaces.reserve(U.box_array().size());
  tfaces.reserve(U.box_array().size());
  for (const Box2D& b : U.box_array().boxes()) {
    rfaces.push_back(xface_box(b));
    tfaces.push_back(yface_box(b));
  }
  MultiFab Fr(BoxArray(std::move(rfaces)), U.dmap(), Model::n_vars, 0);
  MultiFab Ft(BoxArray(std::move(tfaces)), U.dmap(), Model::n_vars, 0);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fr = Fr.fab(li).array();
    Array4 ft = Ft.fab(li).array();
    const Box2D v = R.box(li);
    // Radial faces: i in [lo..hi+1], j in [lo..hi] (cf. xface_box).
    for_each_cell(xface_box(v), detail::PolarFaceFluxRKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fr, r_min, dr, lim, nflux, recon_prim,
                                    wall_radial, i_lo_face, i_hi_face, pos_floor, pos_comp});
    // Azimuthal faces: i in [lo..hi], j in [lo..hi+1] (cf. yface_box).
    for_each_cell(yface_box(v), detail::PolarFaceFluxThetaKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, ft, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 fr = Fr.fab(li).const_array();
    const ConstArray4 ft = Ft.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(
        v, detail::PolarAssembleRhsKernel<Model>{model, u, ax, fr, ft, r, r_min, dr, dtheta});
  }
}

}  // namespace pops

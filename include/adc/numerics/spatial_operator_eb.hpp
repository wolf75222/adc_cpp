#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/elliptic/cut_fraction.hpp>  // detail::cut_fraction, CutFraction (PR1)
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>  // reconstruct<>, load_state/load_aux, *_face_box (REUSED verbatim)
#include <adc/runtime/wall_predicate.hpp>     // detail::DiscDomain (single-source level set of the disc)

#include <utility>
#include <vector>

/// @file
/// @brief CUT-CELL / EMBEDDED BOUNDARY (EB) spatial operator: R = -div_eb F + S on a disc, in
///        CONSERVATIVE finite volumes with face apertures alpha_f and volume fraction
///        kappa derived from detail::cut_fraction (project T5-PR1).
///
/// CONTEXT (the "Cartesian-ring-edge lock"; cf. docs/HOFFART_FIDELITY.md, line "Domain (disc of
/// radius R)" of the fidelity table). The T2 path
/// (spatial_operator.hpp: assemble_rhs_masked) approximates the disc with a STAIRCASE MASK: an
/// active/inactive face is a 0/1 GATE (normal flux set to zero), the boundary is crenellated. This operator
/// GENERALIZES the 0/1 gate to an APERTURE alpha_f in [0, 1] (the linear fraction of the face inside the
/// disc, EXACTLY the cut_distance/h of the elliptic wall -> bit-for-bit consistency with Poisson) AND divides
/// the residual by the cell's VOLUME FRACTION kappa (volume of the cut cell). This is a 2nd-order EB scheme
/// for smooth transport inside the disc, which renders the growth of the diocotron mode
/// without the structural l-dependent over-rate of the Cartesian baseline.
///
/// NOTE ON alpha_f AND THE ACTIVITY MODEL. Cell activity follows the center-in-the-disc criterion
/// (ls(center) < 0), like GeometricMG and the T2 mask. With this criterion, the geometric aperture
/// cut_distance(lc, ln, h)/h equals EXACTLY 1 on a face between two ACTIVE cells (lc < 0 and
/// ln < 0 -> "interior neighbor" branch of cut_distance -> distance = h) and FRACTIONAL in (0, 1) on
/// the leg from an active cell toward an INACTIVE neighbor (ln >= 0: linear crossing). The immersed
/// boundary is therefore carried by TWO continuous quantities derived from cut_fraction: (i) the fractional
/// leg (which closes the face toward the inactive side, cf. no-penetration wall below) and ABOVE ALL
/// (ii) the VOLUME FRACTION kappa in (0, 1] of the cut cell, which corrects the divergence by the true
/// immersed volume. It is kappa (and not the aperture of internal faces) that de-crenellates the boundary and
/// restores 2nd-order interior transport, which the T2 mask (implicit volume = 1) does not.
///
/// CONSERVATIVE EB FORM (cell (i, j), volume kappa dx dy):
///   kappa dx dy d_t U = - [ alpha_xp Fx_{i+1} - alpha_xm Fx_i ] dy
///                       - [ alpha_yp Fy_{j+1} - alpha_ym Fy_i ] dx
///                       - alpha_wall |wall| F_wall                       (immersed WALL term)
///                       + kappa dx dy S
/// that is, after dividing by kappa dx dy (with kappa CLAMPED, cf. small-cell stability):
///   R = S - (1/kappa) [ (alpha_xp Fx_{i+1} - alpha_xm Fx_i) / dx
///                     + (alpha_yp Fy_{j+1} - alpha_ym Fy_i) / dy ]
///         - (1/kappa) (alpha_wall |wall| / (dx dy)) F_wall
/// The immersed WALL flux is a NO-PENETRATION flux (zero-normal-flux): F_wall = 0. It is the
/// FV counterpart of the conducting wall (the elliptic side applies Dirichlet; the transport applies a solid
/// wall at the SAME geometric boundary). The wall term is therefore IDENTICALLY ZERO; it is written
/// explicitly (and kept at zero) so the contract stays readable and so that a future nonzero wall flux
/// (injection, slip) has its single attachment point.
///
/// MASS CONSERVATION. On two neighboring active cells, the aperture of the SHARED FACE is the
/// SAME on both sides (geometric alpha_f, a function of the level set only: alpha_xp(i) == alpha_xm(i+1),
/// cf. cut_fraction symmetric per face) -> the flux of that face telescopes EXACTLY in the sum
/// Sum_cells kappa dx dy R. A face whose neighbor cell is INACTIVE is CLOSED (alpha_f forced to 0)
/// AND the wall flux is zero -> no mass crosses the immersed boundary. The total mass over
/// the active cells Sum n_ij kappa_ij dx dy is therefore conserved TO MACHINE PRECISION. This is the EB analogue
/// (continuous apertures) of the T2 conservation (0/1 gates).
///
/// SMALL-CELL STABILITY (small-cell problem). The 1/kappa factor amplifies the residual when kappa
/// becomes small on the r0/r1 shear layer; at a FIXED time step dt calibrated on full cells, an
/// unbounded amplification blows up (Inf/NaN) the explicit step on a strongly cut cell. Two STACKED
/// complementary guards:
///   1. The PR1 primitive floor: cut_distance clamps each half-face at theta >= 1e-3 (anti-division
///      guard INHERITED from the elliptic wall). kappa = product of half-face averages can therefore
///      never equal EXACTLY 0 -> no strict division by zero. BUT the induced lower bound is
///      ~ (1e-3)^2 / 4 ~ 2.5e-7: 1/kappa can reach ~4e6, which is enough to overflow the fixed step.
///   2. RETAINED SCHEME = VOLUME CLAMP (this header): kappa_eff = max(kappa, kappa_min), kappa_min by
///      default 1e-2 (1% of the full volume). A SCHEME-LEVEL guard, INDEPENDENT of the elliptic floor:
///      it bounds the 1/kappa amplification to 1/kappa_min = 100, a value CALIBRATED so the fixed explicit
///      step stays stable whatever the degree of cut. This is the simplest and most robust implicit
///      "volume merging" for a FIXED step, at the cost of a slight LOCAL NON-conservation on the
///      most-cut cells (effective volume > real volume).
/// The GLOBAL mass stays conserved TO MACHINE PRECISION because the clamp acts only on the DENOMINATOR (volume),
/// NOT on the face fluxes (numerator): the telescoping sum of fluxes is unchanged (the discrete mass
/// consistent with the scheme uses the SAME kappa_eff, cf. conservation test). Documented alternative
/// (outside PR2): flux redistribution (AMReX-EB's flux redistribution) spreads the excess
/// divergence of small cells onto the full neighbors -> exact LOCAL conservation but a non-local stencil;
/// the clamp suffices for the target (calibrated fixed step, smooth MMS, global conservation).
///
/// NAMED FUNCTORS (and not extended lambdas), like spatial_operator.hpp / _polar.hpp (#64/#97):
/// robust device emission when the Model-template kernel is instantiated cross-TU. The RECONSTRUCTION and the
/// numerical FLUX are REUSED verbatim from the Cartesian operator (reconstruct<>, RusanovFlux).
/// The level set is passed BY VALUE (ADC_HD callable, e.g. captured detail::DiscDomain): device-safe.
///
/// INVARIANT: this header is PURELY ADDITIVE and OPT-IN. The Cartesian operator (assemble_rhs) and the
/// T2 mask path (assemble_rhs_masked) stay STRICTLY UNTOUCHED; a run without an EB disc is
/// bit-identical. assemble_rhs_eb is called only on explicit opt-in (geometry = cutcell).

namespace adc {

namespace detail {

/// Default aperture below which a face is treated as CLOSED (immersed wall). Below
/// this threshold the linear aperture is numerically zero (the face barely crosses the disc);
/// the anti-division clamp of cut_distance already stops at 1e-3, this threshold aligns it with the FV closure.
constexpr Real kEbFaceOpenEps = Real(1e-6);

/// Volume fraction floor (small-cell clamp). kappa_eff = max(kappa, kEbKappaMin) bounds
/// the 1/kappa amplification to 1/kEbKappaMin = 100 -> finite residual on an arbitrarily cut cell,
/// stable fixed explicit step. See the SMALL-CELL STABILITY note of @file.
constexpr Real kEbKappaMin = Real(1e-2);

/// Device-safe adapter: makes DiscDomain (which exposes level_set/cell_active, NOT operator()) usable
/// as the Real(Real, Real) callable expected by cut_fraction and the EB operator. NAMED FUNCTOR (captures
/// the DiscDomain BY VALUE: three doubles, device-safe), not an extended lambda. operator() forwards
/// EXACTLY DiscDomain::level_set -> same cut geometry as the elliptic wall (bit consistency).
struct DiscLevelSet {
  DiscDomain disc;
  ADC_HD Real operator()(Real x, Real y) const { return disc.level_set(x, y); }
};

/// Builds the disc level set callable from a DiscDomain (sugar: disc_level_set(d)).
ADC_HD inline DiscLevelSet disc_level_set(const DiscDomain& d) { return DiscLevelSet{d}; }

/// Activity indicator (center in the disc, ls < 0) from a callable level set. ADC_HD.
template <class LevelSet>
ADC_HD inline bool eb_cell_active(const LevelSet& ls, Real xc, Real yc) {
  return ls(xc, yc) < Real(0);
}

/// Aperture of ONE face between the active cell (xc, yc) and its neighbor at (xn, yn), step h.
///
/// FV EB convention:
///   - INACTIVE neighbor (ls(xn,yn) >= 0): the face touches the immersed wall -> CLOSED (alpha = 0,
///     no-penetration). This is the generalization of the T2 0/1 gate: the inactive side closes the face.
///   - ACTIVE neighbor (ln < 0): the aperture reuses VERBATIM the shared primitive cut_distance
///     (hence bit-consistent with the elliptic wall), alpha = cut_distance(lc, ln, h) / h. But for an
///     active neighbor cut_distance takes the "interior neighbor" branch and returns h -> alpha = 1 EXACTLY,
///     far and near the edge alike: the shared face of two active cells is always FULL. The face
///     apertures are therefore BINARY {0, 1}; it is the VOLUME FRACTION kappa in (0, 1], and
///     not the aperture of internal faces, that carries the cut geometry (cf. NOTE alpha_f of @file).
/// SYMMETRY (key to conservation): cut_distance(lc, ln, h) depends only on (lc, ln); the shared
/// face seen from cell i (center lc, neighbor ln) and seen from cell i+1 (center ln, neighbor lc)
/// gives the SAME aperture as soon as both are active (ln < 0: the "interior neighbor" branch returns
/// h on both sides -> alpha = 1). The internal active/active boundary is therefore treated SYMMETRICALLY.
ADC_HD inline Real eb_face_aperture(Real lc, Real ln, Real h) {
  if (ln >= Real(0)) return Real(0);          // inactive neighbor: closed face (wall, no-penetration)
  return cut_distance(lc, ln, h) / h;          // active neighbor: linear aperture (== elliptic wall)
}

/// FACE FLUX kernel for x (dir 0) of the EB transport: numerical flux at the face between (i-1, j) and
/// (i, j), WEIGHTED by the aperture alpha_x of that face. We store alpha_x * Fx so the EB divergence
/// is a simple difference (like the r weighting of the polar operator). NAMED FUNCTOR (device-clean).
///
/// The aperture of face i is computed FROM THE SIDE of the active cell: if (i, j) is active, we take
/// the aperture seen from (i, j) toward its neighbor (i-1); if (i, j) is inactive but (i-1, j) is active, we
/// take the aperture seen from (i-1) toward (i). If BOTH are inactive, the face is outside the active domain
/// (alpha = 0). This symmetry guarantees the uniqueness of alpha on the shared face (conservation).
template <class Limiter, class NumericalFlux, class Model, class LevelSet>
struct EbFaceFluxXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fx;            // output: alpha_x * Fx at the face between i-1 and i (ncomp components)
  Real dx;
  Geometry geom;        // cell centers (level set evaluated at the center, like cut_fraction)
  LevelSet ls;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  ADC_HD void operator()(int i, int j) const {
    const Real xL = geom.x_cell(i - 1), xR = geom.x_cell(i), yc = geom.y_cell(j);
    const Real lL = ls(xL, yc), lR = ls(xR, yc);
    const bool aL = lL < Real(0), aR = lR < Real(0);
    Real alpha;
    if (aL && aR) {
      alpha = eb_face_aperture(lL, lR, dx);     // internal active/active face: symmetric aperture
    } else if (aR) {
      alpha = eb_face_aperture(lR, lL, dx);     // (i) active, (i-1) inactive: closed face (0)
    } else if (aL) {
      alpha = eb_face_aperture(lL, lR, dx);     // (i-1) active, (i) inactive: closed face (0)
    } else {
      alpha = Real(0);                           // both inactive: face outside the active domain
    }
    if (alpha < kEbFaceOpenEps) {                // closed face (immersed wall): zero normal flux
      for (int c = 0; c < Model::n_vars; ++c) fx(i, j, c) = Real(0);
      return;
    }
    const auto L = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i - 1, j), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 0);
    for (int c = 0; c < Model::n_vars; ++c) fx(i, j, c) = alpha * F[c];
  }
};

/// FACE FLUX kernel for y (dir 1) of the EB transport: analogue of EbFaceFluxXKernel in j. Stores
/// alpha_y * Fy at the face between (i, j-1) and (i, j). NAMED FUNCTOR (device-clean).
template <class Limiter, class NumericalFlux, class Model, class LevelSet>
struct EbFaceFluxYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fy;
  Real dy;
  Geometry geom;
  LevelSet ls;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< Zhang-Shu positivity limiter (<= 0: inactive, bit-identical)
  int pos_comp = 0;          ///< component of the Density role (resolved by the host caller)
  ADC_HD void operator()(int i, int j) const {
    const Real xc = geom.x_cell(i), yL = geom.y_cell(j - 1), yR = geom.y_cell(j);
    const Real lL = ls(xc, yL), lR = ls(xc, yR);
    const bool aL = lL < Real(0), aR = lR < Real(0);
    Real alpha;
    if (aL && aR) {
      alpha = eb_face_aperture(lL, lR, dy);
    } else if (aR) {
      alpha = eb_face_aperture(lR, lL, dy);
    } else if (aL) {
      alpha = eb_face_aperture(lL, lR, dy);
    } else {
      alpha = Real(0);
    }
    if (alpha < kEbFaceOpenEps) {
      for (int c = 0; c < Model::n_vars; ++c) fy(i, j, c) = Real(0);
      return;
    }
    const auto L = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i, j - 1), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 1);
    for (int c = 0; c < Model::n_vars; ++c) fy(i, j, c) = alpha * F[c];
  }
};

/// Kernel assembling the EB residual at cell (i, j):
///   INACTIVE cell -> residual 0 (not advanced, like T2);
///   ACTIVE cell -> R = S - (1/kappa_eff) [ (fx_{i+1} - fx_i)/dx + (fy_{j+1} - fy_i)/dy ] - wall_term.
/// fx/fy ALREADY contain alpha_f * F (produced by the face kernels). kappa_eff = max(kappa, min)
/// (small-cell clamp). The immersed WALL term is a no-penetration F_wall = 0 -> zero, written
/// explicitly (kept at zero) as a single attachment point. NAMED FUNCTOR (device-clean).
template <class Model, class LevelSet>
struct EbAssembleRhsKernel {
  Model model;
  ConstArray4 u, ax, fx, fy;  // state, aux, x flux weighted by alpha, y flux weighted by alpha
  Array4 r;                    // output: residual
  Real dx, dy;
  Geometry geom;
  LevelSet ls;
  Real kappa_min;
  ADC_HD void operator()(int i, int j) const {
    const Real xc = geom.x_cell(i), yc = geom.y_cell(j);
    if (!eb_cell_active(ls, xc, yc)) {            // outside the disc: zero residual, not advanced (cf. T2)
      for (int c = 0; c < Model::n_vars; ++c) r(i, j, c) = Real(0);
      return;
    }
    // Volume fraction kappa derived EXACTLY from the same cut_fraction as the elliptic wall: the
    // cut geometry is the single source of truth (face apertures AND volume). kappa in (0, 1].
    const CutFraction cf = cut_fraction(ls, xc, yc, dx, dy);
    // SMALL-CELL CLAMP: bounds 1/kappa to 1/kappa_min -> finite residual, stable fixed step. Acts ONLY on
    // the denominator (volume); the fluxes (numerator) are unchanged -> GLOBAL conservation preserved.
    const Real kappa_eff = cf.kappa > kappa_min ? cf.kappa : kappa_min;
    const Real inv_kappa = Real(1) / kappa_eff;

    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const auto S = model.source(load_state<Model>(u, i, j), Ac);

    // Immersed WALL flux (no-penetration): F_wall = 0 -> zero term. Single attachment point for a
    // future nonzero wall flux. Stays at 0 in PR2 (solid wall, like the elliptic Dirichlet wall).
    constexpr Real wall_flux = Real(0);

    for (int c = 0; c < Model::n_vars; ++c) {
      const Real div_x = (fx(i + 1, j, c) - fx(i, j, c)) / dx;  // discrete d_x(alpha Fx)
      const Real div_y = (fy(i, j + 1, c) - fy(i, j, c)) / dy;  // discrete d_y(alpha Fy)
      // TERM-BY-TERM accumulation (and not inv_kappa*(div_x + div_y)): when kappa_eff = 1 and all
      // alpha = 1 (case WITHOUT cut), inv_kappa = 1 and each inv_kappa*div_* = div_* by IEEE identity
      // (x*1.0 == x), so r = S - div_x - div_y - 0 reproduces BIT FOR BIT the Cartesian operator
      // (S - (Fxp-Fxm)/dx - (Fyp-Fym)/dy), term by term. A grouping (div_x + div_y) would break
      // floating-point associativity and the bit-identity of the default path.
      r(i, j, c) = S[c] - inv_kappa * div_x - inv_kappa * div_y - inv_kappa * wall_flux;
    }
  }
};

}  // namespace detail

/// assemble_rhs_eb<Limiter, NumericalFlux>: residual R = -div_eb F + S on a DISC in cut-cell / EB,
/// with face apertures alpha_f in [0, 1] and volume fraction kappa derived from detail::cut_fraction
/// (T5-PR1). This is the EB generalization of the T2 mask path (assemble_rhs_masked, 0/1 gates): the
/// immersed boundary is no longer crenellated, the scheme is 2nd order for smooth transport
/// inside the disc, and the mass is conserved TO MACHINE PRECISION (no flux crosses the wall).
///
/// @tparam Limiter        reconstruction (NoSlope / Minmod / VanLeer / Weno5), like the Cartesian operator.
/// @tparam NumericalFlux  flux policy (RusanovFlux by default).
/// @param  ls             ADC_HD callable level set (e.g. detail::DiscDomain): ls < 0 inside.
/// @param  kappa_min      volume fraction floor (small-cell clamp), default kEbKappaMin.
///
/// IMPLEMENTATION in TWO PASSES (structure REUSED from the polar operator): pass 1 computes the
/// FACE fluxes weighted by alpha_f into temporary MultiFabs; pass 2 differences and divides by
/// kappa_eff. NO mask MultiFab is required: cell activity AND face apertures
/// all derive from the SINGLE level set (single geometric source), like cut_fraction.
///
/// BOUNDARY CONDITIONS: the caller fills the ghosts (fill_ghosts) before the call, as for
/// assemble_rhs. Faces touching an inactive cell are CLOSED by the kernel (immersed wall),
/// so the physical BC of the square box does not influence the interior disc (the EB masks it).
///
/// INVARIANT: SEPARATE entry point; the default path (assemble_rhs) stays bit-identical as long
/// as it does NOT call this overload.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model, class LevelSet>
void assemble_rhs_eb(const Model& model, const MultiFab& U, const MultiFab& aux, const LevelSet& ls,
                     const Geometry& geom, MultiFab& R, bool recon_prim = false,
                     Real kappa_min = detail::kEbKappaMin, Real pos_floor = Real(0)) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  // FACE BoxArrays (cf. compute_face_fluxes / polar operator): x faces = surroundingNodes in x
  // (xface_box), y faces in y (yface_box). fx(i, .) is the face between i-1 and i, fy(., j) between j-1 and j.
  std::vector<Box2D> xfaces, yfaces;
  xfaces.reserve(U.box_array().size());
  yfaces.reserve(U.box_array().size());
  for (const Box2D& b : U.box_array().boxes()) {
    xfaces.push_back(xface_box(b));
    yfaces.push_back(yface_box(b));
  }
  MultiFab Fx(BoxArray(std::move(xfaces)), U.dmap(), Model::n_vars, 0);
  MultiFab Fy(BoxArray(std::move(yfaces)), U.dmap(), Model::n_vars, 0);
  // PASS 1: face fluxes weighted by alpha_f.
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(xface_box(v),
                  detail::EbFaceFluxXKernel<Limiter, NumericalFlux, Model, LevelSet>{
                      model, u, ax, fx, dx, geom, ls, lim, nflux, recon_prim, pos_floor, pos_comp});
    for_each_cell(yface_box(v),
                  detail::EbFaceFluxYKernel<Limiter, NumericalFlux, Model, LevelSet>{
                      model, u, ax, fy, dy, geom, ls, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
  // PASS 2: EB divergence / kappa_eff + source; inactive cell -> residual 0.
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 fx = Fx.fab(li).const_array();
    const ConstArray4 fy = Fy.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::EbAssembleRhsKernel<Model, LevelSet>{model, u, ax, fx, fy, r, dx, dy,
                                                                  geom, ls, kappa_min});
  }
}

}  // namespace adc

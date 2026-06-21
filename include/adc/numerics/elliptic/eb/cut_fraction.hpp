#pragma once

#include <adc/core/types.hpp>  // Real, ADC_HD

/// @file
/// @brief SHARED cut-fraction primitive (cut-cell / embedded boundary).
///
/// A SINGLE face-crossing computation shared between:
///   - the elliptic solver (geometric_mg.hpp: Shortley-Weller weights of the Poisson wall),
///   - the future EB transport (FV aperture of the disc faces).
/// Both MUST read the SAME aperture geometry so the FV aperture is bit-consistent with the
/// elliptic wall (the "Cartesian ring edge" lock: the FV aperture and the elliptic wall must
/// agree on the same disc-of-radius-R boundary; cf. docs/HOFFART_FIDELITY.md for the validation
/// fidelity table).
///
/// The canonical level-set is detail::DiscDomain::level_set (numerics/embedded_boundary.hpp):
///   ls(x, y) = hypot(x - cx, y - cy) - R, < 0 INSIDE, sign of the boundary.
/// This primitive is header-only, ADC_HD (device-safe) and STATELESS: it takes a level-set
/// callback by value and the cell, and returns purely geometric distances/aperture.

namespace adc {
namespace detail {

/// Cut distance of ONE face along a direction, starting from the active center (ls < 0).
///
/// HISTORICAL GeometricMG convention (geometric_mg.hpp, 'cut' lambda) reused IDENTICALLY to
/// guarantee bit-identical Shortley-Weller weights:
///   - INTERIOR neighbor (ln < 0): no cut, the face is full -> distance = h;
///   - the level-set changes sign (ln >= 0): linear fraction theta = lc / (lc - ln) (linear
///     crossing between the center lc < 0 and the neighbor ln >= 0), distance = theta * h;
///   - anti division-by-zero guard: theta is clamped to [1e-3, 1] (theta -> 0 would make the weight diverge).
/// lc is assumed < 0 (active cell); the clamp bounds are the original ones.
ADC_HD inline Real cut_distance(Real lc, Real ln, Real h) {
  if (ln < Real(0))
    return h;                // interior neighbor: no cut (full face)
  Real th = lc / (lc - ln);  // ls changes sign: linear cut fraction
  if (th < Real(1e-3))
    th = Real(1e-3);  // anti division-by-zero guard (theta -> 0)
  if (th > Real(1))
    th = Real(1);
  return th * h;
}

/// Geometric result of crossing a cut cell: 4 cut distances per face, the 4 apertures alpha_f
/// normalized in [0, 1] (alpha_f = face_distance / h), and the volume fraction kappa of the cell
/// (share of the cell in the active domain). axm/axp carry the x direction (neighbors i-1 / i+1),
/// aym/ayp the y direction (neighbors j-1 / j+1).
///
/// alpha_f and kappa are the quantities that the EB TRANSPORT will consume (face flux attenuated by
/// alpha_f, cell volume kappa); axm/axp/aym/ayp are what the ELLIPTIC already consumes for the
/// Shortley-Weller weights. Everything is derived from the SAME cut_distance -> bit-for-bit consistency.
struct CutFraction {
  Real axm;       ///< cut distance of face x- (toward i-1), in [1e-3*dx, dx]
  Real axp;       ///< cut distance of face x+ (toward i+1)
  Real aym;       ///< cut distance of face y- (toward j-1)
  Real ayp;       ///< cut distance of face y+ (toward j+1)
  Real alpha_xm;  ///< aperture of face x- = axm / dx, in [1e-3, 1]
  Real alpha_xp;  ///< aperture of face x+ = axp / dx
  Real alpha_ym;  ///< aperture of face y- = aym / dy
  Real alpha_yp;  ///< aperture of face y+ = ayp / dy
  Real kappa;     ///< volume fraction of the cell (share in the active domain), in (0, 1]
};

/// Computes the cut geometry of an ACTIVE cell (center (xc, yc) with ls < 0) from a
/// level-set @p ls evaluated at the center and at the 4 cardinal neighbors at distance @p dx / @p dy.
///
/// @tparam LevelSet callable Real(Real, Real) device-safe (e.g. DiscDomain::level_set capture).
///
/// The cell is assumed ACTIVE (the caller has already tested ls(xc, yc) < 0, as GeometricMG skips
/// conductor cells). The 4 face distances reuse cut_distance (so strictly the original logic). The
/// apertures normalize by the step. kappa is a volume fraction DERIVED from the same apertures
/// (average of the two half-faces per direction, product of the two directions): far from the
/// boundary (all apertures = 1) kappa = 1; near the boundary kappa < 1. kappa does NOT alter the
/// elliptic (which only uses axm/axp/aym/ayp); it is provided for the upcoming EB transport.
template <class LevelSet>
ADC_HD inline CutFraction cut_fraction(const LevelSet& ls, Real xc, Real yc, Real dx, Real dy) {
  const Real lc = ls(xc, yc);
  const Real axm = cut_distance(lc, ls(xc - dx, yc), dx);
  const Real axp = cut_distance(lc, ls(xc + dx, yc), dx);
  const Real aym = cut_distance(lc, ls(xc, yc - dy), dy);
  const Real ayp = cut_distance(lc, ls(xc, yc + dy), dy);
  const Real alpha_xm = axm / dx;
  const Real alpha_xp = axp / dx;
  const Real alpha_ym = aym / dy;
  const Real alpha_yp = ayp / dy;
  // Volume fraction: average of the half-faces per axis (mean extent of the cell along each
  // direction, normalized), product of the two axes. Far from the boundary -> 1; cut cell -> < 1.
  const Real kappa = Real(0.5) * (alpha_xm + alpha_xp) * Real(0.5) * (alpha_ym + alpha_yp);
  return CutFraction{axm, axp, aym, ayp, alpha_xm, alpha_xp, alpha_ym, alpha_yp, kappa};
}

/// Shortley-Weller weights (5-point cut-cell stencil) from the 4 cut distances. Returns
/// EXACTLY the 5 coefficients that GeometricMG writes into its coef field (components 0..4):
///   w_xm = 2 / (axm (axm + axp)),  w_xp = 2 / (axp (axm + axp)),
///   w_ym = 2 / (aym (aym + ayp)),  w_yp = 2 / (ayp (aym + ayp)),
///   w_diag = 2 / (axm axp) + 2 / (aym ayp).
/// Centralizes the formula so the elliptic assembly stays the single source of cut-cell truth.
struct ShortleyWellerWeights {
  Real w_xm, w_xp, w_ym, w_yp, w_diag;
};

ADC_HD inline ShortleyWellerWeights shortley_weller(const CutFraction& cf) {
  const Real sx = cf.axm + cf.axp;
  const Real sy = cf.aym + cf.ayp;
  return ShortleyWellerWeights{
      Real(2) / (cf.axm * sx),                                     // w_xm on p(i-1)
      Real(2) / (cf.axp * sx),                                     // w_xp on p(i+1)
      Real(2) / (cf.aym * sy),                                     // w_ym on p(i,j-1)
      Real(2) / (cf.ayp * sy),                                     // w_yp on p(i,j+1)
      Real(2) / (cf.axm * cf.axp) + Real(2) / (cf.aym * cf.ayp)};  // w_diag
}

}  // namespace detail
}  // namespace adc

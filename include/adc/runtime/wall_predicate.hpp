#pragma once

#include <adc/core/types.hpp>  // Real

#include <cmath>       // std::hypot
#include <functional>  // std::function
#include <stdexcept>   // std::runtime_error
#include <string>

/// @file
/// @brief Wall predicate (embedded conductor) shared by the System and AmrSystem runtimes.
///        Both derived the same predicate from the same parameters (wall, radius, L);
///        only the error-message prefix differed. Centralized here by PURE extraction:
///        the body (centered circle, std::hypot < R comparison) is reused identically.

namespace adc {
namespace detail {

/// Builds the "inside the conductor" predicate (embedded wall for the Poisson solver)
/// from the wall mode @p wall, the radius @p wall_radius and the domain size @p L.
///   - "none": no wall -> empty predicate.
///   - "circle": disc centered at (L/2, L/2) with radius @p wall_radius.
///   - other: error, prefixed by @p err_context (e.g. "System::set_poisson").
/// Body reused identically from the System / AmrSystem runtimes (bit-identical).
inline std::function<bool(Real, Real)> wall_predicate(const std::string& wall,
                                                      double wall_radius, double L,
                                                      const std::string& err_context) {
  if (wall == "none") return {};
  if (wall == "circle") {
    const double cx = 0.5 * L, cy = 0.5 * L, R = wall_radius;
    return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
  }
  throw std::runtime_error(err_context + ": unknown wall '" + wall + "'");
}

/// DISC geometry descriptor: SINGLE SOURCE of truth for the paper's real physical domain
/// (Hoffart et al., arXiv:2510.11808, Sec 5.3: disc D of radius R). It is the "transport"
/// counterpart of the Poisson wall: the wall acts only on the elliptic part (cf. wall_predicate /
/// geometric_mg cut_cell), whereas this descriptor is used to build a cell-centered DOMAIN MASK to
/// make the FV transport path aware of the disc ("Cartesian-ring-edge" lock, cf.
/// docs/HOFFART_FIDELITY.md, line "Domain (disc of radius R)" of the fidelity table,
/// documented as the "Cartesian-ring-edge lock").
///
/// REUSES EXACTLY the conductor wall level set (geometric_mg.hpp line 71):
///   ls(x, y) = hypot(x - cx, y - cy) - R, < 0 INSIDE.
/// A cell is ACTIVE when its CENTER is inside the disc (ls < 0), exactly like the inside predicate
/// of GeometricMG (active = ls < 0). The default center (cx, cy) = (L/2, L/2) coincides with that of
/// wall_predicate("circle", ...).
///
/// CONTRACT (T2 work item, inert by default): this descriptor changes NOTHING in the default
/// behavior. It is materialized (mask MultiFab, mask-aware transport) only on explicit opt-in
/// (System::set_disc_domain). As long as no disc is set, the mask is "all active" and the
/// FV/AMR/MPI path stays BIT-IDENTICAL.
struct DiscDomain {
  double cx = 0.0;  ///< center x (default L/2 when built from L)
  double cy = 0.0;  ///< center y (default L/2 when built from L)
  double R = 0.0;   ///< disc radius

  /// Disc centered in a square box [0, L]^2 with radius @p radius (same center as
  /// wall_predicate("circle", radius, L): (L/2, L/2)).
  static DiscDomain centered_in_box(double L, double radius) {
    return DiscDomain{0.5 * L, 0.5 * L, radius};
  }

  /// Level set ls(x, y) = hypot(x - cx, y - cy) - R: < 0 inside, 0 at the boundary, > 0 outside.
  /// Identical to the conductor wall convention (geometric_mg cut_cell).
  ADC_HD Real level_set(Real x, Real y) const {
    return static_cast<Real>(std::hypot(static_cast<double>(x) - cx, static_cast<double>(y) - cy) - R);
  }

  /// ACTIVE cell: its center (x, y) is inside the disc (ls < 0). Same inside test as
  /// the GeometricMG active one (ls < 0).
  ADC_HD bool cell_active(Real x, Real y) const { return level_set(x, y) < Real(0); }
};

}  // namespace detail
}  // namespace adc

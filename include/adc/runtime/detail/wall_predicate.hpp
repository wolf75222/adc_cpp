#pragma once

#include <adc/core/foundation/types.hpp>  // Real
#include <adc/numerics/embedded_boundary.hpp>  // detail::DiscDomain (the level-set domain it lives in since ADC-327)

#include <cmath>       // std::hypot
#include <functional>  // std::function
#include <stdexcept>   // std::runtime_error
#include <string>

/// @file
/// @brief ELLIPTIC conductor-wall predicate shared by the System and AmrSystem runtimes (the wall acts
///        on the Poisson side). Both derived the same predicate from the same parameters (wall, radius,
///        L); only the error-message prefix differed. Centralized here by PURE extraction: the body
///        (centered circle, std::hypot < R comparison) is reused identically.
///
///        The TRANSPORT-side domain descriptor (detail::DiscDomain) and its generic contract moved to
///        numerics/embedded_boundary.hpp (ADC-327); this header re-includes it so existing includers
///        keep seeing detail::DiscDomain unchanged.

namespace adc {
namespace detail {

/// Builds the "inside the conductor" predicate (embedded wall for the Poisson solver)
/// from the wall mode @p wall, the radius @p wall_radius and the domain size @p L.
///   - "none": no wall -> empty predicate.
///   - "circle": disc centered at (L/2, L/2) with radius @p wall_radius.
///   - other: error, prefixed by @p err_context (e.g. "System::set_poisson").
/// Body reused identically from the System / AmrSystem runtimes (bit-identical).
inline std::function<bool(Real, Real)> wall_predicate(const std::string& wall, double wall_radius,
                                                      double L, const std::string& err_context) {
  if (wall == "none")
    return {};
  if (wall == "circle") {
    const double cx = 0.5 * L, cy = 0.5 * L, R = wall_radius;
    return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
  }
  throw std::runtime_error(err_context + ": unknown wall '" + wall + "'");
}

}  // namespace detail
}  // namespace adc

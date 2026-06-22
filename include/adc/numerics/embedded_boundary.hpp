#pragma once

#include <adc/core/foundation/types.hpp>  // Real, ADC_HD

#include <cmath>     // std::hypot
#include <concepts>  // std::convertible_to (diagnostics only)

/// @file
/// @brief Generic EMBEDDED-BOUNDARY / LEVEL-SET DOMAIN contract (ADC-327).
///
/// A level-set domain restricts the transport to a SUB-REGION of the fixed 2D Cartesian plane
/// (it masks cells; it adds NO third index, cf. ADR-0001 Decision 1: ADC-327 is boundary geometry
/// at fixed dimension, not dimensionality). The cut-cell / embedded-boundary numerics
/// (numerics/elliptic/cut_fraction.hpp, numerics/spatial_operator_eb.hpp) and the staircase mask path
/// (numerics/spatial_operator.hpp) are ALREADY generic over any device-safe level-set callable; this
/// header gives that contract a NAME and the two built-in instances:
///   - @ref detail::DiscDomain   : the circle, the historical/canonical instance;
///   - @ref detail::HalfPlaneDomain : a linear half-plane, the simplest NON-disc instance.
///
/// CONTRACT (duck-typed; a domain type provides):
///   - @c ADC_HD Real level_set(Real x, Real y) const : signed level set, < 0 INSIDE the active
///     domain, 0 on the boundary, > 0 outside (the conducting-wall convention of GeometricMG);
///   - @c ADC_HD Real operator()(Real x, Real y) const : the CALLABLE form consumed by cut_fraction /
///     assemble_rhs_eb (forwards to level_set); a domain is therefore directly usable as the LevelSet
///     template argument, with no adapter;
///   - @c ADC_HD bool cell_active(Real x, Real y) const : a cell is ACTIVE when its CENTER is inside
///     (level_set < 0), the same activity criterion as GeometricMG and the staircase mask.
///
/// DEVICE-CLEAN: instances are PODs captured BY VALUE into the kernels (no std::function, no runtime
/// polymorphism, no heap), so they cross the host/device boundary like any other kernel argument.
///
/// The @ref LevelSetDomain concept below is for DIAGNOSTICS ONLY (static_assert on the built-in
/// instances, clearer template errors). It deliberately does NOT constrain the hot-path operator
/// signatures (assemble_rhs_eb, cut_fraction): those stay plain templates so a bare callable (e.g. a
/// local test functor or a future device lambda) keeps working and so the Kokkos/CUDA overload
/// resolution is unchanged.

namespace adc {

/// Diagnostics-only contract for an embedded-boundary / level-set domain (see @file). Used in
/// static_assert; NEVER as a constraint on the numerics templates (would risk changing overload
/// resolution / CUDA compatibility on the device hot path).
template <class D>
concept LevelSetDomain = requires(const D d, Real x, Real y) {
  { d.level_set(x, y) } -> std::convertible_to<Real>;
  { d(x, y) } -> std::convertible_to<Real>;
  { d.cell_active(x, y) } -> std::convertible_to<bool>;
};

namespace detail {

/// CIRCLE / DISC level-set domain: the canonical instance of the contract and the SINGLE SOURCE of
/// truth for the active circular domain (a disc of radius R; see docs/HOFFART_FIDELITY.md for the
/// reference scenario it was validated against). It is the "transport" counterpart of the Poisson
/// conductor wall: the wall acts only on the
/// elliptic part (cf. runtime/wall_predicate.hpp / geometric_mg cut_cell), whereas this descriptor
/// drives a cell-centered DOMAIN MASK / cut-cell aperture so the FV transport is disc-aware (the
/// "Cartesian-ring-edge lock", cf. docs/HOFFART_FIDELITY.md).
///
/// REUSES EXACTLY the conductor-wall level set (geometric_mg.hpp):
///   ls(x, y) = hypot(x - cx, y - cy) - R, < 0 INSIDE. A cell is ACTIVE when its CENTER is inside
/// (ls < 0), exactly like GeometricMG's inside predicate. The default center (cx, cy) = (L/2, L/2)
/// coincides with that of wall_predicate("circle", ...).
///
/// CONTRACT (inert by default): this descriptor changes NOTHING in the default behavior. It is
/// materialized (mask MultiFab, mask-aware transport) only on explicit opt-in
/// (System::set_disc_domain). With no domain set the mask is "all active" and the FV/AMR/MPI path
/// stays BIT-IDENTICAL.
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
  /// Identical to the conductor-wall convention (geometric_mg cut_cell).
  ADC_HD Real level_set(Real x, Real y) const {
    return static_cast<Real>(std::hypot(static_cast<double>(x) - cx, static_cast<double>(y) - cy) -
                             R);
  }

  /// Callable form (the LevelSet argument of cut_fraction / assemble_rhs_eb); forwards to level_set.
  ADC_HD Real operator()(Real x, Real y) const { return level_set(x, y); }

  /// ACTIVE cell: its center (x, y) is inside the disc (ls < 0). Same inside test as GeometricMG.
  ADC_HD bool cell_active(Real x, Real y) const { return level_set(x, y) < Real(0); }
};

/// HALF-PLANE level-set domain: ls(x, y) = a*x + b*y - c, ACTIVE on the side a*x + b*y < c. The
/// simplest NON-disc instance of the contract; it proves the embedded-boundary machinery is generic
/// over an arbitrary device-safe level set, not the circle alone. POD, device-clean (three doubles).
struct HalfPlaneDomain {
  double a = 0.0;  ///< x coefficient of the linear level set
  double b = 0.0;  ///< y coefficient
  double c = 0.0;  ///< offset: active where a*x + b*y - c < 0

  /// Level set ls(x, y) = a*x + b*y - c: < 0 inside (active side), 0 on the line, > 0 outside.
  ADC_HD Real level_set(Real x, Real y) const {
    return static_cast<Real>(a * static_cast<double>(x) + b * static_cast<double>(y) - c);
  }

  /// Callable form (forwards to level_set), so the domain is directly usable as a LevelSet argument.
  ADC_HD Real operator()(Real x, Real y) const { return level_set(x, y); }

  /// ACTIVE cell: its center (x, y) is on the active side (ls < 0).
  ADC_HD bool cell_active(Real x, Real y) const { return level_set(x, y) < Real(0); }
};

}  // namespace detail
}  // namespace adc

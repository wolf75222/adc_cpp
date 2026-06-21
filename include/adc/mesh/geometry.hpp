/// @file
/// @brief Geometry: index-space (Box2D) <-> Cartesian physical-space mapping;
///        PolarGeometry: SIBLING for a global annular domain (r, theta).
///
/// Physical domain is FIXED, no dx/dy cell size shrinking with refinement (refine() refines
/// the index space at constant physical EXTENT). Cell centers (and faces, in polar) defined
/// for EVERY index, ghosts included (negative indices). Accessors are ADC_HD: pure arithmetic,
/// capturable by value and callable from a device kernel. Trivial POD: the annotation is
/// free and keeps the host path bit-identical.

#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>

namespace adc {

/// Cartesian geometry of a level: index domain + physical bounds [xlo, xhi] x [ylo, yhi].
/// Trivial POD; ADC_HD accessors. Uniform cell size (dx = (xhi-xlo)/nx, same for dy).
struct Geometry {
  Box2D domain{};
  Real xlo = 0, xhi = 1, ylo = 0, yhi = 1;

  // ADC_HD accessors: pure arithmetic, capturable by value and callable FROM A device KERNEL.
  // Without ADC_HD, geom.x_cell(i) inside a Kokkos::Cuda kernel is a __host__ call from
  // __device__: nvcc returns a GARBAGE value (often 0) on device, WITHOUT a compile or
  // runtime error. An init kernel that sets x = geom.x_cell(i) then sees x = 0 on GPU (sin(pi*0) = 0)
  // -> field silently zero (defect observed on test_condensed_schur). Geometry is a trivial
  // POD: the annotation is free and keeps the host path bit-identical.
  /// Grid spacing in x (= (xhi - xlo) / domain.nx()). ADC_HD.
  ADC_HD Real dx() const { return (xhi - xlo) / domain.nx(); }
  /// Grid spacing in y (= (yhi - ylo) / domain.ny()). ADC_HD.
  ADC_HD Real dy() const { return (yhi - ylo) / domain.ny(); }
  /// Abscissa at the CENTER of cell index i (i = 0 -> xlo + dx/2; defined for negative i). ADC_HD.
  ADC_HD Real x_cell(int i) const { return xlo + (i + Real(0.5)) * dx(); }
  /// Ordinate at the CENTER of cell index j. ADC_HD.
  ADC_HD Real y_cell(int j) const { return ylo + (j + Real(0.5)) * dy(); }

  /// Geometry refined by ratio r: SAME physical extent, refined index domain (dx -> dx/r).
  Geometry refine(int r) const { return Geometry{domain.refine(r), xlo, xhi, ylo, yhi}; }
};

// PolarGeometry: SIBLING of Geometry for a GLOBAL ANNULAR domain (r, theta).
// "Annular polar grid" effort, Phase 1 (TRANSPORT only, opt-in via adc.PolarMesh).
// The Phase-0 proto (test_polar_ring_advection) quantified that the Cartesian grid diffuses
// the RADIAL gradient of a ring in azimuthal rotation (~18%/5 turns) where the polar grid
// preserves it (ratio 73): carrying the radial direction onto a grid AXIS lifts this lock.
//
// AXIS CONVENTION (fixed):
//   - index direction 0 = RADIAL    (i runs over r, from r_min to r_max)
//   - index direction 1 = AZIMUTHAL (j runs over theta, from 0 to 2pi)
// The domain is r in [r_min, r_max] (PHYSICAL BC at r_min/r_max) x theta in [0, 2pi)
// (PERIODIC in theta). This is a global ring, NOT a local Cartesian<->polar patch
// (the hybrid interface + interpolation + boundary conservation = Phase 2, out of scope here).
//
// The cell size (dr, dtheta) is uniform in INDEX; the PHYSICAL cell size in theta is r*dtheta
// and thus grows with r (hence the 1/r metric of the divergence: cf. assemble_rhs_polar). The
// centers and faces are defined for every index (ghosts included, i negative or >= nr).
struct PolarGeometry {
  Box2D domain{};             ///< nx() = nr (radial cells), ny() = ntheta (azimuthal cells)
  Real r_min = 0, r_max = 1;  ///< physical radial bounds of the ring
  // theta covers [0, 2pi) (periodic): we store no bounds, dtheta = 2pi/ntheta.

  // Local pi (no global adc::kPi constant: it would collide with the local kPi from
  // 'using namespace adc;' in several tests). Constexpr -> no overhead.
  static constexpr Real kTwoPi = Real(2) * Real(3.14159265358979323846);

  // ADC_HD accessors (same pattern as Geometry): device-callable from a kernel without returning
  // garbage under nvcc. Pure arithmetic, host bit-identical.
  ADC_HD Real dr() const { return (r_max - r_min) / domain.nx(); }
  ADC_HD Real dtheta() const { return kTwoPi / domain.ny(); }
  /// Radius at the CENTER of radial cell i (i = 0 -> r_min + dr/2).
  ADC_HD Real r_cell(int i) const { return r_min + (i + Real(0.5)) * dr(); }
  /// Radius at radial FACE i (face between cells i-1 and i; i = 0 -> r_min, i = nr -> r_max).
  ADC_HD Real r_face(int i) const { return r_min + i * dr(); }
  /// Angle at the CENTER of azimuthal cell j (j = 0 -> dtheta/2).
  ADC_HD Real theta_cell(int j) const { return (j + Real(0.5)) * dtheta(); }
  /// Angle at azimuthal FACE j (face between cells j-1 and j).
  ADC_HD Real theta_face(int j) const { return j * dtheta(); }

  // Same annular physical extent, refined index domain (counterpart of Geometry::refine).
  PolarGeometry refine(int r) const { return PolarGeometry{domain.refine(r), r_min, r_max}; }
};

}  // namespace adc

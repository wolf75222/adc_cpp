#pragma once

#include <pops/core/state/state.hpp>        // kAuxBaseComps
#include <pops/mesh/index/box2d.hpp>        // Box2D
#include <pops/mesh/storage/fab2d.hpp>        // Fab2D
#include <pops/mesh/geometry/geometry.hpp>     // Geometry (x_cell / y_cell)
#include <pops/mesh/boundary/physical_bc.hpp>  // BCRec / BCType

/// @file
/// @brief Helpers shared by the three couplers (single-block Coupler, SystemAssembler,
///        AmrSystemCoupler) for the aux channel. Centralizes two bodies that were duplicated
///        verbatim:
///          - derive_aux_bc: aux-channel BC derived from the phi BC (periodic preserved,
///            everything else -> Foextrap).
///          - fill_bz_box:   kernel that writes B_z(x, y) at component kAuxBaseComps on
///            a box of a Fab2D, from a given geometry.
///        PURE extraction: the bodies are taken verbatim (bit-identical). What DIFFERS
///        between couplers (compile-time vs runtime guard, valid vs grown box, per-level
///        geometry, call to fill_ghosts) stays on the caller side; only the common content is here.

namespace pops {
namespace detail {

/// Aux-channel BC derived from the potential phi BC: a periodic BC stays periodic,
/// any other becomes Foextrap (order-0 extrapolation). Body taken verbatim from the
/// three couplers.
inline BCRec derive_aux_bc(const BCRec& b) {
  auto t = [](BCType x) { return x == BCType::Periodic ? BCType::Periodic : BCType::Foextrap; };
  BCRec a;
  a.xlo = t(b.xlo);
  a.xhi = t(b.xhi);
  a.ylo = t(b.ylo);
  a.yhi = t(b.yhi);
  return a;
}

/// Writes B_z(x, y) at component kAuxBaseComps on box @p box of fab @p f, sampling
/// @p bz at the cell centers of geometry @p g. Kernel common to the three couplers:
/// only the traversed box (valid or grown) and the geometry (global or per-level) differ
/// on the caller side; the loop body is bit-identical.
template <class Bz>
inline void fill_bz_box(Fab2D& f, const Box2D& box, const Geometry& g, const Bz& bz) {
  for (int j = box.lo[1]; j <= box.hi[1]; ++j)
    for (int i = box.lo[0]; i <= box.hi[0]; ++i)
      f(i, j, kAuxBaseComps) = bz(g.x_cell(i), g.y_cell(j));
}

}  // namespace detail
}  // namespace pops

/// @file
/// @brief Diagnostics extracted from the AMR couplers: mass and max drift speed (responsibility c).
///
/// Namespace-scope free functions (same reason as detail:: in coupler.hpp: GPU seam, an
/// extended lambda cannot live in a private method). amr_mass_mb goes through the reducer seam
/// (for_each_cell_reduce_sum: a real Kokkos reduction, Kokkos::Sum reassociated per tile --
/// deterministic/idempotent). amr_max_drift_speed_mb stays a host loop
/// (std::hypot not confirmed device-callable under nvcc; routing it would change the last bit). The
/// mono-box variants (...) reduce to the _mb variants (a single fab covering the domain, bit for bit). NO
/// MPI reduction here: the coupler decides whether to all_reduce according to its ownership policy.

#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/multifab.hpp>

#include <algorithm>
#include <cmath>

namespace adc {

// --- MULTI-BOX form (canonical): sum/max over the valid cells of ALL local fabs,
// WITHOUT MPI reduction (the coupler decides whether to all_reduce according to its
// ownership policy). This is the single implementation; the mono-box variants below reduce
// to it (a single fab whose box equals the domain -> bit for bit identical). This removes
// the duplication between AmrCoupler (mono-box) and AmrCouplerMP (multi-box / distributed).

// local sum of u(.,.,0) * dV over the valid cells. dV multiplied INSIDE the kernel.
/// LOCAL mass: sum of u(.,.,0) * dx * dy over the valid cells of ALL local fabs, WITHOUT
/// MPI reduction (the caller decides whether to all_reduce). Canonical multi-box form.
inline Real amr_mass_mb(const MultiFab& coarse, Real dx, Real dy) {
  const Real dV = dx * dy;
  Real M = 0;
  for (int li = 0; li < coarse.local_size(); ++li) {
    const ConstArray4 u = coarse.fab(li).const_array();
    M += for_each_cell_reduce_sum(coarse.box(li),
                                  [u, dV] ADC_HD(int i, int j) { return u(i, j, 0) * dV; });
  }
  return M;
}

// local max of |grad phi| / B0 (aux comp 1,2 = grad phi). Host loop (std::hypot not
// confirmed device: see the header). WITHOUT floor (applied by the caller).
/// LOCAL max drift speed: max of |grad phi| / B0 (aux comp 1, 2 = grad phi) over the valid cells,
/// WITHOUT floor (applied by the caller) nor MPI reduction. Host loop (std::hypot).
inline Real amr_max_drift_speed_mb(const MultiFab& aux0, Real B0) {
  device_fence();
  Real v = 0;
  for (int li = 0; li < aux0.local_size(); ++li) {
    const ConstArray4 a = aux0.fab(li).const_array();
    const Box2D b = aux0.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        v = std::max(v, std::hypot(a(i, j, 1), a(i, j, 2)) / B0);
  }
  return v;
}

// mass of component 0 on the coarse level (single box): degenerate case of
// amr_mass_mb (one fab covering the domain), bit for bit identical. dom kept for the API.
/// Mono-box mass: degenerate case of amr_mass_mb (bit for bit). @p dom is ignored (kept for the API).
inline Real amr_mass(const MultiFab& coarse, const Box2D& dom, Real dx, Real dy) {
  (void)dom;
  return amr_mass_mb(coarse, dx, dy);
}

// max drift speed on the coarse level (single box) + floor 1e-12 (CFL guard).
/// Mono-box max drift speed + floor 1e-12 (CFL guard). @p dom ignored (kept for the API).
inline Real amr_max_drift_speed(const MultiFab& aux0, const Box2D& dom, Real B0) {
  (void)dom;
  return std::max(amr_max_drift_speed_mb(aux0, B0), Real(1e-12));
}

}  // namespace adc

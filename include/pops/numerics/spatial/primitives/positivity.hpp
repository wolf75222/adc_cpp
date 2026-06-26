/// @file
/// @brief Zhang-Shu positivity limiter and Density-role resolution.
///
/// CONTRACT: the OPT-IN positivity layer of the spatial operator. Inactive (bit-identical)
/// unless pos_floor > 0.
///   - zhang_shu_scale: order-1 local fallback when a reconstructed face state drops below floor.
///   - positivity_comp: resolves the Density-role component once per operator call (HOST).
///
/// reconstruct_pp (reconstruction + this limiter) lives in face_flux.hpp, which depends on this
/// module. POINTWISE device-clean (POPS_HD) for the limiter; positivity_comp is HOST-only.

#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>  // VariableRole, VariableSet: Density-role resolution
#include <pops/mesh/storage/fab2d.hpp>      // ConstArray4: face-state source cell read

#include <stdexcept>  // positivity_comp: model without Density role -> clear error

namespace pops {

/// zhang_shu_scale: POSITIVITY limiter on a reconstructed face state -- LOCAL ORDER-1 FALLBACK
/// (vacuum-robust variant of the Zhang & Shu scaling, JCP 2010).
///
/// If component @p pos_comp (Density role) of the face state @p s falls below @p floor, the WHOLE
/// face state is replaced by the average of its SOURCE cell u(i,j,.) (locally zero slope).
/// WHY not the paper's colinear theta-scaling (s <- ubar + theta (s - ubar), theta such that
/// rho_face = floor): in CONSERVATIVE variables at the edge of a QUASI-VACUUM (a ~1e-6 background under a
/// ~1e6 top-hat contrast), it sets rho_face = floor while leaving a face momentum O(average) -> the
/// face VELOCITY v = m/rho diverges (~1e6) -> the Rusanov wave speed blows up whereas dt was chosen
/// BEFORE on the cell velocities -> immediate blow-up (measured: NaN within a couple of steps,
/// independent of the floor value). The paper couples its limiter to the recomputed CFL bound; here the fallback
/// to the average bounds the face velocity by CONSTRUCTION (v_face = v_cell), stays conservative
/// (the average is not touched), positive as soon as the average is, and degrades the order only on
/// the offending faces (WENO5 intact everywhere else).
/// Inactive if floor <= 0 (bit-identical path) or if the face is already >= floor. Motivation: WENO5
/// undershoots at the top-hat jump with 1e6 contrast -> negative face rho -> 1/rho and the Lorentz
/// source detonate -> NaN (positivity-fallback provenance: docs/validation/HEADER_PROVENANCE.md).
/// POINTWISE device-clean function. POPS_HD.
template <class Model>
POPS_HD inline void zhang_shu_scale(typename Model::State& s, const ConstArray4& u, int i, int j,
                                   Real floor, int pos_comp) {
  if (!(floor > Real(0)))
    return;  // strict opt-in: floor <= 0 -> no effect
  if (!(s[pos_comp] < floor))
    return;  // face already above the floor
  for (int c = 0; c < Model::n_vars; ++c)
    s[c] = u(i, j, c);  // order-1 fallback: face = average
}

namespace detail {
/// Component of the Density role for the positivity limiter (HOST, resolved once per spatial
/// operator call, never per cell). pos_floor <= 0 -> 0 (never read, the scaling is short-circuited
/// in zhang_shu_scale). A model without VariableSet introspection or without a Density role cannot
/// request positivity: clear error rather than a silent scaling of an arbitrary component.
template <class Model>
inline int positivity_comp(Real pos_floor) {
  if (!(pos_floor > Real(0)))
    return 0;
  if constexpr (requires { Model::conservative_vars(); }) {
    const int c = Model::conservative_vars().index_of(VariableRole::Density);
    if (c >= 0)
      return c;
    throw std::runtime_error(
        "positivity_floor > 0: the model does not expose the Density role (scaling target)");
  } else {
    throw std::runtime_error(
        "positivity_floor > 0: model without VariableSet introspection (conservative_vars)");
  }
}
}  // namespace detail

}  // namespace pops

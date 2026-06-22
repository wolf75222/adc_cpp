/// @file
/// @brief Cartesian spatial operator: assembles R(U, aux) = -div F + S over the cells of a level.
///
/// This is the "PDE -> ODE system" arrow of the method of lines. The time integrator
/// (time/) only knows R; it is unaware of the geometry and the reconstruction scheme.
///
/// UMBRELLA HEADER (ADC-328): the operator is split into focused modules under numerics/spatial/.
/// This header keeps the historical include path working (it re-exports every symbol below) so
/// existing `#include <adc/numerics/spatial_operator.hpp>` users are unaffected and the numerics
/// are bit-identical. Prefer including the specific module for new code.
///
/// Modules (one-way dependency DAG, bottom to top):
///   - spatial/state_access.hpp     DiffusiveModel, SourceFreeModel, load_state, load_aux.
///   - spatial/positivity.hpp       zhang_shu_scale, detail::positivity_comp.
///   - spatial/face_flux.hpp        reconstruct, reconstruct_pp, require_reconstruction_ghosts,
///                                  xface_box / yface_box, rusanov_flux, compute_face_fluxes.
///   - spatial/wave_speed.hpp       max_wave_speed_mf and step-bound reductions, the hotspot
///                                  diagnostic, fill_wave_speed_cache.
///   - spatial/cartesian_operator.hpp  assemble_rhs, assemble_rhs_hll_cached.
///   - spatial/masked_operator.hpp     assemble_rhs_masked.
///
/// INVARIANT: the Cartesian operator is STRICTLY UNTOUCHED by the polar operator
/// (spatial_operator_polar.hpp); a run on a Cartesian mesh is bit-identical.

#pragma once

#include <adc/numerics/spatial/primitives/state_access.hpp>
#include <adc/numerics/spatial/primitives/positivity.hpp>
#include <adc/numerics/spatial/primitives/face_flux.hpp>
#include <adc/numerics/spatial/primitives/wave_speed.hpp>
#include <adc/numerics/spatial/operators/cartesian_operator.hpp>
#include <adc/numerics/spatial/operators/masked_operator.hpp>

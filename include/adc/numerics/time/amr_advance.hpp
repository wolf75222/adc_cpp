#pragma once

#include <adc/numerics/time/amr_subcycling.hpp>

/// @file
/// @brief Unified production facade for the AMR time engine: LevelHierarchy type (the AMR
///        hierarchy as a named object), OwnershipPolicy alias, and the advance_amr entry point.
///
/// Layer: `include/adc/numerics/time`.
/// Role: expose ONE production entry, advance_amr, that delegates to the N-level multi-patch
///        engine (detail::amr_step_multilevel_multipatch). Two forms: "pieces" (the coupler
///        passes levels/base_dom/... directly) and LevelHierarchy (aggregate object).
/// Contract: advance_amr advances the hierarchy by ONE time step dt; selectors fixed when the
///           block is added (coarse_replicated, recon_prim, imex, time_method) forwarded as-is
///           to the engine.
///
/// Invariants:
/// - coarse_replicated=true (default) reproduces the historical behavior; without forwarding,
///   a de-replicated coarse level would wrongly switch back to replicated;
/// - recon_prim selects the primitive reconstruction (cf. assemble_rhs) instead of conservative;
///   false (default) = strictly bit-identical;
/// - kSsprk3 requires imex == false (rejected otherwise, enforced on the engine side).

namespace adc {

// --- Unified AMR engine (review, point 5) ---
// The AMR hierarchy as a named OBJECT that the engine advances, rather than a family of
// amr_step_* functions whose case (2/N levels, mono/multi-box) is encoded in the NAME.
// Unified entry: advance_amr(m, LevelHierarchy&, dt), faithful facade of the N-level
// multi-patch engine (verified at 2 AND 3 levels, maxdiff = 0, by test_advance_amr). The ROLES
// from the review, their current backing (named types, or remaining code to promote):
//   OwnershipPolicy     = DistributionMapping (who owns which patch)            -> alias below
//   AmrLevel            = AmrLevelMP (box + data + aux + dx of a level)
//   PatchRange          = NAMED TYPE: coarse footprint [I0..I1]x[J0..J1] of a fine patch
//                         (ratio 2), shared by average_down, coverage and registers
//   CoarseFineGhost     = NAMED TYPE/HELPER: fill_cf_ghost_cell (space+time interp per ghost
//                         cell), shared by the three mf_fill_fine_ghosts_*
//   CoarseFineInterface = NAMED TYPE: coverage (CoverageMask) + bordering reflux routing
//                         (route_reflux), shared by subcycle_level_mp and amr_step_2level_multipatch
//   FluxRegister        = NAMED TYPE: avg/ref buffers at global index + all_reduce_sum_inplace
//   SubcyclingSchedule  = NAMED TYPE: Berger-Oliger cadence (ratio r, dt/r, frac s/r) per level
//   RegridPolicy        = amr_regrid_finest (Berger-Rigoutsos), on the coupler side
using OwnershipPolicy = DistributionMapping;

struct LevelHierarchy {
  std::vector<AmrLevelMP> levels;    // level 0 = coarse, levels >0 = fine patches
  Box2D base_dom;                    // footprint of the base level
  Periodicity base_per{true, true};  // BC of the base domain
  bool coarse_replicated = true;     // level 0 replicated (true) or multi-box distributed (false)
  bool recon_prim = false;           // primitive reconstruction (cf. compute_face_fluxes)
  bool imex = false;  // stiff implicit source (backward_euler) instead of forward Euler
  // NEWTON OPTIONS of the IMEX step (default {} = historical constants 2 iters / 1e-7 -> bit-identical).
  // Honored only when imex==true; forwarded to backward_euler_source by mf_apply_source_treatment.
  NewtonOptions newton_options{};
  // TIME METHOD: kEuler (default, forward Euler per substep, bit-identical to the historical) or
  // kSsprk3 (SSPRK3 order 3 + per-stage reflux). kSsprk3 requires imex == false (rejected otherwise, cf. engine).
  AmrTimeMethod time_method = AmrTimeMethod::kEuler;
  // Zhang-Shu positivity floor (ADC-259): Density-role face-state + C/F-ghost-mean floor on the AMR
  // transport. <= 0 (default) -> inactive, bit-identical to the historical path.
  Real pos_floor = Real(0);
};

// Unified production entry: advances the hierarchy by one time step dt. "pieces" form (the coupler
// owns its own stack and passes the vectors directly) and LevelHierarchy form. The gate FORWARDS
// coarse_replicated to the engine; without it a de-replicated coarse level would switch back to
// replicated (mf_find_box instead of parallel_copy). coarse_replicated=true (default) -> identical
// to the historical behavior. recon_prim selects the primitive reconstruction (variables
// (rho, u, p)) instead of conservative: same parameter as assemble_rhs, fixed when the block is added;
// false (default) -> conservative, strictly bit-identical to the historical.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_amr(const Model& m, std::vector<AmrLevelMP>& levels, const Box2D& base_dom, Real dt,
                 Periodicity base_per = Periodicity{true, true}, bool coarse_replicated = true,
                 bool recon_prim = false, bool imex = false, const NewtonOptions& nopts = {},
                 AmrTimeMethod tmethod = AmrTimeMethod::kEuler, Real pos_floor = Real(0)) {
  detail::amr_step_multilevel_multipatch<Limiter, NumericalFlux>(m, levels, base_dom, dt, base_per,
                                                                 coarse_replicated, recon_prim,
                                                                 imex, nopts, tmethod, pos_floor);
}

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_amr(const Model& m, LevelHierarchy& h, Real dt) {
  advance_amr<Limiter, NumericalFlux>(m, h.levels, h.base_dom, dt, h.base_per, h.coarse_replicated,
                                      h.recon_prim, h.imex, h.newton_options, h.time_method,
                                      h.pos_floor);
}

}  // namespace adc

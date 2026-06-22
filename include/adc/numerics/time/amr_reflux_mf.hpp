#pragma once

#include <adc/numerics/time/amr_flux_helpers.hpp>  // mf_advance_faces, mf_apply_source*, mf_average_down, fill_cf_ghost_cell, mf_fill_fine_ghosts_t
#include <adc/numerics/time/reference/amr_level.hpp>  // detail::AmrLevelMF, amr_step_2level_mf, subcycle_level_mf, amr_step_multilevel_mf
#include <adc/numerics/time/amr_patch_range.hpp>  // PatchRange, FluxRegister, CoverageMask, SubcyclingSchedule, CoarseFineInterface, fill_periodic_local, mf_fill_fine_ghosts_multi, mf_average_down_multi
#include <adc/numerics/time/amr_subcycling.hpp>  // AmrLevelMP, RegMP, mf_find_box, coarsen_grown, mf_fill_fine_ghosts_mb, mf_average_down_mb, amr_step_2level_multipatch, detail::subcycle_level_mp, detail::amr_step_multilevel_multipatch
#include <adc/numerics/time/amr_advance.hpp>     // OwnershipPolicy, LevelHierarchy, advance_amr

/// @file
/// @brief Umbrella for the AMR MultiFab stack: includes the numerics/time sub-headers in
///        dependency order (flux_helpers -> level -> patch_range -> subcycling -> advance).
///
/// Layer: `include/adc/numerics/time`.
/// Role: single entry point for the AMR MultiFab/multi-patch stack. Every existing includer of
///        this header keeps compiling without modification after the split into sub-headers;
///        the public production API stays advance_amr (see amr_advance.hpp).

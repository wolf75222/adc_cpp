#pragma once

/// @file
/// @brief POPS_COLD_FN: opt out the COLD host factories from the optimizer (ADC-337, P1-B).
///
/// The string->closure wiring (dispatch_*, make_block*, build_*_for, resolve_implicit_components)
/// runs ONCE per block at construction time, on the host, never inside the time loop. Compiling it
/// at -O3 is pure waste: the backend inlines and optimizes the entire CompositeModel instantiation
/// tree into one giant factory function -- the dominant slice of the heavy translation units'
/// -O3 cost (cf. docs/BUILD_PROFILING.md). Marking these factories no-optimize keeps that wiring at
/// -O0 while the HOT kernels (BlockRhsEval / Advance* / take_step / Kokkos for_each_cell) stay -O3:
/// they are SEPARATE functions, reached through std::function closures, so the factory's -O0 does
/// not de-optimize them. No `-ffast-math` anywhere, and -O0 vs -O3 never changes IEEE results, so
/// the numerics stay bit-identical (the dmax==0 parity suite guards this).
///
/// Placement: between the template clause and the return type, e.g.
///   template <class Visitor> POPS_COLD_FN void dispatch_transport(...) { ... }
///
/// NEVER apply it to a hot kernel or to a function that runs per cell / per step.
#if defined(__clang__)
#define POPS_COLD_FN __attribute__((optnone))
#elif defined(__GNUC__)
#define POPS_COLD_FN __attribute__((optimize("O0")))
#else
#define POPS_COLD_FN
#endif

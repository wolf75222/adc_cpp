#pragma once

#include <stdexcept>
#include <string>

/// @file
/// @brief SINGLE registry of spatial scheme tags (limiters + Riemann fluxes): shared source of
///        truth for ALL dispatches (System make_block, AMR dispatch_amr_block /
///        dispatch_amr_compiled, polar make_block_polar).
///
/// Before this header each dispatch carried its OWN tag table (limiters x fluxes) and its OWN
/// error message; the tables diverged silently (a weno5 case forgotten on an hllc/roe AMR branch
/// gave "unknown limiter" where System accepted it). Here: ONE kLimiters / kRiemanns table,
/// shared VALIDATION functions (same messages, or clearer) and limiter_n_ghost (halo width). The
/// call-sites keep their template if/else dispatch (the Limiter / Flux types are COMPILE-TIME, not
/// tabulable without a heavy X-macro) BUT validate HERE FIRST -> centralized rejection; the final
/// throw of the if/else becomes a "registry/dispatch inconsistency" guard (defense in depth, never
/// reached in practice).
///
/// This header is deliberately LIGHT (no numerical dependency): it carries only strings and
/// integers, not the Limiter / Flux types. It thus stays included early and cost-free. The
/// capability NEEDS of the fluxes (hll: signed waves; hllc/roe: 2D Euler structure or model
/// capability) are DOCUMENTED in kRiemanns but the real guard stays an `if constexpr` PER MODEL at
/// the call-site (capabilities depend on the Model type, unavailable here).

namespace adc {

/// Tag of a reconstruction LIMITER: user-facing name + halo width (n_ghost) required by its
/// stencil. MIRROR of the::n_ghost constants of the types (NoSlope=1, Minmod=2, VanLeer=2,
/// Weno5=3, cf. numerics/reconstruction.hpp): a static_assert on the block_builder.hpp side (which
/// sees both) locks the absence of drift between this table and the real types.
struct LimiterTag {
  const char* name;
  int n_ghost;
};

/// SINGLE SOURCE of the wired limiters (order = display priority: none < minmod < vanleer < weno5).
inline constexpr LimiterTag kLimiters[] = {
    {"none", 1}, {"minmod", 2}, {"vanleer", 2}, {"weno5", 3}};

/// Tag of a Riemann FLUX: name + model CAPABILITY needs (DOCUMENTARY: the real guard is an
/// `if constexpr` per model at the call-site -- these flags do NOT drive the dispatch, they
/// document the contract and serve the tests). polar_ok = wired in polar geometry.
///   - needs_wave_speeds: hll requires model.wave_speeds (signed waves);
///   - needs_hllc_struct: hllc requires HasHLLCStructure (or canonical 2D Euler path);
///   - needs_roe_diss: roe  requires HasRoeDissipation (or canonical 2D Euler path).
struct RiemannTag {
  const char* name;
  bool needs_wave_speeds;
  bool needs_hllc_struct;
  bool needs_roe_diss;
  bool polar_ok;
};

/// SINGLE SOURCE of the wired Riemann fluxes (order = message "(rusanov|hll|hllc|roe)").
inline constexpr RiemannTag kRiemanns[] = {{"rusanov", false, false, false, true},
                                           {"hll", true, false, false, true},
                                           {"hllc", false, true, false, false},
                                           {"roe", false, false, true, false}};

/// Halo width required by the limiter @p lim (source: kLimiters). Default 2 (MUSCL) for an unknown
/// limiter: this is the HISTORICAL allocation of block_n_ghost -> bit-identical (the make_block
/// dispatch will throw anyway on an invalid limiter). Used to size the state MultiFab of a block
/// (wide WENO5 stencil: 5 points, 3 ghosts).
inline int limiter_n_ghost(const std::string& lim) {
  for (const LimiterTag& t : kLimiters)
    if (lim == t.name) return t.n_ghost;
  return 2;  // MUSCL fallback (historical allocation)
}

namespace detail {
/// COMPILE-TIME C string equality (no constexpr <cstring> guaranteed everywhere).
constexpr bool ct_str_eq(const char* a, const char* b) {
  while (*a != '\0' && *b != '\0') {
    if (*a != *b) return false;
    ++a;
    ++b;
  }
  return *a == *b;
}
}  // namespace detail

/// COMPILE-TIME variant of limiter_n_ghost (const char* literal): -1 if unknown. Used ONLY by the
/// non-drift static_assert on the block_builder.hpp side (this TU sees BOTH kLimiters AND the
/// ::n_ghost constants of the types) -- guards that the table never diverges from the real types.
constexpr int limiter_n_ghost_ct(const char* lim) {
  for (const LimiterTag& t : kLimiters)
    if (detail::ct_str_eq(lim, t.name)) return t.n_ghost;
  return -1;
}

/// Validates a LIMITER tag against kLimiters. Throws if unknown, with the HISTORICAL message
/// "<ctx>: unknown limiter '<lim>'" (some tests grep "unknown limiter"). @p ctx = call-site prefix
/// ("System", "add_block(AmrSystem, multi-blocks)", "add_compiled_model(AmrSystem)",
/// "System (polar)") -> message STRICTLY identical to the old inline throw of each dispatch.
inline void validate_limiter(const std::string& lim, const char* ctx = "System") {
  for (const LimiterTag& t : kLimiters)
    if (lim == t.name) return;
  throw std::runtime_error(std::string(ctx) + ": unknown limiter '" + lim + "'");
}

/// Validates a Riemann FLUX tag against kRiemanns. @p polar: annular geometry (rusanov and hll are
/// wired there). Throws if unknown (cartesian) or not wired in polar, with the HISTORICAL messages
/// (some tests grep "Riemann flux", "rusanov", "unsupported"). Does NOT validate the model
/// capabilities (hll/hllc/roe on a transport without signed waves / without pressure): these guards
/// stay `if constexpr` PER MODEL at the call-site, with their "requires ..." messages unchanged.
inline void validate_riemann(const std::string& riem, bool polar = false,
                             const char* ctx = "System") {
  if (polar) {
    // Polar: wired fluxes = those of kRiemanns with polar_ok (rusanov + hll since the audit
    // settlement; hll keeps its model.wave_speeds capability gate at the call-site). HLLC/Roe and
    // unknown tags -> single polar message.
    for (const RiemannTag& t : kRiemanns)
      if (riem == t.name && t.polar_ok) return;
    throw std::runtime_error(
        std::string(ctx) + ": Riemann flux '" + riem +
        "' unsupported (polar -> 'rusanov' | 'hll' (signed wave speeds); HLLC/Roe "
        "assume n_vars==4 (Euler with energy), not applicable to scalar ExB or the polar "
        "isothermal fluid)");
  }
  for (const RiemannTag& t : kRiemanns)
    if (riem == t.name) return;
  throw std::runtime_error(std::string(ctx) + ": unknown Riemann flux '" + riem +
                           "' (rusanov|hll|hllc|roe)");
}

/// DEFENSE-IN-DEPTH guard: reached only if a VALID tag (already accepted by validate_*) is routed
/// by NO branch of the if/else dispatch -- this is an inconsistency between the registry
/// (kLimiters/kRiemanns) and the dispatch, hence a programming bug, not a user input. Replaces the
/// old final `throw "unknown limiter" / "unknown Riemann flux"`, now unreachable since the
/// centralized validation precedes the dispatch. @p kind = "limiter" or "flux".
[[noreturn]] inline void throw_registry_dispatch_mismatch(const char* ctx, const char* kind,
                                                          const std::string& tag) {
  throw std::runtime_error(std::string(ctx) + ": registry/dispatch inconsistency -- " + kind + " '" +
                           tag + "' valid but not routed (add the case to the dispatch)");
}

}  // namespace adc

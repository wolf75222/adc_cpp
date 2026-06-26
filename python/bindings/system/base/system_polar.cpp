// ADC-335 (P0-B): block-build seam for the POLAR (ring) path. Instantiates ONLY the polar leaves
// (assemble_rhs_polar over the two polar transports, scalar ExBVelocityPolar + fluid IsothermalFluxPolar)
// in its own translation unit. VERBATIM move of the dispatch_model_polar visitor body that used to live
// in System::add_block; the polar makers read the System aux through the @p aux pointer (was &P->aux).
// IMEX is rejected on the ring by add_block before this is called.
#include <pops/runtime/builders/block/block_seam.hpp>

namespace pops::detail {

BuiltBlock build_block_polar(const ModelSpec& model, const std::string& limiter,
                             const std::string& riemann, const PolarGridContext& pctx,
                             bool recon_prim, const std::string& method, Real positivity_floor,
                             const MultiFab* aux) {
  BuiltBlock out;
  dispatch_model_polar(model, [&](auto m) {
    using M = decltype(m);
    out.ncomp = M::n_vars;
    out.cons_vs = M::conservative_vars();
    out.prim_vs = M::primitive_vars();
    // ADC-291: aux channel width the model READS (canonical extras B_z/T_e AND model-named extra[k]).
    // The caller (System::add_block, polar branch) widens the shared aux to this via ensure_aux_width,
    // exactly like the Cartesian path. Without it a polar model with n_aux>3 read past the aux fab
    // (load_aux<aux_comps<M>> on a 3-wide channel) -- a silent out-of-bounds (#51-class).
    out.aux_width = aux_comps<M>();
    // wall_radial = true: solid wall at both radial edges (no-penetration) -> zero radial flux at
    // r_min / r_max -> mass Sum n r dr dtheta conserved TO MACHINE precision (diocotron ring bounded by
    // two conducting walls). This is the BC that makes the coupled step conservative.
    out.clo = make_block_polar(m, limiter, riemann, pctx, recon_prim, method, /*wall_radial=*/true,
                               positivity_floor);
    // POLAR StabilityPolicy (audit wave 3): same policy as the Cartesian -- stability lambda* (trait)
    // otherwise max_wave_speed; source/admissible-step bounds if declared, EMPTY closures otherwise
    // (historical step policy, bit-identical).
    out.max_speed = make_cfl_speed_polar(m, aux);
    out.src_freq = make_source_frequency_polar(m, aux);
    out.stab_dt = make_stability_dt_polar(m, aux);
    out.add_poisson_rhs = make_poisson_rhs_polar(m);
    auto conv = make_cell_convert(m);
    out.prim_to_cons = std::move(conv.first);
    out.cons_to_prim = std::move(conv.second);
  });
  return out;
}

}  // namespace pops::detail

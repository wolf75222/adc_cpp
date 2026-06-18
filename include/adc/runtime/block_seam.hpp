#pragma once

#include <adc/core/variables.hpp>           // VariableSet (block descriptor carried in BuiltBlock)
#include <adc/runtime/block_builder.hpp>    // make_block + makers + BlockClosures + NewtonOptions/Report
#include <adc/runtime/block_builder_polar.hpp>  // make_block_polar + polar makers + PolarGridContext
#include <adc/runtime/model_factory.hpp>    // dispatch_model_for + resolve_implicit_components + ModelSpec

#include <functional>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief Per-transport block-build seam (ADC-335 / P0-B).
///
/// System::add_block used to call detail::dispatch_model(spec, lambda) directly; the lambda body
/// instantiated make_block / the makers for EVERY transport in ONE translation unit (python/system.cpp),
/// which is the ~1700-leaf combinatorial product whose -O3 backend pass dominates the build. This header
/// moves that lambda body behind a fixed-signature, NON-template, hidden-visibility free function per
/// transport (build_block_exb / _isothermal / _compressible for the Cartesian path, build_block_polar for
/// the ring). Each is defined in its own .cpp (python/system_<transport>.cpp), so the per-transport leaves
/// compile in parallel. system.cpp keeps only the string validation + a thin transport if/else.
///
/// The seam carries NO template surface: the block boundary is already type-erased (BlockClosures is a POD
/// of std::function, the makers return std::function), so BuiltBlock is a plain bundle. The build_block_for
/// helper is the VERBATIM Cartesian visitor body (so the reachable instantiation set is unchanged and the
/// production binary stays byte-identical); it is instantiated only inside the per-transport TU that names
/// its TR. The seam functions are INTERNAL (no ADC_EXPORT): the module's exported symbol table is unchanged.

namespace adc::detail {

/// Everything System::add_block reads back from the (former) dispatch_model visitor. Mirrors the local
/// variables the visitor wrote by reference; aux_width replaces the in-visitor P->ensure_aux_width call
/// (re-applied host-side by add_block, address-stable, so order vs make_block does not matter).
struct BuiltBlock {
  int ncomp = 1;
  VariableSet cons_vs, prim_vs;
  BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  std::function<Real(const MultiFab&)> src_freq, stab_dt;  // optional step bounds (model traits)
  std::function<void(const double*, double*)> prim_to_cons, cons_to_prim;  // System::CellConvert
  int aux_width = 0;  // aux_comps<Model>() (Cartesian); unused on the polar path (no ensure_aux_width)
};

/// The non-model inputs of a Cartesian block build (a thin bundle so the seam signature stays fixed).
/// Held by value: these are cheap setup-time copies (GridContext is already value-semantic and copied
/// per add_block; NewtonOptions is a POD), which avoids any lifetime coupling to add_block locals.
struct BlockBuildArgs {
  std::string name;
  std::string limiter;
  std::string riemann;
  GridContext ctx;
  bool imex;
  bool recon_prim;
  std::string method;
  std::vector<std::string> implicit_vars;
  std::vector<std::string> implicit_roles;
  NewtonOptions nopts;
  NewtonReport* nreport;  // non-owning (report lives in System::Impl::newton_reports_)
  Real positivity_floor;
  bool wave_speed_cache;
};

/// VERBATIM Cartesian visitor body of the former dispatch_model lambda (system.cpp), with the transport
/// brick @p tr already chosen. Instantiated ONLY in the per-transport TU that calls it (its make_block /
/// makers leaves land there, not in system.cpp). Behavior identical to add_block before ADC-335.
template <class TR>
BuiltBlock build_block_for(TR tr, const ModelSpec& model, const BlockBuildArgs& a) {
  BuiltBlock out;
  dispatch_model_for(model, std::move(tr), [&](auto m) {
    using M = decltype(m);
    out.ncomp = M::n_vars;
    out.cons_vs = M::conservative_vars();  // names + physical ROLES (single source of truth)
    out.prim_vs = M::primitive_vars();
    // AUX WIDTH of the composed model (n_aux > 3 for a magnetized brick reading B_z): returned so
    // add_block widens the SHARED channel host-side (ensure_aux_width preserves the aux ADDRESS captured
    // by the closures, so applying it after make_block is equivalent).
    out.aux_width = aux_comps<M>();
    const std::vector<int> impl_components =
        resolve_implicit_components(a.name, out.cons_vs, a.implicit_vars, a.implicit_roles);
    out.clo = make_block(m, a.limiter, a.riemann, a.ctx, a.imex, a.recon_prim, a.method, impl_components,
                         a.nopts, a.nreport, a.positivity_floor, a.wave_speed_cache);
    out.max_speed = make_max_speed(m, a.ctx);
    out.add_poisson_rhs = make_poisson_rhs(m);
    out.src_freq = make_source_frequency(m, a.ctx);
    out.stab_dt = make_stability_dt(m, a.ctx);
    auto conv = make_cell_convert(m);
    out.prim_to_cons = std::move(conv.first);
    out.cons_to_prim = std::move(conv.second);
  });
  return out;
}

// Per-transport seam functions (defined in python/system_<transport>.cpp). The TR construction matches
// dispatch_transport's branches VERBATIM (ExBVelocity{B0} / CompressibleFlux{gamma} / IsothermalFlux{cs2}).
BuiltBlock build_block_exb(const ModelSpec& model, const BlockBuildArgs& a);
BuiltBlock build_block_compressible(const ModelSpec& model, const BlockBuildArgs& a);
BuiltBlock build_block_isothermal(const ModelSpec& model, const BlockBuildArgs& a);

// Polar (ring) seam: VERBATIM polar visitor body (make_block_polar + polar makers). IMEX is rejected on
// the ring by add_block before this is called. @p aux is &System::Impl::aux (the polar makers read it).
BuiltBlock build_block_polar(const ModelSpec& model, const std::string& limiter,
                             const std::string& riemann, const PolarGridContext& pctx, bool recon_prim,
                             const std::string& method, Real positivity_floor, const MultiFab* aux);

}  // namespace adc::detail

#pragma once

#include <adc/core/variables.hpp>          // VariableSet/VariableRole/role_from_name (resolve mask)
#include <adc/runtime/amr_dsl_block.hpp>   // dispatch_amr_block / dispatch_amr_compiled + AmrBuildParams
#include <adc/runtime/amr_runtime.hpp>     // AmrRuntimeBlock + AmrTimeMethod
#include <adc/runtime/model_factory.hpp>   // dispatch_model_for + compiled bricks + ModelSpec

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief Per-transport block-build seam for AmrSystem (ADC-335 / P0-B), mirror of block_seam.hpp.
///
/// AmrSystem::Impl had TWO inline `detail::dispatch_model(b.spec, lambda)` sites -- the multi-block
/// build_multi (-> dispatch_amr_block -> AmrRuntimeBlock) and the single-block lazy build
/// (-> dispatch_amr_compiled -> AmrCompiledHooks) -- each of which instantiated the full AMR dispatch
/// product (all transports x flux x limiter) in one TU (python/amr_system.cpp). This header moves each
/// lambda body behind a fixed-signature, hidden-visibility, non-template free function per transport,
/// defined in its own .cpp (python/amr_{block,compiled}_<transport>.cpp), so the per-transport leaves
/// compile in parallel. The boundaries are already type-erased (AmrRuntimeBlock / AmrCompiledHooks), so
/// the seam carries no template surface. The lambda bodies move VERBATIM -> byte-identical.

namespace adc::detail {

/// AMR partial-IMEX-mask resolution, moved out of amr_system.cpp's anonymous namespace (ADC-335) so the
/// per-transport seam TUs share one definition. Distinct from the System resolve_implicit_components
/// (model_factory.hpp): the AmrSystem error wording differs, kept VERBATIM here.
inline std::vector<int> resolve_implicit_components_amr(const std::string& block, const VariableSet& cons,
                                                        const std::vector<std::string>& names,
                                                        const std::vector<std::string>& roles) {
  std::vector<int> out;
  auto push_unique = [&out](int c) {
    if (std::find(out.begin(), out.end(), c) == out.end()) out.push_back(c);
  };
  for (const std::string& nm : names) {
    int idx = -1;
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == nm) { idx = i; break; }
    if (idx < 0) {
      std::string have;
      for (std::size_t i = 0; i < cons.names.size(); ++i) {
        if (i) have += ", ";
        have += cons.names[i];
      }
      throw std::runtime_error("AmrSystem::add_block : implicit_vars : variable '" + nm +
                               "' missing from block '" + block + "' (conserved variables : " + have + ")");
    }
    push_unique(idx);
  }
  for (const std::string& rn : roles) {
    const VariableRole role = role_from_name(rn);
    const int idx = cons.index_of(role);
    if (role == VariableRole::Custom || idx < 0)
      throw std::runtime_error("AmrSystem::add_block : implicit_roles : role '" + rn +
                               "' missing from block '" + block + "' (the block does not provide this role)");
    push_unique(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Non-model inputs of a MULTI-block AMR build (the fields the build_multi visitor read off the BlockSpec).
/// Held by value (cheap setup-time copies); @p state points at the BlockSpec's owned vector (non-owning).
struct AmrBlockBuildArgs {
  ModelSpec spec;
  std::string name;
  std::string limiter;
  std::string riemann;
  std::vector<double> density;
  bool has_density;
  double gamma;
  int substeps;
  bool recon_prim;
  bool imex;
  int stride;
  std::vector<std::string> implicit_vars;
  std::vector<std::string> implicit_roles;
  NewtonOptions newton;
  const std::vector<double>* state;  // &BlockSpec::state when has_state, else nullptr (non-owning)
  bool newton_diagnostics;
  int time_method;  // 1 == ssprk3 -> AmrTimeMethod::kSsprk3, else kEuler (build_multi mapping)
  double pos_floor;
};

/// VERBATIM build_multi visitor body with the transport pinned: resolve the partial IMEX mask against the
/// concrete model, map the temporal method, and produce the type-erased AmrRuntimeBlock via dispatch_amr_block.
template <class TR>
AmrRuntimeBlock build_amr_block_for(TR tr, const AmrBlockBuildArgs& a, const SharedAmrLayout& S) {
  AmrRuntimeBlock out;
  dispatch_model_for(a.spec, std::move(tr), [&](auto m) {
    using M = decltype(m);
    const std::vector<int> impl_components =
        a.imex ? resolve_implicit_components_amr(a.name, M::conservative_vars(), a.implicit_vars,
                                                 a.implicit_roles)
               : std::vector<int>{};
    const AmrTimeMethod tmethod = a.time_method == 1 ? AmrTimeMethod::kSsprk3 : AmrTimeMethod::kEuler;
    out = dispatch_amr_block(m, a.limiter, a.riemann, S, a.name, a.density, a.has_density, a.gamma,
                             a.substeps, a.recon_prim, a.imex, a.stride, impl_components, a.newton,
                             a.state, a.newton_diagnostics, tmethod, a.pos_floor);
  });
  return out;
}

/// VERBATIM single-block visitor body with the transport pinned: produce the type-erased AmrCompiledHooks
/// via dispatch_amr_compiled. @p bp (AmrBuildParams) already bundles every single-block parameter.
template <class TR>
AmrCompiledHooks build_amr_compiled_for(TR tr, const ModelSpec& spec, const std::string& limiter,
                                        const std::string& riemann, const AmrBuildParams& bp) {
  AmrCompiledHooks out;
  dispatch_model_for(spec, std::move(tr), [&](auto m) {
    out = dispatch_amr_compiled(m, limiter, riemann, bp);
  });
  return out;
}

// Per-transport seam functions (defined in python/amr_block_<transport>.cpp / amr_compiled_<transport>.cpp).
// TR construction matches dispatch_transport VERBATIM (ExBVelocity{B0}/CompressibleFlux{gamma}/IsothermalFlux{cs2}).
AmrRuntimeBlock build_amr_block_exb(const AmrBlockBuildArgs& a, const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_isothermal(const AmrBlockBuildArgs& a, const SharedAmrLayout& S);
AmrRuntimeBlock build_amr_block_compressible(const AmrBlockBuildArgs& a, const SharedAmrLayout& S);

AmrCompiledHooks build_amr_compiled_exb(const ModelSpec& spec, const std::string& limiter,
                                        const std::string& riemann, const AmrBuildParams& bp);
AmrCompiledHooks build_amr_compiled_isothermal(const ModelSpec& spec, const std::string& limiter,
                                               const std::string& riemann, const AmrBuildParams& bp);
AmrCompiledHooks build_amr_compiled_compressible(const ModelSpec& spec, const std::string& limiter,
                                                 const std::string& riemann, const AmrBuildParams& bp);

}  // namespace adc::detail

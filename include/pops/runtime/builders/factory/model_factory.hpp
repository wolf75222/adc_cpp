#pragma once

#include <pops/core/foundation/cold.hpp>  // POPS_COLD_FN: COLD-factory no-optimize attribute (ADC-337)
#include <pops/core/state/variables.hpp>  // VariableSet/VariableRole/role_from_name/roles_csv (resolve_implicit_components)
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/dynamic/model_registry.hpp>  // kTransports/kSources/kElliptics: builtin-brick tag registry (ADC-331)
#include <pops/runtime/config/model_spec.hpp>

#include <algorithm>  // std::find, std::sort (resolve_implicit_components)
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief Assemble a CompositeModel from a ModelSpec (bricks + parameters).
///
/// The core knows only generic BRICKS; a scenario is a composition, named on the
/// application side (adc_cases). dispatch_model(spec, visitor) builds the
/// CompositeModel<Hyperbolic, Source, Elliptic> designated by the spec and calls visitor(model).
/// Invalid combinations (fluid source on a scalar transport) are rejected.

namespace pops::detail {

/// Completeness contract of a ModelSpec (ADC-290): `transport` and `elliptic` MUST be chosen
/// explicitly. An unset (empty) tag is rejected here with a clear message, instead of letting the
/// old physics default (`compressible`/`charge`) be selected silently. `source` may stay "none" (the
/// explicit, neutral no-source choice); an empty source is also rejected so a cleared tag fails loud
/// rather than tripping dispatch_source's "invalid here" message. This is a CONTRACT guard (mirrors
/// throw_registry_dispatch_mismatch in dispatch_tags.hpp), distinct from a user-tag typo: an unknown
/// (non-empty) tag is still caught downstream by dispatch_transport / dispatch_source /
/// dispatch_elliptic, which list the valid values. Call at every public ModelSpec entry point.
inline void validate_model_spec(const ModelSpec& m) {
  if (m.transport.empty())
    throw std::runtime_error(
        "ModelSpec: transport not set (required) -- choose 'exb' | 'compressible' | 'isothermal'. "
        "The core infers no physics default (no silent 'compressible').");
  if (m.elliptic.empty())
    throw std::runtime_error(
        "ModelSpec: elliptic not set (required) -- choose 'charge' | 'background' | 'gravity'. "
        "The core infers no physics default (no silent 'charge').");
  if (m.source.empty())
    throw std::runtime_error("ModelSpec: source not set -- choose " + source_choices() +
                             " ('none' = no source term).");
}

/// Non-drift guard (ADC-331): the registry's n_vars column (model_registry.hpp, a LIGHT header with
/// no brick types) MUST agree with the real brick types' ::n_vars. This TU sees BOTH, so we lock it
/// at compile time -- a registry row that disagrees with its brick fails the build here.
static_assert(ExBVelocity::n_vars == transport_n_vars_ct("exb"), "registry n_vars drift: exb");
static_assert(CompressibleFlux::n_vars == transport_n_vars_ct("compressible"),
              "registry n_vars drift: compressible");
static_assert(IsothermalFlux::n_vars == transport_n_vars_ct("isothermal"),
              "registry n_vars drift: isothermal");

/// Builds the transport brick and calls v(transport).
template <class Visitor>
POPS_COLD_FN void dispatch_transport(const ModelSpec& m, Visitor&& v) {
  validate_transport(
      m.transport);  // registry rejection (single source of the valid tags + message)
  if (m.transport == "exb")
    return v(ExBVelocity{Real(m.B0)});
  if (m.transport == "compressible")
    return v(CompressibleFlux{Real(m.gamma)});
  if (m.transport == "isothermal")
    return v(IsothermalFlux{Real(m.cs2), Real(m.vacuum_floor)});
  // Reached only if a registry tag is not routed by the if-chain (a registry/dispatch inconsistency,
  // i.e. a programming bug); user typos were already rejected by validate_transport above.
  throw std::runtime_error("transport '" + m.transport +
                           "' valid in registry but not routed (add the dispatch case)");
}

/// Builds the source brick and calls v(source). Fluid sources (force) require
/// >= 3 variables: on a scalar transport (exb), only "none" is valid.
///   - "none": NoSource (neutral);
///   - "potential": PotentialForce (q/m) rho E (electrostatic);
///   - "gravity": GravityForce rho g;
///   - "magnetic" | "lorentz": MagneticLorentzForce q v x B_z (B_z read from aux, EXPLICIT
///                                    regime; the stiff regime goes through the condensed Schur);
///   - "potential_magnetic" | "potential_lorentz": CompositeSource<PotentialForce, MagneticLorentz>
///                                    = electrostatic + Lorentz summed (the full magnetized force in a
///                                    polar setup, with no centrifugal workaround needed).
/// qom (q/m, sign included) is shared by the two charged forces (same species). The magnetized bricks
/// declare n_aux = 4 -> CompositeModel propagates the aux width up to the system (B_z channel).
template <int NV, class Visitor>
POPS_COLD_FN void dispatch_source(const ModelSpec& m, Visitor&& v) {
  if (m.source == "none")
    return v(NoSource{});
  if constexpr (NV >= 3) {
    if (m.source == "potential")
      return v(PotentialForce{Real(m.qom)});
    if (m.source == "gravity")
      return v(GravityForce{});
    if (m.source == "magnetic" || m.source == "lorentz")
      return v(MagneticLorentzForce{Real(m.qom)});
    if (m.source == "potential_magnetic" || m.source == "potential_lorentz")
      return v(CompositeSource<PotentialForce, MagneticLorentzForce>{
          PotentialForce{Real(m.qom)}, MagneticLorentzForce{Real(m.qom)}});
  }
  throw std::runtime_error("source '" + m.source +
                           "' invalid here (requires a fluid transport >= 3 variables, or 'none')");
}

/// Builds the elliptic right-hand-side brick and calls v(elliptic).
template <class Visitor>
POPS_COLD_FN void dispatch_elliptic(const ModelSpec& m, Visitor&& v) {
  validate_elliptic(m.elliptic);  // registry rejection (single source of the valid tags + message)
  if (m.elliptic == "charge")
    return v(ChargeDensity{Real(m.q)});
  if (m.elliptic == "background")
    return v(BackgroundDensity{Real(m.alpha), Real(m.n0)});
  if (m.elliptic == "gravity")
    return v(GravityCoupling{Real(m.sign), Real(m.four_pi_G), Real(m.rho0)});
  // Reached only on a registry/dispatch inconsistency (see dispatch_transport): unknown user tags
  // were already rejected by validate_elliptic above.
  throw std::runtime_error("elliptic '" + m.elliptic +
                           "' valid in registry but not routed (add the dispatch case)");
}

/// AUTOMATIC resolution by ROLES (audit sec.5): fills the component indices of a SOURCE or ELLIPTIC
/// brick (c_rho / c_mx / c_my / c_E) from the conservative descriptor @p cons of the TRANSPORT.
/// This is a TRANSPARENT resolution, with no new user parameter: the native bricks adapt to the
/// transport layout (density/momentum/energy located by their ROLE and not by a hard-coded index).
/// An index is only WRITTEN if the role exists in @p cons; otherwise the brick KEEPS its canonical
/// default (historical behavior for a transport without roles).
///
/// Member detection via `requires` (if constexpr): the bricks have HETEROGENEOUS index sets
/// (PotentialForce/GravityForce: rho/mx/my/E; MagneticLorentzForce: mx/my only;
/// ChargeDensity/Background/GravityCoupling: rho; NoSource: none); only the EXISTING members
/// are touched. CompositeSource<A,B> has no indices of its own: we recurse into its two sub-bricks.
///
/// BIT-IDENTICAL for the NATIVE transports: Euler (rho=0, m_x=1, m_y=2, E=3), Isothermal
/// (rho=0, m_x=1, m_y=2) and ExB (density=0) declare CANONICAL roles -> the resolved indices ==
/// the brick defaults -> no value changes. Resolved AT CONSTRUCTION (host, std::string); never on device.
template <class Brick>
POPS_COLD_FN void bind_variable_roles(Brick& brk, const VariableSet& cons) {
  const int i_rho = cons.index_of(VariableRole::Density);
  const int i_mx = cons.index_of(VariableRole::MomentumX);
  const int i_my = cons.index_of(VariableRole::MomentumY);
  const int i_E = cons.index_of(VariableRole::Energy);
  if constexpr (requires { brk.c_rho; }) {
    if (i_rho >= 0)
      brk.c_rho = i_rho;
  }
  if constexpr (requires { brk.c_mx; }) {
    if (i_mx >= 0)
      brk.c_mx = i_mx;
  }
  if constexpr (requires { brk.c_my; }) {
    if (i_my >= 0)
      brk.c_my = i_my;
  }
  if constexpr (requires { brk.c_E; }) {
    if (i_E >= 0)
      brk.c_E = i_E;
  }
  if constexpr (requires {
                  brk.a;
                  brk.b;
                }) {  // CompositeSource<A,B>: recursion into the sub-bricks
    bind_variable_roles(brk.a, cons);
    bind_variable_roles(brk.b, cons);
  }
}

/// Assembles the CompositeModel designated by @p m and calls `visitor(model)`.
/// @throws std::runtime_error on unknown tag or invalid combination.
template <class Visitor>
POPS_COLD_FN void dispatch_model(const ModelSpec& m, Visitor&& visitor) {
  validate_model_spec(m);  // explicit completeness contract (ADC-290): no silent physics default
  dispatch_transport(m, [&](auto tr) {
    using TR = decltype(tr);
    // Transport roles (host): used to resolve the indices of the source / elliptic bricks before
    // freezing the composite. Native transport -> canonical roles -> resolved indices == defaults.
    const VariableSet cons = TR::conservative_vars();
    dispatch_source<TR::n_vars>(m, [&](auto src) {
      dispatch_elliptic(m, [&](auto ell) {
        bind_variable_roles(src,
                            cons);  // AUTOMATIC resolution by roles (transparent, bit-identical)
        bind_variable_roles(ell, cons);
        visitor(CompositeModel<TR, decltype(src), decltype(ell)>{tr, src, ell});
      });
    });
  });
}

/// Same as dispatch_model but with the transport brick ALREADY chosen (@p tr). Runs ONLY the
/// source/elliptic dispatch for that fixed transport @p TR and calls visitor(CompositeModel<...>).
/// This is the seam that lets the per-transport translation units (system_{exb,isothermal,
/// compressible}.cpp, ADC-335) each instantiate ONLY their own transport's leaves: a TU calling
/// dispatch_model_for<CompressibleFlux> never sees the exb/isothermal branches of dispatch_transport,
/// so the ~1700-leaf combinatorial product splits cleanly across files for `-j`. The body is the inner
/// part of dispatch_model VERBATIM (same role binding, same CompositeModel<TR,...> synthesis), so the
/// reachable instantiation set is unchanged: dispatch_model itself is UNTOUCHED (still used by the .so /
/// add_compiled_model loader path), and the union over the three transports is byte-identical.
template <class TR, class Visitor>
POPS_COLD_FN void dispatch_model_for(const ModelSpec& m, TR tr, Visitor&& visitor) {
  const VariableSet cons = TR::conservative_vars();
  dispatch_source<TR::n_vars>(m, [&](auto src) {
    dispatch_elliptic(m, [&](auto ell) {
      bind_variable_roles(src, cons);
      bind_variable_roles(ell, cons);
      visitor(CompositeModel<TR, decltype(src), decltype(ell)>{tr, src, ell});
    });
  });
}

/// Resolves the IMPLICIT MASK of a block (add_block: implicit_vars / implicit_roles) into a list of
/// conserved-component indices, against the block descriptor @p cons. The mask lives on the BLOCK /
/// time-policy side (and NOT the model): same model, distinct implicit treatments per block. A name
/// or role absent from the block raises an EXPLICIT error (no silent ignore). Returns the UNIQUE,
/// sorted indices (order is irrelevant). Empty input -> empty -> inactive mask. Moved out of
/// system.cpp's anonymous namespace (ADC-335) so the per-transport seam TUs share one definition.
inline POPS_COLD_FN std::vector<int> resolve_implicit_components(
    const std::string& block, const VariableSet& cons, const std::vector<std::string>& names,
    const std::vector<std::string>& roles) {
  std::vector<int> out;
  auto push_unique = [&out](int c) {
    if (std::find(out.begin(), out.end(), c) == out.end())
      out.push_back(c);
  };
  for (const std::string& nm : names) {
    int idx = -1;
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == nm) {
        idx = i;
        break;
      }
    if (idx < 0) {
      std::string have;
      for (std::size_t i = 0; i < cons.names.size(); ++i) {
        if (i)
          have += ", ";
        have += cons.names[i];
      }
      throw std::runtime_error("System::add_block : implicit_vars : variable '" + nm +
                               "' absent from block '" + block +
                               "' (conserved variables : " + have + ")");
    }
    push_unique(idx);
  }
  for (const std::string& rn : roles) {
    const int idx = cons.index_of(rn);  // canonical role name OR user-defined role label (ADC-292)
    if (idx < 0) {
      std::string have = roles_csv(cons);
      throw std::runtime_error(
          "System::add_block : implicit_roles : role '" + rn + "' absent from block '" + block +
          "' (roles : " + (have.empty() ? std::string("<not provided>") : have) + ")");
    }
    push_unique(idx);
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Resolves a SINGLE selector variable of @p block (the AMR regrid variable, ADC-296) into its
/// conserved-component index, by NAME (@p name) XOR by physical ROLE (@p role), against @p cons. STRICT,
/// like resolve_implicit_components: an absent name/role raises an EXPLICIT error (NO silent fallback to
/// component 0 -- the whole point of letting a model put its refinement variable off component 0). Empty
/// name AND empty role -> -1, the caller keeps its default (component 0, historical density criterion,
/// bit-identical). At most one of name/role may be set. @p origin labels the error (e.g.
/// "AmrSystem::set_refinement").
inline POPS_COLD_FN int resolve_selected_component(const std::string& origin,
                                                  const std::string& block, const VariableSet& cons,
                                                  const std::string& name,
                                                  const std::string& role) {
  if (name.empty() && role.empty())
    return -1;  // default selector -> caller's component 0
  if (!name.empty() && !role.empty())
    throw std::runtime_error(origin +
                             " : select the refinement variable by NAME or by ROLE, not both");
  if (!name.empty()) {
    for (int i = 0; i < static_cast<int>(cons.names.size()); ++i)
      if (cons.names[i] == name)
        return i;
    std::string have;
    for (std::size_t i = 0; i < cons.names.size(); ++i) {
      if (i)
        have += ", ";
      have += cons.names[i];
    }
    throw std::runtime_error(origin + " : variable '" + name + "' absent from block '" + block +
                             "' (conserved variables : " + have + ")");
  }
  const int idx = cons.index_of(role);  // canonical role name OR user-defined role label (ADC-292)
  if (idx < 0) {
    const std::string have = roles_csv(cons);
    throw std::runtime_error(origin + " : role '" + role + "' absent from block '" + block +
                             "' (roles : " + (have.empty() ? std::string("<not provided>") : have) +
                             ")");
  }
  return idx;
}

}  // namespace pops::detail

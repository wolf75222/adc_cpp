#pragma once

#include <adc/physics/bricks.hpp>
#include <adc/runtime/model_spec.hpp>

#include <stdexcept>
#include <string>

/// @file
/// @brief Assemble a CompositeModel from a ModelSpec (bricks + parameters).
///
/// The core knows only generic BRICKS; a scenario is a composition, named on the
/// application side (adc_cases). dispatch_model(spec, visitor) builds the
/// CompositeModel<Hyperbolic, Source, Elliptic> designated by the spec and calls visitor(model).
/// Invalid combinations (fluid source on a scalar transport) are rejected.

namespace adc::detail {

/// Builds the transport brick and calls v(transport).
template <class Visitor>
void dispatch_transport(const ModelSpec& m, Visitor&& v) {
  if (m.transport == "exb") return v(ExBVelocity{Real(m.B0)});
  if (m.transport == "compressible") return v(CompressibleFlux{Real(m.gamma)});
  if (m.transport == "isothermal") return v(IsothermalFlux{Real(m.cs2)});
  throw std::runtime_error("unknown transport '" + m.transport +
                           "' (exb|compressible|isothermal)");
}

/// Builds the source brick and calls v(source). Fluid sources (force) require
/// >= 3 variables: on a scalar transport (exb), only "none" is valid.
///   - "none": NoSource (neutral);
///   - "potential": PotentialForce (q/m) rho E (electrostatic);
///   - "gravity": GravityForce rho g;
///   - "magnetic" | "lorentz": MagneticLorentzForce q v x B_z (B_z read from aux, EXPLICIT
///                                    regime; the stiff regime goes through the condensed Schur);
///   - "potential_magnetic" | "potential_lorentz": CompositeSource<PotentialForce, MagneticLorentz>
///                                    = electrostatic + Lorentz summed (full force of the NATIVE polar
///                                    diocotron, without the centrifugal workaround).
/// qom (q/m, sign included) is shared by the two charged forces (same species). The magnetized bricks
/// declare n_aux = 4 -> CompositeModel propagates the aux width up to the system (B_z channel).
template <int NV, class Visitor>
void dispatch_source(const ModelSpec& m, Visitor&& v) {
  if (m.source == "none") return v(NoSource{});
  if constexpr (NV >= 3) {
    if (m.source == "potential") return v(PotentialForce{Real(m.qom)});
    if (m.source == "gravity") return v(GravityForce{});
    if (m.source == "magnetic" || m.source == "lorentz")
      return v(MagneticLorentzForce{Real(m.qom)});
    if (m.source == "potential_magnetic" || m.source == "potential_lorentz")
      return v(CompositeSource<PotentialForce, MagneticLorentzForce>{PotentialForce{Real(m.qom)},
                                                                     MagneticLorentzForce{Real(m.qom)}});
  }
  throw std::runtime_error("source '" + m.source +
                           "' invalid here (requires a fluid transport >= 3 variables, or 'none')");
}

/// Builds the elliptic right-hand-side brick and calls v(elliptic).
template <class Visitor>
void dispatch_elliptic(const ModelSpec& m, Visitor&& v) {
  if (m.elliptic == "charge") return v(ChargeDensity{Real(m.q)});
  if (m.elliptic == "background") return v(BackgroundDensity{Real(m.alpha), Real(m.n0)});
  if (m.elliptic == "gravity")
    return v(GravityCoupling{Real(m.sign), Real(m.four_pi_G), Real(m.rho0)});
  throw std::runtime_error("unknown elliptic '" + m.elliptic + "' (charge|background|gravity)");
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
void bind_variable_roles(Brick& brk, const VariableSet& cons) {
  const int i_rho = cons.index_of(VariableRole::Density);
  const int i_mx = cons.index_of(VariableRole::MomentumX);
  const int i_my = cons.index_of(VariableRole::MomentumY);
  const int i_E = cons.index_of(VariableRole::Energy);
  if constexpr (requires { brk.c_rho; }) { if (i_rho >= 0) brk.c_rho = i_rho; }
  if constexpr (requires { brk.c_mx; })  { if (i_mx >= 0) brk.c_mx = i_mx; }
  if constexpr (requires { brk.c_my; })  { if (i_my >= 0) brk.c_my = i_my; }
  if constexpr (requires { brk.c_E; })   { if (i_E >= 0) brk.c_E = i_E; }
  if constexpr (requires { brk.a; brk.b; }) {  // CompositeSource<A,B>: recursion into the sub-bricks
    bind_variable_roles(brk.a, cons);
    bind_variable_roles(brk.b, cons);
  }
}

/// Assembles the CompositeModel designated by @p m and calls `visitor(model)`.
/// @throws std::runtime_error on unknown tag or invalid combination.
template <class Visitor>
void dispatch_model(const ModelSpec& m, Visitor&& visitor) {
  dispatch_transport(m, [&](auto tr) {
    using TR = decltype(tr);
    // Transport roles (host): used to resolve the indices of the source / elliptic bricks before
    // freezing the composite. Native transport -> canonical roles -> resolved indices == defaults.
    const VariableSet cons = TR::conservative_vars();
    dispatch_source<TR::n_vars>(m, [&](auto src) {
      dispatch_elliptic(m, [&](auto ell) {
        bind_variable_roles(src, cons);  // AUTOMATIC resolution by roles (transparent, bit-identical)
        bind_variable_roles(ell, cons);
        visitor(CompositeModel<TR, decltype(src), decltype(ell)>{tr, src, ell});
      });
    });
  });
}

}  // namespace adc::detail

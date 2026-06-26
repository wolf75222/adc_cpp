#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>

/// @file
/// @brief Elliptic RIGHT-HAND-SIDE bricks f(U): a block's contribution to the right-hand side
///        of the system elliptic equation (Poisson). ChargeDensity (q n), BackgroundDensity
///        (alpha (n - n0)), GravityCoupling (s 4piG (rho - rho0)). Composable as the Elliptic
///        parameter of a CompositeModel (physics/composite.hpp). The elliptic OPERATOR (div eps grad)
///        and the solve live on the system side (runtime); here only the per-block right-hand side.

namespace pops {

/// Charge density f = q n. Elliptic right-hand side of the ion or electron block.
///
/// CONTRACT: pointwise ELLIPTIC brick, device-callable (POPS_HD), no global state.
/// Reads only the density. Sign of q included (ion: q=+1, electron: q=-1).
///
/// ROLE-AWARE (audit sec.5): c_rho member, default = canonical layout (density in component 0), resolved
/// by model_factory via TR::conservative_vars().index_of(Density). Canonical == default for every native
/// transport -> STRICTLY bit-identical. int POD -> rhs stays device-clean.
struct ChargeDensity {
  Real q = 1;
  int c_rho = 0;  // default = density in component 0 (bit-identical)
  template <class State>
  POPS_HD Real rhs(const State& u) const {
    return q * u[c_rho];
  }
};

/// Neutralizing background f = alpha (n - n0). Models a fixed neutralization background of density n0.
///
/// CONTRACT: pointwise ELLIPTIC brick, device-callable (POPS_HD), no global state.
/// Reads only the density. alpha = effective charge of the background; n0 = density of the neutral background.
///
/// ROLE-AWARE (audit sec.5): c_rho member, default = component 0, resolved by model_factory via the Density
/// role of the transport. Canonical == default -> bit-identical. See ChargeDensity.
struct BackgroundDensity {
  Real alpha = 1, n0 = 0;
  int c_rho = 0;  // default = density in component 0 (bit-identical)
  template <class State>
  POPS_HD Real rhs(const State& u) const {
    return alpha * (u[c_rho] - n0);
  }
};

/// Self-consistent coupling f = sign * 4piG * (rho - rho0).
///
/// CONTRACT: pointwise ELLIPTIC brick, device-callable (POPS_HD), no global state.
/// sign = +1 gravity (standard Poisson), sign = -1 plasma (Gauss sign). rho0 = background.
///
/// ROLE-AWARE (audit sec.5): c_rho member, default = component 0, resolved by model_factory via the Density
/// role of the transport. Canonical == default -> bit-identical. See ChargeDensity.
struct GravityCoupling {
  Real sign = 1, four_pi_G = 1, rho0 = 1;
  int c_rho = 0;  // default = density in component 0 (bit-identical)
  template <class State>
  POPS_HD Real rhs(const State& u) const {
    return sign * four_pi_G * (u[c_rho] - rho0);
  }
};

}  // namespace pops

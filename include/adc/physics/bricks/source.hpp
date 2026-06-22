#pragma once

#include <adc/core/model/physical_model.hpp>  // aux_comps<>: propagates the aux channel of a composite source
#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>

/// @file
/// @brief SOURCE bricks S(U, aux): local term, generic over state size (acts on
///        energy if 4 variables). NoSource, PotentialForce ((q/m) rho E), GravityForce (rho g),
///        MagneticLorentzForce (q v x B_z). Composable as the Source parameter of a CompositeModel
///        (physics/composite.hpp); CompositeSource<A, B> SUMS two sources (electrostatic +
///        Lorentz). INTER-species sources (ionization, collision) live at the system level
///        (operator-split).

namespace adc {

// OPTIONAL CONTRACT frequency(U, aux) -> Real (audit 2026-06, step_cfl work): a source brick
// may declare its local FREQUENCY mu [1/s] (relaxation/collision/reaction rate). When the brick
// exposes it, CompositeModel forwards it (source_frequency) and System::step_cfl enforces the
// bound dt <= cfl * substeps / (stride * max_cells(mu)) -- the meeting's "second CFL" (source),
// distinct from the transport CFL (no h: a source is bounded in 1/time). A brick WITHOUT
// frequency (all of those in this file today) does not constrain the step (historical). Must be
// ADC_HD (evaluated inside a reduction kernel).

/// No source: S(U, aux) = 0. Neutral brick (model without potential/gravity coupling).
/// Device-callable, no internal state.
struct NoSource {
  template <class State>
  ADC_HD State apply(const State&, const Aux&) const {
    return State{};
  }
};

/// Electrostatic potential force (q/m) rho E on momentum (+ work on energy if 4 variables).
/// E = -grad phi = -(aux.grad_x, aux.grad_y).
///
/// CONTRACT: pointwise SOURCE brick, device-callable (ADC_HD), no global state.
/// Formula: s[c_mx] += qom*rho*Ex, s[c_my] += qom*rho*Ey, s[c_E] += qom*(rho_u*Ex + rho_v*Ey)
/// (the c_E work term is active only if State::size() == 4: compressible Euler).
///
/// ROLE-AWARE (audit section 5): the component indices (density c_rho, momentum c_mx/c_my,
/// energy c_E) are MEMBERS, defaulting to the CANONICAL fluid layout (rho=0, m_x=1, m_y=2, E=3).
/// model_factory RESOLVES them at construction (host) via TR::conservative_vars().index_of(role);
/// for any NATIVE transport (Euler/Isothermal, canonical roles) the resolved indices == these
/// defaults -> STRICTLY bit-identical. POD integers -> apply stays device-clean (reads u[c_rho],
/// never resolves on device). No new user parameter: automatic, transparent resolution.
struct PotentialForce {
  Real qom = 1;                                // q/m (sign included)
  int c_rho = 0, c_mx = 1, c_my = 2, c_E = 3;  // defaults = canonical fluid layout (bit-identical)
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    const Real Ex = -a.grad_x, Ey = -a.grad_y;
    State s{};
    s[c_mx] = qom * u[c_rho] * Ex;
    s[c_my] = qom * u[c_rho] * Ey;
    if constexpr (State::size() == 4)
      s[c_E] = qom * (u[c_mx] * Ex + u[c_my] * Ey);
    return s;
  }
};

/// Gravitational force rho g (+ work if 4 variables). g = -grad phi.
///
/// CONTRACT: pointwise SOURCE brick, device-callable (ADC_HD), no global state.
/// Formula: s[c_mx] += rho*gx, s[c_my] += rho*gy, s[c_E] += rho_u*gx + rho_v*gy
/// (the c_E work term is active only if State::size() == 4: compressible Euler).
/// No q/m coefficient (unlike PotentialForce): g is gravity directly.
///
/// ROLE-AWARE (audit section 5): c_rho/c_mx/c_my/c_E are members, defaults = canonical layout
/// (rho=0, m_x=1, m_y=2, E=3), resolved by model_factory via the transport roles. Canonical
/// indices == defaults for any native transport -> bit-identical. See PotentialForce for the
/// full contract.
struct GravityForce {
  int c_rho = 0, c_mx = 1, c_my = 2, c_E = 3;  // defaults = canonical fluid layout (bit-identical)
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    const Real gx = -a.grad_x, gy = -a.grad_y;
    State s{};
    s[c_mx] = u[c_rho] * gx;
    s[c_my] = u[c_rho] * gy;
    if constexpr (State::size() == 4)
      s[c_E] = u[c_mx] * gx + u[c_my] * gy;
    return s;
  }
};

/// MAGNETIC Lorentz force q (v x B) on momentum, field B = B_z z_hat out of plane.
/// EXPLICIT regime (moderate omega_c): ALGEBRAIC pointwise term (no derivative), coded once for
/// BOTH geometries because it is INVARIANT under orientation of the local orthonormal frame (x,y)
/// or (e_r, e_theta):
///   (rho v_x, rho v_y) x B_z z_hat = (+B_z rho v_y, -B_z rho v_x) = (+B_z m_y, -B_z m_x)  [cartesian]
///   (rho v_r, rho v_th) x B_z z_hat = (+B_z rho v_th, -B_z rho v_r) = (+B_z m_th, -B_z m_r)  [polar]
/// So s[1] = +qom*B_z*m[2], s[2] = -qom*B_z*m[1] with m[1]=u[1] (1st momentum component),
/// m[2]=u[2] (2nd). v x B is PERPENDICULAR to v: the work F . v = 0 -> s[3] (energy) stays ZERO
/// even at 4 variables (the magnetic force does not change kinetic energy). qom = q/m (sign
/// included, consistent with PotentialForce); the gyration sense (cyclotron) follows the sign
/// of qom*B_z.
///
/// CONTRACT: pointwise SOURCE brick, device-callable (ADC_HD), no global state. Reads B_z from the
/// aux (canonical component 3, as set_magnetic_field populates it) -> declares n_aux = 4 so the aux
/// channel is sized and load_aux fills a.B_z. The STIFF regime (large omega_c) goes through the
/// condensed Schur (ElectrostaticLorentzCondensation), NOT through this explicit brick.
/// PRECONDITION: requires a fluid transport >= 3 variables (momentum on 2 axes); moot on a scalar.
struct MagneticLorentzForce {
  Real qom = 1;                    // q/m (sign included)
  static constexpr int n_aux = 4;  // reads B_z (extra aux channel, canonical index 3)
  // ROLE-AWARE (audit section 5): only the MOMENTUM components are read/written (the magnetic
  // force touches neither density nor energy -- zero work). c_mx/c_my are members, defaults =
  // canonical layout (m_x=1, m_y=2), resolved by model_factory via the transport roles. Canonical
  // == defaults -> bit-identical. POD integers -> device-clean.
  int c_mx = 1, c_my = 2;
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    static_assert(
        State::size() >= 3,
        "MagneticLorentzForce : requires a fluid transport >= 3 variables (momentum on 2 axes)");
    const Real c = qom * a.B_z;
    State s{};
    s[c_mx] = c * u[c_my];   // +qom B_z m_(y/theta)
    s[c_my] = -c * u[c_mx];  // -qom B_z m_(x/r)
    // energy stays 0: v x B is perpendicular to v, zero work.
    return s;
  }
};

/// SUM of two source bricks: S(U, aux) = A.apply(U, aux) + B.apply(U, aux). Allows COMPOSING
/// several pointwise forces in the SINGLE Source slot of CompositeModel (e.g. electrostatic
/// PotentialForce + magnetic MagneticLorentzForce for a magnetized E x B plasma). Generic over state size
/// (StateVec addition is defined component-wise, cf. core/state.hpp).
///
/// CONTRACT: pointwise SOURCE brick, device-callable (ADC_HD), no global state beyond the two
/// sub-bricks (themselves POD). PROPAGATES the aux channel: n_aux = max(aux_comps<A>, aux_comps<B>)
/// -> if a sub-brick reads B_z (n_aux=4), the composite exposes it and CompositeModel raises it to
/// the system.
template <class A, class B>
struct CompositeSource {
  A a{};
  B b{};
  static constexpr int n_aux = aux_comps<A>() > aux_comps<B>() ? aux_comps<A>() : aux_comps<B>();
  template <class State>
  ADC_HD State apply(const State& u, const Aux& ax) const {
    return a.apply(u, ax) + b.apply(u, ax);
  }
};

}  // namespace adc

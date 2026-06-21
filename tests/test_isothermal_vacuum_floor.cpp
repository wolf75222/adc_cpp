// ADC-77: quasi-vacuum velocity bound on IsothermalFlux (and the inherited IsothermalFluxPolar).
// At ~52% of the diocotron rollup the background is evacuated (rho -> ~1e-9) and the Schur source
// stage writes O(1) momentum onto those cells, so the raw u = m/rho explodes and collapses the CFL.
// The model now computes u = m/max(rho, vacuum_floor) when vacuum_floor > 0, which bounds BOTH the
// CFL wave speed and the advective flux. This test pins:
//   (1) vacuum_floor <= 0 is bit-identical to the raw 1/rho path (feature OFF, and the one-arg
//       aggregate init defaults to OFF);
//   (2) vacuum_floor > 0 with rho < floor bounds the velocity to m/floor and keeps the wave speed
//       and flux finite, while rho itself (and the pressure cs2*rho) stay the RAW conserved value;
//   (3) vacuum_floor > 0 with rho >= floor is inactive (identical to OFF);
//   (4) the polar geometric source is floored too (finite at vacuum, identical above the floor).

#include <adc/physics/hyperbolic.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const Aux aux{};
  const Real cs2 = Real(0.5);

  // Quasi-vacuum cell: rho ~ 1e-9 with O(1) momentum -> raw u = m/rho ~ 1e8 (blows up).
  StateVec<3> uvac{};
  uvac[0] = Real(1e-9);
  uvac[1] = Real(0.3);
  uvac[2] = Real(-0.2);

  // (1) floor OFF: bit-identical to the raw 1/rho path.
  {
    const IsothermalFlux off{cs2, Real(0)};
    const Real rho = uvac[0];
    const Real vx = uvac[1] / rho, vy = uvac[2] / rho;
    const auto p = off.to_primitive(uvac);
    chk(p[0] == rho && p[1] == vx && p[2] == vy, "off_to_primitive_raw");
    chk(off.max_wave_speed(uvac, aux, 0) == (vx < 0 ? -vx : vx) + std::sqrt(cs2), "off_mws_raw_x");
    chk(off.flux(uvac, aux, 0)[1] == uvac[1] * vx + cs2 * rho, "off_flux_raw_x");
    // The one-arg aggregate init must also be OFF (vacuum_floor defaults to 0): bit-identical.
    const IsothermalFlux dflt{cs2};
    chk(dflt.max_wave_speed(uvac, aux, 0) == off.max_wave_speed(uvac, aux, 0), "default_is_off");
  }

  // (2) floor ON, rho < floor: the velocity uses max(rho, floor) = floor; rho/pressure stay raw.
  {
    const Real floor = Real(1e-3);
    const IsothermalFlux on{cs2, floor};
    const Real vx_b = uvac[1] / floor, vy_b = uvac[2] / floor;
    const auto p = on.to_primitive(uvac);
    chk(p[1] == vx_b && p[2] == vy_b, "on_velocity_bounded");
    chk(p[0] == uvac[0], "on_rho_is_raw");
    const Real mws = on.max_wave_speed(uvac, aux, 0);
    chk(std::isfinite(mws) && mws == (vx_b < 0 ? -vx_b : vx_b) + std::sqrt(cs2), "on_mws_bounded");
    // flux: advective velocity floored, pressure cs2*rho still uses the raw rho.
    chk(on.flux(uvac, aux, 0)[1] == uvac[1] * vx_b + cs2 * uvac[0], "on_flux_bounded_raw_pressure");
  }

  // (3) floor ON but rho >= floor: inactive (identical to OFF).
  {
    const Real floor = Real(1e-3);
    const IsothermalFlux on{cs2, floor};
    const IsothermalFlux off{cs2, Real(0)};
    StateVec<3> u{};
    u[0] = Real(2.0);
    u[1] = Real(0.6);
    u[2] = Real(0.4);
    chk(on.to_primitive(u)[1] == u[1] / u[0], "on_inactive_above_floor");
    chk(on.max_wave_speed(u, aux, 1) == off.max_wave_speed(u, aux, 1), "on_eq_off_above_floor");
  }

  // (4) polar geometric source: floored at vacuum (finite), identical above the floor.
  {
    const IsothermalFluxPolar on{IsothermalFlux{cs2, Real(1e-3)}};
    const StateVec<3> s = on.polar_geom_source(uvac, Real(1.5));
    chk(s[0] == Real(0) && std::isfinite(s[1]) && std::isfinite(s[2]),
        "polar_geom_finite_at_vacuum");
    const IsothermalFluxPolar off{IsothermalFlux{cs2, Real(0)}};
    StateVec<3> u{};
    u[0] = Real(2.0);
    u[1] = Real(0.6);
    u[2] = Real(0.4);
    const StateVec<3> so = off.polar_geom_source(u, Real(1.5));
    const StateVec<3> sn = on.polar_geom_source(u, Real(1.5));
    chk(so[1] == sn[1] && so[2] == sn[2], "polar_geom_eq_above_floor");
  }

  if (fails == 0)
    std::printf("OK test_isothermal_vacuum_floor\n");
  return fails;
}

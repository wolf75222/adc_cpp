"""ADC-77 end-to-end: the FluidState(vacuum_floor=...) knob threads to the native isothermal model.

Builds two identical isothermal Systems that differ ONLY by vacuum_floor and shows:
  (1) on a QUASI-VACUUM state (rho << floor) with O(1) velocity, the bounded model velocity
      u = m/max(rho, floor) changes the trajectory -> the two runs differ (the knob is wired through
      FluidState -> ModelSpec -> IsothermalFlux);
  (2) on a NORMAL state (rho >> floor) the floor is inactive -> the two runs are BIT-IDENTICAL (the
      bound does not perturb normal runs -- the reason it is a SEPARATE knob from positivity_floor);
  (3) vacuum_floor < 0 is rejected at the python boundary.

Native bricks only (no DSL / no compiler). source=NoSource isolates the transport (where the floor
lives): phi is identical between the two runs, so any difference comes from the velocity bound.
"""
from pops.numerics.variables import Conservative
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import sys

import numpy as np

try:
    import pops
except ImportError as e:
    print("skip  module pops absent (PYTHONPATH ?) : %s" % e)
    sys.exit(0)


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        raise AssertionError(label)


def run(n, L, vacuum_floor, rho_scale, nsteps, dt):
    """One short isothermal transport run; returns the conservative state (3, n, n)."""
    sim = pops.System(n=n, L=L, periodic=False)
    sim.set_poisson(bc="dirichlet")
    sim.set_magnetic_field(np.ones((n, n)))
    sim.add_equation(
        "ions",
        model=pops.Model(
            state=pops.FluidState(kind="isothermal", cs2=1.0, vacuum_floor=vacuum_floor),
            transport=pops.IsothermalFlux(),
            source=pops.NoSource(),
            elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0),
        ),
        spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov(), variables=Conservative()),
        time=pops.Explicit(),
    )
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    # Smooth profiles, zero at the boundaries (Dirichlet-compatible); rho scaled to sit below/above
    # the floor depending on rho_scale.
    rho = rho_scale * (1.0 + 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L))
    u = 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    v = -0.3 * np.sin(2.0 * np.pi * X / L) * np.sin(np.pi * Y / L)
    sim.set_primitive_state("ions", rho=rho, u=u, v=v)
    for _ in range(nsteps):
        sim.step(dt)
    return np.array(sim.get_state("ions")).reshape(3, n, n)


def main():
    n, L = 24, 1.0
    dt = 1.0e-3
    nsteps = 5
    floor = 1.0e-2

    # (1) quasi-vacuum: rho ~ 1.5e-3 << floor -> bounded velocity changes the momentum flux.
    s_off = run(n, L, 0.0, 1.0e-3, nsteps, dt)
    s_on = run(n, L, floor, 1.0e-3, nsteps, dt)
    chk(np.all(np.isfinite(s_off)) and np.all(np.isfinite(s_on)), "(1) both runs finite")
    dmax_vac = float(np.max(np.abs(s_on - s_off)))
    chk(dmax_vac > 1e-9,
        "(1) vacuum_floor wired: bounded velocity changes the trajectory at rho<<floor (dmax=%.3e)"
        % dmax_vac)

    # (2) normal density: rho ~ 1.5 >> floor -> floor inactive -> bit-identical.
    t_off = run(n, L, 0.0, 1.0, nsteps, dt)
    t_on = run(n, L, floor, 1.0, nsteps, dt)
    dmax_norm = float(np.max(np.abs(t_on - t_off)))
    chk(dmax_norm == 0.0,
        "(2) floor inactive at rho>>floor: bit-identical to vacuum_floor=0 (dmax=%.3e)" % dmax_norm)

    # (3) validation at the python boundary.
    try:
        pops.FluidState(kind="isothermal", cs2=1.0, vacuum_floor=-1.0)
        chk(False, "(3) vacuum_floor < 0 rejected")
    except ValueError:
        chk(True, "(3) vacuum_floor < 0 rejected")

    print("test_isothermal_vacuum_floor_system : tout est vert (3 verifications)")


if __name__ == "__main__":
    main()

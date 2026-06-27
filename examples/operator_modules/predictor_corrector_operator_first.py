#!/usr/bin/env python3
"""Operator-first predictor-corrector authored as a PURE pops.model.Module (Spec 2 / ADC-447).

This is the spec's "model-free example target": the model is an ``pops.model.Module`` whose operators
are declared by signature with IR (``Expr``) bodies -- a field operator (Poisson), a flux
(grid operator), an electric source, a Lorentz local linear operator -- plus a composite rate
``explicit_rhs``. No PDE method (``m.flux`` / ``m.source_term``) is called on the model; the Module
IS the model. The time algorithm is the GENERIC macro
``pops.lib.time.predictor_corrector_local_linear`` keyed on the three operator names -- it mentions no
flux / source / poisson / lorentz.

``compile_problem(model=module, time=P)`` lowers the Module to the dsl codegen engine (Module.to_dsl,
a translation -- not a second backend) and compiles the combined .so; ``sim.step`` runs it C++-side.
The result is checked against an offline replay of the same stages, exactly as the dsl-authored
version was, demonstrating that the pure-Module path compiles, runs, and is correct.

Run::

    python examples/operator_modules/predictor_corrector_operator_first.py

Requires a compiler + a visible Kokkos (``POPS_KOKKOS_ROOT``); prints a skip notice and exits 0
otherwise (run it on ROMEO). cf. docs/sphinx/reference/operator-modules.md.
"""
import sys

try:
    import numpy as np

    import pops
    from pops.ir.expr import Const, Expr, Var
    from pops.ir.ops import sqrt
    from pops.physics.facade import Model
    from pops import model
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # noqa: BLE001
    print("skip predictor_corrector_operator_first (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16
BZ = 3.0
DT = 0.02
CS2 = 0.5


def operator_module(name="euler_poisson_lorentz_operator_first"):
    """The model as a pure Module: typed spaces + operators with IR bodies, no PDE method calls."""
    mod = model.Module(name)
    u = mod.state_space("U", ("rho", "mx", "my"),
                        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    mod.aux_fields(B_z="cell_scalar")
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    gx, gy = Var("grad_x", "aux"), Var("grad_y", "aux")
    bz = Var("B_z", "aux")
    cs = sqrt(CS2)
    mod.operator(name="fields_from_state", signature=(u,) >> fields,
                 kind="field_operator", expr=rho)
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [mx, mx * mx / rho + CS2 * rho, mx * my / rho],
                       "y": [my, mx * my / rho, my * my / rho + CS2 * rho]})
    mod.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                    y=[my / rho - cs, my / rho, my / rho + cs])
    mod.operator(name="electric", signature=(u, fields) >> model.Rate(u),
                 kind="local_source", expr=[Const(0.0), -rho * gx, -rho * gy])
    mod.operator(name="lorentz", signature=(fields,) >> model.LocalLinearOperator(u, u),
                 kind="local_linear_operator",
                 expr=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    mod.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return mod


def operator_first_program(module, name="predictor_corrector_operator_first"):
    """The GENERIC model-free macro keyed on the three operator names. No physics here."""
    P = adctime.Program(name).bind_operators(module)
    libtime.predictor_corrector_local_linear(
        P, "plasma",
        fields_operator="fields_from_state",
        explicit_rate_operator="explicit_rhs",
        implicit_operator="lorentz")
    return P


def default_source_model(name="opfirst_ref"):
    """Same physics with the electric force as the DEFAULT source (a Model) so eval_rhs returns
    -div F + electric directly -- used to build the offline reference."""
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs = sqrt(CS2)
    m.flux(x=[mx, mx * mx / rho + CS2 * rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho + CS2 * rho])
    m.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                  y=[my / rho - cs, my / rho, my / rho + cs])
    m.primitive_vars(rho, mx, my)            # identity primitive state (Prim = the conserved vars)
    m.conservative_from([rho, mx, my])
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.aux("B_z")
    m.elliptic_rhs(rho)
    m.source([Const(0.0), -rho * gx, -rho * gy])
    return m


def initial_state():
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def make_sim(block_model):
    """The native reference System (lower-level add_equation path) + shared Poisson + B_z."""
    sim = pops.System(n=N, L=1.0, periodic=True)
    compiled = block_model.compile(backend="production")
    sim.add_equation("plasma", compiled,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    sim.set_magnetic_field(BZ * np.ones(N * N))
    u0 = initial_state()
    sim.set_state("plasma", u0)
    return sim, u0


def offline_rhs(ref, u):
    ref.set_state("plasma", u)
    ref.solve_fields()
    return np.array(ref.eval_rhs("plasma"))


def lorentz_solve(u, a):
    k = a * BZ
    den = 1.0 + k * k
    rho, mx, my = u[0], u[1], u[2]
    return np.stack([rho, (mx + k * my) / den, (-k * mx + my) / den])


def lorentz_apply(u):
    rho, mx, my = u[0], u[1], u[2]
    return np.stack([np.zeros_like(rho), BZ * my, -BZ * mx])


def main():
    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip predictor_corrector_operator_first (_pops lacks install_program; rebuild _pops)")
        return 0
    try:
        mod = operator_module()
        compiled = pops.compile_problem(model=mod, time=operator_first_program(mod))
        ref = make_sim(default_source_model())[0]
    except RuntimeError as exc:
        print("skip predictor_corrector_operator_first (compile_problem could not build the .so: %s)"
              % str(exc)[:160])
        return 0

    u0 = initial_state()
    # Compiled path via the unified headline entry: install() pre-resolves the Module (lowered to its
    # dsl block model via to_dsl), wires its initial state, the B_z aux and the Poisson solver, then
    # installs the compiled time Program. The Module IS the block model.
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install(compiled,
                instances={"plasma": {"model": mod.to_dsl(),
                                      "spatial": pops.FiniteVolume(limiter="none",
                                                                   riemann="rusanov"),
                                      "time": pops.Explicit(method="euler"),
                                      "initial": u0}},
                aux={"B_z": BZ * np.ones(N * N)},
                solvers={"phi": pops.lib.fields.GeometricMG()})
    sim.step(DT)
    u_pc = np.array(sim.get_state("plasma"))

    r_n = offline_rhs(ref, u0)
    u_star = lorentz_solve(u0 + DT * r_n, DT)
    r_star = offline_rhs(ref, u_star)
    c_star = lorentz_apply(u_star)
    q = u0 + 0.5 * DT * r_n + 0.5 * DT * r_star + 0.5 * DT * c_star
    u_ref = lorentz_solve(q, 0.5 * DT)

    err = float(np.abs(u_pc - u_ref).max())
    print("pure-Module operator-first predictor-corrector vs offline reference: max|d| = %.2e" % err)
    ok = err < 1e-10
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

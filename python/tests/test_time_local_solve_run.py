#!/usr/bin/env python3
"""pops.time Phase-4b LOCAL LINEAR SOLVE codegen, end to end (epic ADC-399 / ADC-403).

`emit_cpp_program` now LOWERS the Phase-4 IR ops -- ``source`` (a named ``m.source_term``),
``apply`` (LU for a named ``m.linear_source``) and ``solve_local_linear`` ((I -/+ a*L) U = rhs solved
cell by cell via a dense per-cell inverse) -- reusing ProgramContext + for_each_cell + the existing
``pops::detail::mat_inverse`` (no flux / solver reimplementation, no heap in the device kernel).

(A) Codegen (pure Python, always runs): the generated C++ of a Lorentz ``solve_local_linear`` contains
    the per-cell dense-inverse kernel (M = I - dt*L assembled from the aux B_z, mat_inverse, the
    matvec onto the rhs state); the n_cons > 8 dense-fallback guard fires; the Phase-4b ops are refused
    without a model.

(B) End-to-end Lorentz parity (skips unless the full toolchain is present): a 3-variable model
    (rho, mx, my) with a Lorentz ``linear_source`` L = [[0,0,0],[0,0,B_z],[0,-B_z,0]], a constant B_z,
    a non-trivial momentum IC; a Program W = solve_local_linear(I - dt*L, rhs=U); compile_problem ->
    problem.so, install_program, step(dt). The implicit Lorentz rotation has a closed form: with
    k = dt*B_z, den = 1 + k*k, mx' = (mx + k*my)/den, my' = (-k*mx + my)/den, rho unchanged. The
    stepped (mx, my) must match it to round-off. Runs in CI (gate-python rebuilds _pops) and locally
    once _pops is rebuilt; skips if _pops lacks install_program, numpy/_pops is absent, no compiler/Kokkos
    is visible, or the .so compile fails -- never faking the engine.
"""
import sys


def _skip(msg):
    print("skip test_time_local_solve_run (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
    from pops import dsl
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001  -- numpy or _pops unavailable in this interpreter
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001  -- wrong exception type is a failure, not a pass
        return False
    return False


def lorentz_model(name="lorentz_local"):
    """Isothermal 2D fluid (rho, mx, my; p = cs2*rho) with a Lorentz linear source L(B_z).

    A complete, compilable production block (flux + primitives + eigenvalues). The Program never runs
    a transport rhs: it only solves the LOCAL implicit Lorentz operator, which acts on (mx, my)
    independently of the flux. B_z is read off the System aux."""
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.param("cs2", 0.5)  # const sound speed^2
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = dsl.sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    bz = m.aux("B_z")
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    return m


def lorentz_program(name="lorentz_step"):
    """W = (I - dt*L_lorentz)^{-1} U, committed: one implicit Lorentz rotation of the momentum."""
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("plasma")
    Q = P.linear_combine("Q", 1.0 * U)  # a State scratch == U (the solve rhs)
    W = P.solve_local_linear(name="W", operator=P.I - dt * P.linear_source("lorentz"), rhs=Q)
    P.commit("plasma", W)
    return P


# ---- (A) codegen: pure Python, always runs ----
print("== (A) solve_local_linear codegen ==")
m = lorentz_model()
src = lorentz_program().emit_cpp_program(model=m)
for frag in ("pops::for_each_cell(", "pops::detail::mat_inverse<3>(", "pops::Real M_[3][3];",
             "pops::Real Minv_[3][3];", "auxA(i, j, 3)", "ctx.aux()"):
    chk(frag in src, "generated solve_local_linear kernel has %r" % frag)
chk("a_ * (B_z)" in src and "a_ * ((-B_z))" in src,
    "the Lorentz operator M = I - dt*L reads +/- B_z off the aux")

# Phase-4b ops are refused without the physical model (cannot read the coefficients).
chk(raises(NotImplementedError, lambda: lorentz_program().emit_cpp_program()),
    "solve_local_linear refused without a model")

# n_cons > 8 dense-fallback guard.
big = dsl.Model("too_big")
cons = big.conservative_vars(*["c%d" % i for i in range(9)])
big.aux("B_z")
zero9 = [[0.0] * 9 for _ in range(9)]
big.linear_source("L", zero9)
Pbig = adctime.Program("big")
Ub = Pbig.state("blk")
Qb = Pbig.linear_combine("Qb", 1.0 * Ub)
Pbig.commit("blk", Pbig.solve_local_linear(name="Wb", operator=Pbig.I - Pbig.dt * Pbig.linear_source("L"),
                                           rhs=Qb))
chk(raises(ValueError, lambda: Pbig.emit_cpp_program(model=big)),
    "n_cons > 8 dense-fallback guard fires")

# ---- (B) end-to-end Lorentz parity: skips unless the full toolchain is present ----
if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
    print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
    print("%s test_time_local_solve_run (A only)" % ("FAIL" if fails else "PASS"))
    sys.exit(1 if fails else 0)

print("== (B) end-to-end: implicit Lorentz solve vs analytic rotation ==")


def make_sim():
    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)
    # Production-backend DSL model added as a native block; the Program drives the step.
    try:
        compiled_model = lorentz_model("lorentz_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        _skip("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("plasma", compiled_model,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method="euler"))
    bz = 3.0
    sim.set_magnetic_field(bz * np.ones(n * n))  # constant B_z over the grid
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    mx = 0.5 * rho
    my = -0.2 * rho
    sim.set_state("plasma", np.stack([rho, mx, my]))
    return sim, bz, np.stack([rho, mx, my])


dt = 0.05

try:
    compiled = pops.compile_problem(model=lorentz_model("lorentz_prog"), time=lorentz_program())
except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

chk(compiled.program_name == "lorentz_step", "handle carries the program name")

prog, bz, U0 = make_sim()
prog.install_program(compiled.so_path)  # dlopen + ABI-key check + pops_install_program(this)
prog.step(dt)
U = np.array(prog.get_state("plasma"))

# Analytic implicit Lorentz rotation of (mx, my): (I - dt*L) U' = U, L = [[0,0,0],[0,0,b],[0,-b,0]].
k = dt * bz
den = 1.0 + k * k
rho0, mx0, my0 = U0[0], U0[1], U0[2]
mx_ref = (mx0 + k * my0) / den
my_ref = (-k * mx0 + my0) / den

e_rho = float(np.abs(U[0] - rho0).max())
e_mx = float(np.abs(U[1] - mx_ref).max())
e_my = float(np.abs(U[2] - my_ref).max())
e_lorentz = max(e_mx, e_my)
print("  Lorentz parity: max|d(rho)| = %.2e  max|d(mx,my)| = %.2e" % (e_rho, e_lorentz))
chk(e_rho < 1e-13, "rho unchanged by the Lorentz solve (max|d| = %.2e)" % e_rho)
chk(e_lorentz < 1e-12, "stepped (mx, my) == analytic implicit rotation (max|d| = %.2e)" % e_lorentz)
chk(float(np.abs(U[1] - mx0).max()) > 1e-6, "the step actually rotated the momentum")

print("%s test_time_local_solve_run" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)

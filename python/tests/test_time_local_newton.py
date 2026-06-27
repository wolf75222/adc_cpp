#!/usr/bin/env python3
"""pops.time LOCAL NON-LINEAR SOLVE codegen, end to end (epic ADC-399 / ADC-422).

`P.solve_local_nonlinear` (spec op 10) now LOWERS a per-cell Newton iteration: from the initial guess
U0, ``emit_cpp_program`` emits a device kernel that re-evaluates an inlined residual ``r(U)`` (built
from the residual sub-block -- named ``source`` / ``apply`` per-cell Exprs + the iterate / frozen guess
+ affine combines), forms an in-kernel finite-difference Jacobian, and solves the Newton step
``J dU = -r`` with the SAME stack dense inverse ``pops::detail::mat_inverse<N>`` ``solve_local_linear``
uses -- iterating to ``max_c |r_c| < tol`` or the budget. No heap / std::function / Eigen in the kernel
(only stack scalars + fixed ``[N]`` / ``[N][N]`` arrays).

(A) Validation + codegen (pure Python, always runs): the builder rejects a non-callable residual, a
    non-State guess, a non-positive max_iter, a non-local residual op, and (with n_cons > 8) the dense
    fallback; a valid implicit reaction lowers to a per-cell Newton kernel whose generated C++ has the
    residual lambda, the FD Jacobian, the mat_inverse step and the convergence break; refused w/o model.

(B) End-to-end scalar implicit reaction parity (skips unless the full toolchain is present): a
    1-variable model (rho) with a NON-LINEAR named source ``S(rho) = -k*rho^2``; a Program W solving
    ``r(rho) = rho - rho0 - dt*S(rho) = 0`` per cell; compile_problem -> problem.so, install_program,
    step(dt). The implicit step has the closed form rho = (-1 + sqrt(1 + 4*dt*k*rho0))/(2*dt*k); the
    stepped rho must match it AND an offline numpy Newton on the identical residual to ~1e-10, with the
    offline Newton taking > 1 iteration and its residual dropping by many orders. Skips (exit 0) without
    numpy / _pops / a compiler / a visible Kokkos, or if the .so compile fails -- never faking the engine.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # noqa: BLE001 -- pops not importable here -> skip, never fake
        print("skip test_time_local_newton (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _skip(msg):
    print("skip test_time_local_newton (%s)" % msg)
    sys.exit(0)


def raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return True
    except Exception:  # noqa: BLE001 -- a wrong exception type is a failure, not a pass
        return False
    return False


# --- a 1-variable model with a NON-LINEAR named source S(rho) = -k*rho^2 (a Riccati-type reaction) ---
def reaction_model(name, k):
    """rho only, ZERO flux, a NAMED non-linear source ``react`` = -k*rho^2 (the implicit step rotates
    no transport: the Program drives only the LOCAL non-linear solve). A complete compilable block
    (flux + primitive + eigenvalue + named source_term)."""
    from pops.physics.facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source_term("react", [-k * rho * rho])  # S(rho) = -k rho^2
    return m


def reaction_program(t, name="implicit_reaction"):
    """W = the per-cell solution of r(rho) = rho - rho0 - dt*S(rho) = 0 (an implicit Euler reaction
    step). The residual is built from the named source ``react`` + the iterate / frozen guess."""
    P = t.Program(name)
    dt = P.dt
    U = P.state("blk")

    def residual(P, Uit, U0):
        S = P.source("react", state=Uit)  # S(U) = -k U^2 (named non-linear source)
        return P.linear_combine("r", Uit - U0 - dt * S)  # r = U - U0 - dt*S(U)

    W = P.solve_local_nonlinear(name="W", residual=residual, initial_guess=U,
                                tol=1e-12, max_iter=50)
    P.commit("blk", W)
    return P


fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


# ============================ (A) validation + codegen (pure Python) ============================
def section_a(t):
    print("== (A) solve_local_nonlinear validation + codegen ==")
    try:
        from pops.physics.facade import Model
    except Exception as exc:  # noqa: BLE001 -- dsl needs _pops; A still skips cleanly, never fakes
        print("-- (A) skipped: pops.dsl unavailable (%s) --" % exc)
        return

    # --- builder validation ---
    P = t.Program("v")
    U = P.state("blk")
    chk(raises(ValueError, lambda: P.solve_local_nonlinear(residual=U, initial_guess=U)),
        "a non-callable residual is rejected")

    def resid(P, Uit, U0):
        return P.linear_combine(Uit - U0)
    chk(raises(ValueError, lambda: P.solve_local_nonlinear(residual=resid, initial_guess="x")),
        "a non-State initial_guess is rejected")
    chk(raises(ValueError, lambda: P.solve_local_nonlinear(residual=resid, initial_guess=U, max_iter=0)),
        "max_iter <= 0 is rejected")
    chk(raises((ValueError, NotImplementedError),
               lambda: P.solve_local_nonlinear(residual=resid, initial_guess=U, method="broyden")),
        "an unsupported method is rejected")

    # A non-local residual op (P.rhs carries a divergence / halo) cannot live in a per-cell kernel.
    def bad_resid(P, Uit, U0):
        R = P.rhs(state=Uit, sources=["default"])
        return P.linear_combine(Uit - U0 - P.dt * R)
    chk(raises(ValueError, lambda: P.solve_local_nonlinear(residual=bad_resid, initial_guess=U)),
        "a non-local residual op (P.rhs) is rejected")

    # --- a valid Newton IR validates + hashes; the residual sub-block is recorded ---
    Pr = reaction_program(t)
    chk(Pr.validate() is True, "the implicit-reaction Newton IR validates")
    chk(bool(Pr._ir_hash()), "the Newton IR serializes to a stable hash")
    nl = [v for v in Pr._values if v.op == "solve_local_nonlinear"][0]
    chk(nl.attrs["max_iter"] == 50 and nl.attrs["tol"] == 1e-12, "tol / max_iter recorded on the op")
    chk(len(nl.attrs["residual_block"]) >= 3,
        "the residual sub-block holds the iterate + guess + ops")

    # --- the IR hash is sensitive to the Newton parameters (a different tol / max_iter rehashes) ---
    def _h(tol, mi):
        Q = t.Program("h")
        dt = Q.dt
        u = Q.state("blk")

        def r(Q, Uit, U0):
            return Q.linear_combine(Uit - U0 - dt * Q.source("react", state=Uit))
        Q.commit("blk", Q.solve_local_nonlinear(name="W", residual=r, initial_guess=u,
                                                 tol=tol, max_iter=mi))
        return Q._ir_hash()
    chk(_h(1e-10, 20) != _h(1e-8, 20), "a different tol rehashes the IR")
    chk(_h(1e-10, 20) != _h(1e-10, 30), "a different max_iter rehashes the IR")

    # --- the codegen lowers a per-cell Newton kernel ---
    m = reaction_model("react_cg", 2.0)
    src = reaction_program(t, "react_cg").emit_cpp_program(model=m)
    for frag in ("auto residual_eval = [&]", "pops::detail::mat_inverse<1>(",
                 "for (int it_ = 0;", "J_[1][1]", "std::fmax(rmax_, std::fabs(r_",
                 "if (rmax_ < static_cast<pops::Real>(1e-12)) break;",
                 "const pops::Real eps_", "U_[i_] -= du_;", "pops::for_each_cell("):
        chk(frag in src, "the Newton kernel has %r" % frag)
    # The residual is the affine r = U - U0 - dt*S(U); S(U) = -k U^2 reads the iterate stack.
    chk("Gval[0] = u" in src, "the frozen guess is read into a stack vector")
    chk("U_[0] = Gval[0]" in src, "the Newton iterate is seeded to the guess")
    chk("rout[0] =" in src and "Ueval[0]" in src, "the residual is re-evaluated at the iterate")
    # No forbidden constructs in the device kernel.
    for forbidden in ("std::function", "std::vector", "Eigen::", "new ", "malloc"):
        chk(forbidden not in src, "the Newton kernel has no %r (device-clean)" % forbidden)

    # --- refused without a model (the residual's named source needs the model coefficients) ---
    chk(raises(NotImplementedError, lambda: reaction_program(t, "react_nm").emit_cpp_program()),
        "the Newton codegen is refused without a model")

    # --- n_cons > 8 dense-fallback guard fires (the FD Jacobian is a fixed N x N stack inverse) ---
    big = Model("too_big_nl")
    cons = big.conservative_vars(*["c%d" % i for i in range(9)])
    big.source_term("react", [-1.0 * c for c in cons])
    Pbig = t.Program("big_nl")
    Ub = Pbig.state("blk")

    def big_resid(P, Uit, U0):
        return P.linear_combine(Uit - U0 - P.dt * P.source("react", state=Uit))
    Pbig.commit("blk", Pbig.solve_local_nonlinear(name="W", residual=big_resid, initial_guess=Ub))
    chk(raises(ValueError, lambda: Pbig.emit_cpp_program(model=big)),
        "n_cons > 8 dense-fallback guard fires")


# ============================ (B) end-to-end implicit-reaction parity ============================
def section_b(t):
    try:
        import numpy as np

        import pops
        from pops.physics.facade import Model
    except Exception as exc:  # noqa: BLE001
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return

    if not hasattr(pops.System(n=8, L=1.0, periodic=True), "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return

    print("== (B) end-to-end: per-cell implicit reaction vs analytic + offline Newton ==")
    k = 1.5            # reaction rate
    dt = 0.2           # a big step so the implicit move is non-trivial (Newton must iterate)
    n = 16

    # ---- offline numpy Newton on the IDENTICAL per-cell residual r(U) = U - U0 - dt*S(U), S = -k U^2 ----
    def offline_newton(u0, tol=1e-12, max_iter=50):
        u = np.array(u0, dtype=float, copy=True)
        iters = np.zeros_like(u, dtype=int)
        first_res = None
        last_res = None
        for _ in range(max_iter):
            r = u - u0 - dt * (-k * u * u)           # r(U)
            res = np.abs(r)
            if first_res is None:
                first_res = float(res.max())
            last_res = float(res.max())
            if res.max() < tol:
                break
            jac = 1.0 - dt * (-2.0 * k * u)          # r'(U) = 1 + 2*dt*k*U
            u = u - r / jac
            iters += (res >= tol).astype(int)
        return u, iters, first_res, last_res

    # ---- compile the Program + a native reaction block, run one implicit step ----
    try:
        compiled = pops.compile_problem(model=reaction_model("react_prog", k),
                                       time=reaction_program(t, "react_step"))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

    chk(compiled.program_name == "react_step", "handle carries the program name")

    sim = pops.System(n=n, L=1.0, periodic=True)
    try:
        compiled_model = reaction_model("react_block", k).compile(backend="production")
    except RuntimeError as exc:
        _skip("model compile could not build the .so: %s" % str(exc)[:160])
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))

    # A KNOWN positive field with spatial variation (each cell solves its own scalar Newton).
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # in [0.5, 1.5]
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    sim.step(dt)
    rho = np.array(sim.get_state("blk"))[0]

    # ---- references ----
    # Closed form: dt*k*U^2 + U - U0 = 0 -> U = (-1 + sqrt(1 + 4*dt*k*U0)) / (2*dt*k)  (positive root).
    a = dt * k
    rho_closed = (-1.0 + np.sqrt(1.0 + 4.0 * a * rho0)) / (2.0 * a)
    rho_newton, iters, first_res, last_res = offline_newton(rho0)

    e_closed = float(np.abs(rho - rho_closed).max())
    e_newton = float(np.abs(rho - rho_newton).max())
    moved = float(np.abs(rho - rho0).max())
    max_iters = int(iters.max())
    # Residual of the COMPILED solution against the per-cell residual (must be ~0 at convergence).
    res_compiled = float(np.abs(rho - rho0 - dt * (-k * rho * rho)).max())

    print("  implicit reaction: max|rho-closed| = %.2e  max|rho-offline_newton| = %.2e" %
          (e_closed, e_newton))
    print("  offline Newton: iters(max) = %d  residual %.2e -> %.2e  |compiled residual| = %.2e  moved %.2e"
          % (max_iters, first_res, last_res, res_compiled, moved))

    chk(e_closed < 1e-10, "stepped rho == analytic implicit reaction (max|d| = %.2e)" % e_closed)
    chk(e_newton < 1e-10, "stepped rho == offline numpy Newton (max|d| = %.2e)" % e_newton)
    chk(res_compiled < 1e-10, "the compiled solution drives the per-cell residual to ~0 (%.2e)"
        % res_compiled)
    chk(max_iters > 1, "the Newton solve takes more than one iteration (%d)" % max_iters)
    chk(last_res < 1e-12 and first_res > 1e-3,
        "the residual drops from O(%.1e) to below tol (Newton converges)" % first_res)
    chk(moved > 1e-2, "the implicit step actually moved the state (max|d| = %.2e)" % moved)


def _run():
    t = _pops_time()
    section_a(t)
    section_b(t)
    print("%s test_time_local_newton" % ("FAIL (%d)" % fails if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    _run()

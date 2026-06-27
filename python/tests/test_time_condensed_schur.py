#!/usr/bin/env python3
"""pops.time condensed-Schur implicit source stage as a compiled Program (epic ADC-399 / ADC-421).

ADC-421 adds the anisotropic position-dependent operator-coefficient assembly the condensed-Schur
operator needs: ``P.schur_coeffs`` assembles the per-cell tensor ``A = I + c*rho*B^{-1}`` (the native
detail::SchurOperatorCoeffKernel), ``P.apply_laplacian_coeff`` applies ``div(A grad phi)`` matrix-free
(pops::apply_laplacian's coefficient path), ``P.schur_rhs`` assembles the fused RHS
``-Lap(phi^n) - theta*dt*alpha*div(B^{-1}(mx,my))`` (the native assemble_rhs), and
``P.schur_reconstruct`` reconstructs ``v = B^{-1}(v^n - theta*dt*grad phi)`` (the closed B^{-1}). The
``pops.lib.time.std.condensed_schur`` macro composes them with ``P.solve_linear`` (matrix-free BiCGStab) into
the same assemble / solve / reconstruct sequence as the native CondensedSchurSourceStepper (epic
acceptance 32). The native ``pops.CondensedSchur`` stepper is untouched.

ADC-427 extends the macro to theta != 1: the n+1 extrapolation by factor 1/theta is lowered with the
EXISTING affine algebra (no component-restricted IR op) because schur_reconstruct freezes rho, so the
plain state affine ``U^n + (1/theta)(U^{n+theta} - U^n)`` leaves rho untouched and equals the native
momentum-only extrapolation; an OPTIONAL energy component (c_E) adds the native kinetic-energy increment
via a new P.schur_energy op. The cross-step persistent-phi carry stays deferred (it needs a 1-component
history runtime path); each step solves from phi^n = 0.

(A) Pure Python, always runs:
    - the builder ops record + validate their operands and serialize;
    - the ``std.condensed_schur`` macro lowers theta == 1 (backward Euler, historical IR byte-identical)
      AND theta < 1 (the 1/theta extrapolation as a copy-then-reconstruct + affine combine);
    - an energy component lowers the P.schur_energy op; theta out of (0, 1] raises ValueError.

(B) End-to-end parity (skips unless the full toolchain is present): the macro is compiled + installed +
    one step is taken on a field-coupled rho/mx/my block with a constant B_z, for theta == 1 AND
    theta == 0.5, then compared to an OFFLINE numpy reference of the IDENTICAL discrete steps (the same
    anisotropic 5-point operator with harmonic face means + arithmetic cross means, the same centered-
    divergence RHS, the same closed B^{-1} reconstruction, BiCGStab from phi^n = 0, the same 1/theta
    extrapolation). Asserts max|compiled - offline| <= 1e-6 for both thetas.

    DOCUMENTED GAP vs the native pops.CondensedSchur: the native solve is BiCGStab + a GeometricMG
    preconditioner while the Program solve is matrix-free BiCGStab WITHOUT a preconditioner -- the same
    operator and RHS, a different Krylov path. Both converge to the same phi at tolerance, so the firm
    parity is checked against the matrix-free-equivalent offline reference (not bit-against-native); a
    native pops.CondensedSchur(theta=0.5) step is also REPORTED as a diagnostic (it is confounded by the
    explicit transport half-flow of pops.Split, so it is not asserted). The cross-step phi^n carry is
    deferred (see the macro docstring).

Self-skips (exit 0) without numpy / _pops / install_program / a compiler / a visible Kokkos -- never
fakes the engine (project policy: no fake pops in tests).
"""
import sys


def _pops_time():
    global lt  # ready schemes live in pops.lib.time (Spec 4)
    try:
        import pops.time as t
        import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_condensed_schur (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_N = 16
_L = 1.0
_DT = 0.05
_ALPHA = 1.0
_BZ = 0.7
_THETA = 1.0
_TOL = 1e-10


# ---- (A) builder ops + macro lowering: pure Python, always runs ----
def test_schur_coeffs_records_and_validates(t):
    P = t.Program("p")
    U = P.state("blk")
    coeffs = P.schur_coeffs(state=U, c=0.25, th_dt=0.5, c_rho=0, c_bz=3)
    assert coeffs.vtype == "schur_coeffs", "schur_coeffs yields a schur_coeffs bundle value"
    assert coeffs.attrs["c_rho"] == 0 and coeffs.attrs["c_bz"] == 3
    # a number c / th_dt are stored as dt-polynomials (power 0).
    assert coeffs.attrs["c"] == {0: 0.25} and coeffs.attrs["th_dt"] == {0: 0.5}


def test_schur_coeffs_operand_types(t):
    P = t.Program("p")
    U = P.state("blk")
    g = P.scalar_field("g")
    bad = []
    for kw in (dict(state=g, c=1.0, th_dt=1.0),           # a scalar_field is not a State
               dict(state=U, c="x", th_dt=1.0),           # c not a number / dt-poly
               dict(state=U, c=1.0, th_dt=1.0, c_rho=-1)):  # negative component
        try:
            P.schur_coeffs(**kw)
        except ValueError:
            bad.append(True)
        else:
            bad.append(False)
    assert all(bad), "schur_coeffs must reject a non-State, a non-numeric coeff and a negative comp"


def test_apply_laplacian_coeff_operand_types(t):
    P = t.Program("p")
    U = P.state("blk")
    A = P.matrix_free_operator("A")
    seen = []

    def apply(P, out, x):
        coeffs = P.schur_coeffs(state=U, c=1.0, th_dt=1.0)
        try:
            P.apply_laplacian_coeff(out, U, coeffs)  # in_ must be a scalar_field, not a State
        except ValueError:
            seen.append(True)
        else:
            seen.append(False)
        lap = P.scalar_field("lap")
        P.apply_laplacian_coeff(lap, x, coeffs)  # valid
        return -1.0 * lap

    P.set_apply(A, apply)
    assert seen and all(seen), "apply_laplacian_coeff rejects a non-scalar_field in_"


def test_schur_rhs_and_reconstruct_record(t):
    P = t.Program("p")
    U = P.state("blk")
    phi_n = P.scalar_field("phi_n")
    rhs = P.scalar_field("rhs")
    r = P.schur_rhs(rhs, phi_n, U, th_dt=0.5, g=0.25, c_mx=1, c_my=2, c_bz=3)
    assert r.vtype == "scalar_field" and r.attrs["g"] == {0: 0.25}
    out = P.schur_reconstruct(state=U, phi=phi_n, th_dt=0.5, c_rho=0, c_mx=1, c_my=2, c_bz=3)
    assert out.vtype == "state" and out.block == "blk"


def test_condensed_schur_macro_lowers(t):
    P = t.Program("cs")
    lt.std.condensed_schur(P, "blk", alpha=_ALPHA, theta=1.0)
    assert P.validate() is True, "the condensed-Schur macro must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"
    src = P.emit_cpp_program()
    for frag in ("ctx.solve_fields_from_state", "ctx.assemble_schur_coeffs", "ctx.assemble_schur_rhs",
                 "ctx.apply_laplacian_coeff", "pops::bicgstab_solve", "ctx.schur_reconstruct"):
        assert frag in src, "the condensed-Schur macro must contain %r\n%s" % (frag, src)


def test_condensed_schur_theta_half_lowers(t):
    """ADC-427: theta != 1 now lowers (the n+1 extrapolation by factor 1/theta is the affine algebra,
    no component-restricted IR op). The macro reconstructs on a COPY of U^n so the extrapolation can
    read mom^n, then commits U^n + (1/theta)(U^{n+theta} - U^n)."""
    P = t.Program("cs")
    lt.std.condensed_schur(P, "blk", alpha=_ALPHA, theta=0.5)
    assert P.validate() is True, "the theta=0.5 condensed-Schur macro must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"
    src = P.emit_cpp_program()
    for frag in ("ctx.assemble_schur_coeffs", "ctx.assemble_schur_rhs", "pops::bicgstab_solve",
                 "ctx.schur_reconstruct"):
        assert frag in src, "the theta=0.5 macro must contain %r\n%s" % (frag, src)
    # th_dt = theta*dt is lowered into the reconstruction; the extrapolation is an axpy(2.0, ...) (1/0.5).
    assert "0.5 * dt" in src, "th_dt = theta*dt must reach the reconstruction\n%s" % src
    assert "static_cast<pops::Real>(2.0)" in src, "the 1/theta extrapolation must axpy by 2.0\n%s" % src


def test_condensed_schur_theta_out_of_range_raises(t):
    for bad in (0.0, -0.5, 1.5):
        try:
            lt.std.condensed_schur(t.Program("p"), "blk", alpha=1.0, theta=bad)
        except ValueError as exc:
            assert "theta must be in (0, 1]" in str(exc), str(exc)
        else:
            raise AssertionError("condensed_schur(theta=%r) must raise ValueError" % bad)


def test_condensed_schur_energy_lowers(t):
    """ADC-427: an energy component (c_E) adds the native kinetic-energy increment via P.schur_energy."""
    P = t.Program("cs")
    lt.std.condensed_schur(P, "blk", alpha=_ALPHA, theta=0.5, c_E=3)
    assert P.validate() is True
    src = P.emit_cpp_program()
    assert "ctx.schur_energy" in src, "the energy variant must emit ctx.schur_energy\n%s" % src


def test_condensed_schur_theta_one_ir_unchanged(t):
    """ADC-427 no-regression: theta == 1 keeps its historical IR (reconstruct IN PLACE on U^n, no copy /
    extrapolation / energy op), so an existing theta==1 program's .so cache key is byte-identical."""
    P = t.Program("cs")
    lt.std.condensed_schur(P, "blk", alpha=_ALPHA, theta=1.0)
    src = P.emit_cpp_program()
    assert "ctx.schur_energy" not in src, "theta=1 must NOT emit an energy op"
    # No copy-then-reconstruct: the reconstruction writes U^n in place, the commit is the reconstruction.
    assert src.count("ctx.schur_reconstruct") == 1, src
    assert "static_cast<pops::Real>(2.0)" not in src, "theta=1 must NOT emit a 1/theta extrapolation"


# ---- offline reference of the identical discrete steps (numpy, periodic) ----
def _binv(theta_dt, bz):
    """Closed B^{-1} = (1/det)[[1, w],[-w, 1]], w = theta*dt*B_z, det = 1 + w^2 (LorentzEliminator)."""
    w = theta_dt * bz
    det = 1.0 + w * w
    return (1.0 / det, w / det, -w / det, 1.0 / det)  # (b11, b12, b21, b22)


def _eps_harmonic(a, b):
    s = a + b
    import numpy as np
    return np.where(s > 0.0, 2.0 * a * b / s, 0.0)


def _apply_aniso(phi, eps_x, eps_y, a_xy, a_yx, h):
    """div(A grad phi) on a periodic grid -- the exact pops::apply_laplacian coefficient stencil:
    harmonic face means for the diagonal eps, arithmetic face means for the cross terms (cross_div)."""
    import numpy as np

    idx2 = idy2 = 1.0 / (h * h)
    idx = idy = 1.0 / h
    exm = _eps_harmonic(eps_x, np.roll(eps_x, 1, 0))   # x- face (between i-1 and i)
    exp = _eps_harmonic(eps_x, np.roll(eps_x, -1, 0))  # x+ face
    eym = _eps_harmonic(eps_y, np.roll(eps_y, 1, 1))   # y- face
    eyp = _eps_harmonic(eps_y, np.roll(eps_y, -1, 1))  # y+ face
    wxm, wxp, wym, wyp = exm * idx2, exp * idx2, eym * idy2, eyp * idy2
    lap = (wxp * np.roll(phi, -1, 0) + wxm * np.roll(phi, 1, 0) +
           wyp * np.roll(phi, -1, 1) + wym * np.roll(phi, 1, 1) -
           (wxm + wxp + wym + wyp) * phi)
    # cross fluxes (arithmetic face mean of a_xy / a_yx; 4-corner tangential gradient), cross_div().
    axy_xp = 0.5 * (a_xy + np.roll(a_xy, -1, 0))
    axy_xm = 0.5 * (a_xy + np.roll(a_xy, 1, 0))
    dyf_xp = (np.roll(phi, -1, 1) + np.roll(np.roll(phi, -1, 0), -1, 1) -
              np.roll(phi, 1, 1) - np.roll(np.roll(phi, -1, 0), 1, 1)) * (0.25 * idy)
    dyf_xm = (np.roll(np.roll(phi, 1, 0), -1, 1) + np.roll(phi, -1, 1) -
              np.roll(np.roll(phi, 1, 0), 1, 1) - np.roll(phi, 1, 1)) * (0.25 * idy)
    lap = lap + (axy_xp * dyf_xp - axy_xm * dyf_xm) * idx
    ayx_yp = 0.5 * (a_yx + np.roll(a_yx, -1, 1))
    ayx_ym = 0.5 * (a_yx + np.roll(a_yx, 1, 1))
    dxf_yp = (np.roll(phi, -1, 0) + np.roll(np.roll(phi, -1, 1), -1, 0) -
              np.roll(phi, 1, 0) - np.roll(np.roll(phi, -1, 1), 1, 0)) * (0.25 * idx)
    dxf_ym = (np.roll(np.roll(phi, 1, 1), -1, 0) + np.roll(phi, -1, 0) -
              np.roll(np.roll(phi, 1, 1), 1, 0) - np.roll(phi, 1, 0)) * (0.25 * idx)
    lap = lap + (ayx_yp * dxf_yp - ayx_ym * dxf_ym) * idy
    return lap


def _offline_step(U0, alpha, theta, bz, h, dt, tol):
    """Offline replay with an EXPLICIT dt, mirroring the generated C++ exactly (phi^n = 0). For
    theta < 1 it also applies the n+1 extrapolation v^{n+1} = v^n + (1/theta)(v^{n+theta} - v^n) =
    the macro's affine ``U^n + (1/theta)(U^{n+theta} - U^n)`` (rho frozen)."""
    import numpy as np

    rho, mx, my = U0[0].copy(), U0[1].copy(), U0[2].copy()
    th_dt = theta * dt
    g = theta * dt * alpha
    c = theta * theta * dt * dt * alpha
    b11, b12, b21, b22 = _binv(th_dt, bz)
    # 1) coefficients A = I + c*rho*B^{-1}.
    eps_x = 1.0 + c * rho * b11
    eps_y = 1.0 + c * rho * b22
    a_xy = c * rho * b12
    a_yx = c * rho * b21
    # 2) explicit flux F = B^{-1}(mx, my); RHS = -Lap(0) - g*div(F) (centered divergence).
    Fx = b11 * mx + b12 * my
    Fy = b21 * mx + b22 * my
    divF = (np.roll(Fx, -1, 0) - np.roll(Fx, 1, 0)) / (2 * h) + \
           (np.roll(Fy, -1, 1) - np.roll(Fy, 1, 1)) / (2 * h)
    rhs = -g * divF
    # 3) solve -div(A grad phi) = rhs  <=>  apply(phi) = -div(A grad phi) = rhs, matrix-free BiCGStab.
    def apply(phi):
        return -_apply_aniso(phi, eps_x, eps_y, a_xy, a_yx, h)
    phi, iters = _np_bicgstab(apply, rhs, tol=tol)
    # 4) reconstruct v^{n+theta} = B^{-1}(v^n - theta*dt*grad phi); mom = rho*v (rho frozen).
    inv_rho = np.where(rho != 0.0, 1.0 / rho, 0.0)
    vx = mx * inv_rho
    vy = my * inv_rho
    gx = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2 * h)
    gy = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2 * h)
    ax = vx - th_dt * gx
    ay = vy - th_dt * gy
    nx = b11 * ax + b12 * ay  # v^{n+theta}
    ny = b21 * ax + b22 * ay
    # 5) n+1 extrapolation (theta < 1): v^{n+1} = v^n + (1/theta)(v^{n+theta} - v^n). theta == 1 -> id.
    inv_theta = 1.0 / theta
    nx = vx + inv_theta * (nx - vx)
    ny = vy + inv_theta * (ny - vy)
    return np.stack([rho, rho * nx, rho * ny]), iters


def _np_bicgstab(apply, b, *, tol=1e-10, max_iter=1000):
    """Plain numpy unpreconditioned BiCGStab solving A x = b from x = 0 (matches pops::bicgstab_solve
    with an identity preconditioner -- the Program's solve path)."""
    import numpy as np

    x = np.zeros_like(b)
    r = b - apply(x)
    r0 = r.copy()
    rho_old = alpha_ = omega = 1.0
    v = p = np.zeros_like(b)
    bnorm = float(np.sqrt(np.sum(b * b))) or 1.0
    iters = 0
    for _ in range(max_iter):
        rho_new = float(np.sum(r0 * r))
        if abs(rho_new) < 1e-300:
            break
        beta = (rho_new / rho_old) * (alpha_ / omega)
        p = r + beta * (p - omega * v)
        v = apply(p)
        denom = float(np.sum(r0 * v))
        if abs(denom) < 1e-300:
            break
        alpha_ = rho_new / denom
        s = r - alpha_ * v
        if float(np.sqrt(np.sum(s * s))) <= tol * bnorm:
            x = x + alpha_ * p
            iters += 1
            break
        tt = apply(s)
        tt2 = float(np.sum(tt * tt))
        omega = float(np.sum(tt * s)) / tt2 if tt2 > 1e-300 else 0.0
        x = x + alpha_ * p + omega * s
        r = s - omega * tt
        rho_old = rho_new
        iters += 1
        if float(np.sqrt(np.sum(r * r))) <= tol * bnorm:
            break
    return x, iters


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _run_section_b(t):
    try:
        import numpy as np

        import pops
        from pops.ir.ops import sqrt
        from pops.physics.facade import Model
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable here
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    sim_probe = pops.System(n=8, L=_L, periodic=True)
    if not hasattr(sim_probe, "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None
    if not hasattr(sim_probe, "set_magnetic_field"):
        print("-- (B) skipped: _pops lacks set_magnetic_field (rebuild _pops) --")
        return None

    def schur_model(name):
        """Isothermal 2D fluid block (rho, mx, my) with a Poisson coupling + a B_z aux: the canonical
        condensed-Schur block (Density / MomentumX / MomentumY roles + B_z)."""
        m = Model(name)
        rho, mx, my = m.conservative_vars("rho", "mx", "my")
        cs2 = m.param("cs2", 0.5)
        u = m.primitive("u", mx / rho)
        v = m.primitive("v", my / rho)
        p = m.primitive("p", cs2 * rho)
        m.primitive_vars(rho=rho, u=u, v=v, p=p)
        m.conservative_from([rho, rho * u, rho * v])
        m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
        cs = sqrt(cs2)
        m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
        m.elliptic_rhs(rho)
        m.aux("grad_x")
        m.aux("grad_y")
        m.aux("B_z")
        return m

    def make_sim(name):
        sim = pops.System(n=_N, L=_L, periodic=True)
        try:
            compiled_model = schur_model(name).compile(backend="production")
        except RuntimeError as exc:  # no compiler / no Kokkos visible
            print("-- (B) skipped: model compile could not build the .so: %s --" % str(exc)[:160])
            return None, None
        sim.add_equation("blk", compiled_model,
                         spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                         time=pops.Explicit(method="euler"))
        sim.set_poisson("charge_density", "geometric_mg")
        sim.set_magnetic_field(_BZ * np.ones(_N * _N))
        x = (np.arange(_N) + 0.5) / _N
        X, Y = np.meshgrid(x, x, indexing="ij")
        rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        mx0 = 0.4 * rho0
        my0 = -0.2 * rho0
        U0 = np.stack([rho0, mx0, my0])
        sim.set_state("blk", U0)
        return sim, U0

    h = _L / _N

    def compiled_vs_offline(theta):
        """One compiled step of std.condensed_schur(theta) vs the matrix-free offline reference."""
        sim, U0 = make_sim("cs_block_%d" % int(round(theta * 100)))
        if sim is None:
            return None
        P = t.Program("cs_step_%d" % int(round(theta * 100)))
        lt.std.condensed_schur(P, "blk", alpha=_ALPHA, theta=theta, tol=_TOL, max_iter=400)
        try:
            compiled = pops.compile_problem(model=schur_model("cs_prog_%d" % int(round(theta * 100))),
                                           time=P)
        except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
            print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
            return None
        sim.install_program(compiled.so_path)
        sim.step(_DT)
        out = np.array(sim.get_state("blk"))
        ref, iters = _offline_step(U0, _ALPHA, theta, _BZ, h, _DT, _TOL)
        err = float(np.abs(out - ref).max())
        moved = float(np.abs(out - U0).max())
        rho_drift = float(np.abs(out[0] - U0[0]).max())
        print("  compiled-vs-offline theta=%.2f: max|compiled - offline| = %.2e  iters = %d  "
              "max|U - U0| = %.2e  rho drift = %.2e" % (theta, err, iters, moved, rho_drift))
        assert err <= 1e-6, "compiled condensed-Schur(theta=%.2f) == offline reference (max|d| = " \
                            "%.2e)" % (theta, err)
        assert moved > 1e-6, "the source stage must change the momentum (theta=%.2f, max|d| = %.2e)" \
                             % (theta, moved)
        assert rho_drift < 1e-12, "rho must stay frozen (theta=%.2f, drift = %.2e)" % (theta, rho_drift)
        assert iters > 1, "the solve must take > 1 iteration (theta=%.2f), got %d" % (theta, iters)
        return out, U0, ref

    # theta == 1 (no-regression: the historical backward-Euler path) + theta == 0.5 (ADC-427).
    compiled_vs_offline(1.0)
    half = compiled_vs_offline(0.5)

    # NATIVE diagnostic (ADC-427): std.condensed_schur(theta=0.5) compiled vs pops.CondensedSchur(
    # theta=0.5) through pops.Split, taken as a SINGLE step (both start from phi^n = 0 -- the System
    # initializes phi to zero, the macro has no persistent phi carry). This is REPORTED, not asserted:
    # the native pops.Split also runs the EXPLICIT transport half-flow that the source-only Program omits,
    # so the two states differ by the transport advection (plus the MG-preconditioned vs unpreconditioned
    # BiCGStab path -- the documented ADC-421 Krylov gap). The FIRM parity is compiled-vs-offline above,
    # where the offline reference IS the source stage exactly (same matrix-free BiCGStab). Faking a tight
    # native bound here would mean asserting against a transport-confounded step -- we do not.
    if half is not None:
        out_c, U0, _ = half
        sim_n = pops.System(n=_N, L=_L, periodic=True)
        try:
            native_model = schur_model("cs_native").compile(backend="production")
        except RuntimeError as exc:
            print("-- (B) native diagnostic skipped: model compile failed: %s --" % str(exc)[:160])
            native_model = None
        if native_model is not None:
            try:
                # B_z must exist BEFORE add_equation: the CondensedSchur source stage is wired during
                # add_equation (set_source_stage), which reads the B_z aux. set_poisson + the magnetic
                # field first, then the block.
                sim_n.set_poisson("charge_density", "geometric_mg")
                sim_n.set_magnetic_field(_BZ * np.ones(_N * _N))
                sim_n.add_equation(
                    "blk", native_model,
                    spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                    time=pops.Split(hyperbolic=pops.Explicit(method="euler"),
                                   source=pops.CondensedSchur(theta=0.5, alpha=_ALPHA)))
            except Exception as exc:  # noqa: BLE001 -- Split/CondensedSchur wiring unavailable here
                print("-- (B) native diagnostic skipped: pops.Split/CondensedSchur unavailable: %s --"
                      % str(exc)[:160])
            else:
                sim_n.set_state("blk", U0)
                sim_n.step(_DT)
                out_n = np.array(sim_n.get_state("blk"))
                d_native = float(np.abs(out_c - out_n).max())
                print("  [diagnostic] compiled(theta=0.5) source-only vs native pops.CondensedSchur("
                      "theta=0.5) Split(transport+source): max|d| = %.2e  (native includes the explicit "
                      "transport half-flow + the MG-preconditioned BiCGStab path; firm parity is the "
                      "compiled-vs-offline assertion above)" % d_native)
    return half


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_condensed_schur (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()

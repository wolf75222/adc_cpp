"""pops.lib.time.strang -- Strang and Lie splitting macros, and the condensed Schur source stage.

Exports: strang, lie, condensed_schur.
"""


def strang(P, block, half_flow, source, *, commit=True):
    """Strang splitting macro H(dt/2); S(dt); H(dt/2), the macro form of pops.Strang (lowers to the SAME
    IR, no special class). @p half_flow and @p source are IR-building callables (prog, state, frac) ->
    state that advance the hyperbolic flow and the source by a fraction @p frac of dt. Returns the final
    state (committed when @p commit)."""
    U = P.state(block)
    U1 = half_flow(P, U, 0.5)
    U2 = source(P, U1, 1.0)
    U3 = half_flow(P, U2, 0.5)
    if commit:
        P.commit(block, U3)
    return U3


def lie(P, block, half_flow, source, *, commit=True):
    """Lie (Godunov) splitting macro H(dt); S(dt) -- the sequential first-order sibling of `strang`
    (ADC-423). @p half_flow and @p source are the SAME IR-building callables `strang` takes
    ``(prog, state, frac) -> state`` (each advances its sub-flow by a fraction @p frac of dt); Lie
    just composes them sequentially over the FULL step (H over dt, then S over dt) with no half-steps.
    Lowers to the SAME IR primitives as `strang` (no scheme-specific class). Returns the final state
    (committed when @p commit)."""
    U = P.state(block)
    U1 = half_flow(P, U, 1.0)
    U2 = source(P, U1, 1.0)
    if commit:
        P.commit(block, U2)
    return U2


def condensed_schur(P, block, *, alpha, theta=1.0, c_rho=0, c_mx=1, c_my=2, c_bz=3, c_E=None,
                    method="bicgstab", tol=1e-10, max_iter=400, commit=True):
    """Condensed-Schur implicit electrostatic-Lorentz SOURCE stage as a compiled Program (epic ADC-399,
    acceptance 32), mirroring the native ``pops.CondensedSchur`` (CondensedSchurSourceStepper) sequence:

      1. assemble the anisotropic tensor coefficient ``A = I + c*rho*B^{-1}`` (``P.schur_coeffs``,
         ``c = theta^2 dt^2 alpha``);
      2. assemble the fused RHS ``-Lap(phi^n) - theta*dt*alpha*div(B^{-1}(mx,my))`` (``P.schur_rhs``);
      3. solve ``-div(A grad phi^{n+theta}) = RHS`` matrix-free (``P.matrix_free_operator`` +
         ``P.apply_laplacian_coeff`` negated, ``P.solve_linear``), warm-started from phi^n;
      4. reconstruct ``v^{n+theta} = B^{-1}(v^n - theta*dt*grad phi)`` and write ``mom = rho*v``
         (``P.schur_reconstruct``, the closed B^{-1}); rho stays frozen;
      5. (``theta < 1``) extrapolate the theta-stage state to ``n+1`` by the native factor ``1/theta``:
         ``U^{n+1} = U^n + (1/theta)(U^{n+theta} - U^n)`` (the affine algebra, see THETA below);
      6. (``c_E`` given) update the total energy ``E^{n+1} = E^n + (1/2)rho(|v^{n+1}|^2 - |v^n|^2)``
         (``P.schur_energy``, the native kinetic-energy increment).

    phi^n is a fresh zero scalar field each step (NO persistent history -- see the cross-step carry note
    under DEFERRED below). The phi solve runs to tolerance and the velocity reconstruction reads only the
    solved phi^{n+theta}, so a single step matches the native single step taken from ``phi^n = 0`` (the
    System also initializes phi to zero). Every numerical kernel REUSES a native primitive (no stencil /
    B^{-1} / elimination reimplementation); the native ``pops.CondensedSchur`` stepper is untouched.

    THETA != 1 (ADC-427). The native stepper takes the implicit stage at ``n+theta`` and extrapolates phi
    and the MOMENTUM (not rho) to ``n+1`` by the factor ``1/theta``. This macro lowers that extrapolation
    with the EXISTING affine algebra, no component-restricted IR op: ``schur_reconstruct`` freezes rho
    (and energy), so ``rho^{n+theta} = rho^n`` and ``mom^{n+theta} = rho v^{n+theta}``,
    ``mom^n = rho v^n``. The plain STATE affine ``U^n + (1/theta)(U^{n+theta} - U^n)`` therefore leaves
    rho (and a yet-unwritten energy) untouched -- ``rho^{n+1} = (1-1/theta)rho^n + (1/theta)rho^n =
    rho^n`` -- and on the momentum it equals the native ``mom^{n+1} = mom^n + (1/theta)(mom^{n+theta} -
    mom^n) = rho(v^n + (1/theta)(v^{n+theta} - v^n))``. The phi extrapolation is a no-op here because
    phi^n = 0 (no carry) and the reconstruction already read phi^{n+theta}; phi^{n+1} would only matter
    as the NEXT step's warm start (the deferred persistent-phi carry).

    @p alpha is the electrostatic coupling constant; @p theta the theta-scheme implicitness in ``(0, 1]``;
    @p c_rho / @p c_mx / @p c_my the conserved-variable components, @p c_bz the aux component of B_z
    (canonical 3, filled by ``solve_fields``) and @p c_E the OPTIONAL energy component (None = no energy
    update, like a rho/mx/my isothermal block). @p method / @p tol / @p max_iter configure the Krylov phi
    solve.

    DEFERRED (documented partial, spec's "if too large" clause):
      - **cross-step phi^n carry**. The native stepper freezes phi^n (the previous stage's potential)
        and keeps it in the RHS (``-Lap(phi^n)``) and as the solve warm start. The System history ring
        is sized to the block's ncomp (a full state) and stores via ``pops::lincomb`` (matching ncomp), so
        a 1-component phi cannot be carried through it without a scalar-history runtime path (a new
        ncomp-aware ``register_history`` + scalar-typed history IR ops + the extrapolated-phi store/read
        dataflow) -- a runtime change too large for this slice. This macro therefore solves each step
        from ``phi^n = 0`` (a fresh zero scalar field). At theta != 1 the FIRST step still matches the
        native first step (both start from phi = 0); the cross-step difference is the warm-start /
        ``-Lap(phi^n)`` term the native stepper carries (a smoother convergence, the same fixed point).

    NEAR-MATCH to native, not bit-exact: the native solve is BiCGStab + GeometricMG preconditioner while
    the Program solve is matrix-free BiCGStab WITHOUT a preconditioner -- the SAME operator and RHS, a
    different Krylov path (both converge to the same phi at tolerance). ``python/tests/
    test_time_condensed_schur.py`` checks against an offline reference of the identical assemble / solve
    / reconstruct / extrapolate steps and documents the gap vs native (theta == 1 and theta == 0.5)."""
    if not (0.0 < float(theta) <= 1.0):
        raise ValueError("condensed_schur: theta must be in (0, 1] (got %r)" % (theta,))
    if c_E is not None and (isinstance(c_E, bool) or not isinstance(c_E, int) or c_E < 0):
        raise ValueError("condensed_schur: c_E must be None or a Python int >= 0 (got %r)" % (c_E,))
    U = P.state(block)
    P.solve_fields(U)  # fill the shared aux (B_z at c_bz) from the current state, like the native stage
    # phi^n = 0 (a fresh zero scalar field): the RHS Laplacian term -Lap(phi^n) vanishes and the solve
    # warm starts from zero. Cross-step phi^n carry is deferred (see the docstring).
    phi_n = P.scalar_field(block + ".schur_phi_n")
    c_coeff = (float(theta) * float(theta) * float(alpha)) * P.dt * P.dt  # c = theta^2 dt^2 alpha
    th_dt = float(theta) * P.dt  # theta dt
    g = (float(theta) * float(alpha)) * P.dt  # theta dt alpha (coefficient of the div(F) term)
    coeffs = P.schur_coeffs(state=U, c=c_coeff, th_dt=th_dt, c_rho=c_rho, c_bz=c_bz)
    rhs = P.scalar_field(block + ".schur_rhs")
    P.schur_rhs(rhs, phi_n, U, th_dt, g, c_mx=c_mx, c_my=c_my, c_bz=c_bz)
    A = P.matrix_free_operator(block + ".schur_op")

    def apply(P, out, x):  # out <- A(x) = -div((I + c rho B^{-1}) grad x) = -apply_laplacian_coeff(x)
        lap = P.scalar_field("schur_lap")
        P.apply_laplacian_coeff(lap, x, coeffs)
        return -1.0 * lap  # the condensed operator -div(A grad phi); the affine is the lowered result

    P.set_apply(A, apply)
    phi = P.solve_linear(operator=A, rhs=rhs, method=method, tol=tol, max_iter=max_iter)
    # The reconstruction overwrites the MOMENTUM in place. theta == 1 with no energy keeps the historical
    # IR byte-identical (reconstruct directly on U). For theta < 1 OR an energy update we need U^n
    # (mom^n / E^n) AFTER the reconstruction, so reconstruct on a fresh COPY of U^n and keep U^n intact.
    needs_un = float(theta) != 1.0 or c_E is not None
    target = P.linear_combine(block + ".schur_un_copy", 1.0 * U) if needs_un else U
    out = P.schur_reconstruct(state=target, phi=phi, th_dt=th_dt, c_rho=c_rho, c_mx=c_mx, c_my=c_my,
                              c_bz=c_bz)
    # 5) theta-stage -> n+1 extrapolation (ADC-427). theta < 1 lowers U^n + (1/theta)(U^{n+theta} - U^n)
    # with the affine algebra (out is the theta-stage on the copy, U^n is the untouched original). rho is
    # frozen by the reconstruction, so this affine leaves rho (and the not-yet-written energy) at U^n.
    if float(theta) != 1.0:
        inv_theta = 1.0 / float(theta)
        out = P.linear_combine(block + ".schur_extrap", U + inv_theta * (out - U))
    # 6) energy role (ADC-427). E^{n+1} = E^n + (1/2)rho(|v^{n+1}|^2 - |v^n|^2): the kinetic-energy
    # increment from v^n (= mom^n/rho, read from U^n) to v^{n+1} (= mom^{n+1}/rho, in `out`). Skipped
    # for an isothermal rho/mx/my block (c_E is None).
    if c_E is not None:
        out = P.schur_energy(state=out, state_old=U, c_rho=c_rho, c_mx=c_mx, c_my=c_my, c_E=c_E)
    if commit:
        P.commit(block, out)
    return out

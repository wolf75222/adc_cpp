"""pops.time time-stepping macros and Butcher tableaux (explicit + split schemes).

Each scheme (``forward_euler`` / ``ssprk2`` / ``ssprk3`` / ``rk4`` / ``adams_bashforth`` /
``adams_bashforth2`` / ``strang`` / ``lie`` / ``condensed_schur`` / ``imex_local`` / the
generic ``rk`` over a ``ButcherTableau``) BUILDS Program IR via the same builder ops -- no
scheme-specific C++ class. The BDF / operator-first schemes and the optimization passes live
in ``pops.time.equations_implicit``. Authoring only.
"""

# --- Standard library: time-stepping macros that LOWER to the Program IR (pops.time.std, ADC-407) ----
# These are NOT separate C++ steppers: each builds pops.time.Program IR via the same builder ops + the
# affine algebra over dt, so a scheme is expressed ONCE with no scheme-specific class (spec acceptance
# 25-29; RK4 has no special RK4 class). The generated problem.so (compile_problem, Phase 2c) executes
# the lowered IR. forward_euler / ssprk2 / ssprk3 reproduce pops.Explicit(method="euler"/"ssprk2"/"ssprk3").
def _stage_rhs(P, U, sources, flux):
    """Solve the elliptic fields from U and assemble its RHS for one stage. The FieldContext is
    distinct per stage (no stale global aux). flux=False builds a source-only sub-flow (e.g. Strang S)."""
    fields = P.solve_fields(U) if flux else None
    return P.rhs(state=U, fields=fields, flux=flux, sources=list(sources))


def forward_euler(P, block, *, sources=("default",), flux=True):
    """Forward Euler: U^{n+1} = U + dt * R(U)."""
    U = P.state(block)
    R = _stage_rhs(P, U, sources, flux)
    P.commit(block, P.linear_combine("fe_step", U + P.dt * R))


def ssprk2(P, block, *, sources=("default",), flux=True):
    """SSPRK2 (Heun / Shu-Osher): U1 = U0 + dt k0; U^{n+1} = 1/2 U0 + 1/2 (U1 + dt k1)."""
    U0 = P.state(block)
    k0 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("ssprk2_U1", U0 + P.dt * k0)
    k1 = _stage_rhs(P, U1, sources, flux)
    P.commit(block, P.linear_combine("ssprk2_step", 0.5 * U0 + 0.5 * (U1 + P.dt * k1)))


def ssprk3(P, block, *, sources=("default",), flux=True):
    """SSPRK3 (Shu-Osher): U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1);
    U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    U0 = P.state(block)
    k0 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("ssprk3_U1", U0 + P.dt * k0)
    k1 = _stage_rhs(P, U1, sources, flux)
    U2 = P.linear_combine("ssprk3_U2", 0.75 * U0 + 0.25 * (U1 + P.dt * k1))
    k2 = _stage_rhs(P, U2, sources, flux)
    P.commit(block, P.linear_combine("ssprk3_step", (1.0 / 3.0) * U0 + (2.0 / 3.0) * (U2 + P.dt * k2)))


def rk4(P, block, *, sources=("default",), flux=True):
    """Classic RK4, expressed with NO special RK4 class (spec acceptance 29):
    U^{n+1} = U0 + dt/6 (k1 + 2 k2 + 2 k3 + k4)."""
    U0 = P.state(block)
    k1 = _stage_rhs(P, U0, sources, flux)
    U1 = P.linear_combine("rk4_U1", U0 + 0.5 * P.dt * k1)
    k2 = _stage_rhs(P, U1, sources, flux)
    U2 = P.linear_combine("rk4_U2", U0 + 0.5 * P.dt * k2)
    k3 = _stage_rhs(P, U2, sources, flux)
    U3 = P.linear_combine("rk4_U3", U0 + P.dt * k3)
    k4 = _stage_rhs(P, U3, sources, flux)
    P.commit(block, P.linear_combine(
        "rk4_step", U0 + P.dt / 6.0 * k1 + P.dt / 3.0 * k2 + P.dt / 3.0 * k3 + P.dt / 6.0 * k4))


# Adams-Bashforth weights b_j on R_{n-j} (j = 0..order-1), per order (ADC-423). AB1 is Forward Euler.
_AB_WEIGHTS = {
    1: (1.0,),
    2: (1.5, -0.5),                       # 3/2, -1/2
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
}


def adams_bashforth(P, block, order, *, sources=("default",), flux=True):
    """Adams-Bashforth, explicit ``order``-step, over the System-owned history ring (ADC-406a / ADC-423):

        R_n     = R(U)
        U^{n+1} = U + dt * sum_{j=0}^{order-1} b_j * R_{n-j}
        store_history(block.R, R_n)

    ``order`` selects the classic AB weights b_j:
      - **AB1** == Forward Euler (b = 1), with NO history (it never reads or stores the ring);
      - **AB2** == (3/2, -1/2) on (R_n, R_{n-1});
      - **AB3** == (23/12, -16/12, 5/12) on (R_n, R_{n-1}, R_{n-2}).

    COLD START: the store of R_n is recorded BEFORE the lag reads, and the runtime fills EVERY history
    slot on the FIRST store, so step 0 reads R_{n-j} = R_0 for all j and the recurrence degenerates to a
    single Forward-Euler step (U^1 = U^0 + dt*R_0, since sum_j b_j = 1). From step ``order-1`` on it is
    the true AB recurrence; in between it runs the same partially-filled ring the runtime exposes. This
    is deterministic and exact; an offline reference mirrors it (FE-fill cold start then AB). The history
    name is ``"<block>.R"`` (the block's previous RHS).

    AB1 keeps Forward Euler's exact IR (no history op); AB2 keeps the historical ``"ab2_step"`` combine
    so a pre-ADC-423 AB2 program's ``.so`` cache key is byte-identical."""
    if isinstance(order, bool) or not isinstance(order, int) or order not in _AB_WEIGHTS:
        raise ValueError("adams_bashforth: order must be an int in %s (got %r)"
                         % (sorted(_AB_WEIGHTS), order))
    b = _AB_WEIGHTS[order]
    if order == 1:  # AB1 == Forward Euler: no history, identical IR to forward_euler.
        forward_euler(P, block, sources=sources, flux=flux)
        return
    name = block + ".R"
    step_name = "ab2_step" if order == 2 else ("ab%d_step" % order)
    U = P.state(block)
    R_n = _stage_rhs(P, U, sources, flux)
    # Store R_n FIRST (so the first store cold-start-fills the ring), then read R_{n-j} = lag j.
    P.store_history(name, R_n)
    expr = U + (P.dt * b[0]) * R_n
    for j in range(1, order):
        expr = expr + (P.dt * b[j]) * P.history(name, lag=j)
    P.commit(block, P.linear_combine(step_name, expr))


def adams_bashforth2(P, block, *, sources=("default",), flux=True):
    """Adams-Bashforth 2, a thin back-compat alias for ``adams_bashforth(P, block, 2)`` (ADC-423).

    Kept so existing callers and the historical ``"ab2_step"`` IR are unchanged: this lowers to the
    SAME IR as before (R_n stored first, R_{n-1} read at lag 1, weights 3/2 / -1/2)."""
    adams_bashforth(P, block, 2, sources=sources, flux=flux)


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


# Classic explicit Butcher tableaux (A lower-triangular, b weights, c nodes) for `rk` (ADC-423).
class ButcherTableau:
    """An explicit Butcher tableau ``(A, b, c)`` for `rk`: ``A`` is strictly lower-triangular (stage i
    depends only on stages j < i), ``b`` the final weights, ``c`` the (unused-by-the-lowering) nodes.
    Validated as explicit and consistent (``len(A) == len(b)``, row i has i entries, ``sum(b) == 1``)."""

    def __init__(self, A, b, c=None, name=None):
        self.A = [list(row) for row in A]
        self.b = list(b)
        self.c = list(c) if c is not None else [sum(row) for row in self.A]
        self.name = name
        s = len(self.b)
        if len(self.A) != s or len(self.c) != s:
            raise ValueError("ButcherTableau: A, b, c must share the stage count")
        for i, row in enumerate(self.A):
            if len(row) > i and any(row[j] != 0.0 for j in range(i, len(row))):
                raise ValueError(
                    "ButcherTableau: A must be strictly lower-triangular (stage %d reads stage >= %d); "
                    "rk lowers EXPLICIT tableaux only" % (i, i))
        if abs(sum(self.b) - 1.0) > 1e-12:
            raise ValueError("ButcherTableau: weights b must sum to 1 (got %r)" % (sum(self.b),))

    @property
    def stages(self):
        return len(self.b)


# RK4 (classic): the same tableau the rk4 macro hard-codes, written data-driven.
RK4_TABLEAU = ButcherTableau(
    A=[[],
       [0.5],
       [0.0, 0.5],
       [0.0, 0.0, 1.0]],
    b=[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    c=[0.0, 0.5, 0.5, 1.0],
    name="rk4")

# SSPRK2 (Heun) in NON-Shu-Osher Butcher form: k1 at U, k2 at U+dt*k1, U^{n+1}=U+dt(1/2 k1+1/2 k2).
SSPRK2_TABLEAU = ButcherTableau(
    A=[[],
       [1.0]],
    b=[0.5, 0.5],
    c=[0.0, 1.0],
    name="ssprk2")


def rk(P, block, tableau, *, sources=("default",), flux=True):
    """Generic explicit Runge-Kutta from a Butcher @p tableau (ADC-423), lowered to the SAME stage chain
    the hard-coded `rk4` macro emits -- ``solve_fields`` + ``rhs`` + ``linear_combine``, no RK class:

        k_i      = R( U + dt * sum_{j<i} A[i][j] * k_j )       (the i-th stage RHS)
        U^{n+1}  = U + dt * sum_i b[i] * k_i

    @p tableau is a `ButcherTableau` (or a raw ``(A, b, c)`` triple); ``A`` must be strictly
    lower-triangular (explicit). ``RK4_TABLEAU`` and ``SSPRK2_TABLEAU`` are provided as the classic
    constants: ``rk(P, blk, RK4_TABLEAU)`` builds the identical final affine combination as
    ``rk4(P, blk)`` (a permutation of the same ``U0 + dt(1/6 k1 + 1/3 k2 + 1/3 k3 + 1/6 k4)`` inputs),
    and ``rk(P, blk, SSPRK2_TABLEAU)`` matches Heun's ``U + dt(1/2 k1 + 1/2 k2)``."""
    if not isinstance(tableau, ButcherTableau):
        A, b, c = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(A, b, c)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    U0 = P.state(block)
    ks = []
    for i in range(tableau.stages):
        if i == 0:
            Ui = U0  # the first stage reads U^n directly (no scratch combine, like rk4)
        else:
            expr = U0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0.0:
                    expr = expr + (P.dt * aij) * ks[j]
            Ui = P.linear_combine("%sU%d" % (tag, i), expr)
        ks.append(_stage_rhs(P, Ui, sources, flux))
    final = U0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0.0:
            final = final + (P.dt * bi) * ks[i]
    P.commit(block, P.linear_combine("%sstep" % tag, final))


def imex_local(P, block, *, linear_source, sources=("default",), flux=True, theta=1.0):
    """IMEX with an EXPLICIT flux/source and an IMPLICIT cell-local linear source (ADC-423).

    One step of a theta-implicit splitting of ``dU/dt = R_explicit(U) + L U`` where ``L`` is a named
    model ``m.linear_source`` (e.g. a Lorentz operator) solved cell by cell:

        R   = R_explicit(U)                                     (P.rhs: -div F + the named sources)
        U^{n+1} = (I - theta*dt*L)^{-1} (U + dt*R)              (P.solve_local_linear)

    The explicit part is assembled with `P.rhs` (flux + the requested named @p sources, on the fields
    solved from U); the implicit part is the local solve of ``(I - theta*dt*L) U^{n+1} = U + dt*R``
    via `P.solve_local_linear`, exactly the predictor half of the codebase's predictor-corrector
    pattern (``test_time_local_solve``). At ``theta == 1`` this is backward Euler on the L term and
    forward Euler on R; ``theta == 0`` would drop the implicit solve (use `forward_euler` instead) and
    is rejected. @p linear_source is the name of the model ``m.linear_source``; @p theta the
    implicitness of the L term (0 < theta <= 1)."""
    if not (isinstance(linear_source, str) and linear_source):
        raise ValueError("imex_local: linear_source must be a non-empty m.linear_source name")
    if not (0.0 < float(theta) <= 1.0):
        raise ValueError(
            "imex_local: theta must be in (0, 1] (got %r); theta == 0 is fully explicit -- use "
            "forward_euler instead" % (theta,))
    U = P.state(block)
    fields = P.solve_fields(U) if flux else None
    R = P.rhs(state=U, fields=fields, flux=flux, sources=list(sources))
    rhs = P.linear_combine(block + "_imex_rhs", U + P.dt * R)
    operator = P.I - (float(theta) * P.dt) * P.linear_source(linear_source)
    out = P.solve_local_linear(name=block + "_imex_step", operator=operator, rhs=rhs, fields=fields)
    P.commit(block, out)
    return out

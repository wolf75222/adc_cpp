"""pops.time BDF / operator-first schemes and IR optimization passes.

BDF (``bdf`` + the local-linear / implicit-flux lowerings), the operator-first
``predictor_corrector_local_linear`` / ``explicit_rk`` / ``imex_local_linear`` builders, plus
the free-function optimization passes (``eliminate_*`` / ``optimize``) that delegate to the
Program methods. Authoring only; builds Program IR via the shared builder ops.
"""
from pops.time.equations import ButcherTableau, rk  # noqa: F401

def _bdf_local_linear(P, block, order, linear_source, sources, flux):
    """The cell-LOCAL linear-source BDF fast path (the historical lowering): the BDF system is
    block-diagonal, so ``(c0*I - dt*L) U^{n+1} = rhs`` is solved per cell by `P.solve_local_linear`.

      - **BDF1** (backward Euler): ``(I - dt*L) U^{n+1} = U^n [+ dt R]``;
      - **BDF2**: ``(I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1}) [+ dt R]`` over the System
        history ring, with a BDF1 cold start (the first store fills every slot -> U^{n-1} = U^n)."""
    U = P.state(block)
    fields = P.solve_fields(U) if flux else None
    # Optional EXPLICIT flux/source RHS folded into the BDF right-hand side (lagged at U^n).
    R = P.rhs(state=U, fields=fields, flux=flux, sources=list(sources)) if (flux or sources) else None

    def _with_explicit(expr):
        return (expr + P.dt * R) if R is not None else expr

    if order == 1:  # (I - dt*L) U^{n+1} = U^n [+ dt R]
        rhs = P.linear_combine(block + "_bdf1_rhs", _with_explicit(1.0 * U))
        operator = P.I - P.dt * P.linear_source(linear_source)
        out = P.solve_local_linear(name=block + "_bdf1_step", operator=operator, rhs=rhs, fields=fields)
        P.commit(block, out)
        return out
    # BDF2: (3/2 I - dt*L) U^{n+1} = 2 U^n - 1/2 U^{n-1} [+ dt R], over the history ring.
    name = block + ".U"
    P.store_history(name, U)                       # store U^n first (cold-start fills the ring)
    U_nm1 = P.history(name, lag=1)                 # U^{n-1} (== U^n on step 0 -> BDF1 cold start)
    rhs = P.linear_combine(block + "_bdf2_rhs", _with_explicit(2.0 * U - 0.5 * U_nm1))
    operator = P.I - (P.dt * (2.0 / 3.0)) * P.linear_source(linear_source)
    # Divide both sides by 3/2: (I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1} [+ dt R]).
    rhs = P.linear_combine(block + "_bdf2_rhs_scaled", (2.0 / 3.0) * rhs)
    out = P.solve_local_linear(name=block + "_bdf2_step", operator=operator, rhs=rhs, fields=fields)
    P.commit(block, out)
    return out


def _bdf_implicit_flux(P, block, order, sources, flux, ncomp, newton_tol, newton_max, krylov_tol,
                       krylov_max, krylov_restart, eps):
    """The IMPLICIT-FLUX BDF lowering (ADC-431): a matrix-free Newton-Krylov solve of the coupled
    nonlinear system, composed PURELY from existing IR primitives (no new C++ stepper).

    The implicit BDF step solves ``F(U^{n+1}) = 0`` with::

        BDF1:  F(U) = U - U^n            - dt*rhs(U)
        BDF2:  F(U) = U - (4/3)U^n + (1/3)U^{n-1} - (2/3)*dt*rhs(U)

    (BDF2 reads ``U^{n-1}`` from the System history ring with a BDF1 cold start.) ``rhs(U) = -div F(U)
    [+ sources]`` is the SAME hyperbolic residual the explicit schemes use, so the flux couples the
    cells through its stencil and the Newton system is GLOBAL.

    Newton's method (the outer loop) is a fixed `static_range` unroll of @p newton_max iterations -- each
    iteration is independent IR (its own matrix-free operator + Krylov solve), which the codegen lowers
    at the top level (the install-time apply lambda the Krylov loop needs cannot live inside a runtime
    while/range body). Each iteration:

      1. ``R^k = rhs(U^k)`` (one rhs evaluation; also the frozen base of the matvec FD);
      2. ``F^k = U^k - U^n_terms - c*dt*R^k`` (the residual; ``c = 1`` BDF1, ``c = 2/3`` BDF2);
      3. solve ``J dU = -F^k`` with GMRES (J nonsymmetric), J applied matrix-free via `rhs_jacvec`
         (``J v = v - c*dt * d(rhs)/dU v``, a finite-difference Jacobian-vector product around U^k);
      4. ``U^{k+1} = U^k + dU``.

    The final residual norm ``||F||`` is recorded as the diagnostic ``"<block>.bdf_residual"`` (read via
    ``sim.program_diagnostic``). @p ncomp is the block component count (1 by default -- a scalar model
    like inviscid Burgers / linear advection; pass the model's n_cons for a multi-component block)."""
    c = 1.0 if order == 1 else (2.0 / 3.0)
    U0 = P.state(block)
    fields = P.solve_fields(U0) if flux else None  # frozen-Poisson coupling, solved once from U^n
    # Snapshot U^n into a scratch: the commit writes ctx.state(0) IN PLACE at the very end, so the lagged
    # term must read this frozen copy (not the live state) -- otherwise the post-commit residual
    # diagnostic would read U^{n+1} as U^n. The Newton-loop residuals (before the commit) would be correct
    # either way; the snapshot keeps every residual (loop + diagnostic) reading the true U^n.
    Un = P.linear_combine(block + "_bdf_Un", 1.0 * U0)
    if order == 2:
        name = block + ".U"
        P.store_history(name, U0)                   # store U^n (cold-start fills the ring)
        U_nm1 = P.history(name, lag=1)              # U^{n-1} (== U^n on step 0 -> BDF1 cold start)

    def _un_terms():
        # The lagged (constant-in-Newton) part of the residual: U^n for BDF1, (4/3)U^n - (1/3)U^{n-1}
        # for BDF2 (the constant-state coefficients of the BDF residual normalized to a unit U^{n+1}).
        if order == 1:
            return 1.0 * Un
        return (4.0 / 3.0) * Un - (1.0 / 3.0) * U_nm1

    src = list(sources) if sources is not None else None
    kind = "scalar" if ncomp == 1 else "state"

    def _residual(P, Uk, tag):
        # F^k = U^k - U^n_terms - c*dt*rhs(U^k); returns (F^k, R^k) so the matvec can reuse R^k.
        Rk = P.rhs(name="%s_R" % tag, state=Uk, fields=fields, flux=flux, sources=src)
        Fk = P.linear_combine("%s_F" % tag, _un_terms() * (-1.0) + 1.0 * Uk - (c * P.dt) * Rk)
        return Fk, Rk

    def _newton_step(P, Uk, k):
        tag = "%s_bdf%d_n%d" % (block, order, k)
        Fk, Rk = _residual(P, Uk, tag)
        negF = P.linear_combine("%s_negF" % tag, -1.0 * Fk)
        A = P.matrix_free_operator("%s_J" % tag, domain=kind, range_=kind,
                                   ncomp=(None if ncomp == 1 else ncomp))

        def apply(P, out, v):
            # J v = v - c*dt * d(rhs)/dU v, matrix-free FD around the frozen iterate U^k (r0 = R^k).
            return P.rhs_jacvec(out, v, iterate=Uk, r0=Rk, c_dt=(c * P.dt), eps=eps, flux=flux,
                                sources=sources)

        P.set_apply(A, apply)
        dU = P.solve_linear(name="%s_dU" % tag, operator=A, rhs=negF, method="gmres", tol=krylov_tol,
                            max_iter=krylov_max, restart=krylov_restart)
        return P.linear_combine("%s_next" % tag, 1.0 * Uk + 1.0 * dU)

    # Outer Newton loop: a fixed unroll of newton_max iterations (each independent top-level IR).
    Uk = U0
    for k in range(newton_max):
        Uk = _newton_step(P, Uk, k)
    # Record the final residual norm for diagnostics (sim.program_diagnostic("<block>.bdf_residual")).
    Ffinal, _ = _residual(P, Uk, "%s_bdf%d_final" % (block, order))
    P.record_scalar(block + ".bdf_residual", P.norm2(Ffinal))
    P.commit(block, Uk)
    return Uk


def bdf(P, block, order, *, linear_source=None, sources=("default",), flux=True, ncomp=1,
        newton_tol=1e-10, newton_max=20, krylov_tol=1e-10, krylov_max=200, krylov_restart=None,
        eps=1e-7):
    """Backward Differentiation Formula, IMPLICIT ``order``-step (ADC-423 / ADC-431).

    Two lowerings share this entry point, selected by whether an implicit @p linear_source is named:

      - **implicit FLUX** (the default, ADC-431): ``F(U^{n+1}) = 0`` for the coupled nonlinear system
        ``U - U^n - dt*rhs(U)`` (BDF1) / ``U - (4/3)U^n + (1/3)U^{n-1} - (2/3)dt*rhs(U)`` (BDF2) is
        solved by a matrix-free Newton-Krylov iteration -- ``rhs(U) = -div F [+ sources]`` couples the
        cells through the flux stencil, so the Jacobian ``J = I - c*dt*d(rhs)/dU`` is GLOBAL and applied
        matrix-free by a finite-difference Jacobian-vector product (`P.rhs_jacvec`); each Newton step
        solves ``J dU = -F`` with GMRES (J nonsymmetric). The outer Newton loop is a fixed unroll of
        @p newton_max iterations. The final ``||F||`` is recorded as ``"<block>.bdf_residual"``. This is
        a pure-macro composition of existing primitives (matrix_free_operator + solve_linear + the affine
        algebra + history) -- no new C++ runtime stepper.

      - **cell-local linear SOURCE** (the fast path, ADC-423): when @p linear_source names a model
        ``m.linear_source`` ``L``, the BDF system is block-diagonal and ``(c0*I - dt*L) U^{n+1} = rhs``
        is solved per cell by `P.solve_local_linear` (no Newton / Krylov). @p flux / @p sources then add
        an EXPLICIT flux/source RHS lagged at U^n (like `imex_local`).

    @p order is 1 (backward Euler) or 2 (BDF2, over the System history ring with a BDF1 cold start).
    @p ncomp is the block component count for the implicit-flux path (1 for a scalar model such as
    inviscid Burgers / linear advection; pass the model's n_cons for a multi-component block).
    @p newton_max / @p newton_tol bound the Newton iteration; @p krylov_tol / @p krylov_max /
    @p krylov_restart configure each GMRES inner solve; @p eps is the relative finite-difference step of
    the Jacobian-vector product."""
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("bdf: order must be the int 1 or 2 (got %r)" % (order,))
    if linear_source is not None:
        if not (isinstance(linear_source, str) and linear_source):
            raise ValueError("bdf: linear_source must be a non-empty model linear-source name or None")
        return _bdf_local_linear(P, block, order, linear_source, sources, flux)
    # The implicit-flux Newton-Krylov path (ADC-431): a flux-less BDF with no implicit term is a no-op.
    if not flux:
        raise ValueError(
            "bdf with flux=False needs a cell-local implicit linear_source (there is no implicit term to "
            "solve); pass linear_source='<name>' for the relaxation BDF, or flux=True for the "
            "implicit-flux Newton-Krylov BDF")
    if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
        raise ValueError("bdf: ncomp must be a positive int (the block component count); got %r"
                         % (ncomp,))
    if isinstance(newton_max, bool) or not isinstance(newton_max, int) or newton_max < 1:
        raise ValueError("bdf: newton_max must be a positive int (got %r)" % (newton_max,))
    return _bdf_implicit_flux(P, block, order, sources, flux, ncomp, newton_tol, newton_max,
                              krylov_tol, krylov_max, krylov_restart, eps)


# pops.time.std.<scheme>(Program, block, ...) -- the spec's standard library entry point.
# --- operator-first standard macros (Spec 2) --------------------------------------------------
# These macros are MODEL-FREE: they take typed operator NAMES (not physical terms) and compose them
# with P.call against the registry bound to the Program (P.bind_operators(module)). The SAME macro
# runs against any Module that provides operators with the expected signatures. They never mention
# flux / source / poisson / lorentz / rho / mx / my.
def _op_space_arity(P, name):
    """Number of space-typed inputs (State / FieldSpace) of operator @p name in the bound registry."""
    if P._registry is None:
        raise ValueError("operator-first macro: bind a module first (P.bind_operators(module))")
    op = P._registry.get(name)
    return sum(1 for t in op.signature.inputs if getattr(t, "kind", None) in ("state", "field"))


def _opcall(P, name, *candidate_args, value_name=None):
    """Call operator @p name passing exactly as many leading args as its signature's space inputs
    (so an operator that ignores the fields is called with the state alone, and a fields-free linear
    operator with no args)."""
    arity = _op_space_arity(P, name)
    return P.call(name, *candidate_args[:arity], name=value_name)


def predictor_corrector_local_linear(P, block, *, fields_operator, explicit_rate_operator,
                                     implicit_operator, state_space="U", commit=True):
    """Generic predictor-corrector for ``dU/dt = R(U, fields) + L(fields) U`` (Spec 2, operator-first).

    Composes THREE typed operators by name -- a field operator ``fields_operator: U -> Fields``, an
    explicit rate ``explicit_rate_operator: (U, Fields) -> Rate(U)`` and a local linear operator
    ``implicit_operator: Fields -> LocalLinearOperator(U, U)`` -- into one trapezoidal step with the
    L term treated implicitly via local solves::

        U*    = (I - dt L_n)^{-1} (U^n + dt R_n)
        U^n+1 = (I - 1/2 dt L*)^{-1} (U^n + 1/2 dt R_n + 1/2 dt R* + 1/2 dt L* U*)

    It mentions no physics; ``state_space`` is informational. Requires ``P.bind_operators(module)``.
    """
    u_n = P.state(block)
    fields_n = _opcall(P, fields_operator, u_n, value_name="fields_n")
    r_n = _opcall(P, explicit_rate_operator, u_n, fields_n, value_name="R_n")
    l_n = _opcall(P, implicit_operator, fields_n, value_name="L_n")
    u_star = P.solve_local_linear("U_star", operator=P.I - P.dt * l_n,
                                  rhs=P.linear_combine("U_star_rhs", u_n + P.dt * r_n),
                                  fields=fields_n)
    fields_star = _opcall(P, fields_operator, u_star, value_name="fields_star")
    r_star = _opcall(P, explicit_rate_operator, u_star, fields_star, value_name="R_star")
    l_star = _opcall(P, implicit_operator, fields_star, value_name="L_star")
    c_star = P.apply(l_star, u_star, fields=fields_star, name="C_star")
    q = P.linear_combine("Q", u_n + 0.5 * P.dt * r_n + 0.5 * P.dt * r_star + 0.5 * P.dt * c_star)
    u_np1 = P.solve_local_linear("U_np1", operator=P.I - 0.5 * P.dt * l_star, rhs=q,
                                 fields=fields_star)
    if commit:
        P.commit(block, u_np1)
    return u_np1


def explicit_rk(P, block, *, rhs_operator, fields_operator=None, tableau=None, A=None, b=None,
                c=None, state_space="U"):
    """Generic explicit Runge-Kutta over a typed rate operator (Spec 2, operator-first).

    Each stage is ``k_i = rhs_operator(U_i[, fields_operator(U_i)])``; the tableau lowers to the same
    affine stage chain as :func:`rk`. Pass a ``ButcherTableau`` / ``(A, b, c)`` via ``tableau`` or the
    raw ``A`` / ``b`` / ``c``. ``fields_operator`` is optional (a pure-flux rate needs no fields).
    """
    if tableau is None:
        if A is None or b is None:
            raise ValueError("explicit_rk: provide a tableau or A and b")
        tableau = ButcherTableau(A, b, c)
    elif not isinstance(tableau, ButcherTableau):
        ta, tb, tc = tableau if len(tableau) == 3 else (tableau[0], tableau[1], None)
        tableau = ButcherTableau(ta, tb, tc)
    tag = (tableau.name + "_") if tableau.name else "rk_"
    u0 = P.state(block)
    ks = []
    for i in range(tableau.stages):
        if i == 0:
            u_i = u0
        else:
            expr = u0
            for j in range(i):
                aij = tableau.A[i][j]
                if aij != 0.0:
                    expr = expr + (P.dt * aij) * ks[j]
            u_i = P.linear_combine("%sU%d" % (tag, i), expr)
        if fields_operator is not None:
            f_i = _opcall(P, fields_operator, u_i)
            ks.append(_opcall(P, rhs_operator, u_i, f_i, value_name="%sk%d" % (tag, i)))
        else:
            ks.append(_opcall(P, rhs_operator, u_i, value_name="%sk%d" % (tag, i)))
    final = u0
    for i in range(tableau.stages):
        bi = tableau.b[i]
        if bi != 0.0:
            final = final + (P.dt * bi) * ks[i]
    P.commit(block, P.linear_combine("%sstep" % tag, final))


def imex_local_linear(P, block, *, explicit_operator, implicit_operator, fields_operator=None,
                      theta=1.0, state_space="U"):
    """Generic IMEX with an explicit rate and an implicit local linear operator (Spec 2).

    One theta-implicit step of ``dU/dt = R(U[, fields]) + L([fields]) U``::

        U^{n+1} = (I - theta dt L)^{-1} (U^n + dt R)

    composing the typed ``explicit_operator`` and ``implicit_operator`` (and an optional
    ``fields_operator``) by name. Requires ``P.bind_operators(module)``.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("imex_local_linear: theta must be in (0, 1]")
    u = P.state(block)
    fields = _opcall(P, fields_operator, u, value_name="fields") if fields_operator else None
    r = _opcall(P, explicit_operator, u, fields, value_name="R")
    lin = _opcall(P, implicit_operator, fields, value_name="L")
    q = P.linear_combine("imex_rhs", u + P.dt * r)
    u1 = P.solve_local_linear("imex_step", operator=P.I - theta * P.dt * lin, rhs=q, fields=fields)
    P.commit(block, u1)
    return u1


def eliminate_dead_nodes(program):
    """Return a NEW Program with dead flat-list nodes removed (free-function form of
    :meth:`Program.eliminate_dead_nodes`, Spec 3 s28 / ADC-465). OPT-IN: it optimizes a copy and never
    touches the default ``emit_cpp_program`` path. See the method for the dead-node rule."""
    return program.eliminate_dead_nodes()


def eliminate_common_subexpressions(program):
    """Return a NEW Program with duplicated PURE sub-IR computed once and aliased (free-function form
    of :meth:`Program.eliminate_common_subexpressions`, Spec 3 s28 / ADC-465). OPT-IN, proven-safe."""
    return program.eliminate_common_subexpressions()


def eliminate_redundant_field_solves(program):
    """Return a NEW Program with a provably-redundant second ``solve_fields`` removed (free-function
    form of :meth:`Program.eliminate_redundant_field_solves`, Spec 3 s28 / ADC-465). OPT-IN,
    conservative: only when no state/aux mutation intervenes between the two solves."""
    return program.eliminate_redundant_field_solves()


def optimize(program):
    """Return a NEW Program with the proven-safe Spec 3 s28 transform passes applied (free-function
    form of :meth:`Program.optimize`, ADC-465). OPT-IN: byte-identical when nothing is optimizable."""
    return program.optimize()


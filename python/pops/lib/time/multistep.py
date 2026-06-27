"""pops.lib.time.multistep -- Adams-Bashforth and BDF (Backward Differentiation Formula) schemes.

Exports: adams_bashforth, adams_bashforth2, bdf.
Private helpers: _AB_WEIGHTS, _bdf_local_linear, _bdf_implicit_flux.
"""

from ._helpers import _stage_rhs
from .euler import forward_euler as _forward_euler_macro


def _forward_euler(P, block, sources, flux):
    # AB1 degenerates to Forward Euler; reuse the local euler macro for byte-identical IR.
    _forward_euler_macro(P, block, sources=sources, flux=flux)


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
        _forward_euler(P, block, sources, flux)
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

"""pops.lib.time.imex -- IMEX (implicit-explicit) time-stepping schemes.

Exports: imex_local, imex_local_linear.
"""

from ._helpers import _opcall


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

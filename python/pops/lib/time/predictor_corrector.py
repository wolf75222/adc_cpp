"""pops.lib.time.predictor_corrector -- Predictor-corrector scheme (operator-first, Spec 2).

Exports: predictor_corrector_local_linear.
"""

from ._helpers import _opcall


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

"""Spec 3 board-like time program, and proof it lowers to the primitive IR.

Builds a Heun/Crank-Nicolson-style predictor-corrector with the blackboard sugar
(T.fields / T.define / T.solve / T.commit) and with the equivalent primitive calls
(solve_fields / linear_combine / solve_local_linear / commit), then checks the two
Program IRs are identical -- the board notation is sugar, not a new IR.

Run: python3 examples/spec3/board_time_predictor_corrector.py
"""
from pops.time import Program
from pops.math import unknown


def _ir(P):
    idx = {id(v): k for k, v in enumerate(P._values)}
    return [(v.vtype, v.op, tuple(idx[id(i)] for i in v.inputs),
             repr(sorted(v.attrs.items())), v.block) for v in P._values]


def board():
    T = Program("pc_board")
    dt = T.dt
    u_n = T.state("plasma")
    f_n = T.fields("fields_n", from_state=u_n)
    r_n = T.rhs(name="R_n", state=u_n, fields=f_n, flux=True, sources=["electric"])
    u_star = T.solve(
        "U_star",
        (T.I - dt * T.linear_source("lorentz")) @ unknown("U_star") == u_n + dt * r_n,
    )
    T.commit("plasma", u_star)
    return T


def primitive():
    P = Program("pc_primitive")
    dt = P.dt
    u_n = P.state("plasma")
    f_n = P.solve_fields("fields_n", u_n)
    r_n = P.rhs(name="R_n", state=u_n, fields=f_n, flux=True, sources=["electric"])
    op = P.I - dt * P.linear_source("lorentz")
    rhs = P.linear_combine("U_star_rhs", u_n + dt * r_n)
    u_star = P.solve_local_linear(name="U_star", operator=op, rhs=rhs)
    P.commit("plasma", u_star)
    return P


if __name__ == "__main__":
    same = _ir(board()) == _ir(primitive())
    print("board IR == primitive IR:", same)
    assert same, "board sugar must lower to the same IR as the primitive calls"
    print("OK: blackboard notation is sugar over the operator-first IR")

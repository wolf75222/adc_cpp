"""Spec 3: the board facade and the operator-first kernel share ONE IR.

Builds the same forward-Euler step two ways -- the board sugar (T.fields / T.define /
T.commit) and the explicit operator-first builder (P.solve_fields / P.call /
P.linear_combine / P.commit) -- and asserts the two Program IRs are identical.
This is the anti-duplication guarantee: the facade only generates the Spec 2 IR.

Run: python3 examples/spec3/board_vs_operator_ir_equivalence.py
"""
from adc.time import Program


def _ir(P):
    idx = {id(v): k for k, v in enumerate(P._values)}
    return [(v.vtype, v.op, tuple(idx[id(i)] for i in v.inputs),
             repr(sorted(v.attrs.items())), v.block) for v in P._values]


def board():
    T = Program("fe_board")
    dt = T.dt
    u = T.state("plasma")
    f = T.fields("f", from_state=u)
    r = T.rhs(name="R", state=u, fields=f, flux=True, sources=["electric"])
    u1 = T.define("U1", u + dt * r)
    T.commit("plasma", u1)
    return T


def operator_first():
    P = Program("fe_operator_first")
    dt = P.dt
    u = P.state("plasma")
    f = P.solve_fields("f", u)
    r = P.rhs(name="R", state=u, fields=f, flux=True, sources=["electric"])
    u1 = P.linear_combine("U1", u + dt * r)
    P.commit("plasma", u1)
    return P


if __name__ == "__main__":
    T = board()
    same = _ir(T) == _ir(operator_first())
    print("board IR == operator-first IR:", same)
    print()
    print(T.dump_operator_ir())
    print()
    print(T.dump_cpp_plan())
    assert same, "the board facade must generate the same IR as the operator-first kernel"

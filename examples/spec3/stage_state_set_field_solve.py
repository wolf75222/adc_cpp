"""Spec 3: resolve a field solve from a coherent set of stage states (StageStateSet).

When several blocks are at DIFFERENT stages (e.g. electrons and ions already at U*,
neutrals still at U^n), a field solve must read an unambiguous version of each block.
``T.state_set`` packages that choice; ``T.fields(from_state_set=...)`` solves the
coupled fields from exactly those stage states (lowering to the multi-block
``solve_fields_from_blocks`` operator-first op). The alternative, ``from_states=[...]``,
is equivalent and lighter for a few blocks.

Run: python3 examples/spec3/stage_state_set_field_solve.py
"""
from pops.time import Program


def main():
    P = Program("multi_species_stage")
    dt = P.dt

    e_n = P.state("electrons")
    i_n = P.state("ions")
    n_n = P.state("neutrals")

    # electrons and ions advanced to a predictor stage; neutrals held at n.
    e_star = P.linear_combine("e_star", e_n + dt * P.rhs(name="Re", state=e_n, flux=True))
    i_star = P.linear_combine("i_star", i_n + dt * P.rhs(name="Ri", state=i_n, flux=True))

    star = P.state_set("star", {"electrons": e_star, "ions": i_star, "neutrals": n_n})
    assert len(star) == 3
    assert [s.block for s in star.states()] == ["electrons", "ions", "neutrals"]

    fields_star = P.fields("fields_star", from_state_set=star,
                           operator="fields_from_species")
    # the coherent solve lowers to the multi-block operator-first op:
    assert fields_star.vtype == "fields"
    assert fields_star.op == "solve_fields_from_blocks"
    assert len(fields_star.inputs) == 3  # exactly the three chosen stage states

    print("StageStateSet 'star' ->", [b for b, _ in star.items()])
    print("fields_star op       :", fields_star.op, "(inputs:", len(fields_star.inputs), ")")
    print("\nOK: the field solve reads an unambiguous stage of each block.")


if __name__ == "__main__":
    main()

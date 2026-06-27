#!/usr/bin/env python3
"""Spec 3 generic multi-species board facade (ADC-457, sections 12 + 16, criteria 25-30).

``pops.physics.Model.species`` now authors N >= 2 species; the board facade LOWERS them to the
existing operator-first multi-block IR (PR #287/#299/#300) rather than a parallel runtime:

* each ``m.species(...)`` -> one ``pops.model.StateSpace`` (a named block instance);
* ``m.coupled_rate(...)`` -> the existing ``coupled_rate`` operator kind: a ``RateBundle``
  signature over the input species, the SAME operator the hand-written operator-first model
  registers;
* ``m.solve_fields_from_species(...)`` -> a multi-input ``field_operator`` (the operator-first
  surface of ``P.solve_fields_from_blocks``).

The equivalence pinned here is structural AND at emit: a 3-fluid board model lowers to 3
StateSpaces + a coupled_rate + a multi-block field operator, and a two-fluid board model emits the
SAME C++ as the hand-written ``pops.model.Module`` reference (test 34.10). Arbitrary arity (3 + 4
inputs) works; a wrong-species rate in an affine combine errors (test 34.11). This is the pure
authoring / IR-equivalence slice; the compiled multi-block ``.so`` collision STEP runs on ROMEO
(Kokkos-only AOT), NOT here. Real engine only; skips cleanly if pops.time is unavailable, never
faking.

Runs BOTH as a script (``python3 test_board_multispecies.py``, the CI-style invocation) and under
pytest (the test_* functions take no args and importorskip pops.time / pops.physics).
"""
import sys

import pytest

adctime = pytest.importorskip("pops.time")
physics = pytest.importorskip("pops.physics")
from pops import model
from pops.ir.expr import Var


def _three_fluid_board():
    """A 3-fluid (electrons / ions / neutrals) board model with a coupled rate + field solve."""
    m = physics.Model("three_fluid")
    e = m.species("electrons", state=["ne", "mex", "mey"],
                  roles={"ne": "density", "mex": "momentum_x", "mey": "momentum_y"})
    i = m.species("ions", state=["ni", "mix", "miy"])
    n = m.species("neutrals", state=["nn", "mnx", "mny"])
    m.coupled_rate(
        "collision", inputs=[e, i, n],
        outputs={
            e: [i["ni"] - e["ne"], e["ne"], e["ne"]],
            i: [e["ne"] - i["ni"], i["ni"], i["ni"]],
            n: [n["nn"], n["nn"], n["nn"]],
        })
    m.solve_fields_from_species("fields", inputs=[e, i, n],
                                outputs={"phi": None}, solver="geometric_mg")
    return m, e, i, n


def _two_fluid_board():
    """A two-fluid board model whose coupled rate matches the hand-written operator-first one."""
    m = physics.Model("two_fluid")
    e = m.species("electron_state", state=["ne", "mex", "mey"])
    i = m.species("ion_state", state=["ni", "mix", "miy"])
    m.coupled_rate("collision", inputs=[e, i],
                   outputs={e: [i["ni"] - e["ne"], e["ne"], e["ne"]],
                            i: [e["ne"] - i["ni"], i["ni"], i["ni"]]})
    return m, e, i


def _two_fluid_handwritten():
    """The operator-first reference the two-fluid board must match (test_coupled_rate shape)."""
    mod = model.Module("two_fluid")
    e = mod.state_space("electron_state", ("ne", "mex", "mey"))
    i = mod.state_space("ion_state", ("ni", "mix", "miy"))
    bundle = model.RateBundle({"electron_state": model.Rate(e), "ion_state": model.Rate(i)})
    ne, ni = Var("ne", "cons"), Var("ni", "cons")
    mod.operator(name="collision", signature=model.Signature((e, i), bundle),
                 kind="coupled_rate",
                 expr={"electron_state": [ni - ne, ne, ne], "ion_state": [ne - ni, ni, ni]})
    return mod, e, i


def _two_fluid_program(mod, e_space, i_space):
    """A forward-Euler collision step over the two-fluid module (board or hand-written)."""
    P = adctime.Program("two_fluid_collision").bind_operators(mod)
    e_n = P.state("electron_state", space=e_space)
    i_n = P.state("ion_state", space=i_space)
    C = P.call("collision", e_n, i_n)
    P.commit_many({"electron_state": P.linear_combine("e1", e_n + P.dt * C["electron_state"]),
                   "ion_state": P.linear_combine("i1", i_n + P.dt * C["ion_state"])})
    return P


def test_species_no_longer_raises_for_a_second_species():
    # The N > 1 rejection is gone: a 2nd species lowers to the multi-block IR, not NotImplementedError.
    m = physics.Model("two")
    m.species("electrons", state=["ne"])
    m.species("ions", state=["ni"])  # must not raise
    assert set(m.module.list_state_spaces()) == {"electrons", "ions"}


def test_three_species_lower_to_three_state_spaces():
    m, _e, _i, _n = _three_fluid_board()
    mod = m.module
    assert mod.list_state_spaces() == ["electrons", "ions", "neutrals"]
    assert list(mod.state_spaces()["electrons"].components) == ["ne", "mex", "mey"]


def test_coupled_rate_lowers_to_coupled_rate_operator_with_a_bundle():
    m, _e, _i, _n = _three_fluid_board()
    op = m.module.operator_registry().get("collision")
    assert op.kind == "coupled_rate"
    assert isinstance(op.signature.output, model.RateBundle)
    assert set(op.signature.output.keys()) == {"electrons", "ions", "neutrals"}
    # arity 3: three StateSpace inputs
    assert len(op.signature.inputs) == 3


def test_field_solve_lowers_to_a_multi_input_field_operator():
    m, _e, _i, _n = _three_fluid_board()
    op = m.module.operator_registry().get("fields")
    assert op.kind == "field_operator"
    assert len(op.signature.inputs) == 3  # over all three species (solve_fields_from_blocks surface)
    assert isinstance(op.signature.output, model.FieldSpace)


def test_state_handle_indexes_by_component_name():
    m, e, _i, _n = _three_fluid_board()
    # e["ne"] is the conservative Var of that component (the board access of section 12.3/16),
    # identical to the operator-first Var("ne", "cons"). Var has no __eq__ (== builds an
    # expression), so compare by name + kind.
    ne = e["ne"]
    assert isinstance(ne, Var) and ne.name == "ne" and ne.kind == "cons"
    with pytest.raises(KeyError):
        _ = e["not_a_component"]


def test_board_two_fluid_matches_handwritten_operator_first_ir():
    # test 34.10: the board IR is EQUIVALENT to the hand-written operator-first version.
    bm, _be, _bi = _two_fluid_board()
    bmod = bm.module
    hmod, _he, _hi = _two_fluid_handwritten()
    assert bmod.state_spaces() == hmod.state_spaces()
    assert bmod.list_operators() == hmod.list_operators()
    bop = bmod.operator_registry().get("collision")
    hop = hmod.operator_registry().get("collision")
    assert bop.kind == hop.kind
    assert bop.signature == hop.signature
    # the strongest equivalence: the same module hash (the .so cache key)
    assert bmod.module_hash() == hmod.module_hash()


def test_board_two_fluid_emits_identical_cpp_to_handwritten():
    # the lowered program emits BYTE-identical C++ to the hand-written operator-first program.
    bm, be, bi = _two_fluid_board()
    hmod, he, hi = _two_fluid_handwritten()
    bsrc = _two_fluid_program(bm.module, be.space, bi.space).emit_cpp_program(model=None)
    hsrc = _two_fluid_program(hmod, he, hi).emit_cpp_program(model=None)
    assert bsrc == hsrc
    # one shared multi-state kernel binds both species, reads cons from each state (sanity)
    assert bsrc.count("pops::for_each_cell") == 1
    assert "const pops::Real ne = u0A(i, j, 0);" in bsrc
    assert "const pops::Real ni = u1A(i, j, 0);" in bsrc


def test_arbitrary_arity_four_inputs():
    # criterion 25-26: arbitrary arity, no two-input limit.
    m = physics.Model("four_fluid")
    sp = {nm: m.species(nm, state=[c])
          for nm, c in [("a", "na"), ("b", "nb"), ("c", "nc"), ("d", "nd")]}
    m.coupled_rate("quad", inputs=list(sp.values()),
                   outputs={sp["a"]: [sp["b"]["nb"]], sp["b"]: [sp["a"]["na"]],
                            sp["c"]: [sp["d"]["nd"]], sp["d"]: [sp["c"]["nc"]]})
    op = m.module.operator_registry().get("quad")
    assert len(op.signature.inputs) == 4
    assert set(op.signature.output.keys()) == {"a", "b", "c", "d"}


def test_coupled_rate_read_only_catalyst_input():
    # a species may be a READ-ONLY input (catalyst) without being an output block (ionization).
    m = physics.Model("ioniz")
    e = m.species("e", state=["ne"])
    i = m.species("i", state=["ni"])
    n = m.species("n", state=["nn"])  # catalyst: an input, NOT an output block
    m.coupled_rate("ioniz", inputs=[e, i, n],
                   outputs={e: [i["ni"] + n["nn"]], i: [e["ne"] + n["nn"]]})
    op = m.module.operator_registry().get("ioniz")
    assert len(op.signature.inputs) == 3                 # all three are inputs
    assert set(op.signature.output.keys()) == {"e", "i"}  # only two are output blocks


def test_wrong_species_rate_in_affine_combine_errors():
    # test 34.11 (typing): a Rate of the WRONG species cannot be added to a State of another.
    m = physics.Model("tf")
    e = m.species("electrons", state=["ne"])
    i = m.species("ions", state=["ni"])
    m.coupled_rate("collision", inputs=[e, i],
                   outputs={e: [i["ni"] - e["ne"]], i: [e["ne"] - i["ni"]]})
    P = adctime.Program("s").bind_operators(m.module)
    e_n = P.state("electrons", space=e.space)
    i_n = P.state("ions", space=i.space)
    C = P.call("collision", e_n, i_n)
    with pytest.raises(ValueError, match="different state spaces"):
        P.linear_combine("bad", e_n + P.dt * C["ions"])  # electron state + ion rate


def test_coupled_rate_output_component_count_must_match_state():
    # a per-block rate must be full-rank over its block state (one formula per cons component).
    m = physics.Model("tf")
    e = m.species("electrons", state=["ne", "mex"])
    i = m.species("ions", state=["ni"])
    with pytest.raises(ValueError, match="component formula"):
        m.coupled_rate("collision", inputs=[e, i],
                       outputs={e: [e["ne"]], i: [i["ni"]]})  # electron rate is rank 1, not 2


def test_multispecies_compile_routes_through_a_program():
    # a multi-species model is compiled as a multi-block Program, not the single-state shortcut.
    m, _e, _i, _n = _three_fluid_board()
    assert m.check() is None                              # model-level check is a single-species notion
    with pytest.raises(NotImplementedError, match="multi-block"):
        m.compile("ignored.so")


def test_single_species_is_byte_identical_to_state():
    # the N == 1 path is unchanged: m.species == m.state (no multi-block Module created).
    def via_state():
        m = physics.Model("euler")
        m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
        return m

    def via_species():
        m = physics.Model("euler")
        m.species("U", state=["rho", "mx", "my"],
                  roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
        return m

    s = via_species()
    assert s._multi_module is None                       # no parallel runtime for one species
    assert via_state().module.module_hash() == s.module.module_hash()


def test_field_solve_call_lowers_to_solve_fields_from_blocks_over_all_species():
    # Regression (adversarial review): a multi-input field_operator CALLED in a Program must lower to
    # the COUPLED multi-block solve over ALL species -- not solve_fields(args[0]), which would silently
    # drop every species but the first and read only the first charge into the elliptic RHS.
    m, _e, _i, _n = _three_fluid_board()
    mod = m.module
    P = adctime.Program("ms_fields").bind_operators(mod)
    sp = mod.state_spaces()
    e_n = P.state("electrons", space=sp["electrons"])
    i_n = P.state("ions", space=sp["ions"])
    n_n = P.state("neutrals", space=sp["neutrals"])
    f = P.call("fields", e_n, i_n, n_n)
    assert f.op == "solve_fields_from_blocks", "multi-input field op lowers to the coupled solve"
    assert len(f.inputs) == 3, "all three species contribute to the field solve (none dropped)"


def test_duplicate_species_name_raises():
    # A reused species name would silently alias the StateSpace -> fail loud at authoring instead.
    m = physics.Model("dup")
    m.species("electrons", state=["ne"])
    try:
        m.species("electrons", state=["ne2"])
    except ValueError as exc:
        assert "already declared" in str(exc)
    else:
        raise AssertionError("a duplicate species name must raise ValueError")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("PASS test_board_multispecies (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
    sys.exit(0)

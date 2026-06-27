"""Spec 3 multi-species: coupled_rate operator kind + multi-output P.call (ADC-457).

A coupled operator (collisions, ionization, radiation) takes an arbitrary arity of states and
returns a typed RateBundle -- one Rate per participating block. P.call lowers it to a bundle of
per-block rate values usable in affine combinations. This is the IR/authoring slice (pure
Python, locally testable); the C++ coupled-rate kernel codegen is the deferred runtime part.
"""
import pytest

from pops import model
from pops.ir.expr import Var

adctime = pytest.importorskip("pops.time")


def _two_fluid_module():
    """(e, i) -> RateBundle{electrons: Rate(e), ions: Rate(i)} collision operator."""
    mod = model.Module("two_fluid")
    e = mod.state_space("electron_state", ("ne", "mex", "mey"))
    i = mod.state_space("ion_state", ("ni", "mix", "miy"))
    bundle = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    ne, ni = Var("ne", "cons"), Var("ni", "cons")
    mod.operator(name="collision", signature=model.Signature((e, i), bundle),
                 kind="coupled_rate",
                 expr={"electrons": [ni - ne, ne, ne], "ions": [ne - ni, ni, ni]})
    return mod, e, i, bundle


def test_coupled_rate_is_a_valid_kind():
    assert "coupled_rate" in model.OPERATOR_KINDS


def test_rate_bundle_equality_and_hash():
    e = model.StateSpace("e", ("a",))
    i = model.StateSpace("i", ("b",))
    b1 = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    b2 = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    assert b1 == b2 and hash(b1) == hash(b2)
    assert b1 != model.RateBundle({"electrons": model.Rate(e)})


def test_coupled_rate_operator_registers_with_bundle_output():
    mod, _, _, bundle = _two_fluid_module()
    op = mod.operator_registry().get("collision")
    assert op.kind == "coupled_rate"
    assert op.signature.output == bundle


def test_p_call_coupled_rate_returns_indexable_bundle():
    mod, e, i, _ = _two_fluid_module()
    P = adctime.Program("step").bind_operators(mod)
    e_n = P.state("electrons", space=e)
    i_n = P.state("ions", space=i)
    C = P.call("collision", e_n, i_n)
    re_, ri_ = C["electrons"], C["ions"]
    assert re_.vtype == "rhs" and ri_.vtype == "rhs"
    # each per-block rate is usable in an affine combination of its block's state
    e1 = P.linear_combine("e1", e_n + P.dt * re_)
    i1 = P.linear_combine("i1", i_n + P.dt * ri_)
    assert e1.vtype == "state" and i1.vtype == "state"


def test_coupled_rate_arbitrary_arity_three_blocks():
    mod = model.Module("three_fluid")
    e = mod.state_space("e", ("ne",))
    i = mod.state_space("i", ("ni",))
    n = mod.state_space("n", ("nn",))
    bundle = model.RateBundle({"e": model.Rate(e), "i": model.Rate(i), "n": model.Rate(n)})
    z = Var("ne", "cons")
    mod.operator(name="coll3", signature=model.Signature((e, i, n), bundle),
                 kind="coupled_rate", expr={"e": [z], "i": [z], "n": [z]})
    P = adctime.Program("s").bind_operators(mod)
    en, inn, nn = P.state("e", space=e), P.state("i", space=i), P.state("n", space=n)
    C = P.call("coll3", en, inn, nn)
    assert set(C.keys()) == {"e", "i", "n"}


def test_coupled_rate_bundle_unknown_block_errors():
    mod, e, i, _ = _two_fluid_module()
    P = adctime.Program("step").bind_operators(mod)
    C = P.call("collision", P.state("electrons", space=e), P.state("ions", space=i))
    with pytest.raises(KeyError):
        _ = C["neutrals"]


def test_coupled_rate_rejects_schedule_clearly():
    # schedule= on a coupled_rate has no single output to schedule yet -> clear error, not a raw
    # AttributeError from the _CoupledResult having no .attrs.
    mod, e, i, _ = _two_fluid_module()
    P = adctime.Program("step").bind_operators(mod)
    with pytest.raises(ValueError, match="coupled_rate"):
        P.call("collision", P.state("electrons", space=e), P.state("ions", space=i),
               schedule=adctime.every(2))


def test_dump_cpp_plan_shows_coupled_rate_kernel():
    # the C++ plan shows the coupled_rate as ONE multi-state kernel (ADC-457), never a
    # ctx.coupled_rate(...) call that does not exist.
    mod, e, i, _ = _two_fluid_module()
    P = adctime.Program("step").bind_operators(mod)
    e_n, i_n = P.state("electrons", space=e), P.state("ions", space=i)
    C = P.call("collision", e_n, i_n)
    P.linear_combine("e1", e_n + P.dt * C["electrons"])
    plan = P.dump_cpp_plan()
    assert "ADC-457" in plan and "ctx.coupled_rate(" not in plan
    assert "multi-state for_each_cell rate kernel" in plan
    assert "electrons" in plan and "ions" in plan


def test_coupled_rate_now_lowers_to_cpp():
    # ADC-457 part B: a coupled_rate lowers to a multi-state kernel rather than refusing. The honest
    # deferral is now scoped to prim/aux formulas (the cons-only MVP) -- see
    # test_coupled_rate_codegen.py for the emitted kernel shape and the prim-var raise.
    mod, e, i, _ = _two_fluid_module()
    P = adctime.Program("step").bind_operators(mod)
    e_n, i_n = P.state("electrons", space=e), P.state("ions", space=i)
    C = P.call("collision", e_n, i_n)
    P.commit_many({"electrons": P.linear_combine("e1", e_n + P.dt * C["electrons"]),
                   "ions": P.linear_combine("i1", i_n + P.dt * C["ions"])})
    P._check_lowerable(None)  # no longer raises for a cons-only coupled_rate
    src = P.emit_cpp_program(model=None)
    assert src.count("pops::for_each_cell") == 1
    assert "const pops::Real ne =" in src and "const pops::Real ni =" in src

"""Spec 3 unified-scheduler AUTHORING (ADC-458, epic ADC-450).

The schedule vocabulary, the policy chaining, recording a schedule on a Program node, the
cacheable-capability validation, and the honest refusal to lower a non-always schedule (the
runtime that honors caches / accumulate_dt / checkpoint is the C++ part of ADC-458). These are
pure-Python: only pops.time / pops.model are needed, no compiled step is run.
"""
import pytest

from pops import dsl, model

adctime = pytest.importorskip("pops.time")


def _module(cacheable=True):
    """A module with a cacheable field_operator and a (non-cacheable) flux grid_operator."""
    mod = model.Module("sched_demo")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    rho = dsl.Var("rho", "cons")
    mod.operator(name="fields_from_state", signature=(u,) >> fields,
                 kind="field_operator", expr=rho)
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [rho, rho, rho], "y": [rho, rho, rho]})
    if cacheable:
        mod.operator_capabilities("fields_from_state", cacheable=True)
    return mod, u, fields


# --- schedule vocabulary -----------------------------------------------------
def test_always_is_default_recompute():
    s = adctime.always()
    assert s.kind == "always" and s.policy == "recompute" and s.is_always()


def test_every_carries_n_and_is_not_always():
    s = adctime.every(10)
    assert s.kind == "every" and s.params["n"] == 10 and not s.is_always()


def test_every_rejects_non_positive():
    with pytest.raises(ValueError):
        adctime.every(0)


def test_other_kinds_exist():
    assert adctime.when(lambda: True).kind == "when"
    assert adctime.on_start().kind == "on_start"
    assert adctime.on_end().kind == "on_end"
    assert adctime.subcycle(4).kind == "subcycle"


# --- policy chaining ---------------------------------------------------------
def test_policy_chaining():
    assert adctime.every(10).hold().policy == "hold"
    assert adctime.always().skip().policy == "skip"
    assert adctime.every(5).accumulate_dt().policy == "accumulate_dt"
    assert adctime.on_end().zero().policy == "zero"
    assert adctime.every(2).error().policy == "error"
    # chaining keeps the kind + params
    s = adctime.every(7).hold()
    assert s.kind == "every" and s.params["n"] == 7


def test_schedule_repr_reads_like_the_api():
    assert repr(adctime.every(10).hold()) == "every(10).hold()"
    assert repr(adctime.always()) == "always()"


# --- operator_capabilities setter/getter -------------------------------------
def test_operator_capabilities_setter_then_getter():
    mod, _, _ = _module(cacheable=True)
    assert mod.operator_capabilities("fields_from_state")["cacheable"] is True
    # getter form is unchanged for an operator with no declared caps
    assert mod.operator_capabilities("flux").get("cacheable") is None


# --- recording a schedule on a node ------------------------------------------
def test_call_records_schedule_on_value():
    mod, u, _ = _module()
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    f = P.call("fields_from_state", U, schedule=adctime.every(10).hold())
    assert f.attrs["schedule"].policy == "hold"
    assert "schedule" in P.dump_operator_ir()       # inspectable: recorded, not dropped


def test_call_without_schedule_is_unchanged():
    mod, u, _ = _module()
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    f = P.call("fields_from_state", U)
    assert "schedule" not in f.attrs


# --- cacheable validation (criterion 27) -------------------------------------
def test_hold_on_non_cacheable_operator_raises():
    mod, u, _ = _module(cacheable=False)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    with pytest.raises(ValueError, match="not cacheable"):
        P.call("fields_from_state", U, schedule=adctime.every(10).hold())


def test_accumulate_dt_on_non_cacheable_raises():
    mod, u, _ = _module(cacheable=False)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    with pytest.raises(ValueError, match="not cacheable"):
        P.call("fields_from_state", U, schedule=adctime.every(4).accumulate_dt())


def test_hold_on_cacheable_operator_ok():
    mod, u, _ = _module(cacheable=True)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=adctime.every(10).hold())   # no raise


def test_skip_does_not_require_cacheable():
    # skip / recompute / zero produce nothing cached, so they do not require cacheable
    mod, u, _ = _module(cacheable=False)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=adctime.every(10).skip())   # no raise


# --- honesty gate: the two genuinely-unlowerable cases must fail loud, never silently no-op ---
# (ADC-458 codegen lowers every kind/policy EXCEPT on_end() -- no end-of-run signal in a compiled step
# loop -- and a when() over a Python callable. The full policy/kind matrix is in test_scheduler_codegen.)
def test_on_end_schedule_refuses_to_lower():
    mod, u, _ = _module(cacheable=True)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=adctime.on_end().hold())
    with pytest.raises(NotImplementedError, match="ADC-458"):
        P._check_schedules_lowerable()


def test_when_python_callable_refuses_to_lower():
    mod, u, _ = _module(cacheable=True)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    # a when() over a bare Python callable is not a Program value -> cannot lower
    P.call("fields_from_state", U, schedule=adctime.when(lambda: True).hold())
    with pytest.raises(NotImplementedError, match="ADC-458"):
        P._check_schedules_lowerable()


def test_held_solve_fields_now_lowers():
    # ADC-458 codegen: a held field solve lowers to the cache branch -- it must NOT raise (the runtime
    # cadence is exercised in the compiled .so / ROMEO).
    mod, u, _ = _module(cacheable=True)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=adctime.every(10).hold())
    P._check_schedules_lowerable()   # no raise


def test_skip_now_lowers():
    # ADC-458: skip on a field solve lowers (the op runs only when due; the aux is stale off-cadence).
    mod, u, _ = _module(cacheable=True)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=adctime.every(10).skip())
    P._check_schedules_lowerable()   # no raise


def test_always_schedule_lowers_fine():
    mod, u, _ = _module(cacheable=True)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=adctime.always())
    P._check_schedules_lowerable()   # no raise: always() == the default cadence


def test_scheduled_node_serializes_for_codegen():
    # a Schedule object is not JSON-serializable; it must be reduced to its repr in the IR hash
    # (regression: an always()-scheduled node passed the gate then crashed _ir_hash with a TypeError).
    mod, u, _ = _module(cacheable=True)
    P = adctime.Program("p").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=adctime.always())
    h = P._ir_hash()                 # must not raise
    assert isinstance(h, str) and h
    # the schedule is part of the IR identity: a different cadence yields a different hash
    P2 = adctime.Program("p").bind_operators(mod)
    U2 = P2.state("plasma", space=u)
    P2.call("fields_from_state", U2, schedule=adctime.every(10).skip())
    assert P2._ir_hash() != h

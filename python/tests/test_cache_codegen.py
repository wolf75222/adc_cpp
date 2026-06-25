"""Spec 3 scheduler cache CODEGEN (ADC-458): a held solve_fields lowers to the cache branch.

A `solve_fields` node carrying an `every(N).hold()` schedule must codegen to
`if (ctx.cache_should_update(id, N)) { solve; ctx.cache_store_aux(id); } else { ctx.cache_restore_aux(id); }`
-- recompute + cache the System aux every N macro-steps, reuse it in between. This is the emit-level
check (the cache RUNTIME cadence runs in a compiled .so -- ROMEO; the CacheManager is unit-tested by
tests/test_cache_manager.cpp). Other ops/policies still refuse to lower (not yet supported).
"""
import pytest

from adc import dsl, model

adctime = pytest.importorskip("adc.time")


def _module():
    mod = model.Module("held_fields")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    rho = dsl.Var("rho", "cons")
    mod.operator(name="fields_from_state", signature=(u,) >> fields, kind="field_operator", expr=rho)
    mod.operator_capabilities("fields_from_state", cacheable=True)
    return mod, u


def _held_program(schedule):
    mod, u = _module()
    P = adctime.Program("held").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U, schedule=schedule)
    P.commit("plasma", U)
    return P


def test_held_solve_fields_lowers_and_emits_cache_branch():
    P = _held_program(adctime.every(3).hold())
    P._check_schedules_lowerable()                # must NOT raise: solve_fields + hold is lowerable
    cpp = P.emit_cpp_program()
    assert "cache_should_update" in cpp, "due check emitted"
    assert "cache_store_aux" in cpp, "recompute branch stores the aux"
    assert "cache_restore_aux" in cpp, "held branch restores the cached aux"
    # the period N from every(3) reaches the due check
    assert "cache_should_update" in cpp and ", 3)" in cpp


def test_unscheduled_solve_fields_has_no_cache_branch():
    mod, u = _module()
    P = adctime.Program("plain").bind_operators(mod)
    U = P.state("plasma", space=u)
    P.call("fields_from_state", U)               # no schedule
    P.commit("plasma", U)
    cpp = P.emit_cpp_program()
    assert "cache_should_update" not in cpp       # plain unconditional solve, no cache
    assert "ctx.solve_fields_from_state(" in cpp


def test_always_solve_fields_has_no_cache_branch():
    P = _held_program(adctime.always())
    cpp = P.emit_cpp_program()
    assert "cache_should_update" not in cpp       # always() == default cadence, no caching


def test_skip_policy_on_solve_fields_still_refused():
    # only hold is lowered so far; skip/zero/accumulate_dt on solve_fields refuse to lower (ADC-458)
    P = _held_program(adctime.every(5).skip())
    with pytest.raises(NotImplementedError, match="ADC-458"):
        P._check_schedules_lowerable()


def test_hold_on_non_every_kind_still_refused():
    # only every(N).hold lowers; a hold on subcycle/on_start/on_end carries no N and must NOT
    # silently lower to cache_should_update(id, 1) (every-step) -- it refuses (ADC-458).
    for sched in (adctime.subcycle(3).hold(), adctime.on_start().hold(), adctime.on_end().hold()):
        P = _held_program(sched)
        with pytest.raises(NotImplementedError, match="ADC-458"):
            P._check_schedules_lowerable()

#!/usr/bin/env python3
"""Unified Program scheduler CODEGEN (ADC-458, Spec 3 sections 17-18).

Every schedule kind/policy now LOWERS to C++ (`Program._emit_schedule_wrap`), generalizing the
#297 held-field-solve cache branch to any schedulable node. This test pins the EMITTED guard shape
per policy/kind on both a field-solve node (output = the System aux) and a scratch node (output = a
named MultiFab):

  - `every(N)`   -> `if (ctx.cache_should_update(id, N)) { ... }`
  - `on_start()` -> `if ((ctx.macro_step() == 0)) { ... }`
  - `when(cond)` -> reuses the Program Bool predicate token as the due test
  - `subcycle(c)`-> a `for (...) { ... }` sub-loop over the sub-dt
  - `recompute`  -> the body runs only when due, no else
  - `hold`       -> store/restore the cached value (aux or named scratch) off-cadence
  - `skip`       -> the op runs only when due; the value keeps its stale content (no else)
  - `zero`       -> a `set_val(0)` else-branch
  - `accumulate_dt` -> `ctx.cache_accumulate_dt` off-cadence + `ctx.cache_effective_dt` on the due step
  - `error`      -> a `ctx.scheduler_error(...)` else-branch

It also pins that the always()/default lowering is byte-identical to the unscheduled body (the only
file-level difference is the IR hash, which legitimately tracks the schedule attr), and that the two
genuinely-unlowerable cases (on_end(), a when() over a Python callable) still fail loud naming
ADC-458. The cache RUNTIME cadence in a stepping .so is exercised on ROMEO; the CacheManager is
unit-tested by tests/test_cache_manager.cpp. Pure Python: only adc.time / adc.model / adc.dsl.
"""
import sys


def _skip(msg):
    print("skip test_scheduler_codegen (%s)" % msg)
    sys.exit(0)


try:
    from adc import dsl, model
    from adc import time as adctime
except Exception as exc:  # noqa: BLE001  -- _adc unavailable in this interpreter
    _skip("adc unavailable: %s" % exc)


# --- builders ---------------------------------------------------------------
def _field_program(schedule):
    """A program whose held node is a field SOLVE (output = the System aux)."""
    mod = model.Module("sched_fields")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    rho = dsl.Var("rho", "cons")
    mod.operator(name="fields_from_state", signature=(u,) >> fields, kind="field_operator", expr=rho)
    mod.operator_capabilities("fields_from_state", cacheable=True)
    P = adctime.Program("sched").bind_operators(mod)
    U = P.state("plasma", space=u)
    if schedule is None:
        P.call("fields_from_state", U)
    else:
        P.call("fields_from_state", U, schedule=schedule)
    P.commit("plasma", U)
    return P


def _transport_model():
    import adc

    return adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                     transport=adc.IsothermalFlux(), source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))


def _scratch_program(schedule):
    """A program whose scheduled node is an RHS (output = a named scratch MultiFab). The schedule is
    attached directly to the IR node (the cacheable-capability check is exercised by
    test_schedule_authoring); here the focus is the emitted guard shape."""
    P = adctime.Program("sched_rhs")
    dt = P.dt
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    R.attrs["schedule"] = schedule
    P.commit("ions", P.linear_combine("U1", U + dt * R))
    return P


def _emit_field(schedule):
    return _field_program(schedule).emit_cpp_program()


def _emit_scratch(schedule):
    return _scratch_program(schedule).emit_cpp_program(model=_transport_model())


# --- always / default byte-identity -----------------------------------------
def test_always_body_identical_to_unscheduled():
    # the LOWERED body of always() equals the unscheduled body (no guard). The whole-file emit differs
    # only by the IR hash, which legitimately tracks the schedule attr (cache invalidation by design).
    _, plain = _field_program(None)._emit_body()
    _, always = _field_program(adctime.always())._emit_body()
    assert plain == always
    assert "cache_should_update" not in _emit_field(adctime.always())


def test_unscheduled_has_no_guard():
    cpp = _emit_field(None)
    assert "cache_should_update" not in cpp
    assert "ctx.solve_fields_from_state(" in cpp


# --- every(N) cadence (kind) ------------------------------------------------
def test_every_due_test_carries_period():
    cpp = _emit_field(adctime.every(7).hold())
    assert "ctx.cache_should_update(" in cpp and ", 7)" in cpp


# --- on_start (kind) --------------------------------------------------------
def test_on_start_lowers_to_macro_step_zero():
    cpp = _emit_field(adctime.on_start().hold())
    assert "(ctx.macro_step() == 0)" in cpp


# --- when(cond) (kind) ------------------------------------------------------
def test_when_reuses_program_predicate_token():
    P = adctime.Program("when_sched")
    dt = P.dt
    U = P.state("ions")
    f = P.solve_fields(U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    cond = P.norm2(R) < 1e-6  # a Program Bool predicate emitted before the scheduled node
    R2 = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    R2.attrs["schedule"] = adctime.when(cond).hold()
    P.commit("ions", P.linear_combine("U1", U + dt * R2))
    P._check_schedules_lowerable()  # a Program Bool when() lowers
    cpp = P.emit_cpp_program(model=_transport_model())
    assert "< static_cast<adc::Real>(1e-06)" in cpp  # the predicate is the due test
    assert "ctx.cache_should_update" not in cpp       # when() is a predicate, not a period


def test_when_over_python_callable_refuses():
    P = _scratch_program(adctime.when(lambda: True).hold())
    try:
        P._check_schedules_lowerable()
    except NotImplementedError as exc:
        assert "ADC-458" in str(exc)
    else:
        raise AssertionError("when(callable) must refuse to lower")


# --- subcycle (kind) --------------------------------------------------------
def test_subcycle_emits_a_subloop():
    cpp = _emit_field(adctime.subcycle(4))
    assert "for (int _sub" in cpp
    assert "dt / static_cast<adc::Real>(4)" in cpp  # default sub-dt = macro_dt / count
    assert "< 4;" in cpp


def test_subcycle_explicit_dt():
    cpp = _emit_field(adctime.subcycle(3, dt=0.01))
    assert "for (int _sub" in cpp
    assert "0.01" in cpp


def test_subcycle_on_scratch_node_refuses():
    # subcycle is lowerable only for a field solve (aux output): a scratch-output op sub-cycled would
    # declare its scratch inside the loop, out of scope downstream -> refuse loud (ADC-458).
    P = _scratch_program(adctime.subcycle(4))
    try:
        P._check_schedules_lowerable()
    except NotImplementedError as exc:
        assert "ADC-458" in str(exc) and "subcycle" in str(exc)
    else:
        raise AssertionError("subcycle on a scratch-output node must refuse to lower")


# --- policies on a FIELD-SOLVE node (output = aux) --------------------------
def test_field_hold_stores_and_restores_aux():
    cpp = _emit_field(adctime.every(10).hold())
    assert "ctx.cache_store_aux(" in cpp
    assert "ctx.cache_restore_aux(" in cpp


def test_field_zero_emits_aux_set_val_else():
    cpp = _emit_field(adctime.every(4).zero())
    assert "} else {" in cpp
    assert "ctx.aux().set_val(static_cast<adc::Real>(0));" in cpp


def test_field_accumulate_dt_reads_effective_dt():
    cpp = _emit_field(adctime.every(7).accumulate_dt())
    assert "ctx.cache_effective_dt(" in cpp     # the due step reads the summed skipped dt
    assert "ctx.cache_accumulate_dt(" in cpp    # the skip step accumulates the real dt
    assert "ctx.cache_store_aux(" in cpp


def test_field_skip_runs_only_when_due():
    cpp = _emit_field(adctime.every(5).skip())
    assert "skip: stale aux off-cadence" in cpp
    assert "ctx.cache_restore_aux" not in cpp   # skip does not cache (stale, no restore)
    assert "} else {" not in cpp.split("skip: stale aux off-cadence")[1].split("\n", 1)[0]


def test_field_error_emits_scheduler_error_else():
    cpp = _emit_field(adctime.every(3).error())
    assert "ctx.scheduler_error(" in cpp
    assert "policy=error" in cpp


def test_field_recompute_runs_only_when_due():
    cpp = _emit_field(adctime.every(2).recompute())
    assert "if (ctx.cache_should_update(" in cpp
    assert "cache_store_aux" not in cpp  # recompute does not cache
    assert "cache_restore_aux" not in cpp


# --- policies on a SCRATCH node (output = a named MultiFab) ------------------
def test_scratch_hold_caches_named_scratch():
    # a held NON-solve_fields scratch now caches: the output decl is hoisted out of the guard, the
    # fill + cache_store_scratch run when due, cache_restore_scratch off-cadence.
    cpp = _emit_scratch(adctime.every(10).hold())
    assert "ctx.cache_store_scratch(" in cpp
    assert "ctx.cache_restore_scratch(" in cpp
    # the output scratch is DECLARED before the guard (so both branches see it)
    decl_idx = cpp.index("adc::MultiFab r")
    guard_idx = cpp.index("if (ctx.cache_should_update(")
    assert decl_idx < guard_idx


def test_scratch_zero_sets_the_scratch_to_zero():
    cpp = _emit_scratch(adctime.every(4).zero())
    assert ".set_val(static_cast<adc::Real>(0));" in cpp
    assert "ctx.aux().set_val" not in cpp  # a scratch node zeroes its OWN buffer, not the aux


def test_scratch_accumulate_dt_uses_scratch_cache():
    cpp = _emit_scratch(adctime.every(7).accumulate_dt())
    assert "ctx.cache_effective_dt(" in cpp
    assert "ctx.cache_accumulate_dt(" in cpp
    assert "ctx.cache_store_scratch(" in cpp
    assert "ctx.cache_restore_scratch(" in cpp


def test_scratch_decl_hoisted_for_skip():
    # the scratch decl must be OUTSIDE the guard so the (stale) buffer stays in scope for downstream
    cpp = _emit_scratch(adctime.every(5).skip())
    decl_idx = cpp.index("adc::MultiFab r")
    guard_idx = cpp.index("if (ctx.cache_should_update(")
    assert decl_idx < guard_idx


# --- genuinely unlowerable: on_end (no end-of-run signal) -------------------
def test_on_end_refuses_to_lower():
    P = _field_program(adctime.on_end().hold())
    try:
        P._check_schedules_lowerable()
    except NotImplementedError as exc:
        assert "ADC-458" in str(exc) and "on_end" in str(exc)
    else:
        raise AssertionError("on_end() must refuse to lower (no end-of-run signal)")


# --- script entry point (CI runs each test file as `python3 file.py`) -------
def _run_as_script():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print("  [OK ] %s" % fn.__name__)
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print("  [XX ] %s: %s" % (fn.__name__, exc))
    if fails:
        print("FAIL test_scheduler_codegen: %d failure(s)" % fails)
        sys.exit(1)
    print("PASS test_scheduler_codegen (%d checks)" % len(fns))


if __name__ == "__main__":
    _run_as_script()

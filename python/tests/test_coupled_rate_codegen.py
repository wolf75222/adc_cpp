#!/usr/bin/env python3
"""Spec 3 coupled-rate multi-state kernel codegen (ADC-457 part B).

A ``coupled_rate`` operator (collisions / ionization, Spec 3 criterion 27) takes N input state
spaces and returns a RateBundle (one Rate per block); its component formulas reference cons vars from
MULTIPLE input states. ``emit_cpp_program`` now lowers it to ONE multi-state ``adc::for_each_cell``
kernel filling every block's rate scratch at once -- not independent single-block rates.

This test pins the EMIT + the un-gate + the cons-only deferral (pure Python, no compile): the
``_check_lowerable`` no longer raises for a cons-only coupled_rate; the emitted C++ holds ONE
for_each_cell that binds BOTH input states' Array4s, binds ``ne``/``ni`` from the respective states,
writes BOTH block rate scratches, and the per-block out composes in a forward step; a formula that
references a PRIM var raises the ADC-457 NotImplementedError. The compiled-.so collision STEP (the
runtime that fills the rate and advances the species) is validated on ROMEO (Kokkos-only AOT), NOT
here. Real engine only; skips if adc.time is unavailable, never faking.

Runs BOTH as a script (``python3 test_coupled_rate_codegen.py``, the CI-style invocation) and under
pytest (the test_* functions take no args and importorskip adc.time).
"""
import sys

import pytest

adctime = pytest.importorskip("adc.time")
from adc import dsl, model  # noqa: E402 (after importorskip so a missing adc skips cleanly)


def _two_fluid_module(electron_expr=None):
    """(e, i) -> RateBundle{electrons: Rate(e), ions: Rate(i)} collision operator.

    The electron block's component formulas default to the cons-only [ni - ne, ne, ne]; pass
    @p electron_expr to inject a non-cons formula (the prim-var deferral check)."""
    mod = model.Module("two_fluid")
    e = mod.state_space("electron_state", ("ne", "mex", "mey"))
    i = mod.state_space("ion_state", ("ni", "mix", "miy"))
    bundle = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    ne, ni = dsl.Var("ne", "cons"), dsl.Var("ni", "cons")
    e_comps = electron_expr if electron_expr is not None else [ni - ne, ne, ne]
    mod.operator(name="collision", signature=model.Signature((e, i), bundle),
                 kind="coupled_rate",
                 expr={"electrons": e_comps, "ions": [ne - ni, ni, ni]})
    return mod, e, i, bundle


def _two_fluid_program():
    """A two-block program: solve the collision rate, then forward-Euler each species by its rate."""
    mod, e, i, _ = _two_fluid_module()
    P = adctime.Program("two_fluid_collision").bind_operators(mod)
    e_n = P.state("electrons", space=e)
    i_n = P.state("ions", space=i)
    C = P.call("collision", e_n, i_n)
    P.commit_many({"electrons": P.linear_combine("e1", e_n + P.dt * C["electrons"]),
                   "ions": P.linear_combine("i1", i_n + P.dt * C["ions"])})
    return mod, P


def test_check_lowerable_no_longer_raises_for_coupled_rate():
    # un-gate: a cons-only coupled_rate lowers (was a hard NotImplementedError before ADC-457 part B).
    _mod, P = _two_fluid_program()
    P._check_lowerable(None)  # must not raise


def test_emit_one_multi_state_for_each_cell():
    _mod, P = _two_fluid_program()
    src = P.emit_cpp_program(model=None)
    # ONE shared kernel fills both blocks' rate scratches (not two independent single-block rates).
    assert src.count("adc::for_each_cell") == 1


def test_emit_binds_both_input_state_array4s():
    _mod, P = _two_fluid_program()
    src = P.emit_cpp_program(model=None)
    # The two species states are ctx.state(0) / ctx.state(1) -> u0 / u1; each binds its OWN read handle.
    assert "const adc::ConstArray4 u0A = u0.fab(li).const_array();" in src
    assert "const adc::ConstArray4 u1A = u1.fab(li).const_array();" in src


def test_emit_binds_cons_vars_from_respective_states():
    _mod, P = _two_fluid_program()
    src = P.emit_cpp_program(model=None)
    # ne is component 0 of the electron state (u0), ni component 0 of the ion state (u1): each cons
    # local reads from ITS OWN state's Array4, so the coupled formulas reference the right cells.
    assert "const adc::Real ne = u0A(i, j, 0);" in src
    assert "const adc::Real ni = u1A(i, j, 0);" in src


def test_emit_writes_both_block_rate_scratches():
    _mod, P = _two_fluid_program()
    src = P.emit_cpp_program(model=None)
    # Both blocks' rate scratches are allocated (shaped like their own state) and WRITTEN in the kernel.
    assert "= ctx.rhs_scratch_like(u0);" in src and "= ctx.rhs_scratch_like(u1);" in src
    # electron rate = [ni - ne, ne, ne]; ion rate = [ne - ni, ni, ni], each into its block's scratch.
    assert "_electronsA(i, j, 0) = (ni - ne);" in src
    assert "_electronsA(i, j, 1) = ne;" in src
    assert "_ionsA(i, j, 0) = (ne - ni);" in src
    assert "_ionsA(i, j, 2) = ni;" in src


def test_per_block_out_composes_in_a_forward_step():
    _mod, P = _two_fluid_program()
    src = P.emit_cpp_program(model=None)
    # e1 = e_n + dt * re : the electron rate scratch is axpy'd onto the electron state with dt; same
    # for the ion block. Each coupled_rate_out aliases its block's scratch (no separate compute).
    electron_scratch = next(ln.split("=")[0].strip().split()[-1]
                            for ln in src.splitlines()
                            if "ctx.rhs_scratch_like(u0)" in ln)
    ion_scratch = next(ln.split("=")[0].strip().split()[-1]
                       for ln in src.splitlines()
                       if "ctx.rhs_scratch_like(u1)" in ln)
    assert ("ctx.axpy(acc" in src) and ("static_cast<adc::Real>(dt), %s);" % electron_scratch) in src
    assert ("static_cast<adc::Real>(dt), %s);" % ion_scratch) in src


def test_coupled_rate_with_prim_var_is_deferred():
    # cons-only MVP: a component formula referencing a PRIM var raises the ADC-457 NotImplementedError
    # (never silently emits an undefined `ue` local), naming the deferral precisely.
    ne, ni = dsl.Var("ne", "cons"), dsl.Var("ni", "cons")
    ue = dsl.Var("ue", "prim")  # a PRIM reference -> deferred
    mod, e, i, _ = _two_fluid_module(electron_expr=[ni - ne + ue, ne, ne])
    P = adctime.Program("two_fluid_collision_prim").bind_operators(mod)
    e_n = P.state("electrons", space=e)
    i_n = P.state("ions", space=i)
    C = P.call("collision", e_n, i_n)
    P.commit_many({"electrons": P.linear_combine("e1", e_n + P.dt * C["electrons"]),
                   "ions": P.linear_combine("i1", i_n + P.dt * C["ions"])})
    with pytest.raises(NotImplementedError, match="ADC-457"):
        P._check_lowerable(None)


def test_unbound_registry_is_deferred():
    # Reaching the operator body needs the bound registry; emitting a coupled_rate node whose Program
    # has no registry raises a clear ADC-457 error rather than an opaque AttributeError.
    _mod, P = _two_fluid_program()
    P._registry = None  # simulate an unbound program holding a coupled_rate node
    with pytest.raises(NotImplementedError, match="ADC-457"):
        P._check_lowerable(None)


def test_coupled_rate_codegen_emits_no_forbidden_cpp_tokens():
    # Guard (mirrors test_time_local_newton): the emitted .cpp must use ProgramContext primitives only,
    # never raw std::vector / std::function / Eigen:: / new / malloc -- in code OR comments.
    _mod, P = _two_fluid_program()
    src = P.emit_cpp_program(model=None)
    for tok in ("std::vector", "std::function", "Eigen::", "new ", "malloc"):
        assert tok not in src, "emitted coupled-rate .cpp must not contain %r" % tok


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("PASS test_coupled_rate_codegen (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
    sys.exit(0)

#!/usr/bin/env python3
"""Spec 5 sec.10: strict EARLY install validation + AmrSystem.install parity (ADC-479 / ADC-463).

``System.install`` (and now ``AmrSystem.install``) reads the compiled artifact's DECLARED bind
inputs -- ``compiled.arguments()`` (ADC-509) -- and rejects, BEFORE any native call, an install that
does not supply a REQUIRED argument (instance / runtime param / aux / solver), with one clear
actionable error naming exactly what is missing and how to supply it. The check is INERT: it reads
metadata and compares dicts (no compile, no bind, no allocation), so it needs NO live Kokkos.

The tests build a SYNTHETIC ``CompiledProblem`` -- a real in-memory ``pops.time.Program`` + a real
``CompiledModel`` metadata carrier, no .so on disk (the same stub ADC-509's introspection test uses)
-- and assert:

  - the pure router ``collect_missing_arguments`` flags only REQUIRED-and-missing inputs (a const
    param / the default-Poisson solver are NOT demanded);
  - ``System.install`` raises the clear early error when a required param / aux / instance is
    missing, BEFORE the native ``install_program`` runs (mocked to detect ordering);
  - a VALID install (everything required supplied, or a model with NO required inputs) PASSES
    validation unchanged -- the no-break discipline;
  - a NATIVE install (``compiled=None``) carries no declared arguments and is skipped;
  - ``AmrSystem.install`` has signature parity with ``System.install`` and runs the SAME validation,
    then a clear NotImplementedError for the genuinely un-wired AMR compiled-program path.

The Kokkos-gated end-to-end (a real ``compile_problem`` .so whose native install actually runs) is
covered by ``test_unified_install.py`` / ``test_install_requirement_validation.py``; here we test the
PURE-PYTHON validation logic, which is the new surface. Pytest + __main__ guard (CI runs
``python3 <file>``).
"""
import inspect
import sys

try:
    import numpy as np

    import pops
    from pops.codegen.loader import CompiledModel, CompiledProblem
    from pops.physics.model import Param
    from pops.runtime._system_unified_install import (collect_missing_arguments,
                                                      validate_install_arguments)
    from pops import time as adctime
except Exception as exc:  # noqa: BLE001 -- pops/numpy unavailable in this interpreter
    print("skip test_install_validation (pops/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 8


# ---------------------------------------------------------------------------
# Synthetic compiled artifact: a real lowered Program + a real CompiledModel, no .so.
# ---------------------------------------------------------------------------

def _program(name="installval_demo"):
    """A real in-memory Program: a state, an elliptic field solve, a Forward-Euler commit on
    'plasma' (so arguments().instances commits the 'plasma' block)."""
    P = adctime.Program(name)
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields("phi", U)
    R = P.rhs(state=U, fields=f, flux=True, sources=["default"])
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    return P


def _model(*, n_vars=3, aux_names=("B_z",), params=None):
    """A real CompiledModel metadata carrier (no .so) -- the engine class, carrying only metadata."""
    cons = ["rho", "mx", "my", "E"][:n_vars]
    roles = ["Density", "MomentumX", "MomentumY", "Energy"][:n_vars]
    return CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=cons, cons_roles=roles, prim_names=cons, n_vars=n_vars, gamma=1.4,
        n_aux=len(aux_names), params=params or {}, caps={"cpu": True}, abi_key="SIG|c++|c++23",
        model_hash="modelhash", cxx="c++", std="c++23", aux_extra_names=list(aux_names))


def _compiled(*, aux_names=("B_z",), params=None):
    """A SYNTHETIC CompiledProblem carrying a known arguments(): a real Program + a real model."""
    return CompiledProblem("/tmp/pops-cache/problem.so", _program(),
                           _model(aux_names=aux_names, params=params),
                           "SIG|c++|c++23", "c++", "c++23")


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    assert cond, label


# ---------------------------------------------------------------------------
# Pure router: collect_missing_arguments (host-testable, no engine call).
# ---------------------------------------------------------------------------

def test_pure_router_flags_only_required_and_missing():
    """collect_missing_arguments flags a missing required instance / param / aux, but NOT a const
    param and NOT the default-Poisson solver (arguments() never marks 'phi' required)."""
    print("== pure router: only required-and-missing inputs are flagged ==")
    params = {"cs2": Param("cs2", 1.0, kind="runtime"),
              "g": Param("g", 1.4, kind="const")}
    args = _compiled(aux_names=("B_z",), params=params).arguments()

    missing = collect_missing_arguments(args, set(), set(), set(), set())
    chk(any("'plasma'" in m for m in missing), "missing required instance 'plasma' is flagged")
    chk(any("'cs2'" in m for m in missing), "missing required runtime param 'cs2' is flagged")
    chk(any("'B_z'" in m for m in missing), "missing required aux 'B_z' is flagged")
    chk(not any("'g'" in m for m in missing), "a CONST param ('g') is NOT required")
    chk(not any("solver" in m for m in missing),
        "the default Poisson solver is NOT required (it has a working default)")
    # Each line is actionable: it names the install keyword to supply the input.
    chk(any("install(params=" in m for m in missing), "the param line names install(params=)")
    chk(any("install(aux=" in m for m in missing), "the aux line names install(aux=)")


def test_pure_router_passes_when_everything_supplied():
    """collect_missing_arguments returns [] once every required input is supplied (no false
    positive) -- a block already added on the sim counts as provided."""
    print("== pure router: nothing missing once everything required is supplied ==")
    params = {"cs2": Param("cs2", 1.0, kind="runtime")}
    args = _compiled(aux_names=("B_z",), params=params).arguments()
    chk(collect_missing_arguments(args, {"plasma"}, {"cs2"}, {"B_z"}, {"phi"}) == [],
        "all supplied -> no missing argument")
    chk(collect_missing_arguments(args, {"plasma"}, {"cs2"}, {"B_z"}, set()) == [],
        "the solver is not required, so omitting it is still complete")


def test_pure_router_aggregates_multiple_missing():
    """Several missing required inputs aggregate into one list (one line each)."""
    print("== pure router: multiple missing inputs aggregate ==")
    params = {"cs2": Param("cs2", 1.0, kind="runtime"),
              "nu": Param("nu", 0.1, kind="runtime")}
    args = _compiled(aux_names=("B_z",), params=params).arguments()
    missing = collect_missing_arguments(args, {"plasma"}, set(), set(), set())
    chk(len(missing) == 3, "two missing params + one missing aux -> 3 lines (got %d)" % len(missing))


# ---------------------------------------------------------------------------
# System.install: the early error, raised BEFORE the native install_program.
# ---------------------------------------------------------------------------

def test_install_raises_before_native_when_required_missing():
    """System.install raises the clear early error (naming cs2 + B_z) BEFORE install_program runs.

    install_program is mocked to flip a flag; the validation must raise before it is ever called, so
    a misuse cannot leave a half-configured System."""
    print("== System.install raises BEFORE the native install_program ==")
    params = {"cs2": Param("cs2", 1.0, kind="runtime")}
    cp = _compiled(aux_names=("B_z",), params=params)
    sim = pops.System(n=N, L=1.0, periodic=True)
    called = {"native": False}
    sim.install_program = lambda *a, **k: called.__setitem__("native", True)
    try:
        sim.install(cp, instances={"plasma": {"model": _model(params=params),
                                              "initial": np.ones((3, N, N))}})
        chk(False, "install should have raised (missing cs2 + B_z)")
    except ValueError as exc:
        msg = str(exc)
        chk("cs2" in msg and "B_z" in msg, "the error names both missing required inputs")
        chk("install(params=" in msg and "install(aux=" in msg, "the error is actionable")
        chk(called["native"] is False, "install_program did NOT run (validation fired first)")


def test_install_missing_instance_is_flagged():
    """A program that commits a block not supplied (and not already added) is rejected."""
    print("== System.install flags a missing required instance ==")
    cp = _compiled(aux_names=())  # no aux, no params -> only the 'plasma' instance is required
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.install_program = lambda *a, **k: None
    try:
        sim.install(cp, instances={})  # 'plasma' neither passed nor added
        chk(False, "install should have raised for the missing 'plasma' instance")
    except ValueError as exc:
        chk("plasma" in str(exc), "the error names the missing instance 'plasma'")


def _stub_lower_layer(sim, record):
    """Replace the lower-layer mutators install() lowers to with recorders, so a host test can drive
    install() past validation WITHOUT a Kokkos .so (the synthetic CompiledModel has no dlopen-able
    .so). The block name is recorded so block_names() reflects the wiring. install() reads the model
    via _resolve_instance_model (passes a CompiledModel through unchanged), so only the engine-call
    methods need stubbing."""
    record.setdefault("blocks", [])
    record.setdefault("native", False)
    sim.add_equation = lambda name, model, **k: record["blocks"].append(name)
    sim.set_state = lambda *a, **k: None
    sim._install_solver = lambda *a, **k: None
    sim.install_program = lambda *a, **k: record.__setitem__("native", True)
    sim.block_names = lambda: list(record["blocks"])


def test_valid_install_passes_validation_unchanged():
    """A model with NO required aux / params passes validation and REACHES the native call.

    The no-break discipline: a previously valid install (the ssprk2-style shape: instances + a
    solver, no aux) must not be rejected. The lower-layer add path is stubbed (the synthetic model
    has no dlopen-able .so); we assert validation passed and install_program was reached."""
    print("== a valid install (no required aux/params) passes through unchanged ==")
    cp = _compiled(aux_names=())  # Euler-style: no aux, no runtime params
    sim = pops.System(n=N, L=1.0, periodic=True)
    record = {}
    _stub_lower_layer(sim, record)
    sim.install(cp,
                instances={"plasma": {"model": _model(aux_names=()),
                                      "initial": np.ones((3, N, N)),
                                      "spatial": pops.FiniteVolume()}},
                solvers={"phi": pops.lib.fields.GeometricMG()})
    chk(record["native"] is True, "validation passed and the native install_program was reached")
    chk(record["blocks"] == ["plasma"], "the instance was wired (validation did not block it)")


def test_native_install_skips_validation():
    """A NATIVE install (compiled=None) carries no declared arguments -> validation is skipped (the
    instance model is the native source of truth, not a compiled artifact)."""
    print("== native install (compiled=None) skips the compiled-argument validation ==")
    sim = pops.System(n=N, L=1.0, periodic=True)
    record = {}
    _stub_lower_layer(sim, record)
    sim.install(None, instances={"plasma": {"model": _model(aux_names=()),
                                            "initial": np.ones((3, N, N)),
                                            "spatial": pops.FiniteVolume()}})
    chk(record["blocks"] == ["plasma"], "native install wired the block without a compiled check")
    chk(record["native"] is False, "compiled=None skips install_program (no compiled Program)")


def test_validate_helper_is_inert_on_bad_handle():
    """validate_install_arguments never breaks a valid install: a handle without arguments(), or one
    whose arguments() raises, is skipped (conservative -- a missing check is better than a false
    reject)."""
    print("== validate_install_arguments is conservative on an un-introspectable handle ==")
    sim = pops.System(n=N, L=1.0, periodic=True)

    class _NoArgs:  # a handle that is not a CompiledProblem (no arguments())
        so_path = "/tmp/x.so"

    # No arguments() -> skipped silently (no raise).
    validate_install_arguments(sim, _NoArgs(), {}, {}, {}, {})

    class _Raises:
        so_path = "/tmp/x.so"

        def arguments(self):
            raise RuntimeError("introspection blew up")

    # arguments() raising -> swallowed (a valid install must not break on an introspection bug).
    validate_install_arguments(sim, _Raises(), {}, {}, {}, {})
    chk(True, "an un-introspectable handle does not raise (conservative skip)")


# ---------------------------------------------------------------------------
# AmrSystem.install: signature parity + the SAME validation.
# ---------------------------------------------------------------------------

def test_amr_install_signature_parity():
    """AmrSystem.install mirrors System.install's signature exactly (parameter list parity)."""
    print("== AmrSystem.install signature parity with System.install ==")
    sys_params = list(inspect.signature(pops.System.install).parameters)
    amr_params = list(inspect.signature(pops.AmrSystem.install).parameters)
    chk(sys_params == amr_params, "same parameter list (got %r vs %r)" % (amr_params, sys_params))


def test_amr_install_runs_the_same_validation():
    """AmrSystem.install runs the SAME early validation: a compiled install missing a required aux
    raises the clear ValueError BEFORE the compiled-path NotImplementedError."""
    print("== AmrSystem.install runs the same early validation ==")
    cp = _compiled(aux_names=("B_z",))
    amr = pops.AmrSystem(n=N, L=1.0)
    try:
        amr.install(cp, instances={"plasma": {"model": _model(), "initial": np.ones((3, N, N))}})
        chk(False, "AMR install should have raised for the missing B_z")
    except ValueError as exc:
        chk("B_z" in str(exc), "the AMR validation names the missing required aux 'B_z'")


def test_amr_compiled_path_is_honest_notimplemented():
    """Once validation passes, a COMPILED time Program is not installable on AMR today -> a clear
    NotImplementedError (no install_program seam), not a silent or faked install."""
    print("== AmrSystem.install: compiled path is an honest NotImplementedError ==")
    cp = _compiled(aux_names=("B_z",))
    amr = pops.AmrSystem(n=N, L=1.0)
    try:
        amr.install(cp, instances={"plasma": {"model": _model(), "initial": np.ones((3, N, N))}},
                    aux={"B_z": np.ones(N * N)})
        chk(False, "the compiled AMR path should raise NotImplementedError")
    except NotImplementedError as exc:
        chk("install_program" in str(exc) or "single-level" in str(exc),
            "the message explains why a compiled Program is not installable on AMR")


def test_amr_native_params_and_cadence_rejected():
    """A native AMR install honestly rejects params= (no set_block_params) and cadence= (no
    set_program_cadence) rather than dropping them silently."""
    print("== AmrSystem.install rejects un-wired native params / cadence ==")
    amr = pops.AmrSystem(n=N, L=1.0)
    try:
        amr.install(None, params={"nu": 1.0})
        chk(False, "params= should raise on AMR")
    except NotImplementedError as exc:
        chk("set_block_params" in str(exc) or "params" in str(exc), "params rejection is explicit")
    try:
        amr.install(None, cadence=adctime.CompiledTime(substeps=2))
        chk(False, "cadence= should raise on AMR")
    except NotImplementedError as exc:
        chk("cadence" in str(exc) or "set_program_cadence" in str(exc),
            "cadence rejection is explicit")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, exc))
    print("\n%d/%d test functions passed" % (len(fns) - failed, len(fns)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)

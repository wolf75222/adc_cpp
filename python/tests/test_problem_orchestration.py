"""ADC-492 (Spec 5 sec.5.16 / sec.11): pops.Problem assembly + pops.compile/bind + PhysicsModel.

These are PURE-PYTHON tests of the inert assembly, the alias, and the thin dispatch wiring.
The real ``.so`` compile (``compile_problem``) and the runtime install/run are Kokkos-gated
and validated on CI / ROMEO, so the dispatch tests MONKEYPATCH ``compile_problem`` /
``System`` / ``AmrSystem`` to assert routing WITHOUT a real compile. Every deferred route is
asserted to raise loudly (never fake success).

Runs both under pytest and as a plain script (``python3 test_problem_orchestration.py``); the
CI runner executes it as a script (the ``__main__`` guard below).
"""
import sys

try:
    import pops
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    from pops.fields import FieldProblem
    from pops.math import laplacian
    from pops.codegen import orchestration
except Exception as exc:  # noqa: BLE001
    print("skip test_problem_orchestration (pops unavailable: %s)" % exc)
    sys.exit(0)


# --- tiny stand-ins (no compiler / no runtime) -----------------------------
class _StubModel:
    """A physics stand-in exposing the ``.dsl`` engine model pops.compile resolves."""

    def __init__(self, name="stub"):
        self.name = name
        self.dsl = object()  # what _resolve_problem_model returns


class _StubSolver:
    name = "GeometricMG"
    scheme = "geometric_mg"
    options = {}


class _StubCompiled:
    """A compiled-handle stand-in: only ``.so_path`` + the carried problem/target."""

    def __init__(self, target="system", problem=None):
        self.so_path = "/tmp/stub.so"
        self.model = None
        self._target = target
        self._problem = problem


def _poisson_problem():
    """A minimal valid Poisson FieldProblem named 'phi' (the default-served field)."""
    return FieldProblem(name="phi", unknown="phi",
                        equation=(-laplacian("phi") == "charge_density"),
                        solver=_StubSolver())


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --- assembly + chaining + inspect -----------------------------------------
def test_assembly_chaining_and_inspect():
    model = _StubModel("ne")
    prob = (pops.Problem(name="plasma")
            .block("ne", physics=model, spatial=None)
            .param("alpha", default=1.0, kind="const")
            .aux("B_z", value=None))
    _check(prob is prob.block.__self__, "setters operate on the same problem")
    _check(prob.layout.name == "Uniform", "default layout is Uniform")
    info = prob.inspect()
    _check(info["name"] == "plasma", "name carried")
    _check(set(info["blocks"]) == {"ne"}, "block recorded")
    _check(info["params"]["alpha"]["default"] == 1.0, "param recorded")
    _check("B_z" in info["aux"], "aux recorded")
    _check(prob.options()["n_blocks"] == 1, "options report n_blocks")
    print("ok test_assembly_chaining_and_inspect")


def test_block_requires_physics_and_no_duplicate():
    prob = pops.Problem()
    try:
        prob.block("ne", physics=None)
        raise AssertionError("block with no physics must raise")
    except ValueError:
        pass
    prob.block("ne", physics=_StubModel())
    try:
        prob.block("ne", physics=_StubModel())
        raise AssertionError("duplicate block must raise")
    except ValueError:
        pass
    print("ok test_block_requires_physics_and_no_duplicate")


def test_field_type_checked():
    prob = pops.Problem()
    try:
        prob.field("not a FieldProblem")
        raise AssertionError("field must reject a non-FieldProblem")
    except TypeError:
        pass
    prob.field(_poisson_problem())
    _check("phi" in prob._fields, "field registered by name")
    print("ok test_field_type_checked")


def test_amr_property():
    # Uniform layout -> amr raises ValueError.
    try:
        pops.Problem().amr
        raise AssertionError("amr on a Uniform layout must raise")
    except ValueError:
        pass
    # AMR layout -> returns a handle whose .refine chains back to the problem.
    prob = pops.Problem(layout=AMR(CartesianMesh()))
    handle = prob.amr
    from pops.mesh.amr import RegridEvery
    returned = handle.refine(regrid=RegridEvery(20))
    _check(returned is prob, "amr.refine chains back to the problem")
    _check(prob.layout.regrid is not None, "refine recorded the regrid policy")
    print("ok test_amr_property")


# --- validate(): structural pass + each deferred case raising ---------------
def test_validate_structural_pass():
    prob = pops.Problem().block("ne", physics=_StubModel()).field(_poisson_problem())
    _check(prob.validate() is True, "a single-block + Poisson-field problem validates")
    _check(bool(prob.available()) is True, "available() is yes for a valid problem")
    print("ok test_validate_structural_pass")


def test_validate_requires_a_block():
    try:
        pops.Problem().validate()
        raise AssertionError("a problem with no block must not validate")
    except ValueError:
        pass
    print("ok test_validate_requires_a_block")


def test_validate_multi_block_deferred():
    prob = (pops.Problem().block("ne", physics=_StubModel())
            .block("ni", physics=_StubModel()))
    try:
        prob.validate()
        raise AssertionError("multi-block must raise NotImplementedError")
    except NotImplementedError as exc:
        _check("multi-block" in str(exc), "multi-block message is explicit")
    print("ok test_validate_multi_block_deferred")


def test_validate_non_poisson_field_deferred():
    field = FieldProblem(name="temperature", unknown="T",
                         equation=(-laplacian("T") == "src"), solver=_StubSolver())
    prob = pops.Problem().block("ne", physics=_StubModel()).field(field)
    try:
        prob.validate()
        raise AssertionError("a non-Poisson field must raise NotImplementedError")
    except NotImplementedError as exc:
        _check("non-Poisson" in str(exc), "non-Poisson message is explicit")
    print("ok test_validate_non_poisson_field_deferred")


def test_validate_outputs_deferred():
    class _Policy:
        name = "checkpoint"
    prob = pops.Problem().block("ne", physics=_StubModel()).output(_Policy())
    try:
        prob.validate()
        raise AssertionError("a non-empty output policy must raise NotImplementedError")
    except NotImplementedError as exc:
        _check("output" in str(exc), "output message is explicit")
    print("ok test_validate_outputs_deferred")


def test_validate_name_collision():
    field = _poisson_problem()
    prob = pops.Problem().block("phi", physics=_StubModel()).field(field)
    try:
        prob.validate()
        raise AssertionError("a block/field name collision must raise")
    except ValueError as exc:
        _check("share name" in str(exc), "collision message is explicit")
    print("ok test_validate_name_collision")


# --- PhysicsModel alias identity -------------------------------------------
def test_physics_model_alias_identity():
    _check(pops.physics.PhysicsModel is pops.physics.Model, "alias is the same class object")
    _check(pops.PhysicsModel is pops.physics.Model, "top-level alias is the same class object")
    _check(pops.physics.Model.__name__ == "Model", "class __name__ stays 'Model' (alias not rename)")
    # Existing pops.physics.Model consumers keep working.
    instance = pops.physics.Model("legacy")
    _check(type(instance).__name__ == "Model", "instances still report __name__ == 'Model'")
    _check(isinstance(instance, pops.PhysicsModel), "PhysicsModel is usable in isinstance")
    print("ok test_physics_model_alias_identity")


# --- compile(): layout-driven dispatch via a monkeypatched compile_problem --
def test_compile_layout_drives_target(monkeypatch=None):
    captured = {}

    def _fake_compile_problem(*, time, model, backend, target, **kw):
        captured.update(time=time, model=model, backend=backend, target=target)
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = pops.Problem(name="u").block("ne", physics=_StubModel())
        compiled = orchestration.compile(prob, time=object())
        _check(captured["target"] == "system", "Uniform layout routes to target='system'")
        _check(captured["backend"] == "production", "default backend forwarded")
        _check(compiled._target == "system", "target carried on the handle")
        _check(compiled._problem is prob, "problem carried on the handle")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_layout_drives_target")


def test_compile_amr_deferred():
    prob = pops.Problem(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
    try:
        orchestration.compile(prob, time=object())
        raise AssertionError("AMR layout compile must raise NotImplementedError")
    except NotImplementedError as exc:
        _check("AMR" in str(exc), "AMR-deferred message is explicit")
    print("ok test_compile_amr_deferred")


def test_compile_missing_time_raises():
    prob = pops.Problem().block("ne", physics=_StubModel())
    try:
        orchestration.compile(prob)  # no time= and no problem.time(...)
        raise AssertionError("missing time scheme must raise (no silent default)")
    except NotImplementedError as exc:
        _check("time scheme is required" in str(exc), "missing-time message is explicit")
    print("ok test_compile_missing_time_raises")


def test_compile_problem_time_setter_honored(monkeypatch=None):
    captured = {}

    def _fake_compile_problem(*, time, model, backend, target, **kw):
        captured.update(time=time, target=target)
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        sentinel = object()
        prob = pops.Problem().block("ne", physics=_StubModel()).time(sentinel)
        orchestration.compile(prob)  # time taken from problem._time
        _check(captured["time"] is sentinel, "problem.time(...) is honored when time= omitted")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_problem_time_setter_honored")


def test_compile_multi_block_raises(monkeypatch=None):
    prob = (pops.Problem().block("ne", physics=_StubModel())
            .block("ni", physics=_StubModel()))
    try:
        orchestration.compile(prob, time=object())
        raise AssertionError("multi-block compile must raise")
    except NotImplementedError:
        pass
    print("ok test_compile_multi_block_raises")


# --- bind(): System vs AmrSystem dispatch via a monkeypatched runtime -------
class _RecordingSim:
    """A System / AmrSystem stand-in that records the install(...) call."""

    last = {}

    def install(self, compiled=None, *, instances=None, params=None, aux=None,
                solvers=None, cadence=None):
        _RecordingSim.last = {"compiled": compiled, "instances": instances, "params": params,
                              "aux": aux, "solvers": solvers, "cadence": cadence}


def _bind_with_stub_runtime(target):
    """Run bind() with System/AmrSystem replaced by recording stubs; return the chosen class."""
    import pops.runtime.system as rtsys

    class _StubSystem(_RecordingSim):
        pass

    class _StubAmrSystem(_RecordingSim):
        pass

    orig_sys, orig_amr = rtsys.System, rtsys.AmrSystem
    rtsys.System, rtsys.AmrSystem = _StubSystem, _StubAmrSystem
    try:
        prob = pops.Problem().block("ne", physics=_StubModel()).field(_poisson_problem())
        compiled = _StubCompiled(target=target, problem=prob)
        sim = orchestration.bind(compiled, initial_state={"ne": [1.0]})
        return type(sim), _RecordingSim.last, _StubSystem, _StubAmrSystem
    finally:
        rtsys.System, rtsys.AmrSystem = orig_sys, orig_amr


def test_bind_system_dispatch():
    sim_class, last, stub_system, _ = _bind_with_stub_runtime("system")
    _check(sim_class is stub_system, "target='system' binds a System")
    _check(last["compiled"].so_path == "/tmp/stub.so", "compiled handle passed to install")
    _check(set(last["instances"]) == {"ne"}, "the block becomes one install instance")
    _check(last["instances"]["ne"]["initial"] == [1.0], "initial state routed by block name")
    _check("phi" in last["solvers"], "the Poisson field solver derived from the problem")
    print("ok test_bind_system_dispatch")


def test_bind_amr_dispatch():
    sim_class, _, _, stub_amr = _bind_with_stub_runtime("amr_system")
    _check(sim_class is stub_amr, "target='amr_system' binds an AmrSystem")
    print("ok test_bind_amr_dispatch")


def test_bind_rejects_non_compiled():
    try:
        orchestration.bind(object())
        raise AssertionError("bind must reject a handle without .so_path")
    except TypeError:
        pass
    print("ok test_bind_rejects_non_compiled")


def test_bind_unknown_initial_state_raises():
    prob = pops.Problem().block("ne", physics=_StubModel())
    compiled = _StubCompiled(problem=prob)
    import pops.runtime.system as rtsys
    orig = rtsys.System
    rtsys.System = _RecordingSim
    try:
        orchestration.bind(compiled, initial_state={"nope": [0.0]})
        raise AssertionError("an unknown block in initial_state must raise")
    except ValueError as exc:
        _check("unknown block" in str(exc), "unknown-block message is explicit")
    finally:
        rtsys.System = orig
    print("ok test_bind_unknown_initial_state_raises")


# --- monkeypatch helpers (work under pytest fixture OR the bare __main__ runner) ---
_SAVED = []


def _patch(monkeypatch, dotted, value):
    module_name, attr = dotted.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    if monkeypatch is not None:
        monkeypatch.setattr(module, attr, value)
    else:
        _SAVED.append((module, attr, getattr(module, attr)))
        setattr(module, attr, value)


def _unpatch(monkeypatch):
    if monkeypatch is not None:
        return  # pytest restores it
    while _SAVED:
        module, attr, original = _SAVED.pop()
        setattr(module, attr, original)


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()

"""pops.codegen.orchestration -- thin pops.compile / pops.bind over the existing runtime.

These are the Spec 5 sec.11 lowering entry points for a :class:`pops.problem.Problem`:

* :func:`compile` validates the assembly, picks the compile target from the LAYOUT
  (``Uniform`` -> ``"system"``, ``AMR`` -> ``"amr_system"``; no user ``target=`` string),
  resolves the block's physics to the model the existing ``compile_problem`` wants, and calls
  ``compile_problem`` unchanged. It carries the originating problem + target on the handle.
* :func:`bind` dispatches ``System`` vs ``AmrSystem`` from the carried target, assembles the
  per-instance state mapping, and calls the unified ``sim.install(compiled, instances=, ...)``
  -- the ONE install seam (``pops.runtime._system_unified_install``). No parallel runtime.

There is NO new codegen and NO new install machinery here: this module ORCHESTRATES the
proven pieces. Every not-yet-wired route raises a clear ``NotImplementedError``.

Import-graph rule (Spec 4 / sec.4): ``codegen`` may import only ir / model / physics / time /
lib at module scope. The runtime (System / AmrSystem), mesh (AMR) and problem types are pulled
LAZILY inside the function bodies, so this module adds no forbidden cross-layer edge.
"""


def compile(problem, backend="production", time=None, **kwargs):
    """Lower a :class:`pops.problem.Problem` to a compiled handle (thin over ``compile_problem``).

    Validates @p problem, derives the compile target from its LAYOUT (``Uniform`` -> system,
    ``AMR`` -> amr_system), resolves the single block's physics to the model ``compile_problem``
    accepts, and returns the ``CompiledProblem`` with ``_problem`` / ``_target`` attached. The
    time scheme is explicit: @p time (a ``pops.time.Program``), else ``problem.time(...)``; a
    missing scheme raises -- there is NO silent default. The deferred routes (``layout=AMR``,
    multi-block) raise a clear ``NotImplementedError``.

    Args:
        problem: The :class:`pops.problem.Problem` assembly to lower.
        backend: The codegen backend forwarded to ``compile_problem`` (default "production").
        time: The ``pops.time.Program`` time scheme; falls back to ``problem._time``.
        **kwargs: Extra keyword args forwarded verbatim to ``compile_problem`` (so_path /
            force / cxx / include / std / debug / libraries).

    Returns:
        The ``CompiledProblem`` handle, with ``._problem`` and ``._target`` set.
    """
    # Lazy imports keep the codegen layer's module-scope import graph clean (no mesh / runtime).
    from pops.mesh.layouts import AMR

    problem.validate()
    target = "amr_system" if isinstance(problem.layout, AMR) else "system"

    if target == "amr_system":
        raise NotImplementedError(
            "pops.compile: layout=AMR is deferred (PR-2); use layout=Uniform(...) for now")
    if len(problem._blocks) != 1:
        raise NotImplementedError(
            "pops.compile: multi-block lowering is deferred; declare exactly one block "
            "(got %d)" % len(problem._blocks))

    time = time if time is not None else problem._time
    if time is None:
        raise NotImplementedError(
            "pops.compile: a time scheme is required; pass time=pops.time.Program(...) or set "
            "it on the problem with problem.time(...). There is no default time scheme.")

    _, spec = next(iter(problem._blocks.items()))
    model = _resolve_problem_model(spec["physics"])

    from pops.codegen.compile_drivers import compile_problem
    compiled = compile_problem(time=time, model=model, backend=backend, target=target, **kwargs)
    compiled._problem = problem
    compiled._target = target
    return compiled


def bind(compiled, *, initial_state=None, state=None, params=None, aux=None,
         solvers=None, cadence=None):
    """Wire a compiled handle onto the runtime (thin over the unified ``sim.install``).

    Dispatches ``System`` vs ``AmrSystem`` from the target carried on @p compiled (set by
    :func:`compile`), builds the per-instance state mapping from the problem's blocks and the
    supplied initial state, derives the field solvers from the problem's field problems (an
    explicit @p solvers overrides), and calls ``sim.install(compiled, instances=, params=,
    aux=, solvers=, cadence=)`` -- the single Spec-3 install seam. Returns the bound
    simulation (the ``System`` / ``AmrSystem`` is the Simulation facade for now).

    Args:
        compiled: A ``CompiledProblem`` from :func:`compile` (carries ``_problem`` / ``_target``).
        initial_state: dict {block_name: array} of per-block initial state (alias: @p state).
        state: Alias for @p initial_state (only one may be given).
        params: dict {param_name: value} of runtime parameter overrides.
        aux: dict {aux_name: array} of static aux inputs.
        solvers: dict {field: solver} overriding the per-field solvers from the problem.
        cadence: optional ``pops.CompiledTime`` macro-step cadence.

    Returns:
        The bound ``System`` / ``AmrSystem`` simulation handle.
    """
    so_path = getattr(compiled, "so_path", None)
    if so_path is None:
        raise TypeError(
            "pops.bind: expected a compiled handle from pops.compile(...) (with .so_path); "
            "got %r" % type(compiled).__name__)
    if initial_state is not None and state is not None:
        raise TypeError("pops.bind: pass either initial_state= or state=, not both")
    initial = initial_state if initial_state is not None else state

    problem = getattr(compiled, "_problem", None)
    target = getattr(compiled, "_target", "system")

    from pops.runtime.system import AmrSystem, System
    sim_class = AmrSystem if target == "amr_system" else System
    sim = sim_class()

    instances = _assemble_instances(problem, initial or {})
    field_solvers = _problem_field_solvers(problem)
    field_solvers.update(solvers or {})

    sim.install(compiled, instances=instances, params=params or {}, aux=aux or {},
                solvers=field_solvers, cadence=cadence)
    return sim


def _resolve_problem_model(physics):
    """Resolve a block's physics to the model ``compile_problem`` accepts.

    A blackboard :class:`pops.physics.Model` exposes the underlying ``pops.dsl`` engine model
    via ``.dsl`` -- that is what ``compile_problem(model=...)`` wants. A ``pops.model.Module``
    or a raw ``pops.dsl`` model is forwarded as-is (``compile_problem`` lowers a ``Module``
    itself). ``None`` raises, so a block with no physics never reaches codegen.
    """
    if physics is None:
        raise ValueError("pops.compile: the block has no physics model to resolve")
    dsl_model = getattr(physics, "dsl", None)
    if dsl_model is not None:
        return dsl_model
    return physics


def _assemble_instances(problem, initial):
    """Build the ``sim.install`` instances mapping from the problem's blocks + initial state.

    Each block becomes ``{name: {"model": physics_dsl, "spatial": spatial, "initial": state}}``
    -- the shape the unified install consumes. The per-block initial state comes from @p initial
    (keyed by block name); an unknown key raises so a typo is not silently dropped.
    """
    if problem is None:
        raise TypeError("pops.bind: the compiled handle carries no problem assembly "
                        "(was it produced by pops.compile?)")
    unknown = sorted(set(initial) - set(problem._blocks))
    if unknown:
        raise ValueError("pops.bind: initial state for unknown block(s) %s; declared blocks: %s"
                         % (unknown, sorted(problem._blocks)))
    instances = {}
    for name, spec in problem._blocks.items():
        entry = {"model": _resolve_problem_model(spec["physics"]), "spatial": spec["spatial"]}
        if name in initial:
            entry["initial"] = initial[name]
        instances[name] = entry
    return instances


def _problem_field_solvers(problem):
    """The {field_name: solver} mapping derived from the problem's field problems."""
    if problem is None:
        return {}
    return {name: fp.solver for name, fp in problem._fields.items() if fp.solver is not None}


__all__ = ["compile", "bind"]

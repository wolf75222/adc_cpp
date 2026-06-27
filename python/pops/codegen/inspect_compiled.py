"""pops.codegen.inspect_compiled -- INERT introspection of a compiled artifact (Spec 5 sec.12).

The compiled-artifact introspection surface (criteria #44-49, epic ADC-479): three value
classes and the pure builders that populate them from the metadata a
:class:`pops.codegen.loader.CompiledProblem` ALREADY carries (its lowered ``pops.time.Program``
and its physical model), plus the compile artifacts on disk.

  - :class:`Arguments` (sec.12.2, #44-45) lists the RUNTIME inputs the artifact expects at
    :meth:`pops.System.install` (instances / params / aux / solvers / outputs / layout), WITHOUT
    binding or reading any runtime array. It mirrors ``System.install``'s five keyword groups.
  - :class:`MemoryEstimate` (sec.12.3, #46) turns the Program's GRID-RELATIVE static cost
    (``Program.estimate``: field-sized passes) into an ABSOLUTE byte estimate over a mesh shape,
    as a FORMULA -- it allocates nothing (no ``MultiFab``). Every assumption is inspectable.
  - the metadata attributes (sec.12.4, #48-49) live on :class:`CompiledProblem` itself; the
    helpers here only feed its ``arguments`` / ``estimate_memory`` methods.

Nothing here compiles, binds, dlopens or allocates: the builders read Python-side metadata only.
The module imports ``pops.mesh`` lazily (in-function) to respect the codegen layering (a codegen
module may not import ``pops.mesh`` at module scope; cf. tests/architecture/test_import_graph.py).
"""

import json

# Bytes per double-precision cell value. The core is 2D (n x n cells), one field component is a
# full grid traversal; a "field pass" in Program.estimate is one such state-sized buffer.
_BYTES_PER_CELL = 8


# ---------------------------------------------------------------------------
# sec.12.2 -- Arguments: the runtime inputs the artifact expects at bind
# ---------------------------------------------------------------------------

class Arguments:
    """The runtime inputs a compiled artifact expects at ``System.install`` (Spec 5 sec.12.2).

    A plain, inert value describing what a caller must SUPPLY to bind the artifact -- distinct
    from ``CompiledProblem.requirements`` (the compile-time constraints). It lists, per group:

      - ``instances``: the physics blocks the Program commits (name -> state space / component
        count / required), the ``instances=`` dict ``install`` consumes;
      - ``params``: the model's declared parameters (name -> type / kind / required), routed to
        ``install(params=...)`` (only ``kind == "runtime"`` is settable at bind);
      - ``aux``: the static aux inputs the model declares (name -> layout / required), the
        ``install(aux=...)`` dict;
      - ``solvers``: the elliptic field solves the Program performs (field -> problem / solver),
        the ``install(solvers=...)`` dict;
      - ``outputs``: the field outputs / diagnostics the Program records (informational);
      - ``layout_runtime``: the mesh layout the artifact targets (layout / requires_mpi /
        ghost_depth).

    It is built by :func:`build_arguments` from the carried Program + model; it neither compiles,
    binds nor reads any runtime array. ``str(args)`` is a readable table; :meth:`to_dict` /
    :meth:`to_json` serialise it.
    """

    def __init__(self, *, instances, params, aux, solvers, outputs, layout_runtime,
                 program_name=None):
        self.instances = dict(instances)
        self.params = dict(params)
        self.aux = dict(aux)
        self.solvers = dict(solvers)
        self.outputs = dict(outputs)
        self.layout_runtime = dict(layout_runtime)
        self.program_name = program_name

    def to_dict(self):
        """A plain-dict view of every argument group (JSON-ready)."""
        return {"program": self.program_name,
                "instances": {k: dict(v) for k, v in self.instances.items()},
                "params": {k: dict(v) for k, v in self.params.items()},
                "aux": {k: dict(v) for k, v in self.aux.items()},
                "solvers": {k: dict(v) for k, v in self.solvers.items()},
                "outputs": {k: dict(v) for k, v in self.outputs.items()},
                "layout_runtime": dict(self.layout_runtime)}

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __str__(self):
        lines = ["arguments for compiled artifact %r (bind inputs)"
                 % (self.program_name or "problem")]
        lines.append("  instances (install instances=):")
        for name, spec in sorted(self.instances.items()):
            lines.append("    %-14s state=%s comps=%s required=%s"
                         % (name, spec.get("state"), spec.get("components"),
                            spec.get("required")))
        lines.append("  params (install params=):")
        for name, spec in sorted(self.params.items()):
            lines.append("    %-14s type=%s kind=%s required=%s"
                         % (name, spec.get("type"), spec.get("kind"), spec.get("required")))
        lines.append("  aux (install aux=):")
        for name, spec in sorted(self.aux.items()):
            lines.append("    %-14s layout=%s required=%s"
                         % (name, spec.get("layout"), spec.get("required")))
        lines.append("  solvers (install solvers=):")
        for name, spec in sorted(self.solvers.items()):
            lines.append("    %-14s problem=%s solver=%s"
                         % (name, spec.get("problem"), spec.get("solver")))
        lines.append("  outputs:")
        for name, spec in sorted(self.outputs.items()):
            lines.append("    %-14s kind=%s" % (name, spec.get("kind")))
        lr = self.layout_runtime
        lines.append("  layout_runtime : layout=%s requires_mpi=%s ghost_depth=%s"
                     % (lr.get("layout"), lr.get("requires_mpi"), lr.get("ghost_depth")))
        return "\n".join(lines)

    def __repr__(self):
        return ("Arguments(instances=%d, params=%d, aux=%d, solvers=%d)"
                % (len(self.instances), len(self.params), len(self.aux), len(self.solvers)))


def _model_metadata(compiled):
    """Read the carried model's (name -> ...) metadata WITHOUT loading the .so.

    Returns ``(cons_names, n_cons, params, aux_names, n_aux, state_space)`` from the physical model
    a :class:`CompiledProblem` carries (a ``pops.physics.facade.Model`` or a ``CompiledModel``).
    Either exposes ``cons_names`` / ``n_vars`` / ``params``; the named-aux table is
    ``aux_extra_names`` (CompiledModel) and the count is ``n_aux``. A handle that carries no model
    yields empty metadata (the Program structure is still introspectable)."""
    model = getattr(compiled, "model", None)
    if model is None:
        return [], 0, {}, [], 0, "U"
    cons = list(getattr(model, "cons_names", []) or [])
    n_cons = int(getattr(model, "n_vars", len(cons)) or len(cons))
    params = dict(getattr(model, "params", {}) or {})
    aux_names = list(getattr(model, "aux_extra_names", []) or [])
    n_aux = int(getattr(model, "n_aux", len(aux_names)) or len(aux_names))
    spaces = getattr(model, "list_state_spaces", None)
    state_space = "U"
    if callable(spaces):
        names = spaces()
        if names:
            state_space = names[0]
    return cons, n_cons, params, aux_names, n_aux, state_space


def _solver_arguments(program):
    """Elliptic field solves the Program performs (field name -> {problem, solver}).

    Read from the lowered IR: every ``solve_fields`` / ``solve_fields_from_blocks`` node names an
    elliptic field; ``solve_linear`` is a Krylov solve. The runtime serves these via
    ``install(solvers={field: <GeometricMG/...>})`` (today only the default Poisson field is wired;
    cf. ``System._install_solver``). We do not know the chosen solver brick at compile time -- it is
    a BIND input -- so ``solver`` is reported as ``None`` ("to be supplied")."""
    solvers = {}
    for value in getattr(program, "_values", []):
        op = value.op
        if op in ("solve_fields", "solve_fields_from_blocks"):
            field = value.name or "phi"
            solvers[field] = {"problem": "elliptic", "solver": None}
        elif op == "solve_linear":
            field = value.name or "krylov"
            solvers[field] = {"problem": "linear_system", "solver": None}
    return solvers


def build_arguments(compiled):
    """Build the :class:`Arguments` of a compiled artifact from its carried metadata (sec.12.2).

    Sources, all Python-side (no compile / bind / runtime read):

      - instances: the blocks the Program COMMITS (``program.commits()`` -- the blocks it advances);
        each is required and carries the model's conservative state space + component count;
      - params: the model's declared parameters (``model.params``); ``kind`` is the declared kind
        (``runtime`` settable at bind, ``const`` frozen at compile);
      - aux: the model's named aux inputs (``model.aux_extra_names``), each required;
      - solvers: the elliptic / Krylov solves in the Program IR (:func:`_solver_arguments`);
      - outputs: the values the Program records for output (``store_history`` / ``record`` ops);
      - layout_runtime: the target layout (System, single level -- the only ``target`` a compiled
        Program supports today), MPI optionality and the model ghost depth.
    """
    program = getattr(compiled, "program", None)
    cons, n_cons, params, aux_names, n_aux, state_space = _model_metadata(compiled)

    # Instances: the blocks the Program commits. A read-only block (never committed) is still a
    # bind input, but the Program only references blocks it commits or reads; the commit set is the
    # authoritative list of advanced blocks (criterion 23: the block is bound by name).
    commits = {}
    if program is not None and hasattr(program, "commits"):
        commits = program.commits()
    instances = {}
    for block in sorted(commits):
        instances[block] = {"state": state_space, "components": n_cons,
                            "required": True, "conservative": list(cons)}
    if not instances and n_cons:
        # A Program that commits no block (pure field/diagnostic) still needs the physics block the
        # model describes; surface it under the model name so the table is never silently empty.
        instances[getattr(compiled, "program_name", None) or "block"] = {
            "state": state_space, "components": n_cons, "required": True,
            "conservative": list(cons)}

    param_args = {}
    for name, param in params.items():
        kind = getattr(param, "kind", "const")
        ptype = getattr(param, "type", None) or type(getattr(param, "value", 0.0)).__name__
        param_args[name] = {"type": str(ptype), "kind": str(kind),
                            "required": kind == "runtime"}

    aux_args = {name: {"layout": "cell", "required": True} for name in aux_names}

    solver_args = _solver_arguments(program) if program is not None else {}

    outputs = {}
    if program is not None:
        for value in getattr(program, "_values", []):
            if value.op == "store_history":
                outputs[value.name or "history"] = {"kind": "history"}
            elif value.op == "record" or value.op == "record_scalar":
                outputs[value.name or "diagnostic"] = {"kind": "diagnostic"}

    caps = getattr(getattr(compiled, "model", None), "caps", {}) or {}
    layout_runtime = {"layout": "system", "requires_mpi": False,
                      "ghost_depth": _ghost_depth(compiled),
                      "supports_mpi": bool(caps.get("mpi", False))}

    return Arguments(instances=instances, params=param_args, aux=aux_args,
                     solvers=solver_args, outputs=outputs, layout_runtime=layout_runtime,
                     program_name=getattr(compiled, "program_name", None))


def _ghost_depth(compiled):
    """Conservative ghost (halo) depth of the model: 2 for a finite-volume MUSCL stencil.

    The artifact does not record its reconstruction stencil width in today's metadata, so we report
    the conservative default the runtime uses for second-order MUSCL (a 2-cell halo). A richer
    manifest (a follow-up) would carry the per-block ghost depth; until then this is a documented
    upper-bound assumption, surfaced in the estimate's ``assumptions``."""
    return 2


# ---------------------------------------------------------------------------
# sec.12.3 -- MemoryEstimate: an absolute byte FORMULA over a mesh shape
# ---------------------------------------------------------------------------

class MemoryEstimate:
    """An ABSOLUTE memory estimate for a compiled artifact on a given mesh (Spec 5 sec.12.3).

    A FORMULA, not an allocation: it multiplies the Program's grid-relative static cost
    (``Program.estimate``: field-sized buffer passes, scratch buffer count) by the cell count and
    the per-cell byte size, and adds the persistent state / field-output / aux footprint. It never
    constructs a ``MultiFab``. Every figure is a category in :attr:`categories` (bytes); the
    :attr:`assumptions` list records what the estimate takes for granted (it is CONSERVATIVE: it
    over-counts scratch as if no codegen reuse happened beyond the static reuse report, and ignores
    in-solver V-cycle traffic). :meth:`by_block` / :meth:`by_solver` / :meth:`by_scratch` slice it.
    """

    def __init__(self, *, categories, cells, mesh_shape, n_cons, n_aux, scratch_buffers,
                 assumptions, conservative=True, layout="system"):
        self.categories = dict(categories)   # category -> bytes
        self.cells = int(cells)
        self.mesh_shape = tuple(mesh_shape)
        self.n_cons = int(n_cons)
        self.n_aux = int(n_aux)
        self.scratch_buffers = int(scratch_buffers)
        self.assumptions = list(assumptions)
        self.conservative = bool(conservative)
        self.layout = str(layout)

    @property
    def total_bytes(self):
        """Sum of every category, in bytes."""
        return sum(self.categories.values())

    def by_block(self):
        """The per-block (state-sized) categories: persistent state, RHS / state scratch."""
        keys = ("state", "rhs_scratch", "state_scratch", "field_output", "aux")
        return {k: self.categories[k] for k in keys if k in self.categories}

    def by_solver(self):
        """The elliptic / Krylov / multigrid categories (the field solves)."""
        keys = ("scalar_field", "krylov", "multigrid")
        return {k: self.categories[k] for k in keys if k in self.categories}

    def by_scratch(self):
        """The transient scratch categories (RHS / state scratch, halo, MPI buffers)."""
        keys = ("rhs_scratch", "state_scratch", "halo", "mpi_buffer", "amr_patch")
        return {k: self.categories[k] for k in keys if k in self.categories}

    def to_dict(self):
        """A plain-dict view: every category, the total, the mesh + assumptions (JSON-ready)."""
        return {"total_bytes": self.total_bytes, "categories": dict(self.categories),
                "cells": self.cells, "mesh_shape": list(self.mesh_shape),
                "n_cons": self.n_cons, "n_aux": self.n_aux,
                "scratch_buffers": self.scratch_buffers, "layout": self.layout,
                "conservative": self.conservative, "assumptions": list(self.assumptions)}

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def _mib(self, n_bytes):
        return n_bytes / (1024.0 * 1024.0)

    def __str__(self):
        lines = ["memory estimate on mesh %s (%d cells, %d cons, %d aux) -- %s formula"
                 % (self.mesh_shape, self.cells, self.n_cons, self.n_aux,
                    "conservative" if self.conservative else "tight")]
        for name in sorted(self.categories):
            lines.append("  %-14s %12d B  (%8.2f MiB)"
                         % (name, self.categories[name], self._mib(self.categories[name])))
        lines.append("  %-14s %12d B  (%8.2f MiB)"
                     % ("TOTAL", self.total_bytes, self._mib(self.total_bytes)))
        if self.assumptions:
            lines.append("  assumptions:")
            for note in self.assumptions:
                lines.append("    - %s" % note)
        return "\n".join(lines)

    def __repr__(self):
        return ("MemoryEstimate(total=%d B, cells=%d, categories=%d)"
                % (self.total_bytes, self.cells, len(self.categories)))


def _mesh_shape(mesh):
    """Return ``(cells, (nx, ny))`` for a mesh argument WITHOUT touching the runtime.

    Accepts a ``pops.mesh.CartesianMesh`` (or any object exposing ``n``), or a plain int / 2-tuple.
    The core is 2D (n x n), so a scalar ``n`` means ``n * n`` cells. A richer mesh carrying an
    explicit ``(nx, ny)`` is honoured if present."""
    n = getattr(mesh, "n", mesh)
    if isinstance(n, (tuple, list)):
        if len(n) != 2:
            raise ValueError("estimate_memory: mesh shape must be 2D (got %r)" % (n,))
        nx, ny = int(n[0]), int(n[1])
    else:
        nx = ny = int(n)
    if nx <= 0 or ny <= 0:
        raise ValueError("estimate_memory: mesh extents must be positive (got %dx%d)" % (nx, ny))
    return nx * ny, (nx, ny)


def build_memory_estimate(compiled, mesh, *, platform=None, layout=None):
    """Build the :class:`MemoryEstimate` for a compiled artifact on ``mesh`` (sec.12.3).

    A pure FORMULA over the Program's static cost (``Program.estimate``) and the carried model's
    component counts -- it allocates nothing. The byte budget per category, with ``C`` the cell
    count, ``B`` = 8 bytes/cell, ``n_cons`` / ``n_aux`` the model's component counts:

      - ``state``        = n_cons * C * B           (the persistent conservative state)
      - ``field_output`` = (#field solves) * C * B  (one scalar field per elliptic solve)
      - ``aux``          = n_aux * C * B            (the static aux channel)
      - ``rhs_scratch``  = (#scratch buffers after reuse) * n_cons * C * B   (the step-body scratch)
      - ``state_scratch``= 1 state buffer * n_cons * C * B (the committed-state staging copy)
      - ``scalar_field`` = (#field solves) * C * B  (the elliptic unknown buffer)
      - ``krylov``       = (#linear solves) * 4 * C * B (Krylov needs ~4 work vectors per solve)
      - ``multigrid``    = (#field solves) * (4/3) * C * B (the geometric V-cycle hierarchy ~ 4/3 C)
      - ``halo``         = ghost_depth * perimeter * n_cons * B (the ghost ring, 2D)
      - ``mpi_buffer``   = same as halo, only when ``platform`` requests MPI (else 0)
      - ``amr_patch``    = for an ``AMR`` layout: a CONSERVATIVE per-level patch budget

    @p platform optional hint (e.g. ``"mpi"`` / ``"cpu"``) to include the MPI halo exchange buffer;
    @p layout optional ``pops.mesh.layouts.AMR``/``Uniform`` to estimate an AMR hierarchy. For an
    AMR layout the estimate is CONSERVATIVE (full refinement of every level); a tight figure needs a
    bind (the regrid pattern is data-dependent).
    """
    program = getattr(compiled, "program", None)
    cells, shape = _mesh_shape(mesh)
    _cons, n_cons, _params, _aux_names, n_aux, _space = _model_metadata(compiled)
    n_cons = max(n_cons, 1)  # a degenerate empty model still stages 1 component (never 0 bytes)

    est = program.estimate() if (program is not None and hasattr(program, "estimate")) \
        else {"buffer_count": 1, "heavy_kernels": 0}
    scratch_buffers = int(est.get("buffer_count", 1))
    n_field_solves = int(est.get("heavy_kernels", 0))
    n_linear_solves = sum(1 for v in getattr(program, "_values", [])
                          if v.op == "solve_linear") if program is not None else 0
    n_elliptic = max(n_field_solves - n_linear_solves, 0)

    cell_field = cells * _BYTES_PER_CELL          # one scalar field (n x n doubles)
    state_field = n_cons * cell_field             # one full conservative-state buffer

    categories = {
        "state": state_field,
        "state_scratch": state_field,             # the committed-state staging copy
        "rhs_scratch": scratch_buffers * state_field,
        "field_output": n_elliptic * cell_field,
        "aux": n_aux * cell_field,
        "scalar_field": n_elliptic * cell_field,
        "krylov": n_linear_solves * 4 * cell_field,
        "multigrid": int(n_elliptic * (4.0 / 3.0) * cell_field),
    }

    ghost = _ghost_depth(compiled)
    nx, ny = shape
    perimeter = 2 * (nx + ny)                      # cells on the domain boundary ring (2D)
    halo = ghost * perimeter * n_cons * _BYTES_PER_CELL
    categories["halo"] = halo

    requires_mpi = bool(platform) and "mpi" in str(platform).lower()
    categories["mpi_buffer"] = halo if requires_mpi else 0

    assumptions = [
        "double precision: %d bytes per cell value" % _BYTES_PER_CELL,
        "2D core: %d cells = %d x %d" % (cells, nx, ny),
        "scratch counted AFTER the Program's static buffer-reuse report (%d buffers); the codegen "
        "may keep more, so this is a lower bound on scratch reuse" % scratch_buffers,
        "ghost halo depth assumed %d (conservative MUSCL stencil; not recorded in today's metadata)"
        % ghost,
        "Krylov work vectors assumed 4 per linear solve; multigrid hierarchy ~ 4/3 of the fine grid",
        "in-solver V-cycle / smoother traffic is NOT counted (solver-dependent, out of a static "
        "structural estimate)",
    ]

    layout_kind = "system"
    if layout is not None:
        layout_kind, amr_bytes, amr_notes = _amr_patch_budget(layout, state_field, cell_field,
                                                              n_elliptic)
        if amr_bytes is not None:
            categories["amr_patch"] = amr_bytes
            assumptions.extend(amr_notes)

    if requires_mpi:
        assumptions.append("MPI halo-exchange buffer included (platform=%r); a rank-local subdomain "
                           "would be smaller -- this is the single-rank whole-domain ring" % platform)

    return MemoryEstimate(categories=categories, cells=cells, mesh_shape=shape, n_cons=n_cons,
                          n_aux=n_aux, scratch_buffers=scratch_buffers, assumptions=assumptions,
                          conservative=True, layout=layout_kind)


def _amr_patch_budget(layout, state_field, cell_field, n_elliptic):
    """A CONSERVATIVE AMR patch budget from an ``AMR`` layout descriptor (no bind).

    Returns ``(layout_kind, amr_patch_bytes, notes)``. For a ``Uniform`` layout there is no extra
    patch budget (``amr_patch_bytes`` is ``None``). For an ``AMR(max_levels=L, ratio=r)`` layout the
    worst case fully refines every level: a level ``k`` covering the whole domain at refinement
    ``r^k`` has ``r^(2k)`` times the base cells (2D). Summing the geometric series over the refined
    levels (1..L-1) gives the extra fine-grid footprint on top of the base level. This is an UPPER
    bound (real regrids refine a fraction of the domain); a tight figure needs a bind. The mesh
    import is lazy to respect the codegen layering."""
    from pops.mesh.layouts import AMR, Uniform  # lazy: codegen may not import mesh at module scope
    if isinstance(layout, Uniform):
        return "uniform", None, []
    if not isinstance(layout, AMR):
        raise TypeError("estimate_memory(layout=): expected a pops.mesh.layouts.AMR / Uniform; "
                        "got %r" % type(layout).__name__)
    max_levels = int(getattr(layout, "max_levels", 1) or 1)
    ratio = int(getattr(layout, "ratio", 2) or 2)
    if max_levels <= 1:
        return "amr", 0, ["AMR layout with a single level: no extra patch budget"]
    # Sum r^(2k) for k = 1 .. max_levels-1 (each refined level fully covering the domain).
    refine_factor = sum(ratio ** (2 * k) for k in range(1, max_levels))
    # Each refined cell carries the same per-cell footprint as the base (state + one elliptic field).
    per_cell_levels = state_field + n_elliptic * cell_field
    amr_bytes = refine_factor * per_cell_levels
    notes = [
        "AMR estimate is CONSERVATIVE: assumes EVERY level (1..%d) fully refines the whole domain "
        "at ratio %d (worst case); a real regrid tags a fraction of cells, so the true footprint is "
        "smaller. A tight AMR figure needs a bind (the regrid pattern is data-dependent)."
        % (max_levels - 1, ratio),
        "AMR refine factor (sum of r^(2k), k=1..%d) = %d base-grid equivalents"
        % (max_levels - 1, refine_factor),
    ]
    return "amr", amr_bytes, notes


__all__ = ["Arguments", "MemoryEstimate", "build_arguments", "build_memory_estimate"]

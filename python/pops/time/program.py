"""pops.time.Program -- the compiled time-program authoring class (builder-mode IR).

A ``Program`` BUILDS a typed SSA IR for one time step; Python never executes a numerical stage.
The C++ lowering (``emit_cpp_program``) is a thin delegator to
``pops.codegen.program_codegen`` (lazy import), so this package keeps a strictly acyclic
graph (it imports only ``pops.ir`` / ``pops.model`` and never ``pops.codegen`` / ``_pops`` at
module scope). The class is composed from focused authoring mixins.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from pops.time.program_authoring import _ProgramAuthoring
from pops.time.program_core import _ProgramCore
from pops.time.program_inspect import _ProgramInspect
from pops.time.program_local import _ProgramLocal
from pops.time.program_passes import _ProgramPasses
from pops.time.program_solve import _ProgramSolve
from pops.time.values import _Coeff, Value  # noqa: F401  (Value used by mixins via prog ref)


class Program(_ProgramCore, _ProgramLocal, _ProgramSolve, _ProgramAuthoring,
              _ProgramPasses, _ProgramInspect):
    """A compiled time program (builder mode). Holds the SSA value list and the committed
    blocks. The Python object only BUILDS the IR; it is never executed numerically during
    ``sim.step``. Authoring methods come from the mixins; C++ emission is delegated to
    ``pops.codegen.program_codegen`` via :meth:`emit_cpp_program`.
    """

    def __init__(self, name):
        self.name = name
        self._values = []
        self._next_id = 0
        self._commits = {}      # block -> State value
        self._recording = []    # stack of sub-block lists (a control-flow body); see _new / while_
        self._histories = {}    # name -> max declared lag (multistep histories; ADC-406a)
        # OPTIONAL dt bound (spec s18 / ADC-417): a recorded scalar sub-program (cfl -> Scalar) the
        # generated .so exports as pops_program_dt_bound; None = no bound (the native CFL is used).
        self._dt_bound = None        # (block, scalar_value) once set; the block is the scalar sub-block
        self.dt = _Coeff({1: 1.0})   # symbolic time step; participates in coefficient arithmetic
        # OPTIONAL bound operator registry (Spec 2, operator-first): set by bind_operators so P.call
        # can resolve and type-check operators at build time. None = legacy PDE-shortcut-only Program.
        self._registry = None
        # Per-emit scratch names of coupled_rate blocks, keyed by (coupled node id, block): the
        # coupled_rate kernel fills them and each coupled_rate_out projection aliases its block's
        # scratch (ADC-457). Populated during _emit_op; harmless to keep across emits (keys are unique
        # per node id).
        self._coupled_scratch = {}


    # --- C++ codegen (lowering to a problem.so source) lives in pops.codegen; the authoring
    # Program delegates via a LAZY import so pops.time stays free of any codegen/_pops edge. ---
    def emit_cpp_program(self, model=None):
        """Generate the C++ source of a problem.so implementing this Program (codegen).

        Thin authoring entry point: delegates to the free function
        :func:`pops.codegen.program_codegen.emit_cpp_program`, imported lazily so the
        ``pops.time`` package never imports ``pops.codegen`` / ``_pops`` at module scope. See
        the codegen function for the full lowering contract.
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg.emit_cpp_program(self, model=model)

    def _check_lowerable(self, model=None):
        """Raise if the IR uses a construct the codegen cannot lower (delegates to
        :func:`pops.codegen.program_codegen._check_lowerable`, lazy import).
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._check_lowerable(self, model)

    def _check_schedules_lowerable(self):
        """Raise if a node carries a schedule the codegen cannot lower (delegates to
        :func:`pops.codegen.program_codegen._check_schedules_lowerable`, lazy import).
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._check_schedules_lowerable(self)

    def _emit_body(self, model=None):
        """Lower the install-function body to ``(prelude, body)`` C++ (delegates to
        :func:`pops.codegen.program_codegen._emit_body`, lazy import). Exposed for the codegen
        tests that assert the body shape directly.
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._emit_body(self, model)


class CompiledTime:
    """Record of a compiled `Program`'s macro-step cadence (`substeps` / `stride`).

    A compiled Program OWNS the whole step body: it is installed via `sim.install_program` and driven
    by `sim.step(dt)`. Its cadence is applied to the System with `sim.set_program_cadence(substeps,
    stride)` (call it after `install_program`); a `CompiledTime` just records those values. The
    compiled program is NOT attached via `sim.add_equation(time=CompiledTime(...))` -- that path is
    rejected with an explicit error (the transport policy passed to `add_equation` is a native
    `pops.Explicit`/etc.; the compiled program is installed separately). `substeps` and
    `stride` are wired (ADC-411) as a SYSTEM-level orchestration AROUND the opaque program closure
    (`System.set_program_cadence`, mirroring the native per-block advance loop): `substeps=n` runs the
    program n times over `eff_dt/n`; `stride=M` runs the whole program once per M macro-steps with
    `eff_dt = M*dt` (GLOBAL hold-then-catch-up, the clock still ticks every macro-step).

    Two semantic limits to keep in mind (cf. system_stepper.hpp):
      - `substeps > 1` is bit-exact vs native `pops.Explicit(substeps=n)` ONLY for an UNCOUPLED /
        transport-only program: `program_step_(h)` re-runs the WHOLE program (its `solve_fields`
        included), whereas native substeps subdivides ONLY the transport (solve_fields runs once).
      - `stride` here is GLOBAL (a compiled program is one whole-system closure), so it equals native
        per-block stride only for a single-block system (or all blocks sharing the stride).

    A non-default `cfl` is still deferred (the Program receives a bare `dt`; pass an explicit `dt` to
    `sim.step(dt)`) -- it fails loud rather than being silently ignored."""

    def __init__(self, substeps=1, stride=1, cfl="default"):
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("CompiledTime: substeps must be a positive int (got %r)" % (substeps,))
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("CompiledTime: stride must be a positive int (got %r)" % (stride,))
        if cfl != "default":
            raise NotImplementedError(
                "CompiledTime: cfl != 'default' is deferred (ADC-401 Phase 2c); pass an explicit dt "
                "to sim.step(dt)")
        self.substeps = substeps
        self.stride = stride
        self.cfl = cfl
        self.kind = "compiled"

    def __repr__(self):
        return "CompiledTime(substeps=%d, stride=%d, cfl=%r)" % (self.substeps, self.stride, self.cfl)

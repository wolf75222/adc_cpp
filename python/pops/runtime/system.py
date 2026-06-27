"""System : the runtime coupler (Spec-4 PR-F composed class).

``System`` composes blocks, shares a Poisson and advances the whole. Its ~1300 lines of methods
are split into cohesive mixins (``_system_install`` / ``_system_unified_install`` /
``_system_aux_state`` / ``_system_diagnostics`` / ``_system_io``) to satisfy the per-file
<=500-line cap ; this module composes them and keeps the constructor + the delegation glue.
``AmrSystem`` lives in :mod:`pops.runtime.amr_system` and is re-exported here for the
``from pops.runtime.system import System, AmrSystem`` import in the slim ``pops`` hub.
"""

from pops._bootstrap import SystemConfig, _System
from pops.runtime import threading as _threading
from pops.runtime.amr_system import AmrSystem  # noqa: F401  (re-exported via this module)
from pops.runtime._system_aux_state import _SystemAuxState
from pops.runtime._system_diagnostics import _SystemDiagnostics
from pops.runtime._system_install import _SystemInstall
from pops.runtime._system_io import _SystemIO
from pops.runtime._system_unified_install import _SystemUnifiedInstall
from pops.runtime.profile import PerformanceSummary, Profile


class _ProfileSession:
    """The typed profiling context manager System.profile() returns (Spec 5 sec.12.5).

    ``__enter__`` resets + enables the native profiler; ``__exit__`` snapshots the report into a
    :class:`PerformanceSummary` and disables the profiler. ``summary()`` works inside OR after the
    ``with`` block (it re-reads the live report while open, returns the closing snapshot after).
    The off-by-default contract holds: nothing here enables until the block is entered.
    """

    def __init__(self, system, profile):
        self._system = system
        self._profile = profile
        self._summary = None

    def __enter__(self):
        self._system.reset_profiling()
        self._system.enable_profiling()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._summary = PerformanceSummary(self._system.profile_report(), self._profile)
        self._system.disable_profiling()
        return False

    def summary(self):
        """Return a :class:`PerformanceSummary` of the run.

        Inside the ``with`` block it reads the live native report; after the block it returns the
        snapshot taken on ``__exit__``.
        """
        if self._summary is not None:
            return self._summary
        return PerformanceSummary(self._system.profile_report(), self._profile)


class System(_SystemInstall, _SystemUnifiedInstall, _SystemAuxState,
             _SystemDiagnostics, _SystemIO):
    """The system/coupler: composes blocks, shares a Poisson, advances the whole.

    add_block takes a composed model (pops.Model(...)) + Spatial / Explicit / IMEX objects.
    Everything else (set_poisson, set_density, step, step_cfl, step_adaptive, diagnostics,
    primitives eval_rhs/get_state/set_state) is forwarded to the compiled facade.

    GEOMETRY: the choice lives in a MESH object passed as mesh= (pops.CartesianMesh / pops.PolarMesh),
    NOT in the scheme (pops.FiniteVolume stays reconstruction + Riemann + variables). Default (mesh=None
    or pops.CartesianMesh) = square domain, bit-identical to the history. pops.PolarMesh (global ring)
    is WIRED in System.step (Phase 2b): polar ExB transport + polar Poisson + aux in local basis
    (e_r, e_theta). Limits: scalar ExB transport, single-rank, no cart<->polar coupling."""

    def __init__(self, config=None, mesh=None, **cfg_kw):
        if config is None:
            config = SystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        # The mesh (if provided) carries the geometry CHOICE and overrides the corresponding fields
        # of the config. Applied AFTER cfg_kw: mesh= takes precedence over the n=/L= passed as keywords.
        if mesh is not None:
            if not hasattr(mesh, "_apply"):
                raise TypeError("System: mesh must be an pops.CartesianMesh / pops.PolarMesh (got %r)"
                                % type(mesh).__name__)
            mesh._apply(config)
        # Mark the Kokkos init as imminent: _System(config) allocates Fabs -> Kokkos initializes
        # (lazy) here. After this point, pops.set_threads has no further effect (warned by set_threads).
        _threading._first_system_built = True
        self._s = _System(config)  # geometry == 'polar' builds a global ring (Phase 2b, cf. PolarMesh)
        # Table of NAMED aux fields per block (ADC-70 phase 1): block -> {name: canonical component}.
        # Filled by add_equation from CompiledModel.aux_extra_names (the component of the k-th name =
        # dsl.AUX_NAMED_BASE + k). The FACADE holds the names: the C++ only manipulates component
        # indices (set_aux_field_component / aux_field_component). Empty for a block without a
        # named aux field. cf. set_aux_field / aux_field.
        self._aux_field_index = {}

    def run(self, t_end, cfl=0.4, max_steps=1_000_000):
        """Advance up to t_end by CFL steps (sugar: `while time() < t_end: step_cfl(cfl)`).

        @p cfl: Courant number passed to step_cfl. @p max_steps: guard (avoids an infinite
        loop if dt -> 0). Returns the number of steps taken. cf. DSL_MODEL_DESIGN.md section 6."""
        steps = 0
        while self.time() < t_end and steps < max_steps:
            self.step_cfl(cfl)
            steps += 1
        return steps

    def profile(self, profile=None):
        """Typed profiling context manager (Spec 5 sec.12.5, criteria 41-44).

        Usage::

            with sim.profile(pops.Profile.Basic()) as prof:
                sim.run(0.1)
            print(prof.summary())

        @p profile is a :class:`pops.Profile` level (``Profile.Basic()`` / ``Profile.Advanced()``);
        with no argument the level comes from ``POPS_PROFILE`` (unset / ``off`` -> Basic()). The
        manager enables the native profiler on entry and disables it on exit, so a plain run (no
        ``with sim.profile()``) leaves profiling off -- the off-by-default contract. ``prof.summary()``
        returns a :class:`pops.PerformanceSummary`.
        """
        if profile is None:
            profile = Profile.from_env(default=Profile.Basic())
        elif not isinstance(profile, Profile):
            raise TypeError(
                "System.profile: expected a pops.Profile (Profile.Basic()/Advanced()), got %r"
                % type(profile).__name__)
        return _ProfileSession(self, profile)

    def block_names(self):
        """Names of the added blocks, in order (useful for a Python integrator).

        Delegates to the C++ block registry (single source), so it includes the blocks loaded via
        add_dynamic_block (.so JIT) and add_compiled_block (.so AOT), not only add_block.
        """
        return list(self._s.block_names())

    def __str__(self):
        """Short, array-free summary: the installed block names (Spec 5 sec.12.1).

        Deliberately field-data-free -- it prints the block registry, never a Fab dump.
        """
        try:
            blocks = self.block_names()
        except Exception:  # pragma: no cover - defensive: _System not fully wired
            blocks = []
        return "System(blocks=%s)" % (blocks,)

    @property
    def amr(self):
        """The AMR runtime inspection surface does not apply to a uniform ``System``.

        ``System`` is single-level: it carries no AMR hierarchy, so ``sim.amr`` (the live
        patch / regrid / ghost / reflux / checkpoint reports of Spec 5 sec.8.12) is an
        ``AmrSystem``-only handle. Build an ``pops.AmrSystem`` for a refined run, or use the
        STATIC authoring report ``pops.inspect_amr(layout)`` for a layout descriptor. Accessing it
        raises a clear ``AttributeError`` (sourced in ``__getattr__`` so the message is single).
        """
        # The AttributeError routes through __getattr__('amr'), which raises the clear message.
        raise AttributeError("amr")

    @staticmethod
    def abi_key():
        """Module ABI key (compiler, C++ standard, signature of the pops headers). Compared to
        that of a native loader by add_native_block. Also exposed as a class attribute (the
        __getattr__ delegate only covers instances), so pops.System.abi_key() works."""
        return _System.abi_key()

    def __getattr__(self, attr):
        # 'amr' is an AmrSystem-only inspection handle; the System @property raises AttributeError,
        # which routes here -- intercept it so the clear message surfaces instead of the raw _pops
        # "object has no attribute 'amr'" delegation (Spec 5 sec.8.12).
        if attr == "amr":
            raise AttributeError(
                "System has no 'amr' inspection handle: System is a uniform single-level runtime "
                "with no AMR hierarchy. Use pops.AmrSystem (its sim.amr returns an AmrRuntimeView), "
                "or pops.inspect_amr(layout) for the static authoring report.")
        return getattr(self._s, attr)

"""The live AMR runtime view (Spec 5 sec.8.12 ``sim.amr``).

:class:`AmrRuntimeView` is the runtime-bound handle returned by ``AmrSystem.amr``. It is the only
part of :mod:`pops.runtime.amr` tied to a live system: it reads the already-built box accessors
(``patch_rectangles`` / ``patch_boxes`` / ``coarse_local_boxes`` / ``coarse_total_boxes``) and the
static config the ``AmrSystem`` retained, then packages them into the inert report value classes.

It is INERT: it builds nothing, allocates nothing, steps no clock. The native box accessors trigger
only the same lazy box build that ``n_patches()`` does; before the hierarchy is built (no block
added / no first step) they raise, and the view reports an unbuilt hierarchy instead of forcing a
build. A measure the native build cannot answer (per-level ghost depth, composite Poisson) is
reported as honestly unavailable.
"""

from pops.runtime.amr._reports import (
    PatchReport,
    RegridReport,
    GhostReport,
    RefluxReport,
    CheckpointReport,
    HierarchySnapshot,
)


# The build raises this exact message before the first block is added / the hierarchy is built; we
# treat it as "not built yet" rather than letting it propagate (the report is inert, never a build).
_NOT_BUILT = "call add_block first"


class AmrRuntimeView:
    """Inert, live-bound inspection handle for an :class:`AmrSystem` (Spec 5 sec.8.12).

    Construct via ``sim.amr`` (do not instantiate directly). Every method returns one of the inert
    report value classes (:class:`PatchReport`, :class:`RegridReport`, :class:`GhostReport`,
    :class:`RefluxReport`, :class:`CheckpointReport`, :class:`HierarchySnapshot`).
    """

    def __init__(self, amr_system):
        # Bound to the AmrSystem facade (not the raw _AmrSystem): we read its box helpers, its
        # block registry and the config snapshot it retained.
        self._sim = amr_system

    # --- live box readers (guard the pre-build RuntimeError) -----------------
    def _is_built(self):
        """True once the native hierarchy has been built (a block added; boxes available)."""
        try:
            self._sim._s.n_levels()
            return True
        except RuntimeError as exc:
            if _NOT_BUILT in str(exc):
                return False
            raise

    def _coarse_boxes(self):
        """(local, total) coarse base box counts, or (None, None) before the build."""
        try:
            return (int(self._sim.coarse_local_boxes()), int(self._sim.coarse_total_boxes()))
        except RuntimeError as exc:
            if _NOT_BUILT in str(exc):
                return (None, None)
            raise

    def _block_names(self):
        try:
            return list(self._sim._s.block_names())
        except RuntimeError:
            return []

    def _per_level(self):
        """Per-level patch census from patch_boxes() + patch_rectangles(), level 0 = base box."""
        s = self._sim._s
        n_levels = int(s.n_levels())
        base_n = int(s.nx())
        # Level 0: the coarse base covers the whole domain; report it as one covering box.
        levels = {
            0: {"level": 0, "n_patches": 1, "cells": base_n * base_n, "boxes": [], "rectangles": []}
        }
        for lvl in range(1, n_levels):
            levels[lvl] = {"level": lvl, "n_patches": 0, "cells": 0, "boxes": [], "rectangles": []}
        rects = self._sim.patch_rectangles()
        # patch_boxes() and patch_rectangles() are parallel (one rectangle per box); strict=True
        # asserts that invariant rather than silently truncating to the shorter.
        for (level, ilo, jlo, ihi, jhi), rect in zip(s.patch_boxes(), rects, strict=True):
            level = int(level)
            entry = levels.setdefault(
                level, {"level": level, "n_patches": 0, "cells": 0, "boxes": [], "rectangles": []})
            entry["n_patches"] += 1
            entry["cells"] += (int(ihi) - int(ilo) + 1) * (int(jhi) - int(jlo) + 1)
            entry["boxes"].append((int(ilo), int(jlo), int(ihi), int(jhi)))
            entry["rectangles"].append(tuple(float(v) for v in rect))
        return [levels[k] for k in sorted(levels)]

    # --- the sec.8.12 reports ------------------------------------------------
    def patch_table(self):
        """Return a :class:`PatchReport` of the live patches + coarse box distribution."""
        coarse_local, coarse_total = self._coarse_boxes()
        if not self._is_built():
            return PatchReport(
                built=False, n_levels=None, base_n=None, domain_l=self._sim._L, per_level=[],
                coarse_local_boxes=coarse_local, coarse_total_boxes=coarse_total)
        return PatchReport(
            built=True, n_levels=int(self._sim._s.n_levels()), base_n=int(self._sim._s.nx()),
            domain_l=self._sim._L, per_level=self._per_level(),
            coarse_local_boxes=coarse_local, coarse_total_boxes=coarse_total)

    def explain_regrid(self):
        """Return a :class:`RegridReport` of the regrid cadence + union-tag criteria."""
        regrid_every = int(self._sim._regrid_every)
        frozen = regrid_every == 0
        # The union-of-tags criteria shape, as the multi-block route documents it (the exact
        # threshold lives on the native model; we name the criteria, not a fabricated number).
        criteria = [
            "per-block variable threshold (set_refinement(threshold, variable=/role=); "
            "default component 0)",
            "grad phi (set_phi_refinement(grad_threshold); multi-block only, disabled when "
            "grad_threshold <= 0)",
        ]
        notes = []
        if frozen:
            notes.append("regrid_every == 0: the hierarchy is built once and frozen "
                         "(bit-identical, no dynamic regrid).")
        else:
            notes.append("a cell is tagged when ANY criterion fires (cell-by-cell OR).")
        return RegridReport(regrid_every=regrid_every, frozen=frozen, criteria=criteria, notes=notes)

    def explain_ghosts(self):
        """Return a :class:`GhostReport`; the per-level ghost depth is honestly unavailable."""
        return GhostReport(
            per_level_depth=None,
            requirement_note=("the reconstruction stencil sets the ghost depth "
                              "(minmod / vanleer -> 1, weno5 -> 3); the coarse-fine fine ghosts "
                              "are re-derived per path on the AMR transport."),
            notes=["per-level ghost depth is not exposed by this native build."])

    def explain_reflux(self):
        """Return a :class:`RefluxReport` of the route reflux requirement."""
        return RefluxReport(
            enabled=True,
            per_stage=None,
            notes=["coarse-fine flux refluxing is a native AMR route requirement (the AMR layout "
                   "descriptor reports reflux=True); it runs on the single-block coupler path."])

    def explain_checkpoint(self):
        """Return a :class:`CheckpointReport` of the live system's restartability (sec.8.11)."""
        constraints = ["single block", "single rank", "frozen hierarchy (regrid_every == 0)"]
        violations = []
        # Block count: a multi-block AmrRuntime engine is not checkpointable in v1.
        try:
            n_blocks = int(self._sim._s.n_blocks())
        except RuntimeError:
            n_blocks = len(self._block_names())
        if n_blocks > 1:
            violations.append("multi-block (n_blocks=%d): the v1 checkpoint is single-block "
                              "(AmrRuntime engine carries no per-block restart yet)." % n_blocks)
        if int(self._sim._regrid_every) != 0:
            violations.append("regrid_every=%d != 0: a bit-identical resume requires a frozen "
                              "hierarchy (the post-restart regrid would re-diverge)."
                              % int(self._sim._regrid_every))
        notes = ["MPI np>1 is rejected at checkpoint time (single-rank v1); rank count is a "
                 "runtime property, not read here.",
                 "composite multi-level Poisson restart: unavailable (single-level System has "
                 "bit-identical checkpoint/restart including under MPI)."]
        return CheckpointReport(
            restartable=not violations, constraints=constraints, violations=violations, notes=notes)

    def hierarchy_snapshot(self):
        """Return a :class:`HierarchySnapshot`: config envelope (reusing :func:`pops.inspect_amr`)
        composed with the live patch table."""
        # Config envelope from the inert authoring report (native max_levels / ratio / limitations).
        from pops import inspect_amr  # runtime layer may import the flat root facade.

        envelope = inspect_amr().to_dict()
        regrid_every = int(self._sim._regrid_every)
        patch_table = self.patch_table()
        return HierarchySnapshot(
            blocks=self._block_names(),
            max_levels=envelope["max_levels"],
            ratio=envelope["ratio"],
            regrid_every=regrid_every,
            frozen=regrid_every == 0,
            patch_table=patch_table,
            limitations=envelope["limitations"],
            config_available=envelope["available"])

    def __repr__(self):
        return "AmrRuntimeView(blocks=%r)" % (self._block_names(),)

    def __str__(self):
        """Short, array-free handle summary (Spec 5 sec.12.1)."""
        built = "built" if self._is_built() else "not built"
        return "AmrRuntimeView(blocks=%s, hierarchy %s)" % (self._block_names(), built)

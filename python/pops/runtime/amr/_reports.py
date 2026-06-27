"""Inert report value classes for the AMR runtime inspection surface (Spec 5 sec.8.4).

Each class is a plain record: it holds numbers / strings already read from the live runtime and
the descriptor metadata, exposes a deterministic ``to_dict()`` and a short, array-free ``__str__``
(Spec 5 sec.12.1 print rules: no Fab dumps, no per-cell arrays), and imports nothing -- no
``_pops``, no numpy, no runtime. The :class:`AmrRuntimeView` (bound to the live system) constructs
these; they carry no reference back to the system.

A field whose value the current native build cannot answer is set to ``None`` and rendered as an
explicit "unavailable (<reason>)" line rather than a fabricated zero (sec.8 honesty rule).
"""


def _fmt_unavailable(reason):
    """Render an honestly-deferred measure as a short, stable string."""
    return "unavailable (%s)" % reason


class PatchReport:
    """The live patch table of an AMR hierarchy (Spec 5 sec.8.12 ``patch_table()``).

    A per-level census of the refined patches that ACTUALLY exist on the built hierarchy right now
    (read from ``patch_rectangles`` + ``patch_boxes``), plus the coarse (base) box distribution
    (``coarse_local_boxes`` / ``coarse_total_boxes``). It dumps no field data: each patch is its
    integer box corners + physical rectangle, summarized per level. ``built`` is False when the
    hierarchy has not been built yet (no block added / no first step), in which case there are no
    patches to report.
    """

    def __init__(self, *, built, n_levels, base_n, domain_l, per_level, coarse_local_boxes,
                 coarse_total_boxes):
        self.built = bool(built)
        self.n_levels = n_levels
        self.base_n = base_n
        self.domain_l = domain_l
        # per_level: ordered list of dicts {level, n_patches, cells, boxes=[(ilo,jlo,ihi,jhi)],
        # rectangles=[(x0,y0,w,h)]}; level 0 (the coarse base) is reported as a single covering box.
        self.per_level = list(per_level)
        self.coarse_local_boxes = coarse_local_boxes
        self.coarse_total_boxes = coarse_total_boxes

    @property
    def n_patches(self):
        """Total fine patches (every refined level combined)."""
        return sum(lvl["n_patches"] for lvl in self.per_level if lvl["level"] > 0)

    @property
    def coarse_is_distributed(self):
        """True when this rank owns a strict subset of the coarse base (MPI strong-scaling)."""
        if self.coarse_local_boxes is None or self.coarse_total_boxes is None:
            return None
        return self.coarse_local_boxes < self.coarse_total_boxes

    def to_dict(self):
        return {
            "built": self.built,
            "n_levels": self.n_levels,
            "base_n": self.base_n,
            "domain_l": self.domain_l,
            "n_patches": self.n_patches,
            "coarse_local_boxes": self.coarse_local_boxes,
            "coarse_total_boxes": self.coarse_total_boxes,
            "coarse_is_distributed": self.coarse_is_distributed,
            "per_level": [dict(lvl) for lvl in self.per_level],
        }

    def __repr__(self):
        return "PatchReport(built=%r, n_levels=%r, n_patches=%r)" % (
            self.built, self.n_levels, self.n_patches)

    def __str__(self):
        if not self.built:
            return ("AMR patch table: hierarchy not built yet (add a block and take a step, or "
                    "set the initial density, to build the levels).")
        lines = ["AMR patch table (built hierarchy):"]
        lines.append("  base: n=%s on [0, %s]^2, levels=%s"
                     % (self.base_n, self.domain_l, self.n_levels))
        for lvl in self.per_level:
            if lvl["level"] == 0:
                lines.append("  level 0 (base): %d box(es), %d cells"
                             % (lvl["n_patches"], lvl["cells"]))
            else:
                lines.append("  level %d: %d patch(es), %d cells"
                             % (lvl["level"], lvl["n_patches"], lvl["cells"]))
        coarse_l, coarse_t = self.coarse_local_boxes, self.coarse_total_boxes
        if coarse_l is not None and coarse_t is not None:
            tag = "distributed" if self.coarse_is_distributed else "replicated/single-box"
            lines.append("  coarse base boxes: local=%d total=%d (%s)"
                         % (coarse_l, coarse_t, tag))
        return "\n".join(lines)


class RegridReport:
    """The regrid policy in force on the live hierarchy (Spec 5 sec.8.12 ``explain_regrid()``).

    Reports the regrid cadence (``regrid_every``, read from the AmrSystem), whether the hierarchy
    is FROZEN (cadence 0 -> built once, bit-identical) or DYNAMIC, and the union-tag refinement
    criteria as the route documents them (per-block variable threshold OR ``grad phi``). The exact
    per-run threshold values live on the native model; this report names the criteria SHAPE, not a
    fabricated threshold it cannot read back.
    """

    def __init__(self, *, regrid_every, frozen, criteria, notes):
        self.regrid_every = regrid_every
        self.frozen = bool(frozen)
        self.criteria = list(criteria)
        self.notes = list(notes)

    def to_dict(self):
        return {
            "regrid_every": self.regrid_every,
            "frozen": self.frozen,
            "criteria": list(self.criteria),
            "notes": list(self.notes),
        }

    def __repr__(self):
        return "RegridReport(regrid_every=%r, frozen=%r)" % (self.regrid_every, self.frozen)

    def __str__(self):
        mode = "frozen" if self.frozen else "dynamic"
        lines = ["AMR regrid policy: %s (regrid_every=%s)" % (mode, self.regrid_every)]
        if self.criteria:
            lines.append("  union-of-tags criteria:")
            for c in self.criteria:
                lines.append("    - %s" % c)
        for note in self.notes:
            lines.append("  note: %s" % note)
        return "\n".join(lines)


class GhostReport:
    """The ghost-cell envelope of the hierarchy (Spec 5 sec.8.12 ``explain_ghosts()``).

    The reconstruction stencil sets the ghost depth (minmod / van Leer -> 1, weno5 -> 3); the
    coarse-fine ghosts are re-derived per path on the AMR transport. The PER-LEVEL ghost depth is
    not exposed by the current native build, so ``per_level_depth`` is honestly None and the report
    states the requirement shape (stencil -> ghost depth) instead of a fabricated number.
    """

    def __init__(self, *, per_level_depth, requirement_note, notes):
        self.per_level_depth = per_level_depth
        self.requirement_note = requirement_note
        self.notes = list(notes)

    def to_dict(self):
        return {
            "per_level_depth": self.per_level_depth,
            "requirement_note": self.requirement_note,
            "notes": list(self.notes),
        }

    def __repr__(self):
        return "GhostReport(per_level_depth=%r)" % (self.per_level_depth,)

    def __str__(self):
        lines = ["AMR ghost cells:"]
        if self.per_level_depth is None:
            lines.append("  per-level ghost depth: %s"
                         % _fmt_unavailable("not queryable from this build"))
        else:
            lines.append("  per-level ghost depth: %s" % (self.per_level_depth,))
        lines.append("  requirement: %s" % self.requirement_note)
        for note in self.notes:
            lines.append("  note: %s" % note)
        return "\n".join(lines)


class RefluxReport:
    """The reflux policy of the route (Spec 5 sec.8.12 ``explain_reflux()``).

    On the native AMR route, coarse-fine flux refluxing is a route REQUIREMENT (the descriptor
    layout reports ``reflux=True``); it runs per stage on the single-block coupler path. The
    per-stage timing detail is a route property, not a runtime-readable counter, so it is reported
    as the documented requirement rather than a measured count.
    """

    def __init__(self, *, enabled, per_stage, notes):
        self.enabled = bool(enabled)
        self.per_stage = per_stage
        self.notes = list(notes)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "per_stage": self.per_stage,
            "notes": list(self.notes),
        }

    def __repr__(self):
        return "RefluxReport(enabled=%r, per_stage=%r)" % (self.enabled, self.per_stage)

    def __str__(self):
        lines = ["AMR reflux: %s" % ("enabled" if self.enabled else "disabled")]
        if self.per_stage is None:
            lines.append("  per-stage timing: %s"
                         % _fmt_unavailable("route property, not a runtime counter"))
        else:
            lines.append("  per-stage: %s" % (self.per_stage,))
        for note in self.notes:
            lines.append("  note: %s" % note)
        return "\n".join(lines)


class CheckpointReport:
    """The checkpoint / restart policy of the live system (Spec 5 sec.8.12 ``explain_checkpoint()``).

    Surfaces the native AMR v1 checkpoint envelope (sec.8.11): bit-identical resume is wired for a
    SINGLE block, a SINGLE rank, and a FROZEN hierarchy (regrid_every == 0). For the live system it
    reports whether the current configuration is RESTARTABLE under that envelope and, if not, which
    constraint(s) it violates -- the same constraints ``AmrSystem.checkpoint`` rejects at call time,
    surfaced here as an inert explanation rather than a raised error.
    """

    def __init__(self, *, restartable, constraints, violations, notes):
        self.restartable = bool(restartable)
        self.constraints = list(constraints)
        self.violations = list(violations)
        self.notes = list(notes)

    def to_dict(self):
        return {
            "restartable": self.restartable,
            "constraints": list(self.constraints),
            "violations": list(self.violations),
            "notes": list(self.notes),
        }

    def __repr__(self):
        return "CheckpointReport(restartable=%r)" % (self.restartable,)

    def __str__(self):
        head = "restartable" if self.restartable else "NOT restartable"
        lines = ["AMR checkpoint policy: %s (bit-identical v1 envelope)" % head]
        lines.append("  envelope: single block, single rank, frozen hierarchy "
                     "(regrid_every == 0)")
        if self.violations:
            lines.append("  this system violates:")
            for v in self.violations:
                lines.append("    - %s" % v)
        for note in self.notes:
            lines.append("  note: %s" % note)
        return "\n".join(lines)


class HierarchySnapshot:
    """A single inert snapshot of the live AMR hierarchy (Spec 5 sec.8.4 / sec.8.12).

    Composes the config envelope (levels / ratio / regrid cadence / patch layout / native
    limitations, reusing :func:`pops.inspect_amr` for the config-level metadata) with the LIVE
    patch census (:class:`PatchReport`). It is a value object: a deterministic, array-free picture
    of the hierarchy at the moment it was taken, suitable to ``print()`` or diff between snapshots.
    """

    def __init__(self, *, blocks, max_levels, ratio, regrid_every, frozen, patch_table,
                 limitations, config_available):
        self.blocks = list(blocks)
        self.max_levels = max_levels
        self.ratio = ratio
        self.regrid_every = regrid_every
        self.frozen = bool(frozen)
        self.patch_table = patch_table
        self.limitations = list(limitations)
        self.config_available = config_available

    def to_dict(self):
        return {
            "blocks": list(self.blocks),
            "max_levels": self.max_levels,
            "ratio": self.ratio,
            "regrid_every": self.regrid_every,
            "frozen": self.frozen,
            "config_available": self.config_available,
            "patch_table": self.patch_table.to_dict(),
            "limitations": list(self.limitations),
        }

    def __repr__(self):
        return ("HierarchySnapshot(blocks=%r, max_levels=%r, n_patches=%r)"
                % (self.blocks, self.max_levels, self.patch_table.n_patches))

    def __str__(self):
        mode = "frozen" if self.frozen else "dynamic"
        lines = ["AMR hierarchy snapshot:"]
        lines.append("  blocks: %s" % (self.blocks,))
        lines.append("  levels: max_levels=%s ratio=%s (config available: %s)"
                     % (self.max_levels, self.ratio, self.config_available))
        lines.append("  regrid: %s (regrid_every=%s)" % (mode, self.regrid_every))
        if self.patch_table.built:
            lines.append("  live patches: %d on %d level(s)"
                         % (self.patch_table.n_patches, self.patch_table.n_levels))
        else:
            lines.append("  live patches: hierarchy not built yet")
        if self.limitations:
            lines.append("  limitations:")
            for note in self.limitations:
                lines.append("    - %s" % note)
        return "\n".join(lines)

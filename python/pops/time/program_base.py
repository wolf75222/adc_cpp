"""pops.time Program constant tables (shared mixin base).

``_ProgramConstants`` carries every class-level op allow-list / threshold table the Program
authoring and the codegen lowering key on. It is a pure-data mixin (no methods, no imports):
the assembled :class:`pops.time.Program` and the emission free functions in
``pops.codegen.program_codegen`` both read these through the instance (``program._X``).
"""


class _ProgramConstants:
    """Class-level constant tables shared across the Program authoring mixins."""

    _RESIDUAL_LOCAL_OPS = frozenset({"state", "source", "apply", "linear_combine"})

    _KRYLOV_METHODS = frozenset({"cg", "bicgstab", "richardson", "gmres"})
    _GMRES_RESTART_DEFAULT = 30  # GMRES(m) restart length when the caller does not override it

    _OPERATOR_KINDS = frozenset({"scalar", "vector", "state"})

    _CELL_CMPS = {">": "cell_gt", ">=": "cell_ge", "<": "cell_lt", "<=": "cell_le"}

    # --- IR optimization passes (Spec 3 s28, ADC-465) --------------------------------------------
    # SAFE-BY-DEFAULT ALLOW-LIST. A flat node is removable ONLY if its op is enumerated here; EVERY
    # other op (known side-effecting, buffer-writing, sub-block-owning, OR new/unknown) is treated as
    # live even when its result looks unconsumed. This is the inverse of a blacklist: a buffer-writing
    # op whose ``_emit_op`` lowering ALIASES a caller-allocated input buffer (``var[v.id] =
    # var[out_in.id]``) is side-effecting on that buffer even though it has no dataflow output edge --
    # e.g. ``schur_rhs`` fills the ``rhs`` scratch that ``solve_linear`` then reads by BUFFER IDENTITY,
    # not via an input edge. A blacklist silently drops such an op (corrupting the codegen while
    # ``validate()`` stays True); a whitelist cannot, because the op is not listed.
    #
    # An op qualifies for this list ONLY if its ``_emit_op`` branch was verified to (a) ALLOCATE A
    # FRESH result scratch (``ctx.rhs_scratch_like`` / ``ctx.scratch_state_like`` / ``ctx.alloc_*`` /
    # a fresh ``s%d`` scalar local), NOT alias an input, AND (b) have no other observable side effect.
    #   rhs / source / apply  -> ``r%d = ctx.rhs_scratch_like(state)`` then a pure compute fill
    #   linear_combine        -> ``u%d = ctx.scratch_state_like(base)`` (the NON-commit branch; a
    #                            committed linear_combine is a commit root, never a dead-node candidate)
    #   linear_source         -> a pure operator DECLARATION node (no inputs, no emitted statement)
    #   solve_local_linear    -> ``u%d = ctx.scratch_state_like(base)`` (fresh; per-cell dense solve)
    #   cell_compare / where  -> a fresh ``ctx.alloc_scalar_field`` / ``ctx.scratch_state_like`` mask
    #   reduce / scalar_op    -> a fresh ``s%d`` scalar local (a collective reduce is recomputed if
    #                            re-added later; dropping an UNCONSUMED one removes only dead arithmetic)
    #   compare               -> an inline boolean expression, no statement of its own
    # Deliberately EXCLUDED (kept live): the buffer-writers schur_rhs / schur_explicit_flux / laplacian
    # / gradient / divergence / apply_laplacian_coeff / schur_coeffs / schur_reconstruct / schur_energy
    # (alias an input buffer); the side-effecting solve_fields[_from_blocks] / project / fill_boundary /
    # store_history / record_scalar; solve_linear (reads its rhs by buffer identity); scalar_field /
    # state / history (scratch/state bindings other ops fill or alias); and the sub-block ops below.
    _REMOVABLE_OPS = frozenset({
        "rhs", "source", "apply", "linear_combine", "linear_source", "solve_local_linear",
        "cell_compare", "where", "reduce", "scalar_op", "compare",
    })

    # Ops that own a recorded sub-block (a while / if / range body, a matrix-free apply, a Newton
    # residual). v1 does NOT descend into sub-blocks, so these are treated as always-live roots and
    # every value they (or their sub-blocks) read is conservatively kept. They are simply absent from
    # the allow-list above (hence live); listed here only to drive the sub-block reference walk.
    _SUBBLOCK_OPS = frozenset({
        "while", "if", "range", "matrix_free_operator", "solve_local_nonlinear",
    })

    # Ops PROVEN PURE for common-subexpression elimination (Spec 3 s28, ADC-465): each allocates a
    # FRESH result scratch from its inputs ALONE, reads nothing through a buffer-identity side channel,
    # and has no side effect (no aux/Poisson/history/diagnostic write, no in-place mutation). Two such
    # nodes with the same op + vtype + block + inputs + attrs compute the SAME value, so the second can
    # alias the first with no change to the result. This is a STRICT subset of ``_REMOVABLE_OPS``:
    #   - ``reduce`` is EXCLUDED (a collective reduction is a global communication; conservatively never
    #     deduplicated even though two identical reduces give the same scalar -- CSE soundness, not perf,
    #     drives the list, and a duplicated reduce is dead-node territory, not CSE);
    #   - ``solve_local_linear`` is EXCLUDED (it reads its rhs State by buffer; a later op may overwrite
    #     that rhs buffer in place between two solves, so two same-input solves are NOT provably equal);
    #   - ``where`` / ``cell_compare`` ARE pure (fresh scratch, component-wise from inputs only).
    #   - ``rhs`` / ``source`` / ``apply`` are EXCLUDED: their per-cell kernels read the SHARED System
    #     aux (``ctx.aux()``) BY BUFFER IDENTITY, not through a dataflow input edge (see
    #     ``_emit_source_kernel`` / ``_emit_apply_kernel`` / ``_cell_locals`` -> ``auxA(i,j,...)``). A
    #     ``solve_fields`` mutates that aux in place, so two same-(state,fields)-input source/apply/rhs
    #     nodes straddling a field solve are NOT equal (the second reads the freshly solved field). CSE
    #     keys only on dataflow inputs, so collapsing them would silently read the stale pre-solve aux.
    # Every op OUTSIDE this set is treated as non-CSE-able, so a buffer-writer / side-effecting / aux-
    # reading / unknown op is never collapsed (safe-by-default, mirroring the dead-node allow-list).
    _PURE_OPS = frozenset({
        "linear_combine", "linear_source", "cell_compare", "where", "scalar_op", "compare",
    })

    # Ops that WRITE a block state or otherwise change what a subsequent ``solve_fields`` over the same
    # state would see (the Poisson RHS reads every block's live state + the shared aux). A redundant
    # ``solve_fields`` may be eliminated ONLY when none of these appears between the two solves over the
    # same state input -- conservatively, ANY commit, in-place state mutation (project), boundary fill,
    # history store, or a second field solve into the shared aux counts as a state/aux barrier.
    _STATE_BARRIER_OPS = frozenset({
        "project", "fill_boundary", "store_history",
        "solve_fields", "solve_fields_from_blocks",
    })

    _OPTIMIZE_PASSES = (
        ("eliminate_dead_nodes", "dead-node elimination"),
        ("eliminate_common_subexpressions", "common-subexpression elimination"),
        ("eliminate_redundant_field_solves", "redundant field-solve elimination"),
    )

    _SCRATCH_OPS = frozenset({
        "rhs", "source", "apply", "linear_combine", "linear_source", "solve_local_linear",
        "solve_local_nonlinear", "cell_compare", "where", "coupled_rate",
    })

    _PERCELL_KERNEL_OPS = frozenset({
        "rhs", "source", "apply", "linear_combine", "linear_source", "solve_local_linear",
        "solve_local_nonlinear", "cell_compare", "where", "coupled_rate", "project", "fill_boundary",
    })
    _HEAVY_KERNEL_OPS = frozenset({
        "solve_fields", "solve_fields_from_blocks", "solve_linear",
    })

    # GPU heuristic thresholds (Spec 3 s28 detectors, ADC-465). A warning report, never a hard error:
    # a small step legitimately trips none; a pathological IR (a long chain of tiny per-cell kernels, a
    # buffer explosion, or heavy field traffic) trips one and the report names it. The numbers are
    # launch-overhead / occupancy rules of thumb, not a measured limit -- the MEASURED GPU kernel count
    # is a ROMEO profile (this is the host-side static prediction).
    _GPU_MAX_SMALL_KERNELS = 12
    _GPU_MAX_SCRATCHES = 16
    _GPU_MAX_TRAFFIC_FIELDS = 40

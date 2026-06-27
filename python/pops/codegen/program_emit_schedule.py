"""pops.codegen.program_emit_schedule : the unified scheduler wrap (ADC-458).

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  ``_emit_schedule_wrap`` wraps the statements a node
emitted in its schedule's due-test guard + policy branch; ``program_emit_ops._emit_op``
calls it after each op lowers itself.  ``_schedule_due_test`` / ``_split_output_decl`` are
its helpers.  Reuses the op tables in ``program_emit_kernels``.
"""
import json

from pops.codegen.program_emit_kernels import _AUX_OUTPUT_OPS, Value


def _schedule_due_test(program, v, sched):
    """The C++ boolean 'is this node due this step' for a non-subcycle schedule kind. Reused as the
    guard of the policy branch. Raises (naming ADC-458) for a kind that needs a runtime primitive the
    compiled .so does not have (on_end: no end-of-run signal reaches a sim.step(dt) loop)."""
    kind = sched.kind
    if kind == "every":
        # Cadence: due cold-start, then every N macro-steps (CacheManager::is_due via macro_step()).
        return "ctx.cache_should_update(%d, %d)" % (v.id, int(sched.params.get("n", 1)))
    if kind == "on_start":
        return "(ctx.macro_step() == 0)"
    if kind == "when":
        # A runtime predicate: a Program Bool value already lowered to a parenthesized C++ expr token
        # (a compare over reductions). A bare Python callable cannot lower (it is not a Program value).
        cond = sched.params.get("cond")
        if not (isinstance(cond, Value) and cond.vtype == "bool"):
            raise NotImplementedError(
                "when(cond) lowers only a Program Bool predicate (e.g. P.norm2(r) < tol), not a "
                "Python callable: node %r (ADC-458). Build the condition with Program compares."
                % v.name)
        if cond.id not in program._when_tokens:
            raise ValueError(
                "when(cond) on node %r references a Bool value not emitted before it; build the "
                "predicate earlier in the Program (ADC-458)" % v.name)
        return program._when_tokens[cond.id]
    raise NotImplementedError(
        "schedule kind %r on node %r is not lowerable: on_end() needs an end-of-run signal that a "
        "compiled sim.step(dt) loop never sees (the .so cannot know the last step); use on_start()/"
        "every()/when()/subcycle() or an on_end host hook (ADC-458)." % (kind, v.name))

def _emit_schedule_wrap(program, v, var, lines, start):
    """Wrap the C++ statements node @p v emitted (``lines[start:]``) in its schedule's due-test guard
    + policy branch (ADC-458, Spec 3 sections 17-18). Generic over the op: a field solve caches the
    System aux, any other node caches its named scratch (var[v.id]). An always()/absent schedule
    leaves the lines untouched (byte-identical to the unscheduled lowering)."""
    sched = v.attrs.get("schedule")
    if sched is None or sched.is_always():
        return
    body = lines[start:]
    del lines[start:]
    if sched.kind == "subcycle":
        # Structured sub-cycling of a field solve (the gate restricts subcycle to an aux-output op):
        # run the op body COUNT times over the sub-dt (macro_dt / count by default, or an explicit
        # dt), refreshing the persistent System aux each pass. A pure recompute cadence -- no cache.
        # The sub-dt is a const local exposed for a dt-scaled body; the MVP field-solve body is
        # dt-free, so it documents the cadence (the aux solve re-runs COUNT times).
        count = int(sched.params["count"])
        sub_dt = sched.params.get("dt")
        sd = "_subdt%d" % v.id
        if sub_dt is None:
            lines.append("const pops::Real %s = dt / static_cast<pops::Real>(%d);" % (sd, count))
        else:
            lines.append("const pops::Real %s = static_cast<pops::Real>(%s);"
                         % (sd, repr(float(sub_dt))))
        lines.append("(void)%s;" % sd)  # the MVP body is self-contained; sd documents the cadence
        lines.append("for (int _sub%d = 0; _sub%d < %d; ++_sub%d) {" % (v.id, v.id, count, v.id))
        lines += ["  " + ln for ln in body]
        lines.append("}")
        return
    due = _schedule_due_test(program, v, sched)
    policy = sched.policy
    is_aux = v.op in _AUX_OUTPUT_OPS
    # The scratch node's output token (the MultiFab the policy holds / zeroes). A field solve writes
    # the System aux and sets no var[v.id], so out is read only on the scratch path.
    out = None if is_aux else var.get(v.id)
    if policy == "recompute":
        # Run only when due; on a NOT-due step do nothing (the aux / scratch keeps its last content).
        # recompute off-cadence is simply 'run when due' -- no cache, no else branch. A scratch node
        # hoists its output declaration so the buffer stays in scope when the body does not run.
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("}")
        return
    if policy == "skip":
        # Do not run the op off-cadence: the value keeps its previous content (the cacheable contract
        # -- downstream must tolerate a stale value). A scratch node hoists its output declaration so
        # the stale buffer stays in scope across the guard (no else branch: nothing happens off-
        # cadence); a field solve writes the persistent aux, so its whole body simply guards.
        if is_aux:
            lines.append("if (%s) {  // skip: stale aux off-cadence" % due)
            lines += ["  " + ln for ln in body]
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {  // skip: stale value off-cadence" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("}")
        return
    if policy == "zero":
        # Off-cadence, zero the node's output. The output must EXIST in both branches: for a scratch
        # node hoist its allocation out of the guard (the first emitted line declares var[v.id]); the
        # aux always exists (System-owned).
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("} else {")
            lines.append("  ctx.aux().set_val(static_cast<pops::Real>(0));")
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("} else {")
            lines.append("  %s.set_val(static_cast<pops::Real>(0));" % out)
            lines.append("}")
        return
    if policy == "hold":
        # Recompute + cache when due; restore the cached value off-cadence (no recompute). The aux
        # path uses cache_store_aux/restore_aux; a scratch node hoists its allocation and uses the
        # named-scratch cache. _validate_schedule already rejected hold on a non-cacheable operator.
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("  ctx.cache_store_aux(%d);" % v.id)
            lines.append("} else {")
            lines.append("  ctx.cache_restore_aux(%d);" % v.id)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
            lines.append("} else {")
            lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
            lines.append("}")
        return
    if policy == "accumulate_dt":
        # Off-cadence: accumulate THIS step's dt (the real skipped dt, never N*dt_current) and hold the
        # cached value. When due: read eff_dt = dt + sum(skipped) (resets the accumulator), recompute,
        # cache. eff_dt is bound so a dt-dependent recompute can read it (the MVP field solve / scratch
        # fill is dt-free, but eff_dt is exposed for a dt-scaled body). Cacheable (validated upstream).
        ed = "_effdt%d" % v.id
        if is_aux:
            lines.append("if (%s) {" % due)
            lines.append("  const pops::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                         % (ed, v.id, ed))
            lines += ["  " + ln for ln in body]
            lines.append("  ctx.cache_store_aux(%d);" % v.id)
            lines.append("} else {")
            lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
            lines.append("  ctx.cache_restore_aux(%d);" % v.id)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines.append("  const pops::Real %s = ctx.cache_effective_dt(%d, dt); (void)%s;"
                         % (ed, v.id, ed))
            lines += ["  " + ln for ln in rest]
            lines.append("  ctx.cache_store_scratch(%d, %s);" % (v.id, out))
            lines.append("} else {")
            lines.append("  ctx.cache_accumulate_dt(%d, dt);" % v.id)
            lines.append("  ctx.cache_restore_scratch(%d, %s);" % (v.id, out))
            lines.append("}")
        return
    if policy == "error":
        # Guard that a stale value is never read off-cadence: run when due, else fail loud (the node
        # asserts it is only consumed on its cadence). Emitted as a runtime throw on the not-due path.
        # A scratch node hoists its output declaration so the buffer stays in scope (the throw never
        # returns, but the C++ must still be well-scoped); a field solve guards the aux body directly.
        err = ('ctx.scheduler_error(%s);'
               % json.dumps("node '%s' (op '%s') read off its schedule cadence (policy=error)"
                            % (v.name, v.op)))
        if is_aux:
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in body]
            lines.append("} else {")
            lines.append("  " + err)
            lines.append("}")
        else:
            decl, rest = _split_output_decl(program, body, out, v)
            lines.append(decl)
            lines.append("if (%s) {" % due)
            lines += ["  " + ln for ln in rest]
            lines.append("} else {")
            lines.append("  " + err)
            lines.append("}")
        return
    raise NotImplementedError(
        "schedule policy %r on node %r is not lowerable (ADC-458)" % (policy, v.name))

def _split_output_decl(program, body, out, v):
    """Split a scratch node's emitted @p body into (declaration_line, rest): the OUTPUT scratch
    ``out`` must be declared OUTSIDE the policy guard so both branches see it, while the fill stays
    inside. The op declares its output as its FIRST emitted line (``pops::MultiFab <out> = ...;``);
    hoist exactly that one line. Raises if the shape is unexpected (a node whose output is not a
    freshly-declared scratch cannot use a cache/zero policy through this path)."""
    decl_prefix = "pops::MultiFab %s = " % out
    if not body or not body[0].startswith(decl_prefix):
        raise NotImplementedError(
            "schedule policy on node %r (op '%s') needs its output scratch %r declared as its first "
            "emitted line to hoist it out of the guard; got %r (ADC-458)"
            % (v.name, v.op, out, body[0] if body else None))
    return body[0], body[1:]

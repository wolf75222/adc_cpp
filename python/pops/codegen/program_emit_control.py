"""pops.codegen.program_emit_control : the body walk + control-flow op emitters.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  ``_emit_body`` is the two-phase body walk;
``_emit_while`` / ``_emit_range`` / ``_emit_if`` lower the control-flow ops (they re-run
the per-op lowering on their sub-blocks); ``_coupled_rate_components`` / ``_walk_expr``
resolve and scan a coupled_rate node.  The op dispatcher ``_emit_op`` lives in
``program_emit_ops`` and is imported LAZILY inside the functions below to break the
``ops`` <-> ``control`` recursion cycle at import time (it resolves fine at call time).
"""
import json


def _coupled_rate_components(program, v):
    """Resolve a ``coupled_rate`` node @p v to its per-block component formulas (Spec 3 criterion
    27, ADC-457), validated for the cons-only MVP. Returns ``{block: [Expr, ...]}`` (one formula
    per component of that block's StateSpace).

    The component formulas live in the BOUND operator's body (``op.body`` = the ``expr=`` dict
    passed to ``Module.operator``), reachable through the registry the node's ``operator`` attr
    names; the input states' cons names come from each input value's StateSpace (set by
    ``P.state(space=...)``). Raises a clear NotImplementedError naming ADC-457 when a coupled_rate
    cannot lower in this MVP: no bound registry, no operator body, a block whose component count
    does not match its StateSpace, or a formula referencing a non-cons (prim / aux) Var."""
    from pops.ir.expr import Var
    op_name = v.attrs["operator"]
    if program._registry is None:
        raise NotImplementedError(
            "the coupled_rate kernel codegen (ADC-457) needs the bound operator registry to reach "
            "operator %r's component formulas; call P.bind_operators(module) before emitting "
            "(node %r)" % (op_name, v.name))
    op = program._registry.get(op_name)
    expr = op.body
    if not isinstance(expr, dict):
        raise NotImplementedError(
            "the coupled_rate kernel codegen (ADC-457) needs operator %r to carry its per-block "
            "component formulas as an expr={block: [Expr, ...]} dict (got %r); a decorator-body "
            "coupled_rate is a later phase (node %r)" % (op_name, type(expr).__name__, v.name))
    # Each coupled_rate_out block must own one input state (its rate scratch is shaped like that
    # block's state) whose StateSpace gives the component count + cons names.
    by_block = {s.block: s for s in v.inputs}
    components = {}
    for blk, comps in expr.items():
        state_in = by_block.get(blk)
        if state_in is None or getattr(state_in, "space", None) is None:
            raise NotImplementedError(
                "the coupled_rate kernel codegen (ADC-457) needs every output block to map to an "
                "input State declared with a StateSpace (P.state(%r, space=...)); operator %r "
                "block %r has none (node %r)" % (blk, op_name, blk, v.name))
        ncons = len(state_in.space.components)
        if len(comps) != ncons:
            raise NotImplementedError(
                "coupled_rate operator %r block %r emits %d component formulas but its StateSpace "
                "has %d components; the rate must be full-rank over the block state (ADC-457, "
                "node %r)" % (op_name, blk, len(comps), ncons, v.name))
        for e in comps:
            for node in _walk_expr(e):
                if isinstance(node, Var) and node.kind != "cons":
                    raise NotImplementedError(
                        "coupled_rate formulas referencing prim/aux vars are deferred (ADC-457): "
                        "operator %r block %r references %s var %r; the MVP per-cell binding is "
                        "cons-only (node %r)" % (op_name, blk, node.kind, node.name, v.name))
        components[blk] = list(comps)
    # Every cons var a formula references must be a component of SOME input state (an output block
    # OR a read-only catalyst input). A name in no input state -- a typo, or a name the author
    # forgot to add to a P.state(space=...) -- would emit an undefined C++ identifier that only
    # fails at the AOT compile, far from the authoring site; reject it loud here, like prim/aux.
    all_cons = {c for s in v.inputs if getattr(s, "space", None) is not None
                for c in s.space.components}
    referenced = set()
    for comps in components.values():
        for e in comps:
            referenced |= e.deps()
    missing = referenced - all_cons
    if missing:
        raise NotImplementedError(
            "coupled_rate operator %r references cons var(s) %s that are a component of no input "
            "state; declare them via P.state(space=...) or fix the formula (ADC-457, node %r)"
            % (op_name, sorted(missing), v.name))
    return components

def _walk_expr(e):
    """Yield every node of a dsl Expr tree (used to scan a coupled_rate formula for non-cons Vars)."""
    from pops.ir.visitors import _children
    stack = [e]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(_children(node))

def _emit_body(program, model=None):
    """Generate the C++ of the install function in TWO phases (each list indented uniformly by the
    template). Assumes `_check_lowerable` has passed. @p model supplies the symbolic coefficients of
    the Phase-4b source / apply / solve_local_linear ops. Returns ``(prelude, body)``:

      - ``prelude``: INSTALL-TIME C++ (before ``ctx.install``) -- persistent scratch fields (held
        via ``std::shared_ptr`` so they outlive the install call and are reused across every step
        and every Krylov iteration) and the matrix-free apply lambdas. Captured by value into the
        step closure (shared_ptr / lambda / ctx all copy cheaply).
      - ``body``: the STEP closure body (one macro-step over dt).

    Multi-block (ADC-426): the SSA walk allocates a per-block base (``ctx.state(idx)`` for each
    declared block) and routes every op to ITS block's index via ``_block_indices`` / ``v.block``.
    Each committed block's final value is copied into that block's state (a scratch commit) or was
    written in place (a linear_combine commit). A single block reduces to the historical lowering."""
    from pops.codegen.program_emit_ops import _emit_op
    block_idx = program._block_indices()
    # The first-declared state Value per block: the "base" any op of that block clones / commits into.
    bases = {}
    for v in program._values:
        if v.op == "state" and v.block not in bases:
            bases[v.block] = v
    # IR value id -> C++ token: a MultiFab variable name (states / RHS scratches), a scalar variable
    # name (reductions, ``s{id}``) or a parenthesized boolean expression (compares).
    var = {}
    prelude = []
    lines = []
    # Bool-predicate value id -> its C++ token, for a when(cond) schedule whose cond is a Program
    # compare value emitted earlier in the body (ADC-458). Reset per emit (tokens are body-local).
    program._when_tokens = {}
    committed_ids = {s.id for s in program._commits.values()}
    # Multistep histories (ADC-406a): register each declared history at its MAX lag FIRST (a
    # registration-only call, NOT a read -- a read before the first store fails loud), so the ring
    # depth is locked before any store. The first ctx.store_history then cold-start-fills every
    # (already-allocated) slot -- step 0 reads the same value at every lag and the scheme degenerates
    # to a one-step method. register_history is idempotent (no-op once registered).
    for name, lag in sorted(program._histories.items()):
        lines.append("ctx.register_history(%s, %d);" % (json.dumps(name), int(lag)))
    for v in program._values:
        base = bases.get(v.block)  # the block-state value of THIS op's block (None: a scalar op)
        _emit_op(program, v, base, committed_ids, var, model, lines, prelude, block_idx)
    # Each committed block: a scratch commit (solve_local_linear / solve_linear / a non-base
    # linear_combine wrote a scratch) is copied into the block state; a linear_combine commit already
    # wrote ctx.state(idx) in place (var == base), so its copy is a no-op (skipped).
    for block, committed in program._commits.items():
        base = bases[block]
        if var[committed.id] != var[base.id]:
            lines.append(
                "ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                % (var[base.id], var[base.id], var[committed.id]))
    # Rotate the history rings ONCE at the very end of the step (after the commit), so the next step
    # reads lag k as the value k stores ago. Only emitted when the Program uses histories.
    if program._histories:
        lines.append("ctx.rotate_histories();")
    prelude_src = "\n".join("  " + ln for ln in prelude)
    body_src = "\n".join("    " + ln for ln in lines)
    return prelude_src, body_src

def _emit_while(program, v, base, var, model, lines, block_idx=None):
    """Lower a while op to an infinite C++ loop with a break (the condition re-evaluates each pass).
    The loop variable is a single MultiFab mutated IN PLACE across iterations; the cond / body sub-
    blocks re-run the per-op lowering each pass, with the loop-variable value id seeded to the loop
    var so their references resolve to it."""
    from pops.codegen.program_emit_ops import _emit_op
    loop_in = v.inputs[0]  # the initial loop-variable state
    x = "x%d" % v.id
    var[v.id] = x
    # Hoist + initialize the loop variable from the entry state (x <- loop_in).
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[loop_in.id]))
    lines.append("for (;;) {")
    # The sub-blocks see the loop variable in place of the entry-state value id (the body / cond were
    # built reading the loop-var State; they resolve to x here). A fresh sub-var map keeps the inner
    # scratch names from leaking out, but inherits the outer bindings (the loop var, target, ...).
    sub = dict(var)
    sub[loop_in.id] = x
    body_lines = []
    for w in v.attrs["cond_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    cond_expr = sub[v.attrs["cond"].id]
    body_lines.append("if (!(%s)) break;" % cond_expr)
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    # Write the next state into the loop variable in place (x <- body result).
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")

def _emit_range(program, v, base, var, model, lines, block_idx=None):
    """Lower a range op to a C++ ``for`` over a fixed count. Like a while, the loop variable is one
    MultiFab mutated in place and the body sub-block is emitted ONCE inside the loop (re-run each
    pass at runtime); the loop-variable value id is seeded to the loop var for the sub-block."""
    from pops.codegen.program_emit_ops import _emit_op
    loop_in = v.inputs[0]
    x = "x%d" % v.id
    i = "i%d" % v.id
    var[v.id] = x
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[loop_in.id]))
    lines.append("for (int %s = 0; %s < %d; ++%s) {" % (i, i, int(v.attrs["count"]), i))
    sub = dict(var)
    sub[loop_in.id] = x
    body_lines = []
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")

def _emit_if(program, v, base, var, model, lines, block_idx=None):
    """Lower an if op to a C++ branch. @p cond was emitted at the top level (its boolean expression
    is var[cond.id]); the loop variable is a copy of the input state, overwritten in place only when
    the branch is taken (so the result is the input state when the condition is false at runtime)."""
    from pops.codegen.program_emit_ops import _emit_op
    state_in, cond = v.inputs[0], v.inputs[1]
    x = "x%d" % v.id
    var[v.id] = x
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[state_in.id]))
    lines.append("if (%s) {" % var[cond.id])
    sub = dict(var)
    sub[state_in.id] = x
    body_lines = []
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx)
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")

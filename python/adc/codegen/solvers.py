"""adc.codegen.solvers -- C++ lowering for custom solver IR (Spec 4 split).

Verbatim extraction from python/adc/lib.py lines 590-616, 622, 625-889, 932-952,
955-1004. No logic changed. Cross-references resolved:
- SolverIR, SolverContext, _as_descriptor, _require_field, _operator_name,
  _iter_all_nodes, _walk_nodes, _SOLVER_MAX_ITERS -> adc.lib.solvers
  (those helpers were in the same file; now imported from the new sibling module).
- _fold_scalar_literal, _SolverCppLowering, _SOLVER_CPP_TEMPLATE: defined locally here
  (they are the codegen lowering internals).
- json: kept at module level exactly as lib.py has it.
"""
import json

from adc.lib.solvers import (  # SPEC4: was same-file in lib.py
    SolverIR,           # noqa: F401 -- re-exported for callers that import from here
    SolverContext,      # noqa: F401
    _as_descriptor,
    _require_field,     # noqa: F401 -- used by SolverContext (in lib/solvers.py); kept for completeness
    _operator_name,     # noqa: F401 -- used by SolverContext
    _iter_all_nodes,
    _walk_nodes,        # noqa: F401
    _SOLVER_MAX_ITERS,
    build_solver_ir,
)


def generate_solver_cpp(solver_brick, func=None):
    """Lower a custom solver IR to GENERATED C++ that RUNS (Spec 3 section 20, criterion 23).

    @p solver_brick is a ``@adc.lib.solver`` descriptor (or its registered name). Builds the
    solver IR with :func:`build_solver_ir` and walks the SSA, emitting a self-contained C++
    free function that drives the solve entirely in C++ -- it calls the SHARED matrix-free HPC
    primitives (``adc::dot`` / ``adc::saxpy`` / ``adc::lincomb`` from ``mf_arith.hpp``) and
    runs the convergence loop as a REAL C++ ``for (;;)`` whose predicate re-evaluates each
    iteration. Returns the C++ source string (a header-only templated kernel).

    The emitted signature is::

        template <class Op>
        adc::KrylovResult <name>_solve(const Op& A, adc::MultiFab& x,
                                       const adc::MultiFab& b);

    The operator ``A`` is a value-typed TEMPLATE parameter (a callable
    ``void(adc::MultiFab&, const adc::MultiFab&)``, the same shape ``adc::ApplyFn`` and the
    native Krylov loops take), so it inlines at the call site -- there is NO ``std::function``
    in the kernel, NO Python callback in the loop, NO heap allocation inside the loop (the
    scratch fields are allocated ONCE before it), and NO per-cell string lookup (criterion
    24.9). The DSL path is for CUSTOM solvers; a DSL solver that maps onto a native scheme
    keeps the ``adc::*_solve`` free functions as its backend.
    """
    desc = _as_descriptor(solver_brick)
    ir = build_solver_ir(desc)
    return _SolverCppLowering(ir, func or desc.name).emit()


# A hard upper bound on a custom solver's convergence loop: a generated kernel MUST terminate even if
# the authored predicate never goes false (a stalled / diverging custom solver). The authored
# ``it < max_iter`` cap normally stops the loop first; this is the backstop.
# NOTE: _SOLVER_MAX_ITERS is imported from adc.lib.solvers; the local alias is kept for
# the _SOLVER_CPP_TEMPLATE format call below and the _emit_while safety-cap literal.


class _SolverCppLowering:
    """Walk a :class:`SolverIR` and emit a standalone, self-contained C++ solver kernel.

    The walker mirrors the proven ``adc.time.Program`` op lowering for the solver subset
    (``state`` / ``linear_source`` / ``reduce`` (norm2/dot) / ``apply`` / ``linear_combine`` /
    ``scalar_op`` / ``compare`` / ``logical_and`` / ``while``) but emits a free function that
    is NOT bound to a ProgramContext / model: the matrix-free operator is the template
    parameter ``A`` and the vector operands are bare ``adc::MultiFab`` scratch fields combined
    with the shared ``adc::dot`` / ``adc::saxpy`` / ``adc::lincomb`` primitives. It is kept in
    ``adc.lib`` (the solver-codegen path) so it does not perturb the shared time/dsl codegen."""

    def __init__(self, ir, func):
        self._ir = ir
        self._func = str(func)
        self._var = {}          # IR value id -> C++ token (a MultiFab lvalue or a scalar / bool expr)
        self._scratch = []      # MultiFab scratch field declarations (alloc-once, before the loop)
        # Iteration counters (the ``it`` of ``it = it + 1``): an SSA scalar literal cannot mutate, so
        # the authored counter is a constant in the IR. We bind the FIRST such counter to the real C++
        # loop counter ``adc_iters`` so ``it < max_iter`` is a GENUINE trip bound, not a frozen test.
        self._counter_id = self._find_counter()
        # Every scalar_op that aliases the counter (the base ``it`` and its ``it + 1`` increments,
        # transitively) lowers to the live C++ ``adc_iters``. Pre-resolved so the cond_block (emitted
        # before the body) can reference the body's counter-update id, which is the same loop counter.
        self._counter_aliases = self._counter_alias_ids()
        # The rhs ``b`` is the first State the builder receives (ctx.unknown('b')); the iterate ``x``
        # is the warm-start State the result folds back onto (ctx.zeros_like(b)). Both lower to a
        # SINGLE C++ variable each, so the in-place loop update mutates one field, the cond re-reads it.
        self._b_id = self._first_state_id()
        self._iterate_id = self._find_iterate_id()
        if self._iterate_id is None:
            # Fail loud: a result with no iterate State (other than the rhs ``b``) would otherwise
            # lower to a kernel that silently returns its input. build_solver_ir validates the
            # result upstream, so a well-formed @adc.lib.solver never reaches this.
            raise ValueError(
                "solver IR has no solution iterate to lower (the result must fold onto a "
                "warm-start State distinct from the rhs); cannot generate C++")

    def emit(self):
        """The full C++ translation unit: the templated kernel + an explicit-double ABI wrapper."""
        body = self._emit_kernel_body()
        scratch = "\n".join("  " + ln for ln in self._scratch)
        body_src = "\n".join("  " + ln for ln in body)
        return _SOLVER_CPP_TEMPLATE.format(
            name=json.dumps(self._ir.descriptor.name), func=self._func,
            scratch=scratch, body=body_src, max_iters=_SOLVER_MAX_ITERS)

    # --- top-level walk ----------------------------------------------------
    def _emit_kernel_body(self):
        """Lower the top-level SSA list into the kernel body lines (the while op recurses).

        ``r`` (the result iterate) is the function's output ``x``; the warm start (``zeros_like``)
        zeroes it. Every other field op allocates an alloc-once scratch field. Reductions /
        scalars / compares lower inline; the ``while`` op becomes a real C++ loop."""
        lines = []
        result_id = self._ir.result.id
        for v in self._ir.program._values:
            self._emit_op(v, lines, result_id)
        return lines

    def _emit_op(self, v, lines, result_id):
        """Lower a SINGLE solver-IR op to C++, appending statements to @p lines and recording its
        C++ token in ``self._var``. Shared by the top-level walk and the while sub-blocks."""
        op = v.op
        if op == "linear_source":
            # The matrix-free operator A: its token is the kernel's template-parameter callable.
            self._var[v.id] = "A"
        elif op == "state":
            # A solver unknown / rhs / warm-start iterate. The rhs ``b`` is the kernel argument; the
            # warm-start iterate (``zeros_like(b)``) IS the output ``x`` zeroed (the loop updates it in
            # place); any other bare state is an alloc-once scratch field.
            if v.id == self._b_id:
                self._var[v.id] = "b"
            elif v.id == self._iterate_id:
                self._var[v.id] = "x"
                lines.append("x.set_val(static_cast<adc::Real>(0));  // warm start: zeros_like(b)")
            else:
                self._var[v.id] = self._alloc_field(None, v)
        elif op == "reduce":
            self._emit_reduce(v, lines)
        elif op == "apply":
            self._emit_apply(v, lines)
        elif op == "linear_combine":
            self._emit_combine(v, lines, result_id)
        elif op == "scalar_op":
            self._emit_scalar_op(v, lines)
        elif op == "compare":
            self._emit_compare(v)
        elif op == "logical_and":
            a, b = v.inputs
            self._var[v.id] = "(%s && %s)" % (self._var[a.id], self._var[b.id])
        elif op == "while":
            self._emit_while(v, lines, result_id)
        else:
            raise NotImplementedError(
                "generate_solver_cpp: solver-IR op %r (value %r) is not lowerable yet; the custom "
                "solver DSL supports zeros_like / norm2 / dot / apply / residual / combine / "
                "scalar_int / logical_and / while_ (Spec 3 section 20)" % (op, v.name))

    # --- per-op lowering ---------------------------------------------------
    def _emit_reduce(self, v, lines):
        """A collective reduction -> a C++ scalar local. ``norm2 = sqrt(adc::dot(u,u))``;
        ``dot`` calls ``adc::dot`` directly (the SAME shared primitive the native Krylov uses)."""
        tok = "s%d" % v.id
        self._var[v.id] = tok
        kind = v.attrs["kind"]
        if kind == "norm2":
            (u,) = v.inputs
            lines.append("const adc::Real %s = std::sqrt(adc::dot(%s, %s));"
                         % (tok, self._var[u.id], self._var[u.id]))
        elif kind == "dot":
            a, b = v.inputs
            lines.append("const adc::Real %s = adc::dot(%s, %s);"
                         % (tok, self._var[a.id], self._var[b.id]))
        else:
            raise NotImplementedError(
                "generate_solver_cpp: reduction kind %r is not lowerable in a custom solver "
                "(use norm2 / dot)" % (kind,))

    def _emit_apply(self, v, lines):
        """``A(x)`` -> the matrix-free matvec: call the template-parameter operator into an
        alloc-once scratch field. NO std::function -- ``A`` is the inlined template callable."""
        operator = v.inputs[0]
        state = v.inputs[1] if len(v.inputs) > 1 else None
        # The solver context records apply inputs as (state[, fields]); the operator name is in
        # attrs. The first State input is the vector A is applied to.
        x_in = next((inp for inp in v.inputs if getattr(inp, "vtype", None) == "state"), state)
        out = self._alloc_field(None, v)
        self._var[v.id] = out
        lines.append("A(%s, %s);" % (out, self._var[x_in.id]))
        _ = operator  # the operator is the single template callable A (no per-name dispatch)

    def _emit_combine(self, v, lines, result_id):
        """An affine combine ``sum_k c_k * field_k`` -> a scratch field (or the in-place iterate
        update). ``c_k`` are IR literal coefficients.

        The Richardson update ``x <- x + omega*r`` is the in-place iterate write: its term ``x`` is
        the loop variable (the kernel output ``x``), so the combine TARGETS ``x`` and keeps its
        content, adding only the other terms with ``adc::saxpy``. The residual ``r = b - A x``
        (whose terms are ``b`` / ``A(x)``, neither the iterate) targets a fresh alloc-once scratch
        that is zeroed first. So both the warm-start state ``#result`` and every in-place update map
        to the single token ``x`` -- the loop's cond re-reads ``A(x)`` on the updated iterate."""
        terms = list(zip(v.inputs, v.attrs["coeffs"], strict=True))
        # The combine carries the iterate iff one of its terms IS the iterate token "x" (the
        # x + omega*r update) or it is the returned warm-start itself.
        carries_iterate = v.id == result_id or any(self._var[inp.id] == "x" for inp, _ in terms)
        target = "x" if carries_iterate else self._alloc_field(None, v)
        self._var[v.id] = target
        self_term = next((c for inp, c in terms if self._var[inp.id] == target), None)
        if self_term is None:
            lines.append("%s.set_val(static_cast<adc::Real>(0));" % target)
        else:
            scale = float(self_term.get(0, 0.0))
            if scale != 1.0:
                lines.append("adc::saxpy(%s, static_cast<adc::Real>(%s), %s);  // scale self term"
                             % (target, repr(scale - 1.0), target))
        for inp, coeff in terms:
            tok = self._var[inp.id]
            if tok == target:
                continue
            lines.append("adc::saxpy(%s, static_cast<adc::Real>(%s), %s);"
                         % (target, repr(float(coeff.get(0, 0.0))), tok))

    def _emit_scalar_op(self, v, lines):
        """Scalar arithmetic (add/sub/mul/div). The bound iteration counter lowers to the real C++
        loop counter ``adc_iters`` (so ``it < max_iter`` is a genuine trip bound); the counter's
        ``it = it + 1`` increment is the loop's own ``++adc_iters`` and emits nothing. A pure-literal
        ``scalar_int(n)`` collapses to its compile-time value; any other scalar arithmetic lowers to
        a C++ expression over its operands."""
        # The counter and its increments alias the live C++ ``adc_iters`` (no statement emitted).
        if v.id in self._counter_aliases:
            self._var[v.id] = "static_cast<adc::Real>(adc_iters)"
            return
        operands = v.attrs["operands"]
        fn = v.attrs["fn"]
        # A literal-only scalar_int (scalar_int(n) is built as n + 0.0): fold to its constant value.
        if all(kind == "c" for kind, _ in operands):
            self._var[v.id] = "static_cast<adc::Real>(%s)" % repr(_fold_scalar_literal(operands, fn))
            return
        toks = []
        for kind, val in operands:
            if kind == "v":
                toks.append(self._var[v.inputs[val].id])
            else:
                toks.append("static_cast<adc::Real>(%s)" % repr(float(val)))
        cppop = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[fn]
        self._var[v.id] = "(%s %s %s)" % (toks[0], cppop, toks[1])

    def _counter_alias_ids(self):
        """The set of scalar_op ids that alias the loop counter: the base ``it`` plus every
        ``it + literal`` increment (transitively). All lower to the live C++ ``adc_iters``."""
        if self._counter_id is None:
            return frozenset()
        aliases = {self._counter_id}
        nodes = list(_iter_all_nodes(self._ir.program._values))
        changed = True
        while changed:
            changed = False
            for v in nodes:
                if v.op != "scalar_op" or v.id in aliases:
                    continue
                reads = [v.inputs[idx].id for kind, idx in v.attrs["operands"] if kind == "v"]
                if any(rid in aliases for rid in reads):
                    aliases.add(v.id)
                    changed = True
        return frozenset(aliases)

    def _find_counter(self):
        """The IR id of the iteration counter: a top-level literal ``scalar_int`` (built as ``n+0``)
        that a body ``scalar_op`` reads to increment it (the ``it = it + scalar_int(1)`` idiom). Bound
        to the live C++ loop counter so the authored ``it < max_iter`` cap actually bounds the
        generated loop. ``None`` if the solver authors no counter (the loop then relies on the
        residual test / the hard safety cap to terminate)."""
        prog = self._ir.program
        # A counter SEED is a pure-literal scalar_op declared at the top level (before the loop).
        seeds = {v.id for v in prog._values
                 if v.op == "scalar_op" and all(k == "c" for k, _ in v.attrs["operands"])}
        # The counter is a seed that some OTHER scalar_op reads (its increment); a bare unread literal
        # (e.g. a max_iter constant compared directly) is not a counter.
        for v in _iter_all_nodes(prog._values):
            if v.op != "scalar_op":
                continue
            reads = [v.inputs[idx].id for kind, idx in v.attrs["operands"] if kind == "v"]
            for rid in reads:
                if rid in seeds:
                    return rid
        return None

    def _emit_compare(self, v):
        """A scalar predicate -> an inline C++ boolean expression (no statement; the while op embeds
        it in ``if (!(<expr>)) break;``)."""
        lhs = v.inputs[0]
        if len(v.inputs) == 2:
            rhs_tok = self._var[v.inputs[1].id]
        else:
            rhs_tok = "static_cast<adc::Real>(%s)" % repr(float(v.attrs["rhs"]))
        self._var[v.id] = "(%s %s %s)" % (self._var[lhs.id], v.attrs["cmp"], rhs_tok)

    def _emit_while(self, v, lines, result_id):
        """Lower the convergence loop to a REAL C++ ``for (;;)`` with a break: the cond sub-block is
        re-emitted each pass (the predicate re-evaluates against the loop-updated iterate), then the
        body sub-block runs. The C++ loop counter ``adc_iters`` is the authored iteration counter (so
        the author's ``it < max_iter`` cap really bounds the loop); a hard safety cap also breaks the
        loop so a stalled custom solver can never spin forever even if its IR omits a cap."""
        # The convergence predicate (cond_block) was RECORDED AFTER the body so it reads the
        # loop-updated iterate -- in the IR it references the body's iterate-update State (result_id).
        # In the C++ loop the iterate is the single in-place variable ``x``, so pre-bind the body's
        # iterate-update id to ``x``: the cond (emitted first, runs each pass against the prior body's
        # x) and the body (which writes x in place) then both resolve to the one loop variable.
        self._var[result_id] = "x"
        # Pre-bind every counter alias to the live loop counter, so the cond_block (emitted before the
        # body) can reference the body's counter-update id -- they are all the one C++ ``adc_iters``.
        for cid in self._counter_aliases:
            self._var[cid] = "static_cast<adc::Real>(adc_iters)"
        lines.append("adc_iters = 0;")  # function-scope counter (declared in the kernel template)
        lines.append("for (;; ++adc_iters) {")
        body = ["if (adc_iters >= %d) break;  // hard safety cap (custom solver)" % _SOLVER_MAX_ITERS]
        for w in v.attrs["cond_block"]:
            self._emit_op(w, body, result_id)
        cond_expr = self._var[v.attrs["cond"].id]
        body.append("if (!(%s)) break;" % cond_expr)
        for w in v.attrs["body_block"]:
            self._emit_op(w, body, result_id)
        lines += ["  " + ln for ln in body]
        lines.append("}")

    # --- helpers -----------------------------------------------------------
    def _alloc_field(self, fixed, v):
        """The C++ token of a vector field for IR value @p v. ``fixed="x"`` is the kernel output (no
        new scratch); otherwise allocate an ALLOC-ONCE scratch field shaped like the rhs ``b`` (one
        ghost, ncomp(b)) before the loop -- never inside it (no heap churn in the kernel)."""
        if fixed == "x":
            return "x"
        tok = "v%d" % v.id
        self._scratch.append("adc::MultiFab %s(b.box_array(), b.dmap(), b.ncomp(), 1);" % tok)
        return tok

    def _first_state_id(self):
        """The id of the rhs ``b`` IR State: the first State the builder receives (build_solver_ir
        binds it as ``ctx.unknown('b')`` right after the operator linear_source 'A')."""
        for v in self._ir.program._values:
            if v.op == "state":
                return v.id
        return None

    def _find_iterate_id(self):
        """The id of the solution iterate ``x``: the warm-start State the result folds back onto. The
        result is either a State (the warm start returned directly) or a ``linear_combine`` whose State
        input (not the rhs ``b``) is the iterate base (the ``x`` of ``x + omega*r``)."""
        res = self._ir.result
        if res.op == "state":
            return res.id
        # Walk the result combine's State inputs for the non-``b`` State (the iterate base).
        seen = set()
        frontier = [res]
        while frontier:
            node = frontier.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            if node.op == "state" and node.id != self._b_id:
                return node.id
            for inp in node.inputs:
                if getattr(inp, "vtype", None) == "state":
                    frontier.append(inp)
        return None


def _fold_scalar_literal(operands, fn):
    """Fold a pure-literal scalar_op (the ``scalar_int(n) = n + 0.0`` idiom) to its constant float."""
    a = float(operands[0][1])
    b = float(operands[1][1]) if len(operands) > 1 else 0.0
    if fn == "add":
        return a + b
    if fn == "sub":
        return a - b
    if fn == "mul":
        return a * b
    return a / b


_SOLVER_CPP_TEMPLATE = '''\
// GENERATED by adc.lib.generate_solver_cpp (epic ADC-462, Spec 3 section 20 / criterion 23).
// Do not edit by hand. A CUSTOM solver authored in the @adc.lib.solver IR-DSL, lowered to a
// self-contained C++ kernel that drives the solve entirely in C++ via the SHARED matrix-free
// HPC primitives (adc::dot / adc::saxpy / adc::lincomb). The matrix-free operator A is a
// value-typed TEMPLATE parameter (a callable void(MultiFab&, const MultiFab&), the same shape
// the native Krylov loops take), so it INLINES at the call site: no type-erased indirection in
// the kernel, no Python callback in the loop, no heap allocation inside the loop (the scratch
// fields below are allocated once, before it), and no per-cell name dispatch (criterion 24.9).
#include <adc/mesh/storage/multifab.hpp>                     // adc::MultiFab
#include <adc/mesh/storage/mf_arith.hpp>                     // adc::dot / adc::saxpy / adc::lincomb
#include <adc/numerics/elliptic/linear/krylov_result.hpp>    // adc::KrylovResult
#include <adc/core/foundation/types.hpp>                     // adc::Real
#include <cmath>                                             // std::sqrt in lowered norms

// The custom solver kernel: solve A x = b, writing the solution into x (warm-started from x).
// Returns the iteration count / final relative residual in an adc::KrylovResult.
template <class Op>
inline adc::KrylovResult {func}_solve(const Op& A, adc::MultiFab& x, const adc::MultiFab& b) {{
  int adc_iters = 0;  // convergence-loop counter (0 for a loop-free solver)
{scratch}
{body}
  // Final relative residual ||b - A x|| / ||b|| over the SAME shared primitives (a diagnostic, once,
  // after the loop): A(x) into a scratch, r = b - A x, then the global L2 norms via adc::dot.
  adc::MultiFab adc_resid(b.box_array(), b.dmap(), b.ncomp(), 1);
  A(adc_resid, x);
  adc::lincomb(adc_resid, static_cast<adc::Real>(1), b, static_cast<adc::Real>(-1), adc_resid);
  const adc::Real adc_bnorm = std::sqrt(adc::dot(b, b));
  const adc::Real adc_rnorm = std::sqrt(adc::dot(adc_resid, adc_resid));
  adc::KrylovResult res;
  res.iters = adc_iters;
  res.rel_residual = adc_bnorm > adc::Real(0) ? adc_rnorm / adc_bnorm : adc_rnorm;
  res.converged = adc_iters < {max_iters};
  return res;
}}
'''

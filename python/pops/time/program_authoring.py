"""pops.time Program authoring mixin -- dumps + control flow + reductions + dt bound.

IR dumps, decorator step(), control flow (while_/if_/range), reductions and the dt bound.
"""
from pops.time.program_base import _ProgramConstants
from pops.time.values import Value, _is_field_value


class _ProgramAuthoring(_ProgramConstants):
    """IR dumps, decorator step(), control flow (while_/if_/range), reductions and the dt bound."""

    @staticmethod
    def _render_node(v):
        """Render one IR value as an operator-first line (introspection, not codegen)."""
        ins = ", ".join(i.name for i in v.inputs)
        extra = ""
        keys = {k: val for k, val in v.attrs.items() if k != "coeffs"}
        if "coeffs" in v.attrs:
            extra = "  # coeffs=%s" % (v.attrs["coeffs"],)
        elif keys:
            extra = "  # %s" % (keys,)
        return "%-16s = P.%s(%s)%s" % (v.name, v.op, ins, extra)

    def dump_operator_ir(self):
        """The operator-first Program IR (one line per node): P.call/linear_combine/
        solve_local_linear/commit. The board sugar lowers to exactly this -- the dump
        proves the board and the operator-first writings share one IR."""
        lines = ["# operator-first Program IR: %s" % self.name]
        for v in self._values:
            lines.append("  " + self._render_node(v))
        for block, st in self._commits.items():
            lines.append("  P.commit(%r, %s)" % (block, st.name))
        return "\n".join(lines)

    def dump_board(self):
        """The board-level view; board notation lowers to the operator-first IR below."""
        return ("# board program %s lowers to the operator-first IR (board == operator-first):\n%s"
                % (self.name, self.dump_operator_ir()))

    def dump_cpp_plan(self):
        """A textual C++ plan of the generated step (ProgramContext calls), NOT the exact
        codegen -- it shows which ctx / GeneratedModule call each node lowers to."""
        lines = ["// C++ plan for GeneratedProgram step of %s" % self.name]
        for v in self._values:
            ins = ", ".join(i.name for i in v.inputs)
            if v.op == "coupled_rate":
                # the coupled-rate kernel lowers to ONE multi-state for_each_cell filling every block's
                # rate scratch at once (ADC-457); there is no single ctx.coupled_rate(...) call.
                blks = ", ".join(v.attrs.get("blocks", []))
                lines.append("  // %s: multi-state for_each_cell rate kernel over (%s) for blocks "
                             "[%s];  // ADC-457" % (v.name, ins, blks))
            elif v.op == "coupled_rate_out":
                # a pure projection: it aliases its block's rate scratch (no code of its own).
                lines.append("  // %s = %s.rate[%r];  // ADC-457 (block projection, no ctx call)"
                             % (v.name, ins, v.attrs.get("out_block")))
            else:
                lines.append("  ctx.%s(%s);  // -> %s" % (v.op, ins, v.name))
        for block, st in self._commits.items():
            lines.append("  ctx.commit(%r, %s);" % (block, st.name))
        return "\n".join(lines)

    # --- decorator mode (ADC-423): record the step body from a function ---
    def step(self, fn):
        """Record this Program's IR by calling @p fn(self) ONCE, at build time (decorator mode).

        ``@P.step`` is sugar for an inline builder body: the decorated function receives the Program
        and builds the IR exactly as if its statements had been written at module scope. It is a
        BUILD-TIME callback -- it runs once, here, to populate the SSA value list; it is NEVER executed
        numerically during ``sim.step`` (the compiled ``.so`` owns the runtime step). So

            P = pops.time.Program("fe")

            @P.step
            def _(P):
                pops.lib.time.forward_euler(P, "plasma")

        produces byte-identical IR (same ``_ir_hash``) to calling
        ``pops.lib.time.forward_euler(P, "plasma")`` inline. Returns the Program so a one-liner
        ``P = pops.time.Program("p").step(build)`` also reads
        cleanly. @p fn must be callable; it is invoked with the Program as its single argument."""
        if not callable(fn):
            raise TypeError("Program.step expects a callable build_fn(P); got %r" % (fn,))
        fn(self)
        return self

    # --- ghost fill / positivity projection (spec ops 22 / 21) ---
    def fill_boundary(self, x):
        """Fill the ghost cells (halos) of a State/scalar_field @p x in place: the transport BC
        (periodic by default), the same exchange `laplacian` / `gradient` / `divergence` do internally
        before differencing (spec op 22). Returns @p x (the SAME value -- a side effect on its ghosts,
        the valid cells are untouched). Lowered to ``ctx.fill_boundary(x)``."""
        if not _is_field_value(x):
            raise ValueError("fill_boundary: a State/RHS/scalar_field value is required (got %r)"
                             % (x,))
        return self._new(x.vtype, "fill_boundary", (x,), {}, x.name, x.block)

    def project(self, name=None, state=None, projection="block"):
        """Apply the block's post-step positivity projection to @p state in place (spec op 21):
        ``U <- project(U, aux)`` over the valid cells, the SAME Zhang-Shu / floor projection the native
        per-step path runs (ADC-177). Returns a State value (the projected state). @p projection selects
        the projection primitive; only ``"block"`` (the block's own, set at add_block time) is supported
        -- a custom projection is a later phase. Lowered to ``ctx.apply_projection(idx, state)`` for the
        state's own block (ADC-426)."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("project: a State value is required (state=...)")
        if projection != "block":
            raise NotImplementedError(
                "project: only projection='block' (the block's own positivity projection) is "
                "supported; a custom projection is a later phase (got %r)" % (projection,))
        return self._new("state", "project", (state,), {"projection": projection}, name,
                         state.block)

    # --- per-cell conditional select (spec op 17, ADC-418) ---

    def cell_compare(self, field, value, cmp, name=None):
        """A PER-CELL comparison ``field <cmp> value`` -> a fresh 1-component 0/1 mask scalar_field (1.0
        where the comparison holds, 0.0 otherwise), evaluated cell by cell on component 0 of @p field
        (its sole / first conserved component). @p field is a State/RHS/scalar_field value; @p value is a
        Python float threshold (a per-cell field threshold is a later phase); @p cmp is one of
        ``'>' '>=' '<' '<='``. The mask is the input the per-cell `where` selects on. Lowered to a
        for_each_cell kernel ``maskA(i,j,0) = fieldA(i,j,0) <cmp> value ? 1 : 0``. Convenience wrappers:
        `cell_gt` / `cell_ge` / `cell_lt` / `cell_le`."""
        if not _is_field_value(field):
            raise ValueError("cell_compare: a State/RHS/scalar_field value is required (got %r)"
                             % (field,))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("cell_compare: value must be a Python float threshold (a per-cell field "
                            "threshold is a later phase); got %r" % (value,))
        if cmp not in self._CELL_CMPS:
            raise ValueError("cell_compare: cmp must be one of %s; got %r"
                             % (sorted(self._CELL_CMPS), cmp))
        return self._new("scalar_field", "cell_compare", (field,),
                         {"cmp": cmp, "value": float(value)}, name, field.block)

    def cell_gt(self, field, value, name=None):
        """Per-cell ``field > value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, ">", name)

    def cell_ge(self, field, value, name=None):
        """Per-cell ``field >= value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, ">=", name)

    def cell_lt(self, field, value, name=None):
        """Per-cell ``field < value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, "<", name)

    def cell_le(self, field, value, name=None):
        """Per-cell ``field <= value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, "<=", name)

    def where(self, mask, a, b, name=None):
        """A PER-CELL conditional select (spec op 17): ``out(i,j,c) = mask(i,j,*) != 0 ? a(i,j,c) :
        b(i,j,c)`` COMPONENT-WISE over the field. This is NOT the scalar runtime branch `if_` -- the
        condition is decided per cell INSIDE a Kokkos kernel.

          - @p mask: a 0/1 (or any nonzero/zero) mask field. Either 1-component (one mask shared by all
            components -- read at component 0) or with the SAME ncomp as @p a / @p b (a per-component
            mask). Built with `cell_ge` / `cell_gt` / ... (a threshold) or any scalar_field.
          - @p a, @p b: the two field values to choose between, on the SAME grid and ncomp (a State or a
            scalar_field). The result has @p a's vtype / block / ncomp.

        Lowered to a for_each_cell select kernel (a ternary, no branch divergence concern at MVP)."""
        for nm, fv in (("mask", mask), ("a", a), ("b", b)):
            if not _is_field_value(fv):
                raise ValueError("where: %s must be a State/RHS/scalar_field value (got %r)"
                                 % (nm, fv))
        if a.vtype != b.vtype:
            raise ValueError("where: a and b must have the same value type (a is %s, b is %s)"
                             % (a.vtype, b.vtype))
        if a.block != b.block:
            raise ValueError("where: a and b must belong to the same block (a is %r, b is %r)"
                             % (a.block, b.block))
        na, nb, nm_ = self._ncomp(a), self._ncomp(b), self._ncomp(mask)
        if na is not None and nb is not None and na != nb:
            raise ValueError("where: a and b must have the same ncomp (a has %d, b has %d)" % (na, nb))
        ncomp = na if na is not None else nb
        if nm_ is not None and ncomp is not None and nm_ not in (1, ncomp):
            raise ValueError("where: mask must be 1-component or match a/b's ncomp (mask has %d, "
                             "a/b have %d)" % (nm_, ncomp))
        attrs = {"ncomp": ncomp} if ncomp is not None else {}
        return self._new(a.vtype, "where", (mask, a, b), attrs, name, a.block)

    @staticmethod
    def _ncomp(value):
        """The statically-known component count of a field value, or None when it is not pinned in the
        IR (a State / RHS ncomp is the model's n_cons, known only at codegen): a scalar_field carries its
        own ``ncomp`` attr. Used by `where` for the static a/b/mask ncomp consistency check."""
        if value.vtype == "scalar_field":
            return int(value.attrs.get("ncomp", 1))
        return None

    # --- reductions / comparisons / control flow (ADC-404a) ---
    def norm2(self, state):
        """The Euclidean norm ``||u||_2`` of a State (a collective all_reduce). Returns a Scalar value.
        Lowered as ``sqrt(pops::dot(u, u))`` -- the same collective reduction every rank must run. NOTE:
        ``pops::dot`` reduces COMPONENT 0 only, so for a multi-component State this is the L2 norm of the
        first conserved variable, not the full state norm; a full multi-component reduction is a later
        phase (the convergence loops this enables are single-residual-component for now)."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("norm2: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "norm2"}, None, state.block)

    def dot(self, a, b):
        """The inner product ``<a, b>`` of two State values (a collective all_reduce). Returns a Scalar.
        Lowered as ``pops::dot(a, b)`` -- COLLECTIVE, called on every rank (empty ranks included)."""
        if not (isinstance(a, Value) and a.is_field() and isinstance(b, Value) and b.is_field()):
            raise ValueError("dot: two State/RHS values are required")
        return self._new("scalar", "reduce", (a, b), {"kind": "dot"}, None, a.block)

    def norm_inf(self, state):
        """The infinity norm ``max|u|`` of a State (a collective all_reduce). Returns a Scalar value.
        Lowered as ``pops::norm_inf(u)``. Like norm2/dot it reduces COMPONENT 0 only (a multi-component
        reduction is a later phase) and MUST run on every rank (it goes through the collective seam)."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("norm_inf: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "norm_inf"}, None, state.block)

    def sum(self, state):
        """The sum ``sum_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``pops::reduce_sum(u, 0)`` -- COLLECTIVE, called on every rank (empty
        ranks included), the same seam pops::dot uses. Like norm2/dot it reduces COMPONENT 0 only (a
        full multi-component reduction is a later phase). For a specific component use
        `sum_component`."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("sum: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "sum", "comp": 0}, None, state.block)

    def max(self, state):
        """The maximum ``max_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``pops::reduce_max(u, 0)`` (the SIGNED max, not the magnitude -- use
        `norm_inf` for max|u|). COLLECTIVE: called on every rank. Component 0 only."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("max: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "max", "comp": 0}, None, state.block)

    def min(self, state):
        """The minimum ``min_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``pops::reduce_min(u, 0)``. COLLECTIVE: called on every rank.
        Component 0 only."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("min: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "min", "comp": 0}, None, state.block)

    def sum_component(self, state, comp):
        """The sum ``sum_cells u(.,comp)`` of a State over conservative component @p comp (a collective
        all_reduce). Returns a Scalar value. Lowered as ``pops::reduce_sum(u, comp)``. COLLECTIVE:
        called on every rank. @p comp must be a Python int >= 0 (a runtime component is meaningless)."""
        if not (isinstance(state, Value) and state.is_field()):
            raise ValueError("sum_component: a State/RHS value is required")
        if isinstance(comp, bool) or not isinstance(comp, int) or comp < 0:
            raise ValueError("sum_component: comp must be a Python int >= 0 (got %r)" % (comp,))
        return self._new("scalar", "reduce", (state,), {"kind": "sum", "comp": int(comp)}, None,
                         state.block)

    def record_scalar(self, name, value):
        """Record a runtime Scalar @p value (e.g. ``P.norm2(R)``) into the System diagnostics map under
        @p name, retrievable after the step via ``sim.program_diagnostic(name)`` /
        ``sim.program_diagnostics()`` (spec op 23). A side-effecting op (no value): it stores the scalar
        for inspection / logging, it does not feed the numerics. @p name must be a non-empty string;
        @p value must be a Scalar value (a P.norm2 / P.dot / P.sum / ... result), not a field. Lowered
        to ``ctx.record_scalar("<name>", <scalar>)``."""
        if not isinstance(name, str) or not name:
            raise ValueError("record_scalar: name must be a non-empty string")
        if not (isinstance(value, Value) and value.vtype == "scalar"):
            raise ValueError("record_scalar: value must be a Scalar value (e.g. P.norm2(R)); got %r"
                             % (value,))
        return self._new("scalar", "record_scalar", (value,), {"diagnostic": name}, name,
                         value.block)

    def _scalar_binop(self, a, b, fn):
        """Build a Scalar arithmetic node ``a <fn> b`` (fn in add/sub/mul/div). Each operand is a Scalar
        Value or a Python number (a literal constant, stored in attrs). Used by the Value scalar dunders
        so a dt_bound can express cfl * hmin / max_wave_speed (spec s18); never evaluated in Python."""
        inputs = []
        operands = []  # per operand: ("v", index-into-inputs) or ("c", literal float)
        for x in (a, b):
            if isinstance(x, Value):
                if x.vtype != "scalar":
                    raise TypeError("scalar arithmetic operands must be Scalar values or numbers; got "
                                    "a %s value %r" % (x.vtype, x.name))
                operands.append(("v", len(inputs)))
                inputs.append(x)
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                operands.append(("c", float(x)))
            else:
                raise TypeError("scalar arithmetic operands must be Scalar values or numbers; got %r"
                                % (x,))
        block = next((x.block for x in (a, b) if isinstance(x, Value)), None)
        return self._new("scalar", "scalar_op", tuple(inputs), {"fn": fn, "operands": operands}, None,
                         block)

    def max_wave_speed(self, state):
        """The maximum |wave speed| of @p state's block (a collective reduction): the SAME per-block
        wave speed the native CFL uses (BlockState::max_speed). Returns a Scalar value. Lowered to
        ``ctx.max_wave_speed(idx, u)`` for @p state's own block (ADC-426). The denominator of a
        CFL-style dt bound cfl * hmin / w (spec s18). REUSES the block's wave-speed closure -- it does
        not recompute the speed."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("max_wave_speed: a State value is required")
        return self._new("scalar", "max_wave_speed", (state,), {}, None, state.block)

    def hmin(self):
        """The MIN physical cell size of the grid (Cartesian min(dx, dy); polar min(dr, r_min*dtheta)):
        the SAME hmin the native CFL uses. Returns a Scalar value. Lowered to ``ctx.hmin()``. The
        numerator factor of a CFL-style dt bound cfl * hmin / max_wave_speed (spec s18)."""
        return self._new("scalar", "hmin", (), {}, None, None)

    def _compare(self, lhs, rhs, cmp):
        """Build a Bool predicate ``s_lhs <cmp> rhs`` (re-evaluated each loop pass). @p rhs is a Python
        float tolerance (stored as a literal) or another Scalar value (compared at runtime). Inputs are
        the Scalar operand(s); the float bound lives in attrs['rhs']."""
        if isinstance(rhs, (int, float)):
            return self._new("bool", "compare", (lhs,), {"cmp": cmp, "rhs": float(rhs)}, None,
                             lhs.block)
        if isinstance(rhs, Value) and rhs.vtype == "scalar":
            return self._new("bool", "compare", (lhs, rhs), {"cmp": cmp}, None, lhs.block)
        raise TypeError("compare: the right-hand side must be a float tolerance or a Scalar value, "
                        "got %r" % (rhs,))

    def while_(self, state, cond_fn, body_fn):
        """A convergence loop: starting from @p state, while ``cond_fn(self, x)`` holds, replace x by
        ``body_fn(self, x)``; return the final State.

        The condition and body are RE-EVALUATED each iteration, so the ops they build are captured into
        a separate recording sub-block (NOT the flat SSA list) and re-emitted inside a C++ loop. The
        loop variable is the SAME C++ MultiFab across passes (mutated in place).

          - ``cond_fn(self, x)`` must return a Bool value (e.g. ``self.norm2(diff) > tol``);
          - ``body_fn(self, x)`` must return the next-iteration State (e.g. a linear_combine)."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("while_: the loop variable must be a State value")
        if self._recording:
            raise NotImplementedError(
                "while_: nested control flow is a later phase; a while_ body cannot itself open a "
                "while_ yet")
        cond_block, cond_val = self._record(cond_fn, state)
        if not (isinstance(cond_val, Value) and cond_val.vtype == "bool"):
            raise ValueError("while_: cond_fn must return a Bool value (e.g. P.norm2(d) > tol)")
        body_block, next_state = self._record(body_fn, state)
        if not (isinstance(next_state, Value) and next_state.vtype == "state"):
            raise ValueError("while_: body_fn must return the next-iteration State value")
        if next_state.block != state.block:
            raise ValueError("while_: body_fn must return a State of the same block as the loop "
                             "variable")
        return self._new("state", "while", (state,),
                         {"cond_block": cond_block, "cond": cond_val,
                          "body_block": body_block, "body": next_state},
                         None, state.block)

    def static_range(self, state, count, body_fn):
        """A COMPILE-TIME (unrolled) loop: apply ``body_fn(self, x)`` to the State @p count times,
        threading the result, and return the final State. @p count must be a Python int known at IR
        build time -- the loop is unrolled HERE (no IR control-flow op, no C++ loop): it simply builds
        @p count copies of the body ops in order. Use `range` for a C++ ``for`` over a fixed count."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("static_range count must be a Python int (compile-time); use P.range for "
                            "a runtime / C++-loop count")
        if count < 0:
            raise ValueError("static_range count must be non-negative")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("static_range: the loop variable must be a State value")
        x = state
        for _ in range(count):
            x = body_fn(self, x)
            if not (isinstance(x, Value) and x.vtype == "state" and x.block == state.block):
                raise ValueError("static_range: body_fn must return a State of the loop variable's "
                                 "block")
        return x

    def range(self, state, count, body_fn):
        """A C++ ``for`` loop over a FIXED count: from @p state, apply ``body_fn(self, x)`` @p count
        times, threading the loop-variable State in place, and return the final State. @p count must be
        a Python int (a runtime/Scalar count is a later phase). The body is RE-EXECUTED each pass, so
        its ops are captured into a recording sub-block (NOT the flat SSA list) and emitted ONCE inside
        the loop. Use `static_range` to unroll instead."""
        if isinstance(count, Value):
            if count.vtype == "scalar":
                raise NotImplementedError("range with a runtime Scalar count is deferred; use a "
                                          "Python int")
            raise TypeError("range count must be a Python int")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("range count must be a Python int")
        if count < 0:
            raise ValueError("range count must be non-negative")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("range: the loop variable must be a State value")
        if self._recording:
            raise NotImplementedError("range: nested control flow is a later phase; a control-flow "
                                      "body cannot itself open a range yet")
        body_block, next_state = self._record(body_fn, state)
        if not (isinstance(next_state, Value) and next_state.vtype == "state"
                and next_state.block == state.block):
            raise ValueError("range: body_fn must return the next-iteration State of the same block")
        return self._new("state", "range", (state,),
                         {"count": int(count), "body_block": body_block, "body": next_state},
                         None, state.block)

    def if_(self, state, cond, body_fn):
        """A C++ ``if`` branch: from @p state, if the runtime Bool @p cond holds, replace the state by
        ``body_fn(self, x)``; otherwise leave it unchanged. Returns the (possibly updated) State. @p
        cond is a Bool value built BEFORE if_ (e.g. ``P.norm2(d) > tol``), evaluated ONCE. The body ops
        are captured into a recording sub-block and emitted inside the branch."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("if_: the state must be a State value")
        if not (isinstance(cond, Value) and cond.vtype == "bool"):
            raise ValueError("if_: cond must be a Bool value (e.g. P.norm2(d) > tol)")
        if self._recording:
            raise NotImplementedError("if_: nested control flow is a later phase; a control-flow body "
                                      "cannot itself open an if_ yet")
        body_block, next_state = self._record(body_fn, state)
        if not (isinstance(next_state, Value) and next_state.vtype == "state"
                and next_state.block == state.block):
            raise ValueError("if_: body_fn must return a State of the same block as the input state")
        return self._new("state", "if", (state, cond),
                         {"body_block": body_block, "body": next_state}, None, state.block)

    def _record(self, fn, x):
        """Run a control-flow callable ``fn(self, x)`` with a fresh recording scope active, capturing the
        ops it builds into a sub-block (returned with the value fn produced). The sub-block ops are NOT
        appended to self._values (they belong to the owning control-flow op)."""
        sub = []
        self._recording.append(sub)
        try:
            out = fn(self, x)
        finally:
            self._recording.pop()
        return sub, out

    # --- optional dt bound (spec s18 / ADC-417) ----------------------------------------------------
    def set_dt_bound(self, expr_or_fn):
        """Set an OPTIONAL dt bound for this Program (spec s18). The generated .so exports it as a
        SECOND ABI function (``pops_program_dt_bound``) alongside the macro step; ``step_cfl`` then uses
        ``min(native CFL dt, program dt bound)``. Without a dt bound the native CFL is UNCHANGED.

        @p expr_or_fn is either a callable ``f(P, cfl)`` returning a Scalar (the common form -- it reads
        the runtime ``cfl`` to build e.g. ``cfl * P.hmin() / P.max_wave_speed(U)``), or a Scalar Value
        already built (a fixed bound, cfl-independent). The body is recorded as a scalar sub-program; it
        is NOT run in Python during ``sim.step_cfl`` -- it lowers to C++ that reads the live state /
        reductions. The dt_bound body may read state / fields / scalars only (no commit, no field write):
        a State / RHS / field op in it is rejected."""
        if self._dt_bound is not None:
            raise ValueError("set_dt_bound: a dt bound is already set (set it at most once)")
        if self._recording:
            raise NotImplementedError("set_dt_bound: a dt bound cannot be set inside a control-flow body")
        sub = []
        self._recording.append(sub)
        try:
            if callable(expr_or_fn):
                # The cfl placeholder is a runtime Scalar the body composes with (cfl * hmin / w).
                cfl = self._new("scalar", "cfl", (), {}, "cfl", None)
                result = expr_or_fn(self, cfl)
            else:
                result = expr_or_fn
        finally:
            self._recording.pop()
        if not (isinstance(result, Value) and result.vtype == "scalar"):
            raise ValueError("set_dt_bound: the body must return a Scalar value (e.g. "
                             "cfl * P.hmin() / P.max_wave_speed(U)); got %r" % (result,))
        # The dt_bound is a READ-ONLY scalar program: it may READ state / fields (P.state, P.solve_fields)
        # and produce scalars (reductions, hmin, max_wave_speed, cfl, scalar arithmetic, comparisons),
        # but it must NOT write a field or commit -- a flux / linear_combine / source / projection in it
        # is rejected fail-loud (it would mutate the state during the CFL dt evaluation).
        _DT_BOUND_OPS = frozenset({"state", "solve_fields", "reduce", "compare", "cfl", "hmin",
                                   "max_wave_speed", "scalar_op"})
        for v in sub:
            if v.op not in _DT_BOUND_OPS:
                raise ValueError("set_dt_bound: the dt bound body may read state / fields and compute "
                                 "scalars only (no field write / commit); op '%s' (value '%s') is not "
                                 "allowed" % (v.op, v.name))
        self._dt_bound = (sub, result)
        return result

    def dt_bound(self, fn):
        """Decorator form of `set_dt_bound`: ``@P.dt_bound`` over ``def f(P, cfl): return ...`` records
        the dt bound (spec s18) and returns the function unchanged (so the name stays usable)."""
        self.set_dt_bound(fn)
        return fn

    def has_dt_bound(self):
        """True iff an optional dt bound was set (spec s18)."""
        return self._dt_bound is not None



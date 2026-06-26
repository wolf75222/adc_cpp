"""pops.time Program authoring mixin -- solve / commit / record ops.

Krylov solve_linear, histories, commits, board sugar (fields/define/solve) and records.
"""
from pops.time.program_base import _ProgramConstants
from pops.time.values import StageStateSet, Value, _Affine, _is_field_value


class _ProgramSolve(_ProgramConstants):
    """Krylov solve_linear, histories, commits, board sugar (fields/define/solve) and records."""

    def solve_linear(self, name=None, operator=None, rhs=None, initial_guess=None, method="cg",
                     preconditioner="identity", tol=1e-8, max_iter=None, restart=None):
        """Solve the matrix-free linear system ``operator x = rhs`` with the runtime's Krylov loop and
        return the solution as a scalar_field. The iteration is DYNAMIC (C++-side, inside the loop):
        the IR only carries the operator (its apply lambda), the rhs, the initial guess, and the
        method / tolerance / iteration budget.

          - @p operator: a ``matrix_free_operator`` value (with a ``set_apply`` body);
          - @p rhs: the right-hand side -- a scalar_field, or (MVP) a 1-component State value;
          - @p initial_guess: warm start (defaults to zero);
          - @p method: ``"cg"`` (SPD), ``"bicgstab"`` (general), ``"richardson"``, or ``"gmres"``
            (restarted GMRES(m), the robust choice for a NON-symmetric operator);
          - @p preconditioner: ``"identity"`` only for now;
          - @p tol: relative L2 residual stop (> 0);
          - @p max_iter: iteration budget (REQUIRED, > 0: a dynamic solver loop with no budget is a
            configuration error -- ``pops::*_solve`` itself throws on a non-positive budget);
          - @p restart: GMRES restart length m (a positive int; defaults to 30). Ignored by the other
            methods; passing it to a non-gmres solve is rejected."""
        if not (isinstance(operator, Value) and operator.vtype == "matrix_free_op"):
            raise ValueError("solve_linear: operator must be a matrix_free_operator value")
        if operator.attrs["apply_block"] is None:
            raise ValueError("solve_linear: operator '%s' has no apply; call P.set_apply first"
                             % operator.name)
        if not _is_field_value(rhs):
            raise ValueError("solve_linear: rhs must be a scalar_field or State value (rhs=...)")
        if initial_guess is not None and not _is_field_value(initial_guess):
            raise ValueError("solve_linear: initial_guess must be a scalar_field or State value")
        op_ncomp = int(operator.attrs["ncomp"])
        # The rhs / initial guess must carry at least the operator's component count: the solve runs on
        # an op_ncomp buffer. A scalar_field exposes its ncomp here; a State's n_cons is only known at
        # compile (against the model), so a State is accepted now and checked there.
        for label, fld in (("rhs", rhs), ("initial_guess", initial_guess)):
            if fld is None or fld.vtype != "scalar_field":
                continue
            fld_ncomp = int(fld.attrs.get("ncomp", 1))
            if fld_ncomp < op_ncomp:
                raise ValueError(
                    "solve_linear: %s has %d component(s) but the operator needs %d (a scalar_field "
                    "with ncomp >= the operator ncomp, or a State)" % (label, fld_ncomp, op_ncomp))
        if method not in self._KRYLOV_METHODS:
            raise ValueError("solve_linear: method must be one of %s; got %r"
                             % (sorted(self._KRYLOV_METHODS), method))
        if preconditioner != "identity":
            raise NotImplementedError(
                "solve_linear: only preconditioner='identity' is supported yet (got %r)"
                % preconditioner)
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("solve_linear: tol must be a positive number (got %r)" % (tol,))
        if max_iter is None or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("dynamic solver loops require max_iter")
        # restart is a gmres-only knob; the GMRES(m) basis size. Other methods have no restart concept,
        # so passing one to them is a config error (fail loud rather than silently ignore it).
        if method == "gmres":
            if restart is None:
                restart = self._GMRES_RESTART_DEFAULT
            elif isinstance(restart, bool) or not isinstance(restart, int) or restart <= 0:
                raise ValueError("solve_linear: restart must be a positive integer for gmres (got %r)"
                                 % (restart,))
        elif restart is not None:
            raise ValueError("solve_linear: restart only applies to method='gmres' (got method=%r)"
                             % (method,))
        inputs = (operator, rhs) if initial_guess is None else (operator, rhs, initial_guess)
        return self._new("scalar_field", "solve_linear", inputs,
                         {"method": method, "preconditioner": preconditioner, "tol": float(tol),
                          "max_iter": int(max_iter), "has_guess": initial_guess is not None,
                          "ncomp": op_ncomp,
                          "restart": int(restart) if method == "gmres" else None}, name, rhs.block)

    # --- multistep histories (ADC-406a) ---
    def history(self, name, lag=1):
        """Read a SYSTEM-OWNED history field carried across macro-steps: the value stored @p lag steps
        back (e.g. ``P.history("plasma.R", lag=1)`` is R_{n-1} for Adams-Bashforth). Returns a
        State-typed value usable in the affine algebra. The history is owned by the System (a
        HistoryManager), not the Program, so a later checkpoint slice can serialize it; reading it
        before it has ever been stored is a fail-loud runtime error (it must be written by
        `store_history` every step). @p lag must be a Python int >= 1."""
        if not isinstance(name, str) or not name:
            raise ValueError("history: name must be a non-empty string")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("history: lag must be a Python int >= 1 (got %r)" % (lag,))
        self._histories[name] = max(self._histories.get(name, 0), lag)
        return self._new("state", "history", (), {"history": name, "lag": int(lag)}, name, None)

    def store_history(self, name, value):
        """Store @p value (a State/RHS field) into the CURRENT slot of history @p name at the end of the
        step (rotated to lag 1 on the next step). A multistep scheme stores its current RHS so the next
        step can read it back via `history`. The history is System-owned; this is a side-effecting op
        (no value). @p value must be a State/RHS field of the Program."""
        if not isinstance(name, str) or not name:
            raise ValueError("store_history: name must be a non-empty string")
        if not _is_field_value(value):
            raise ValueError("store_history: value must be a State/RHS field (got %r)" % (value,))
        if value.prog is not self:
            raise ValueError("store_history: the value belongs to a different Program")
        self._histories.setdefault(name, 1)
        return self._new("state", "store_history", (value,), {"history": name}, name, value.block)

    def commit(self, block, state):
        """Replace the current state of ``block`` with ``state`` at the end of the step. Each block
        is committed AT MOST once; read-only blocks need no commit.

        @p state is normally a State value; a 1-component model's conservative state doubles as a
        scalar field, so a ``scalar_field`` (e.g. a ``solve_linear`` solution) is also accepted and
        copied back into the block state at commit (the final ``ctx.lincomb`` in the lowered body)."""
        if not (isinstance(state, Value) and state.vtype in ("state", "scalar_field")):
            raise ValueError("commit: a State (or scalar_field) value is required")
        if state.prog is not self:
            raise ValueError("commit: the State value belongs to a different Program")
        if block in self._commits:
            raise ValueError("block '%s' committed more than once" % block)
        self._commits[block] = state

    def commits(self):
        """Map of committed block -> committed State value (copy)."""
        return dict(self._commits)

    # --- board-like sugar (Spec 3): T.define / T.fields / T.solve / T.commit_many ---
    # These lower to the SAME primitive ops as the P.call / linear_combine /
    # solve_local_linear / commit style; they are blackboard notation, not a new IR.
    def op(self, name):
        """Return a callable board handle for a bound operator: ``expl = P.op("explicit_rate")``
        then ``expl(U, fields)`` builds ``P.call("explicit_rate", U, fields)``."""
        def _handle(*args, value_name=None):
            return self.call(name, *args, name=value_name)
        _handle.__name__ = str(name)
        return _handle

    def fields(self, name, from_state=None, from_states=None, from_state_set=None, operator=None):
        """Board sugar for a field solve. Lowers to ``P.call(operator, ...)`` when a named operator
        is bound, else to ``P.solve_fields`` (single state) or ``P.solve_fields_from_blocks``."""
        if from_state_set is not None:
            states = from_state_set.states()
        elif from_states is not None:
            states = list(from_states)
        elif from_state is not None:
            states = [from_state]
        else:
            raise ValueError("fields: provide from_state=, from_states= or from_state_set=")
        named = operator is not None and operator != "fields_from_state"
        if len(states) == 1:
            if named and self._registry is not None:
                return self.call(operator, states[0], name=name)
            return self.solve_fields(name, states[0])
        if named and self._registry is not None:
            return self.call(operator, *states, name=name)
        return self.solve_fields_from_blocks(states, name=name)

    def define(self, name, value):
        """Board sugar to name a value. An affine combination of states materializes via
        ``linear_combine``; a ``rate(U) == <expr>`` equation keeps its right-hand side; any other
        Value is named in place."""
        from pops import math as _bm
        if isinstance(value, _bm.Equation):
            if not isinstance(value.lhs, _bm.TimeDerivative):
                raise ValueError("define(%r): an equation must read 'rate(U) == <rate expression>'"
                                 % (name,))
            value = value.rhs
        if isinstance(value, _Affine):
            return self.linear_combine(name, value)
        if isinstance(value, Value):
            value.name = name
            return value
        raise TypeError(
            "define(%r): expected a Value, an affine combination, or a rate equation; got %r"
            % (name, value))

    def solve(self, name, equation):
        """Board sugar for an implicit local solve ``(I -/+ a*L) @ unknown("x") == rhs``.

        Lowers to ``linear_combine`` (if the rhs is an affine combination) then
        ``solve_local_linear``; identical IR to writing those two calls by hand.
        """
        from pops import math as _bm
        if not isinstance(equation, _bm.Equation):
            raise TypeError("solve(%r): expected '(I - dt*C) @ unknown(\"x\") == rhs'" % (name,))
        lhs, rhs = equation.lhs, equation.rhs
        if not isinstance(lhs, _bm.OpApply):
            raise ValueError("solve(%r): left-hand side must be 'operator @ unknown(name)'" % (name,))
        if isinstance(rhs, _Affine):
            rhs = self.linear_combine(name + "_rhs", rhs)
        elif not (isinstance(rhs, Value) and rhs.vtype == "state"):
            raise ValueError("solve(%r): right-hand side must be a State or an affine of States"
                             % (name,))
        return self.solve_local_linear(name=name, operator=lhs.operator, rhs=rhs)

    def commit_many(self, mapping, fields=None):
        """Atomically commit several coupled blocks (Spec 3). ALL entries are validated before any
        commit, so a partial or double commit of a coupled group is rejected as a unit and no block
        is left half-committed. ``fields`` (optional) is validated as a coherent FieldContext but is
        RESERVED: the IR commit has no fields slot yet (the runtime association lands with ADC-457)."""
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("commit_many: a non-empty {block: State} mapping is required")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("commit_many: fields must be a FieldContext from solve_fields")
        for block, state in mapping.items():
            if not (isinstance(state, Value) and state.vtype in ("state", "scalar_field")):
                raise ValueError("commit_many: block %r needs a State value" % (block,))
            if state.prog is not self:
                raise ValueError("commit_many: the State for %r belongs to a different Program"
                                 % (block,))
            if block in self._commits:
                raise ValueError("block '%s' committed more than once" % (block,))
        for block, state in mapping.items():
            self._commits[block] = state

    def state_set(self, name, mapping):
        """Build a :class:`StageStateSet` -- a coherent set of stage states for a field solve."""
        return StageStateSet(name, mapping)

    def record(self, name, value):
        """Record a scalar diagnostic (board sugar over :meth:`record_scalar`).

        ``value`` is a Program scalar -- a reduction result such as ``P.sum(U)`` or
        ``P.norm2(U)`` (the runtime value of a generic invariant). The automatic
        reduction of an arbitrary ``integral(expr)`` over a per-cell expression is a
        follow-up (it needs the scheduler / a generated reduction kernel, ADC-458)."""
        if not (isinstance(value, Value) and value.vtype == "scalar"):
            raise ValueError(
                "record(%r): value must be a Program scalar (e.g. P.sum / P.norm2); got %r"
                % (name, value))
        return self.record_scalar(name, value)

    def check_invariant(self, name, before=None, after=None, tolerance=1e-10):
        """Record the drift of a generic invariant between two stages (board diagnostic).

        ``before`` / ``after`` are Program scalars (reduction results); the recorded
        diagnostic ``"<name>_drift"`` is ``after - before``. ``tolerance`` is carried as
        metadata for a later assertion stage (the scheduled runtime check is ADC-458)."""
        if not (isinstance(before, Value) and before.vtype == "scalar"
                and isinstance(after, Value) and after.vtype == "scalar"):
            raise ValueError(
                "check_invariant(%r): before/after must be Program scalars" % (name,))
        drift = after - before
        out = self.record_scalar(name + "_drift", drift)
        out.attrs["tolerance"] = float(tolerance)
        return out

    # --- inspection / debug (Spec 3 section 33): show the lowering ---

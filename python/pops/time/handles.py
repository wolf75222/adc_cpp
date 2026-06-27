"""pops.time typed temporal-version handles (Spec 5 sec.5.3.1).

Typed handles for the temporal versions of one block state -- the current state ``U.n``,
the named stages ``U.stage(k)``, the end-of-step ``U.next``, and the lagged history
``U.prev(lag)`` -- expressed as SUGAR over the existing SSA IR. There is NO new IR and NO
Python runtime data here: every handle lowers to the EXACT primitive ops the positional
``P.state`` / ``P.linear_combine`` / ``P.commit`` / ``P.history`` / ``P.store_history``
style already builds, so a scheme written with handles produces a byte-identical
``_ir_hash``.

A :class:`TimeState` is a family of handles; a :class:`_Version` is one typed handle that
PROXIES the affine algebra to its resolved :class:`pops.time.values.Value` once
``T.define`` has lowered it. Before definition, using a stage in arithmetic raises a clear
error; the handles themselves carry no ndarray.
"""
from pops.time.history import CopyCurrent


def _proxy_value(handle):
    """Return the resolved Value behind a handle, or raise the handle's undefined error."""
    value = handle._resolved()
    if value is None:
        handle._undefined()
    return value


class _Version:
    """A typed temporal-version handle for one block state (a stage, or ``.next``).

    Carries only its parent :class:`TimeState`, its ``key`` (an int stage index, a stage
    name, or the string ``"next"``), and -- once :meth:`TimeState` lowers it via
    ``T.define`` -- the resolved :class:`pops.time.values.Value`. It holds NO runtime data.

    Before definition, every arithmetic use raises so a stage cannot be read before it is
    written (SSA single assignment). After definition, ``__add__`` / ``__radd__`` /
    ``__sub__`` / ``__rsub__`` / ``__mul__`` / ``__rmul__`` / ``__truediv__`` delegate to
    the resolved Value's affine algebra, so a handle composes exactly like the Value would.
    """

    def __init__(self, timestate, key):
        self._timestate = timestate
        self._key = key
        self._value = None  # set by TimeState._define once T.define lowers this version

    @property
    def block(self):
        return self._timestate.block

    @property
    def value(self):
        """The resolved Value (raises if the version was never defined)."""
        return _proxy_value(self)

    def _resolved(self):
        return self._value

    def _as_value(self):
        """The resolved Value, for the State-accepting boundaries (``solve_fields`` / ``rhs``)."""
        return _proxy_value(self)

    def _undefined(self):
        raise ValueError(
            "stage %r is undefined (define it with T.define first)" % (self._key,))

    # --- affine algebra: delegate to the resolved Value (else fail loud) -----------------
    def __add__(self, other):
        return _proxy_value(self).__add__(other)

    def __radd__(self, other):
        return _proxy_value(self).__radd__(other)

    def __sub__(self, other):
        return _proxy_value(self).__sub__(other)

    def __rsub__(self, other):
        return _proxy_value(self).__rsub__(other)

    def __mul__(self, other):
        return _proxy_value(self).__mul__(other)

    def __rmul__(self, other):
        return _proxy_value(self).__rmul__(other)

    def __truediv__(self, other):
        return _proxy_value(self).__truediv__(other)

    def __repr__(self):
        state = "undefined" if self._value is None else ("#%d" % self._value.id)
        return "<_Version %s.%s key=%r %s>" % (
            self._timestate.block, self._timestate.name, self._key, state)


class _Prev:
    """The lagged-history accessor of a :class:`TimeState` (``U.prev`` / ``U.prev(lag)``).

    ``U.prev`` is this object; calling it ``U.prev(lag)`` returns the history Value at that
    lag (``P.history("<block>.<name>", lag)``). The bare ``U.prev`` PROXIES the lag-1
    history through the affine algebra, so ``U.prev`` reads like ``U.prev(1)``. It is a
    READ-ONLY handle: ``T.define(U.prev, ...)`` is rejected (the ring is produced by the
    history policy, not by an SSA definition). Carries no runtime data.
    """

    def __init__(self, timestate):
        self._timestate = timestate

    def __call__(self, lag=1):
        return self._timestate._history(lag)

    def _lag1(self):
        return self._timestate._history(1)

    def _as_value(self):
        """The lag-1 history Value, for the State-accepting boundaries (bare ``U.prev``)."""
        return self._lag1()

    # --- affine algebra: bare U.prev behaves as U.prev(1) --------------------------------
    def __add__(self, other):
        return self._lag1().__add__(other)

    def __radd__(self, other):
        return self._lag1().__radd__(other)

    def __sub__(self, other):
        return self._lag1().__sub__(other)

    def __rsub__(self, other):
        return self._lag1().__rsub__(other)

    def __mul__(self, other):
        return self._lag1().__mul__(other)

    def __rmul__(self, other):
        return self._lag1().__rmul__(other)

    def __truediv__(self, other):
        return self._lag1().__truediv__(other)

    def __repr__(self):
        return "<_Prev %s.%s>" % (self._timestate.block, self._timestate.name)


class TimeState:
    """A family of typed temporal-version handles for one block state (Spec 5 sec.5.3.1).

    Built by ``P.state("U", block="plasma")``. Exposes:

      - ``.n``        -- the current state ``U^n`` (a cached Value; read-only);
      - ``.stage(k)`` -- a cached :class:`_Version` for stage ``k`` (int or str), undefined
                         until ``T.define``;
      - ``.next``     -- a cached :class:`_Version` for the end-of-step state ``U^{n+1}``;
      - ``.prev`` / ``.prev(lag)`` -- the lagged history, requiring ``keep_history`` first.

    The handles carry NO runtime arrays: ``.n`` is an SSA Value (an IR node, not data), the
    versions hold only a key, and ``.prev`` holds only its parent.
    """

    def __init__(self, program, block, name="U"):
        if not isinstance(block, str) or not block:
            raise ValueError("TimeState: block must be a non-empty string")
        self.program = program
        self.block = block
        self.name = name or "U"
        self._n = None
        self._stages = {}
        self._next = None
        self._prev = _Prev(self)
        self._history_depth = None
        self._cold_start = None

    # --- the current state U^n (read-only) -----------------------------------------------
    @property
    def n(self):
        """The current conservative state ``U^n`` (cached Value, byte-identical to the
        positional ``P.state(block)``)."""
        if self._n is None:
            self._n = self.program.state(self.block)
        return self._n

    # --- stages / next (SSA versions, defined via T.define) ------------------------------
    def stage(self, key):
        """A cached :class:`_Version` handle for stage ``key`` (an int or str). Undefined
        until lowered with ``T.define``; reusing the same key returns the same handle."""
        if not isinstance(key, (int, str)) or isinstance(key, bool):
            raise ValueError("TimeState.stage: key must be an int or a str (got %r)" % (key,))
        if key not in self._stages:
            self._stages[key] = _Version(self, key)
        return self._stages[key]

    @property
    def next(self):
        """A cached :class:`_Version` for the end-of-step state ``U^{n+1}``."""
        if self._next is None:
            self._next = _Version(self, "next")
        return self._next

    # --- lagged history (requires keep_history) ------------------------------------------
    @property
    def prev(self):
        """The lagged-history accessor: ``U.prev`` (lag 1) or ``U.prev(lag)``."""
        return self._prev

    def _history(self, lag):
        """Resolve the history Value at ``lag``, after validating ``keep_history`` set up a
        deep-enough ring. Lowers to the existing ``P.history`` op."""
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("TimeState.prev: lag must be a Python int >= 1 (got %r)" % (lag,))
        if self._history_depth is None:
            raise ValueError(
                "%s.prev requires keep_history first: declare T.keep_history(%s, depth=...) "
                "before reading a lagged state" % (self.block, self.name))
        if lag > self._history_depth:
            raise ValueError(
                "%s.prev(%d) exceeds the kept history depth %d; raise the keep_history depth"
                % (self.block, lag, self._history_depth))
        return self.program.history(self._history_name(), lag)

    def _history_name(self):
        return "%s.%s" % (self.block, self.name)

    # --- lowering hooks driven by Program.define / Program.keep_history ------------------
    def _is_n(self, version):
        return version is self._n

    def _define(self, version, value):
        """Lower a stage/next definition through the existing ``define`` path and bind the
        resulting Value onto the handle (SSA single assignment)."""
        if version is self._n:
            raise ValueError("current state is read-only in Program")
        if isinstance(version, _Prev):
            raise ValueError("history is produced by the history policy")
        if not isinstance(version, _Version) or version._timestate is not self:
            raise ValueError("T.define: target is not a version handle of this TimeState")
        if version._value is not None:
            raise ValueError("SSA version already defined")
        gen_name = "%s_%s_%s" % (self.block, self.name, version._key)
        out = self.program.define(gen_name, value)
        version._value = out
        return out

    def _keep_history(self, depth, cold_start=None):
        """Record the history depth / cold-start policy and lower a ``store_history`` of the
        current state so the ring is populated each step."""
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("keep_history: depth must be a Python int >= 1 (got %r)" % (depth,))
        if cold_start is None:
            cold_start = CopyCurrent()
        self._history_depth = depth
        self._cold_start = cold_start
        return self.program.store_history(self._history_name(), self.n)

    def __repr__(self):
        return "TimeState(block=%r, name=%r)" % (self.block, self.name)

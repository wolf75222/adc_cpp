"""pops.time scheduler annotations (Spec 3 unified scheduler).

``Schedule`` is an inert IR annotation deciding WHEN a node is due and what to do when it is
not; the module helpers (``always`` / ``every`` / ``when`` / ``on_start`` / ``on_end`` /
``subcycle``) build the kinds. Authoring only.
"""
class Schedule:
    """When a Program node is due, and what to do when it is not (Spec 3 unified scheduler).

    A Schedule is an inert IR annotation recorded on a node (``Value.attrs['schedule']``). The
    ``kind`` decides WHEN the node is due (``always`` every step, ``every(N)``, ``when(cond)``,
    ``on_start`` / ``on_end``, ``subcycle``); the ``policy`` decides what happens when it is NOT
    due (``recompute`` the default, ``hold`` the cached value, ``skip``, ``zero``,
    ``accumulate_dt``, or ``error``). Build a kind with the module helpers and set the policy by
    chaining: ``every(10).hold()``.

    Only ``always()`` runs at ``sim.step`` today: the runtime that honors a non-trivial schedule
    (the typed cache, ``accumulate_dt``, the checkpoint) is the C++ part of ADC-458, so a node
    carrying a non-always schedule is recorded and inspectable but refuses to lower (it is never
    silently ignored). See ``docs/sphinx/reference/program-scheduler.md``.
    """

    _KINDS = ("always", "every", "when", "on_start", "on_end", "subcycle")
    _POLICIES = ("recompute", "hold", "skip", "zero", "accumulate_dt", "error")
    # policies that reuse a stored value, so the operator must be cacheable
    _CACHING = ("hold", "accumulate_dt")

    def __init__(self, kind, policy="recompute", **params):
        if kind not in Schedule._KINDS:
            raise ValueError("schedule kind %r must be one of %s"
                             % (kind, ", ".join(Schedule._KINDS)))
        if policy not in Schedule._POLICIES:
            raise ValueError("schedule policy %r must be one of %s"
                             % (policy, ", ".join(Schedule._POLICIES)))
        self.kind = kind
        self.policy = policy
        self.params = dict(params)

    def is_always(self):
        """True for the default cadence (every step, recompute) -- the only schedule that lowers."""
        return self.kind == "always" and self.policy == "recompute"

    def needs_cache(self):
        """True if the policy reuses a stored value (so the operator must be cacheable)."""
        return self.policy in Schedule._CACHING

    def _with_policy(self, policy):
        return Schedule(self.kind, policy=policy, **self.params)

    def recompute(self):
        return self._with_policy("recompute")

    def hold(self):
        return self._with_policy("hold")

    def skip(self):
        return self._with_policy("skip")

    def zero(self):
        return self._with_policy("zero")

    def accumulate_dt(self):
        return self._with_policy("accumulate_dt")

    def error(self):
        return self._with_policy("error")

    def __repr__(self):
        if self.kind == "every":
            base = "every(%r)" % (self.params.get("n"),)
        elif self.kind == "subcycle":
            base = "subcycle(%r)" % (self.params.get("count"),)
        elif self.kind == "when":
            base = "when(...)"
        else:
            base = "%s()" % self.kind
        return base if self.policy == "recompute" else "%s.%s()" % (base, self.policy)


def always():
    """Due every step, recomputed -- the default cadence (the only schedule that runs today)."""
    return Schedule("always")


def every(n):
    """Due every ``n`` macro-steps (``n`` a positive int)."""
    if not (isinstance(n, int) and n > 0):
        raise ValueError("every(n): n must be a positive int, got %r" % (n,))
    return Schedule("every", n=n)


def when(cond):
    """Due when the runtime condition ``cond`` holds (a Program Bool value or a callable)."""
    return Schedule("when", cond=cond)


def on_start():
    """Due only at the first step."""
    return Schedule("on_start")


def on_end():
    """Due only at the last step."""
    return Schedule("on_end")


def subcycle(count, dt=None):
    """Structured sub-cycling: ``count`` inner steps (of ``dt`` each, default ``macro_dt/count``)."""
    if not (isinstance(count, int) and count > 0):
        raise ValueError("subcycle(count): count must be a positive int, got %r" % (count,))
    return Schedule("subcycle", count=count, dt=dt)


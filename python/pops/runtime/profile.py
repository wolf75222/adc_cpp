"""pops.runtime.profile -- typed profiling surface (Spec 5 sec.12.5, criteria 41-44).

A typed replacement for the stringly ``sim.enable_profiling()`` / ``sim.profile_report()``
dance. Two pieces:

* :class:`Profile` -- a typed profiling *level* (``Profile.Basic()`` / ``Profile.Advanced()``),
  NOT ``profile="advanced"``. It is an inert descriptor: it carries no timers and computes
  nothing in Python; it only declares WHICH native counters a level wants surfaced.
* :class:`PerformanceSummary` -- a printable wrapper around the native profile report. It parses
  the report the C++ :class:`pops::runtime::program::Profiler` produces (a per-scope timing table
  plus integer counters) into a structured dict and exposes typed views:
  :meth:`~PerformanceSummary.by_program_node` / :meth:`~PerformanceSummary.by_native_brick` /
  :meth:`~PerformanceSummary.by_solver` / :meth:`~PerformanceSummary.by_memory`. When a measure is
  not available on the current build (the heavy per-brick / scheduler counters are Kokkos-gated and
  only move under a compiled ``.so`` step), the view DECLARES it unavailable honestly rather than
  fabricating a zero.

The off-by-default contract (criterion 44): profiling adds no heavy timers unless explicitly
enabled. The native ``enable_profiling`` already gates this -- a plain run leaves the profiler
disabled. :meth:`System.profile` (the context manager in :mod:`pops.runtime.system`) is the typed
front door: it enables on ``__enter__`` and disables on ``__exit__``, and exposes
``prof.summary()`` -> :class:`PerformanceSummary`.

This module is a pure typed/parsing wrapper: it imports neither ``_pops`` nor numpy. The native
extension is reached only through the :class:`System` instance the context manager is bound to.
"""

import json
import os


# Native scope-name conventions the C++ Profiler emits (program_context.hpp / system.cpp):
# coarse System phases, per-Program-node scopes ("node:<name>"), and the integer counters.
_COARSE_PHASES = ("step", "field_solve")
_NODE_PREFIX = "node:"
# A field-solve Program node ("node:solve_fields...") is the solver-attributable scope on the
# native path; the coarse "field_solve" phase is the System-level elliptic solve.
_SOLVER_SCOPES = ("field_solve",)
_SOLVER_NODE_HINT = "solve_fields"
# Memory counters (program_context.hpp count_scratch): allocation count + the largest single
# scratch buffer in bytes. A live-bytes total is deliberately NOT tracked by the native runtime.
_MEMORY_COUNTERS = ("scratch_allocs", "scratch_peak_bytes")
# Scheduler / cache counters that only move under a compiled .so step body (Kokkos/ROMEO); absent
# (the honest zero) on the native host path.
_ADVANCED_COUNTERS = ("cache_hits", "cache_misses", "nodes_due", "nodes_skipped")

# POPS_PROFILE: map sim.profile() called with NO argument to a default level.
_ENV_VAR = "POPS_PROFILE"
_ENV_OFF = ("", "0", "off", "false", "no", "none")
_ENV_ADVANCED = ("advanced", "2", "full")


class Profile:
    """A typed profiling level (Spec 5 sec.12.5). Inert: it carries no timers.

    Use the named constructors rather than a string flag::

        with sim.profile(pops.Profile.Basic()) as prof:
            sim.run(0.1)
        print(prof.summary())

    ``Basic`` surfaces the coarse phase timings + the kernel/step counters; ``Advanced`` also asks
    for the per-program-node timings and the scheduler/memory counters (which only populate under a
    compiled step on a Kokkos build -- declared unavailable, never faked, otherwise).
    """

    __slots__ = ("level",)

    #: The two recognised levels.
    _LEVELS = ("basic", "advanced")

    def __init__(self, level="basic"):
        if level not in self._LEVELS:
            raise ValueError(
                "Profile level must be one of %s (got %r)" % (self._LEVELS, level))
        self.level = level

    @classmethod
    def Basic(cls):
        """Coarse phase timings + step / kernel counters."""
        return cls("basic")

    @classmethod
    def Advanced(cls):
        """Per-program-node timings + scheduler / memory counters (Kokkos-gated; honest about gaps)."""
        return cls("advanced")

    @property
    def advanced(self):
        """True for the Advanced level (asks for the per-node / scheduler / memory views)."""
        return self.level == "advanced"

    @classmethod
    def from_env(cls, default=None):
        """Resolve the level from ``POPS_PROFILE`` (sim.profile() with no argument).

        Unset / ``0`` / ``off`` -> @p default (a Basic() when @p default is None); ``advanced`` /
        ``full`` -> Advanced(); anything else -> Basic(). Returns None when the env asks for OFF and
        no @p default is given, so the caller can leave profiling disabled.
        """
        raw = os.environ.get(_ENV_VAR)
        if raw is None or raw.strip().lower() in _ENV_OFF:
            return default
        if raw.strip().lower() in _ENV_ADVANCED:
            return cls.Advanced()
        return cls.Basic()

    def __eq__(self, other):
        return isinstance(other, Profile) and other.level == self.level

    def __hash__(self):
        return hash(("Profile", self.level))

    def __repr__(self):
        return "Profile.%s()" % self.level.capitalize()


def _parse_report(report):
    """Parse the native ``profile_report()`` string into a structured dict.

    The C++ Profiler renders (profiler.hpp ``report()``)::

        Profiler report (total 0.010849 s, 2 scopes)
          step  count=2  total=0.007229s  mean=0.003614s  min=...s  max=...s
          field_solve  count=1  total=...s  ...
        counters:  steps=2  kernels=3

    Returns ``{"scopes": {name: {count,total_s,mean_s,min_s,max_s}}, "counters": {name: int},
    "total_s": float}``. An empty / unrecognised report yields empty tables (never raises).
    """
    scopes = {}
    counters = {}
    total_s = 0.0
    for line in (report or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Profiler report"):
            total_s = _extract_float(stripped, "total ")
            continue
        if stripped.startswith("counters:"):
            for tok in stripped[len("counters:"):].split():
                if "=" in tok:
                    key, val = tok.split("=", 1)
                    counters[key] = _to_int(val)
            continue
        # A scope line: "<name>  count=..  total=..s  mean=..s  min=..s  max=..s".
        if "count=" in stripped:
            name = stripped.split("  ", 1)[0].strip()
            fields = {}
            for tok in stripped.split():
                if "=" in tok:
                    key, val = tok.split("=", 1)
                    if key in ("count",):
                        fields["count"] = _to_int(val)
                    elif key in ("total", "mean", "min", "max"):
                        fields["%s_s" % key] = _to_float(val.rstrip("s"))
            if name:
                scopes[name] = fields
    return {"scopes": scopes, "counters": counters, "total_s": total_s}


def _extract_float(text, after):
    """Best-effort: the float token that follows @p after in @p text (else 0.0)."""
    idx = text.find(after)
    if idx < 0:
        return 0.0
    return _to_float(text[idx + len(after):].split()[0]) if text[idx + len(after):].split() else 0.0


def _to_float(token):
    try:
        return float(token)
    except (TypeError, ValueError):
        return 0.0


def _to_int(token):
    try:
        return int(token)
    except (TypeError, ValueError):
        return 0


class _Unavailable:
    """A sentinel view: a measure the current build does not surface (declared, not faked)."""

    __slots__ = ("measure", "reason")

    def __init__(self, measure, reason):
        self.measure = measure
        self.reason = reason

    @property
    def available(self):
        return False

    def to_dict(self):
        return {"available": False, "measure": self.measure, "reason": self.reason, "entries": {}}

    def __bool__(self):
        return False

    def __repr__(self):
        return "<unavailable %s: %s>" % (self.measure, self.reason)


class PerformanceSummary:
    """A printable, typed wrapper around the native profile report (Spec 5 criteria 41-43).

    Built from the string :meth:`System.profile_report` returns (and the :class:`Profile` level the
    run requested). It exposes the report as a structured dict (:meth:`to_dict` / :meth:`to_json`)
    and typed views: :meth:`by_program_node`, :meth:`by_native_brick`, :meth:`by_solver`,
    :meth:`by_memory`. Views read the parsed native tables; a view the build does not surface
    returns an :class:`_Unavailable` sentinel (``bool(view) is False``) rather than a faked zero.
    """

    def __init__(self, report, profile=None):
        self._report_text = report or ""
        self._profile = profile if profile is not None else Profile.Basic()
        self._parsed = _parse_report(self._report_text)

    # ---- raw access -------------------------------------------------------------------------
    @property
    def profile(self):
        """The :class:`Profile` level the run requested."""
        return self._profile

    @property
    def raw_report(self):
        """The exact string the native profiler returned (the source of truth)."""
        return self._report_text

    def scopes(self):
        """All timed scopes: ``{name: {count, total_s, mean_s, min_s, max_s}}``."""
        return dict(self._parsed["scopes"])

    def counters(self):
        """All integer counters: ``{name: int}``."""
        return dict(self._parsed["counters"])

    def total_s(self):
        """Sum of every scope's total wall-clock time (seconds)."""
        return self._parsed["total_s"]

    # ---- typed views ------------------------------------------------------------------------
    def by_program_node(self):
        """Per-program-node timings (the ``node:<name>`` scopes the compiled step emits).

        Keys are the bare node names (``rhs2``, ``solve_fields1``, ...). Empty on a native step
        (no compiled Program); populated under a compiled ``.so`` step.
        """
        nodes = {name[len(_NODE_PREFIX):]: dict(fields)
                 for name, fields in self._parsed["scopes"].items()
                 if name.startswith(_NODE_PREFIX)}
        return nodes

    def by_native_brick(self):
        """Per-native-brick timings.

        The native runtime times Program nodes and coarse phases, not individual bricks: there is no
        per-brick scope to read. Declared unavailable rather than faked (the per-brick granularity is
        a documented follow-up wired through the compiled ProgramContext).
        """
        return _Unavailable(
            "by_native_brick",
            "native runtime times program nodes / phases, not individual bricks")

    def by_solver(self):
        """Solver-attributable timings: the elliptic field-solve phase + any solve_fields node.

        Reads the coarse ``field_solve`` phase and the ``node:solve_fields*`` program nodes. Empty
        when no field solve ran under profiling.
        """
        out = {}
        for name, fields in self._parsed["scopes"].items():
            if name in _SOLVER_SCOPES:
                out[name] = dict(fields)
            elif name.startswith(_NODE_PREFIX) and _SOLVER_NODE_HINT in name:
                out[name[len(_NODE_PREFIX):]] = dict(fields)
        return out

    def by_memory(self):
        """Scratch-memory counters: allocation count + the largest single scratch buffer (bytes).

        Reads ``scratch_allocs`` / ``scratch_peak_bytes`` (program_context.hpp ``count_scratch``).
        These move only under a compiled step on a Kokkos build; on a native host step neither
        counter is created, so this view declares itself unavailable rather than faking a 0.
        """
        present = {name: self._parsed["counters"][name]
                   for name in _MEMORY_COUNTERS if name in self._parsed["counters"]}
        if not present:
            return _Unavailable(
                "by_memory",
                "scratch memory counters populate only under a compiled Kokkos step")
        return present

    # ---- serialisation ----------------------------------------------------------------------
    def to_dict(self):
        """The full structured report: level + scopes + counters + total, plus the typed views.

        ``by_native_brick`` / ``by_memory`` serialise their availability honestly (an unavailable
        view records ``{"available": False, "reason": ...}``).
        """
        return {
            "profile": self._profile.level,
            "total_s": self.total_s(),
            "scopes": self.scopes(),
            "counters": self.counters(),
            "views": {
                "by_program_node": self.by_program_node(),
                "by_native_brick": _view_to_dict(self.by_native_brick()),
                "by_solver": self.by_solver(),
                "by_memory": _view_to_dict(self.by_memory()),
            },
        }

    def to_json(self, path=None):
        """Serialise :meth:`to_dict` to JSON. Writes to @p path when given; returns the JSON string."""
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            with open(path, "w", encoding="ascii") as handle:
                handle.write(text)
        return text

    # ---- printable --------------------------------------------------------------------------
    def __str__(self):
        if not self._parsed["scopes"] and not self._parsed["counters"]:
            return "PerformanceSummary(%s): no profiling data recorded" % self._profile.level
        lines = ["PerformanceSummary (%s, total %.6f s, %d scopes)"
                 % (self._profile.level, self.total_s(), len(self._parsed["scopes"]))]
        for name, fields in self._parsed["scopes"].items():
            lines.append("  %-24s count=%d total=%.6fs mean=%.6fs"
                         % (name, fields.get("count", 0),
                            fields.get("total_s", 0.0), fields.get("mean_s", 0.0)))
        if self._parsed["counters"]:
            counters = "  ".join("%s=%d" % (k, v) for k, v in self._parsed["counters"].items())
            lines.append("counters: %s" % counters)
        return "\n".join(lines)

    def print(self):
        """Print the human-readable summary (``print(summary)`` sugar)."""
        print(str(self))

    def __repr__(self):
        return "PerformanceSummary(profile=%r, scopes=%d, counters=%d)" % (
            self._profile.level, len(self._parsed["scopes"]), len(self._parsed["counters"]))


def _view_to_dict(view):
    """Serialise a typed view: a dict passes through; an _Unavailable records its availability."""
    if isinstance(view, _Unavailable):
        return view.to_dict()
    return {"available": True, "entries": view}


__all__ = ["Profile", "PerformanceSummary"]

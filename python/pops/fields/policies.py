"""pops.fields.policies -- inert field-solve cadence policies (Spec 5 sec.13.11.2).

A field solve does not always run every macro-step: a self-consistent Poisson coupling
may be held over several steps and only recomputed on a cadence, or recomputed only when a
residual exceeds a threshold. The CADENCE (WHEN to solve) is a :class:`pops.time.Schedule`;
the POLICY here decides what happens on a step where the solve is NOT due:

* :class:`HoldPrevious` -- reuse the cached field from the last solve (requires a cacheable
  output, so the result can be stored and re-read);
* :class:`Recompute` -- always recompute, never reuse a cached value (the default cadence
  behaviour, surfaced as an explicit typed policy).

Both are inert typed :class:`~pops.descriptors.Descriptor` objects: they declare their
cacheability / math-admissibility through :meth:`capabilities` / :meth:`requirements` and
compute nothing. The C++ runtime / codegen consumes the recorded cadence; see
:meth:`pops.fields.FieldProblem.solve`.
"""
from pops.descriptors import Descriptor


class FieldSolvePolicy(Descriptor):
    """Base of the field-solve cadence policies (what to do when a solve is not due).

    A policy is inert: it only declares whether it reuses a cached field (so the field
    output must be cacheable) through :meth:`capabilities` / :meth:`requirements`. The
    runtime honours it; the policy computes nothing.
    """

    category = "field_solve_policy"


class HoldPrevious(FieldSolvePolicy):
    """Hold the previously solved field on a step where the solve is not due.

    Reuses the cached field from the last solve, so it REQUIRES a cacheable output (the
    field must be storable and re-readable between solves). Declares ``reuses_cache`` so a
    consumer can check math-admissibility before the runtime is touched.
    """

    def requirements(self):
        return {"cacheable_output": True}

    def capabilities(self):
        return {"reuses_cache": True, "recomputes": False}

    def options(self):
        return {"policy": "hold_previous"}

    def __repr__(self):
        return "HoldPrevious()"


class Recompute(FieldSolvePolicy):
    """Recompute the field on every due step; never reuse a cached value.

    The explicit typed form of the default cadence behaviour. Requires nothing of the
    output (no cache is read), and declares ``recomputes`` so the distinction from
    :class:`HoldPrevious` is inspectable.
    """

    def requirements(self):
        return {}

    def capabilities(self):
        return {"reuses_cache": False, "recomputes": True}

    def options(self):
        return {"policy": "recompute"}

    def __repr__(self):
        return "Recompute()"


__all__ = ["FieldSolvePolicy", "HoldPrevious", "Recompute"]

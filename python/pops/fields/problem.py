"""pops.fields.problem -- the typed elliptic field-problem descriptor (Spec 5 sec.5.5).

:class:`FieldProblem` is the inert, typed declaration of a field solve: the unknown, the
governing :class:`pops.math.Equation`, the input/coefficient/boundary/nullspace objects,
the outputs and the solver brick. It declares its requirements / capabilities / options
and validates that it was built from a real :class:`~pops.math.Equation` (not a Python
bool or some other object) and that a solver was provided -- before the runtime is ever
touched.

It computes nothing; codegen / runtime consume the descriptor.
"""
from pops.descriptors import Availability, Descriptor
from pops.math import Equation

from .policies import FieldSolvePolicy


def _summarize(side, limit=60):
    """A short repr of one equation side, truncated to keep the print() summary small."""
    text = repr(side)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class SolveCadence(Descriptor):
    """The inert record of a field-solve cadence: a schedule + a not-due policy.

    Recorded on a :class:`FieldProblem` by :meth:`FieldProblem.solve` and surfaced by the
    problem's :meth:`~FieldProblem.inspect`. It pairs the typed :class:`pops.time.Schedule`
    (WHEN to solve) with the typed :class:`~pops.fields.policies.FieldSolvePolicy` (what to
    do when the solve is not due). It is authoring metadata only: it carries the schedule and
    policy, it does NOT lower into the Program / codegen. The runtime wiring is a
    codegen-adjacent follow-up (see :meth:`FieldProblem.solve`).
    """

    category = "solve_cadence"

    def __init__(self, schedule, policy):
        self.schedule = schedule
        self.policy = policy

    def options(self):
        return {"schedule": repr(self.schedule), "policy": self.policy.name}

    def requirements(self):
        return dict(self.policy.requirements())

    def capabilities(self):
        return dict(self.policy.capabilities())

    def inspect(self):
        info = super().inspect()
        info["schedule"] = repr(self.schedule)
        info["policy"] = self.policy.inspect()
        return info

    def __repr__(self):
        return "SolveCadence(schedule=%r, policy=%r)" % (self.schedule, self.policy)


class FieldProblem(Descriptor):
    """A typed elliptic field problem: an unknown solved from an :class:`~pops.math.Equation`.

    ``FieldProblem(unknown=phi, equation=(-laplacian(phi) == rhs), solver=GeometricMG())``.
    The ``inputs`` / ``coefficients`` / ``bcs`` / ``nullspace`` / ``outputs`` / ``postprocess``
    fields are typed descriptors (from :mod:`pops.fields`); they are stored and surfaced, not
    interpreted, here. :meth:`validate` rejects a non-:class:`~pops.math.Equation` equation
    (a Python bool produced by ``==`` on plain values is the common mistake) and a missing
    solver.
    """

    category = "field_problem"

    def __init__(self, name=None, unknown=None, equation=None, inputs=(),
                 coefficients=None, bcs=(), nullspace=None, outputs=None,
                 postprocess=None, solver=None):
        self._name = None if name is None else str(name)
        self.unknown = unknown
        self.equation = equation
        self.inputs = tuple(inputs)
        self.coefficients = coefficients
        self.bcs = tuple(bcs)
        self.nullspace = nullspace
        self.outputs = outputs
        self.postprocess = postprocess
        self.solver = solver
        # The inert field-solve cadence recorded by solve(); None until authored.
        self.cadence = None

    @property
    def name(self):
        return self._name if self._name is not None else type(self).__name__

    def options(self):
        return {"name": self._name,
                "unknown": getattr(self.unknown, "name", repr(self.unknown))
                if self.unknown is not None else None,
                "n_inputs": len(self.inputs),
                "n_bcs": len(self.bcs),
                "has_coefficients": self.coefficients is not None,
                "has_nullspace": self.nullspace is not None,
                "solver": getattr(self.solver, "name", self.solver),
                "has_cadence": self.cadence is not None}

    def requirements(self):
        req = {"equation": True, "solver": True}
        if self.nullspace is not None:
            req["nullspace"] = getattr(self.nullspace, "name", repr(self.nullspace))
        return req

    def capabilities(self):
        return {"elliptic": True}

    def available(self, context=None):
        if not isinstance(self.equation, Equation):
            return Availability.no(
                "%s needs a pops.math.Equation; got %r" % (self.name, type(self.equation).__name__),
                missing=["equation"])
        if self.solver is None:
            return Availability.no("%s needs a solver" % self.name, missing=["solver"])
        return Availability.yes()

    def validate(self, context=None):
        if isinstance(self.equation, bool):
            raise TypeError(
                "%s: equation must be a pops.math.Equation built from PoPS operators "
                "(e.g. -laplacian(phi) == rhs); got a Python bool, which usually means '==' "
                "was applied to plain values instead of PoPS expressions" % self.name)
        if not isinstance(self.equation, Equation):
            raise TypeError(
                "%s: equation must be a pops.math.Equation; got %r"
                % (self.name, type(self.equation).__name__))
        if self.solver is None:
            raise ValueError("%s: a solver must be provided" % self.name)
        return True

    def solve(self, schedule, policy):
        """Record an inert field-solve cadence (a schedule + a not-due policy).

        Pairs a typed :class:`pops.time.Schedule` (WHEN to solve -- e.g. ``every(4)`` to
        solve every fourth macro-step, or ``when(cond)``) with a typed
        :class:`~pops.fields.policies.FieldSolvePolicy`
        (:class:`~pops.fields.policies.HoldPrevious` /
        :class:`~pops.fields.policies.Recompute`) deciding what happens on a step where the
        solve is not due. The pair is stored as :attr:`cadence` and surfaced by
        :meth:`inspect`; the method returns ``self`` so it chains after construction.

        This is AUTHORING metadata only. It deliberately does NOT lower the cadence into the
        Program / codegen: honouring a non-trivial field-solve cadence at runtime (the cached
        field carry, the residual gate) is the codegen-adjacent follow-up. The cadence is
        recorded and inspectable, never silently executed.

        Args:
            schedule: A typed :class:`pops.time.Schedule` (built with ``every`` / ``when`` /
                ``always`` / ...). A bare ``int`` or ``str`` is rejected with a clear
                ``TypeError`` -- the cadence is a typed object, not a free string or count.
            policy: A typed :class:`~pops.fields.policies.FieldSolvePolicy`
                (:class:`~pops.fields.policies.HoldPrevious` or
                :class:`~pops.fields.policies.Recompute`). A string such as ``"hold"`` is
                rejected with a clear ``TypeError``.

        Returns:
            FieldProblem: ``self``, so the call chains after construction.

        Raises:
            TypeError: When @p schedule is not a :class:`pops.time.Schedule` or @p policy is
                not a :class:`~pops.fields.policies.FieldSolvePolicy`.
        """
        # Lazy import: keep pops.fields free of a module-scope pops.time edge (the acyclic
        # layering of test_import_graph; fields.aux defers its mesh import the same way).
        from pops.time.schedule import Schedule

        if not isinstance(schedule, Schedule):
            raise TypeError(
                "%s.solve(schedule=...): schedule must be a typed pops.time.Schedule "
                "(e.g. pops.time.every(4) or pops.time.when(cond)); got %r. A bare int / "
                "string is not a cadence." % (self.name, schedule))
        if not isinstance(policy, FieldSolvePolicy):
            raise TypeError(
                "%s.solve(policy=...): policy must be a typed field-solve policy "
                "(pops.fields.HoldPrevious() or pops.fields.Recompute()); got %r"
                % (self.name, policy))
        self.cadence = SolveCadence(schedule, policy)
        return self

    def inspect(self):
        info = super().inspect()
        info["equation"] = self._equation_summary()
        info["bcs"] = [getattr(b, "name", repr(b)) for b in self.bcs]
        info["outputs"] = getattr(self.outputs, "name", self.outputs)
        info["cadence"] = self.cadence.inspect() if self.cadence is not None else None
        return info

    def _equation_summary(self):
        if not isinstance(self.equation, Equation):
            return repr(self.equation)
        return "%s == %s" % (_summarize(self.equation.lhs), _summarize(self.equation.rhs))

    def __str__(self):
        solver = getattr(self.solver, "name", self.solver)
        outputs = getattr(self.outputs, "name", self.outputs)
        return "%s [%s] %s | bcs=%d | solver=%s | outputs=%s" % (
            self.name, self.category, self._equation_summary(), len(self.bcs), solver, outputs)


__all__ = ["FieldProblem", "SolveCadence"]

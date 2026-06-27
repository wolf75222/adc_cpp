"""Spec 5 (sec.13.11.2): the inert field-solve cadence (schedule + not-due policy).

ADC-501 lets a :class:`~pops.fields.PoissonProblem` / :class:`~pops.fields.FieldProblem`
RECORD a solve cadence: a typed :class:`pops.time.Schedule` (WHEN to solve) paired with a
typed field-solve :class:`~pops.fields.policies.FieldSolvePolicy`
(:class:`~pops.fields.HoldPrevious` / :class:`~pops.fields.Recompute`) deciding what happens
on a step where the solve is not due. These exercise the authoring + validation surface:
``solve()`` records the cadence (surfaced by ``inspect()``), rejects a bare int / string
schedule and a string policy, and the policies themselves inspect / validate. Pure Python;
needs only ``import pops`` (nothing here computes on a grid). The cadence is authoring
metadata: it deliberately does NOT lower into the Program / codegen here.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.math import laplacian, unknown  # noqa: E402
from pops.ir.expr import Var  # noqa: E402
from pops.fields import (  # noqa: E402
    FieldProblem, PoissonProblem, HoldPrevious, Recompute, FieldSolvePolicy)
from pops.fields.problem import SolveCadence  # noqa: E402
from pops.time import every, when, always  # noqa: E402
from pops.time.schedule import Schedule  # noqa: E402


def _poisson():
    """A real -laplacian(phi) == rho PoissonProblem with a dummy solver object."""
    phi = unknown("phi")
    rho = Var("rho", "cons")
    return PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho), solver=object())


# --- the policies are inert typed descriptors -------------------------------------------
def test_hold_previous_is_a_field_solve_policy():
    policy = HoldPrevious()
    assert isinstance(policy, FieldSolvePolicy)
    assert policy.category == "field_solve_policy"
    # HoldPrevious reuses the cached field, so it requires a cacheable output.
    assert policy.requirements()["cacheable_output"] is True
    assert policy.capabilities()["reuses_cache"] is True
    assert policy.capabilities()["recomputes"] is False
    assert policy.validate() is True


def test_recompute_is_a_field_solve_policy():
    policy = Recompute()
    assert isinstance(policy, FieldSolvePolicy)
    # Recompute reads no cache, so it requires nothing of the output.
    assert policy.requirements() == {}
    assert policy.capabilities()["recomputes"] is True
    assert policy.capabilities()["reuses_cache"] is False
    assert policy.validate() is True


def test_policy_repr_is_short():
    assert repr(HoldPrevious()) == "HoldPrevious()"
    assert repr(Recompute()) == "Recompute()"


def test_policy_inspect_surfaces_cacheability():
    info = HoldPrevious().inspect()
    assert info["category"] == "field_solve_policy"
    assert info["requirements"]["cacheable_output"] is True
    assert info["capabilities"]["reuses_cache"] is True


def test_policies_exported_from_fields():
    assert pops.fields.HoldPrevious is HoldPrevious
    assert pops.fields.Recompute is Recompute
    # also reachable via the policies submodule.
    assert pops.fields.policies.HoldPrevious is HoldPrevious


# --- solve() records the cadence and surfaces it via inspect() --------------------------
def test_solve_records_cadence_and_inspects():
    prob = _poisson()
    assert prob.cadence is None
    returned = prob.solve(schedule=every(4), policy=HoldPrevious())
    # solve() chains (returns self) and records a typed cadence.
    assert returned is prob
    assert isinstance(prob.cadence, SolveCadence)
    assert isinstance(prob.cadence.schedule, Schedule)
    assert isinstance(prob.cadence.policy, HoldPrevious)

    info = prob.inspect()
    assert info["cadence"] is not None
    assert info["cadence"]["schedule"] == repr(every(4))
    assert info["cadence"]["policy"]["category"] == "field_solve_policy"
    assert info["cadence"]["policy"]["capabilities"]["reuses_cache"] is True
    # options() flags that a cadence is recorded.
    assert info["options"]["has_cadence"] is True


def test_solve_with_recompute_and_when_schedule():
    prob = _poisson()
    prob.solve(schedule=when("cond"), policy=Recompute())
    assert prob.cadence.policy.capabilities()["recomputes"] is True
    assert prob.cadence.schedule.kind == "when"


def test_inspect_cadence_none_before_solve():
    info = _poisson().inspect()
    assert info["cadence"] is None
    assert info["options"]["has_cadence"] is False


def test_field_problem_solve_also_works():
    # The method lives on the FieldProblem base, not only the Poisson shortcut.
    phi = unknown("phi")
    prob = FieldProblem(unknown=phi, equation=(-laplacian(phi) == Var("rho", "cons")),
                        solver=object())
    prob.solve(schedule=always(), policy=Recompute())
    assert isinstance(prob.cadence, SolveCadence)


# --- solve() rejects un-typed schedule / policy with a clear TypeError -------------------
def test_solve_rejects_bare_int_schedule():
    prob = _poisson()
    with pytest.raises(TypeError, match="schedule must be a typed pops.time.Schedule"):
        prob.solve(schedule=4, policy=HoldPrevious())
    # the rejected call left no cadence behind.
    assert prob.cadence is None


def test_solve_rejects_string_schedule():
    prob = _poisson()
    with pytest.raises(TypeError, match="schedule must be a typed pops.time.Schedule"):
        prob.solve(schedule="every-4", policy=HoldPrevious())


def test_solve_rejects_string_policy():
    prob = _poisson()
    with pytest.raises(TypeError, match="policy must be a typed field-solve policy"):
        prob.solve(schedule=every(4), policy="hold")
    assert prob.cadence is None


def test_solve_cadence_inspect_and_repr():
    cadence = SolveCadence(every(2), HoldPrevious())
    assert "schedule" in cadence.inspect()
    assert cadence.inspect()["policy"]["category"] == "field_solve_policy"
    assert "SolveCadence" in repr(cadence)
    # the cadence forwards the policy's requirements (cacheable output).
    assert cadence.requirements()["cacheable_output"] is True


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

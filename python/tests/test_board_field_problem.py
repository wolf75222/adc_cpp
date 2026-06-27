"""Spec 5 (sec.5.1 / sec.9.6): the inert pops.physics.Model.field_problem shortcut.

These exercise the ergonomic typed-object entry point ``Model.field_problem(name, equation,
...)`` on the blackboard physics model: it CONSTRUCTS and RETURNS an inspectable
:class:`pops.fields.PoissonProblem` (or a :class:`pops.fields.FieldProblem` when coefficients
are present) describing ``-laplacian(phi) == rhs``, and records it on the model's authoring
state so ``m.inspect()`` surfaces it. The shortcut is INERT: it lowers ONLY to the descriptor;
it does not touch the dsl model, the operator graph, codegen or the runtime. Pure Python: only
pops.physics / pops.math / pops.fields are needed (nothing computes on a grid).
"""
import sys

import pytest

physics = pytest.importorskip("pops.physics")
amath = pytest.importorskip("pops.math")
fields = pytest.importorskip("pops.fields")


def _model_with_state():
    """A board model with a 3-component state and a solved field phi."""
    m = physics.Model("plasma")
    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    phi = m.field("phi")
    return m, U, phi


def test_field_problem_returns_an_inspectable_poisson_problem():
    from pops.math import laplacian, grad

    m, U, phi = _model_with_state()
    rho, mx, my = U
    prob = m.field_problem(
        "poisson",
        equation=(-laplacian(phi) == rho),
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver="geometric_mg")
    assert isinstance(prob, fields.PoissonProblem)
    # The equation / solver / outputs inspect correctly off the returned descriptor.
    info = prob.inspect()
    assert "Laplacian" in info["equation"]
    assert info["options"]["solver"] == "geometric_mg"
    assert prob.outputs is not None and set(prob.outputs) == {"phi", "grad_x", "grad_y"}
    assert prob.capabilities()["poisson"] is True


def test_field_problem_validates_and_is_available_with_a_solver():
    from pops.math import laplacian

    m, U, phi = _model_with_state()
    rho, mx, my = U
    prob = m.field_problem("poisson", equation=(-laplacian(phi) == rho), solver="geometric_mg")
    assert prob.validate() is True
    assert prob.available().ok


def test_field_problem_without_solver_is_flagged_unavailable():
    from pops.math import laplacian

    m, U, phi = _model_with_state()
    rho, mx, my = U
    prob = m.field_problem("poisson", equation=(-laplacian(phi) == rho), solver=None)
    # The descriptor's own contract flags the missing solver (the shortcut stays inert).
    assert not prob.available()
    with pytest.raises(ValueError):
        prob.validate()


def test_field_problem_carries_bcs():
    from pops.math import laplacian

    m, U, phi = _model_with_state()
    rho, mx, my = U
    wall = fields.bcs.Dirichlet(value=0.0)
    prob = m.field_problem("poisson", equation=(-laplacian(phi) == rho),
                           solver="geometric_mg", bcs=[wall])
    assert prob.bcs == (wall,)
    assert prob.options()["n_bcs"] == 1


def test_field_problem_with_coefficients_is_a_field_problem():
    from pops.math import laplacian
    from pops.fields.coefficients import ScalarCoefficient

    m, U, phi = _model_with_state()
    rho, mx, my = U
    prob = m.field_problem("screened", equation=(-laplacian(phi) == rho),
                           solver="geometric_mg", coefficients=ScalarCoefficient("eps"))
    # With coefficients present the descriptor is the general FieldProblem (not a PoissonProblem).
    assert isinstance(prob, fields.FieldProblem)
    assert not isinstance(prob, fields.PoissonProblem)
    assert prob.coefficients is not None


def test_field_problem_rejects_a_non_equation():
    m, U, phi = _model_with_state()
    with pytest.raises(TypeError):
        m.field_problem("poisson", equation="-laplacian(phi) == rho", solver="geometric_mg")


def test_model_inspect_surfaces_the_field_problem():
    from pops.math import laplacian

    m, U, phi = _model_with_state()
    rho, mx, my = U
    m.field_problem("poisson", equation=(-laplacian(phi) == rho), solver="geometric_mg")
    snapshot = m.inspect()
    assert snapshot["name"] == "plasma"
    assert "U" in snapshot["states"]
    assert "poisson" in snapshot["field_problems"]
    # The recorded entry is the descriptor's own inspect() dict.
    assert "Laplacian" in snapshot["field_problems"]["poisson"]["equation"]


def test_field_problem_does_not_perturb_the_dsl_elliptic_rhs():
    # The shortcut is INERT: unlike solve_field it must not register an elliptic rhs on the
    # underlying dsl model (no lowering / operator-graph mutation).
    from pops.math import laplacian

    m, U, phi = _model_with_state()
    rho, mx, my = U
    m.field_problem("poisson", equation=(-laplacian(phi) == rho), solver="geometric_mg")
    assert m.dsl._m._elliptic is None


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this
# module so the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

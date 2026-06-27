"""Spec 5 (sec.9.2, ADC-491): the board-node elliptic-operator algebra.

The field-equation forms Spec 5 itself uses must be AUTHORABLE as inert, inspectable IR:

  -laplacian(phi) + k*phi == rhs            (screened Poisson)
  -div(eps*grad(phi)) + kappa*phi == rhs    (anisotropic / variable coefficient, sec.9.2)

These build elliptic operator terms (Reaction / CoeffGradient / DivCoeffGrad / EllipticSum);
nothing computes in Python -- the C++ elliptic solver executes. This test covers the IR
construction, the principal-kind inspection, and the Poisson-family validation.
"""
import sys

import pytest

pytest.importorskip("pops")

from pops.math import (  # noqa: E402
    laplacian, grad, div, unknown, principal_kinds,
    Reaction, CoeffGradient, DivCoeffGrad, EllipticSum, Laplacian, Divergence, Equation)
from pops.ir.expr import Var, RateTerm  # noqa: E402
from pops.fields.coefficients import ScalarCoefficient, ReactionCoefficient  # noqa: E402
from pops.fields import (  # noqa: E402
    PoissonProblem, ScreenedPoissonProblem, AnisotropicPoissonProblem)


def test_reaction_term_constructs():
    phi = unknown("phi")
    react = 0.5 * phi
    assert isinstance(react, Reaction)
    assert principal_kinds(react) == {"reaction"}
    # coefficient (not just a scalar) is accepted
    react2 = ReactionCoefficient("kappa") * phi
    assert isinstance(react2, Reaction)


def test_coeff_gradient_and_div():
    phi = unknown("phi")
    eps = ScalarCoefficient("eps")
    cg = eps * grad(phi)
    assert isinstance(cg, CoeffGradient)
    op = div(cg)
    assert isinstance(op, DivCoeffGrad)
    assert principal_kinds(op) == {"div_coeff_grad"}


def test_div_grad_is_laplacian():
    phi = unknown("phi")
    op = div(grad(phi))  # no coefficient -> the constant-coefficient Laplacian
    assert isinstance(op, Laplacian)
    assert principal_kinds(op) == {"laplacian"}


def test_div_of_a_flux_stays_hyperbolic():
    # Regression: div of a model flux handle must still build a hyperbolic flux term.
    op = div("default_flux")
    assert isinstance(op, Divergence)
    assert isinstance(op, RateTerm)  # composes into a rate equation, not an elliptic sum


def test_screened_form_constructs_and_inspects():
    phi = unknown("phi")
    rhs = Var("charge", "cons")
    lhs = -laplacian(phi) + 0.5 * phi
    assert isinstance(lhs, EllipticSum)
    assert principal_kinds(lhs) == {"laplacian", "reaction"}
    eq = (lhs == rhs)
    assert isinstance(eq, Equation)
    assert "EllipticSum" in repr(lhs) and "Reaction" in repr(lhs)


def test_spec_9_2_headline_form_constructs():
    # -div(eps*grad(phi)) + kappa*phi == charge  (Spec 5 sec.9.2 headline)
    phi = unknown("phi")
    eps = ScalarCoefficient("eps")
    kappa = ReactionCoefficient("kappa")
    charge = Var("charge", "cons")
    eq = (-div(eps * grad(phi)) + kappa * phi == charge)
    assert isinstance(eq, Equation)
    assert principal_kinds(eq.lhs) == {"div_coeff_grad", "reaction"}


def test_poisson_accepts_principal_operator():
    phi = unknown("phi")
    rhs = Var("charge", "cons")
    PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rhs), solver=object()).validate()
    eps = ScalarCoefficient("eps")
    PoissonProblem(unknown=phi, equation=(-div(eps * grad(phi)) == rhs),
                   solver=object()).validate()


def test_screened_requires_reaction():
    phi = unknown("phi")
    rhs = Var("charge", "cons")
    # missing reaction -> rejected
    bad = ScreenedPoissonProblem(unknown=phi, equation=(-laplacian(phi) == rhs), solver=object())
    with pytest.raises(ValueError, match="reaction"):
        bad.validate()
    # with reaction -> ok
    ok = ScreenedPoissonProblem(unknown=phi, equation=(-laplacian(phi) + 0.5 * phi == rhs),
                                solver=object())
    assert ok.validate() is True


def test_anisotropic_requires_div_coeff_grad():
    phi = unknown("phi")
    rhs = Var("charge", "cons")
    eps = ScalarCoefficient("eps")
    # plain laplacian -> rejected (no variable-coefficient principal operator)
    bad = AnisotropicPoissonProblem(unknown=phi, equation=(-laplacian(phi) == rhs),
                                    solver=object())
    with pytest.raises(ValueError, match="div"):
        bad.validate()
    # div(eps*grad(phi)) -> ok
    ok = AnisotropicPoissonProblem(unknown=phi, equation=(-div(eps * grad(phi)) == rhs),
                                   solver=object())
    assert ok.validate() is True


def test_terms_are_inert_no_runtime_data():
    # The elliptic nodes carry references/coefficients, not arrays; they compute nothing.
    phi = unknown("phi")
    react = ScalarCoefficient("eps") * phi
    assert not hasattr(react, "eval")           # no host evaluator
    assert react.field is phi                   # a reference, not a value
    assert react.coeff.name == "eps"            # the coefficient descriptor, not an array
    assert "Reaction" in repr(react)            # inspectable


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

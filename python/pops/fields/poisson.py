"""pops.fields.poisson -- Poisson-family field-problem shortcuts (Spec 5 sec.5.5).

Thin subclasses of :class:`pops.fields.problem.FieldProblem` that name the common elliptic
shapes:

* :class:`PoissonProblem` -- ``-laplacian(phi) == rhs`` (the standard self-consistent field);
* :class:`ScreenedPoissonProblem` -- adds a zeroth-order reaction term (Debye screening);
* :class:`AnisotropicPoissonProblem` -- a tensor / anisotropic principal coefficient.

They only refine the declared capabilities and the validation; the runtime / codegen treat
them as a :class:`FieldProblem`.
"""
from pops.math import Equation, principal_kinds
from .problem import FieldProblem

# An elliptic LHS must carry a principal operator: a Laplacian or a div(coeff*grad).
_PRINCIPAL_OPERATORS = {"laplacian", "div_coeff_grad"}


class PoissonProblem(FieldProblem):
    """A standard Poisson problem ``-laplacian(phi) == rhs``.

    Refines :class:`FieldProblem` by requiring the equation's principal operator to be an
    elliptic Laplacian / div(coeff*grad) (so a non-elliptic equation is rejected up front).
    """

    category = "poisson_problem"

    def capabilities(self):
        caps = super().capabilities()
        caps["poisson"] = True
        return caps

    def validate(self, context=None):
        super().validate(context)
        if isinstance(self.equation, Equation):
            if not (principal_kinds(self.equation.lhs) & _PRINCIPAL_OPERATORS):
                raise ValueError(
                    "%s: a Poisson problem expects an elliptic principal operator on the "
                    "left-hand side (e.g. -laplacian(phi) == rhs or "
                    "-div(eps*grad(phi)) == rhs); got %r" % (self.name, self.equation.lhs))
        return True


class ScreenedPoissonProblem(PoissonProblem):
    """A screened Poisson problem ``-laplacian(phi) + k*phi == rhs`` (Debye screening).

    Requires a zeroth-order reaction term (``k*phi``) on top of the principal operator.
    """

    category = "screened_poisson_problem"

    def capabilities(self):
        caps = super().capabilities()
        caps["screened"] = True
        return caps

    def validate(self, context=None):
        super().validate(context)
        if isinstance(self.equation, Equation):
            if "reaction" not in principal_kinds(self.equation.lhs):
                raise ValueError(
                    "%s: a screened Poisson expects a zeroth-order reaction term "
                    "(e.g. -laplacian(phi) + k*phi == rhs); got %r"
                    % (self.name, self.equation.lhs))
        # Reject only a solver that declares it cannot serve a screened operator (criterion 11).
        self._require_solver_capability("screened", "a screened operator", "GeometricMG()")
        return True


class AnisotropicPoissonProblem(PoissonProblem):
    """An anisotropic Poisson problem ``-div(A grad phi) == rhs`` (variable / tensor coefficient).

    Requires a ``div(coeff*grad(phi))`` principal operator (the variable-coefficient form).

    NOTE: the form is authorable + validated here, but lowering a variable / tensor
    coefficient in the elliptic codegen is the coordinated follow-up (ADC-491); today the
    native elliptic operator solves a constant-coefficient ``div(eps grad)``.
    """

    category = "anisotropic_poisson_problem"

    def capabilities(self):
        caps = super().capabilities()
        caps["anisotropic"] = True
        return caps

    def validate(self, context=None):
        super().validate(context)
        if isinstance(self.equation, Equation):
            if "div_coeff_grad" not in principal_kinds(self.equation.lhs):
                raise ValueError(
                    "%s: an anisotropic Poisson expects a div(coeff*grad(phi)) principal "
                    "operator (e.g. -div(eps*grad(phi)) == rhs); got %r"
                    % (self.name, self.equation.lhs))
        # Reject only a solver that declares it cannot serve an anisotropic operator (criterion 11).
        self._require_solver_capability(
            "anisotropic", "an anisotropic operator", "GeometricMG()")
        return True


__all__ = ["PoissonProblem", "ScreenedPoissonProblem", "AnisotropicPoissonProblem"]

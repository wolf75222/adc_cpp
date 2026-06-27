"""pops.fields.poisson -- Poisson-family field-problem shortcuts (Spec 5 sec.5.5).

Thin subclasses of :class:`pops.fields.problem.FieldProblem` that name the common elliptic
shapes:

* :class:`PoissonProblem` -- ``-laplacian(phi) == rhs`` (the standard self-consistent field);
* :class:`ScreenedPoissonProblem` -- adds a zeroth-order reaction term (Debye screening);
* :class:`AnisotropicPoissonProblem` -- a tensor / anisotropic principal coefficient.

They only refine the declared capabilities and the validation; the runtime / codegen treat
them as a :class:`FieldProblem`.
"""
from pops.math import Equation, Laplacian
from .problem import FieldProblem


def _has_laplacian(node):
    """True if @p node (or its negation, or a leading-coefficient scaling) is a Laplacian."""
    return isinstance(node, Laplacian)


class PoissonProblem(FieldProblem):
    """A standard Poisson problem ``-laplacian(phi) == rhs``.

    Refines :class:`FieldProblem` by requiring the equation's principal operator to be a
    Laplacian (so a non-elliptic equation is rejected up front).
    """

    category = "poisson_problem"

    def capabilities(self):
        caps = super().capabilities()
        caps["poisson"] = True
        return caps

    def validate(self, context=None):
        super().validate(context)
        if isinstance(self.equation, Equation) and not _has_laplacian(self.equation.lhs):
            raise ValueError(
                "%s: a Poisson problem expects a laplacian form on the left-hand side "
                "(e.g. -laplacian(phi) == rhs); got %r" % (self.name, self.equation.lhs))
        return True


class ScreenedPoissonProblem(PoissonProblem):
    """A screened Poisson problem ``-laplacian(phi) + k*phi == rhs`` (Debye screening)."""

    category = "screened_poisson_problem"

    def capabilities(self):
        caps = super().capabilities()
        caps["screened"] = True
        return caps


class AnisotropicPoissonProblem(PoissonProblem):
    """An anisotropic Poisson problem ``-div(A grad phi) == rhs`` (tensor coefficient)."""

    category = "anisotropic_poisson_problem"

    def capabilities(self):
        caps = super().capabilities()
        caps["anisotropic"] = True
        return caps


__all__ = ["PoissonProblem", "ScreenedPoissonProblem", "AnisotropicPoissonProblem"]

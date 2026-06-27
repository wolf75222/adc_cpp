"""Spec 5 EARLY-VALIDATION gaps (sec.7 / sec.8.6; criteria 11 / 31; epic ADC-479).

Three validations the spec requires BEFORE runtime, each implemented under the OVERRIDING
no-false-positive discipline (only reject a KNOWN / declared incompatibility; a valid problem
must still pass):

* GAP 1 (sec.7, criterion 11): a :class:`pops.fields.FieldProblem` cross-checks the chosen
  elliptic solver's declared capabilities against the problem kind. A screened / anisotropic
  Poisson paired with a solver that declares ``supports_screened`` / ``supports_anisotropic``
  KNOWN-False (the spectral ``FFT``) is refused; ``GeometricMG`` (variable-epsilon) and a
  capability-less solver pass.
* GAP 2 (sec.7, criterion 11): a reconstruction whose DECLARED ghost depth exceeds an EXPLICIT
  block halo is refused (WENO5 needs 3; an explicit depth-2 block is too thin). The native
  runtime grows the halo to match the scheme, so WENO5 with no explicit constraint -- and MUSCL
  at depth 2 -- pass.
* GAP 3 (sec.8.6, criterion 31): a :class:`pops.mesh.amr.Refine` validated against a model
  rejects a bogus subject and lists the declared subjects; a real declared role / component
  validates, and a context that advertises no subjects (or no context) is not rejected.

Pure Python; needs only ``import pops`` (nothing computes on a grid). The validations read
descriptor metadata and run nothing.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.math import div, grad, laplacian, unknown  # noqa: E402
from pops.ir.expr import Var  # noqa: E402
from pops.fields import (  # noqa: E402
    AnisotropicPoissonProblem, PoissonProblem, ScreenedPoissonProblem)
from pops.fields.coefficients import ScalarCoefficient  # noqa: E402
from pops.solvers.elliptic import FFT, GeometricMG  # noqa: E402
from pops.numerics.reconstruction import (  # noqa: E402
    FirstOrder, MUSCL, WENO5, required_ghost_depth, validate_ghost_depth)
from pops.runtime._bricks_scheme import FiniteVolume  # noqa: E402
from pops.mesh.amr import Refine, TagUnion, _declared_subjects  # noqa: E402
from pops.mesh.layouts import AMR  # noqa: E402
from pops.mesh import CartesianMesh  # noqa: E402


# ----------------------------------------------------------------------------------------
# GAP 1 -- FieldProblem incompatible-solver validation (sec.7, criterion 11)
# ----------------------------------------------------------------------------------------

def _screened_problem(solver):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    return ScreenedPoissonProblem(
        unknown=phi, equation=(-laplacian(phi) + 0.5 * phi == rho), solver=solver)


def _anisotropic_problem(solver):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    eps = ScalarCoefficient("eps")
    return AnisotropicPoissonProblem(
        unknown=phi, equation=(-div(eps * grad(phi)) == rho), solver=solver)


def test_screened_with_fft_is_rejected_with_actionable_message():
    # The FFT solver declares supports_screened KNOWN-False -> refused before runtime.
    with pytest.raises(ValueError) as exc:
        _screened_problem(FFT()).validate()
    msg = str(exc.value)
    assert "does not support a screened operator" in msg
    assert "supports_screened is False" in msg
    assert "pops.solvers.elliptic.GeometricMG()" in msg


def test_anisotropic_with_fft_is_rejected_with_actionable_message():
    with pytest.raises(ValueError) as exc:
        _anisotropic_problem(FFT()).validate()
    msg = str(exc.value)
    assert "does not support an anisotropic operator" in msg
    assert "supports_anisotropic is False" in msg
    assert "pops.solvers.elliptic.GeometricMG()" in msg


def test_screened_with_geometric_mg_is_fine():
    # NO FALSE POSITIVE: GeometricMG declares supports_variable_epsilon -> it serves the
    # screened reaction term; supports_screened=False is not a real incompatibility.
    assert GeometricMG().capabilities()["supports_screened"] is False
    assert GeometricMG().capabilities()["supports_variable_epsilon"] is True
    assert _screened_problem(GeometricMG()).validate() is True


def test_anisotropic_with_geometric_mg_is_fine():
    assert _anisotropic_problem(GeometricMG()).validate() is True


def test_capabilityless_solver_is_not_rejected():
    # NO FALSE POSITIVE: a solver that exposes no capabilities() dict (a bare object, an
    # external brick) has an ABSENT capability, not a declared-False one -> never rejected.
    assert _screened_problem(object()).validate() is True
    assert _anisotropic_problem(object()).validate() is True


def test_plain_poisson_with_fft_is_not_rejected():
    # NO FALSE POSITIVE: a plain Poisson needs neither screened nor anisotropic, so the FFT
    # solver (a real periodic constant-coefficient route) is not refused by this cross-check.
    phi = unknown("phi")
    rho = Var("rho", "cons")
    prob = PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho), solver=FFT())
    assert prob.validate() is True


# ----------------------------------------------------------------------------------------
# GAP 2 -- WENO5 ghost-depth insufficiency (sec.7, criterion 11)
# ----------------------------------------------------------------------------------------

def test_declared_required_ghost_depths():
    # The declared per-scheme requirement: WENO5 >= 3, MUSCL >= 2, first-order 1.
    assert required_ghost_depth(WENO5()) == 3
    assert required_ghost_depth(MUSCL()) == 2
    assert required_ghost_depth(FirstOrder()) == 1
    assert required_ghost_depth("weno5") == 3
    assert required_ghost_depth("minmod") == 2


def test_weno5_on_explicit_depth2_block_is_rejected():
    # An EXPLICIT depth-2 block is too thin for WENO5's 3-cell stencil -> clear error.
    with pytest.raises(ValueError) as exc:
        FiniteVolume(weno5=True).validate(ghost_depth=2, block="plasma")
    msg = str(exc.value)
    assert "WENO5 requires ghost_depth >= 3" in msg
    assert "block 'plasma' has ghost_depth=2" in msg
    # The same check fires on the raw token form too.
    with pytest.raises(ValueError, match="WENO5 requires ghost_depth >= 3"):
        validate_ghost_depth("weno5", available=2, block="plasma")


def test_muscl_on_depth2_block_is_fine():
    # NO FALSE POSITIVE: a second-order MUSCL scheme fits the default 2-cell halo.
    assert FiniteVolume(minmod=True).validate(ghost_depth=2) is True
    assert validate_ghost_depth("minmod", available=2) is True


def test_weno5_without_explicit_constraint_is_not_rejected():
    # NO FALSE POSITIVE: the native runtime grows the block halo to match the scheme
    # (block_n_ghost("weno5") == 3), so WENO5 on a default block is a VALID problem.
    assert FiniteVolume(weno5=True).validate() is True
    assert validate_ghost_depth(WENO5()) is True
    assert validate_ghost_depth("weno5") is True


def test_undeclared_reconstruction_is_not_rejected():
    # NO FALSE POSITIVE: an unknown token has no declared requirement -> never rejected.
    assert validate_ghost_depth("custom_user_scheme", available=1) is True


# ----------------------------------------------------------------------------------------
# GAP 3 -- Refine role-existence validation (sec.8.6, criterion 31)
# ----------------------------------------------------------------------------------------

class _FakeModel:
    """A minimal model advertising its declared subjects the way HyperbolicModel does."""

    cons_names = ["rho", "rho_u", "rho_v", "E"]
    cons_roles = None
    aux_extra_names = ["B_z"]

    def state_space(self):
        from pops.model.spaces import StateSpace
        return StateSpace(components=self.cons_names,
                          roles={"rho": "Density", "E": "Energy"})


def test_declared_subjects_collects_components_roles_and_aux():
    subjects = _declared_subjects(_FakeModel())
    # Components, the named aux, and BOTH the role keys and role values are legal subjects.
    for name in ("rho", "rho_u", "E", "B_z", "Density", "Energy"):
        assert name in subjects, "%r missing from declared subjects %s" % (name, sorted(subjects))


def test_refine_on_declared_role_validates():
    model = _FakeModel()
    # A real state component and a real physical role both validate.
    assert Refine.on("rho").above(0.05).validate(model) is True
    assert Refine.on("Density").above(0.05).validate(model) is True
    assert Refine.on("B_z").above(0.05).validate(model) is True


def test_refine_on_bogus_role_is_rejected_listing_declared_subjects():
    model = _FakeModel()
    with pytest.raises(ValueError) as exc:
        Refine.on("bogus_role").above(0.05).validate(model)
    msg = str(exc.value)
    assert "bogus_role" in msg
    assert "is not a declared subject" in msg
    # The message lists the real declared subjects so the user can fix it.
    assert "rho" in msg and "Density" in msg


def test_refine_without_context_self_validates_only():
    # NO FALSE POSITIVE: with no model the subject check DEFERS (it cannot know the roles);
    # only the predicate/threshold shape is checked here.
    assert Refine.on("bogus_role").above(0.05).validate() is True
    # An incomplete criterion is still rejected on shape, with or without a context.
    with pytest.raises(ValueError, match="incomplete"):
        Refine.on("rho").validate()


def test_refine_against_surfaceless_context_is_not_rejected():
    # NO FALSE POSITIVE: a context that advertises no subject surface is not rejected.
    assert Refine.on("bogus_role").above(0.05).validate(object()) is True


def test_tag_union_forwards_the_model_context():
    model = _FakeModel()
    # A union of REAL subjects validates against the model.
    assert TagUnion(Refine.on("rho").above(0.05),
                    Refine.on("Density").gradient_above(0.5)).validate(model) is True
    # A union containing a bogus subject is rejected through the forwarded context.
    with pytest.raises(ValueError, match="is not a declared subject"):
        TagUnion(Refine.on("rho").above(0.05),
                 Refine.on("nope").below(0.1)).validate(model)


def test_problem_amr_refine_validates_the_subject_against_the_block_model():
    model = _FakeModel()
    # The role check fires where the model IS available: problem.amr.refine.
    prob = pops.Problem(layout=AMR(base=CartesianMesh(n=64))).block("plasma", physics=model)
    # A real role passes and the call chains back to the problem.
    assert prob.amr.refine(Refine.on("rho").above(0.05)) is prob
    # A bogus role is refused before runtime.
    prob2 = pops.Problem(layout=AMR(base=CartesianMesh(n=64))).block("plasma", physics=model)
    with pytest.raises(ValueError, match="is not a declared subject"):
        prob2.amr.refine(Refine.on("definitely_not_a_role").above(0.05))


# The CI python runner invokes each test file as `python3 <file>`; run pytest on this module so
# the assertions execute (a bare import would only define the test functions).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

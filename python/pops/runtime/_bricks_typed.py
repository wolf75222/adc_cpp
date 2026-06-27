"""Typed native-brick constructors (Spec 5 sec.14.2.5).

The native bricks are NAMED by typed constructors instead of magic ``kind=`` / ``bc=`` strings,
ADDITIVELY (the string path keeps working). This module homes the sec.14.2.5 typed ctors that do
not belong to a state/source/time module :

* the typed native ELLIPTIC boundary bricks ``Dirichlet`` / ``Neumann`` / ``Periodic`` ;
* ``ElectrostaticLorentzSchur`` -- the typed constructor for the (currently unique)
  ``CondensedSchur`` kind, kept here (not in ``_bricks_time``) to stay under the 500-line cap.

The native elliptic (Poisson) boundary is selected today by a magic ``bc=`` string token on
``System.set_poisson`` / ``System.add_elliptic_model`` : ``"auto" | "periodic" | "dirichlet" |
"neumann"`` (cf. the C++ binding ``System::set_poisson``). The typed constructors NAME that choice
with a type instead of a string : ``pops.Dirichlet()`` lowers to the SAME ``bc="dirichlet"`` token,
``pops.Neumann()`` to ``bc="neumann"``, ``pops.Periodic()`` to ``bc="periodic"``. The ``.bc``
attribute carries the token so a consumer can pass it straight through (``set_poisson(bc=b.bc)``).

The boundary bricks are the NATIVE elliptic-solver boundary (the homogeneous BC of the system
Poisson solve), and are DISTINCT from two other typed boundary surfaces already in the package :

* ``pops.fields.bcs`` (Spec 5 sec.5.5) -- inert field-VALUE conditions attached per face of a
  ``pops.fields`` elliptic problem (Dirichlet value, Neumann flux, first-order extrapolation) ;
* ``pops.mesh.boundaries`` (Spec 5 sec.5.9) -- domain-TOPOLOGY descriptors (periodic vs physical
  faces of the mesh).

All objects here are inert (no ``_pops`` / numpy / runtime / codegen compute) : they record /
lower their token (boundary) or pin a kind (Schur). ``pops.runtime.bricks`` re-exports them.
"""

from pops.runtime._bricks_time import CondensedSchur, Role


# The native elliptic boundary tokens accepted by System::set_poisson (C++ binding). "auto" lets the
# solver pick periodic vs wall from the mesh ; the three typed ctors below pin the explicit choices.
_BC_PERIODIC = "periodic"
_BC_DIRICHLET = "dirichlet"
_BC_NEUMANN = "neumann"


class _Boundary:
    """Base of the native elliptic (Poisson) boundary bricks : carries a ``bc=`` token.

    ``bc`` is the string consumed by ``System.set_poisson(bc=...)`` / ``add_elliptic_model(bc=...)``.
    ``lower()`` returns it so the install / runtime path can route the typed object to the existing
    string argument without a new C++ entry point (additive, inert).
    """

    bc = ""

    def lower(self):
        """The native ``bc=`` token this boundary lowers to (e.g. ``"dirichlet"``)."""
        return self.bc

    def __eq__(self, other):
        return isinstance(other, _Boundary) and other.bc == self.bc

    def __hash__(self):
        return hash((type(self).__name__, self.bc))

    def __repr__(self):
        return "%s(bc=%r)" % (type(self).__name__, self.bc)


class Periodic(_Boundary):
    """Native PERIODIC elliptic boundary : lowers to ``bc="periodic"`` (the Poisson solve wraps).

    Typed equivalent of ``set_poisson(bc="periodic")``. The mesh-topology counterpart is
    ``pops.mesh.boundaries.Periodic`` ; this brick is the native ELLIPTIC-solver boundary token.
    """

    bc = _BC_PERIODIC


class Dirichlet(_Boundary):
    """Native DIRICHLET elliptic boundary : lowers to ``bc="dirichlet"`` (homogeneous phi=0 wall).

    Typed equivalent of ``set_poisson(bc="dirichlet")``. The native Poisson solve imposes a
    homogeneous Dirichlet value on the physical faces (a conducting wall is added via
    ``set_poisson(wall="circle", wall_radius=...)``). The field-VALUE per-face Dirichlet of a
    ``pops.fields`` problem (with a non-zero value) lives in ``pops.fields.bcs.Dirichlet``.
    """

    bc = _BC_DIRICHLET


class Neumann(_Boundary):
    """Native NEUMANN elliptic boundary : lowers to ``bc="neumann"`` (homogeneous zero-flux wall).

    Typed equivalent of ``set_poisson(bc="neumann")``. The native Poisson solve imposes a
    homogeneous Neumann (zero normal derivative) condition on the physical faces.
    """

    bc = _BC_NEUMANN


class ElectrostaticLorentzSchur(CondensedSchur):
    """Typed constructor for the electrostatic-Lorentz condensed-Schur source stage (Spec 5
    sec.14.2.5). ``pops.ElectrostaticLorentzSchur(theta=0.5, alpha=1.0, ...)`` is the typed
    equivalent of ``pops.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, alpha=1.0, ...)``:
    it pins the (currently unique) ``kind`` so the algorithm is named by a type instead of a magic
    string. It is a ``CondensedSchur`` subclass, so every consumer that accepts a CondensedSchur
    (``pops.Split`` / ``pops.Strang`` source stage, ``System.add_equation``) accepts it unchanged.

    All other arguments (theta / alpha / density / momentum / energy / magnetic_field / potential /
    krylov_tol / krylov_max_iters) carry through to CondensedSchur with identical validation and
    lowering; see that class for the field-role descriptors and the assembled operator. Passing
    ``kind`` is not allowed (it is fixed to "electrostatic_lorentz").
    """

    def __init__(self, theta=0.5, alpha=1.0, density=Role.Density,
                 momentum=(Role.MomentumX, Role.MomentumY), energy=None,
                 magnetic_field="B_z", potential="phi",
                 krylov_tol=None, krylov_max_iters=None):
        super().__init__(kind="electrostatic_lorentz", theta=theta, alpha=alpha,
                         density=density, momentum=momentum, energy=energy,
                         magnetic_field=magnetic_field, potential=potential,
                         krylov_tol=krylov_tol, krylov_max_iters=krylov_max_iters)


__all__ = ["Periodic", "Dirichlet", "Neumann", "ElectrostaticLorentzSchur"]

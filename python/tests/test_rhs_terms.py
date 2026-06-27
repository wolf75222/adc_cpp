"""P.rhs(terms=[...]) typed RHS composition (Spec 5 sec.14.2.4, ADC-479 criterion 27).

The typed ``terms=`` front door is PURE sugar over the legacy ``flux=``/``sources=`` path: each
:class:`pops.numerics.terms.Flux`/source term lowers onto the existing booleans/name-list, so the
built IR is BYTE-IDENTICAL. These tests pin that equivalence on the ``Program._ir_hash``:

  - ``terms=[Flux(), <source>]`` builds the byte-identical hash to ``flux=True, sources=[<name>]``;
  - the legacy ``flux=``/``sources=`` path is unchanged (the new branch does not perturb it);
  - ``Flux()`` is a typed term, not a bool (a bare bool in terms= is a TypeError);
  - every accepted source form (name str / SourceTerm / OperatorHandle) maps onto the same name;
  - ``terms=`` plus any legacy kwarg is a clear ValueError;
  - a non-term object in terms= is a clear TypeError.

Pure Python; no compilation, no ``_pops``. Run with python3 (PYTHONPATH = built pops package).
"""
import pytest

from pops import time as adctime
from pops.model import OperatorHandle
from pops.numerics.terms import Flux, LocalTerm, SourceTerm


def _terms_program(terms):
    """A one-block forward-Euler Program whose single rhs is built from ``terms=``."""
    P = adctime.Program("rhs_terms")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=f, terms=terms)
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    P.validate()
    return P


def _legacy_program(flux, sources):
    """The same Program built through the legacy flux=/sources= surface."""
    P = adctime.Program("rhs_terms")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=f, flux=flux, sources=sources)
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    P.validate()
    return P


def test_terms_flux_plus_source_is_byte_identical():
    """terms=[Flux(), "electric"] == flux=True, sources=["electric"] (same _ir_hash)."""
    h_terms = _terms_program([Flux(), "electric"])._ir_hash()
    h_legacy = _legacy_program(True, ["electric"])._ir_hash()
    assert h_terms == h_legacy, (h_terms, h_legacy)
    print("OK  1. terms=[Flux(), 'electric'] _ir_hash == flux=True, sources=['electric']")


def test_terms_flux_only_is_byte_identical():
    """terms=[Flux()] (no source) == flux=True, sources=[] (flux only)."""
    assert _terms_program([Flux()])._ir_hash() == _legacy_program(True, [])._ir_hash()
    print("OK  2. terms=[Flux()] _ir_hash == flux=True, sources=[]")


def test_terms_source_only_is_byte_identical():
    """terms=["electric"] (no Flux) == flux=False, sources=["electric"] (source only)."""
    assert _terms_program(["electric"])._ir_hash() == _legacy_program(False, ["electric"])._ir_hash()
    print("OK  3. terms=['electric'] _ir_hash == flux=False, sources=['electric']")


def test_legacy_path_unchanged_by_terms_branch():
    """The legacy flux=/sources= path is byte-identical to a pristine build: adding the terms=
    branch must not perturb the default/legacy lowering (the _UNSET sentinel resolves to the
    historical defaults)."""
    # Default rhs() (no flux/sources/fluxes) keeps the historical flux=True, sources=None.
    P = adctime.Program("rhs_terms")
    dt = P.dt
    U = P.state("plasma")
    f = P.solve_fields(U)
    R = P.rhs("R", state=U, fields=f)
    P.commit("plasma", P.linear_combine("U1", U + dt * R))
    P.validate()
    # flux=True, sources=["default"] is the explicit spelling of the same default lowering.
    assert P._ir_hash() == _legacy_program(True, None)._ir_hash()
    print("OK  4. legacy default rhs() _ir_hash unchanged by the terms= branch")


def test_source_forms_map_to_same_name():
    """Every accepted source form (name str / SourceTerm / OperatorHandle) folds in the SAME
    source name, so all three build the byte-identical IR."""
    h_str = _terms_program([Flux(), "electric"])._ir_hash()
    h_srcterm = _terms_program([Flux(), SourceTerm("electric")])._ir_hash()
    h_handle = _terms_program([Flux(), OperatorHandle("electric", kind="local_source")])._ir_hash()
    h_local = _terms_program([Flux(), LocalTerm("electric")])._ir_hash()
    assert h_str == h_srcterm == h_handle == h_local, (h_str, h_srcterm, h_handle, h_local)
    print("OK  5. source forms (str/SourceTerm/OperatorHandle/LocalTerm) -> same name -> same hash")


def test_flux_is_a_term_not_a_bool():
    """Flux() is a typed term (sets flux=True), and a bare bool in terms= is rejected: the spec
    distinguishes a Flux term from a flux boolean."""
    # Flux() lowers to flux=True (proven by the byte-identical hash above); a bare True does not.
    assert _terms_program([Flux()])._ir_hash() != _terms_program([])._ir_hash()
    P = adctime.Program("rhs_terms")
    U = P.state("plasma")
    f = P.solve_fields(U)
    with pytest.raises(TypeError):
        P.rhs("R", state=U, fields=f, terms=[True])
    print("OK  6. Flux() is a term not a bool; a bare bool in terms= is a TypeError")


def test_terms_with_legacy_kwarg_raises():
    """terms= is mutually exclusive with flux=/sources=/fluxes=: passing both is a clear ValueError
    naming the conflicting kwarg."""
    P = adctime.Program("rhs_terms")
    U = P.state("plasma")
    f = P.solve_fields(U)
    for kw in ({"flux": True}, {"sources": ["electric"]}, {"fluxes": ["default"]}):
        with pytest.raises(ValueError, match="mutually exclusive"):
            P.rhs("R", state=U, fields=f, terms=[Flux()], **kw)
    print("OK  7. terms= plus a legacy kwarg -> ValueError naming the conflict")


def test_bad_term_raises_typeerror():
    """A non-term object in terms= is a clear TypeError (transparent typed surface)."""
    P = adctime.Program("rhs_terms")
    U = P.state("plasma")
    f = P.solve_fields(U)
    for bad in (123, 4.5, object(), ["nested"]):
        with pytest.raises(TypeError):
            P.rhs("R", state=U, fields=f, terms=[Flux(), bad])
    # An unnamed SourceTerm/LocalTerm has no declared source name to fold in.
    for unnamed in (SourceTerm(), LocalTerm()):
        with pytest.raises(ValueError, match="must be named"):
            P.rhs("R", state=U, fields=f, terms=[Flux(), unnamed])
    print("OK  8. a non-term in terms= -> TypeError; an unnamed source term -> ValueError")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

"""Coherence contract for adc.capabilities() (ADC-297).

adc.capabilities() is the published source of truth for what the runtime can dispatch
(Riemann fluxes, time methods, stability bounds, Poisson, geometry, Schur, DSL backends,
IO, AMR layout, aux). It is a hand-written dict, so it can silently drift from the gates it
claims to mirror. These checks pin the capability surface to facts that are verified
elsewhere in the suite, so that changing a capability forces either a test update or a
documentation update:

  T1 - the published top-level keys stay present (the doc and the limitations pages key off
       them; a vanished key means a stale reference).
  T2 - the Riemann surface matches the dispatch gates: hllc/roe are exposed on the cartesian
       and AMR facades but NOT on polar (no polar energy-flux brick, make_block_polar rejects
       them); polar exposes only rusanov + hll (the isothermal fluid declares wave_speeds).
       Guards the "hllc/roe = 2D Euler only" and "polar = scalar ExB only" doc regressions.
  T3 - backends_dsl MPI/AMR flags agree (truthiness) with the dsl._BACKEND_CAPS table that
       actually drives backend selection; catches drift between the two tables.
  T4 - the polar stability bounds (stability_speed / stability_dt / source_frequency) are
       advertised as wired (system_polar.cpp installs them); guards the PolarMesh "NOT wired"
       docstring regression.
  T5 - the AMR Schur stage is advertised as implemented (Phase 4a), not "to be done";
       guards the ALGORITHMS.md section 25 "the implementation does not exist" regression.
  T6 - the spatial-dimension invariant is published as the structured scalar dimension == 2
       (ADC-294 / ADR-0001 Decision 1: the 2D core is an official, introspectable limit, not
       prose); guards against the key silently vanishing or drifting to a non-2D value.
  T7 - the AMR regrid variable is advertised as selectable by name/role (ADC-296 / ADR-0001
       Decision 5), with the mono-block / compiled .so paths declared component-0 only; guards
       the "regrid is component-0 only" doc regression now that a selector exists.

The test is pure Python: it only reads adc.capabilities() and adc.dsl._BACKEND_CAPS, so it
needs the _adc extension to import but does not build or run any model.
"""
import adc
from adc import dsl

EXPECTED_TOP_KEYS = {
    "dimension", "riemann", "time", "stability_policy", "poisson", "geometry", "schur",
    "backends_dsl", "io", "amr_layout", "aux", "regrid",
}


def test_top_level_keys_present():
    caps = adc.capabilities()
    missing = EXPECTED_TOP_KEYS - set(caps)
    assert not missing, "capabilities() lost published top-level key(s): %s" % sorted(missing)


def test_riemann_surface_matches_dispatch():
    riemann = adc.capabilities()["riemann"]
    assert riemann["system_cartesian"] == ["rusanov", "hll", "hllc", "roe"], riemann["system_cartesian"]
    assert riemann["amr"] == ["rusanov", "hll", "hllc", "roe"], riemann["amr"]
    # Polar has no energy-flux brick: hllc/roe are rejected by make_block_polar, only
    # rusanov (any model) + hll (isothermal fluid, declares wave_speeds) are wired.
    assert riemann["system_polar"] == ["rusanov", "hll"], riemann["system_polar"]
    assert "hllc" not in riemann["system_polar"] and "roe" not in riemann["system_polar"]


def test_backends_dsl_flags_match_backend_caps():
    caps_b = adc.capabilities()["backends_dsl"]
    for backend in ("prototype", "aot", "production"):
        ref = dsl._BACKEND_CAPS[backend]
        got = caps_b[backend]
        assert bool(got["mpi"]) == bool(ref["mpi"]), \
            "%s: capabilities() mpi=%r disagrees with _BACKEND_CAPS mpi=%r" % (backend, got["mpi"], ref["mpi"])
        assert bool(got["amr"]) == bool(ref["amr"]), \
            "%s: capabilities() amr=%r disagrees with _BACKEND_CAPS amr=%r" % (backend, got["amr"], ref["amr"])


def test_polar_stability_bounds_advertised_wired():
    polar = " ".join(adc.capabilities()["stability_policy"]["system_polar"])
    for bound in ("stability_speed", "stability_dt", "source_frequency"):
        assert bound in polar, "polar stability bound %r missing from capabilities()" % bound


def test_amr_schur_advertised_implemented():
    amr_schur = adc.capabilities()["schur"]["amr"]
    assert amr_schur and "Phase 4a" in amr_schur, \
        "schur.amr should advertise the implemented Phase 4a composite stage, got: %r" % amr_schur
    assert "the implementation does not" not in amr_schur


def test_dimension_invariant_2d():
    # ADC-294 / ADR-0001 Decision 1: the core is officially 2D. The limit is published as a
    # structured scalar (not prose) so scripts and the limitations doc can introspect it, and it is
    # a SEPARATE top-level key, NOT nested under "geometry" (polar is a second geometry at the SAME
    # dimension, not a third axis).
    caps = adc.capabilities()
    dim = caps["dimension"]
    assert dim == 2, "capabilities()['dimension'] should declare the 2D-core invariant, got %r" % (dim,)
    # bool is a subclass of int in Python; pin to a plain int so True / 2.0 / "2" cannot pass.
    assert isinstance(dim, int) and not isinstance(dim, bool), \
        "capabilities()['dimension'] should be a plain int scalar, got %r" % (dim,)
    assert "dimension" not in caps["geometry"], \
        "the dimension invariant must stay a separate top-level key, not nested under geometry"


def test_regrid_variable_selector_advertised():
    # ADC-296 / ADR-0001 Decision 5: the multi-block regrid variable is selectable by name/role
    # (default = component 0). The mono-block and compiled .so paths stay component-0 only. The
    # surface mirrors AmrSystem.set_refinement(threshold, variable=, role=).
    regrid = adc.capabilities()["regrid"]
    assert set(regrid["variable_selector"]) == {"component_0", "by_name", "by_role"}, \
        regrid["variable_selector"]
    assert "by_name" in regrid["multi_block"] and "by_role" in regrid["multi_block"], regrid["multi_block"]
    assert "component_0 only" in regrid["mono_block"], regrid["mono_block"]
    assert "component_0 only" in regrid["compiled_so"], regrid["compiled_so"]


def test_aux_named_surface_and_limit_parity():
    # ADC-291: named aux is advertised on System (cartesian + polar) AND AMR (single + multi block),
    # no longer "cartesian System only". The remaining compile-time limit (kAuxMaxExtra) is published
    # as an introspectable scalar and MUST match BOTH the C++ source (_adc.__aux_max_extra__) and the
    # DSL mirror (dsl.AUX_NAMED_MAX) -- this pins the hand-maintained Python<->C++ mirror so it cannot
    # silently drift (the historical #51-class risk the issue calls out).
    from adc import _adc
    named = adc.capabilities()["aux"]["named"]
    assert set(named["backends"]) >= {"system_cartesian", "system_polar", "amr_single_block",
                                      "amr_multi_block"}, named["backends"]
    # the limit is the SINGLE C++ source, mirrored by the DSL constant.
    assert named["limit"] == _adc.__aux_max_extra__ == dsl.AUX_NAMED_MAX, \
        "aux named limit drift: caps=%r, C++=%r, dsl=%r" % (
            named["limit"], _adc.__aux_max_extra__, dsl.AUX_NAMED_MAX)
    # the aux ghost width is explicit (the configurable-radius mechanism is a documented follow-up).
    assert named["halo_radius"] == 1, named["halo_radius"]
    # the other mirrored aux constants stay coherent C++ <-> DSL.
    assert _adc.__aux_named_base__ == dsl.AUX_NAMED_BASE, "AUX_NAMED_BASE drift"
    assert _adc.__aux_base_comps__ == dsl.AUX_BASE_COMPS, "AUX_BASE_COMPS drift"
    assert _adc.__aux_max_comps__ == _adc.__aux_named_base__ + _adc.__aux_max_extra__
    # the C++ canonical name->component table mirrors the Python AUX_CANONICAL exactly.
    assert dict(_adc.__aux_canonical__) == dict(dsl.AUX_CANONICAL), \
        "C++ aux_names table != Python AUX_CANONICAL: %r vs %r" % (
            dict(_adc.__aux_canonical__), dict(dsl.AUX_CANONICAL))
    # no stale "cartesian System only" claim survives in the aux surface.
    blob = repr(adc.capabilities()["aux"]).lower()
    assert "cartesian system only" not in blob, "stale 'cartesian System only' aux claim"


if __name__ == "__main__":
    test_top_level_keys_present()
    test_riemann_surface_matches_dispatch()
    test_backends_dsl_flags_match_backend_caps()
    test_polar_stability_bounds_advertised_wired()
    test_amr_schur_advertised_implemented()
    test_dimension_invariant_2d()
    test_regrid_variable_selector_advertised()
    test_aux_named_surface_and_limit_parity()
    print("test_capabilities : OK (top keys, riemann surface, backends_dsl, polar stability, "
          "AMR Schur, 2D dimension, regrid selector, aux named surface + limit parity)")

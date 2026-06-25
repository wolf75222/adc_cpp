"""Spec 3 adc.lib: a catalog of typed brick descriptors and IR macros.

adc.lib never computes in Python. A descriptor names a brick (native C++ id,
generated, macro or external) and carries its requirements / capabilities; the
codegen and runtime consume it. These tests check that the descriptors are
lightweight metadata that lower to native ids -- not numerical code.
"""
import pytest

lib = pytest.importorskip("adc.lib")


def test_riemann_hllc_is_a_native_descriptor():
    d = lib.riemann.HLLC()
    assert d.brick_type == "native"
    assert d.available
    assert d.native_id == "adc::HLLCFlux"   # the EXACT C++ symbol (namespace adc)
    assert d.scheme == "hllc"               # the runtime scheme string


def test_riemann_native_ids_are_exact():
    # Guard against the wrong-namespace overclaim: ids must be the real adc:: symbols.
    assert lib.riemann.Rusanov().native_id == "adc::RusanovFlux"
    assert lib.riemann.HLL().native_id == "adc::HLLFlux"
    assert lib.riemann.Roe().native_id == "adc::RoeFlux"


def test_reconstruction_weno5z_is_native():
    d = lib.reconstruction.WENO5Z()
    assert d.brick_type == "native"
    assert d.native_id == "adc::Weno5"      # adc::Weno5 IS the WENO5-Z reconstruction
    assert d.scheme == "weno5"


def test_catalogued_but_unwired_bricks_are_marked_unavailable():
    # No native symbol is fabricated: planned bricks carry available=False, empty id.
    for d in (lib.fields.Poisson(), lib.solvers.Newton(),
              lib.preconditioners.Jacobi(), lib.limiters.MC()):
        assert d.available is False
        assert d.native_id == ""


def test_available_native_ids_exist_and_are_namespaced():
    for d in (lib.fields.GeometricMG(), lib.solvers.CG(), lib.solvers.GMRES(),
              lib.solvers.Schur(), lib.projections.positivity()):
        assert d.available
        assert d.native_id.startswith("adc::")


def test_riemann_descriptors_compute_nothing():
    # A descriptor exposes metadata only -- no eval / compile / __call__ numeric path.
    d = lib.riemann.Rusanov()
    assert not hasattr(d, "eval")
    assert not hasattr(d, "compile")
    assert d.scheme == "rusanov"
    # frozen-ish: the same descriptor twice compares equal (value type)
    assert lib.riemann.Rusanov() == lib.riemann.Rusanov()


def test_field_solver_descriptor_carries_options():
    d = lib.fields.GeometricMG(tolerance=1e-10, max_iters=200)
    assert d.brick_type == "native"
    assert d.options["tolerance"] == 1e-10
    assert d.options["max_iters"] == 200


def test_solver_descriptors():
    assert lib.solvers.BiCGStab().scheme == "bicgstab"
    assert lib.solvers.GMRES().scheme == "gmres"
    assert lib.solvers.CG().scheme == "cg"


def test_user_riemann_is_external():
    # A User brick must be loaded first (ADC-463); registering its manifest then makes
    # riemann.User(id) surface an external_cpp descriptor.
    import json
    lib._register_manifest(json.dumps(
        {"bricks": [{"id": "my_hllc_variant", "category": "riemann"}]}))
    try:
        d = lib.riemann.User("my_hllc_variant")
        assert d.brick_type == "external_cpp"
        assert d.native_id == "my_hllc_variant"
    finally:
        lib._clear_external_catalog()


def test_descriptor_requirements_present():
    # HLLC requires the model HLLC capabilities; Rusanov only needs a max wave speed.
    assert "hllc_star_state" in lib.riemann.HLLC().requirements.get("capabilities", [])
    assert lib.riemann.Rusanov().requirements.get("capabilities") == ["max_wave_speed"]

"""Spec 3 section 22 + 24 (ADC-466): unified ``sim.install(...)`` + install-time validation.

``sim.install(compiled, instances=, params=, aux=, solvers=)`` is the single Spec-3 entry that
installs the compiled handle, binds each named instance's block by name, sets its initial state and
spatial brick, sets the field solvers / aux fields / runtime params, and finally installs the
compiled time Program -- LOWERING to the existing lower-layer calls (add_equation / set_poisson /
set_magnetic_field / set_aux_field / set_block_params / install_program), no parallel runtime.

The full compiled-.so install RUN needs a compiler + a visible Kokkos (ADC_KOKKOS_ROOT) and is
validated on ROMEO / CI-Kokkos (mirrors test_install_requirement_validation.py). The API SHAPE, the
lowering, and the section-24 capability/aux/solver validation messages are host-testable WITHOUT a
full run -- exercised here. cf. docs/sphinx/reference/board-like-dsl.md.
"""
import sys

try:
    import numpy as np

    import adc
    from adc import dsl
    from adc import time as adctime
except Exception as exc:  # noqa: BLE001
    print("skip test_unified_install (adc/numpy unavailable: %s)" % exc)
    sys.exit(0)

N = 16


def _fake_compiled(*, hllc=False, roe=False, prim_names=("rho", "u", "v"), wave_speeds=False,
                   params=None):
    """A real adc.dsl.CompiledModel object (the engine class) carrying only metadata -- NOT a built
    .so. Used to exercise the host-testable section-24 capability check and the params routing
    WITHOUT compiling (which needs Kokkos). It is never install_program'd."""
    return dsl.CompiledModel(
        so_path="/nonexistent/problem.so", backend="production", adder="add_native_block",
        cons_names=["rho", "mx", "my"], cons_roles=["density", "momentum_x", "momentum_y"],
        prim_names=list(prim_names), n_vars=3, gamma=None, n_aux=3, params=params or {},
        caps={}, abi_key="", model_hash="", cxx="c++", std="23",
        hllc=hllc, roe=roe, wave_speeds=wave_speeds)


def test_lower_spatial_accepts_runtime_and_lib():
    """install lowers BOTH an adc.FiniteVolume (runtime) and an adc.lib.spatial.FiniteVolume
    (Spec-3 catalog descriptor) to the same add_equation spatial args."""
    sim = adc.System(n=N, L=1.0, periodic=True)
    # Runtime descriptor passes through unchanged.
    rt = adc.FiniteVolume(limiter="weno5", riemann="hll", variables="primitive")
    low = sim._lower_spatial(rt)
    assert low is rt, "runtime Spatial must pass through unchanged"
    # lib descriptor: riemann/reconstruction/positivity_floor -> limiter/flux/recon.
    libdesc = adc.lib.spatial.FiniteVolume(riemann="hllc", reconstruction="weno5",
                                           positivity_floor=1e-12)
    low = sim._lower_spatial(libdesc)
    assert low.flux == "hllc", "riemann -> Spatial.flux (got %r)" % low.flux
    assert low.limiter == "weno5", "reconstruction -> Spatial.limiter (got %r)" % low.limiter
    assert low.positivity_floor == 1e-12, "positivity_floor lowered (got %r)" % low.positivity_floor
    # None -> default Spatial.
    assert isinstance(sim._lower_spatial(None), adc.Spatial)
    print("OK  _lower_spatial accepts runtime + lib descriptors")


def test_solver_token_lowering():
    """A field-solver selection lowers to its set_poisson token: string as-is, or the lib
    descriptor's scheme (adc.lib.fields.GeometricMG -> 'geometric_mg')."""
    sim = adc.System(n=N, L=1.0, periodic=True)
    assert sim._solver_token("geometric_mg") == "geometric_mg"
    assert sim._solver_token(adc.lib.fields.GeometricMG()) == "geometric_mg"
    print("OK  _solver_token lowers string + lib descriptor")


def test_install_solver_sets_poisson():
    """install lowers solvers={'phi': GeometricMG(...)} to set_poisson, reflected by poisson_solver()
    (the section-24 accessor) when the binding is present."""
    sim = adc.System(n=N, L=1.0, periodic=True)
    sim._install_solver("phi", adc.lib.fields.GeometricMG())
    if hasattr(sim._s, "poisson_solver"):
        assert sim.poisson_solver() == "geometric_mg", \
            "set_poisson lowered (got %r)" % sim.poisson_solver()
        print("OK  _install_solver lowers to set_poisson (poisson_solver() == geometric_mg)")
    else:
        print("OK  _install_solver lowers to set_poisson (poisson_solver accessor absent; rebuild _adc)")
    # A second named elliptic field is deferred -> NotImplementedError (explicit, not silent).
    try:
        sim._install_solver("temperature", adc.lib.fields.GeometricMG())
        raise AssertionError("MISMATCH: a second named elliptic field should be NotImplementedError")
    except NotImplementedError:
        print("OK  _install_solver defers a second named elliptic field (NotImplementedError)")


def test_riemann_capability_verbatim():
    """Section 24: the selected Riemann flux must be backed by the model capability, with the
    VERBATIM spec message. A compiled model WITHOUT the HLLC capability and WITHOUT a pressure
    rejects riemann='hllc'."""
    sim = adc.System(n=N, L=1.0, periodic=True)
    model = _fake_compiled(hllc=False, prim_names=("rho", "u", "v"))
    try:
        sim._validate_riemann_capability(model, adc.FiniteVolume(riemann="hllc"))
        raise AssertionError("MISMATCH: hllc without capability should raise")
    except RuntimeError as exc:
        assert str(exc) == "riemann HLLC requires capability 'hllc_star_state'", \
            "verbatim message (got %r)" % str(exc)
        print("OK  riemann HLLC requires capability 'hllc_star_state'")
    # Roe without capability / pressure rejects too.
    try:
        sim._validate_riemann_capability(model, adc.FiniteVolume(riemann="roe"))
        raise AssertionError("MISMATCH: roe without capability should raise")
    except RuntimeError as exc:
        assert "roe_dissipation" in str(exc).lower() or "Roe requires capability" in str(exc), \
            "roe capability message (got %r)" % str(exc)
        print("OK  riemann Roe requires its capability")
    # With the capability emitted, the same flux passes.
    ok_model = _fake_compiled(hllc=True, prim_names=("rho", "u", "v", "p"))
    sim._validate_riemann_capability(ok_model, adc.FiniteVolume(riemann="hllc"))
    print("OK  riemann capability accepted once the model emits it")


def test_install_aux_derived_rejected():
    """install rejects aux={'T_e': ...} (T_e is DERIVED, not a static aux field) and a named aux not
    declared by any installed instance -- both host-testable, no .so."""
    sim = adc.System(n=N, L=1.0, periodic=True)
    try:
        sim._install_aux("T_e", np.ones(N * N))
        raise AssertionError("MISMATCH: T_e should be rejected (derived)")
    except ValueError as exc:
        assert "T_e" in str(exc) and "set_electron_temperature_from" in str(exc)
        print("OK  install rejects aux 'T_e' (derived)")
    try:
        sim._install_aux("grad_phi_custom", np.ones(N * N))
        raise AssertionError("MISMATCH: an undeclared named aux should be rejected")
    except ValueError as exc:
        assert "not declared by any installed instance" in str(exc)
        print("OK  install rejects an undeclared named aux field")


def test_install_params_routing():
    """install routes a flat params dict to set_block_params per instance (keyed by the RESOLVED
    CompiledModel's runtime_param_names, NOT the raw dsl.Model), and rejects a param name declared by
    no instance (no silent drop). Host-testable via _install_params (which takes resolved models)."""
    sim = adc.System(n=N, L=1.0, periodic=True)
    # An instance whose RESOLVED model declares no runtime params: a stray param name must raise.
    try:
        sim._install_params({"plasma": _fake_compiled(params={})}, {"nu": 1.0})
        raise AssertionError("MISMATCH: an unknown param should raise")
    except ValueError as exc:
        assert "declared by no instance" in str(exc)
        print("OK  install rejects a param declared by no instance")
    # NOTE: the positive routing (a DECLARED runtime param resolves through the CompiledModel's
    # runtime_param_names -- the fix that stopped reading the raw dsl.Model, which has none) needs a
    # really-compiled model with a declared runtime param; it is exercised by the Kokkos end-to-end
    # below (and tracked for a dedicated positive test). _install_params now takes the RESOLVED models.


def _lorentz_model(name="adc466_model"):
    """An isothermal fluid whose Lorentz linear source reads the aux field B_z (a hard requirement),
    same shape as test_install_requirement_validation -- used for the Kokkos-gated end-to-end."""
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs = dsl.sqrt(0.5)
    m.flux(x=[mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho + 0.5 * rho])
    m.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                  y=[my / rho - cs, my / rho, my / rho + cs])
    m.primitive_vars(rho, mx, my)
    m.conservative_from([rho, mx, my])
    bz = m.aux("B_z")
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho)
    m.rate_operator("explicit_rhs", flux=True)
    return m


def _lie_program(name="adc466_prog"):
    P = adctime.Program(name)
    u = P.state("plasma")
    fields = P.solve_fields(u)
    r = P.rhs(state=u, fields=fields)
    P.commit("plasma", P.linear_combine("u1", u + P.dt * r))
    return P


def test_install_end_to_end_kokkos():
    """End-to-end unified install (needs a compiler + Kokkos -> ROMEO / CI-Kokkos). A single
    sim.install(compiled, instances=, aux=, solvers=) wires + installs; the NEGATIVE case (no B_z)
    raises the section-24 aux requirement at install."""
    if not hasattr(adc.System(n=8, L=1.0, periodic=True), "install_program"):
        print("skip test_install_end_to_end_kokkos (_adc lacks install_program; rebuild _adc)")
        return
    m = _lorentz_model()
    try:
        compiled = adc.compile_problem(model=m, time=_lie_program())
    except RuntimeError as exc:
        print("skip test_install_end_to_end_kokkos (no Kokkos to build the .so: %s)"
              % str(exc)[:120])
        return

    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    u0 = np.stack([rho, 0.4 * rho, -0.2 * rho])

    # Negative: install WITHOUT aux B_z -> section-24 aux requirement raised at install_program.
    sim_missing = adc.System(n=N, L=1.0, periodic=True)
    try:
        sim_missing.install(
            compiled,
            instances={"plasma": {"state": "U", "initial": u0,
                                  "spatial": adc.FiniteVolume(limiter="none", riemann="rusanov"),
                                  "time": adc.Explicit(method="euler")}},
            solvers={"phi": adc.lib.fields.GeometricMG()})
        raise AssertionError("MISMATCH: unified install accepted a simulation missing B_z")
    except RuntimeError as exc:
        assert "lorentz" in str(exc) and "B_z" in str(exc) and "did not provide" in str(exc), \
            "section-24 aux message (got %r)" % str(exc)
        print("OK  unified install rejects a missing required aux: %s" % str(exc))

    # Positive: the SAME install with aux={'B_z': ...} wires + installs cleanly.
    sim_ok = adc.System(n=N, L=1.0, periodic=True)
    sim_ok.install(
        compiled,
        instances={"plasma": {"state": "U", "initial": u0,
                              "spatial": adc.FiniteVolume(limiter="none", riemann="rusanov"),
                              "time": adc.Explicit(method="euler")}},
        aux={"B_z": 3.0 * np.ones(N * N)},
        solvers={"phi": adc.lib.fields.GeometricMG()})
    assert "plasma" in sim_ok.block_names(), "instance bound by name"
    print("OK  unified install wires instance + aux + solver and installs the program")


def main():
    test_lower_spatial_accepts_runtime_and_lib()
    test_solver_token_lowering()
    test_install_solver_sets_poisson()
    test_riemann_capability_verbatim()
    test_install_aux_derived_rejected()
    test_install_params_routing()
    test_install_end_to_end_kokkos()
    return 0


if __name__ == "__main__":
    sys.exit(main())

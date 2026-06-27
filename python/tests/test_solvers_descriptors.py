"""ADC-500 (Spec 5 sec.5.7 / criterion 4 / sec.13.11.1): the pops.solvers central package.

pops.solvers homes the linear / nonlinear / Schur / elliptic solver + preconditioner catalog
as inert typed descriptors. These tests construct each entry, exercise the RICH GeometricMG
parameter surface (typed smoother / coarse / tolerance + capabilities) and its protocol
(inspect / options / capabilities / lower), check that a bare string is rejected where a typed
sub-descriptor is expected (Spec 5 sec.7), and assert lower() carries the right native id /
scheme. They also confirm the pops.lib.solvers shim still resolves the legacy install-path
names (back-compat). The descriptors compute nothing; only their metadata is asserted.
"""
import pytest

pops = pytest.importorskip("pops")
solvers = pytest.importorskip("pops.solvers")

from pops.solvers import elliptic, krylov, nonlinear, schur
from pops.solvers.options import Chebyshev, DirectSmallGrid, RedBlackGaussSeidel
from pops.solvers.tolerances import Absolute, AbsoluteFloor, Relative


# --- the package is wired and exposed ----------------------------------------------------

def test_solvers_is_top_level_and_exposed():
    assert pops.solvers is solvers
    for sub in ("elliptic", "krylov", "nonlinear", "schur",
                "options", "tolerances", "preconditioners", "requirements"):
        assert hasattr(solvers, sub), "pops.solvers missing sub-module %r" % sub


# --- Krylov solvers (moved from pops.lib.solvers) ----------------------------------------

def test_krylov_native_ids_and_schemes():
    assert krylov.CG().native_id == "pops::cg_solve"
    assert krylov.CG().scheme == "cg"
    assert krylov.BiCGStab().native_id == "pops::bicgstab_solve"
    assert krylov.BiCGStab().scheme == "bicgstab"
    assert krylov.GMRES().native_id == "pops::gmres_solve"
    assert krylov.GMRES().scheme == "gmres"
    assert krylov.Richardson().native_id == "pops::richardson_solve"
    assert krylov.Richardson().scheme == "richardson"
    for d in (krylov.CG(), krylov.GMRES(), krylov.BiCGStab(), krylov.Richardson()):
        assert d.brick_type == "native"
        assert d.available
        assert d.category == "solver"


def test_krylov_descriptors_compute_nothing():
    d = krylov.GMRES()
    assert not hasattr(d, "eval")
    assert not hasattr(d, "compile")
    # value identity: the same descriptor twice compares equal.
    assert krylov.GMRES() == krylov.GMRES()


def test_krylov_lower_carries_native_id_and_scheme():
    rec = krylov.CG().lower()
    assert rec["native_id"] == "pops::cg_solve"
    assert rec["scheme"] == "cg"


# --- nonlinear solvers (planned: no native type yet) -------------------------------------

def test_nonlinear_are_planned():
    for d in (nonlinear.Newton(), nonlinear.FixedPoint()):
        assert d.available is False
        assert d.native_id == ""
        assert d.category == "solver"
    assert nonlinear.Newton().scheme == "newton"
    assert nonlinear.FixedPoint().scheme == "fixed_point"


# --- Schur-condensation solver -----------------------------------------------------------

def test_schur_native_id_and_alias():
    assert schur.Schur().native_id == "pops::SchurCondensationOperator"
    assert schur.Schur().scheme == "schur"
    # CondensedSchur is an alias naming the same native operator (distinct from the
    # pops.time CondensedSchur splitting POLICY).
    assert schur.CondensedSchur().native_id == "pops::SchurCondensationOperator"
    assert schur.CondensedSchur() == schur.Schur()


# --- the RICH GeometricMG elliptic solver ------------------------------------------------

def test_geometric_mg_defaults():
    g = elliptic.GeometricMG()
    assert g.name == "geometric_mg"
    assert g.scheme == "geometric_mg"
    assert g.native_id == "pops::GeometricMG"
    assert g.category == "elliptic_solver"
    opts = g.options()
    assert opts == {"smoother": "chebyshev", "coarse": "direct_small_grid",
                    "tolerance": "relative", "max_cycles": 20}


def test_geometric_mg_rich_surface():
    g = elliptic.GeometricMG(
        smoother=RedBlackGaussSeidel(),
        coarse=DirectSmallGrid(threshold=64),
        tolerance=Relative(rel=1e-8, floor=AbsoluteFloor(1e-14)),
        max_cycles=30)
    assert g.smoother.name == "red_black_gauss_seidel"
    assert g.coarse.options() == {"threshold": 64}
    assert g.tolerance.options() == {"rel": 1e-8, "abs_floor": 1e-14}
    assert g.options()["max_cycles"] == 30
    # An Absolute tolerance is accepted too.
    assert elliptic.GeometricMG(tolerance=Absolute(1e-9)).tolerance.name == "absolute"
    # A Chebyshev smoother of a chosen degree is accepted.
    assert elliptic.GeometricMG(smoother=Chebyshev(degree=4)).smoother.options() == {"degree": 4}


def test_geometric_mg_capabilities():
    caps = elliptic.GeometricMG().capabilities()
    assert caps["supports_uniform"] is True
    assert caps["supports_amr"] is True
    assert caps["supports_mpi"] is True
    assert caps["supports_gpu"] is True
    assert caps["supports_variable_epsilon"] is True
    assert caps["supports_anisotropic"] is False
    assert caps["supports_screened"] is False


def test_geometric_mg_inspect_and_lower():
    g = elliptic.GeometricMG(smoother=Chebyshev(degree=3))
    view = g.inspect()
    assert view["name"] == "geometric_mg"
    assert view["native_id"] == "pops::GeometricMG"
    assert view["scheme"] == "geometric_mg"
    assert view["available"] is True
    assert view["capabilities"]["supports_amr"] is True
    rec = g.lower()
    assert rec["native_id"] == "pops::GeometricMG"
    assert rec["scheme"] == "geometric_mg"
    assert rec["smoother"] == {"kind": "chebyshev", "degree": 3}
    assert rec["coarse"]["kind"] == "direct_small_grid"
    assert rec["tolerance"]["kind"] == "relative"
    assert rec["max_cycles"] == 20


def test_geometric_mg_rejects_string_for_typed_subdescriptor():
    # Spec 5 sec.7: a bare string / number for a typed sub-descriptor slot is rejected loud.
    with pytest.raises(TypeError, match="smoother"):
        elliptic.GeometricMG(smoother="chebyshev")
    with pytest.raises(TypeError, match="coarse"):
        elliptic.GeometricMG(coarse="direct")
    with pytest.raises(TypeError, match="tolerance"):
        elliptic.GeometricMG(tolerance=1e-6)
    with pytest.raises(TypeError, match="max_cycles"):
        elliptic.GeometricMG(max_cycles="20")
    # The tolerance floor must itself be a typed AbsoluteFloor.
    with pytest.raises(TypeError, match="floor"):
        Relative(rel=1e-6, floor=1e-12)


# --- the FFT elliptic solver (real pops::PoissonFFTSolver) --------------------------------

def test_fft_is_a_real_solver_with_route_constraints():
    f = elliptic.FFT()
    assert f.name == "fft"
    # A real, runtime-wired solver -- not unimplemented.
    assert f.native_id == "pops::PoissonFFTSolver"
    assert f.scheme == "fft"
    status = f.available()
    # partial = genuine route constraints (periodic / const-coeff / power-of-two), not "no symbol".
    assert status.status == "partial"
    assert any("periodic" in m for m in status.missing)
    assert "pops.solvers.elliptic.GeometricMG()" in status.alternatives
    assert f.inspect()["available"] == "partial"
    # spectral=True selects the fft_spectral token.
    spectral = elliptic.FFT(spectral=True)
    assert spectral.scheme == "fft_spectral"
    assert spectral.options() == {"spectral": True}


# --- preconditioners ---------------------------------------------------------------------

def test_preconditioners_catalog():
    pre = solvers.preconditioners
    assert pre.GeometricMG().native_id == "pops::GeometricMG"
    assert pre.GeometricMG().category == "preconditioner"
    for d in (pre.Identity(), pre.Jacobi(), pre.BlockJacobi()):
        assert d.available is False
        assert d.native_id == ""


# --- requirements vocabulary -------------------------------------------------------------

def test_capability_vocabulary_rejects_unknown_tag():
    from pops.solvers.requirements import capability_map
    assert capability_map(uniform=True)["supports_uniform"] is True
    with pytest.raises(ValueError, match="unknown solver capability tag"):
        capability_map(quantum=True)


# --- back-compat: the pops.lib.solvers shim still resolves the legacy names ---------------

def test_lib_solvers_shim_back_compat():
    lib_solvers = pytest.importorskip("pops.lib.solvers")
    # The legacy flat namespace (pops.lib.solvers.solvers) still resolves to the moved factories.
    ns = lib_solvers.solvers
    assert ns.GMRES().scheme == "gmres"
    assert ns.CG().native_id == "pops::cg_solve"
    assert ns.Schur().native_id == "pops::SchurCondensationOperator"
    assert ns.Newton().available is False
    # The custom-solver registry hooks are wired (the authoring DSL lives in the shim).
    assert callable(ns.custom)
    assert callable(ns.registered)
    # The authoring DSL + the preconditioner namespace re-export through the shim.
    assert callable(lib_solvers.solver)
    assert callable(lib_solvers.generate_solver_cpp)
    assert lib_solvers.preconditioners.GeometricMG().native_id == "pops::GeometricMG"


def test_install_path_token_resolution_for_rich_descriptor():
    # The unified-install solver-token resolver reads .scheme; the new rich GeometricMG
    # resolves to the same 'geometric_mg' token as the untouched pops.lib.fields descriptor.
    from pops.runtime._system_unified_install import _SystemUnifiedInstall
    assert _SystemUnifiedInstall._solver_token(elliptic.GeometricMG()) == "geometric_mg"
    assert _SystemUnifiedInstall._solver_token(pops.lib.fields.GeometricMG()) == "geometric_mg"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))

"""Init et diagnostic en variables PRIMITIVES (System-pipeline P4) :

    sim.set_primitive_state("e", rho=rho0, u=u0, v=v0, p=p0)   # init DEPUIS les primitives
    P = sim.get_primitive_state("e")                            # relit les primitives (diagnostic)

set_density ne pose que la densite (reste au repos). Pour un vrai modele on initialise depuis les
PRIMITIVES (rho, u, v, p) -- converties en CONSERVATIVES par le MODELE du bloc (compressible :
E = p/(g-1) + 1/2 rho|v|^2 ; isotherme : rho u ; scalaire : identite) -- et on veut relire les
primitives pour les diagnostics (vitesses, pression). La conversion est celle du MODELE du bloc, pas
une formule recodee : elle est forwardee a l'execution pour le natif (add_block) ET le DSL compile
(add_compiled_block AOT, add_dynamic_block JIT), via les memes formules que le flux.

On verifie :
 - round-trip set_primitive_state -> get_primitive_state == identite a la precision machine ;
 - apres set_primitive_state, un pas tourne et l'etat conservatif reste physique ;
 - un bloc COMPRESSIBLE (4 var, avec p) ET un bloc ISOTHERME (3 var, sans p) ET un bloc SCALAIRE ;
 - un modele DSL COMPILE (AOT + JIT) -- s'auto-saute sans compilateur C++ ;
 - set_density marche toujours (pas de regression) ;
 - erreur claire sur un nom de primitive inconnu (ou manquant).
"""
import os
import shutil
import tempfile

import numpy as np

import pops

N, L = 24, 1.0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def _fields(n=N):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    rho = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    u = 0.2 * np.sin(2 * np.pi * X)
    v = -0.1 * np.cos(2 * np.pi * Y)
    p = 0.5 + 0.1 * X
    return rho, u, v, p


def _roundtrip_err(sim, name, **prims):
    """set_primitive_state(**prims) puis get_primitive_state : ecart max sur chaque primitive."""
    sim.set_primitive_state(name, **prims)
    out = sim.get_primitive_state(name)
    return max(float(np.max(np.abs(out[k] - np.asarray(prims[k])))) for k in prims)


def test_compressible():
    """Bloc Euler compressible (4 var) : round-trip (rho, u, v, p) exact + pas physique."""
    rho, u, v, p = _fields()
    spec = pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    s = pops.System(n=N, L=L, periodic=True)
    s.add_block("e", spec, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    assert s.variable_names("e", "primitive") == ["rho", "u", "v", "p"]

    err = _roundtrip_err(s, "e", rho=rho, u=u, v=v, p=p)
    assert err < 1e-13, "compressible : round-trip cons<->prim non exact (%.2e)" % err

    # E conservatif coherent avec la formule du modele (pas seulement rho pose) : v != 0, E != rho/(g-1).
    U = np.array(s.get_state("e")).reshape(4, N, N)
    assert float(np.abs(U[1]).max()) > 1e-3 and float(np.abs(U[2]).max()) > 1e-3, "qte de mvt nulle"
    E_expected = p / (1.4 - 1.0) + 0.5 * rho * (u * u + v * v)
    assert float(np.max(np.abs(U[3] - E_expected))) < 1e-13, "E != p/(g-1) + 1/2 rho|v|^2"

    # un pas tourne et l'etat reste physique (densite > 0, fini).
    s.set_poisson()
    for _ in range(5):
        s.step_cfl(0.4)
    U1 = np.array(s.get_state("e")).reshape(4, N, N)
    assert np.isfinite(U1).all() and U1[0].min() > 0, "etat non physique apres set_primitive_state + pas"
    print("OK  compressible (4 var) : round-trip %.1e, E coherent, pas physique" % err)


def test_isothermal():
    """Bloc Euler isotherme (3 var, sans p) : round-trip (rho, u, v) exact."""
    rho, u, v, _ = _fields()
    spec = pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    s = pops.System(n=N, L=L, periodic=True)
    s.add_block("g", spec, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    assert s.variable_names("g", "primitive") == ["rho", "u", "v"]

    err = _roundtrip_err(s, "g", rho=rho, u=u, v=v)
    assert err < 1e-13, "isotherme : round-trip cons<->prim non exact (%.2e)" % err

    U = np.array(s.get_state("g")).reshape(3, N, N)
    assert float(np.max(np.abs(U[1] - rho * u))) < 1e-13, "rho_u != rho * u"
    assert float(np.max(np.abs(U[2] - rho * v))) < 1e-13, "rho_v != rho * v"

    s.set_poisson()
    for _ in range(5):
        s.step_cfl(0.4)
    U1 = np.array(s.get_state("g")).reshape(3, N, N)
    assert np.isfinite(U1).all() and U1[0].min() > 0, "isotherme : etat non physique apres init + pas"
    print("OK  isotherme (3 var, sans p) : round-trip %.1e, rho u coherent, pas physique" % err)


def test_scalar():
    """Bloc scalaire (1 var) : prim == cons -> conversion identite, round-trip exact."""
    rho, _, _, _ = _fields()
    spec = pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    s = pops.System(n=N, L=L, periodic=True)
    s.add_block("q", spec, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    name = s.variable_names("q", "primitive")[0]
    err = _roundtrip_err(s, "q", **{name: rho})
    assert err < 1e-15, "scalaire : round-trip (identite) non exact (%.2e)" % err
    print("OK  scalaire (1 var) : round-trip identite %.1e" % err)


def test_set_density_unchanged():
    """Pas de regression : set_density pose la densite (comp 0) + reste au repos."""
    rho, _, _, _ = _fields()
    spec = pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    s = pops.System(n=N, L=L, periodic=True)
    s.add_block("e", spec, spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    s.set_density("e", rho)
    U = np.array(s.get_state("e")).reshape(4, N, N)
    assert float(np.max(np.abs(U[0] - rho))) < 1e-15, "set_density : densite incorrecte"
    assert float(np.abs(U[1]).max()) == 0.0 and float(np.abs(U[2]).max()) == 0.0, "set_density : pas au repos"
    assert float(np.max(np.abs(U[3] - rho / (1.4 - 1.0)))) < 1e-13, "set_density : E != rho/(g-1)"
    print("OK  set_density inchange (densite posee, reste au repos, pas de regression)")


def test_errors():
    """Erreur CLAIRE sur un nom de primitive inconnu, et sur une primitive manquante."""
    rho, u, v, p = _fields()
    spec = pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0))
    s = pops.System(n=N, L=L, periodic=True)
    s.add_block("e", spec, spatial=pops.Spatial(minmod=True), time=pops.Explicit())

    try:
        s.set_primitive_state("e", rho=rho, u=u, v=v, p=p, bogus=p)
        raise AssertionError("aucune erreur sur une primitive inconnue")
    except ValueError as ex:
        assert "bogus" in str(ex), "message d'erreur sans le nom fautif : %s" % ex

    try:
        s.set_primitive_state("e", rho=rho, u=u)  # v, p manquantes
        raise AssertionError("aucune erreur sur des primitives manquantes")
    except ValueError as ex:
        assert "missing primitive(s)" in str(ex), "message sans la mention 'missing primitive(s)' : %s" % ex
    print("OK  erreurs claires : primitive inconnue + primitive manquante")


def test_dsl_compiled():
    """Modele DSL COMPILE (euler_poisson) : la conversion cons<->prim est forwardee par le .so, donc
    set/get_primitive_state round-trippent. AOT (add_compiled_block, ABI plate) ET JIT
    (add_dynamic_block, IModel virtuel). S'auto-saute sans compilateur C++."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  DSL compile : compilateur ou en-tetes pops absents")
        return
    # build_euler_poisson : Euler 4 var (rho, u, v, p) avec to_primitive / to_conservative declares
    # (set_primitive_state / set_conservative_from dans le DSL). cf. test_dsl_coupled.
    from test_dsl_coupled import build_euler_poisson

    rho, u, v, p = _fields()
    e = build_euler_poisson()
    tmp = tempfile.mkdtemp()
    try:
        # --- AOT : .so a ABI plate, charge par add_compiled_block ---
        so_aot = e.compile_or_jit(os.path.join(tmp, "ep_aot.so"), INCLUDE, mode="compile")
        a = pops.System(n=N, L=L, periodic=True)
        a.add_compiled_block("gas", so_aot, limiter="minmod", riemann="rusanov",
                             recon="conservative", names=["rho", "rho_u", "rho_v", "E"])
        assert a.variable_names("gas", "primitive") == ["rho", "u", "v", "p"]
        err_a = _roundtrip_err(a, "gas", rho=rho, u=u, v=v, p=p)
        assert err_a < 1e-13, "DSL AOT : round-trip cons<->prim non exact (%.2e)" % err_a
        # le pas tourne (etat conservatif physique) apres init depuis les primitives.
        a.set_poisson(rhs="charge_density")
        for _ in range(5):
            a.step_cfl(0.4)
        Ua = np.array(a.get_state("gas")).reshape(4, N, N)
        assert np.isfinite(Ua).all() and Ua[0].min() > 0, "DSL AOT : etat non physique apres init + pas"

        # --- JIT prototype : .so IModel virtuel, charge par add_dynamic_block ---
        so_jit = e.compile_so(os.path.join(tmp, "ep_jit.so"), INCLUDE)
        j = pops.System(n=N, L=L, periodic=True)
        j.add_dynamic_block("gas", so_jit, names=["rho", "rho_u", "rho_v", "E"])
        err_j = _roundtrip_err(j, "gas", rho=rho, u=u, v=v, p=p)
        assert err_j < 1e-13, "DSL JIT : round-trip cons<->prim non exact (%.2e)" % err_j
        print("OK  DSL compile : round-trip AOT %.1e + JIT %.1e, pas physique" % (err_a, err_j))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_compressible()
    test_isothermal()
    test_scalar()
    test_set_density_unchanged()
    test_errors()
    test_dsl_compiled()
    print("test_primitive_state : tout est vert")


if __name__ == "__main__":
    main()

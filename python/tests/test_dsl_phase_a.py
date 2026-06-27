"""DSL Phase A : l'API utilisateur stable (Model facade + Param + CompiledModel + add_equation +
FiniteVolume + run). PUR-PYTHON au-dessus de HyperbolicModel : aucune numerique nouvelle. cf.
docs/DSL_MODEL_DESIGN.md.

Deux niveaux :
(1) PUR-PYTHON (aucun compilateur requis) : Param nomme + runtime supporte (P7-b), flux vs eval_flux distincts,
    primitive_vars kwargs (layout ordonne, rho conservatif rejoint le layout sans etre redefini),
    FiniteVolume(riemann=), et les erreurs explicites (backend inconnu, target amr_system, weno5 sur
    .so, names= longueur, hllc sans pression, names= sur production natif).
(2) BOUT EN BOUT (saute si pas de compilateur / en-tetes) : compile(backend="aot") ET
    compile(backend="production") -> CompiledModel (adder correct : add_compiled_block vs
    add_native_block), add_equation + run ; aot et production donnent le MEME etat (memes briques de
    production). Prouve que production passe bien par le chemin NATIF add_native_block (#85), pas aot.
"""
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.variables import Primitive
from pops.numerics.reconstruction import WENO5
import os
import shutil
import tempfile

import numpy as np

import pops
from pops.codegen.loader import CompiledModel
from pops.ir.ops import sqrt
from pops.physics.facade import Model
from pops.physics.model import Param

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
GAMMA = 1.6667


def build_euler(name="euler_pa"):
    """Euler 2D ecrit via la FACADE Model (kwargs primitive_vars, param gamma nomme)."""
    m = Model(name)
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", GAMMA)                       # Param NOMME, inline au codegen + set_gamma
    u = rhou / rho
    v = rhov / rho
    p = (g - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    H = (E + p) / rho
    c = sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
           y=[rhov, rhov * u, rhov * v + p, rho * H * v])   # DECLARATEUR
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    # KWARGS : rho conservatif rejoint le layout ; renvoie les Var PRIMITIVES (rho/u/v/p comme
    # locales primitives), a utiliser dans conservative_from (qui exprime cons EN FONCTION des prim).
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([prho, prho * pu, prho * pv,
                         pp / (g - 1.0) + 0.5 * prho * (pu * pu + pv * pv)])
    return m


def build_euler_predef(name="euler_predef"):
    """IDENTIQUE a build_euler, mais u/v/p sont des Var PRIMITIVES deja definies (m.primitive(...))
    passees en SELF-REFERENCE a primitive_vars(rho=rho, u=u, v=v, p=p). C'est le style cible avec des
    Var pre-definies : sans le garde-fou self-ref, u=u redefinirait la primitive en `const Real u = u;`
    (auto-init -> NaN). Doit produire le MEME modele que build_euler (formes equivalentes)."""
    m = Model(name)
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", GAMMA)
    u = m.primitive("u", rhou / rho)                  # Var PRIMITIVE deja definie
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
           y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)   # u=u : Var primitive self-ref
    m.conservative_from([prho, prho * pu, prho * pv,
                         pp / (g - 1.0) + 0.5 * prho * (pu * pu + pv * pv)])
    return m


def initial_state(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((4, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    U[3] = 1.0 / (GAMMA - 1.0)
    return U.reshape(-1).tolist()


def expect_raises(exc, fn, label):
    try:
        fn()
    except exc:
        print("OK  %s : %s levee" % (label, exc.__name__))
        return
    raise AssertionError("%s : %s attendue, non levee" % (label, exc.__name__))


def pure_python_checks():
    # Param nomme + identite ; runtime SUPPORTE (P7-b)
    m = build_euler()
    g = m.params["gamma"]
    assert isinstance(g, Param) and g.name == "gamma" and abs(g.value - GAMMA) < 1e-12 \
        and g.kind == "const", "Param identite"
    assert abs(float(g) - GAMMA) < 1e-12, "Param float()"
    # P7-b : les parametres runtime sont desormais implementes (cf. test_dsl_runtime_params). L'ancienne
    # assertion "runtime rejete -> NotImplementedError" etait perimee depuis l'arrivee de la feature et
    # echouait en silence (CI auto-decouverte avalant l'echec, cf. ADC-104).
    kp = m.param("kappa", 1.0, kind="runtime")
    assert isinstance(kp, Param) and kp.name == "kappa" and kp.kind == "runtime" \
        and abs(kp.value - 1.0) < 1e-12, "param runtime supporte (Param kind='runtime')"
    print("OK  Param nomme (name/value/kind) + runtime supporte (P7-b)")

    # flux declarateur vs eval_flux evaluateur : noms distincts, methodes distinctes
    assert m.flux is not m.eval_flux, "flux et eval_flux doivent etre distincts"
    assert m._m._flux.get("x"), "m.flux(...) a bien declare le flux x"
    print("OK  m.flux (declarateur) != m.eval_flux (evaluateur)")

    # primitive_vars kwargs : layout ordonne [rho,u,v,p] ; rho (conservatif) PAS redefini en primitive
    assert m.prim_state == ["rho", "u", "v", "p"], "layout primitif kwargs : %r" % m.prim_state
    assert "rho" not in m._m.prim_defs, "rho conservatif ne doit pas etre redefini comme primitive"
    assert "u" in m._m.prim_defs and "p" in m._m.prim_defs, "u/p definis comme primitives"
    print("OK  primitive_vars kwargs : layout ordonne, rho conservatif rejoint le layout")

    # FiniteVolume : riemann (PAS flux) -> Spatial.flux ; variables -> recon
    fv = pops.FiniteVolume(limiter=Minmod(), riemann=HLLC(), variables=Primitive())
    assert fv.flux == "hllc" and fv.limiter == "minmod" and fv.recon == "primitive", \
        "FiniteVolume(riemann=) -> Spatial.flux"
    print("OK  FiniteVolume(limiter=, riemann=, variables=) remappe sur Spatial")

    # compile : backend inconnu rejete AVANT toute compilation ; target='amr_system' n'existe que pour
    # le backend natif "production" (DSL Phase D : le loader inline add_compiled_model(AmrSystem&)),
    # donc le demander avec un autre backend (aot) leve ValueError.
    expect_raises(ValueError, lambda: m.compile("x.so", INCLUDE, backend="bogus"),
                  "backend inconnu")
    expect_raises(ValueError,
                  lambda: m.compile("x.so", INCLUDE, backend="aot", target="amr_system"),
                  "target amr_system hors backend production")

    # add_equation : erreurs sur un CompiledModel FACTICE (pas de .so reel necessaire, les gardes
    # levent AVANT la frontiere C++).
    sys = pops.System(n=16, periodic=True)
    fake = CompiledModel(so_path="/inexistant.so", backend="aot", adder="add_compiled_block",
                             cons_names=["rho", "rho_u", "rho_v", "E"],
                             cons_roles=["Density", "MomentumX", "MomentumY", "Energy"],
                             prim_names=["rho", "u", "v"],  # PAS de 'p' -> hllc/roe doit lever
                             n_vars=4, gamma=GAMMA, n_aux=3, params={}, caps={},
                             abi_key="k", model_hash="h", cxx="c++", std="c++20")
    # WENO5 est desormais ACCEPTE sur aot/production (la grille .so / le bloc natif allouent
    # block_n_ghost(limiter) = 3 ghosts) : un fake aot+weno5 passe le garde Python et echoue plus loin
    # au dlopen (.so inexistant) -> RuntimeError, PAS ValueError (la garde weno5-aot n'existe plus).
    expect_raises(RuntimeError, lambda: sys.add_equation("g", fake,
                  spatial=pops.FiniteVolume(limiter=WENO5())), "weno5 aot : accepte (echec au dlopen)")
    # WENO5 reste rejete (ValueError) sur le backend 'prototype' (JIT, residu hote Rusanov ordre 1,
    # sans assemble_rhs) : ce chemin n'a pas de stencil large a alimenter.
    fake_proto = CompiledModel(so_path="/inexistant.so", backend="prototype",
                                   adder="add_dynamic_block", cons_names=["rho", "rho_u", "rho_v", "E"],
                                   cons_roles=["Density", "MomentumX", "MomentumY", "Energy"],
                                   prim_names=["rho", "u", "v", "p"], n_vars=4, gamma=GAMMA, n_aux=3,
                                   params={}, caps={}, abi_key="k", model_hash="h", cxx="c++",
                                   std="c++20")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake_proto,
                  spatial=pops.FiniteVolume(limiter=WENO5())), "weno5 sur prototype (JIT)")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake,
                  spatial=pops.FiniteVolume(riemann=HLLC())), "hllc sans pression")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake, names=["a", "b"]),
                  "names= mauvaise longueur")
    fake_prod = CompiledModel(so_path="/inexistant.so", backend="production",
                                  adder="add_native_block", cons_names=["rho"], cons_roles=["Density"],
                                  prim_names=["rho"], n_vars=1, gamma=None, n_aux=3, params={},
                                  caps={}, abi_key="k", model_hash="h", cxx="c++", std="c++20")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake_prod, names=["x"]),
                  "names= sur production natif")
    print("OK  add_equation : erreurs explicites (weno5/prototype, hllc sans p, names=, names= natif)")


def end_to_end_checks(cxx):
    n = 32
    tmp = tempfile.mkdtemp()
    finals = {}
    try:
        for backend, exp_adder in (("aot", "add_compiled_block"), ("production", "add_native_block")):
            m = build_euler("euler_%s" % backend)
            cm = m.compile(os.path.join(tmp, "m_%s.so" % backend), INCLUDE, backend=backend)
            assert isinstance(cm, CompiledModel), "compile -> CompiledModel"
            assert cm.backend == backend and cm.adder == exp_adder, \
                "%s : adder %r (attendu %r)" % (backend, cm.adder, exp_adder)
            assert cm.n_vars == 4 and abs((cm.gamma or 0) - GAMMA) < 1e-12, "metadonnees CompiledModel"
            assert cm.abi_key and cm.model_hash, "abi_key + model_hash presents"
            assert "gamma" in cm.params, "params porte le Param gamma"
            print("OK  %s : compile -> CompiledModel(adder=%s, n_vars=%d, abi_key=%.8s...)"
                  % (backend, cm.adder, cm.n_vars, cm.abi_key))

            s = pops.System(n=n, periodic=True)
            s.add_equation("gas", cm, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLLC(),
                                                               variables=Primitive()))
            s.set_poisson(rhs="charge_density", solver="geometric_mg")
            s.set_state("gas", initial_state(n))
            nsteps = s.run(t_end=0.02, cfl=0.4)
            assert nsteps > 0, "run a avance"
            finals[backend] = np.array(s.get_state("gas"))
            assert np.all(np.isfinite(finals[backend])), "%s : etat fini" % backend
            print("OK  %s : add_equation + run(%d pas) -> etat fini" % (backend, nsteps))

        # aot (host-marshale) et production (natif zero-copie) tournent les MEMES briques de
        # production -> etat final identique (preuve que production != prototype et est numerique-coherent).
        da = float(np.max(np.abs(finals["aot"] - finals["production"])))
        print("[phase A] dmax(aot vs production) = %.3e" % da)
        assert da < 1e-10, "aot et production doivent coincider (memes briques de production), dmax=%.3e" % da
        print("OK  aot == production (memes briques de production, dmax=%.3e)" % da)

        # Garde-fou self-ref kwargs (style cible avec Var pre-definies) : u/v/p definies par m.primitive
        # puis passees en primitive_vars(rho=rho, u=u, v=v, p=p). Doit (a) ne PAS produire de NaN
        # (sans le fix, u=u -> `Real u = u;` auto-init) et (b) donner le MEME modele que la forme expr.
        mp = build_euler_predef("euler_predef")
        cmp_ = mp.compile(os.path.join(tmp, "m_predef.so"), INCLUDE, backend="aot")
        sp = pops.System(n=n, periodic=True)
        sp.add_equation("gas", cmp_, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLLC(),
                                                              variables=Primitive()))
        sp.set_poisson(rhs="charge_density", solver="geometric_mg")
        sp.set_state("gas", initial_state(n))
        sp.run(t_end=0.02, cfl=0.4)
        pf = np.array(sp.get_state("gas"))
        assert np.all(np.isfinite(pf)), "primitive_vars kwargs (Var pre-definies) : etat fini, pas de NaN"
        dp = float(np.max(np.abs(pf - finals["aot"])))
        assert dp < 1e-10, "primitive_vars(u=u) Var pre-definie == forme expr (meme modele), dmax=%.3e" % dp
        print("OK  primitive_vars kwargs Var pre-definies : pas de NaN, == forme expr (dmax=%.3e)" % dp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def modelspec_substeps_check():
    """substeps= doit etre forwarde pour un ModelSpec (pas seulement pour un CompiledModel) : la
    branche ModelSpec d'add_equation appelle _s.add_block DIRECTEMENT avec nsub (pas self.add_block,
    qui retomberait sur time.substeps et IGNORERAIT l'override). Verifie via un espion sur _s.add_block."""
    s = pops.System(n=16, periodic=True)
    spec = pops.Model(state=pops.FluidState("isothermal", cs2=1.0), transport=pops.IsothermalFlux(),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=-1.0))
    calls = []

    class _Spy:
        def add_block(self, *a):
            calls.append(a)

    s._s = _Spy()
    # _s.add_block positional : (name, model, limiter, flux, recon, time_kind, substeps, evolve)
    s.add_equation("ions", spec, time=pops.Explicit(), substeps=10)
    assert calls, "add_equation(ModelSpec) doit appeler _s.add_block"
    assert calls[0][6] == 10, "substeps= ignore pour ModelSpec : recu %r" % (calls[0][6],)
    calls.clear()
    s.add_equation("ions2", spec, time=pops.Explicit(substeps=3))   # defaut = time.substeps
    assert calls[0][6] == 3, "defaut substeps != time.substeps : recu %r" % (calls[0][6],)
    print("OK  substeps= override forwarde pour ModelSpec (10) ; defaut = time.substeps (3)")


def predef_primitive_selfref_check():
    """primitive_vars(rho=rho, u=u, v=v, p=p) avec u/v/p des Var PRIMITIVES deja definies (m.primitive).
    Le garde-fou self-ref ne doit PAS redefinir u en `u = u` (auto-init NaN) : prim_defs garde la
    formule d'origine (rho_u/rho), pas un renvoi a soi. Pur-Python (aucun compilateur requis)."""
    m = build_euler_predef("euler_predef_pp")
    pd = m._m.prim_defs
    for nm in ("u", "v", "p"):
        assert nm in pd, "primitive '%s' absente de prim_defs" % nm
        assert pd[nm].to_cpp() != nm, \
            "primitive '%s' auto-initialisee (self-ref kwargs mal gere : `%s = %s;`)" % (nm, nm, nm)
    assert "rho_u" in pd["u"].deps(), "primitive 'u' doit garder sa formule (depend de rho_u)"
    print("OK  primitive_vars kwargs Var pre-definies : pas d'auto-init (formules prim_defs preservees)")


def main():
    pure_python_checks()
    predef_primitive_selfref_check()
    modelspec_substeps_check()
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  bout-en-bout (compilateur ou en-tetes pops absents)")
    else:
        end_to_end_checks(cxx)
    print("test_dsl_phase_a : tout est vert")


if __name__ == "__main__":
    main()

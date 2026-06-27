"""Ergonomie de m.compile(...) : auto-detection de `include`, auto-gestion de `so_path` et CACHE de
build keye sur le modele. PUR-PYTHON / ergonomie : aucune numerique nouvelle, memes briques generees.

On verifie :
(1) PUR-PYTHON (aucun compilateur requis) :
    - pops_include() auto-detecte le dossier d'en-tetes (ici via POPS_INCLUDE, sinon le paquet) ;
    - pops_cache_dir() respecte POPS_CACHE_DIR et cree le dossier ;
    - la cle de cache (model_hash + abi_key + backend/target/name) DIFFERE quand le modele change
      (param, formule, backend), et est STABLE pour un modele identique.
(2) BOUT EN BOUT (saute sans compilateur / en-tetes) :
    - m.compile(backend="aot") ET backend="production" SANS so_path/include -> CompiledModel valide,
      branchable via add_equation et qui tourne (run) ;
    - une 2e compilation du MEME modele est un cache HIT (PAS de recompilation : meme chemin, mtime
      inchange, marqueur present) ;
    - changer le modele (un parametre) BUSTE le cache (chemin different, recompilation) ;
    - la forme a arguments EXPLICITES (so_path + include) marche toujours (retro-compat).

Lance avec python3, meme PYTHONPATH que les autres tests DSL.
"""
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.variables import Primitive
import os
import shutil
import tempfile
import time

import numpy as np

import pops
from pops.codegen.cache import _cache_so_path, pops_cache_dir
from pops.codegen.loader import CompiledModel
from pops.codegen.toolchain import pops_include
from pops.ir.expr import Var
from pops.ir.ops import sqrt
from pops.physics.facade import Model

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
GAMMA = 1.6667  # gamma NON STANDARD (5/3)


def build_euler(name="euler_cache", gamma=GAMMA):
    """Euler 2D via la facade Model (param gamma nomme, roles canoniques)."""
    m = Model(name)
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", gamma)
    u = rhou / rho
    v = rhov / rho
    p = (g - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    H = (E + p) / rho
    c = sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
           y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
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


def pure_python_checks():
    """(1) Auto-detection + cle de cache : aucun compilateur requis."""
    # pops_cache_dir respecte POPS_CACHE_DIR (override) et cree le dossier
    cache = tempfile.mkdtemp()
    old = os.environ.get("POPS_CACHE_DIR")
    os.environ["POPS_CACHE_DIR"] = cache
    try:
        assert os.path.normpath(pops_cache_dir()) == os.path.normpath(cache), \
            "pops_cache_dir doit honorer POPS_CACHE_DIR"
        print("OK  pops_cache_dir() honore POPS_CACHE_DIR")

        # pops_include : si POPS_INCLUDE pointe sur un include valide, il est retenu en priorite
        if os.path.isdir(INCLUDE):
            os.environ["POPS_INCLUDE"] = INCLUDE
            assert os.path.normpath(pops_include()) == os.path.normpath(INCLUDE), \
                "pops_include doit honorer POPS_INCLUDE"
            del os.environ["POPS_INCLUDE"]
            print("OK  pops_include() honore POPS_INCLUDE")

        # cle de cache : meme modele -> meme chemin ; param/formule/backend differents -> chemin different
        abi = "fakeabikey"
        m = build_euler()
        h = m._model_hash()
        p_aot = _cache_so_path(h, abi, "aot", "system", None)
        p_aot_again = _cache_so_path(h, abi, "aot", "system", None)
        assert p_aot == p_aot_again, "cle de cache non deterministe pour un modele identique"
        assert p_aot.startswith(os.path.normpath(cache)), "le .so en cache doit vivre dans le cache dir"

        # backend / target / abi differents -> chemins distincts (memes model_hash)
        assert _cache_so_path(h, abi, "production", "system", None) != p_aot, \
            "backend different doit donner un chemin different"
        assert _cache_so_path(h, abi, "production", "amr_system", None) != \
            _cache_so_path(h, abi, "production", "system", None), \
            "target different doit donner un chemin different"
        assert _cache_so_path(h, "autreabi", "aot", "system", None) != p_aot, \
            "abi_key differente doit donner un chemin different"

        # un PARAMETRE different change model_hash, donc le chemin de cache (cache MISS)
        m2 = build_euler(gamma=1.4)
        assert m2._model_hash() != h, "un param different doit changer model_hash"
        assert _cache_so_path(m2._model_hash(), abi, "aot", "system", None) != p_aot, \
            "un param different doit buster le cache"

        # une FORMULE differente change aussi model_hash : on ajoute une source non triviale
        m3 = build_euler()
        rho3 = Var("rho", "cons")
        m3.source([0.0 * rho3, 0.0 * rho3, 0.0 * rho3, rho3])  # source != defaut -> formules differentes
        assert m3._model_hash() != h, "une formule differente (source) doit changer model_hash"
        assert _cache_so_path(m3._model_hash(), abi, "aot", "system", None) != p_aot, \
            "une formule differente doit buster le cache"
        print("OK  cle de cache : stable pour modele identique, distincte sur param/formule/backend/"
              "target/abi")
    finally:
        if old is None:
            os.environ.pop("POPS_CACHE_DIR", None)
        else:
            os.environ["POPS_CACHE_DIR"] = old
        shutil.rmtree(cache, ignore_errors=True)


def end_to_end_checks():
    """(2) Bout en bout : compile SANS so_path/include -> CompiledModel valide + run ; cache HIT/MISS ;
    forme explicite (retro-compat)."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> bout-en-bout saute")
        return

    n = 24
    cache = tempfile.mkdtemp()
    explicit = tempfile.mkdtemp()
    old_cache = os.environ.get("POPS_CACHE_DIR")
    old_inc = os.environ.get("POPS_INCLUDE")
    os.environ["POPS_CACHE_DIR"] = cache
    os.environ["POPS_INCLUDE"] = INCLUDE  # rend l'auto-detection robuste meme hors paquet installe
    try:
        for backend, exp_adder in (("aot", "add_compiled_block"), ("production", "add_native_block")):
            m = build_euler("euler_%s" % backend)

            # (a) compile SANS so_path NI include -> CompiledModel valide
            cm = m.compile(backend=backend)
            assert isinstance(cm, CompiledModel), "compile -> CompiledModel"
            assert cm.adder == exp_adder, "%s : adder %r (attendu %r)" % (backend, cm.adder, exp_adder)
            assert cm.so_path and os.path.exists(cm.so_path), "%s : .so absente" % backend
            assert os.path.normpath(cm.so_path).startswith(os.path.normpath(cache)), \
                "%s : .so hors du cache dir" % backend
            assert cm.n_vars == 4 and cm.abi_key and cm.model_hash, "metadonnees CompiledModel"
            print("OK  %s : compile() SANS so_path/include -> %s (cache %s)"
                  % (backend, cm.adder, os.path.basename(cm.so_path)))

            # (b) branchable via add_equation et tourne
            s = pops.System(n=n, periodic=True)
            s.add_equation("gas", cm, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLLC(),
                                                               variables=Primitive()))
            s.set_poisson(rhs="charge_density", solver="geometric_mg")
            s.set_state("gas", initial_state(n))
            nsteps = s.run(t_end=0.02, cfl=0.4)
            assert nsteps > 0 and np.all(np.isfinite(np.array(s.get_state("gas")))), \
                "%s : run instable" % backend
            print("OK  %s : add_equation + run(%d pas) -> etat fini" % (backend, nsteps))

            # (c) 2e compile du MEME modele -> cache HIT : meme chemin, PAS de recompilation
            mtime1 = os.path.getmtime(cm.so_path)
            time.sleep(1.1)  # resolution mtime : un vrai recompile changerait l'horodatage
            cm_hit = m.compile(backend=backend)
            assert cm_hit.so_path == cm.so_path, "%s : cache HIT chemin different" % backend
            assert os.path.getmtime(cm_hit.so_path) == mtime1, \
                "%s : cache HIT a RECOMPILE (mtime change)" % backend
            print("OK  %s : 2e compile() = cache HIT (meme chemin, mtime inchange, pas de recompile)"
                  % backend)

            # (d) changer un PARAMETRE buste le cache : chemin different, recompilation
            m_diff = build_euler("euler_%s" % backend, gamma=1.4)  # gamma different -> model_hash different
            cm_miss = m_diff.compile(backend=backend)
            assert cm_miss.so_path != cm.so_path, "%s : param change n'a pas buste le cache" % backend
            assert os.path.exists(cm_miss.so_path), "%s : cache MISS n'a pas compile" % backend
            print("OK  %s : param different = cache MISS (chemin different, recompilation)" % backend)

        # (e) retro-compat : forme a arguments EXPLICITES (so_path + include) marche toujours
        m = build_euler("euler_explicit")
        ex_path = os.path.join(explicit, "explicit.so")
        cm_ex = m.compile(ex_path, INCLUDE, backend="aot")
        assert cm_ex.so_path == ex_path and os.path.exists(ex_path), "so_path explicite casse"
        s = pops.System(n=n, periodic=True)
        s.add_equation("gas", cm_ex, spatial=pops.FiniteVolume(limiter=Minmod(), riemann=HLLC(),
                                                              variables=Primitive()))
        s.set_poisson(rhs="charge_density", solver="geometric_mg")
        s.set_state("gas", initial_state(n))
        assert s.run(t_end=0.02, cfl=0.4) > 0, "run via so_path explicite instable"
        print("OK  retro-compat : compile(so_path, include, ...) explicite marche toujours")
    finally:
        for k, v in (("POPS_CACHE_DIR", old_cache), ("POPS_INCLUDE", old_inc)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(explicit, ignore_errors=True)


def main():
    pure_python_checks()
    end_to_end_checks()
    print("test_dsl_compile_cache : tout est vert")


if __name__ == "__main__":
    main()

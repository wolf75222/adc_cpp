#!/usr/bin/env python3
"""ADC-186 : le cache de compilation DSL doit etre keye PAR BACKEND sur TOUS les chemins.

Contexte (decouvert en basculant les drivers hyqmom15 en backend production). La cle du cache
hors-source incluait deja le backend (chemins distincts par backend). Restait un cache aveugle au
backend : le cache de HANDLES du chargeur dynamique (dlopen / dyld), keye PAR CHEMIN. Recompiler un
.so 'production' sur un chemin ou un .so 'aot' a deja ete charge dans le MEME process fait resservir
l'ancien handle aot -> add_native_block echoue sur 'pops_native_abi_key absent'. Le chemin so_path
EXPLICITE etant fige par l'appelant, deux backends s'y ecrasaient.

Le fix (python/pops/dsl.py) : un registre EN PROCESS du backend ecrit a chaque chemin ; un so_path
explicite deja occupe par un AUTRE backend est redirige vers un frere distinct ('.<backend>.so') pour
que dlopen recharge un handle neuf. Le cache hors-source reste keye par backend (non-regression).

On verifie :
 (1) PUR-PYTHON (aucun compilateur) : semantique du registre + redirection backend ;
 (2) DECISION (compilateur + en-tetes pops requis ; auto-skip sinon) :
     (a) so_path explicite, aot PUIS production au meme chemin -> chemins RETENUS distincts, chacun
         portant les symboles de SON backend (production exporte pops_native_abi_key, aot non) ;
     (b) l'inverse production puis aot -> idem ;
     (c) cache hors-source aot puis production sans so_path -> chemins distincts, symboles corrects ;
 (3) CHARGEMENT NATIF (production chargeable dans cet environnement ; auto-skip sinon) :
     (a) bout en bout : add_equation(aot) PUIS, au meme chemin, add_equation(production) reussit --
         reproduit le bug (sur le code d'avant le fix, dlopen ressert l'ancien handle aot).

Lance avec python3, meme PYTHONPATH que les autres tests DSL.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

import pops
from pops.codegen.cache import _backend_distinct_so_path, _process_so_backend, _record_so_backend
from pops.ir.ops import sqrt
from pops.physics.facade import Model

INCLUDE = os.environ.get("POPS_INCLUDE") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "include"))

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def build_euler(name="euler_cacheb", gamma=1.6667):
    """Euler 2D minimal via la facade Model (param gamma nomme, roles canoniques)."""
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
    U[3] = 1.0 / (1.6667 - 1.0)
    return U.reshape(-1).tolist()


def is_production_so(path):
    """True si le .so exporte pops_native_abi_key (artefact backend production), False sinon (aot).

    nm (POSIX) si present ; sinon scan d'octets du fichier (le nom d'un symbole EXPORTE figure dans la
    table de chaines, ELF comme Mach-O). Les deux distinguent production (exporte) d'aot (n'exporte
    pas) -- verifie sur les artefacts reels."""
    nm = shutil.which("nm")
    if nm:
        try:
            out = subprocess.check_output([nm, "-gU", path], stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                toks = line.split()
                if toks and toks[-1].lstrip("_") == "pops_native_abi_key":
                    return True
            return False
        except Exception:  # noqa: BLE001  (nm absent d'un format -> repli octets)
            pass
    with open(path, "rb") as f:
        return b"pops_native_abi_key" in f.read()


def pure_python_checks():
    """(1) Registre EN PROCESS + redirection backend : aucun compilateur requis."""
    # etat propre pour ce sous-test (le registre est global au process)
    _process_so_backend.clear()
    p = "/tmp/adc186/model.so"

    # chemin inconnu -> inchange
    chk(_backend_distinct_so_path(p, "production") == p,
        "chemin neuf : inchange (aucun backend ne l'occupe)")

    # une fois 'aot' enregistre, un AUTRE backend est redirige, le MEME backend ne l'est pas
    _record_so_backend(p, "aot")
    chk(_backend_distinct_so_path(p, "aot") == p,
        "meme backend : chemin inchange (pas de collision)")
    redir = _backend_distinct_so_path(p, "production")
    chk(redir == "/tmp/adc186/model.production.so",
        "backend different : redirige vers le frere '.production.so' (recu %r)" % (redir,))

    # la redirection insere le backend AVANT l'extension (frere lisible, pas d'ecrasement)
    _record_so_backend("/a/b.so", "production")
    chk(_backend_distinct_so_path("/a/b.so", "hybrid-aot") == "/a/b.hybrid-aot.so",
        "le backend est insere avant l'extension")
    _process_so_backend.clear()


def _compile_probe():
    """Tente une compilation aot minimale ; renvoie (ok, raison) -- sert a auto-skip sans compilateur
    ni en-tetes (ni Kokkos sous Kokkos-only)."""
    if not os.path.isdir(INCLUDE):
        return False, "en-tetes pops introuvables (%s)" % INCLUDE
    if not (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")):
        return False, "aucun compilateur C++"
    try:
        d = tempfile.mkdtemp()
        build_euler("euler_probe").compile(os.path.join(d, "probe.so"), INCLUDE, backend="aot")
        shutil.rmtree(d, ignore_errors=True)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, "%s : %s" % (type(exc).__name__, str(exc).splitlines()[0])


def decision_checks():
    """(2) Decision de chemin keye par backend (compile reel, PAS de chargement) : symboles via nm."""
    ok, why = _compile_probe()
    if not ok:
        print("skip  decision_checks : %s" % why)
        return

    # (a) so_path EXPLICITE : aot puis production au MEME chemin -> chemins RETENUS distincts
    d = tempfile.mkdtemp()
    so = os.path.join(d, "model.so")
    cm_aot = build_euler().compile(so, INCLUDE, backend="aot")
    cm_prod = build_euler().compile(so, INCLUDE, backend="production")
    chk(cm_prod.so_path != cm_aot.so_path,
        "(a) production NE reutilise PAS le chemin de l'artefact aot (aot=%s prod=%s)"
        % (os.path.basename(cm_aot.so_path), os.path.basename(cm_prod.so_path)))
    chk(cm_aot.so_path == so,
        "(a) le chemin aot explicite est preserve (retro-compat)")
    chk(is_production_so(cm_prod.so_path),
        "(a) l'artefact production retenu exporte pops_native_abi_key")
    chk(not is_production_so(cm_aot.so_path),
        "(a) l'artefact aot (intouche) n'exporte pas pops_native_abi_key")

    # (b) l'inverse : production PUIS aot au meme chemin -> chemins distincts, symboles coherents
    d2 = tempfile.mkdtemp()
    so2 = os.path.join(d2, "model2.so")
    cm_prod2 = build_euler().compile(so2, INCLUDE, backend="production")
    cm_aot2 = build_euler().compile(so2, INCLUDE, backend="aot")
    chk(cm_aot2.so_path != cm_prod2.so_path,
        "(b) aot NE reutilise PAS le chemin de l'artefact production")
    chk(is_production_so(cm_prod2.so_path) and not is_production_so(cm_aot2.so_path),
        "(b) chaque artefact porte les symboles de SON backend")

    # (c) cache hors-source (sans so_path) : aot puis production -> chemins distincts par backend
    cache = tempfile.mkdtemp()
    old = os.environ.get("POPS_CACHE_DIR")
    os.environ["POPS_CACHE_DIR"] = cache
    try:
        mc = build_euler("euler_cacheb_c")
        cm_c_aot = mc.compile(backend="aot", include=INCLUDE)
        cm_c_prod = build_euler("euler_cacheb_c").compile(backend="production", include=INCLUDE)
        chk(cm_c_prod.so_path != cm_c_aot.so_path,
            "(c) cache hors-source : aot et production sur des chemins distincts")
        chk(is_production_so(cm_c_prod.so_path) and not is_production_so(cm_c_aot.so_path),
            "(c) cache hors-source : symboles corrects par backend")
    finally:
        if old is None:
            os.environ.pop("POPS_CACHE_DIR", None)
        else:
            os.environ["POPS_CACHE_DIR"] = old
        shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(d2, ignore_errors=True)


def _native_load_probe():
    """True si un .so production DSL se CHARGE dans cet environnement (module _pops compatible Kokkos).
    Faux sur un module SERIE local (les .so DSL referencent Kokkos non exporte par le module)."""
    try:
        d = tempfile.mkdtemp()
        cm = build_euler("euler_loadprobe").compile(os.path.join(d, "lp.so"), INCLUDE,
                                                    backend="production")
        s = pops.System(n=8, periodic=True)
        s.add_equation("g", cm, spatial=pops.FiniteVolume(limiter="minmod", riemann="hllc",
                                                         variables="primitive"))
        shutil.rmtree(d, ignore_errors=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def native_load_checks():
    """(3) Bout en bout : aot CHARGE puis production au MEME chemin -> add_native_block reussit.

    Reproduit le bug d'origine : sur le code d'avant le fix, dlopen ressert l'ancien handle aot au
    chemin production (pops_native_abi_key absent). Auto-skip si le module ne charge pas les .so DSL."""
    if not _native_load_probe():
        print("skip  native_load_checks : module _pops ne charge pas les .so DSL ici (Kokkos serie ?)")
        return
    n = 16
    d = tempfile.mkdtemp()
    so = os.path.join(d, "model.so")

    # charger l'artefact AOT au chemin so (peuple le cache de handles dlopen pour ce chemin)
    cm_aot = build_euler().compile(so, INCLUDE, backend="aot")
    s_aot = pops.System(n=n, periodic=True)
    s_aot.add_equation("gas", cm_aot, spatial=pops.FiniteVolume(limiter="minmod", riemann="hllc",
                                                              variables="primitive"))

    # recompiler PRODUCTION au MEME chemin, puis brancher via add_native_block : doit reussir
    cm_prod = build_euler().compile(so, INCLUDE, backend="production")
    try:
        s_prod = pops.System(n=n, periodic=True)
        s_prod.add_equation("gas", cm_prod, spatial=pops.FiniteVolume(limiter="minmod", riemann="hllc",
                                                                    variables="primitive"))
        s_prod.set_poisson(rhs="charge_density", solver="geometric_mg")
        s_prod.set_state("gas", initial_state(n))
        steps = s_prod.run(t_end=0.01, cfl=0.4)
        ok = steps > 0 and np.all(np.isfinite(np.array(s_prod.get_state("gas"))))
        chk(ok, "(3a) add_native_block reussit apres un aot charge au meme chemin (run %d pas)" % steps)
    except Exception as exc:  # noqa: BLE001
        chk(False, "(3a) add_native_block a echoue : %s" % str(exc).splitlines()[0])
    shutil.rmtree(d, ignore_errors=True)


def main():
    print("== (1) registre/redirection (pur-python) ==")
    pure_python_checks()
    print("== (2) decision de chemin par backend (compile) ==")
    decision_checks()
    print("== (3) chargement natif bout en bout ==")
    native_load_checks()
    if fails:
        print("test_compile_cache_backend : %d echec(s)" % fails)
        raise SystemExit(1)
    print("test_compile_cache_backend : tout est vert")


if __name__ == "__main__":
    main()

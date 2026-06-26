"""Garde-fou de REGRESSION : la norme C++ du modele NATIF (backend="production") doit suivre celle du
LOADER (module _pops), sinon add_native_block rejette le bloc avec "incompatible ABI".

CONTEXTE (regression observee sur GH200). Le module _pops est compile en C++20 sous Kokkos (CUDA 12.x
n'offre pas -std=c++23 ; cf. POPS_CXX_STD dans CMakeLists.txt), en C++23 sinon. Avant le fix, le DSL
backend="production" figeait le std du modele natif a "c++23" en dur. Sous Kokkos cela donnait :
loader C++20 (__cplusplus=202002L) vs modele C++23 (__cplusplus!=202002L) -> les cles d'ABI (qui
encodent __cplusplus) divergeaient -> add_native_block levait "incompatible ABI" -> AUCUN cas ne
pouvait tourner en natif sur GH200. Le fix derive le std du modele natif de la norme reelle du loader
(pops.dsl.loader_cxx_std() / pops._pops.__cxx_std__), donc les cles concordent SUR TOUTE toolchain.

Ce test :
  1) verifie l'INVARIANT de norme : loader_cxx_std() == norme reellement bakee par le module
     (pops._pops.__cxx_std__), avec fallback sur le std encode dans abi_key() ;
  2) bout-en-bout : un modele trivial compile(backend="production") puis branche par add_native_block
     se charge SANS erreur d'ABI -- c'est exactement ce qui cassait sous Kokkos. Le test echouerait
     sous Kokkos avec l'ancien defaut c++23 (mismatch __cplusplus), il passe avec le std aligne.

S'auto-saute (exit 0) si aucun compilateur C++ ou les en-tetes adc sont absents (chemins JIT/AOT/natif
ont la meme convention). Cable au job CI Kokkos (OpenMP) : c'est le SEUL job ou loader != c++23, donc
le seul ou cette regression se manifeste.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops import dsl


INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
GAMMA = 1.4


def build_trivial_euler(name="euler_abistd"):
    """Modele euler 2D minimal en formules (suffisant pour exercer add_native_block en natif)."""
    e = dsl.HyperbolicModel(name)
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.set_primitive_state(rho, u, v, p)
    # emit_cpp_brick (backend production) exige set_conservative_from (4 expressions to_conservative).
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    return e


def _expected_std_from_module():
    """Norme attendue du loader, lue DIRECTEMENT du module (independamment de loader_cxx_std), pour
    constituer une reference croisee : __cxx_std__ (entier 20/23) sinon le std encode dans abi_key()."""
    n = getattr(pops._pops, "__cxx_std__", None)
    if isinstance(n, int) and n in (20, 23):
        return "c++%d" % n
    key = pops._pops.abi_key()
    for tok in str(key).split(";"):
        if tok.startswith("std="):
            val = tok[len("std="):].rstrip("Ll")
            if val.isdigit():
                return "c++23" if int(val) > 202002 else "c++20"
    raise AssertionError("impossible de deduire la norme du loader (ni __cxx_std__ ni abi_key std=)")


def check_std_invariant():
    """La norme retournee par loader_cxx_std() DOIT coincider avec la norme reelle du module charge."""
    got = dsl.loader_cxx_std()
    assert got in ("c++20", "c++23"), "loader_cxx_std() = %r (attendu c++20|c++23)" % got
    expected = _expected_std_from_module()
    assert got == expected, (
        "loader_cxx_std()=%r != norme du module %r : le modele natif serait compile avec un std "
        "different du loader -> __cplusplus divergent -> cle d'ABI incompatible" % (got, expected))
    print("OK  invariant de norme : loader_cxx_std()=%s == module _pops (%s)" % (got, expected))
    return got


def check_native_loads_without_abi_error(expected_std):
    """Le coeur du fix : compile(backend="production") avec le std PAR DEFAUT (derive du loader) puis
    add_native_block doit charger SANS "incompatible ABI". Sous Kokkos avec l'ancien defaut c++23 en
    dur, ce chemin levait ; avec le std aligne, il passe."""
    n = 16
    tmp = tempfile.mkdtemp()
    try:
        e = build_trivial_euler()
        # std laisse a None -> defaut par backend : production suit loader_cxx_std() (le fix).
        so = e.compile(os.path.join(tmp, "euler_abistd.so"), INCLUDE, backend="production")
        assert os.path.exists(so), "compile(backend='production') n'a pas produit de .so"

        sys = pops.System(n=n, L=1.0, periodic=True)
        # Si le std du modele != std du loader, add_native_block leve RuntimeError("incompatible ABI").
        try:
            sys._s.add_native_block("gas", so, limiter="minmod", riemann="rusanov",
                                    recon="conservative", time="explicit", gamma=GAMMA, substeps=1)
        except RuntimeError as ex:
            if "incompatible ABI" in str(ex):
                raise AssertionError(
                    "REGRESSION : add_native_block rejette le modele production (std du modele != "
                    "std du loader %s). C'est exactement le bug GH200 sous Kokkos. Detail : %s"
                    % (expected_std, ex))
            raise

        # Sanity end-to-end : un etat trivial + eval_rhs renvoie un residu fini (le bloc tourne vraiment).
        U = np.zeros((4, n, n))
        U[0] = 1.0
        U[3] = 1.0 / (GAMMA - 1.0)
        sys.set_state("gas", U.reshape(-1).tolist())
        R = np.array(sys.eval_rhs("gas"))
        assert R.size == 4 * n * n and np.all(np.isfinite(R)), "eval_rhs du bloc natif non fini"
        print("OK  production + add_native_block : charge SANS erreur d'ABI (std modele = loader %s)"
              % expected_std)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents")
        print("test_native_abi_std : OK (rien a compiler)")
        return

    expected_std = check_std_invariant()
    check_native_loads_without_abi_error(expected_std)
    print("test_native_abi_std : tout est vert")


if __name__ == "__main__":
    main()

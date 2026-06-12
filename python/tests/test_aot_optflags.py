"""Flags d'optimisation du backend AOT (compile_aot) : le .so AOT execute le MEME chemin de production
que le natif, il doit donc etre compile avec les MEMES flags ($ADC_DSL_OPTFLAGS, defaut -O3 -DNDEBUG)
et non un -O2 en dur (a -O2 sans -DNDEBUG le noyau marshale est ~1.48x).

Trois volets :
(a) LIGNE DE COMMANDE (hermetique, aucun compilateur ni Kokkos requis) : on intercepte la commande de
    compilation de compile_aot et on verifie qu'elle porte les flags attendus -- defaut -O3 -DNDEBUG,
    et $ADC_DSL_OPTFLAGS custom honore (define traceur).
(b) CLE DE CACHE (hermetique) : les flags entrent dans la cle (un .so -O2 perime n'est pas reservi),
    et la cle des backends natif/jit reste INCHANGEE (pas d'invalidation collaterale).
(c) PARITE NUMERIQUE (auto-skip sans compilateur / Kokkos) : un meme modele compile via aot et via
    production donne le MEME etat apres quelques pas (memes briques de production, memes flags).
"""
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import adc  # noqa: E402  (les chemins .so exigent le module natif, comme les tests AOT voisins)
from adc import dsl  # noqa: E402
from test_dsl_phase_a import INCLUDE, build_euler, initial_state  # noqa: E402


class _Captured(Exception):
    """Sentinelle : porte la commande de compilation interceptee (on coupe avant le vrai compilateur)."""

    def __init__(self, cmd):
        self.cmd = list(cmd)


def _capture_compile_aot_cmd(optflags_env):
    """Renvoie la liste d'arguments que compile_aot passerait au compilateur, SANS rien compiler.

    On neutralise les sondes toolchain (Kokkos / std / compilateur) et on remplace _run_compile par
    une capture : le test reste valable meme sans compilateur ni Kokkos installes."""
    saved_env = os.environ.get("ADC_DSL_OPTFLAGS")
    saved = {nm: getattr(dsl, nm) for nm in (
        "_run_compile", "_native_kokkos_root", "_native_kokkos_compiler",
        "_native_kokkos_flags", "_probe_cxx_std")}
    if optflags_env is None:
        os.environ.pop("ADC_DSL_OPTFLAGS", None)
    else:
        os.environ["ADC_DSL_OPTFLAGS"] = optflags_env
    try:
        def _grab(cmd, what):
            raise _Captured(cmd)

        dsl._run_compile = _grab
        dsl._native_kokkos_root = lambda: "/dummy/kokkos"     # guard Kokkos-only franchi
        dsl._native_kokkos_compiler = lambda cxx=None: "c++"  # pas de which() reel
        dsl._native_kokkos_flags = lambda: ([], [])           # pas d'includes/libs Kokkos
        dsl._probe_cxx_std = lambda cc, std: std              # pas de probe -fsyntax-only
        m = build_euler("euler_optflags")
        try:
            m._m.compile_aot(os.path.join(tempfile.gettempdir(), "unused_optflags.so"), INCLUDE)
        except _Captured as c:
            return c.cmd
        raise AssertionError("compile_aot n'a pas atteint _run_compile (capture manquee)")
    finally:
        for nm, fn in saved.items():
            setattr(dsl, nm, fn)
        if saved_env is None:
            os.environ.pop("ADC_DSL_OPTFLAGS", None)
        else:
            os.environ["ADC_DSL_OPTFLAGS"] = saved_env


def check_default_flags():
    """Sans $ADC_DSL_OPTFLAGS, compile_aot doit batir a -O3 -DNDEBUG (parite natif), jamais -O2 en dur."""
    cmd = _capture_compile_aot_cmd(None)
    assert "-O3" in cmd, "compile_aot par defaut : -O3 absent (recu %r)" % (cmd,)
    assert "-DNDEBUG" in cmd, "compile_aot par defaut : -DNDEBUG absent (recu %r)" % (cmd,)
    assert "-O2" not in cmd, "compile_aot par defaut : -O2 en dur persiste (recu %r)" % (cmd,)
    print("OK  compile_aot defaut -> -O3 -DNDEBUG (plus de -O2 en dur)")


def check_env_override_honored():
    """$ADC_DSL_OPTFLAGS doit etre honore par compile_aot (meme variable que le chemin natif)."""
    cmd = _capture_compile_aot_cmd("-O2 -DADC_TEST_FLAG")
    assert "-O2" in cmd, "ADC_DSL_OPTFLAGS=-O2 ... non honore (recu %r)" % (cmd,)
    assert "-DADC_TEST_FLAG" in cmd, "define traceur ADC_DSL_OPTFLAGS non transmis (recu %r)" % (cmd,)
    assert "-O3" not in cmd and "-DNDEBUG" not in cmd, \
        "le defaut -O3 -DNDEBUG fuit malgre l'override (recu %r)" % (cmd,)
    print("OK  compile_aot honore $ADC_DSL_OPTFLAGS (-O2 -DADC_TEST_FLAG, define traceur transmis)")


def _old_cache_path(model_hash, abi_key, backend, target, name):
    """Reconstitue le nom de fichier .so AVANT le marqueur de schema aot (cle a 5 composantes)."""
    rest = "|".join((abi_key or "", backend or "", target or "", name or "",
                     dsl._platform_cache_key())).encode()
    tag = hashlib.sha256(rest).hexdigest()[:16]
    return os.path.join(dsl.adc_cache_dir(), "%s-%s.so" % ((model_hash or "nohash")[:16], tag))


def check_cache_key():
    """Les flags entrent dans la cle de l'artefact aot (un -O2 perime n'est pas reservi) ; le natif/jit
    gardent un nom inchange (pas d'invalidation collaterale)."""
    saved_env = os.environ.get("ADC_DSL_OPTFLAGS")
    aot_be = "aot;kokkos=on;kcfg=deadbeef"
    prod_be = "production;kokkos=on;kcfg=deadbeef"
    try:
        # (1) les optflags changent le nom du .so aot -> un binaire compile a d'autres flags est distinct
        os.environ.pop("ADC_DSL_OPTFLAGS", None)
        p_o3 = dsl._cache_so_path("mh", "abi", aot_be, "system", None)
        os.environ["ADC_DSL_OPTFLAGS"] = "-O2"
        p_o2 = dsl._cache_so_path("mh", "abi", aot_be, "system", None)
        assert p_o3 != p_o2, "cle de cache aot insensible aux optflags (%s == %s)" % (p_o3, p_o2)
        # (2) un .so aot compile avant l'alignement des flags (cle a 5 composantes) ne collisionne plus
        os.environ.pop("ADC_DSL_OPTFLAGS", None)
        assert dsl._cache_so_path("mh", "abi", aot_be, "system", None) \
            != _old_cache_path("mh", "abi", aot_be, "system", None), \
            "le .so aot -O2 d'avant le fix serait encore reservi (cle non invalidee)"
        # (3) le natif (cle deja fidele a son binaire) garde son nom de fichier : pas d'invalidation
        assert dsl._cache_so_path("mh", "abi", prod_be, "system", None) \
            == _old_cache_path("mh", "abi", prod_be, "system", None), \
            "la cle du backend natif a change (invalidation collaterale)"
    finally:
        if saved_env is None:
            os.environ.pop("ADC_DSL_OPTFLAGS", None)
        else:
            os.environ["ADC_DSL_OPTFLAGS"] = saved_env
    print("OK  cle de cache : optflags integres au .so aot, -O2 perime ecarte, natif inchange")


def _is_local_env_limitation(err):
    """True si l'echec releve de l'ENVIRONNEMENT local, pas d'une regression du code :
    - .so AOT non chargeable (symbole Kokkos absent du namespace plat : _adc serie / macOS deux-niveaux) ;
    - en-tetes du worktree != build du module _adc (signature en-tetes : _adc construit ailleurs).
    Aucun de ces cas n'arrive en CI (memes en-tetes que _adc, runtime Kokkos charge). On NE masque PAS un
    echec de compilation ('la compilation du .so ... a echoue') ni un ecart de parite (AssertionError)."""
    msg = str(err)
    if "dlopen" in msg and "Kokkos" in msg:
        return True
    return "NE CORRESPONDENT PAS" in msg or "signature en-tetes" in msg


def check_numeric_parity():
    """aot (host-marshale, desormais -O3 -DNDEBUG) et production (natif) tournent les memes briques de
    production -> meme etat apres quelques pas. On compile d'abord la .so AOT aux flags courants (preuve
    BRUYANTE que les flags sont acceptes par le compilateur), puis on tente la parite end-to-end ; ce
    second volet depend de l'environnement (natif: en-tetes == module ; aot: .so chargeable) et se SKIP
    proprement si l'env local ne le permet pas. Auto-skip aussi sans compilateur / Kokkos."""
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE) or dsl._native_kokkos_root() is None:
        print("skip  parite numerique (compilateur, en-tetes adc ou Kokkos absents)")
        return
    n = 32
    tmp = tempfile.mkdtemp()
    finals = {}
    try:
        # (1) compile_aot REEL aux flags courants : un echec ici (flags invalides) DOIT etre bruyant.
        cm_aot = build_euler("euler_optflags_aot").compile(
            os.path.join(tmp, "m_aot.so"), INCLUDE, backend="aot")
        print("OK  compile_aot produit une .so aux flags courants (acceptes par le compilateur)")
        # (2) parite end-to-end aot vs production (memes briques, memes flags). Depend de l'env local.
        try:
            cm_prod = build_euler("euler_optflags_production").compile(
                os.path.join(tmp, "m_prod.so"), INCLUDE, backend="production")
            for backend, cm in (("aot", cm_aot), ("production", cm_prod)):
                s = adc.System(n=n, periodic=True)
                s.add_equation("gas", cm, spatial=adc.FiniteVolume(limiter="minmod", riemann="hllc",
                                                                   variables="primitive"))
                s.set_poisson(rhs="charge_density", solver="geometric_mg")
                s.set_state("gas", initial_state(n))
                nsteps = s.run(t_end=0.02, cfl=0.4)
                assert nsteps > 0, "%s : run n'a pas avance" % backend
                finals[backend] = np.array(s.get_state("gas"))
                assert np.all(np.isfinite(finals[backend])), "%s : etat non fini" % backend
        except RuntimeError as e:
            if _is_local_env_limitation(e):
                print("skip  parite end-to-end : env local (%s) ; compilation AOT deja validee"
                      % str(e).splitlines()[0])
                return
            raise
        da = float(np.max(np.abs(finals["aot"] - finals["production"])))
        # meme seuil que test_dsl_phase_a (parite aot==production) : dmax absolu < 1e-10
        assert da < 1e-10, "aot != production apres alignement des flags (dmax=%.3e)" % da
        print("OK  parite aot == production (memes flags, memes briques, dmax=%.3e)" % da)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    check_default_flags()
    check_env_override_honored()
    check_cache_key()
    check_numeric_parity()
    print("test_aot_optflags : tout est vert")


if __name__ == "__main__":
    main()

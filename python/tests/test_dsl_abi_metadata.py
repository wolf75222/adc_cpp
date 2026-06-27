"""ABI etendue des blocs .so : un bloc compile (AOT add_compiled_block) ou charge a l'execution (JIT
add_dynamic_block) transporte desormais ses METADONNEES via des symboles extern "C" OPTIONNELS lus par
dlsym : noms de variables (pops_compiled_var_names), roles physiques (pops_compiled_roles), indice
adiabatique (pops_compiled_gamma). Avant, ces metadonnees etaient PERDUES a la frontiere du .so : le
System retombait sur un fallback (noms u0.., aucun role, gamma=1.4).

Ce test verifie sur les DEUX chemins (.so AOT et JIT) :
(1) PROPAGATION : un modele aux roles + gamma NON STANDARD compile en .so, charge dans le System,
    expose ses BONS noms/roles/gamma via variable_names / variable_roles / block_gamma (pas le fallback).
(2) RETRO-COMPATIBILITE : un .so ANCIEN (ABI legacy, SANS les symboles de metadonnees) charge toujours
    et retombe proprement sur le fallback (noms u0.., roles 'custom', gamma 1.4) -- aucune regression.

Exige un compilateur C++ + les en-tetes pops (sinon saute, comme test_dsl_aot). Lance avec python3.
"""
import os
import shutil
import subprocess
import tempfile

import pops
from pops.ir.ops import sqrt
from pops.physics.model import HyperbolicModel

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
GAMMA = 1.6667  # gamma NON STANDARD (monoatomique 5/3), distinct du defaut historique 1.4


def build_shuffled_euler():
    """Euler au layout NON STANDARD [my, E, mx, rho] (roles explicites) + gamma 5/3. Sert a prouver
    que noms, roles ET gamma traversent l'ABU du .so au lieu du fallback. La physique reste Euler."""
    e = HyperbolicModel("euler_meta")
    my, E, mx, rho = e.conservative_vars(
        "my", "ee", "mx", "rho",
        roles=["MomentumY", "Energy", "MomentumX", "Density"])
    u = e.primitive("u", mx / rho)
    v = e.primitive("v", my / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(GAMMA * p / rho)
    e.set_flux(x=[mx * v, rho * H * u, mx * u + p, mx],
               y=[my * v + p, rho * H * v, my * u, my])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.set_primitive_state(rho, u, v, p,
                          roles=["Density", "VelocityX", "VelocityY", "Pressure"])
    e.set_conservative_from([rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v),
                             rho * u, rho])
    e.set_gamma(GAMMA)  # transporte via pops_compiled_gamma
    return e


# Briques generees (sans set_gamma ni roles utiles) pour fabriquer un .so ANCIEN (ABI legacy) : on les
# enveloppe a la main dans l'ABI HISTORIQUE, SANS appeler les macros de metadonnees. Reproduit un .so
# d'avant ce chantier : aucun des symboles pops_compiled_var_names / _roles / _gamma n'est defini.
def build_legacy_scalar():
    """Transport scalaire (1 var, nom 'q' sans role canonique). Brique simple pour le .so legacy."""
    e = HyperbolicModel("legacy_q")
    (q,) = e.conservative_vars("q")
    e.set_flux(x=[q], y=[q])
    e.set_eigenvalues(x=[q], y=[q])
    e.set_primitive_state(q)
    e.set_conservative_from([q])
    return e


def compile_src(src, so_path, std="c++20"):
    # adc_cpp est Kokkos-only : le .so legacy inclut compiled_block_abi.hpp -> for_each (#error sans
    # Kokkos). pops_loader_build_flags fournit compilateur + flags Kokkos (+ macOS -undefined dynamic_lookup),
    # symboles Kokkos resolus contre _pops au chargement.
    from pops.codegen.toolchain import pops_loader_build_flags
    cc, kflags_c, kflags_l = pops_loader_build_flags()
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "legacy.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        subprocess.run([cc, "-shared", "-fPIC", "-std=" + std, "-O2", *kflags_c, "-I", INCLUDE, cpp,
                        "-o", so_path, *kflags_l], check=True)
    return so_path


def legacy_aot_so(so_path):
    """.so AOT a l'ABI HISTORIQUE : macro POPS_DEFINE_COMPILED_BLOCK seule, AUCUNE metadonnee emise."""
    e = build_legacy_scalar()
    nv, bricks, composite = e._emit_bricks()
    src = ('#include <pops/runtime/builders/compiled/compiled_block_abi.hpp>\n'
           '#include <pops/physics/bricks/bricks.hpp>\n'
           '#include <pops/core/state/variables.hpp>\n'
           + bricks
           + '\nnamespace pops_generated { using AotModel = %s; }\n' % composite
           + 'POPS_DEFINE_COMPILED_BLOCK(pops_generated::AotModel)\n')  # PAS de POPS_EXPORT_BLOCK_METADATA
    return compile_src(src, so_path)


def legacy_jit_so(so_path):
    """.so JIT a l'ABI HISTORIQUE : fabrique pops_make_model seule, AUCUNE metadonnee emise."""
    e = build_legacy_scalar()
    nv, bricks, composite = e._emit_bricks()
    src = ('#include <pops/runtime/dynamic/dynamic_model.hpp>\n'
           '#include <pops/physics/bricks/bricks.hpp>\n'
           '#include <pops/core/state/variables.hpp>\n'
           + bricks
           + '\nnamespace pops_generated { using JitModel = %s; }\n' % composite
           + 'extern "C" int pops_model_nvars() { return %d; }\n' % nv
           + 'extern "C" void* pops_make_model() { return new pops::ModelAdapter<pops_generated::JitModel>(); }\n'
           + 'extern "C" void pops_destroy_model(void* p) { delete static_cast<pops::IModel<%d>*>(p); }\n' % nv)
    return compile_src(src, so_path)


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents")
        print("test_dsl_abi_metadata : OK (rien a compiler)")
        return

    tmp = tempfile.mkdtemp()
    try:
        e = build_shuffled_euler()
        n, L = 16, 1.0

        # (1a) PROPAGATION via le chemin AOT (add_compiled_block) --------------------------------
        so_aot = e.compile_or_jit(os.path.join(tmp, "meta_aot.so"), INCLUDE, mode="compile")
        s = pops.System(n=n, L=L, periodic=True)
        s.add_compiled_block("gas", so_aot, limiter="minmod", riemann="hllc", recon="primitive")
        # noms : ceux DU MODELE (layout shuffle), pas le fallback u0..u3
        assert s.variable_names("gas") == ["my", "ee", "mx", "rho"], \
            "AOT : noms != metadonnees du .so : %r" % s.variable_names("gas")
        assert s.variable_names("gas", "primitive") == ["rho", "u", "v", "p"], \
            "AOT : noms primitifs != metadonnees : %r" % s.variable_names("gas", "primitive")
        # roles : resolus par SENS dans le layout non standard
        assert s.variable_roles("gas") == ["momentum_y", "energy", "momentum_x", "density"], \
            "AOT : roles conservatifs != metadonnees : %r" % s.variable_roles("gas")
        assert s.variable_roles("gas", "primitive") == ["density", "velocity_x", "velocity_y", "pressure"], \
            "AOT : roles primitifs != metadonnees : %r" % s.variable_roles("gas", "primitive")
        # gamma : 5/3 transporte, pas le defaut 1.4
        assert abs(s.block_gamma("gas") - GAMMA) < 1e-12, \
            "AOT : gamma != metadonnees : %r" % s.block_gamma("gas")
        print("OK  AOT (add_compiled_block) : noms/roles/gamma propages depuis le .so (gamma=%.4f)"
              % s.block_gamma("gas"))

        # (1a-bis) names= explicite garde la priorite sur les noms du .so ; roles/gamma restent du .so
        s2 = pops.System(n=n, L=L, periodic=True)
        s2.add_compiled_block("gas", so_aot, limiter="minmod", riemann="hllc", recon="primitive",
                              names=["c0", "c1", "c2", "c3"])
        assert s2.variable_names("gas") == ["c0", "c1", "c2", "c3"], "names= devrait primer"
        assert s2.variable_roles("gas") == ["momentum_y", "energy", "momentum_x", "density"], \
            "roles du .so devraient rester malgre names= explicite"
        assert abs(s2.block_gamma("gas") - GAMMA) < 1e-12, "gamma du .so devrait rester malgre names="
        print("OK  AOT : names= prime sur les noms du .so, roles + gamma restent ceux du .so")

        # (1b) PROPAGATION via le chemin JIT (add_dynamic_block) ---------------------------------
        so_jit = e.compile_or_jit(os.path.join(tmp, "meta_jit.so"), INCLUDE, mode="jit")
        sj = pops.System(n=n, L=L, periodic=True)
        sj.add_dynamic_block("gas", so_jit, recon="minmod")
        assert sj.variable_names("gas") == ["my", "ee", "mx", "rho"], \
            "JIT : noms != metadonnees du .so : %r" % sj.variable_names("gas")
        assert sj.variable_roles("gas") == ["momentum_y", "energy", "momentum_x", "density"], \
            "JIT : roles != metadonnees du .so : %r" % sj.variable_roles("gas")
        assert sj.variable_roles("gas", "primitive") == ["density", "velocity_x", "velocity_y", "pressure"], \
            "JIT : roles primitifs != metadonnees : %r" % sj.variable_roles("gas", "primitive")
        assert abs(sj.block_gamma("gas") - GAMMA) < 1e-12, \
            "JIT : gamma != metadonnees : %r" % sj.block_gamma("gas")
        print("OK  JIT (add_dynamic_block) : noms/roles/gamma propages depuis le .so (gamma=%.4f)"
              % sj.block_gamma("gas"))

        # (1c) GARDE-FOU : names= de mauvaise longueur doit lever (sinon variable_names/roles desync)
        for mode, adder in (("AOT", lambda S: S.add_compiled_block("g", so_aot, names=["a", "b"])),
                            ("JIT", lambda S: S.add_dynamic_block("g", so_jit, names=["a", "b"]))):
            S = pops.System(n=n, L=L, periodic=True)
            try:
                adder(S)
            except Exception:
                print("OK  %s : names= de longueur 2 != 4 variables rejete" % mode)
            else:
                raise AssertionError("%s : names= de mauvaise longueur aurait du lever" % mode)

        # (2a) RETRO-COMPAT : .so AOT ANCIEN (sans metadonnees) -> fallback -----------------------
        old_aot = legacy_aot_so(os.path.join(tmp, "legacy_aot.so"))
        so = pops.System(n=n, L=L, periodic=True)
        so.add_compiled_block("scal", old_aot, limiter="none", riemann="rusanov")
        assert so.variable_names("scal") == ["u0"], \
            "AOT legacy : fallback noms attendu, recu %r" % so.variable_names("scal")
        assert so.variable_roles("scal") == ["custom"], \
            "AOT legacy : fallback roles attendu, recu %r" % so.variable_roles("scal")
        assert abs(so.block_gamma("scal") - 1.4) < 1e-12, \
            "AOT legacy : fallback gamma 1.4 attendu, recu %r" % so.block_gamma("scal")
        print("OK  AOT legacy (sans symboles de metadonnees) : charge + fallback u0../custom/1.4")

        # (2b) RETRO-COMPAT : .so JIT ANCIEN (sans metadonnees) -> fallback -----------------------
        old_jit = legacy_jit_so(os.path.join(tmp, "legacy_jit.so"))
        sjo = pops.System(n=n, L=L, periodic=True)
        sjo.add_dynamic_block("scal", old_jit, recon="none")
        assert sjo.variable_names("scal") == ["u0"], \
            "JIT legacy : fallback noms attendu, recu %r" % sjo.variable_names("scal")
        assert sjo.variable_roles("scal") == ["custom"], \
            "JIT legacy : fallback roles attendu, recu %r" % sjo.variable_roles("scal")
        assert abs(sjo.block_gamma("scal") - 1.4) < 1e-12, \
            "JIT legacy : fallback gamma 1.4 attendu, recu %r" % sjo.block_gamma("scal")
        print("OK  JIT legacy (sans symboles de metadonnees) : charge + fallback u0../custom/1.4")

        print("test_dsl_abi_metadata : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

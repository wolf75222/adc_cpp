"""Backend "production" (NATIF) du DSL : un modele euler_poisson ecrit en formules est compile en un
LOADER .so (compile_native / compile(backend="production")) qui inline le gabarit en-tete
adc::add_compiled_model<ProdModel>, puis branche dans le System via add_native_block.

A la difference du backend "aot" (add_compiled_block : .so a ABI plate, le bloc recalcule sur une
grille LOCALE et MARSHALE des tableaux), le loader natif installe le modele genere comme bloc NATIF
sur le CONTEXTE REEL du System (grid_context) -> le bloc tourne ZERO-COPIE le MEME chemin qu'add_block
(assemble_rhs, fill_boundary, foncteurs nommes device-clean). On verifie donc une parite STRICTE :

  1) eval_rhs ET potentiel du bloc "production" == bloc NATIF add_block (precision machine), pour
     plusieurs schemas (minmod+rusanov+conservatif, minmod+hllc+primitif = flux de production) ;
  2) une avance de quelques pas (SSPRK2) reste bit-identique au bloc natif (etat final) ;
  3) GARDE-FOU ABI : un loader dont la cle adc_native_abi_key est falsifiee est REJETE explicitement
     par add_native_block (pas d'UB silencieux a la frontiere C++).

CRUX (resolution de symboles a travers le dlopen) : le loader appelle des methodes hors-ligne du
module _adc (install_block / grid_context / ensure_aux_width). Elles sont exportees (ADC_EXPORT) et le
loader est compile avec -undefined dynamic_lookup ; le test echouerait au dlopen sinon.
"""
import os
import shutil
import tempfile

import numpy as np

import adc
from test_dsl_coupled import build_euler_poisson, GAMMA, INCLUDE


def _native_spec():
    """Le MEME modele euler_poisson, version NATIVE composee par briques (reference de parite)."""
    return adc.Model(state=adc.FluidState("compressible", gamma=GAMMA),
                     transport=adc.CompressibleFlux(),
                     source=adc.GravityForce(),
                     elliptic=adc.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))


def _initial_state(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((4, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    U[3] = 1.0 / (GAMMA - 1.0)
    return U


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents")
        print("test_dsl_production : OK (rien a compiler)")
        return

    e = build_euler_poisson()
    n, L = 48, 1.0
    U = _initial_state(n)
    Uflat = U.reshape(-1).tolist()
    spec = _native_spec()
    tmp = tempfile.mkdtemp()
    try:
        # Backend "production" via la facade : compile_native sous le capot (loader natif).
        so = e.compile(os.path.join(tmp, "euler_poisson_native.so"), INCLUDE, backend="production")
        assert e.adder_for("production") == "add_native_block"

        def build_native(limiter, riemann, recon, evolve=True):
            sys = adc.System(n=n, L=L, periodic=True)
            sys._s.add_native_block("gas", so, limiter=limiter, riemann=riemann, recon=recon,
                                    time="explicit", gamma=GAMMA, substeps=1, evolve=evolve)
            sys.set_poisson(rhs="charge_density", solver="geometric_mg")
            sys.set_state("gas", Uflat)
            return sys

        def build_ref(limiter, riemann, recon, evolve=True):
            sys = adc.System(n=n, L=L, periodic=True)
            lim = {"none": dict(none=True), "minmod": dict(minmod=True),
                   "vanleer": dict(vanleer=True)}[limiter]
            sys.add_block("gas", spec, spatial=adc.Spatial(flux=riemann, recon=recon, **lim),
                          time=adc.Explicit(), evolve=evolve)
            sys.set_poisson(rhs="charge_density", solver="geometric_mg")
            sys.set_state("gas", Uflat)
            return sys

        def compare(limiter, riemann, recon):
            prod = build_native(limiter, riemann, recon)
            prod.solve_fields()
            R_prod = np.array(prod.eval_rhs("gas")).reshape(4, n, n)
            phi_prod = np.array(prod.potential()).reshape(n, n)

            ref = build_ref(limiter, riemann, recon)
            ref.solve_fields()
            R_ref = np.array(ref.eval_rhs("gas")).reshape(4, n, n)
            phi_ref = np.array(ref.potential()).reshape(n, n)

            dphi = float(np.max(np.abs(phi_prod - phi_ref)))
            assert dphi < 1e-12, "%s : potentiel natif != add_block (%.2e)" % (riemann, dphi)
            assert float(np.max(np.abs(R_prod))) > 1e-3, "%s : residu trivial" % riemann
            dres = float(np.max(np.abs(R_prod - R_ref)))
            # Parite STRICTE : meme chemin compile (install_block), donc bit-identique (pas seulement < 1e-9).
            assert dres == 0.0, "%s : eval_rhs natif != add_block (ecart %.2e, attendu 0)" % (riemann, dres)
            print("OK  bloc production %s+%s : eval_rhs BIT-IDENTIQUE + potentiel == add_block"
                  % (limiter, riemann))

        compare("minmod", "rusanov", "conservative")
        compare("minmod", "hllc", "primitive")  # flux de production (pressure()/wave_speeds() generes)

        # (2) avance SSPRK2 : etat final bit-identique au bloc natif sur 12 pas a dt fixe (meme dt des
        # deux cotes -> pas de derive numerique possible si la numerique est la meme).
        prod = build_native("minmod", "hllc", "primitive")
        ref = build_ref("minmod", "hllc", "primitive")
        dt = 1e-3
        for _ in range(12):
            prod.step(dt)
            ref.step(dt)
        Up = np.array(prod.get_state("gas")).reshape(4, n, n)
        Ur = np.array(ref.get_state("gas")).reshape(4, n, n)
        dstep = float(np.max(np.abs(Up - Ur)))
        assert np.isfinite(Up).all() and Up[0].min() > 0, "etat de production non physique"
        assert float(np.abs(Up[1]).max()) > 1e-4, "la gravite n'a pas mis le gaz en mouvement"
        assert dstep == 0.0, "etat apres 12 pas natif != add_block (ecart %.2e, attendu 0)" % dstep
        print("OK  12 pas SSPRK2 : etat de production BIT-IDENTIQUE au bloc natif add_block")

        # (3) GARDE-FOU ABI : on compile un loader dont la SIGNATURE D'EN-TETES bakee est volontairement
        # FAUSSE (-DADC_HEADER_SIG different). Sa cle adc_native_abi_key differe alors de celle du module
        # -> add_native_block doit lever une erreur EXPLICITE. (On ne patche PAS le binaire : sur macOS
        # ARM cela invaliderait la signature ad-hoc et le noyau tuerait le process ; on recompile un .so
        # valide a la cle differente, ce qui teste exactement la frontiere d'ABI.)
        bad = _compile_wrong_abi(e, os.path.join(tmp, "euler_poisson_wrongabi.so"), cxx)
        sys = adc.System(n=n, L=L, periodic=True)
        raised = False
        try:
            sys._s.add_native_block("gas", bad, limiter="minmod", riemann="rusanov",
                                    recon="conservative", time="explicit", gamma=GAMMA)
        except RuntimeError as ex:
            raised = True
            assert "ABI incompatible" in str(ex), "message inattendu : %s" % ex
        assert raised, "add_native_block a accepte un loader a cle d'ABI fausse (UB silencieux)"
        print("OK  cle d'ABI divergente REJETEE explicitement par add_native_block")

        print("test_dsl_production : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _compile_wrong_abi(model, dst_so, cxx):
    """Compile le MEME loader natif mais avec une signature d'en-tetes FAUSSE (-DADC_HEADER_SIG bidon) :
    le .so produit est valide (signe par le compilateur) mais sa cle d'ABI differe de celle du module,
    ce qui doit declencher le rejet d'add_native_block. Renvoie le chemin du .so."""
    import subprocess
    import tempfile
    from adc.dsl import adc_loader_build_flags
    src = model.emit_cpp_native_loader()
    # adc_cpp est Kokkos-only : le loader inclut les en-tetes adc (for_each), il faut donc Kokkos +
    # (macOS) -undefined dynamic_lookup. adc_loader_build_flags fournit compilateur + flags ; on garde
    # une SIGNATURE D'EN-TETES FAUSSE (-DADC_HEADER_SIG bidon) pour que le .so compile mais soit REJETE
    # a l'ABI par add_native_block (le but du test).
    cc, kflags_c, kflags_l = adc_loader_build_flags(cxx)
    flags = ["-shared", "-fPIC", "-std=c++20", "-O2",
             "-DADC_HEADER_SIG=\"deadbeef_signature_volontairement_fausse\"", *kflags_c]
    with tempfile.TemporaryDirectory() as t:
        cpp = os.path.join(t, "wrong.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        subprocess.run([cc, *flags, "-I", INCLUDE, cpp, "-o", dst_so, *kflags_l], check=True)
    return dst_so


if __name__ == "__main__":
    main()

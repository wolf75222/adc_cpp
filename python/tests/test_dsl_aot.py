"""Backend "compile" (AOT) du DSL : un modele euler_poisson ecrit en formules est compile
ahead-of-time (compile_or_jit(mode="compile")) en une .so qui execute le chemin de PRODUCTION
(assemble_rhs<Limiter, Flux>, SSPRK2 du coeur), puis branche dans le System via add_compiled_block.

Contrairement au bloc dynamique (compile_so -> IModel, dispatch virtuel, Rusanov hote ordre 1), la
numerique du bloc AOT est censee etre IDENTIQUE a celle d'un bloc NATIF add_block (memes briques, meme
schema). On le verifie : pour le meme etat et le meme schema (minmod + rusanov + conservatif), le
residu eval_rhs et le potentiel du bloc AOT egalent ceux du euler_poisson natif a la precision machine.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from test_dsl_coupled import build_euler_poisson, GAMMA, INCLUDE


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents")
        print("test_dsl_aot : OK (rien a compiler)")
        return

    e = build_euler_poisson()
    n, L = 48, 1.0
    tmp = tempfile.mkdtemp()
    try:
        so = e.compile_or_jit(os.path.join(tmp, "euler_poisson_aot.so"), INCLUDE, mode="compile")

        xs = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xs, xs)
        U = np.zeros((4, n, n))
        U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
        U[3] = 1.0 / (GAMMA - 1.0)
        Uflat = U.reshape(-1).tolist()

        spec = pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                         transport=pops.CompressibleFlux(),
                         source=pops.GravityForce(),
                         elliptic=pops.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))

        # Pour chaque schema (rusanov, puis HLLC qui exige pressure() + wave_speeds() generes), le
        # bloc AOT doit etre IDENTIQUE au bloc NATIF add_block (memes briques, meme assemble_rhs).
        def compare(limiter, riemann, recon):
            aot = pops.System(n=n, L=L, periodic=True)
            aot.add_compiled_block("gas", so, limiter=limiter, riemann=riemann, recon=recon,
                                   time="explicit", names=["rho", "rho_u", "rho_v", "E"])
            aot.set_poisson(rhs="charge_density", solver="geometric_mg")
            aot.set_state("gas", Uflat)
            aot.solve_fields()
            R_aot = np.array(aot.eval_rhs("gas")).reshape(4, n, n)
            phi_aot = np.array(aot.potential()).reshape(n, n)

            nat = pops.System(n=n, L=L, periodic=True)
            lim = {"none": dict(none=True), "minmod": dict(minmod=True),
                   "vanleer": dict(vanleer=True)}[limiter]
            nat.add_block("gas", spec, spatial=pops.Spatial(flux=riemann, recon=recon, **lim),
                          time=pops.Explicit())
            nat.set_poisson(rhs="charge_density", solver="geometric_mg")
            nat.set_state("gas", Uflat)
            nat.solve_fields()
            R_nat = np.array(nat.eval_rhs("gas")).reshape(4, n, n)
            phi_nat = np.array(nat.potential()).reshape(n, n)

            dphi = float(np.max(np.abs(phi_aot - phi_nat)))
            assert dphi < 1e-9, "%s : potentiel AOT != natif (%.2e)" % (riemann, dphi)
            assert float(np.max(np.abs(R_aot))) > 1e-3, "%s : residu AOT trivial" % riemann
            dres = float(np.max(np.abs(R_aot - R_nat)))
            assert dres < 1e-9, "%s : eval_rhs AOT != natif (%.2e)" % (riemann, dres)
            print("OK  bloc AOT %s+%s : eval_rhs ET potentiel == bloc natif (ecart %.1e)"
                  % (limiter, riemann, max(dphi, dres)))

        compare("minmod", "rusanov", "conservative")
        compare("minmod", "hllc", "primitive")  # flux de production : pressure()/wave_speeds() generes

        # (C) avance de production (SSPRK2 + HLLC) : tourne, masse conservee, dynamique non triviale
        aot = pops.System(n=n, L=L, periodic=True)
        aot.add_compiled_block("gas", so, limiter="minmod", riemann="hllc", recon="primitive",
                               names=["rho", "rho_u", "rho_v", "E"])
        aot.set_poisson(rhs="charge_density", solver="geometric_mg")
        aot.set_state("gas", Uflat)
        mass0 = float(np.array(aot.get_state("gas")).reshape(4, n, n)[0].sum())
        for _ in range(15):
            aot.step_cfl(0.4)
        U1 = np.array(aot.get_state("gas")).reshape(4, n, n)
        drel = abs(float(U1[0].sum()) - mass0) / mass0
        assert np.isfinite(U1).all() and U1[0].min() > 0, "etat non physique"
        assert drel < 1e-9, "masse non conservee (drel=%.2e)" % drel
        assert float(np.abs(U1[1]).max()) > 1e-4, "la gravite n'a pas mis le gaz en mouvement"
        print("OK  bloc AOT avance dans le System (SSPRK2, 15 pas, masse drel=%.1e)" % drel)
        print("test_dsl_aot : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

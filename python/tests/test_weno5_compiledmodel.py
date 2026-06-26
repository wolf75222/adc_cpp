"""WENO5 sur les chemins .so / CompiledModel (AOT add_compiled_block ET production add_native_block).

Le verrou de ce chantier : les chemins compiles allouent desormais l'etat avec block_n_ghost(limiter)
(3 pour weno5, son stencil 5 points), comme System::add_block (PR #88) -- la grille locale du .so cote
AOT (compiled_block_abi.hpp), le bloc natif via add_compiled_model cote production. WENO5 etait avant
REJETE sur ces chemins ("limiter 'weno5' non expose ... 2 ghosts").

On verifie, pour le MEME euler_poisson et limiter="weno5", flux rusanov :
  (1) AOT (add_compiled_block) : plus de "limiter inconnu" / rejet ; eval_rhs ET potentiel egalent
      le bloc NATIF add_block weno5 a TOLERANCE SERREE (le .so recalcule sur une grille LOCALE et
      MARSHALE des tableaux plats -> arrondi de recopie, comme test_dsl_aot ; < 1e-9). Avance saine.
  (2) production (add_native_block, loader natif zero-copie) : eval_rhs ET potentiel BIT-IDENTIQUES
      au bloc natif add_block weno5 (MEME make_block / install_block / set_block_ghosts(3) sur les
      vrais MultiFab du System).
  (3) NO-DEFAULT-CHANGE : none/minmod restent <= 2 ghosts (set_block_ghosts no-op) -> AOT a la meme
      tolerance serree et production reste BIT-IDENTIQUE au natif. Allocation inchangee.

S'auto-saute (exit 0) sans compilateur / en-tetes adc, comme test_dsl_aot / test_dsl_production.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from test_dsl_coupled import build_euler_poisson, GAMMA, INCLUDE


def _native_spec():
    """euler_poisson NATIF compose par briques (reference de parite, cf. test_dsl_production)."""
    return pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                     transport=pops.CompressibleFlux(),
                     source=pops.GravityForce(),
                     elliptic=pops.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))


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
        print("test_weno5_compiledmodel : OK (rien a compiler)")
        return

    e = build_euler_poisson()
    n, L = 48, 1.0
    U = _initial_state(n)
    Uflat = U.reshape(-1).tolist()
    spec = _native_spec()
    tmp = tempfile.mkdtemp()
    try:
        so_aot = e.compile(os.path.join(tmp, "euler_poisson_aot.so"), INCLUDE, backend="aot")
        so_prod = e.compile(os.path.join(tmp, "euler_poisson_native.so"), INCLUDE,
                            backend="production")
        assert e.adder_for("aot") == "add_compiled_block"
        assert e.adder_for("production") == "add_native_block"

        # --- reference NATIVE add_block (oracle de parite) ---
        def ref(limiter):
            sys = pops.System(n=n, L=L, periodic=True)
            lim = {"none": dict(none=True), "minmod": dict(minmod=True),
                   "weno5": dict(weno5=True)}[limiter]
            sys.add_block("gas", spec, spatial=pops.Spatial(flux="rusanov", recon="conservative",
                                                           **lim), time=pops.Explicit())
            sys.set_poisson(rhs="charge_density", solver="geometric_mg")
            sys.set_state("gas", Uflat)
            sys.solve_fields()
            return (np.array(sys.eval_rhs("gas")).reshape(4, n, n),
                    np.array(sys.potential()).reshape(n, n))

        # --- (1) AOT : add_compiled_block accepte weno5, parite serree au natif ---
        def aot(limiter):
            sys = pops.System(n=n, L=L, periodic=True)
            sys.add_compiled_block("gas", so_aot, limiter=limiter, riemann="rusanov",
                                   recon="conservative", time="explicit",
                                   names=["rho", "rho_u", "rho_v", "E"])
            sys.set_poisson(rhs="charge_density", solver="geometric_mg")
            sys.set_state("gas", Uflat)
            sys.solve_fields()
            return (np.array(sys.eval_rhs("gas")).reshape(4, n, n),
                    np.array(sys.potential()).reshape(n, n))

        for limiter in ("none", "minmod", "weno5"):
            R_ref, phi_ref = ref(limiter)
            R_aot, phi_aot = aot(limiter)
            assert float(np.max(np.abs(R_aot))) > 1e-3, "%s : residu AOT trivial" % limiter
            dphi = float(np.max(np.abs(phi_aot - phi_ref)))
            dres = float(np.max(np.abs(R_aot - R_ref)))
            # Tolerance SERREE (pas bit-identique) : le .so AOT recalcule sur une grille LOCALE et
            # marshale des tableaux plats (arrondi de recopie), comme tout add_compiled_block.
            assert dphi < 1e-9, "AOT %s : potentiel != natif (%.2e)" % (limiter, dphi)
            assert dres < 1e-9, "AOT %s : eval_rhs != natif (%.2e)" % (limiter, dres)
            print("OK  AOT %s : add_compiled_block accepte + eval_rhs/potentiel == add_block (%.1e)"
                  % (limiter, max(dphi, dres)))

        # avance AOT weno5 : tourne, masse conservee, dynamique non triviale
        sys = pops.System(n=n, L=L, periodic=True)
        sys.add_compiled_block("gas", so_aot, limiter="weno5", riemann="rusanov",
                               recon="conservative", names=["rho", "rho_u", "rho_v", "E"])
        sys.set_poisson(rhs="charge_density", solver="geometric_mg")
        sys.set_state("gas", Uflat)
        mass0 = float(np.array(sys.get_state("gas")).reshape(4, n, n)[0].sum())
        for _ in range(15):
            sys.step_cfl(0.4)
        U1 = np.array(sys.get_state("gas")).reshape(4, n, n)
        drel = abs(float(U1[0].sum()) - mass0) / mass0
        assert np.isfinite(U1).all() and U1[0].min() > 0, "AOT weno5 : etat non physique"
        assert drel < 1e-9, "AOT weno5 : masse non conservee (drel=%.2e)" % drel
        print("OK  AOT weno5 avance dans le System (SSPRK2, 15 pas, masse drel=%.1e)" % drel)

        # --- (2) production : add_native_block accepte weno5, parite STRICTE (bit-identique) ---
        def prod(limiter):
            sys = pops.System(n=n, L=L, periodic=True)
            sys._s.add_native_block("gas", so_prod, limiter=limiter, riemann="rusanov",
                                    recon="conservative", time="explicit", gamma=GAMMA, substeps=1,
                                    evolve=True)
            sys.set_poisson(rhs="charge_density", solver="geometric_mg")
            sys.set_state("gas", Uflat)
            sys.solve_fields()
            return (np.array(sys.eval_rhs("gas")).reshape(4, n, n),
                    np.array(sys.potential()).reshape(n, n))

        for limiter in ("none", "minmod", "weno5"):
            R_ref, phi_ref = ref(limiter)
            R_prod, phi_prod = prod(limiter)
            assert float(np.max(np.abs(R_prod))) > 1e-3, "%s : residu production trivial" % limiter
            dphi = float(np.max(np.abs(phi_prod - phi_ref)))
            dres = float(np.max(np.abs(R_prod - R_ref)))
            # Meme chemin compile que add_block (install_block + set_block_ghosts) -> BIT-IDENTIQUE.
            assert dphi == 0.0, "production %s : potentiel != add_block (%.2e)" % (limiter, dphi)
            assert dres == 0.0, "production %s : eval_rhs != add_block (%.2e, attendu 0)" % (limiter,
                                                                                            dres)
            print("OK  production %s : add_native_block accepte + eval_rhs BIT-IDENTIQUE add_block"
                  % limiter)

        # avance production weno5 : etat final bit-identique au natif sur 12 pas a dt fixe.
        def build_prod_step():
            sys = pops.System(n=n, L=L, periodic=True)
            sys._s.add_native_block("gas", so_prod, limiter="weno5", riemann="rusanov",
                                    recon="conservative", time="explicit", gamma=GAMMA, substeps=1,
                                    evolve=True)
            sys.set_poisson(rhs="charge_density", solver="geometric_mg")
            sys.set_state("gas", Uflat)
            return sys

        def build_ref_step():
            sys = pops.System(n=n, L=L, periodic=True)
            sys.add_block("gas", spec, spatial=pops.Spatial(weno5=True, flux="rusanov",
                                                           recon="conservative"),
                          time=pops.Explicit())
            sys.set_poisson(rhs="charge_density", solver="geometric_mg")
            sys.set_state("gas", Uflat)
            return sys

        p_sys, r_sys = build_prod_step(), build_ref_step()
        dt = 1e-3
        for _ in range(12):
            p_sys.step(dt)
            r_sys.step(dt)
        Up = np.array(p_sys.get_state("gas")).reshape(4, n, n)
        Ur = np.array(r_sys.get_state("gas")).reshape(4, n, n)
        dstep = float(np.max(np.abs(Up - Ur)))
        assert np.isfinite(Up).all() and Up[0].min() > 0, "production weno5 : etat non physique"
        assert dstep == 0.0, "production weno5 : etat apres 12 pas != add_block (%.2e)" % dstep
        print("OK  production weno5 : 12 pas SSPRK2 BIT-IDENTIQUES au bloc natif add_block")

        print("test_weno5_compiledmodel : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

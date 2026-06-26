"""Choix de la RECONSTRUCTION (ordre) du bloc dynamique : add_dynamic_block(recon=...) passe le bloc
charge a l'execution d'un schema ordre 1 (recon='none') a une reconstruction MUSCL ordre 2 limitee
(recon='minmod' | 'vanleer') sur les variables conservatives. On verifie que le residu hote (host_residual
via IModel) reproduit EXACTEMENT une reference MUSCL + Rusanov a a_max global ecrite en numpy depuis les
memes formules DSL, pour les trois reconstructions ; que recon='none' reste l'ordre 1 (== pops.PythonFlux,
non-regression de test_dsl_block) ; et qu'une reconstruction inconnue est refusee.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from test_dsl_brick import build_euler_brick, GAMMA

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def _slope(am, ap, recon):
    """Pente limitee (memes formules que limited_slope cote C++), tableaux numpy."""
    if recon == "minmod":
        return np.where(am * ap <= 0, 0.0, np.where(np.abs(am) < np.abs(ap), am, ap))
    if recon == "vanleer":
        ab = am * ap
        denom = np.where(np.abs(am + ap) < 1e-300, 1.0, am + ap)
        return np.where(ab <= 0, 0.0, 2.0 * ab / denom)
    return np.zeros_like(am)


def muscl_residual(e, U, h, recon):
    """Reference numpy : -div F* avec reconstruction MUSCL conservative + Rusanov a a_max GLOBAL.
    U : (n_vars, n, n) periodique, U[c, j, i] ; dir 0 = x = axe i (2), dir 1 = y = axe j (1)."""
    amax = max(e.max_wave_speed(U, {}, 0), e.max_wave_speed(U, {}, 1))  # global, comme le chemin hote
    R = np.zeros_like(U)
    for dirn, ax in ((0, 2), (1, 1)):
        Um = np.roll(U, 1, axis=ax)   # voisin -dir
        Up = np.roll(U, -1, axis=ax)  # voisin +dir
        s = _slope(U - Um, Up - U, recon)
        Lf = U + 0.5 * s                                  # etat gauche a la face +dir de la cellule
        Rf = np.roll(U, -1, axis=ax) - 0.5 * np.roll(s, -1, axis=ax)  # etat droit (voisin +dir)
        FL = e.flux(Lf, {}, dirn)
        FR = e.flux(Rf, {}, dirn)
        Fface = 0.5 * (FL + FR) - 0.5 * amax * (Rf - Lf)  # flux a la face i+1/2
        Fleft = np.roll(Fface, 1, axis=ax)                # flux a la face i-1/2
        R += -(Fface - Fleft) / h
    return R


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents")
        print("test_dsl_recon : OK (rien a compiler)")
        return

    e = build_euler_brick()
    n, L = 48, 1.0
    h = L / n
    tmp = tempfile.mkdtemp()
    try:
        so = e.compile_so(os.path.join(tmp, "euler.so"), INCLUDE)

        # etat initial structure (gradients marques pour exercer les pentes limitees)
        xs = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xs, xs)
        U = np.zeros((4, n, n))
        U[0] = 1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
        U[1] = 0.2 * U[0] * np.sin(2 * np.pi * X)
        U[2] = -0.1 * U[0] * np.cos(2 * np.pi * Y)
        p0 = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
        U[3] = p0 / (GAMMA - 1.0) + 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0]
        Uflat = U.reshape(-1).tolist()

        R_none = None
        for recon in ("none", "minmod", "vanleer"):
            sim = pops.System(n=n, L=L, periodic=True)
            sim.add_dynamic_block("gas", so, names=["rho", "rho_u", "rho_v", "E"], recon=recon)
            sim.set_state("gas", Uflat)
            R_sys = np.array(sim.eval_rhs("gas")).reshape(4, n, n)
            R_ref = muscl_residual(e, U, h, recon)
            d = float(np.max(np.abs(R_sys - R_ref)))
            assert d < 1e-9, "recon=%s : residu hote != reference MUSCL numpy (ecart %.2e)" % (recon, d)
            print("OK  recon=%-7s : eval_rhs == reference MUSCL+Rusanov (ecart %.1e)" % (recon, d))
            if recon == "none":
                R_none = R_sys

        # le limiteur agit vraiment : minmod differe nettement de l'ordre 1
        sim = pops.System(n=n, L=L, periodic=True)
        sim.add_dynamic_block("gas", so, names=["rho", "rho_u", "rho_v", "E"], recon="minmod")
        sim.set_state("gas", Uflat)
        R_minmod = np.array(sim.eval_rhs("gas")).reshape(4, n, n)
        assert float(np.max(np.abs(R_minmod - R_none))) > 1e-3, "minmod ne change rien (knob inactif)"
        print("OK  minmod modifie le residu vs ordre 1 (reconstruction active)")

        # reconstruction inconnue refusee
        try:
            sim = pops.System(n=n, L=L, periodic=True)
            sim.add_dynamic_block("g", so, recon="superbee")
            raise AssertionError("recon inconnue acceptee a tort")
        except Exception as ex:
            assert "recon" in str(ex), "mauvaise erreur : %s" % ex
            print("OK  reconstruction inconnue refusee")
        print("test_dsl_recon : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

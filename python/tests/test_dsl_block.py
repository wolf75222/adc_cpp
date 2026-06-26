"""Bloc DYNAMIQUE de bout en bout : formules Python -> .so JIT -> sim.add_dynamic_block -> le modele
charge a l'execution est pilote DANS le System (eval_rhs / step / get_state). On verifie que le residu
calcule par le bloc dynamique (host Rusanov via IModel) == celui d'pops.PythonFlux (meme schema), et que
le bloc tourne dans le System en conservant la masse. C'est l'aboutissement de l'item (a) : le DSL est
bout-en-bout depuis Python, a travers le runtime pops.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from test_dsl_brick import build_euler_brick, GAMMA

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> bloc dynamique saute")
        print("test_dsl_block : OK (rien a compiler)")
        return

    e = build_euler_brick()                       # Euler en formules (flux/eig/prim_state/cons_from)
    n, L = 48, 1.0
    h = L / n
    tmp = tempfile.mkdtemp()
    try:
        so = e.compile_so(os.path.join(tmp, "euler_model.so"), INCLUDE)   # JIT -> .so chargeable

        sim = pops.System(n=n, L=L, periodic=True)
        sim.add_dynamic_block("euler", so, names=["rho", "rho_u", "rho_v", "E"])
        print("OK  add_dynamic_block : modele charge a l'execution (vars=%s)"
              % sim.variable_names("euler"))

        # condition initiale : bulle de pression (U[c, j, i], aplatie composante-majeur c*n*n + j*n + i)
        xs = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(xs, xs)                 # indexing 'xy' : X[j,i]=x, Y[j,i]=y
        p0 = 1.0 + 0.4 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
        U = np.zeros((4, n, n))
        U[0] = 1.0
        U[3] = p0 / (GAMMA - 1.0)
        sim.set_state("euler", U.reshape(-1).tolist())

        # (1) le residu du bloc dynamique == celui d'pops.PythonFlux (meme schema Rusanov global)
        R_sys = np.array(sim.eval_rhs("euler")).reshape(4, n, n)
        pf = pops.PythonFlux(lambda u, d: e.flux(u, {}, d),
                            lambda u: max(e.max_wave_speed(u, {}, 0), e.max_wave_speed(u, {}, 1)))
        R_py = pf.residual(U, h)
        dres = float(np.max(np.abs(R_sys - R_py)))
        assert dres < 1e-9, "residu bloc dynamique != PythonFlux (ecart %.2e)" % dres
        print("OK  eval_rhs(bloc dynamique) == pops.PythonFlux (ecart max %.1e)" % dres)

        # (2) le bloc tourne DANS le System : masse conservee, etat physique, dynamique non triviale
        mass0 = float(np.array(sim.get_state("euler")).reshape(4, n, n)[0].sum())
        for _ in range(25):
            sim.step_cfl(0.4)
        U1 = np.array(sim.get_state("euler")).reshape(4, n, n)
        drel = abs(float(U1[0].sum()) - mass0) / mass0
        assert np.isfinite(U1).all() and U1[0].min() > 0, "etat non physique"
        assert drel < 1e-9, "masse non conservee (drel=%.2e)" % drel
        assert (U1[0].max() - U1[0].min()) > 1e-3, "dynamique triviale"
        print("OK  bloc dynamique avance dans le System (25 pas, masse drel=%.1e)" % drel)
        print("test_dsl_block : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""System.set_gauss_policy : politique de la loi de Gauss (chantier R0, reproduction Hoffart).

 - "restart" (DEFAUT) : solve_fields re-resout -Delta phi = f a chaque pas (historique).
 - "evolve"  : apres le premier solve (phi^0), solve_fields NE re-resout plus le Poisson ; il derive
   l'aux du phi COURANT que l'etage source condense (Schur) fait evoluer in-place -> evolution
   -Delta phi sans restart du papier.

On verifie :
 (a) API (sans compilateur) : set_gauss_policy accepte "restart"/"evolve", rejette l'inconnu.
 (b) NO-DEFAULT-CHANGE (avec compilateur) : un run SANS set_gauss_policy == un run "restart" explicite,
     a la precision machine (la branche restart est le chemin historique, inchange).
 (c) ACTIF : un run "evolve" DIFFERE de "restart" sur un fluide magnetise + CondensedSchur (le saut du
     re-solve change la trajectoire de phi), et reste FINI.

Reutilise le fluide isotherme magnetise + l'etage Schur de test_strang_split. S'auto-saute (exit 0)
sans compilateur C++ pour (b)/(c) ; (a) tourne toujours.
"""
import os
import shutil
import sys

import numpy as np

import pops
from test_strang_split import INCLUDE, build_sim, isothermal_magnetized, strang

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# --- (a) API (sans compilateur) --------------------------------------------------------------------
print("== (a) set_gauss_policy : restart/evolve acceptes, inconnu rejete ==")
s = pops.System(n=16)
s.set_gauss_policy("restart")
s.set_gauss_policy("evolve")
chk(True, "(a) set_gauss_policy('restart'/'evolve') acceptes")
chk(raises(lambda: s.set_gauss_policy("rejoindre")), "(a) politique inconnue rejetee")

# --- (b)/(c) comportement (necessite un compilateur + en-tetes pops) --------------------------------
cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip  (b)/(c) : compilateur ou en-tetes pops absents")
    print("test_gauss_policy : OK (API verte)" if fails == 0 else f"{fails} ECHEC(S)")
    sys.exit(0 if fails == 0 else 1)

compiled = isothermal_magnetized().compile(backend="aot", include=INCLUDE)
DT, NSTEPS = 2.0e-3, 25


def run(policy):
    sim = build_sim(compiled, strang(theta=0.5, alpha=3.0))
    if policy is not None:
        sim.set_gauss_policy(policy)  # avant le 1er step : le 1er solve_fields resout phi^0 (verrou)
    for _ in range(NSTEPS):
        sim.step(DT)
    return np.array(sim.density("ions"))


d_default = run(None)        # defaut = restart implicite
d_restart = run("restart")   # restart explicite
d_evolve = run("evolve")     # evolution sans restart

# (b) NO-DEFAULT-CHANGE : defaut == restart explicite, bit-identique.
chk(np.array_equal(d_default, d_restart),
    "(b) defaut == set_gauss_policy('restart') (BIT-IDENTIQUE, chemin historique)")

# (c) evolve ACTIF : differe de restart, et reste fini/physique.
chk(np.isfinite(d_evolve).all() and d_evolve.min() > 0,
    "(c) run 'evolve' fini, densite positive")
diff = float(np.max(np.abs(d_evolve - d_restart)))
chk(diff > 0.0, "(c) 'evolve' DIFFERE de 'restart' (saut du re-solve actif, ecart %.2e)" % diff)

print("test_gauss_policy : tout est vert" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)

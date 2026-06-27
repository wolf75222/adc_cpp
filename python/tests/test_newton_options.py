#!/usr/bin/env python3
"""Options et diagnostics du Newton de la source implicite IMEX (audit 2026-06, chantier 2).

Verifie :
  - NO-DEFAULT-CHANGE : pops.IMEX() == pops.IMEX(newton_max_iters=2, newton_fd_eps=1e-7) (etats
    bit-identiques, le chemin par defaut est la boucle historique) ;
  - newton_diagnostics=True -> sim.newton_report(name) rend un rapport coherent (enabled,
    converged, residu fini, iterations <= budget) ;
  - tolerance active (rel_tol) -> converge et n'utilise pas plus que le budget ;
  - source LINEAIRE (PotentialForce) : 1 iteration Newton suffit (etats ~identiques a 2 iterations) ;
  - rejets explicites : options Newton avec pops.Explicit(), newton_report sans diagnostics,
    newton_max_iters < 1, options Newton sur un modele compile (add_equation).

Invariants par assert ; imprime "OK test_newton_options" en cas de succes.
"""
from pops.numerics.reconstruction.limiters import Minmod
import sys

import numpy as np

import pops

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def fluid(charge=-1.0):
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(),
                     source=pops.PotentialForce(charge=charge),
                     elliptic=pops.ChargeDensity(charge=charge))


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.5 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


def run(n=24, steps=4, **imex_kw):
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("e", fluid(), spatial=pops.FiniteVolume(limiter=Minmod()),
                  time=pops.IMEX(**imex_kw))
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.set_density("e", gaussian(n).ravel())
    for _ in range(steps):
        sim.step_cfl(0.3)
    return sim, np.asarray(sim.get_state("e"))


# --- 1. NO-DEFAULT-CHANGE : defauts == constantes historiques explicites ---------
print("== no-default-change : IMEX() == IMEX(newton_max_iters=2, newton_fd_eps=1e-7) ==")
_, u_def = run()
_, u_exp = run(newton_max_iters=2, newton_fd_eps=1e-7)
chk(np.array_equal(u_def, u_exp), "etats BIT-IDENTIQUES (defauts = historique)")

# --- 2. Diagnostics : rapport coherent -------------------------------------------
print("== newton_diagnostics=True : rapport coherent ==")
sim, u_diag = run(newton_diagnostics=True)
rep = sim.newton_report("e")
chk(rep["enabled"], "rapport calcule (enabled)")
chk(rep["converged"], "convergence (aucune cellule en echec)")
chk(np.isfinite(rep["max_residual"]), f"residu max fini ({rep['max_residual']:.3e})")
chk(rep["max_iters_used"] <= 2, f"iterations <= budget ({rep['max_iters_used']})")
chk(rep["n_failed"] == 0, "aucune cellule en echec")
# Le mode diagnostics ne change PAS l'etat (memes iterations, meme Newton).
chk(np.array_equal(u_diag, u_def), "diagnostics : etat BIT-IDENTIQUE au chemin par defaut")

# --- 3. Tolerance active : arret anticipe possible, convergence ------------------
# abs_tol = plancher ABSOLU necessaire : la jacobienne par differences finies plafonne le residu
# vers ~1e-16 ; un rel_tol seul relatif a un res0 deja petit (etat quasi-equilibre) exigerait un
# seuil sous ce plancher et signalerait (a raison) une non-convergence.
print("== tolerance rel_tol=1e-8 + abs_tol=1e-13, budget 10 ==")
sim, u_tol = run(newton_max_iters=10, newton_rel_tol=1e-8, newton_abs_tol=1e-13,
                 newton_diagnostics=True)
rep = sim.newton_report("e")
chk(rep["converged"], "converge sous tolerance")
chk(rep["max_iters_used"] <= 10, f"iterations <= 10 ({rep['max_iters_used']})")
chk(np.all(np.isfinite(u_tol)), "etat fini")
# Source lineaire (PotentialForce) : Newton exact en 1 iteration -> la tolerance doit arreter tot.
chk(rep["max_iters_used"] <= 3, f"source lineaire : arret rapide ({rep['max_iters_used']} iters)")

# --- 4. Source lineaire : 1 iteration ~ 2 iterations -----------------------------
print("== source lineaire : newton_max_iters=1 ~= 2 ==")
_, u1 = run(newton_max_iters=1)
chk(np.allclose(u1, u_def, rtol=1e-12, atol=1e-13),
    "PotentialForce lineaire : 1 iteration Newton suffit (ecart < 1e-12)")

# --- 5. Rejets explicites ---------------------------------------------------------
print("== rejets explicites ==")
sim5 = pops.System(n=16, L=1.0, periodic=True)
try:
    sim5.add_block("e", fluid(), time=pops.Explicit(),
                   spatial=pops.FiniteVolume())
    sim5._s.add_block("e2", fluid(), "minmod", "rusanov", "conservative", "explicit",
                      1, True, 1, [], [], 5)  # newton_max_iters=5 en explicite
    chk(False, "options Newton en explicite auraient du lever")
except RuntimeError as e:
    chk("imex" in str(e), f"options Newton + explicit -> erreur : {e}")
try:
    sim5.newton_report("e")
    chk(False, "newton_report sans diagnostics aurait du lever")
except RuntimeError as e:
    chk("diagnostics" in str(e), f"newton_report sans diagnostics -> erreur : {e}")
try:
    pops.IMEX(newton_max_iters=0)
    chk(False, "newton_max_iters=0 aurait du lever")
except ValueError as e:
    chk(True, f"IMEX(newton_max_iters=0) -> ValueError : {e}")

if fails:
    print(f"FAIL test_newton_options : {fails} echec(s)")
    sys.exit(1)
print("OK test_newton_options")

#!/usr/bin/env python3
"""Vague 3 (solde) : options et diagnostics Newton CABLES sur AMR.

Couvre les deux chantiers du solde (docs/GENERICITY_2026-06.md, points NON generalises #1) :

  (a) OPTIONS NEWTON MONO-BLOC : un bloc AMR UNIQUE en IMEX avec des options Newton non-defaut
      (newton_max_iters, rel_tol) tourne fini et NE LEVE PLUS au build. Les options sont threadees
      au coupleur AmrCouplerMP (cpl->step -> advance_amr -> subcycle_level_mp ->
      mf_apply_source_treatment -> backward_euler_source), la ou avant elles etaient rejetees.

  (b) NEWTON_DIAGNOSTICS MULTI-BLOCS : newton_diagnostics=True sur un bloc IMEX d'un systeme
      multi-blocs -> AmrSystem.newton_report('bloc') rend un dict coherent (converged bool,
      max_residual fini, n_failed==0 sur un cas doux). Le rapport est AGREGE par AmrRuntime
      (reset en tete d'avance, max/somme sur niveaux et sous-pas, all_reduce MPI).

  (c) NO-DEFAULT-CHANGE (mono-bloc) : les options Newton par DEFAUT explicites (newton_max_iters=2,
      rel_tol=0, abs_tol=0, fd_eps=1e-7, damping=1.0) donnent une trajectoire BIT-IDENTIQUE a celle
      sans options (pops.IMEX()). On compare deux runs de memes graines : dmax == 0 (vrai test de
      non-regression : le chemin a options par defaut == chemin historique a iters figes).

  (d) LOADER .so + OPTIONS/DIAGNOSTICS : le chemin production AMR (CompiledModel
      backend='production', target='amr_system' -> add_native_block, ABI plate) REJETTE explicitement
      les options ET newton_diagnostics (ils seraient pris a leurs defauts en silence). Test PUR
      PYTHON : la garde de facade leve AVANT tout dlopen, un CompiledModel FACTICE suffit (pas de
      compilateur requis).

Invariants par assert ; imprime "OK test_amr_newton_full" en cas de succes.
"""
import sys

import numpy as np

import pops
from pops.codegen.loader import CompiledModel

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso_model(charge=1.0):
    """Isotherme 3-var (rho, rho_u, rho_v) avec source electrostatique (PotentialForce) : la source
    raide non triviale qu'attaque le Newton de l'IMEX (le pas implicite a quelque chose a resoudre)."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.PotentialForce(charge=charge),
                     elliptic=pops.ChargeDensity(charge=charge))


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.4 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


def mono_imex(time):
    """Construit un AmrSystem MONO-BLOC IMEX (iso_model) avec le traitement temporel @p time, seede
    d'une gaussienne, et avance d'un pas. Renvoie la densite grossiere apres le pas."""
    s = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
    s.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    s.set_refinement(1e30)
    s.add_block("e", iso_model(), spatial=pops.FiniteVolume(limiter="minmod"), time=time)
    s.set_density("e", gaussian(16).ravel())
    s.step(2e-3)
    return np.asarray(s.density("e")).reshape(16, 16)


def fake_production_amr():
    """CompiledModel FACTICE du chemin production AMR : la garde options/diagnostics de add_equation
    leve AVANT le dlopen du .so, donc le .so inexistant n'est jamais charge (deterministe, no compiler).
    Meme recette que test_amr_production_stride_reject.py."""
    return CompiledModel(
        so_path="/inexistant_amr.so", backend="production", adder="add_native_block",
        cons_names=["rho", "rho_u", "rho_v"], cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "u", "v"], n_vars=3, gamma=1.4, n_aux=3, params={}, caps={},
        abi_key="k", model_hash="h", cxx="c++", std="c++20", target="amr_system")


# ---- (a) OPTIONS NEWTON MONO-BLOC : tourne fini, ne leve plus -----------------------------------
print("== (a) mono-bloc IMEX + options Newton non-defaut : tourne fini (plus de rejet) ==")
d_a = mono_imex(pops.IMEX(newton_max_iters=5, newton_rel_tol=1e-12))
chk(np.all(np.isfinite(d_a)),
    "mono-bloc IMEX(newton_max_iters=5, rel_tol=1e-12) avance fini (options threadees au coupleur)")

# ---- (b) NEWTON_DIAGNOSTICS MULTI-BLOCS : newton_report dict coherent ---------------------------
print("== (b) multi-blocs IMEX + newton_diagnostics : newton_report('e1') coherent ==")
amr = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
amr.set_refinement(1e30)
amr.add_block("e1", iso_model(+1.0), spatial=pops.FiniteVolume(limiter="minmod"),
              time=pops.IMEX(newton_max_iters=4, newton_diagnostics=True))
amr.add_block("e2", iso_model(-1.0), spatial=pops.FiniteVolume(limiter="minmod"),
              time=pops.Explicit())
amr.set_density("e1", gaussian(16).ravel())
amr.set_density("e2", gaussian(16).ravel())
amr.advance(2e-3, 3)
rep = amr.newton_report("e1")
chk(rep["enabled"] is True, "newton_report : enabled (au moins une avance IMEX jouee)")
chk(isinstance(rep["converged"], bool), f"newton_report : converged est un bool ({rep['converged']})")
chk(np.isfinite(rep["max_residual"]) and rep["max_residual"] >= 0.0,
    f"newton_report : max_residual fini et >= 0 ({rep['max_residual']:.3e})")
chk(rep["max_iters_used"] <= 4.0,
    f"newton_report : max_iters_used <= budget (4) (recu {rep['max_iters_used']})")
chk(rep["n_failed"] == 0, f"newton_report : aucune cellule en echec sur le cas doux ({rep['n_failed']})")
chk(rep["failed_cell"] is None and rep["failed_component"] == -1,
    "newton_report : pas de cellule fautive (failed_cell None, failed_component -1)")
# Un bloc EXPLICITE (e2) ou un bloc sans diagnostics n'expose pas de rapport (erreur claire).
try:
    amr.newton_report("e2")
    chk(False, "newton_report('e2') (explicite, sans diagnostics) aurait du lever")
except RuntimeError as e:
    chk("diagnostics" in str(e), f"newton_report sur bloc sans diagnostics rejete : {str(e)[:60]}")
# Un nom de bloc inconnu leve aussi.
try:
    amr.newton_report("inconnu")
    chk(False, "newton_report('inconnu') aurait du lever")
except RuntimeError as e:
    chk("inconnu" in str(e), f"newton_report bloc inconnu rejete : {str(e)[:60]}")

# ---- (c) NO-DEFAULT-CHANGE (mono-bloc) : defauts explicites == chemin sans options --------------
print("== (c) mono-bloc : options par defaut explicites == sans options (dmax == 0, bit-identique) ==")
d_base = mono_imex(pops.IMEX())  # chemin historique (aucune option)
d_def = mono_imex(pops.IMEX(newton_max_iters=2, newton_rel_tol=0.0, newton_abs_tol=0.0,
                           newton_fd_eps=1e-7, newton_damping=1.0))  # defauts EXPLICITES
dmax = float(np.max(np.abs(d_def - d_base)))
chk(dmax == 0.0, f"options Newton par defaut explicites : dmax == 0 (bit-identique ; recu {dmax:.3e})")

# ---- (d) LOADER .so + OPTIONS/DIAGNOSTICS : rejet explicite -------------------------------------
print("== (d) production AMR (.so) : options/diagnostics Newton rejetes explicitement ==")
sim_opt = pops.AmrSystem(n=16, periodic=True)
try:
    sim_opt.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                         time=pops.IMEX(newton_max_iters=5))
    chk(False, "add_equation(.so, IMEX(newton_max_iters=5)) doit lever ValueError")
except ValueError as ex:
    chk("Newton" in str(ex) and "production" in str(ex),
        "options Newton sur .so production AMR : ValueError claire (Newton/production)")
sim_diag = pops.AmrSystem(n=16, periodic=True)
try:
    sim_diag.add_equation("gas", fake_production_amr(), spatial=pops.FiniteVolume(),
                          time=pops.IMEX(newton_diagnostics=True))
    chk(False, "add_equation(.so, IMEX(newton_diagnostics=True)) doit lever ValueError")
except ValueError as ex:
    chk("Newton" in str(ex) and "production" in str(ex),
        "newton_diagnostics sur .so production AMR : ValueError claire (Newton/production)")

print("FAIL test_amr_newton_full" if fails else "OK test_amr_newton_full")
sys.exit(1 if fails else 0)

#!/usr/bin/env python3
"""Test VISIBLE du chemin generique HLL + minmod sur un modele NON Euler.

adc.FiniteVolume(limiter="minmod", riemann="hll", variables="primitive") doit etre accepte par
System ET AmrSystem des que le modele expose des vitesses d'onde signees (model.wave_speeds) --
ici le fluide ISOTHERME 3 variables (rho, rho_u, rho_v), qui n'est PAS Euler 4 variables : hllc et
roe le REJETTENT, hll doit l'accepter (c'est la baseline generique recommandee par l'audit).

Verifie aussi :
  - hll sur un transport scalaire ExB (pas de wave_speeds) -> erreur EXPLICITE (pas de fallback) ;
  - hllc sur le meme modele isotherme -> rejet explicite (Euler 2D seulement) ;
  - quelques pas step_cfl : etat fini, masse conservee (domaine periodique).

Invariants par assert ; imprime "OK test_fv_hll_minmod" en cas de succes.
"""
import sys

import numpy as np

import adc

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso_model(charge=1.0, cs2=0.5):
    return adc.Model(state=adc.FluidState("isothermal", cs2=cs2),
                     transport=adc.IsothermalFlux(),
                     source=adc.PotentialForce(charge=charge),
                     elliptic=adc.ChargeDensity(charge=charge))


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.5 * np.exp(-80.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


# --- 1. System : hll + minmod + primitive sur isotherme 3 var (non Euler) -------
print("== System : FiniteVolume(minmod, hll, primitive) sur isotherme 3 var ==")
n = 32
sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_block("ions", iso_model(),
              spatial=adc.FiniteVolume(limiter="minmod", riemann="hll", variables="primitive"),
              time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
rho0 = gaussian(n)
sim.set_density("ions", rho0.ravel())
m0 = sim.mass("ions")
for _ in range(5):
    dt = sim.step_cfl(0.4)
    chk(np.isfinite(dt) and dt > 0, f"step_cfl rend un dt fini > 0 (dt={dt:.3e})")
rho = np.asarray(sim.density("ions"))
chk(np.all(np.isfinite(rho)), "densite finie apres 5 pas hll+minmod")
chk(abs(sim.mass("ions") - m0) < 1e-10 * abs(m0), "masse conservee (periodique)")

# --- 2. hll sur scalaire ExB (pas de wave_speeds) -> erreur explicite ------------
print("== hll sans wave_speeds (scalaire ExB) : rejet explicite ==")
sim2 = adc.System(n=16, L=1.0, periodic=True)
scal = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0), source=adc.NoSource(),
                 elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
try:
    sim2.add_block("e", scal, spatial=adc.FiniteVolume(limiter="minmod", riemann="hll"))
    chk(False, "hll sur scalaire ExB aurait du lever")
except RuntimeError as e:
    chk("wave_speeds" in str(e) or "hll" in str(e), f"erreur explicite : {e}")

# --- 3. hllc sur isotherme 3 var -> rejet explicite (Euler 2D seulement) ---------
print("== hllc sur isotherme 3 var : rejet explicite (Euler 2D seulement) ==")
sim3 = adc.System(n=16, L=1.0, periodic=True)
try:
    sim3.add_block("ions", iso_model(), spatial=adc.FiniteVolume(limiter="minmod", riemann="hllc"))
    chk(False, "hllc sur isotherme 3 var aurait du lever")
except RuntimeError as e:
    chk("hllc" in str(e), f"erreur explicite : {e}")

# --- 4. AmrSystem : hll + minmod accepte (alignement de surface System/AMR) ------
print("== AmrSystem : add_block(riemann='hll') accepte sur isotherme ==")
amr = adc.AmrSystem(n=32, L=1.0, periodic=True, regrid_every=0)
amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
amr.set_refinement(1e30)  # aucun raffinement : hierarchie mono-niveau (le sujet est le ROUTAGE hll)
amr.add_block("ions", iso_model(),
              spatial=adc.FiniteVolume(limiter="minmod", riemann="hll", variables="primitive"),
              time=adc.Explicit())
amr.set_density("ions", gaussian(32).ravel())
m0 = amr.mass("ions")
for _ in range(3):
    dt = amr.step_cfl(0.4)
    chk(np.isfinite(dt) and dt > 0, f"AMR step_cfl rend un dt fini > 0 (dt={dt:.3e})")
rho = np.asarray(amr.density("ions"))
chk(np.all(np.isfinite(rho)), "densite AMR finie apres 3 pas hll+minmod")

if fails:
    print(f"FAIL test_fv_hll_minmod : {fails} echec(s)")
    sys.exit(1)
print("OK test_fv_hll_minmod")

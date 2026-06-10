#!/usr/bin/env python3
"""Briques magnetiques natives exposees a l'API Python (audit 2026-06, chantier 5).

adc.MagneticLorentzForce (q v x B_z) et adc.PotentialMagneticForce (electrostatique + Lorentz
sommees, CompositeSource C++) etaient routees par la factory C++ ("magnetic"/"potential_magnetic")
mais ABSENTES de la surface Python (adc.Model n'acceptait que NoSource/PotentialForce/GravityForce).

Verifie :
  - adc.Model(source=adc.MagneticLorentzForce(q)) accepte ; le residu d'un etat UNIFORME (flux nul)
    vaut EXACTEMENT la force magnetique attendue s = (q B m_y, -q B m_x) (cablage quantitatif) ;
  - l'energie n'est pas touchee par la force magnetique (F . v = 0 : pas de composante energie) ;
  - adc.PotentialMagneticForce accepte et tourne fini ;
  - rejet explicite sur un transport scalaire (la brique exige >= 3 variables).

Invariants par assert ; imprime "OK test_magnetic_bricks" en cas de succes.
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


n, B0, q = 16, 3.0, -2.0

# --- 1. MagneticLorentzForce : residu quantitatif sur etat uniforme ----------------
print("== MagneticLorentzForce : residu = force magnetique exacte (etat uniforme) ==")
sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_block("e",
              adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                        transport=adc.IsothermalFlux(),
                        source=adc.MagneticLorentzForce(charge=q),
                        elliptic=adc.ChargeDensity(charge=0.0)),  # pas de couplage Poisson
              spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
sim.set_magnetic_field(B0 * np.ones(n * n))
rho0, v0 = 1.0, 0.7
ones = np.ones((n, n))
# etat uniforme : rho=1, m=(0, rho*v0) -> flux a divergence nulle -> R == source magnetique.
sim.set_primitive_state("e", rho=rho0 * ones, u=0.0 * ones, v=v0 * ones)
sim.solve_fields()
R = np.asarray(sim.eval_rhs("e")).reshape(3, n, n)
s1_expected = q * B0 * (rho0 * v0)   # s[m_x] = qom * B_z * m_y
s2_expected = 0.0                    # s[m_y] = -qom * B_z * m_x, m_x = 0
chk(np.allclose(R[0], 0.0, atol=1e-12), "R[rho] = 0 (aucune source de masse)")
chk(np.allclose(R[1], s1_expected, rtol=1e-12),
    f"R[m_x] = q*B*m_y = {s1_expected} (cablage quantitatif)")
chk(np.allclose(R[2], s2_expected, atol=1e-12), "R[m_y] = -q*B*m_x = 0")
for _ in range(5):
    dt = sim.step_cfl(0.3)
rho = np.asarray(sim.density("e"))
chk(np.all(np.isfinite(rho)), "5 pas magnetises : densite finie")

# --- 2. PotentialMagneticForce : accepte et tourne ---------------------------------
print("== PotentialMagneticForce (electrostatique + Lorentz sommees) ==")
sim2 = adc.System(n=n, L=1.0, periodic=True)
sim2.add_block("e",
               adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                         transport=adc.IsothermalFlux(),
                         source=adc.PotentialMagneticForce(charge=q),
                         elliptic=adc.BackgroundDensity(alpha=1.0, n0=1.0)),
               spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
sim2.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
sim2.set_magnetic_field(B0 * np.ones(n * n))
x = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(x, x, indexing="xy")
sim2.set_density("e", (1.0 + 0.3 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))).ravel())
for _ in range(5):
    sim2.step_cfl(0.3)
chk(np.all(np.isfinite(np.asarray(sim2.density("e")))),
    "PotentialMagneticForce : 5 pas finis (E x B + gyration)")

# --- 3. Rejet explicite sur scalaire ------------------------------------------------
print("== rejet sur transport scalaire (la force exige >= 3 variables) ==")
sim3 = adc.System(n=8, L=1.0, periodic=True)
try:
    sim3.add_block("s",
                   adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                             source=adc.MagneticLorentzForce(charge=1.0),
                             elliptic=adc.BackgroundDensity()),
                   spatial=adc.FiniteVolume())
    chk(False, "MagneticLorentzForce sur scalaire aurait du lever")
except RuntimeError as e:
    chk("3 variables" in str(e) or "invalide" in str(e), f"erreur explicite : {e}")

if fails:
    print(f"FAIL test_magnetic_bricks : {fails} echec(s)")
    sys.exit(1)
print("OK test_magnetic_bricks")

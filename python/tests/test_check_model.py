#!/usr/bin/env python3
"""check_model : verification generique d'un modele (audit 2026-06, chantier 6).

Deux niveaux :
  - dsl.Model.check_model(...) : verifie les FORMULES avant compilation (flux/source/elliptic
    finis, valeurs propres finies et reelles, coherence wave_speeds/max_wave_speed, round-trip
    to_conservative(to_primitive(U)) ~= U, positivite Density / 'p') sur des etats echantillons ;
  - System.check_model(block) : verifie le BLOC INSTALLE sur son etat courant (U fini, residu
    fini, positivite par roles, round-trip des conversions du modele, etat restaure).

Verifie le chemin vert ET les detections (round-trip casse, flux non fini, densite non positive).
Invariants par assert ; imprime "OK test_check_model" en cas de succes.
"""
import sys

import numpy as np

import adc
from adc import dsl

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso3(broken_roundtrip=False, nan_flux=False):
    m = dsl.Model("iso3_chk")
    rho, mx, my = m.conservative_vars("rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"])
    cs2 = 0.5
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    m.primitive("p", cs2 * rho)
    c = dsl.sqrt(cs2)
    fx = dsl.sqrt(rho - 10.0) if nan_flux else mx  # rho ~ [0.1, 2] -> sqrt(negatif) = NaN
    m.flux(x=[fx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    # round-trip casse : my reconstruit avec un facteur 2 (to_conservative incoherent).
    m.conservative_from([rho, rho * u, (2.0 if broken_roundtrip else 1.0) * rho * v])
    m.elliptic_rhs(1.0 * rho)
    return m


# --- 1. DSL : chemin vert ----------------------------------------------------------
print("== dsl.Model.check_model : modele sain ==")
rep = iso3().check_model()
chk(rep["ok"], "modele isotherme 3-var : ok")
chk(rep["n_samples"] == 64, "64 echantillons par defaut")

# --- 2. DSL : round-trip casse detecte ----------------------------------------------
print("== round-trip to_conservative(to_primitive) casse ==")
try:
    iso3(broken_roundtrip=True).check_model()
    chk(False, "round-trip casse aurait du lever")
except ValueError as e:
    chk("round-trip" in str(e), f"detecte : {str(e)[:90]}")

# --- 3. DSL : flux non fini detecte --------------------------------------------------
print("== flux non fini (sqrt d'un negatif) ==")
try:
    iso3(nan_flux=True).check_model()
    chk(False, "flux NaN aurait du lever")
except ValueError as e:
    chk("flux" in str(e), f"detecte : {str(e)[:90]}")
rep = iso3(nan_flux=True).check_model(raise_on_error=False)
chk(not rep["ok"] and any("flux" in f for f in rep["failures"]),
    "raise_on_error=False rend le rapport sans lever")

# --- 4. System.check_model : bloc natif sain puis densite cassee --------------------
print("== System.check_model : bloc natif ==")
n = 16
sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_block("ions",
              adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                        transport=adc.IsothermalFlux(),
                        source=adc.PotentialForce(charge=1.0),
                        elliptic=adc.ChargeDensity(charge=1.0)),
              spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
x = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(x, x, indexing="xy")
rho0 = 1.0 + 0.5 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
sim.set_density("ions", rho0.ravel())
u_before = np.asarray(sim.get_state("ions"))
rep = sim.check_model("ions")
chk(rep["ok"], "bloc natif sain : ok")
chk(np.array_equal(np.asarray(sim.get_state("ions")), u_before),
    "etat du bloc RESTAURE apres le round-trip (aucune mutation nette)")

print("== System.check_model : densite non positive detectee ==")
bad = rho0.copy()
bad[0, 0] = 0.0  # densite nulle -> role Density non strictement positif
sim.set_density("ions", bad.ravel())
rep = sim.check_model("ions", raise_on_error=False)
chk(not rep["ok"] and any("Density" in f for f in rep["failures"]),
    f"detecte : {rep['failures'][:1]}")

if fails:
    print(f"FAIL test_check_model : {fails} echec(s)")
    sys.exit(1)
print("OK test_check_model")

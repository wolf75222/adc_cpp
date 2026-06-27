#!/usr/bin/env python3
"""check_model : verification generique d'un modele (audit 2026-06, chantier 6).

Deux niveaux :
  - Model.check_model(...) : verifie les FORMULES avant compilation (flux/source/elliptic
    finis, valeurs propres finies et reelles, coherence wave_speeds/max_wave_speed, round-trip
    to_conservative(to_primitive(U)) ~= U, positivite Density / 'p') sur des etats echantillons ;
  - System.check_model(block) : verifie le BLOC INSTALLE sur son etat courant (U fini, residu
    fini, positivite par roles, round-trip des conversions du modele, etat restaure).

Verifie le chemin vert ET les detections (round-trip casse, flux non fini, densite non positive).
Invariants par assert ; imprime "OK test_check_model" en cas de succes.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops.ir.ops import sqrt
from pops.physics.facade import Model

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso3(broken_roundtrip=False, nan_flux=False):
    m = Model("iso3_chk")
    rho, mx, my = m.conservative_vars("rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"])
    cs2 = 0.5
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    m.primitive("p", cs2 * rho)
    c = sqrt(cs2)
    fx = sqrt(rho - 10.0) if nan_flux else mx  # rho ~ [0.1, 2] -> sqrt(negatif) = NaN
    m.flux(x=[fx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    # round-trip casse : my reconstruit avec un facteur 2 (to_conservative incoherent).
    m.conservative_from([rho, rho * u, (2.0 if broken_roundtrip else 1.0) * rho * v])
    m.elliptic_rhs(1.0 * rho)
    return m


# --- 1. DSL : chemin vert ----------------------------------------------------------
print("== Model.check_model : modele sain ==")
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
sim = pops.System(n=n, L=1.0, periodic=True)
sim.add_block("ions",
              pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                        transport=pops.IsothermalFlux(),
                        source=pops.PotentialForce(charge=1.0),
                        elliptic=pops.ChargeDensity(charge=1.0)),
              spatial=pops.FiniteVolume(limiter="minmod"), time=pops.Explicit())
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

print("== CompiledModel.check_runtime : .so re-verifiable SEUL (solde audit pt 9) ==")
cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
if cxx and os.path.isdir(INCLUDE):
    tmpd = tempfile.mkdtemp()
    try:
        mq = Model("ckrt")
        rho, mx, my = mq.conservative_vars("rho", "mx", "my",
                                           roles=["Density", "MomentumX", "MomentumY"])
        uq = mq.primitive("u", mx / rho)
        vq = mq.primitive("v", my / rho)
        mq.primitive("p", 0.5 * rho)
        cq = sqrt(0.5)
        mq.flux(x=[mx, mx * uq + 0.5 * rho, mx * vq], y=[my, my * uq, my * vq + 0.5 * rho])
        mq.eigenvalues(x=[uq - cq, uq, uq + cq], y=[vq - cq, vq, vq + cq])
        mq.primitive_vars(rho, uq, vq)
        mq.conservative_from([rho, rho * uq, rho * vq])
        mq.elliptic_rhs(0.0 * rho)
        cmq = mq.compile(os.path.join(tmpd, "ckrt.so"), INCLUDE, backend="production")
        repq = cmq.check_runtime(n=16)
        chk(repq["ok"] and not repq["failures"],
            f"CompiledModel seul re-verifie dans un System ephemere (ok={repq['ok']})")
        try:
            cmq.target = "amr_system"
            cmq.check_runtime()
            chk(False, "check_runtime sur target amr aurait du lever")
        except ValueError as e:
            chk("amr_system" in str(e), f"target amr rejete : {str(e)[:60]}")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
else:
    print("skip  check_runtime : compilateur ou en-tetes absents")

if fails:
    print(f"FAIL test_check_model : {fails} echec(s)")
    sys.exit(1)
print("OK test_check_model")

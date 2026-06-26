#!/usr/bin/env python3
"""IO v1 (audit 2026-06) : sim.write (vtk/npz) + sim.checkpoint / sim.restart.

La preuve centrale : un run CHECKPOINTE puis RESTARTE dans une composition rejouee reprend
BIT-IDENTIQUEMENT -- y compris la cadence STRIDE (hold-then-catch-up), qui depend de macro_step
% stride et pas seulement de t (c'est exactement ce que set_clock restaure).

Verifie :
  (1) checkpoint -> restart -> N pas == run continu, BIT-IDENTIQUE (avec un bloc stride=2 pour
      exercer la dependance a macro_step) ;
  (2) restart refuse une composition differente / une grille differente (erreur explicite) ;
  (3) write npz : champs et horloge presents ; write vtk : .vti lisible (en-tete ImageData).
Invariants par assert ; imprime "OK test_io_checkpoint" en cas de succes.
"""
import os
import sys
import tempfile

import numpy as np

import pops

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def build(n=16):
    """Deux blocs couples par le Poisson, le second a STRIDE=2 (cadence hold-then-catch-up) :
    le restart doit reprendre la fenetre stride exactement (macro_step restaure)."""
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_block("ions",
                  pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                            transport=pops.IsothermalFlux(),
                            source=pops.PotentialForce(charge=1.0),
                            elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.FiniteVolume(limiter="minmod"), time=pops.Explicit())
    sim.add_block("slow",
                  pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                            transport=pops.IsothermalFlux(),
                            source=pops.PotentialForce(charge=-1.0),
                            elliptic=pops.ChargeDensity(charge=-1.0)),
                  spatial=pops.FiniteVolume(limiter="minmod"),
                  time=pops.Explicit(stride=2))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    sim.set_density("ions", (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).ravel())
    sim.set_density("slow", (1.0 + 0.3 * np.exp(-50.0 * ((X - 0.6) ** 2 + (Y - 0.5) ** 2))).ravel())
    return sim


tmp = tempfile.mkdtemp()
dt = 2e-3

# --- (1) checkpoint -> restart bit-identique (stride=2 exerce macro_step) -----------
print("== (1) reprise bit-identique (3 pas ; checkpoint ; +4 pas vs restart +4 pas) ==")
sim = build()
for _ in range(3):  # 3 pas (IMPAIR) : le bloc stride=2 est au MILIEU de sa fenetre au checkpoint
    sim.step(dt)
ck = sim.checkpoint(os.path.join(tmp, "chk"))
chk(os.path.exists(ck), f"checkpoint ecrit ({os.path.basename(ck)})")
chk(sim.macro_step() == 3, f"macro_step = 3 ({sim.macro_step()})")
for _ in range(4):
    sim.step(dt)
ref_ions = np.asarray(sim.get_state("ions"))
ref_slow = np.asarray(sim.get_state("slow"))
ref_t = sim.time()

sim2 = build()  # composition REJOUEE (contrat v1)
sim2.restart(os.path.join(tmp, "chk"))
chk(sim2.macro_step() == 3 and abs(sim2.time() - 3 * dt) < 1e-15,
    "horloge restauree (t, macro_step)")
for _ in range(4):
    sim2.step(dt)
chk(np.array_equal(np.asarray(sim2.get_state("ions")), ref_ions),
    "bloc rapide : reprise BIT-IDENTIQUE")
chk(np.array_equal(np.asarray(sim2.get_state("slow")), ref_slow),
    "bloc stride=2 : reprise BIT-IDENTIQUE (fenetre stride reprise via macro_step)")
chk(abs(sim2.time() - ref_t) < 1e-15, "temps final identique")

# --- (2) rejets explicites -----------------------------------------------------------
print("== (2) rejets explicites ==")
bad = pops.System(n=16, L=1.0, periodic=True)
bad.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
bad.add_block("autre",
              pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                        transport=pops.IsothermalFlux(), source=pops.NoSource(),
                        elliptic=pops.ChargeDensity(charge=0.0)),
              spatial=pops.FiniteVolume(limiter="minmod"))
try:
    bad.restart(os.path.join(tmp, "chk"))
    chk(False, "composition differente aurait du lever")
except ValueError as e:
    chk("composition" in str(e), f"composition differente : {str(e)[:70]}")
small = build(n=8)
try:
    small.restart(os.path.join(tmp, "chk"))
    chk(False, "grille differente aurait du lever")
except ValueError as e:
    chk("checkpoint grid" in str(e), f"grille differente : {str(e)[:70]}")

# --- (3) write npz / vtk ---------------------------------------------------------------
print("== (3) write npz / vtk ==")
p_npz = sim.write(os.path.join(tmp, "out"), format="npz", step=7)
d = np.load(p_npz)
chk(p_npz.endswith("_000007.npz") and "state_ions" in d and "phi" in d and "macro_step" in d,
    f"npz ecrit avec etats/phi/horloge ({os.path.basename(p_npz)})")
chk(d["state_ions"].shape == (3, 16, 16), "npz : etat (ncomp, ny, nx)")
p_vti = sim.write(os.path.join(tmp, "out"), format="vtk", step=7)
head = open(p_vti).read(200)
chk("ImageData" in head and "VTKFile" in head, f"vti ecrit (en-tete ImageData) ({os.path.basename(p_vti)})")
chk("ions_rho" in open(p_vti).read(), "vti : DataArray par variable (ions_rho)")
try:
    sim.write(os.path.join(tmp, "out"), format="silo")
    chk(False, "format inconnu aurait du lever")
except ValueError as e:
    chk("format" in str(e), f"format inconnu : {str(e)[:60]}")

if fails:
    print(f"FAIL test_io_checkpoint : {fails} echec(s)")
    sys.exit(1)
print("OK test_io_checkpoint")

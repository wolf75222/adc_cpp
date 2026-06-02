#!/usr/bin/env python3
"""Test des bindings Python de la lib adc_cpp (module `adc`).

Verifie la composition bloc par bloc, le Poisson de systeme (avec paroi conductrice),
le choix libre du traitement temporel par bloc, et l'integrateur temporel ECRIT EN
PYTHON (primitives eval_rhs/get_state/set_state). Invariants par assert ; imprime
"OK test_bindings" en cas de succes.
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


def meshx(n):
    return (np.arange(n) + 0.5) / n


# --- 1. Composition heterogene : un schema par bloc, API objets lisible ----------
print("== composition heterogene (objets Spatial/Explicit/IMEX) ==")
sim = adc.System(n=48, gamma=1.4, cs2=0.5)
sim.add_block("electrons", model="electron_euler", charge=-1.0,
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))
sim.add_block("ions", model="ion_isothermal", charge=+1.0,
              spatial=adc.Spatial(minmod=True, flux="rusanov"), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
chk(sim.n_species() == 2, "deux blocs composes")
chk(sim.block_names() == ["electrons", "ions"], "noms de blocs traces")

xs = meshx(48)
sim.set_density("electrons", 1.0 + 0.02 * np.cos(2 * np.pi * xs)[None, :] * np.ones((48, 1)))
sim.set_density("ions", np.ones((48, 48)))
sim.solve_fields()
phi = sim.potential()
chk(np.abs(phi).max() > 1e-8, "Poisson de systeme actif (phi != 0)")
me0, mi0 = sim.mass("electrons"), sim.mass("ions")
sim.advance(0.001, 6)
chk(abs(sim.mass("electrons") - me0) < 1e-10, "masse electrons conservee (Euler/HLLC/IMEX)")
chk(abs(sim.mass("ions") - mi0) < 1e-10, "masse ions conservee (isotherme/Rusanov)")

# --- 2. implicite/explicite par bloc, REVERSIBLE (pas hardcode) ------------------
print("== implicite/explicite par bloc, reversible ==")
for et, it in [("imex", "explicit"), ("explicit", "imex")]:
    s = adc.System(n=32)
    s.add_block("e", "electron_euler", -1.0, time=(adc.IMEX() if et == "imex" else adc.Explicit()))
    s.add_block("i", "ion_isothermal", +1.0, time=(adc.IMEX() if it == "imex" else adc.Explicit()))
    s.set_poisson()
    s.set_density("e", 1.0 + 0.02 * np.cos(2 * np.pi * meshx(32))[None, :] * np.ones((32, 1)))
    s.set_density("i", np.ones((32, 32)))
    m0 = s.mass("e")
    s.advance(0.001, 4)
    chk(abs(s.mass("e") - m0) < 1e-10, f"electrons={et} ions={it} : masse conservee")

# --- 3. Diocotron compose en Python (PAS de DiocotronSolver), paroi conductrice ---
print("== diocotron compose generiquement + paroi conductrice circulaire ==")
n = 96
dio = adc.System(n=n, L=1.0, B0=1.0, alpha=1.0, n_i0=0.0, periodic=False)
dio.add_block("ne", model="diocotron", charge=1.0,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
dio.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)
# CI anneau ecrite en numpy (cote Python uniquement)
xx, yy = np.meshgrid(meshx(n), meshx(n), indexing="xy")
r = np.hypot(xx - 0.5, yy - 0.5)
th = np.arctan2(yy - 0.5, xx - 0.5)
ne = np.full((n, n), 1e-3)
ring = (r > 0.15) & (r < 0.20)
ne[ring] = 1.0 - 0.01 + 0.01 * np.sin(4 * th[ring])
dio.set_density("ne", ne)
dio.solve_fields()
chk(np.abs(dio.potential()).max() > 1e-6, "diocotron : Poisson a paroi actif")
m0 = dio.mass("ne")
for _ in range(20):
    dio.step_cfl(0.4)
chk(abs(dio.mass("ne") - m0) < 1e-9, "diocotron : masse conservee (transport conservatif)")
chk(dio.time() > 0, "diocotron : step_cfl avance le temps")

# --- 4. Integrateur temporel ECRIT EN PYTHON (take_step custom) ------------------
print("== integrateur temporel ecrit en Python (eval_rhs/get_state/set_state) ==")
pdio = adc.System(n=64, L=1.0, B0=1.0, alpha=1.0, n_i0=0.0, periodic=False)
pdio.add_block("ne", model="diocotron", charge=1.0, spatial=adc.Spatial(minmod=True))
pdio.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
xx, yy = np.meshgrid(meshx(64), meshx(64), indexing="xy")
r = np.hypot(xx - 0.5, yy - 0.5)
ne = np.full((64, 64), 1e-3)
ne[(r > 0.15) & (r < 0.20)] = 1.0
pdio.set_density("ne", ne)
m0 = pdio.mass("ne")
# SSPRK2 entierement Python (Poisson re-resolu par etage), le calcul reste C++ :
for _ in range(10):
    adc.integrate.ssprk2_step(pdio, 0.002)
chk(abs(pdio.mass("ne") - m0) < 1e-9, "integrateur Python : masse conservee")
chk(np.isfinite(pdio.density("ne")).all(), "integrateur Python : etat fini (stable)")

# --- 5. garde-fous -----------------------------------------------------------------
print("== garde-fous ==")


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


chk(raises(lambda: adc.System(n=16)._s.add_block("d", "diocotron", -1.0, "minmod", "hllc",
                                                 "explicit", 1)),
    "hllc refuse sur diocotron")
chk(raises(lambda: adc.System(n=16)._s.add_block("x", "inconnu", 0.0, "minmod", "rusanov",
                                                 "explicit", 1)),
    "modele inconnu refuse")

print("OK test_bindings" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)

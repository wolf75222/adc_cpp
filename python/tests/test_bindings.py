#!/usr/bin/env python3
"""Test des bindings Python de la lib adc_cpp (module `adc`), API par BRIQUES.

Verifie la composition de modeles a partir de briques generiques (aucun scenario nomme
cote C++), le Poisson de systeme (avec paroi), le choix implicite/explicite par bloc, le
multirate, l'integrateur temporel ecrit en Python, et l'AMR generique. Invariants par
assert ; imprime "OK test_bindings" en cas de succes.
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


def electron(charge=-1.0, gamma=1.4):
    return adc.Model(state=adc.FluidState("compressible", gamma=gamma),
                     transport=adc.CompressibleFlux(),
                     source=adc.PotentialForce(charge=charge),
                     elliptic=adc.ChargeDensity(charge=charge))


def ion(charge=1.0, cs2=0.5):
    return adc.Model(state=adc.FluidState("isothermal", cs2=cs2),
                     transport=adc.IsothermalFlux(),
                     source=adc.PotentialForce(charge=charge),
                     elliptic=adc.ChargeDensity(charge=charge))


def diocotron(B0=1.0, alpha=1.0, n_i0=0.0):
    return adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=B0),
                     source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=alpha, n0=n_i0))


# --- 1. Composition de briques : un schema par bloc -----------------------------
print("== composition par briques (electrons Euler/HLLC/IMEX + ions isothermes) ==")
sim = adc.System(n=48)
sim.add_block("electrons", model=electron(),
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))
sim.add_block("ions", model=ion(), spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
chk(sim.n_species() == 2, "deux blocs composes")
xs = meshx(48)
sim.set_density("electrons", 1.0 + 0.02 * np.cos(2 * np.pi * xs)[None, :] * np.ones((48, 1)))
sim.set_density("ions", np.ones((48, 48)))
sim.solve_fields()
chk(np.abs(sim.potential()).max() > 1e-8, "Poisson de systeme actif (phi != 0)")
me0, mi0 = sim.mass("electrons"), sim.mass("ions")
sim.advance(0.001, 6)
chk(abs(sim.mass("electrons") - me0) < 1e-10, "masse electrons conservee (Euler/HLLC/IMEX)")
chk(abs(sim.mass("ions") - mi0) < 1e-10, "masse ions conservee (isotherme/Rusanov)")
mea, mia = sim.mass("electrons"), sim.mass("ions")
sim.step_adaptive(0.4)
chk(abs(sim.mass("electrons") - mea) < 1e-9 and abs(sim.mass("ions") - mia) < 1e-9,
    "step_adaptive (multirate) : masses conservees par bloc")

# --- 2. implicite/explicite par bloc, REVERSIBLE --------------------------------
print("== implicite/explicite par bloc, reversible ==")
for et, it in [("imex", "explicit"), ("explicit", "imex")]:
    s = adc.System(n=32)
    s.add_block("e", electron(), time=(adc.IMEX() if et == "imex" else adc.Explicit()))
    s.add_block("i", ion(), time=(adc.IMEX() if it == "imex" else adc.Explicit()))
    s.set_poisson()
    s.set_density("e", 1.0 + 0.02 * np.cos(2 * np.pi * meshx(32))[None, :] * np.ones((32, 1)))
    s.set_density("i", np.ones((32, 32)))
    m0 = s.mass("e")
    s.advance(0.001, 4)
    chk(abs(s.mass("e") - m0) < 1e-10, f"electrons={et} ions={it} : masse conservee")

# --- 3. Diocotron compose par briques + paroi conductrice -----------------------
print("== diocotron compose par briques (ExB + BackgroundDensity) + paroi ==")
n = 96
dio = adc.System(n=n, L=1.0, periodic=False)
dio.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=0.0), spatial=adc.Spatial(minmod=True))
dio.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
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
chk(abs(dio.mass("ne") - m0) < 1e-9, "diocotron : masse conservee")

# --- 4. Integrateur temporel ECRIT EN PYTHON ------------------------------------
print("== integrateur temporel ecrit en Python (primitives eval_rhs/get_state/set_state) ==")
pd = adc.System(n=64, L=1.0, periodic=False)
pd.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=0.0), spatial=adc.Spatial(minmod=True))
pd.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
xx, yy = np.meshgrid(meshx(64), meshx(64), indexing="xy")
r = np.hypot(xx - 0.5, yy - 0.5)
ne = np.full((64, 64), 1e-3)
ne[(r > 0.15) & (r < 0.20)] = 1.0
pd.set_density("ne", ne)
m0 = pd.mass("ne")
for _ in range(10):
    adc.integrate.ssprk2_step(pd, 0.002)
chk(abs(pd.mass("ne") - m0) < 1e-9, "integrateur Python : masse conservee")
chk(np.isfinite(pd.density("ne")).all(), "integrateur Python : etat fini")

# --- 4b. AmrSystem : diocotron generique sur AMR --------------------------------
print("== AmrSystem (diocotron sur briques, AMR) ==")
nb = 64
xs = meshx(nb)
xx, yy = np.meshgrid(xs, xs, indexing="xy")
y0 = 0.5 + 0.02 * np.cos(2 * np.pi * 4 * xx)
band = 1.0 + np.exp(-((yy - y0) ** 2) / 0.05 ** 2)
nbar = float(band.mean())
amr = adc.AmrSystem(n=nb, regrid_every=10, periodic=True)
amr.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=nbar), spatial=adc.Spatial(none=True))
amr.set_refinement(threshold=nbar + 0.15)
amr.set_poisson()
amr.set_density("ne", band)
am0 = amr.mass()
for _ in range(20):
    amr.step_cfl(0.4)
chk(amr.n_patches() >= 1, "AmrSystem : raffinement actif")
chk(abs(amr.mass() - am0) / abs(am0) < 1e-9, "AmrSystem : masse conservee (reflux)")

# --- 5. garde-fous --------------------------------------------------------------
print("== garde-fous ==")


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# HLLC exige un transport compressible (4 var) : refuse sur un scalaire (ExB).
chk(raises(lambda: adc.System(n=16).add_block("d", diocotron(), spatial=adc.Spatial(flux="hllc"))),
    "hllc refuse sur transport scalaire")
# Source fluide (PotentialForce) sur un transport scalaire (ExB) : invalide.
bad = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                source=adc.PotentialForce(charge=1.0), elliptic=adc.ChargeDensity(charge=1.0))
chk(raises(lambda: adc.System(n=16).add_block("x", bad)),
    "source fluide refusee sur transport scalaire")
# Etat/transport incoherents rejetes cote Python.
chk(raises(lambda: adc.Model(state=adc.Scalar(), transport=adc.CompressibleFlux(),
                             source=adc.NoSource(), elliptic=adc.ChargeDensity())),
    "etat/transport incoherents refuses")

print("OK test_bindings" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)

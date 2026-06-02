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

# --- 4c. Espece gelee (background fixe) : non avancee, mais vue par Poisson ------
print("== espece gelee (evolve=False) : fond fixe vu par Poisson ==")
fz = adc.System(n=32, L=1.0, periodic=True)
fz.add_block("electrons", model=electron(), spatial=adc.Spatial(minmod=True))
fz.add_background("ions", model=ion(charge=1.0), density=np.ones((32, 32)))
fz.set_poisson()
fz.set_density("electrons", 1.0 + 0.05 * np.cos(2 * np.pi * meshx(32))[None, :] * np.ones((32, 32)))
ni0 = np.array(fz.density("ions"))
me0 = fz.mass("electrons")
fz.solve_fields()
chk(np.abs(fz.potential()).max() > 1e-8, "espece gelee : le fond contribue a Poisson")
for _ in range(5):
    fz.step_cfl(0.4)
chk(np.allclose(np.array(fz.density("ions")), ni0), "espece gelee : fond inchange (non avance)")
chk(abs(fz.mass("electrons") - me0) < 1e-9, "espece gelee : electrons avances, masse conservee")

# --- 4d. Source couplee inter-especes : ionisation n_g -> n_i (+ n_e) ------------
print("== source couplee : ionisation (operator-split, masse transferee) ==")


def inert():  # scalaire SANS transport (charge 0 -> phi 0 -> derive nulle) : isole le couplage
    return adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                     source=adc.NoSource(), elliptic=adc.ChargeDensity(charge=0.0))


iz = adc.System(n=24, L=1.0, periodic=True)
iz.add_block("ne", model=inert(), spatial=adc.Spatial(none=True))
iz.add_block("ni", model=inert(), spatial=adc.Spatial(none=True))
iz.add_block("ng", model=inert(), spatial=adc.Spatial(none=True))
iz.set_poisson()
iz.set_density("ne", 0.1 * np.ones((24, 24)))
iz.set_density("ni", np.zeros((24, 24)))
iz.set_density("ng", np.ones((24, 24)))
iz.add_ionization(electron="ne", ion="ni", neutral="ng", rate=0.5)
ne0, ni0i, ng0 = iz.mass("ne"), iz.mass("ni"), iz.mass("ng")
iz.advance(0.05, 10)  # pas FIXE (transport nul) : on teste uniquement la source couplee
ne1, ni1, ng1 = iz.mass("ne"), iz.mass("ni"), iz.mass("ng")
chk(ng1 < ng0 - 1e-6 and ni1 > ni0i + 1e-6, "ionisation : neutres -> ions (n_g diminue, n_i augmente)")
chk(abs((ni1 + ng1) - (ni0i + ng0)) < 1e-9, "ionisation : masse n_i + n_g conservee")
chk(ne1 > ne0, "ionisation : electrons crees (nombre)")

# --- 4e. Source couplee : friction inter-especes (qte de mvt conservee) ---------
print("== source couplee : collision / friction (qte de mvt transferee) ==")


def iso_inert():  # isotherme sans couplage de champ (charge 0) : on isole la friction
    return adc.Model(state=adc.FluidState("isothermal", cs2=0.5), transport=adc.IsothermalFlux(),
                     source=adc.NoSource(), elliptic=adc.ChargeDensity(charge=0.0))


co = adc.System(n=24, L=1.0, periodic=True)
co.add_block("a", model=iso_inert(), spatial=adc.Spatial(minmod=True))
co.add_block("b", model=iso_inert(), spatial=adc.Spatial(minmod=True))
co.set_poisson()
Ua = np.zeros((3, 24, 24)); Ua[0] = 1.0; Ua[1] = 0.3   # a : rho=1, u_x=0.3
Ub = np.zeros((3, 24, 24)); Ub[0] = 1.0; Ub[1] = 0.0   # b : rho=1, au repos
co.set_state("a", Ua.reshape(-1).tolist())
co.set_state("b", Ub.reshape(-1).tolist())
co.add_collision("a", "b", rate=1.0)
pa0 = float(np.array(co.get_state("a")).reshape(3, 24, 24)[1].sum())
pb0 = float(np.array(co.get_state("b")).reshape(3, 24, 24)[1].sum())
co.advance(0.01, 20)  # etat uniforme -> transport nul : on teste la friction seule
pa1 = float(np.array(co.get_state("a")).reshape(3, 24, 24)[1].sum())
pb1 = float(np.array(co.get_state("b")).reshape(3, 24, 24)[1].sum())
chk(pa1 < pa0 - 1e-6 and pb1 > pb0 + 1e-6, "collision : transfert de qte de mvt a -> b")
chk(abs((pa1 + pb1) - (pa0 + pb0)) < 1e-9, "collision : qte de mvt totale conservee")

# --- 4f. Source couplee : echange thermique (energie totale conservee) ----------
print("== source couplee : echange thermique (chaud -> froid) ==")


def euler_inert():  # Euler sans couplage de champ (charge 0) : on isole l'echange thermique
    return adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                     transport=adc.CompressibleFlux(), source=adc.NoSource(),
                     elliptic=adc.ChargeDensity(charge=0.0))


te = adc.System(n=16, L=1.0, periodic=True)
te.add_block("a", model=euler_inert(), spatial=adc.Spatial(minmod=True))
te.add_block("b", model=euler_inert(), spatial=adc.Spatial(minmod=True))
te.set_poisson()
Ua = np.zeros((4, 16, 16)); Ua[0] = 1.0; Ua[3] = 2.0 / 0.4   # rho=1, u=0, p=2 -> T=2
Ub = np.zeros((4, 16, 16)); Ub[0] = 1.0; Ub[3] = 1.0 / 0.4   # rho=1, u=0, p=1 -> T=1
te.set_state("a", Ua.reshape(-1).tolist())
te.set_state("b", Ub.reshape(-1).tolist())
te.add_thermal_exchange("a", "b", rate=1.0)
A0 = np.array(te.get_state("a")).reshape(4, 16, 16)
B0 = np.array(te.get_state("b")).reshape(4, 16, 16)
Ea0, Eb0 = float(A0[3].sum()), float(B0[3].sum())
te.advance(0.01, 20)  # etat uniforme -> transport nul : on teste l'echange seul
A1 = np.array(te.get_state("a")).reshape(4, 16, 16)
B1 = np.array(te.get_state("b")).reshape(4, 16, 16)
Ea1, Eb1 = float(A1[3].sum()), float(B1[3].sum())
Ta1, Tb1 = float((0.4 * A1[3] / A1[0]).mean()), float((0.4 * B1[3] / B1[0]).mean())
chk(Ea1 < Ea0 - 1e-6 and Eb1 > Eb0 + 1e-6, "echange thermique : energie chaud -> froid")
chk(abs((Ea1 + Eb1) - (Ea0 + Eb0)) < 1e-9, "echange thermique : energie totale conservee")
chk(abs(Ta1 - Tb1) < 1.0 - 1e-3, "echange thermique : temperatures relaxent")

# --- 4g. EPM : Poisson comme instance composable d'add_elliptic_model -----------
print("== EPM : Poisson via add_elliptic_model (set_poisson = raccourci) ==")
ep = adc.System(n=48, L=1.0, periodic=False)
ep.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=0.0), spatial=adc.Spatial(minmod=True))
ep.add_elliptic_model("phi", model=adc.elliptic(operator=adc.div_eps_grad(1.0),
                      rhs=adc.charge_density(), output=adc.electric_field_from_potential()),
                      solver=adc.EllipticSolver("geometric_mg"), bc="dirichlet",
                      wall="circle", wall_radius=0.40)
xx, yy = np.meshgrid(meshx(48), meshx(48), indexing="xy")
r = np.hypot(xx - 0.5, yy - 0.5)
ne_ring = np.full((48, 48), 1e-3)
ne_ring[(r > 0.15) & (r < 0.20)] = 1.0
ep.set_density("ne", ne_ring)
ep.solve_fields()
chk(np.abs(ep.potential()).max() > 1e-6, "EPM : add_elliptic_model (Poisson) actif")
try:
    adc.System(n=16).add_elliptic_model("d", adc.elliptic(operator=adc.div_eps_grad(2.0)))
    chk(False, "EPM : eps != 1 refuse (raffinement solveur)")
except NotImplementedError:
    chk(True, "EPM : eps != 1 refuse (raffinement solveur)")

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

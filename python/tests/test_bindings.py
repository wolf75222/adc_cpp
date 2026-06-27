#!/usr/bin/env python3
"""Test des bindings Python de la lib adc_cpp (module `pops`), API par BRIQUES.

Verifie la composition de modeles a partir de briques generiques (aucun scenario nomme
cote C++), le Poisson de systeme (avec paroi), le choix implicite/explicite par bloc, le
multirate, l'integrateur temporel ecrit en Python, et l'AMR generique. Invariants par
assert ; imprime "OK test_bindings" en cas de succes.
"""
from pops.numerics.riemann import HLLC, Roe
from pops.numerics.reconstruction.limiters import Minmod, VanLeer
import sys

import numpy as np

import pops

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def meshx(n):
    return (np.arange(n) + 0.5) / n


def electron(charge=-1.0, gamma=1.4):
    return pops.Model(state=pops.FluidState("compressible", gamma=gamma),
                     transport=pops.CompressibleFlux(),
                     source=pops.PotentialForce(charge=charge),
                     elliptic=pops.ChargeDensity(charge=charge))


def ion(charge=1.0, cs2=0.5):
    return pops.Model(state=pops.FluidState("isothermal", cs2=cs2),
                     transport=pops.IsothermalFlux(),
                     source=pops.PotentialForce(charge=charge),
                     elliptic=pops.ChargeDensity(charge=charge))


def diocotron(B0=1.0, alpha=1.0, n_i0=0.0):
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=B0),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=alpha, n0=n_i0))


# --- 1. Composition de briques : un schema par bloc -----------------------------
print("== composition par briques (electrons Euler/HLLC/IMEX + ions isothermes) ==")
sim = pops.System(n=48)
sim.add_block("electrons", model=electron(),
              spatial=pops.Spatial(vanleer=True, flux=HLLC()), time=pops.IMEX(substeps=10))
sim.add_block("ions", model=ion(), spatial=pops.Spatial(minmod=True), time=pops.Explicit())
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
    s = pops.System(n=32)
    s.add_block("e", electron(), time=(pops.IMEX() if et == "imex" else pops.Explicit()))
    s.add_block("i", ion(), time=(pops.IMEX() if it == "imex" else pops.Explicit()))
    s.set_poisson()
    s.set_density("e", 1.0 + 0.02 * np.cos(2 * np.pi * meshx(32))[None, :] * np.ones((32, 1)))
    s.set_density("i", np.ones((32, 32)))
    m0 = s.mass("e")
    s.advance(0.001, 4)
    chk(abs(s.mass("e") - m0) < 1e-10, f"electrons={et} ions={it} : masse conservee")

# --- 3. Diocotron compose par briques + paroi conductrice -----------------------
print("== diocotron compose par briques (ExB + BackgroundDensity) + paroi ==")
n = 96
dio = pops.System(n=n, L=1.0, periodic=False)
dio.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=0.0), spatial=pops.Spatial(minmod=True))
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
pd = pops.System(n=64, L=1.0, periodic=False)
pd.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=0.0), spatial=pops.Spatial(minmod=True))
pd.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
xx, yy = np.meshgrid(meshx(64), meshx(64), indexing="xy")
r = np.hypot(xx - 0.5, yy - 0.5)
ne = np.full((64, 64), 1e-3)
ne[(r > 0.15) & (r < 0.20)] = 1.0
pd.set_density("ne", ne)
m0 = pd.mass("ne")
for _ in range(10):
    pops.integrate.ssprk2_step(pd, 0.002)
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
amr = pops.AmrSystem(n=nb, regrid_every=10, periodic=True)
amr.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=nbar), spatial=pops.Spatial(none=True))
amr.set_refinement(threshold=nbar + 0.15)
amr.set_poisson()
amr.set_density("ne", band)
am0 = amr.mass()
for _ in range(20):
    amr.step_cfl(0.4)
chk(amr.n_patches() >= 1, "AmrSystem : raffinement actif")
chk(abs(amr.mass() - am0) / abs(am0) < 1e-9, "AmrSystem : masse conservee (reflux)")

# --- 4b-bis. AmrSystem : Euler en reconstruction PRIMITIVE (minmod/vanleer + hllc/roe) ----
# Garde de non-regression du stencil de ghost : un patch reconstruit en MUSCL ordre 2 a besoin
# de 2 ghosts ; le primitif (to_primitive) divise par rho et part en NaN si on lit un 2e ghost
# hors bornes (allocation a 1 ghost). On verifie ici que le schema reconstruit (le meme que
# System) tourne reellement sur la facade AMR : fini, densite positive, masse conservee, et
# proche de System sur une grille mono-niveau equivalente (l'integration AMR est sous-cyclee,
# d'ou un ecart de l'ordre du pourcent, pas zero).
print("== AmrSystem : Euler primitif sur AMR (garde du stencil de ghost) ==")
def euler_gas():
    return pops.Model(state=pops.FluidState(kind="compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.ChargeDensity(charge=1.0))
ne = 32
exs = meshx(ne); exx, eyy = np.meshgrid(exs, exs, indexing="xy")
erho = 1.0 + 0.4 * np.exp(-((exx - 0.5) ** 2 + (eyy - 0.5) ** 2) / 0.02)
for elim, eflux in ((Minmod(), HLLC()), (Minmod(), Roe()), (VanLeer(), HLLC())):
    tag = f"{elim.scheme}+{eflux.scheme}"
    eamr = pops.AmrSystem(n=ne, regrid_every=0, periodic=True)
    eamr.add_block("gas", model=euler_gas(),
                   spatial=pops.Spatial(limiter=elim, flux=eflux, primitive=True))
    eamr.set_refinement(threshold=1e9)  # patch seed coherent, sans tagger de cellule
    eamr.set_poisson(); eamr.set_density("gas", erho)
    em0 = eamr.mass()
    for _ in range(10):
        eamr.step_cfl(0.2)
    eda = np.array(eamr.density())
    chk(np.isfinite(eda).all() and eda.min() > 0,
        f"AMR {tag}+primitif : fini, densite positive")
    chk(abs(eamr.mass() - em0) / abs(em0) < 1e-6,
        f"AMR {tag}+primitif : masse conservee (reflux)")
    esys = pops.System(n=ne, periodic=True)
    esys.add_block("gas", model=euler_gas(),
                   spatial=pops.Spatial(limiter=elim, flux=eflux, primitive=True))
    esys.set_poisson(); esys.set_density("gas", erho)
    for _ in range(10):
        esys.step_cfl(0.2)
    eds = np.array(esys.density("gas"))
    erel = np.abs(eda - eds).max() / np.abs(eds).max()
    chk(erel < 0.05, f"AMR {tag}+primitif vs System : ecart relatif {erel:.1%} < 5%")

# --- 4c. Espece gelee (background fixe) : non avancee, mais vue par Poisson ------
print("== espece gelee (evolve=False) : fond fixe vu par Poisson ==")
fz = pops.System(n=32, L=1.0, periodic=True)
fz.add_block("electrons", model=electron(), spatial=pops.Spatial(minmod=True))
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
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=0.0))


iz = pops.System(n=24, L=1.0, periodic=True)
iz.add_block("ne", model=inert(), spatial=pops.Spatial(none=True))
iz.add_block("ni", model=inert(), spatial=pops.Spatial(none=True))
iz.add_block("ng", model=inert(), spatial=pops.Spatial(none=True))
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
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5), transport=pops.IsothermalFlux(),
                     source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=0.0))


co = pops.System(n=24, L=1.0, periodic=True)
co.add_block("a", model=iso_inert(), spatial=pops.Spatial(minmod=True))
co.add_block("b", model=iso_inert(), spatial=pops.Spatial(minmod=True))
co.set_poisson()
Ua = np.zeros((3, 24, 24)); Ua[0] = 1.0; Ua[1] = 0.3   # a : rho=1, u_x=0.3
Ub = np.zeros((3, 24, 24)); Ub[0] = 1.0; Ub[1] = 0.0   # b : rho=1, au repos
co.set_state("a", Ua.reshape(-1).tolist())
co.set_state("b", Ub.reshape(-1).tolist())
co.add_coupling(pops.Collision("a", "b", rate=1.0))  # forme OBJET (= add_collision)
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
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.ChargeDensity(charge=0.0))


te = pops.System(n=16, L=1.0, periodic=True)
te.add_block("a", model=euler_inert(), spatial=pops.Spatial(minmod=True))
te.add_block("b", model=euler_inert(), spatial=pops.Spatial(minmod=True))
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
ep = pops.System(n=48, L=1.0, periodic=False)
ep.add_block("ne", model=diocotron(B0=1.0, alpha=1.0, n_i0=0.0), spatial=pops.Spatial(minmod=True))
ep.add_elliptic_model("phi", model=pops.elliptic(operator=pops.div_eps_grad(1.0),
                      rhs=pops.charge_density(), output=pops.electric_field_from_potential()),
                      solver=pops.EllipticSolver("geometric_mg"), bc="dirichlet",
                      wall="circle", wall_radius=0.40)
xx, yy = np.meshgrid(meshx(48), meshx(48), indexing="xy")
r = np.hypot(xx - 0.5, yy - 0.5)
ne_ring = np.full((48, 48), 1e-3)
ne_ring[(r > 0.15) & (r < 0.20)] = 1.0
ep.set_density("ne", ne_ring)
ep.solve_fields()
chk(np.abs(ep.potential()).max() > 1e-6, "EPM : add_elliptic_model (Poisson) actif")
# eps != 1 CONSTANT est desormais supporte (div(eps grad phi) = f <=> lap phi = f/eps)
try:
    pops.System(n=16).add_elliptic_model("d", pops.elliptic(operator=pops.div_eps_grad(2.0)))
    chk(True, "EPM : eps != 1 constant accepte")
except NotImplementedError:
    chk(False, "EPM : eps != 1 constant accepte")


class _BogusOperator:  # operateur non div_eps_grad (diffusion / projection : non disponible)
    pass


try:
    pops.System(n=16).add_elliptic_model("d", pops.elliptic(operator=_BogusOperator()))
    chk(False, "EPM : operateur non div_eps_grad refuse")
except NotImplementedError:
    chk(True, "EPM : operateur non div_eps_grad refuse")

# --- 4h. Descripteur de variables (introspection : noms cons/prim par bloc) -----
print("== descripteur Variables : noms des variables par bloc ==")
vn = pops.System(n=16)
vn.add_block("e", model=electron())
vn.add_block("d", model=diocotron())
chk(list(vn.variable_names("e", "conservative")) == ["rho", "rho_u", "rho_v", "E"],
    "noms conservatifs (Euler)")
chk(list(vn.variable_names("e", "primitive")) == ["rho", "u", "v", "p"], "noms primitifs (Euler)")
chk(list(vn.variable_names("d")) == ["n"], "noms scalaire (diocotron)")
# Roles PHYSIQUES (ce que resolvent les couplages : index_of(role) au lieu d'un indice litteral).
chk(list(vn.variable_roles("e", "conservative")) == ["density", "momentum_x", "momentum_y", "energy"],
    "roles conservatifs (Euler)")
chk(list(vn.variable_roles("e", "primitive")) == ["density", "velocity_x", "velocity_y", "pressure"],
    "roles primitifs (Euler)")
chk(list(vn.variable_roles("d")) == ["density"], "role scalaire (diocotron)")

# --- 4i. PythonFlux : backend de prototypage (hote, numpy) ----------------------
print("== PythonFlux : flux defini en Python (prototypage hote, hors Kokkos) ==")
vx, vy = 0.7, -0.3


def adv_flux(U, d):
    return (vx if d == 0 else vy) * U  # advection scalaire F = v U


def adv_speed(U):
    return abs(vx) + abs(vy)


pf = pops.PythonFlux(adv_flux, adv_speed)
nn = 32
Upf = np.zeros((1, nn, nn))
Upf[0] = 1.0 + 0.2 * np.sin(2 * np.pi * meshx(nn))[None, :] * np.ones((nn, nn))
hh = 1.0 / nn
m0pf = float(Upf.sum())
for _ in range(50):
    Upf = Upf + pf.cfl_dt(Upf, hh, 0.4) * pf.residual(Upf, hh)  # Euler avant
chk(abs(float(Upf.sum()) - m0pf) < 1e-9, "PythonFlux : masse conservee (Rusanov conservatif)")
chk(np.isfinite(Upf).all() and float(np.abs(Upf - 1.0).max()) > 1e-3, "PythonFlux : transport actif, fini")

# --- 5. garde-fous --------------------------------------------------------------
print("== garde-fous ==")


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


# HLLC exige un transport compressible (4 var) : refuse sur un scalaire (ExB).
chk(raises(lambda: pops.System(n=16).add_block("d", diocotron(), spatial=pops.Spatial(flux=HLLC()))),
    "hllc refuse sur transport scalaire")
# Source fluide (PotentialForce) sur un transport scalaire (ExB) : invalide.
bad = pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                source=pops.PotentialForce(charge=1.0), elliptic=pops.ChargeDensity(charge=1.0))
chk(raises(lambda: pops.System(n=16).add_block("x", bad)),
    "source fluide refusee sur transport scalaire")
# Etat/transport incoherents rejetes cote Python.
chk(raises(lambda: pops.Model(state=pops.Scalar(), transport=pops.CompressibleFlux(),
                             source=pops.NoSource(), elliptic=pops.ChargeDensity())),
    "etat/transport incoherents refuses")


def err(fn):
    try:
        fn()
        return ""
    except Exception as e:
        return str(e)


# --- ADC-290 : un ModelSpec INCOMPLET echoue clairement, jamais de retombee physique silencieuse. ---
# Avant, transport defaut="compressible" / elliptic="charge" -> un ModelSpec nu valait Euler +
# Poisson-charge par accident. pops.ModelSpec() est desormais NON POSE (transport="" et elliptic="").
chk(raises(lambda: pops.System(n=16).add_block("m", pops.ModelSpec())),
    "ModelSpec incomplet (transport non pose) refuse : pas de 'compressible' silencieux")
chk("transport" in err(lambda: pops.System(n=16).add_block("m", pops.ModelSpec())).lower(),
    "message ModelSpec incomplet nomme 'transport' (erreur lisible)")
_only_transport = pops.ModelSpec()
_only_transport.transport = "exb"  # elliptic encore non pose
chk(raises(lambda: pops.System(n=16).add_block("m", _only_transport)),
    "ModelSpec sans elliptic refuse : pas de 'charge' silencieux")
# Parite AmrSystem : meme contrat a l'entree de add_block.
chk(raises(lambda: pops.AmrSystem(n=16).add_block("m", pops.ModelSpec())),
    "AmrSystem.add_block(ModelSpec incomplet) refuse")
# Un modele COMPLET (via pops.Model) reste accepte : le garde-fou ne sur-rejette pas.
chk(not raises(lambda: pops.System(n=16).add_block("ok", diocotron())),
    "modele complet (pops.Model) accepte par add_block")

# --- ADC-299 : une config invalide est REJETEE avant toute construction interne (System / AmrSystem). ---
chk(raises(lambda: pops.System(n=0)), "System(n=0) refuse (n >= 1)")
chk("n >= 1" in err(lambda: pops.System(n=0)), "message System(n=0) nomme la contrainte (lisible)")
chk(raises(lambda: pops.System(n=16, L=0.0)), "System(L=0) refuse (L > 0)")
chk(raises(lambda: pops.AmrSystem(n=0)), "AmrSystem(n=0) refuse (n >= 1)")
chk(raises(lambda: pops.AmrSystem(n=32, regrid_every=-1)), "AmrSystem(regrid_every=-1) refuse (>= 0)")

print("OK test_bindings" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)

#!/usr/bin/env python3
"""FREQUENCE PAR CELLULE des sources couplees (chantier GENERICITY 7).

CoupledSource.frequency accepte desormais, en plus d'une CONSTANTE mu [1/s], une Expr du MEME
vocabulaire que les termes (champs block().role() + param()). Une frequence-Expr est emise en bytecode
(meme machine a pile / table de registres que les termes) et evaluee PAR CELLULE cote C++ (reduction MAX
+ all_reduce_max), bornant le pas a dt <= cfl / max(mu). Les cas couverts :

  (a) CONSTANTE (regression du chemin actuel) : dt == cfl/mu EXACT, last_dt_bound 'coupled_source:<nom>'.
  (b) Expr mu(U) = k*rho avec rho gaussienne de max connu -> dt == cfl/(k*max(rho)) a 1e-12, raison OK.
  (c) la borne SUIT L'ETAT : re-seeder la densite du bloc lu a 3x resserre le pas d'un facteur 3.
  (d) ERREUR : une frequence-Expr referencant un (bloc,role) NON declare -> ValueError explicite.
  (e) AMR : la frequence-Expr est honoree sur le NIVEAU GROSSIER (dt == cfl/(k*max(rho)), raison OK).

Invariants par assert ; imprime "OK test_coupled_freq_expr" en cas de succes (exit 1 sinon).
"""
import sys

import numpy as np

import pops
from pops import dsl

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def rel(a, b):
    """Erreur relative (denominateur borne pour eviter la division par zero)."""
    return abs(a - b) / max(abs(b), 1e-300)


def iso_model(charge=1.0):
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.PotentialForce(charge=charge),
                     elliptic=pops.ChargeDensity(charge=charge))


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.4 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


CFL = 0.4
N = 16


def make_system():
    sim = pops.System(n=N, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_block("a", iso_model(+1.0), spatial=pops.FiniteVolume(limiter="minmod"))
    sim.add_block("b", iso_model(-1.0), spatial=pops.FiniteVolume(limiter="minmod"))
    return sim


# --- (a) frequence CONSTANTE : regression du chemin actuel ----------------------------------
print("== (a) frequence CONSTANTE : dt == cfl/mu, raison coupled_source:<nom> ==")
sim = make_system()
rho = gaussian(N)
sim.set_density("a", rho.ravel())
sim.set_density("b", rho.ravel())
csrc = dsl.CoupledSource("constfreq").frequency(500.0)  # mu = 500 constante
na = csrc.block("a").role("density")
kc = csrc.param("kc", 1e-9)
csrc.add_pair("a", "b", role="density", expr=kc * na)   # terme benin (la source doit avoir un terme)
sim.add_coupling(csrc.compile())
dt_a = sim.step_cfl(CFL)
chk(abs(dt_a - CFL / 500.0) < 1e-15, f"dt = cfl/mu = {CFL/500.0:.6e} (recu {dt_a:.6e})")
chk(sim.last_dt_bound() == "coupled_source:constfreq",
    f"raison = coupled_source:constfreq (recu {sim.last_dt_bound()!r})")


# --- (b) frequence PAR CELLULE mu(U) = k*rho ------------------------------------------------
print("== (b) frequence Expr mu(U)=k*rho : dt == cfl/(k*max(rho)) a 1e-12 ==")
K = 400.0
sim2 = make_system()
rho = gaussian(N)
sim2.set_density("a", rho.ravel())
sim2.set_density("b", rho.ravel())
fsrc = dsl.CoupledSource("freqexpr")
ne = fsrc.block("a").role("density")          # declare le champ (a, density)
kf = fsrc.param("kf", K)
fsrc.add_pair("a", "b", role="density", expr=1e-9 * ne)  # terme benin
fsrc.frequency(kf * ne)                        # mu(U) = K * rho_a, PAR CELLULE
compiled = fsrc.compile()
# La frequence est bien un programme bytecode (non vide) et non une constante.
chk(len(compiled.freq_prog_ops) > 0 and compiled.frequency == 0.0,
    f"compiled porte un programme de frequence (len={len(compiled.freq_prog_ops)}, const={compiled.frequency})")
# Reference numpy : memes formules que le bytecode C++.
mu_ref = compiled.reference_frequency({("a", "density"): rho})
chk(mu_ref is not None and abs(mu_ref.max() - K * rho.max()) < 1e-12,
    f"reference_frequency.max == K*max(rho) = {K*rho.max():.6e}")
sim2.add_coupling(compiled)
dt_b = sim2.step_cfl(CFL)
expected_b = CFL / (K * rho.max())
chk(rel(dt_b, expected_b) < 1e-12, f"dt = cfl/(K*max rho) = {expected_b:.9e} (recu {dt_b:.9e})")
chk(sim2.last_dt_bound() == "coupled_source:freqexpr",
    f"raison = coupled_source:freqexpr (recu {sim2.last_dt_bound()!r})")


# --- (c) la borne SUIT L'ETAT : re-seed 3x -> pas resserre d'un facteur 3 --------------------
print("== (c) la borne par cellule suit l'etat (re-seed 3x densite -> dt/3) ==")
sim2.set_density("a", (3.0 * rho).ravel())     # max(rho_a) triple a l'instant du calcul de dt
dt_c = sim2.step_cfl(CFL)
expected_c = CFL / (K * (3.0 * rho).max())
chk(rel(dt_c, expected_c) < 1e-12, f"dt = cfl/(K*max 3rho) = {expected_c:.9e} (recu {dt_c:.9e})")
chk(rel(dt_c, dt_b / 3.0) < 1e-12, f"borne suivie : dt_c == dt_b/3 ({dt_c:.9e} vs {dt_b/3.0:.9e})")


# --- (d) ERREUR : frequence-Expr referencant un (bloc,role) NON declare ----------------------
print("== (d) frequence-Expr sur un (bloc,role) non declare -> ValueError ==")
bad = dsl.CoupledSource("bad")
nb = bad.block("a").role("density")            # seuls (a,density) et (b,density) seront declares
bad.add_pair("a", "b", role="density", expr=1e-9 * nb)
# Var construite a la main : 'a::pressure' n'a jamais ete declaree via .block(...).role(...).
ghost = dsl.Var("a::pressure", "coupled_field")
bad.frequency(2.0 * ghost)
raised = False
try:
    bad.compile()
except ValueError as e:
    raised = True
    print(f"      (message : {e})")
chk(raised, "compile() leve ValueError sur la frequence-Expr a registre inconnu")


# --- (e) AMR : frequence-Expr honoree sur le niveau GROSSIER ---------------------------------
print("== (e) AMR multi-blocs : frequence-Expr mu(U)=k*rho sur le grossier ==")
KA = 300.0
amr = pops.AmrSystem(n=N, L=1.0, periodic=True, regrid_every=0)
amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
amr.set_refinement(1e30)
amr.add_block("e1", iso_model(+1.0), spatial=pops.FiniteVolume(limiter="minmod"))
amr.add_block("e2", iso_model(-1.0), spatial=pops.FiniteVolume(limiter="minmod"))
rho = gaussian(N)
amr.set_density("e1", rho.ravel())
amr.set_density("e2", rho.ravel())
asrc = dsl.CoupledSource("amrfreq")
ae = asrc.block("e1").role("density")
ka = asrc.param("ka", KA)
asrc.add_pair("e1", "e2", role="density", expr=1e-9 * ae)
asrc.frequency(ka * ae)                        # mu(U) = KA * rho_e1, PAR CELLULE (grossier)
amr.add_coupling(asrc.compile())
dt_e = amr.step_cfl(CFL)
expected_e = CFL / (KA * rho.max())
chk(rel(dt_e, expected_e) < 1e-12, f"AMR dt = cfl/(KA*max rho) = {expected_e:.9e} (recu {dt_e:.9e})")
chk(amr.last_dt_bound() == "coupled_source:amrfreq",
    f"AMR raison = coupled_source:amrfreq (recu {amr.last_dt_bound()!r})")


if fails:
    print(f"XX test_coupled_freq_expr : {fails} echec(s)")
    sys.exit(1)
print("OK test_coupled_freq_expr")

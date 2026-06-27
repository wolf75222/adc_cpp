#!/usr/bin/env python3
"""Famille IMEX-RK, schema ARS(2,2,2) sur System cartesien (Linear ADC-69).

pops.IMEXRK(scheme="ars222") cable le schema d'Ascher-Ruuth-Spiteri (1997) : transport explicite
(L = -div F) couple a la source raide implicite (backward-Euler local par cellule), ORDRE 2. C'est
une famille DISTINCTE de pops.IMEX (backward-Euler local, ordre 1, INCHANGE).

PROBLEME TEMOIN -- gyration cyclotron pure. Sur un etat UNIFORME (rho, m_x, m_y) + B_z uniforme, le
flux isotherme a divergence NULLE (L == 0) et la dynamique se reduit a l'ODE de rotation
  dm_x/dt = q*B*m_y,  dm_y/dt = -q*B*m_x      (frequence cyclotron omega = q*B),
source LINEAIRE -> le backward-Euler local est EXACT en une iteration de Newton, ce qui ISOLE
l'ORDRE TEMPOREL du schema (avec L == 0, IMEXRK se reduit au SDIRK ARS(2,2,2) ordre 2 ; IMEX au
backward-Euler ordre 1). Reference auto-convergente (Cauchy) : pas besoin de la formule exacte.

Verifie :
  (a) CONVERGENCE ORDRE 2 : erreur(dt)/erreur(dt/2) ~ 4 pour IMEXRK (ordre 2) vs ~ 2 pour IMEX
      (ordre 1) -- le test discriminant ;
  (b) STABILITE RAIDE : omega*dt = 50, etat fini et NON amplifie (pas d'explosion ; un explicite
      donnerait |1 + i*omega*dt|^k -> infini) ;
  (c) DEFAUT INCHANGE / FAMILLE DISTINCTE : pops.IMEX.kind == "imex" != pops.IMEXRK.kind ==
      "imexrk_ars222" ; sur le meme probleme les deux schemas DIVERGENT (chemins C++ distincts,
      pas un alias) ;
  (d) REJETS EXPLICITES : AMR, polaire et splitting Strang refusent IMEXRK (perimetre = System
      cartesien), + masque IMEX partiel refuse (source pleinement implicite).

Modeles NATIFS uniquement (le chemin compile .so rejette IMEXRK cote C++). Invariants par assert ;
imprime "OK test_imexrk" en cas de succes.
"""
from pops.numerics.reconstruction.limiters import Minmod
import sys

import numpy as np

import pops

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def cyclotron_model(q):
    """Fluide isotherme + force magnetique q*(v x B). elliptic charge=0 -> Poisson trivial (phi=0),
    aucune force electrique : la dynamique d'un etat uniforme est la pure gyration cyclotron."""
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.MagneticLorentzForce(charge=q),
                     elliptic=pops.ChargeDensity(charge=0.0))


def build(time_policy, q, B0, rho0=1.0, u0=1.0, v0=0.0, n=8):
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("e", cyclotron_model(q), spatial=pops.FiniteVolume(limiter=Minmod()),
                  time=time_policy)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.set_magnetic_field(B0 * np.ones(n * n))
    ones = np.ones((n, n))
    sim.set_primitive_state("e", rho=rho0 * ones, u=u0 * ones, v=v0 * ones)
    return sim


def momentum_after(time_policy, q, B0, dt, nsteps, n=8):
    """Avance nsteps de pas FIXE dt et renvoie le moment uniforme (m_x, m_y) (moyenne spatiale)."""
    sim = build(time_policy, q, B0, n=n)
    for _ in range(nsteps):
        sim.step(dt)
    U = np.asarray(sim.get_state("e"), dtype=float).reshape(3, n, n)
    return np.array([U[1].mean(), U[2].mean()])


# --- (a) CONVERGENCE ORDRE 2 (auto-convergence en temps, reference fine) ----------------
print("== (a) convergence en temps : IMEXRK ordre 2 (ratio ~4) vs IMEX ordre 1 (ratio ~2) ==")
Q, B = 1.0, 1.0  # omega = q*B = 1 (regime asymptotique sur T = 1)
T = 1.0


def err_ratio(make_policy):
    ref = momentum_after(make_policy(), Q, B, T / 800, 800)  # reference fine (par schema)
    m1 = momentum_after(make_policy(), Q, B, T / 10, 10)
    m2 = momentum_after(make_policy(), Q, B, T / 20, 20)
    e1 = float(np.linalg.norm(m1 - ref))
    e2 = float(np.linalg.norm(m2 - ref))
    return e1, e2, (e1 / e2 if e2 > 0 else float("inf"))


e1_rk, e2_rk, r_rk = err_ratio(lambda: pops.IMEXRK())
e1_be, e2_be, r_be = err_ratio(lambda: pops.IMEX())
chk(3.2 <= r_rk <= 4.8, f"IMEXRK ARS(2,2,2) : ratio d'erreur ~4 (ordre 2) -> {r_rk:.2f}")
chk(1.6 <= r_be <= 2.5, f"IMEX backward-Euler : ratio d'erreur ~2 (ordre 1) -> {r_be:.2f}")
chk(r_rk > r_be + 1.0, f"IMEXRK converge plus vite que IMEX ({r_rk:.2f} > {r_be:.2f})")
chk(e1_rk < e1_be, f"IMEXRK plus precis qu'IMEX a meme dt ({e1_rk:.2e} < {e1_be:.2e})")

# --- (b) STABILITE RAIDE : omega*dt = 50, etat fini et non amplifie ---------------------
print("== (b) stabilite raide : omega*dt = 50, etat fini, pas d'explosion ==")
Q2, B2 = 1.0, 50.0  # omega = 50
DT2 = 1.0           # omega*dt = 50
mag0 = 1.0          # |m0| = sqrt(u0^2 + v0^2) = 1 (rho0 = 1)
m_stiff = momentum_after(pops.IMEXRK(), Q2, B2, DT2, 10)
mag = float(np.linalg.norm(m_stiff))
chk(np.all(np.isfinite(m_stiff)), f"IMEXRK raide : etat fini apres 10 pas (m = {m_stiff})")
chk(mag <= mag0 + 1e-6, f"IMEXRK raide : |m| non amplifie (A-stable) -> {mag:.3e} <= {mag0}")

# --- (c) DEFAUT INCHANGE / FAMILLE DISTINCTE --------------------------------------------
print("== (c) defaut IMEX inchange + famille IMEXRK distincte ==")
chk(pops.IMEX().kind == "imex", "pops.IMEX.kind == 'imex' (backward-Euler local, defaut)")
chk(pops.IMEXRK().kind == "imexrk_ars222", "pops.IMEXRK.kind == 'imexrk_ars222' (famille distincte)")
chk(pops.IMEX().kind != pops.IMEXRK().kind, "kinds distincts -> chemins C++ distincts (pas un alias)")
chk(pops.IMEXRK().scheme == "ars222", "pops.IMEXRK.scheme == 'ars222'")
# Sur le MEME probleme, IMEX (ordre 1) et IMEXRK (ordre 2) donnent des etats DIFFERENTS : preuve
# qu'IMEXRK n'emprunte PAS le chemin d'IMEX (sinon etats identiques).
m_be = momentum_after(pops.IMEX(), Q, B, T / 10, 10)
m_rk = momentum_after(pops.IMEXRK(), Q, B, T / 10, 10)
chk(float(np.linalg.norm(m_be - m_rk)) > 1e-4,
    f"IMEX != IMEXRK sur le meme pas (ecart {float(np.linalg.norm(m_be - m_rk)):.2e})")

# --- (d) REJETS EXPLICITES (perimetre = System cartesien) -------------------------------
print("== (d) rejets explicites : AMR / polaire / Strang / masque partiel ==")

# (d1) AMR
amr = pops.AmrSystem(n=16, L=1.0, periodic=True, regrid_every=0)
try:
    amr.add_block("e", cyclotron_model(1.0), spatial=pops.FiniteVolume(limiter=Minmod()),
                  time=pops.IMEXRK())
    chk(False, "AMR + IMEXRK aurait du lever")
except (RuntimeError, ValueError, TypeError) as e:
    chk("imexrk" in str(e).lower() or "imex-rk" in str(e).lower(), f"AMR rejet explicite : {e}")

# (d2) polaire (anneau) : la source raide implicite n'y est pas cablee
simp = pops.System(mesh=pops.PolarMesh(r_min=0.2, r_max=1.0, nr=16, ntheta=16))
try:
    simp.add_block("e",
                   pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                             source=pops.NoSource(), elliptic=pops.BackgroundDensity()),
                   spatial=pops.FiniteVolume(), time=pops.IMEXRK())
    chk(False, "polaire + IMEXRK aurait du lever")
except (RuntimeError, ValueError, TypeError) as e:
    chk("imex" in str(e).lower(), f"polaire rejet explicite : {e}")

# (d3) Strang / Schur : l'etage hyperbolique doit etre un pops.Explicit (pas IMEXRK)
try:
    pops.Strang(hyperbolic=pops.IMEXRK(),
               source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, alpha=1.0))
    chk(False, "Strang(hyperbolic=IMEXRK) aurait du lever")
except TypeError as e:
    chk("Explicit" in str(e), f"Strang rejet (hyperbolique doit etre Explicit) : {e}")

# (d4) masque IMEX partiel : la source IMEXRK est pleinement implicite -> rejet a l'ajout du bloc
sim_mask = pops.System(n=8, L=1.0, periodic=True)
try:
    # pops.IMEXRK n'expose pas implicit_vars ; on force l'attribut pour exercer la garde C++.
    pol = pops.IMEXRK()
    pol.implicit_vars = ["rho_u"]
    sim_mask.add_block("e", cyclotron_model(1.0), spatial=pops.FiniteVolume(limiter=Minmod()),
                       time=pol)
    chk(False, "IMEXRK + implicit_vars aurait du lever")
except (RuntimeError, ValueError) as e:
    chk("imexrk" in str(e).lower() or "fully implicit" in str(e).lower(),
        f"masque partiel rejete : {e}")

if fails:
    print(f"FAIL test_imexrk : {fails} echec(s)")
    sys.exit(1)
print("OK test_imexrk")

#!/usr/bin/env python3
"""Briques natives ROLE-AWARE (audit GENERICITY_2026-06 §5 "Briques natives layout fluide").

Les briques de source (PotentialForce, GravityForce, MagneticLorentzForce) et d'elliptique
(ChargeDensity, BackgroundDensity, GravityCoupling) lisaient/ecrivaient des indices CODES EN DUR
(u[0]=rho, s[1]/s[2]=qdm, s[3]=energie). Elles portent desormais des MEMBRES d'indices (c_rho, c_mx,
c_my, c_E) que model_factory RESOUT a la construction (hote) via TR::conservative_vars().index_of(role)
-- une resolution AUTOMATIQUE par role, SANS aucun parametre utilisateur nouveau.

CE QUI EST COUVERT ICI (et ses LIMITES, documentees honnetement) :
  - NO-DEFAULT-CHANGE (quantitatif) : pour les transports NATIFS, les roles sont CANONIQUES
    (Euler/Isothermal : rho=0, m_x=1, m_y=2[, E=3]) -> les indices resolus == les DEFAUTS des briques
    -> resultat STRICTEMENT bit-identique. On le verrouille sur la force magnetique (B_z directement
    posable via set_magnetic_field) : sur un etat UNIFORME (flux a divergence nulle) le residu vaut
    EXACTEMENT s = (q B m_y, -q B m_x), donc c_mx==1 et c_my==2 ont bien ete resolus aux composantes
    canoniques. Idem sur Euler 4-var, ou l'ENERGIE (composante 3) reste INTOUCHEE (travail nul).
  - API : PotentialForce + ChargeDensity composent toujours et tournent fini apres l'ajout des membres ;
    deux constructions IDENTIQUES du meme modele donnent dmax == 0 (aucun etat cache introduit).

  - LIMITE (honnete) : depuis Python, les briques natives ne se composent QU'AVEC les transports
    natifs (Euler/Isothermal/ExB), tous a roles CANONIQUES. Il n'existe donc PAS de chemin public ou un
    layout NON canonique rencontre une brique native -- le re-cablage effectif des indices (c_rho!=0,
    etc.) n'est pas observable d'ici. Ce chemin est verrouille cote C++ par la detection `requires`
    (if constexpr) du binder bind_variable_roles (model_factory.hpp) et par le registre des roles
    (core/variables.hpp). Ce test verrouille donc le NO-DEFAULT-CHANGE et l'API, pas le re-cablage.

Invariants par assert ; imprime "OK test_role_aware_bricks" en cas de succes.
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
rho0, v0 = 1.0, 0.7
ones = np.ones((n, n))

# --- 1. NO-DEFAULT-CHANGE quantitatif : magnetique sur ISOTHERME 3-var (rho=0, m_x=1, m_y=2) --------
print("== 1. isotherme 3-var : residu magnetique == force exacte (roles canoniques resolus) ==")
sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_block("e",
              adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                        transport=adc.IsothermalFlux(),
                        source=adc.MagneticLorentzForce(charge=q),
                        elliptic=adc.ChargeDensity(charge=0.0)),  # pas de couplage Poisson
              spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
sim.set_magnetic_field(B0 * np.ones(n * n))
# etat uniforme rho=1, m=(0, rho*v0) -> div F = 0 -> R == source magnetique.
sim.set_primitive_state("e", rho=rho0 * ones, u=0.0 * ones, v=v0 * ones)
sim.solve_fields()
R = np.asarray(sim.eval_rhs("e")).reshape(3, n, n)
chk(np.allclose(R[0], 0.0, atol=1e-12), "R[rho] = 0 (c_rho non ecrit : aucune source de masse)")
chk(np.allclose(R[1], q * B0 * (rho0 * v0), rtol=1e-12),
    "R[m_x] = q*B*m_y EXACT (c_mx resolu a la composante 1)")
chk(np.allclose(R[2], 0.0, atol=1e-12), "R[m_y] = -q*B*m_x = 0 (c_my resolu a la composante 2)")

# --- 2. NO-DEFAULT-CHANGE quantitatif : magnetique sur EULER 4-var (energie c_E=3 intouchee) --------
print("== 2. compressible 4-var : energie (composante 3) intouchee par la force magnetique ==")
sE = adc.System(n=n, L=1.0, periodic=True)
sE.add_block("g",
             adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                       transport=adc.CompressibleFlux(),
                       source=adc.MagneticLorentzForce(charge=q),
                       elliptic=adc.ChargeDensity(charge=0.0)),
             spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
sE.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
sE.set_magnetic_field(B0 * np.ones(n * n))
sE.set_primitive_state("g", rho=rho0 * ones, u=0.0 * ones, v=v0 * ones, p=1.0 * ones)
sE.solve_fields()
RE = np.asarray(sE.eval_rhs("g")).reshape(4, n, n)
chk(np.allclose(RE[0], 0.0, atol=1e-10), "R[rho] = 0 (4-var)")
chk(np.allclose(RE[1], q * B0 * (rho0 * v0), rtol=1e-12), "R[m_x] = q*B*m_y EXACT (4-var, c_mx=1)")
chk(np.allclose(RE[2], 0.0, atol=1e-10), "R[m_y] = 0 (4-var, c_my=2)")
chk(np.allclose(RE[3], 0.0, atol=1e-10),
    "R[E] = 0 : la force magnetique ne touche PAS l'energie (c_E=3 non ecrit, travail nul)")

# --- 3. API + determinisme : PotentialForce + ChargeDensity composent et sont reproductibles --------
print("== 3. PotentialForce + ChargeDensity : compose, tourne fini, deux chemins identiques == ==")
x = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(x, x, indexing="xy")
rho_bump = (1.0 + 0.3 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))).ravel()


def run_potential():
    s = adc.System(n=n, L=1.0, periodic=True)
    s.add_block("e",
                adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                          transport=adc.IsothermalFlux(),
                          source=adc.PotentialForce(charge=-1.0),   # lit c_rho, ecrit c_mx/c_my
                          elliptic=adc.ChargeDensity(charge=-1.0)),  # second membre Poisson = q*u[c_rho]
                spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
    s.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    s.set_density("e", rho_bump.copy())
    for _ in range(8):
        s.step_cfl(0.3)
    return np.asarray(s.density("e"))


d_a = run_potential()
d_b = run_potential()
chk(np.all(np.isfinite(d_a)), "PotentialForce + ChargeDensity : densite finie sur 8 pas")
# couplage NON trivial : la force electrostatique a deforme le profil (sinon dmax==0 serait vide de sens).
chk(float(np.max(np.abs(d_a - rho_bump.reshape(n, n)))) > 1e-6,
    "le couplage electrostatique fait EVOLUER la densite (test non trivial)")
dmax = float(np.max(np.abs(d_a - d_b)))
chk(dmax == 0.0, "deux constructions IDENTIQUES -> dmax == 0 (aucun etat cache, bit-identique)")

if fails:
    print(f"FAIL test_role_aware_bricks : {fails} echec(s)")
    sys.exit(1)
print("OK test_role_aware_bricks")

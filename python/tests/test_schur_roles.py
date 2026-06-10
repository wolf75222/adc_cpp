#!/usr/bin/env python3
"""Roles / champs de l'etage Schur condense TRANSPORTES dans l'ABI (audit 2026-06, vague 2).

CondensedSchur n'est plus fige sur Density/MomentumX/MomentumY/B_z : les descripteurs (role stable
OU nom de variable du bloc) traversent set_source_stage jusqu'au stepper (constructeur a composantes
explicites). Verifie :
  (1) EQUIVALENCE : descripteurs explicites par NOM DE VARIABLE ("rho", "rho_u", "rho_v") sur le
      bloc natif isotherme == defauts canoniques, BIT-IDENTIQUE (memes composantes resolues) ;
  (2) rejet explicite d'un descripteur introuvable (ni role ni variable) ;
  (3) rejet explicite des overrides sur l'etage POLAIRE (non cable, pas d'ignore silencieux) ;
  (4) rejet explicite des overrides sur le chemin amr-schur (idem).
Invariants par assert ; imprime "OK test_schur_roles" en cas de succes.
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


def iso_model(cs2=1.0, alpha=3.0):
    return adc.Model(state=adc.FluidState(kind="isothermal", cs2=cs2),
                     transport=adc.IsothermalFlux(),
                     source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=alpha, n0=0.0))


def smooth(n, L=1.0):
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.5 * np.ones((n, n))
    u0 = 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    v0 = -0.3 * np.sin(2.0 * np.pi * X / L) * np.sin(np.pi * Y / L)
    return rho0, u0, v0


def build(n=20, B0=4.0, schur=None, steps=4):
    sim = adc.System(n=n, L=1.0, periodic=False)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
    sim.set_magnetic_field(B0 * np.ones((n, n)))
    sim.add_equation("e", model=iso_model(),
                     spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                     time=adc.Split(hyperbolic=adc.Explicit(),
                                    source=schur or adc.CondensedSchur(theta=1.0, alpha=3.0)))
    rho0, u0, v0 = smooth(n)
    sim.set_primitive_state("e", rho=rho0, u=u0, v=v0)
    for _ in range(steps):
        sim.step(2e-3)
    return np.asarray(sim.get_state("e"))


# --- (1) descripteurs par NOM DE VARIABLE == defauts canoniques (bit-identique) ------
print("== (1) equivalence : descripteurs explicites par nom == roles canoniques ==")
u_default = build()
u_named = build(schur=adc.CondensedSchur(theta=1.0, alpha=3.0,
                                         density="rho", momentum=("rho_u", "rho_v")))
chk(np.array_equal(u_default, u_named),
    "etats BIT-IDENTIQUES (resolution par nom de variable == roles canoniques)")
chk(np.all(np.isfinite(u_default)), "etat fini (etage Schur actif)")

# --- (2) descripteur introuvable -> rejet explicite ----------------------------------
print("== (2) descripteur introuvable ==")
try:
    build(schur=adc.CondensedSchur(theta=1.0, alpha=3.0, density="rho_inconnue"))
    chk(False, "descripteur inconnu aurait du lever")
except RuntimeError as e:
    chk("ni un role" in str(e) or "n'expose pas" in str(e), f"erreur explicite : {str(e)[:80]}")

# --- (3) overrides sur l'etage POLAIRE : ACCEPTES depuis la vague 3 -----------------
print("== (3) polaire : overrides cables (ctor a composantes explicites) ==")
psim = adc.System(mesh=adc.PolarMesh(r_min=0.5, r_max=1.0, nr=16, ntheta=16))
psim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
psim.set_magnetic_field(4.0 * np.ones(16 * 16))
psim.add_equation("e", model=iso_model(),
                  spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                  time=adc.Split(hyperbolic=adc.Explicit(),
                                 source=adc.CondensedSchur(theta=1.0, alpha=3.0,
                                                           density="rho",
                                                           momentum=("rho_u", "rho_v"))))
rho0p = 1.5 * np.ones(16 * 16)
psim.set_density("e", rho0p)
psim.step(1e-3)
chk(np.all(np.isfinite(np.asarray(psim.get_state("e")))),
    "polaire + descripteurs par nom : un pas Schur fini (overrides cables)")

# --- (4) overrides sur amr-schur : ACCEPTES depuis la vague 3 (krylov + roles) -------
print("== (4) amr-schur : descripteurs + krylov transportes ==")
amr = adc.AmrSystem(n=16, L=1.0, periodic=False, regrid_every=0)
amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
amr.set_refinement(1e30)
amr.set_magnetic_field(4.0 * np.ones((16, 16)))
amr.add_equation("e", model=iso_model(),
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Split(hyperbolic=adc.Explicit(),
                                source=adc.CondensedSchur(theta=1.0, alpha=3.0,
                                                          density="rho",
                                                          momentum=("rho_u", "rho_v"),
                                                          krylov_tol=1e-8,
                                                          krylov_max_iters=200)))
rho0a, u0a, v0a = smooth(16)
amr.set_conservative_state("e", np.stack([rho0a, rho0a * u0a, rho0a * v0a]))
amr.step(1e-3)
chk(np.all(np.isfinite(np.asarray(amr.density("e")))),
    "amr-schur + descripteurs par nom + krylov : un pas fini")
# magnetic_field reste fige sur le tampon B_z grossier dedie -> rejet explicite.
amr2 = adc.AmrSystem(n=16, L=1.0, periodic=False, regrid_every=0)
amr2.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
amr2.set_refinement(1e30)
amr2.set_magnetic_field(4.0 * np.ones((16, 16)))
try:
    amr2.add_equation("e", model=iso_model(),
                      spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                      time=adc.Split(hyperbolic=adc.Explicit(),
                                     source=adc.CondensedSchur(theta=1.0, alpha=3.0,
                                                               magnetic_field="T_e")))
    chk(False, "magnetic_field != B_z aurait du lever sur AMR")
except ValueError as e:
    chk("B_z" in str(e), f"magnetic_field non transporte : {str(e)[:70]}")

if fails:
    print(f"FAIL test_schur_roles : {fails} echec(s)")
    sys.exit(1)
print("OK test_schur_roles")

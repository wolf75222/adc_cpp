#!/usr/bin/env python3
"""Couverture CI : System POLAIRE -> run_source_stage -> etage Schur condense POLAIRE (Voie A etape 2c).

Pendant POLAIRE de test_schur_via_system.py (cartesien). L'etage PolarCondensedSchurSourceStepper est
teste en STANDALONE par le test C++ test_polar_condensed_schur_source_stepper (relation implicite,
stabilite vs Euler, ordre 1) ; ce test comble le chemin FACADE, sur l'anneau (r, theta) :
  pops.System(mesh=pops.PolarMesh(...)).add_equation(time=pops.Split(source=pops.CondensedSchur(...)))
    -> _s.add_block (IsothermalFluxPolar : roles Density / MomentumX (radial) / MomentumY (azimutal))
    -> _s.set_source_stage (geometrie polaire -> PolarCondensedSchurSourceStepper, dispatch C++)
  sim.step(dt)
    -> SystemStepper::step -> s.advance (transport polaire) -> run_source_stage (SCHUR POLAIRE)
       -> s.schur_polar->step(U, phi, bz, kAuxBaseComps, theta, dt)        # cible du test

C'est le diocotron-Schur polaire de PR6, au moins en smoke : un fluide isotherme magnetise RAIDE
(omega_c eleve) reste STABLE sous l'etage source implicite (theta=1), la ou un schema explicite
divergerait.

Le modele isotherme polaire (IsothermalFluxPolar) expose exactement les roles requis par
set_source_stage (Density / MomentumX / MomentumY). On utilise UNIQUEMENT des briques natives : aucune
compilation .so, CI-safe. Mono-rang (le stepper polaire l'exige).

Validations :
  (A) L'etat apres N pas est FINI (nan/inf rejectes avant toute tolerance).
  (B) L'etage source a EFFECTIVEMENT tourne : la quantite de mouvement differe du run sans Schur
      (si run_source_stage etait un no-op, les deux runs seraient identiques).
  (C) La densite reste BORNEE et positive (l'etage source ne traite que la vitesse ; rho est gelee
      dans le sous-systeme source, seul le transport la fait evoluer).
  (D) Stabilite sous source RAIDE (omega_c >> 1/dt) : ||v|| reste finie et bornee sur K pas
      (theta=1, Euler retrograde) -- le verrou de PR6.
"""

import math
import sys

import numpy as np

try:
    import pops
except ImportError as e:
    print("skip  module pops absent (PYTHONPATH ?) : %s" % e)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Construction du systeme polaire
# ---------------------------------------------------------------------------

RMIN, RMAX = 0.30, 1.00


def iso_fluid_model(cs2=1.0, alpha=1.0):
    """Fluide isotherme NATIF (briques natives, sans DSL ni compilateur C++).

    Roles conservatifs : Density / MomentumX (radial) / MomentumY (azimutal). Sur l'anneau, le
    dispatch polaire (block_builder_polar) instancie IsothermalFluxPolar (memes roles). C'est le
    modele MINIMAL accepte par set_source_stage.
    """
    return pops.Model(
        state=pops.FluidState(kind="isothermal", cs2=cs2),
        transport=pops.IsothermalFlux(),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=alpha, n0=0.0),
    )


def _annular_density(nr, nth):
    """Profil annulaire lisse, > 0, nul aux bords radiaux (compatible Dirichlet phi=0), module en
    theta (asymetrie -> Poisson non trivial -> grad phi != 0 -> source active). Layout attendu par
    set_density polaire : axe lent = theta (j), axe rapide = r (i), flat[j*nr + i]."""
    dr = (RMAX - RMIN) / nr
    dth = 2.0 * math.pi / nth
    rho = np.empty(nth * nr, dtype=np.float64)
    for j in range(nth):
        th = (j + 0.5) * dth
        for i in range(nr):
            r = RMIN + (i + 0.5) * dr
            rr = (r - RMIN) / (RMAX - RMIN)
            rho[j * nr + i] = 1.5 + 0.3 * math.cos(2.0 * th) * math.sin(math.pi * rr)
    return rho


def _initial_velocity_state(nr, nth, rho_flat):
    """Etat conservatif (3, ntheta, nr) = (rho, mom_r, mom_theta) avec une vitesse initiale lisse,
    nulle aux bords radiaux. set_state attend l'ordre composante-majeur (c lent, puis j=theta, puis
    i=r), c.-a-d. un tableau numpy (ncomp, ny=ntheta, nx=nr) aplati en C-order."""
    dr = (RMAX - RMIN) / nr
    dth = 2.0 * math.pi / nth
    rho = rho_flat.reshape(nth, nr)
    mr = np.empty((nth, nr), dtype=np.float64)
    mth = np.empty((nth, nr), dtype=np.float64)
    for j in range(nth):
        th = (j + 0.5) * dth
        for i in range(nr):
            r = RMIN + (i + 0.5) * dr
            h = math.sin(math.pi * (r - RMIN) / (RMAX - RMIN))  # 0 aux bords radiaux
            vr = 0.6 * h * math.cos(2.0 * th)
            vth = -0.4 * h * math.sin(th)
            mr[j, i] = rho[j, i] * vr
            mth[j, i] = rho[j, i] * vth
    return np.stack([rho, mr, mth], axis=0)  # (3, ntheta, nr)


def build_system(nr, nth, B0, alpha, theta, with_schur, cs2=1.0):
    """System polaire (anneau) avec un bloc isotherme NATIF + Poisson polaire + B_z constant.

    with_schur=True  : add_equation(Split) -> add_block (IsothermalFluxPolar) + set_source_stage
                       (-> PolarCondensedSchurSourceStepper, dispatch C++ polaire).
    with_schur=False : add_equation(Explicit) -> add_block, sans set_source_stage (chemin de reference).
    """
    sim = pops.System(mesh=pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=nr, ntheta=nth))
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    # B_z constant sur l'anneau. Layout (ntheta, nr) aplati C-order = flat[j*nr+i] (theta lent, r
    # rapide), coherent avec set_density / set_state polaire.
    sim.set_magnetic_field(B0 * np.ones((nth, nr)))
    if with_schur:
        time_policy = pops.Split(
            hyperbolic=pops.Explicit(),
            source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=theta, alpha=alpha),
        )
    else:
        time_policy = pops.Explicit()
    sim.add_equation(
        "ions",
        model=iso_fluid_model(cs2=cs2, alpha=alpha),
        spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative"),
        time=time_policy,
    )
    rho = _annular_density(nr, nth)
    sim.set_density("ions", rho)             # pose rho, vitesse au repos
    sim.set_state("ions", _initial_velocity_state(nr, nth, rho).ravel())  # injecte la vitesse initiale
    return sim


# ---------------------------------------------------------------------------
# Helpers d'assertion
# ---------------------------------------------------------------------------

def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        raise AssertionError(label)


def state3(sim, nr, nth):
    return np.array(sim.get_state("ions")).reshape(3, nth, nr)


def mom_l2(sim, nr, nth):
    u = state3(sim, nr, nth)
    return float(np.sqrt(np.sum(u[1] ** 2 + u[2] ** 2)))


# ---------------------------------------------------------------------------
# Test principal
# ---------------------------------------------------------------------------

def test_polar_schur_via_system():
    nr, nth = 32, 32          # nr == ntheta : carre (le B_z passe par n*n et nr*ntheta identiques)
    B0, alpha, cs2 = 6.0, 3.0, 1.0
    h = min((RMAX - RMIN) / nr, RMIN * (2.0 * math.pi / nth))  # pas physique min polaire
    dt = 0.2 * h / math.sqrt(cs2)   # CFL transport stable, mais dt >> 1/omega_c (B0 eleve -> raide)
    n_steps = 6

    # (A/B/C) run avec etage Schur POLAIRE : fini, source active, densite bornee.
    sim = build_system(nr, nth, B0=B0, alpha=alpha, theta=1.0, with_schur=True, cs2=cs2)
    rho_before = state3(sim, nr, nth)[0]
    for _ in range(n_steps):
        sim.step(dt)
    s = state3(sim, nr, nth)
    rho_after, mx_after, my_after = s[0], s[1], s[2]

    chk(np.all(np.isfinite(s)), "(A) etat fini apres run avec etage Schur polaire")

    # (B) la source a tourne : la quantite de mvt differe du run SANS Schur.
    sim_no = build_system(nr, nth, B0=B0, alpha=alpha, theta=1.0, with_schur=False, cs2=cs2)
    for _ in range(n_steps):
        sim_no.step(dt)
    s_no = state3(sim_no, nr, nth)
    diff_mom = float(np.max(np.abs(
        np.sqrt(s[1] ** 2 + s[2] ** 2) - np.sqrt(s_no[1] ** 2 + s_no[2] ** 2))))
    chk(diff_mom > 1e-6,
        "(B) l'etage Schur polaire modifie la quantite de mvt (diff vs no-source = %.3e)" % diff_mom)

    # (C) densite bornee + positive (la source ne traite que la vitesse, rho via transport seulement).
    chk(np.all(np.isfinite(rho_after)) and float(np.min(rho_after)) > 0.0,
        "(C) densite finie et positive apres l'etage Schur polaire")
    chk(float(np.max(rho_after)) < 3.0 * float(np.max(rho_before)),
        "(C) densite bornee (rho <= 3 rho0) apres l'etage Schur polaire")

    # (D) STABILITE sous source RAIDE : B0 eleve, theta=1, ||v|| reste finie et bornee sur K pas.
    B0_stiff, alpha_stiff, K = 20.0, 8.0, 30
    sim_stiff = build_system(nr, nth, B0=B0_stiff, alpha=alpha_stiff, theta=1.0, with_schur=True, cs2=cs2)
    v0 = mom_l2(sim_stiff, nr, nth)
    dt_stiff = 0.4 * h / math.sqrt(cs2)
    for _ in range(K):
        sim_stiff.step(dt_stiff)
    s_stiff = state3(sim_stiff, nr, nth)
    chk(np.all(np.isfinite(s_stiff)), "(D) etat fini sous source raide (theta=1, implicite)")
    v_stiff = mom_l2(sim_stiff, nr, nth)
    chk(v_stiff < 20.0 * max(v0, 1e-12),
        "(D) ||v|| bornee sous source raide (pas d'explosion, v_stiff=%.3e, v0=%.3e)" % (v_stiff, v0))

    print("test_polar_schur_via_system : tout est vert (6 verifications)")


if __name__ == "__main__":
    test_polar_schur_via_system()

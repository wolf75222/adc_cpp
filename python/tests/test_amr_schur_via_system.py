#!/usr/bin/env python3
"""Couverture CI du chemin amr-schur : AmrSystem -> etage source condense GLOBAL par Schur.

Pendant AMR de test_schur_via_system.py. L'etage AmrCondensedSchurSourceStepper est teste en
STANDALONE (parite mono-niveau) dans test_amr_condensed_schur_source_stepper (C++) ; ici on exerce
le chemin FACADE de bout en bout :

  pops.AmrSystem.add_equation(time=pops.Strang(source=pops.CondensedSchur(...)))
    -> add_equation(time=time.hyperbolic = pops.Explicit())   # transport SOURCE-FREE (NoSource)
      -> _s.add_block (ModelSpec -> dispatch_amr_compiled -> AmrCouplerMP)
    -> _s.set_source_stage(name, kind, theta, alpha)          # etage condense GLOBAL
    -> _s.set_time_scheme("lie"|"strang")                     # splitting
  sim.step(dt)
    -> AmrCouplerMP : update() ; advance_transport (source-free) ; amr_schur_source ; ...

MODELE NATIF (CI-safe, sans compilateur C++) : pops.FluidState(isothermal) + pops.IsothermalFlux()
expose Density / MomentumX / MomentumY (les roles exiges par l'etage condense) avec source=NoSource
(le transport est source-free ; l'etage condense porte la source electrostatique/Lorentz). Hierarchie
MONO-NIVEAU (set_refinement(1e30) -> aucun patch fin) : l'etage degenere en l'etage uniforme.

Validations :
  A) density() / potential() FINIS apres N pas (nan/inf rejectes avant toute tolerance).
  B) MASSE conservee (l'etage condense gele rho ; le transport conserve la masse au reflux).
  C) L'etage source AGIT : la densite diverge d'un run SANS etage (Explicit pur) -- si l'etage etait
     un no-op, density() serait identique et la difference nulle.
  D) STABILITE sous source RAIDE (B0 grand, theta=1 implicite) : density() reste finie et bornee.
  E) GARDES : etage condense sans set_magnetic_field -> erreur claire au build ; add_block(Strang) ->
     erreur claire (le splitting passe par add_equation) ; theta/kind hors domaine -> ValueError.
"""

from pops.numerics.variables import Conservative
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import sys
import numpy as np

try:
    import pops
except ImportError as e:
    print("skip  module pops absent (PYTHONPATH ?) : %s" % e)
    sys.exit(0)


def iso_model(cs2=1.0, alpha=1.0):
    """Fluide isotherme NATIF : roles Density / MomentumX / MomentumY, source NoSource (transport
    source-free), Poisson de fond alpha*(n - 0)."""
    return pops.Model(
        state=pops.FluidState(kind="isothermal", cs2=cs2),
        transport=pops.IsothermalFlux(),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=alpha, n0=0.0),
    )


def smooth_state(n, L):
    """rho0 > 0, vitesse de derive lisse nulle aux bords (compatible Dirichlet)."""
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.5 * np.ones((n, n))
    u0 = 0.5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    v0 = -0.3 * np.sin(2.0 * np.pi * X / L) * np.sin(np.pi * Y / L)
    return rho0, u0, v0


def build_amr(n=24, L=1.0, B0=4.0, alpha=3.0, theta=1.0, with_schur=True, strang=True,
              cs2=1.0, set_bz=True):
    """AmrSystem MONO-NIVEAU (set_refinement desactive) avec un bloc isotherme natif.

    with_schur=True : add_equation(pops.Strang/Split(source=CondensedSchur)) -> etage condense GLOBAL.
    with_schur=False : add_equation(pops.Explicit()) -> transport seul (reference sans source)."""
    sim = pops.AmrSystem(n=n, L=L, periodic=False, regrid_every=0)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet")
    sim.set_refinement(1e30)  # aucun raffinement -> hierarchie MONO-NIVEAU (niveau fin seed vide)
    if set_bz:
        sim.set_magnetic_field(B0 * np.ones((n, n)))
    if with_schur:
        cls = pops.Strang if strang else pops.Split
        time_policy = cls(
            hyperbolic=pops.Explicit(),  # SSPRK2 -> kind 'explicit' (l'AMR n'a pas ssprk3)
            source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=theta, alpha=alpha),
        )
    else:
        time_policy = pops.Explicit()
    sim.add_equation(
        "electrons",
        model=iso_model(cs2=cs2, alpha=alpha),
        spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov(), variables=Conservative()),
        time=time_policy,
    )
    rho0, u0, v0 = smooth_state(n, L)
    sim.set_conservative_state("electrons", np.stack([rho0, rho0 * u0, rho0 * v0]))
    return sim


def chk(cond, label):
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        raise AssertionError(label)


def raises(fn, label):
    try:
        fn()
    except Exception:
        chk(True, label)
        return
    chk(False, label + " (aucune exception levee)")


def main():
    n, L = 24, 1.0
    B0, alpha, cs2 = 4.0, 3.0, 1.0
    dt = 5.0e-4
    nsteps = 5

    # ----------------------------------------------------------------- (A/B) run avec etage condense
    sim = build_amr(n=n, L=L, B0=B0, alpha=alpha, theta=1.0, with_schur=True, strang=True, cs2=cs2)
    mass0 = sim.mass()
    for _ in range(nsteps):
        sim.step(dt)
    rho = np.array(sim.density())
    phi = np.array(sim.potential())
    if not np.all(np.isfinite(rho)):
        raise AssertionError("density() non finie apres l'etage condense")
    if not np.all(np.isfinite(phi)):
        raise AssertionError("potential() non fini apres l'etage condense")
    chk(True, "(A) density()/potential() finis apres run avec etage condense (Strang)")

    mass1 = sim.mass()
    chk(abs(mass1 - mass0) <= 1e-9 * max(abs(mass0), 1e-30),
        "(B) masse conservee par l'etage condense (rel=%.2e)" % (abs(mass1 - mass0) / max(abs(mass0), 1e-30)))

    # ----------------------------------------------------------------- (C) l'etage est CABLE (pas un no-op)
    # On compare au chemin EXPLICIT pur (sans set_source_stage) : un etat distinct prouve que
    # set_source_stage route vers une EXECUTION distincte (pas un no-op silencieux qui perdrait la source).
    # NB : la parite STRICTE de l'etage source est prouvee cote C++ (test_amr_condensed_schur_source_stepper :
    # parite bit-pour-bit vs etage uniforme + relation implicite). Ici on valide le CABLAGE facade.
    sim_ref = build_amr(n=n, L=L, B0=B0, alpha=alpha, with_schur=False, cs2=cs2)
    for _ in range(nsteps):
        sim_ref.step(dt)
    rho_ref = np.array(sim_ref.density())
    diff = float(np.max(np.abs(rho - rho_ref)))
    chk(diff > 1e-9,
        "(C) set_source_stage route vers une execution distincte (vs explicit pur, diff=%.3e)" % diff)

    # (C2) DETERMINISME : deux runs amr-schur identiques -> densite bit-identique (pas d'UB, reproductible).
    sim_a = build_amr(n=n, L=L, B0=B0, alpha=alpha, theta=1.0, with_schur=True, strang=True, cs2=cs2)
    sim_b = build_amr(n=n, L=L, B0=B0, alpha=alpha, theta=1.0, with_schur=True, strang=True, cs2=cs2)
    for _ in range(nsteps):
        sim_a.step(dt)
        sim_b.step(dt)
    chk(float(np.max(np.abs(np.array(sim_a.density()) - np.array(sim_b.density())))) == 0.0,
        "(C2) chemin amr-schur deterministe (deux runs bit-identiques)")

    # ----------------------------------------------------------------- (D) stabilite sous source raide
    sim_stiff = build_amr(n=n, L=L, B0=10.0, alpha=8.0, theta=1.0, with_schur=True, strang=False,
                          cs2=cs2)
    dt_stiff = 0.4 * (L / n) / np.sqrt(cs2)
    for _ in range(30):
        sim_stiff.step(dt_stiff)
    rho_stiff = np.array(sim_stiff.density())
    if not np.all(np.isfinite(rho_stiff)):
        raise AssertionError("density() non finie sous source raide")
    chk(float(np.max(rho_stiff)) < 5.0 * 1.5,
        "(D) density() bornee sous source raide (theta=1, implicite ; max=%.3f)" % float(np.max(rho_stiff)))

    # ----------------------------------------------------------------- (E) gardes d'API
    # E1 : etage condense sans set_magnetic_field -> erreur claire au build (1er step).
    def no_bz():
        s = build_amr(n=n, L=L, B0=B0, alpha=alpha, with_schur=True, set_bz=False)
        s.step(dt)
    raises(no_bz, "(E1) etage condense sans set_magnetic_field -> erreur au build")

    # E2 : add_block(time=Strang) -> rejet (le splitting passe par add_equation).
    def add_block_strang():
        s = pops.AmrSystem(n=n, L=L, periodic=False)
        s.add_block("e", iso_model(),
                    time=pops.Strang(hyperbolic=pops.Explicit(),
                                    source=pops.CondensedSchur(theta=0.5, alpha=alpha)))
    raises(add_block_strang, "(E2) add_block(pops.Strang) rejete (passer par add_equation)")

    # E3 : theta hors (0, 1] et kind inconnu -> ValueError des pops.CondensedSchur (validation Python).
    raises(lambda: pops.CondensedSchur(theta=1.5, alpha=alpha), "(E3a) CondensedSchur(theta=1.5) rejete")
    raises(lambda: pops.CondensedSchur(kind="bidon", alpha=alpha), "(E3b) CondensedSchur(kind invalide) rejete")

    print("test_amr_schur_via_system : tout est vert")


if __name__ == "__main__":
    main()

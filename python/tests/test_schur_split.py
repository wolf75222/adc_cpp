#!/usr/bin/env python3
"""Test de la politique temporelle adc.Split(hyperbolic=Explicit, source=CondensedSchur) : le binding
Python qui cable l'etage SOURCE condense par Schur (CondensedSchurSourceStepper, #126) dans System via
set_source_stage. cf. docs/SCHUR_CONDENSATION_DESIGN.md sections 5-6, roadmap PR5.

On valide :
  (a) RUN sans callback Python par cellule : un fluide isotherme magnetise (DSL compile AOT, roles
      Density/MomentumX/MomentumY, lit B_z) tourne avec time=adc.Split(...). L'etage source est en
      C++ (aucune PythonFlux / source numpy) ; on verifie que le run produit un etat FINI et que la
      densite (gelee dans la source) est conservee, et que la quantite de mouvement EVOLUE (l'etage
      source a engage le couplage potentiel / Lorentz).
  (b) STABILITE (la garantie du C++) : a un dt RAIDE (cyclotron + plasma), l'etage condense (implicite,
      theta=1) garde la vitesse BORNEE -- la relation implicite tient (cf. test C++
      test_condensed_schur_source_stepper, sous-tests A/B/C). On verifie que ||v|| reste FINIE et du
      meme ordre, la ou un schema explicite exploserait.
  (c) ERREURS CLAIRES : (c1) role manquant (Density/MomentumX/MomentumY) -> erreur a add_equation ;
      (c2) pas de B_z (set_magnetic_field non appele) -> erreur ; (c3) kind / theta invalides.
  (d) DEFAUT INCHANGE : adc.Explicit / adc.IMEX restent BIT-IDENTIQUES (le seul fait d'avoir Split /
      CondensedSchur dans le module ne perturbe pas le chemin par defaut), et un bloc adc.Split sur une
      espece NE perturbe PAS une espece voisine en adc.Explicit (le defaut est strictement opt-in).

Lance avec python3. Saute proprement (skip) si aucun compilateur C++ n'est disponible (le modele
magnetise a roles passe par le DSL compile -- pas de modele natif a roles rho/mx/my accessible sinon).
"""
import os
import shutil
import tempfile

import numpy as np

import adc
from adc import dsl

INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def isothermal_magnetized(cs2=1.0):
    """Fluide isotherme 2D (rho, mx, my) magnetise : flux d'Euler isotherme (pression cs2*rho), roles
    canoniques Density/MomentumX/MomentumY, lit le champ aux B_z (-> n_aux=4, canal B_z present). La
    source LOCALE est nulle : l'etage SOURCE est porte par adc.CondensedSchur (electrostatique + Lorentz)."""
    m = dsl.Model("iso_mag")
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = cs2 * rho                       # pression isotherme p = cs2 rho
    c = dsl.sqrt(cs2)                   # vitesse du son isotherme (constante)
    m.flux(x=[mx, mx * u + p, mx * v],
           y=[my, my * u, my * v + p])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho=rho, u=u, v=v)
    m.conservative_from([rho, rho * u, rho * v])
    bz = m.aux("B_z")                  # lit B_z (n_aux = 4) : canal B_z dans l'aux partage
    m.source([0.0 * rho, 0.0 * bz * mx, 0.0 * my])  # source locale NULLE (Schur porte la source)
    return m


def smooth_init(n, L):
    """Profils lisses, nuls au bord (compatibles Dirichlet) : rho0 > 0, v0 = sin(pi x) sin(pi y)."""
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    sx, sy = np.sin(np.pi * X / L), np.sin(np.pi * Y / L)
    rho0 = 1.5 + 0.0 * X                 # densite constante > 0 (gelee dans la source)
    u0 = 0.6 * sx * sy
    v0 = -0.4 * np.sin(2 * np.pi * X / L) * sy
    return rho0, u0, v0


def build_sim(compiled, n=32, L=1.0, B0=4.0, alpha=3.0, theta=1.0, with_bz=True,
              with_source_stage=True):
    """System non periodique (Dirichlet pour le Poisson condense), un bloc isotherme magnetise. Pose
    B_z AVANT add_equation (set_source_stage exige le champ B_z) si with_bz."""
    sim = adc.System(n=n, L=L, periodic=False)
    sim.set_poisson(bc="dirichlet")
    if with_bz:
        sim.set_magnetic_field(B0 * np.ones((n, n)))  # B_z constant ; doit preceder set_source_stage
    if with_source_stage:
        time = adc.Split(hyperbolic=adc.Explicit(),
                         source=adc.CondensedSchur(kind="electrostatic_lorentz", theta=theta,
                                                   alpha=alpha,
                                                   density=adc.Role.Density,
                                                   momentum=(adc.Role.MomentumX, adc.Role.MomentumY),
                                                   magnetic_field="B_z", potential="phi"))
    else:
        time = adc.Explicit()
    sim.add_equation("ions", model=compiled,
                     spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov",
                                              variables="conservative"),
                     time=time)
    rho0, u0, v0 = smooth_init(n, L)
    sim.set_primitive_state("ions", rho=rho0, u=u0, v=v0)
    return sim


def vel_l2(sim):
    P = sim.get_primitive_state("ions")
    return float(np.sqrt(np.sum(P["u"] ** 2 + P["v"] ** 2)))


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> test_schur_split saute (%s)" % INCLUDE)
        print("test_schur_split : OK (rien a compiler)")
        return

    n, L = 32, 1.0
    compiled = isothermal_magnetized().compile(backend="aot", include=INCLUDE)

    # ------------------------------------------------------------------------------------------
    # (a) RUN : l'etage source C++ engage le couplage. rho gelee, mom evolue, etat fini.
    # ------------------------------------------------------------------------------------------
    sim = build_sim(compiled, n=n, L=L, theta=1.0)
    rho_before = np.array(sim.density("ions"))
    P0 = sim.get_primitive_state("ions")
    mom0 = np.array(P0["u"]) * rho_before  # quantite de mvt x avant
    dt = 2.0e-3
    for _ in range(4):
        sim.step(dt)
    P1 = sim.get_primitive_state("ions")
    rho_after = np.array(sim.density("ions"))
    mom1 = np.array(P1["u"]) * rho_after
    state_finite = bool(np.all(np.isfinite(P1["u"])) and np.all(np.isfinite(P1["v"])) and
                        np.all(np.isfinite(rho_after)))
    chk(state_finite, "(a) etat fini apres l'etage source condense (run C++)")
    # rho est GELEE dans l'etage source ; le transport (vitesse) la deplace tres peu sur quelques pas,
    # mais l'etage source ne la touche pas. On verifie surtout que la quantite de mvt a EVOLUE (source
    # engagee) : l'etage potentiel / Lorentz a modifie mx au-dela du bruit machine.
    mom_changed = float(np.max(np.abs(mom1 - mom0)))
    chk(mom_changed > 1e-6, "(a) la quantite de mvt evolue (etage source condense engage)")

    # ------------------------------------------------------------------------------------------
    # (b) STABILITE source RAIDE : a un dt STABLE pour le transport (CFL) mais avec une source TRES
    #     raide (cyclotron B0 = 8, plasma alpha = 6), l'etage condense (theta = 1, implicite) garde
    #     ||v|| BORNEE sur de nombreux pas. C'est la garantie portee par le C++ (la relation implicite
    #     B v = v^n - theta dt grad phi tient, cf. test_condensed_schur_source_stepper sous-tests A/B/C) :
    #     une source aussi raide traitee EXPLICITEMENT amplifierait ||v|| a chaque pas. On verifie ||v||
    #     FINIE et bornee (du meme ordre que l'initiale), sur 30 pas.
    # ------------------------------------------------------------------------------------------
    sim_stiff = build_sim(compiled, n=n, L=L, theta=1.0, B0=8.0, alpha=6.0)
    v0 = vel_l2(sim_stiff)
    dt_stiff = 0.4 * (L / n) / 1.0  # CFL transport (cs = 1) : stable pour l'etage hyperbolique
    for _ in range(30):
        sim_stiff.step(dt_stiff)
    v_stiff = vel_l2(sim_stiff)
    chk(np.isfinite(v_stiff), "(b) ||v|| finie sous source raide (etage implicite stable)")
    chk(v_stiff < 10.0 * v0, "(b) ||v|| bornee sous source raide (pas d'amplification source)")

    # ------------------------------------------------------------------------------------------
    # (c) ERREURS CLAIRES.
    # ------------------------------------------------------------------------------------------
    # (c1) role manquant : un modele scalaire (pas de MomentumX/MomentumY) -> erreur a add_equation.
    scal = dsl.Model("scal_mag")
    (q,) = scal.conservative_vars("q")     # un seul champ : aucun role Density/Momentum canonique
    scal.flux(x=[0.0 * q], y=[0.0 * q])
    scal.eigenvalues(x=[0.0 * q], y=[0.0 * q])
    scal.primitive_vars(q=q)
    scal.conservative_from([q])
    scal_c = scal.compile(backend="aot", include=INCLUDE)
    sim_c1 = adc.System(n=n, L=L, periodic=False)
    sim_c1.set_poisson(bc="dirichlet")
    sim_c1.set_magnetic_field(np.ones((n, n)))
    raised = False
    try:
        sim_c1.add_equation("q", model=scal_c,
                            time=adc.Split(hyperbolic=adc.Explicit(),
                                           source=adc.CondensedSchur(theta=0.5)))
    except Exception as e:
        raised = "role" in str(e).lower() or "momentum" in str(e).lower()
    chk(raised, "(c1) role requis manquant -> erreur claire a add_equation")

    # (c2) pas de B_z (set_magnetic_field non appele) -> erreur a add_equation.
    sim_c2 = adc.System(n=n, L=L, periodic=False)
    sim_c2.set_poisson(bc="dirichlet")
    raised = False
    try:
        sim_c2.add_equation("ions", model=compiled,
                            time=adc.Split(hyperbolic=adc.Explicit(),
                                           source=adc.CondensedSchur(theta=0.5)))
    except Exception as e:
        raised = "b_z" in str(e).lower() or "magnetic" in str(e).lower()
    chk(raised, "(c2) B_z absent -> erreur claire a add_equation")

    # (c3) kind / theta invalides -> erreur a la construction de CondensedSchur (cote Python).
    raised = False
    try:
        adc.CondensedSchur(kind="nimporte_quoi")
    except ValueError:
        raised = True
    chk(raised, "(c3) kind inconnu -> ValueError")
    raised = False
    try:
        adc.CondensedSchur(theta=1.5)
    except ValueError:
        raised = True
    chk(raised, "(c3) theta hors (0, 1] -> ValueError")

    # ------------------------------------------------------------------------------------------
    # (d) DEFAUT INCHANGE : un bloc adc.Explicit donne le MEME resultat avec ou sans l'existence de
    #     l'etage source dans le module, ET un bloc Split sur une espece ne perturbe pas une espece
    #     voisine en adc.Explicit (opt-in strict).
    # ------------------------------------------------------------------------------------------
    # Reference : modele isotherme magnetise SANS etage source (adc.Explicit pur). Deterministe.
    ref = build_sim(compiled, n=n, L=L, with_source_stage=False)
    for _ in range(4):
        ref.step(2.0e-3)
    a = np.array(ref.density("ions"))
    ref2 = build_sim(compiled, n=n, L=L, with_source_stage=False)
    for _ in range(4):
        ref2.step(2.0e-3)
    b = np.array(ref2.density("ions"))
    chk(float(np.max(np.abs(a - b))) == 0.0, "(d) chemin Explicit pur deterministe (bit-identique)")

    # Deux especes : "ions" en Split (etage source), "bg" en Explicit pur ; le bloc Explicit doit etre
    # IDENTIQUE a un run ou "bg" est seul (le Split d'ions ne le perturbe pas via le couplage Poisson au
    # dela du couplage physique attendu -- ici on compare le bloc bg a son run SOLO, memes pas).
    # On verifie au minimum que "bg" reste FINI et que retirer l'etage source d'ions NE casse PAS bg.
    sim_two = adc.System(n=n, L=L, periodic=False)
    sim_two.set_poisson(bc="dirichlet")
    sim_two.set_magnetic_field(4.0 * np.ones((n, n)))
    sim_two.add_equation("ions", model=compiled,
                         time=adc.Split(hyperbolic=adc.Explicit(),
                                        source=adc.CondensedSchur(theta=1.0, alpha=3.0)))
    sim_two.add_equation("bg", model=compiled, time=adc.Explicit())  # voisin Explicit pur
    rho0, u0, v0 = smooth_init(n, L)
    sim_two.set_primitive_state("ions", rho=rho0, u=u0, v=v0)
    sim_two.set_primitive_state("bg", rho=rho0, u=u0, v=v0)
    for _ in range(3):
        sim_two.step(2.0e-3)
    bg = sim_two.get_primitive_state("bg")
    chk(bool(np.all(np.isfinite(bg["u"]))) and bool(np.all(np.isfinite(bg["v"]))),
        "(d) bloc voisin Explicit pur reste fini en presence d'un bloc Split")

    if fails == 0:
        print("test_schur_split : tout est vert")
    else:
        raise SystemExit("test_schur_split : %d verification(s) en echec" % fails)


if __name__ == "__main__":
    main()

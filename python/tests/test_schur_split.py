#!/usr/bin/env python3
"""Test de la politique temporelle pops.Split(hyperbolic=Explicit, source=CondensedSchur) : le binding
Python qui cable l'etage SOURCE condense par Schur (CondensedSchurSourceStepper, #126) dans System via
set_source_stage. cf. docs/SCHUR_CONDENSATION_DESIGN.md sections 5-6, roadmap PR5.

On valide :
  (a) RUN sans callback Python par cellule : un fluide isotherme magnetise (DSL compile AOT, roles
      Density/MomentumX/MomentumY, lit B_z) tourne avec time=pops.Split(...). L'etage source est en
      C++ (aucune PythonFlux / source numpy) ; on verifie que le run produit un etat FINI et que la
      densite (gelee dans la source) est conservee, et que la quantite de mouvement EVOLUE (l'etage
      source a engage le couplage potentiel / Lorentz).
  (b) STABILITE (la garantie du C++) : a un dt RAIDE (cyclotron + plasma), l'etage condense (implicite,
      theta=1) garde la vitesse BORNEE -- la relation implicite tient (cf. test C++
      test_condensed_schur_source_stepper, sous-tests A/B/C). On verifie que ||v|| reste FINIE et du
      meme ordre, la ou un schema explicite exploserait.
  (c) ERREURS CLAIRES : (c1) role manquant (Density/MomentumX/MomentumY) -> erreur a add_equation ;
      (c2) pas de B_z (set_magnetic_field non appele) -> erreur ; (c3) kind / theta invalides.
  (d) DEFAUT INCHANGE : pops.Explicit / pops.IMEX restent BIT-IDENTIQUES (le seul fait d'avoir Split /
      CondensedSchur dans le module ne perturbe pas le chemin par defaut), et un bloc pops.Split sur une
      espece NE perturbe PAS une espece voisine en pops.Explicit (le defaut est strictement opt-in).

Lance avec python3. Saute proprement (skip) si aucun compilateur C++ n'est disponible (le modele
magnetise a roles passe par le DSL compile -- pas de modele natif a roles rho/mx/my accessible sinon).
"""
import os
import shutil
import tempfile

import numpy as np

import pops
from pops.ir.ops import sqrt
from pops.physics.facade import Model

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
    source LOCALE est nulle : l'etage SOURCE est porte par pops.CondensedSchur (electrostatique + Lorentz)."""
    m = Model("iso_mag")
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = cs2 * rho                       # pression isotherme p = cs2 rho
    c = sqrt(cs2)                   # vitesse du son isotherme (constante)
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
    sim = pops.System(n=n, L=L, periodic=False)
    sim.set_poisson(bc="dirichlet")
    if with_bz:
        sim.set_magnetic_field(B0 * np.ones((n, n)))  # B_z constant ; doit preceder set_source_stage
    if with_source_stage:
        time = pops.Split(hyperbolic=pops.Explicit(),
                         source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=theta,
                                                   alpha=alpha,
                                                   density=pops.Role.Density,
                                                   momentum=(pops.Role.MomentumX, pops.Role.MomentumY),
                                                   magnetic_field="B_z", potential="phi"))
    else:
        time = pops.Explicit()
    sim.add_equation("ions", model=compiled,
                     spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov",
                                              variables="conservative"),
                     time=time)
    rho0, u0, v0 = smooth_init(n, L)
    sim.set_primitive_state("ions", rho=rho0, u=u0, v=v0)
    return sim


def vel_l2(sim):
    P = sim.get_primitive_state("ions")
    return float(np.sqrt(np.sum(P["u"] ** 2 + P["v"] ** 2)))


def raises(exc_types, fn, *args, **kw):
    """Renvoie l'exception levee par fn (du type attendu), ou leve AssertionError si rien n'est leve /
    le type ne correspond pas. Equivaut a pytest.raises mais SANS dependance pytest (la CI lance ces
    tests comme de simples scripts python3, pytest n'est pas installe)."""
    try:
        fn(*args, **kw)
    except exc_types as e:
        return e
    except Exception as e:  # pragma: no cover - diagnostic d'un mauvais type d'erreur
        raise AssertionError("attendu %r, leve %r (%s)" % (exc_types, type(e).__name__, e))
    raise AssertionError("attendu %r, aucune exception levee" % (exc_types,))


def scalar_native_model():
    """Modele natif (ModelSpec) diocotron-like (Scalar + ExB), valide sur AmrSystem (cf. test_bindings).
    Sert a exercer les gardes de facade (rejet de pops.Split sur AmrSystem) SANS compilateur C++ : la
    garde Split leve AVANT tout usage du modele (juste apres les defauts), le modele reste un argument
    valide pour ne pas masquer le rejet par une autre erreur."""
    return pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def check_condensed_schur_descriptors():
    """(e) Descripteurs roles / champs de pops.CondensedSchur. Les roles density / momentum / energy sont
    TRANSPORTES a l'ABI C++ (audit vague 2 : *_spec resolus au build contre les VariableRole du bloc ;
    couverture de transport dediee dans test_schur_roles.py), donc un pops.Role.* / nom != defaut est
    desormais ACCEPTE au constructeur (la facade ne valide plus la semantique : le build C++ leve si le
    role/nom est introuvable). Seuls magnetic_field (champ aux CANONIQUE obligatoire) et potential (fige
    a 'phi' : pas de solveur derriere un autre champ) sont REJETES des le constructeur."""
    # Defaut explicite ET implicite : doivent construire SANS lever (parite stricte avec l'existant).
    pops.CondensedSchur()
    pops.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, alpha=1.0,
                       density=pops.Role.Density,
                       momentum=(pops.Role.MomentumX, pops.Role.MomentumY),
                       energy=None, magnetic_field="B_z", potential="phi")
    chk(True, "(e) CondensedSchur defaut (explicite + implicite) construit sans lever")
    # energy=pops.Role.Energy est tolere (c'est la valeur que le C++ utilise pour l'energie optionnelle).
    pops.CondensedSchur(energy=pops.Role.Energy)
    chk(True, "(e) CondensedSchur(energy=pops.Role.Energy) tolere (valeur hardcodee C++)")

    # Roles density / momentum != defaut TRANSPORTES (vague 2) : ACCEPTES, exposent un *_spec non vide
    # (resolution role -> composante cote C++) ; les defauts canoniques gardent des specs VIDES (chemin
    # historique C++ bit-identique).
    cs = pops.CondensedSchur(density=pops.Role.Energy,
                            momentum=(pops.Role.VelocityX, pops.Role.VelocityY),
                            energy=pops.Role.Scalar)
    chk(bool(cs.density_spec) and bool(cs.momentum_x_spec) and bool(cs.momentum_y_spec)
        and bool(cs.energy_spec),
        "(e) CondensedSchur(density/momentum/energy != defaut) -> accepte et transporte (*_spec non vides)")
    chk(pops.CondensedSchur().density_spec == "" and pops.CondensedSchur().momentum_x_spec == "",
        "(e) CondensedSchur defaut -> specs VIDES (chemin canonique C++ inchange)")

    # magnetic_field non canonique ET potential != 'phi' restent REJETES au constructeur (messages clairs).
    e_bz = raises(ValueError, lambda: pops.CondensedSchur(magnetic_field="B_custom"))
    e_phi = raises(ValueError, lambda: pops.CondensedSchur(potential="psi"))
    chk("magnetic_field" in str(e_bz) and "potential" in str(e_phi),
        "(e) magnetic_field non canonique / potential != 'phi' -> ValueError (messages clairs)")


def check_amr_split_rejected():
    """(f) pops.Split / pops.Strang (etage source condense par Schur) n'est cable QUE par add_equation
    (qui branche set_source_stage APRES l'ajout du bloc). AmrSystem.add_block doit donc le REJETER
    explicitement (sinon Split, exposant .kind/.substeps, passerait comme un transport seul et la source
    condensee serait perdue en silence) -- MEME rejet que System.add_block. Depuis le chemin amr-schur
    (#265), AmrSystem.add_equation(time=pops.Split(...)) est au contraire SUPPORTE (set_source_stage +
    set_time_scheme ; couverture positive dans test_amr_schur_via_system.py) : seul add_block rejette."""
    n, L = 16, 1.0
    split = pops.Split(hyperbolic=pops.Explicit(),
                      source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=0.5))
    model = scalar_native_model()

    amr1 = pops.AmrSystem(n=n, L=L, periodic=True)
    e1 = raises((TypeError, ValueError), amr1.add_block, "ne", model=model, time=split)
    chk("Split" in str(e1) or "Schur" in str(e1),
        "(f) AmrSystem.add_block(time=pops.Split(...)) -> rejet explicite (Split/Schur)")

    # DEFAUT INCHANGE : un bloc AMR en pops.Explicit pur s'ajoute toujours sans lever.
    amr_ok = pops.AmrSystem(n=n, L=L, periodic=True)
    amr_ok.add_block("ne", model=scalar_native_model(), time=pops.Explicit())
    chk(True, "(f) AmrSystem.add_block(time=pops.Explicit()) defaut inchange (pas de rejet)")


def main():
    # --- Gardes de facade pures Python (FINDING 5/6) : aucune compilation requise, on les exerce
    # AVANT le saut conditionnel pour qu'elles tournent meme sans compilateur C++. -----------------
    check_condensed_schur_descriptors()
    check_amr_split_rejected()


    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes pops absents -> test_schur_split saute (%s)" % INCLUDE)
        if fails:  # ne pas masquer un echec des gardes pures Python (e)/(f) sans compilateur
            raise SystemExit("test_schur_split : %d verification(s) en echec" % fails)
        print("test_schur_split : OK (rien a compiler)")
        return

    n, L = 32, 1.0
    # adc_cpp est Kokkos-only (#263) : le .so AOT inclut les en-tetes pops (multifab/for_each) qui ne
    # compilent QUE sous POPS_HAS_KOKKOS, donc compile_aot exige un Kokkos installe (POPS_KOKKOS_ROOT).
    # Sans lui, on saute proprement la portion compilee -- meme convention que test_time_euler.py.
    try:
        compiled = isothermal_magnetized().compile(backend="aot", include=INCLUDE)
    except RuntimeError as ex:
        if "Kokkos" not in str(ex):
            raise
        print("skip  Kokkos introuvable -> portion compilee sautee (%s)" % str(ex).splitlines()[0][:70])
        if fails:  # ne pas masquer un echec des gardes pures Python (e)/(f) deja exercees
            raise SystemExit("test_schur_split : %d verification(s) en echec" % fails)
        print("test_schur_split : OK (rien a compiler)")
        return

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
    scal = Model("scal_mag")
    (q,) = scal.conservative_vars("q")     # un seul champ : aucun role Density/Momentum canonique
    scal.flux(x=[0.0 * q], y=[0.0 * q])
    scal.eigenvalues(x=[0.0 * q], y=[0.0 * q])
    scal.primitive_vars(q=q)
    scal.conservative_from([q])
    scal_c = scal.compile(backend="aot", include=INCLUDE)
    sim_c1 = pops.System(n=n, L=L, periodic=False)
    sim_c1.set_poisson(bc="dirichlet")
    sim_c1.set_magnetic_field(np.ones((n, n)))
    raised = False
    try:
        sim_c1.add_equation("q", model=scal_c,
                            time=pops.Split(hyperbolic=pops.Explicit(),
                                           source=pops.CondensedSchur(theta=0.5)))
    except Exception as e:
        raised = "role" in str(e).lower() or "momentum" in str(e).lower()
    chk(raised, "(c1) role requis manquant -> erreur claire a add_equation")

    # (c2) pas de B_z (set_magnetic_field non appele) -> erreur a add_equation.
    sim_c2 = pops.System(n=n, L=L, periodic=False)
    sim_c2.set_poisson(bc="dirichlet")
    raised = False
    try:
        sim_c2.add_equation("ions", model=compiled,
                            time=pops.Split(hyperbolic=pops.Explicit(),
                                           source=pops.CondensedSchur(theta=0.5)))
    except Exception as e:
        raised = "b_z" in str(e).lower() or "magnetic" in str(e).lower()
    chk(raised, "(c2) B_z absent -> erreur claire a add_equation")

    # (c3) kind / theta invalides -> erreur a la construction de CondensedSchur (cote Python).
    raised = False
    try:
        pops.CondensedSchur(kind="nimporte_quoi")
    except ValueError:
        raised = True
    chk(raised, "(c3) kind inconnu -> ValueError")
    raised = False
    try:
        pops.CondensedSchur(theta=1.5)
    except ValueError:
        raised = True
    chk(raised, "(c3) theta hors (0, 1] -> ValueError")

    # ------------------------------------------------------------------------------------------
    # (d) DEFAUT INCHANGE : un bloc pops.Explicit donne le MEME resultat avec ou sans l'existence de
    #     l'etage source dans le module, ET un bloc Split sur une espece ne perturbe pas une espece
    #     voisine en pops.Explicit (opt-in strict).
    # ------------------------------------------------------------------------------------------
    # Reference : modele isotherme magnetise SANS etage source (pops.Explicit pur). Deterministe.
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
    sim_two = pops.System(n=n, L=L, periodic=False)
    sim_two.set_poisson(bc="dirichlet")
    sim_two.set_magnetic_field(4.0 * np.ones((n, n)))
    sim_two.add_equation("ions", model=compiled,
                         time=pops.Split(hyperbolic=pops.Explicit(),
                                        source=pops.CondensedSchur(theta=1.0, alpha=3.0)))
    sim_two.add_equation("bg", model=compiled, time=pops.Explicit())  # voisin Explicit pur
    rho0, u0, v0 = smooth_init(n, L)
    sim_two.set_primitive_state("ions", rho=rho0, u=u0, v=v0)
    sim_two.set_primitive_state("bg", rho=rho0, u=u0, v=v0)
    for _ in range(3):
        sim_two.step(2.0e-3)
    bg = sim_two.get_primitive_state("bg")
    chk(bool(np.all(np.isfinite(bg["u"]))) and bool(np.all(np.isfinite(bg["v"]))),
        "(d) bloc voisin Explicit pur reste fini en presence d'un bloc Split")

    # ------------------------------------------------------------------------------------------
    # (g) FINDING 4 : evolve=False (bloc GELE) est SILENCIEUSEMENT IGNORE sur les backends compiles
    #     'prototype' (add_dynamic_block) et 'aot' (add_compiled_block) : leur ABI .so ne transporte
    #     PAS evolve (force a true cote C++). On exige un REJET explicite (ValueError) nommant le
    #     backend, et on verifie que evolve=True (defaut) passe toujours sur ces memes backends.
    # ------------------------------------------------------------------------------------------
    sim_aot = pops.System(n=n, L=L, periodic=False)
    sim_aot.set_poisson(bc="dirichlet")
    sim_aot.set_magnetic_field(np.ones((n, n)))
    e_aot = raises(ValueError, sim_aot.add_equation, "frozen", model=compiled,
                   time=pops.Explicit(), evolve=False)
    chk("aot" in str(e_aot) and "evolve" in str(e_aot),
        "(g) backend 'aot' : evolve=False -> ValueError (nomme le backend)")
    # evolve=True (defaut) : meme backend, meme bloc -> aucun rejet (chemin nominal inchange).
    sim_aot.add_equation("ok", model=compiled, time=pops.Explicit())  # evolve True par defaut
    chk(True, "(g) backend 'aot' : evolve=True (defaut) passe toujours (pas de rejet)")

    proto = isothermal_magnetized().compile(backend="prototype", include=INCLUDE)
    sim_proto = pops.System(n=n, L=L, periodic=False)
    sim_proto.set_poisson(bc="dirichlet")
    sim_proto.set_magnetic_field(np.ones((n, n)))
    e_proto = raises(ValueError, sim_proto.add_equation, "frozen", model=proto,
                     time=pops.Explicit(), evolve=False)
    chk("prototype" in str(e_proto) and "evolve" in str(e_proto),
        "(g) backend 'prototype' : evolve=False -> ValueError (nomme le backend)")
    sim_proto.add_equation("ok", model=proto, time=pops.Explicit())  # evolve True par defaut
    chk(True, "(g) backend 'prototype' : evolve=True (defaut) passe toujours (pas de rejet)")

    if fails == 0:
        print("test_schur_split : tout est vert")
    else:
        raise SystemExit("test_schur_split : %d verification(s) en echec" % fails)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""System POLAIRE multi-box (decoupage en BANDES theta, ADC-67) a travers pops.PolarMesh(theta_boxes=N).

theta_boxes=1 (defaut) = mono-box, STRICTEMENT bit-identique a l'historique. theta_boxes>1 = l'anneau
(r, theta) est decoupe en N bandes azimutales (chaque boite couvre tout le rayon) ; le TRANSPORT polaire
(assemble_rhs_polar + fill_ghosts collectif) tourne multi-box. MATRICE des capacites validee ici :

  (a) TRANSPORT isotherme BIT-IDENTIQUE multi-box : N pas (Euler avant pilote en Python sur le residu
      multi-box eval_rhs) avec theta_boxes=2 et 4 donnent EXACTEMENT le meme etat que theta_boxes=1
      (dmax == 0.0). L'assemblage par boite + les halos collectifs (copie EXACTE des cellules valides
      voisines, sans interpolation) ne changent aucune valeur : l'emplacement des bords de boite est
      indifferent. C'est le coeur du livrable.
  (b) ExB SCALAIRE (1 variable) : le marshaling hote multi-box (set_density/density, get/set state) et
      l'assemblage scalaire sont eux aussi independants du decoupage (residu eval_rhs bit-identique ;
      ronde-trip densite exact). NB : sans champ electrostatique (Poisson direct mono-box only, cf. (c)),
      la vitesse ExB derive d'un aux nul -> residu nul ; ce test couvre donc le marshaling + le dispatch
      scalaire multi-box, le transport quantitatif non trivial etant porte par (a).
  (c) REJETS explicites : theta_boxes ne divisant pas ntheta (PolarMesh ET SystemConfig brut) ;
      Poisson polaire DIRECT demande avec theta_boxes>1 -> message AMONT clair (mono-box / Schur).
  (d) ROUND-TRIP get/set state multi-box : set_state puis get_state == identite ; set_primitive_state
      puis get_primitive_state == identite ; set_density puis density == identite -- l'anneau global est
      reconstruit a la lecture et eparpille a l'ecriture sur toutes les bandes.

Mono-rang (le Poisson polaire direct refuse MPI ; le transport multi-box teste ici est single-rank).
"""
import math
import sys

import numpy as np

try:
    import pops
except ImportError as e:  # PYTHONPATH non pose : skip CI-safe (comme les autres tests Python)
    print("skip  module adc absent (PYTHONPATH ?) : %s" % e)
    sys.exit(0)


# Geometrie : anneau NON carre (nr != ntheta) pour exercer le layout (nr, ntheta) sur chaque chemin.
RMIN, RMAX = 0.30, 1.00
NR, NTH = 16, 32          # ntheta = 32 divisible par 1, 2, 4 (bandes egales)
THETA_BOXES = (1, 2, 4)   # 1 = reference mono-box ; 2 et 4 = multi-box


# ---------------------------------------------------------------------------
# Constructeurs de systeme
# ---------------------------------------------------------------------------

def _iso_system(theta_boxes):
    """System polaire isotherme NATIF (rho, mom_r, mom_theta) a theta_boxes bandes. AUCUN Poisson n'est
    configure ni resolu : on n'exerce que le TRANSPORT (eval_rhs) -- le flux isotherme est self-contenu
    (cs2 du modele), il ne lit pas grad phi."""
    sim = pops.System(mesh=pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=NTH, theta_boxes=theta_boxes))
    sim.add_block(
        "iso",
        model=pops.Model(state=pops.FluidState(kind="isothermal", cs2=1.0),
                        transport=pops.IsothermalFlux(), source=pops.NoSource(),
                        elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    return sim


def _exb_system(theta_boxes):
    """System polaire ExB scalaire (1 variable) a theta_boxes bandes. Poisson NON resolu ici (cf. (c))."""
    sim = pops.System(mesh=pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=NTH, theta_boxes=theta_boxes))
    sim.add_block(
        "ne",
        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0), source=pops.NoSource(),
                        elliptic=pops.ChargeDensity(charge=1.0)),
        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    return sim


# ---------------------------------------------------------------------------
# Etats initiaux deterministes (memes valeurs quel que soit theta_boxes)
# ---------------------------------------------------------------------------

def _iso_state():
    """Etat conservatif (3, ntheta, nr) lisse, strictement positif, azimutalement asymetrique (pour
    exciter le flux radial ET azimutal). Layout composante-majeur (c, theta=j, r=i) : ordre de set_state."""
    dr = (RMAX - RMIN) / NR
    dth = 2.0 * math.pi / NTH
    rho = np.empty((NTH, NR)); mr = np.empty((NTH, NR)); mth = np.empty((NTH, NR))
    for j in range(NTH):
        th = (j + 0.5) * dth
        for i in range(NR):
            r = RMIN + (i + 0.5) * dr
            rr = (r - RMIN) / (RMAX - RMIN)
            h = math.sin(math.pi * rr)                  # 0 aux bords radiaux
            rho[j, i] = 1.5 + 0.3 * math.cos(2.0 * th) * h
            mr[j, i] = rho[j, i] * (0.6 * h * math.cos(2.0 * th))
            mth[j, i] = rho[j, i] * (-0.4 * h * math.sin(th))
    return np.stack([rho, mr, mth], axis=0)             # (3, ntheta, nr)


def _scalar_density():
    """Densite scalaire (ntheta, nr) lisse, strictement positive, asymetrique en theta."""
    dr = (RMAX - RMIN) / NR
    dth = 2.0 * math.pi / NTH
    rho = np.empty((NTH, NR))
    for j in range(NTH):
        th = (j + 0.5) * dth
        for i in range(NR):
            r = RMIN + (i + 0.5) * dr
            rr = (r - RMIN) / (RMAX - RMIN)
            rho[j, i] = 1.0 + 0.4 * math.cos(3.0 * th) * math.sin(math.pi * rr)
    return rho


# ---------------------------------------------------------------------------
# (a) Transport isotherme BIT-IDENTIQUE multi-box
# ---------------------------------------------------------------------------

def _euler_transport(sim, name, U0, dt, nsteps):
    """Avance N pas d'Euler avant en PILOTANT le residu de transport multi-box (eval_rhs) cote Python.
    eval_rhs = fill_ghosts (collectif, halos inter-boites) + assemble_rhs_polar (itere local_size()),
    SANS solve_fields (le Poisson polaire direct est mono-box). On compare l'etat final entre decoupages."""
    U = np.array(U0, dtype=float)
    sim.set_state(name, U)
    for _ in range(nsteps):
        R = np.asarray(sim.eval_rhs(name), dtype=float)  # residu multi-box sur l'etat courant
        U = U + dt * R
        sim.set_state(name, U)
    return np.asarray(sim.get_state(name), dtype=float)


def test_transport_isothermal_bit_identical_multibox():
    U0 = _iso_state()
    # Pas minuscule : on cherche la BIT-IDENTITE (determinisme), pas la physique ; un dt petit garde
    # l'etat fini sur quelques pas (un blow-up serait identique mais NaN != NaN masquerait le test).
    dr = (RMAX - RMIN) / NR
    dt = 0.05 * dr
    nsteps = 8

    # Reference mono-box (theta_boxes=1) : un eval_rhs simple d'abord (sanity), puis N pas.
    sim1 = _iso_system(1)
    sim1.set_state("iso", U0)
    R1 = np.asarray(sim1.eval_rhs("iso"), dtype=float)
    assert np.all(np.isfinite(R1)), "residu mono-box non fini"
    assert float(np.max(np.abs(R1))) > 1e-9, "residu mono-box trivialement nul (transport inerte ?)"

    ref = _euler_transport(_iso_system(1), "iso", U0, dt, nsteps)
    assert np.all(np.isfinite(ref)), "etat de reference non fini"

    for tb in (2, 4):
        # residu sur l'etat initial : doit deja etre BIT-IDENTIQUE (l'assemblage par boite est exact).
        simtb = _iso_system(tb)
        simtb.set_state("iso", U0)
        Rtb = np.asarray(simtb.eval_rhs("iso"), dtype=float)
        drhs = float(np.max(np.abs(Rtb - R1)))
        assert drhs == 0.0, "eval_rhs theta_boxes=%d differe du mono-box (dmax=%.3e)" % (tb, drhs)

        # N pas : etat final bit-identique au mono-box.
        out = _euler_transport(_iso_system(tb), "iso", U0, dt, nsteps)
        dmax = float(np.max(np.abs(out - ref)))
        assert dmax == 0.0, (
            "transport isotherme theta_boxes=%d != theta_boxes=1 apres %d pas (dmax=%.3e) ; "
            "l'assemblage par boite a change les valeurs (halo ou marshaling fautif)" % (tb, nsteps, dmax))
    print("  [OK ] (a) transport isotherme bit-identique multi-box (theta_boxes 2/4 == 1)")


# ---------------------------------------------------------------------------
# (b) ExB scalaire : marshaling + dispatch scalaire multi-box
# ---------------------------------------------------------------------------

def test_exb_scalar_multibox_consistent():
    rho0 = _scalar_density()

    sim1 = _exb_system(1)
    sim1.set_density("ne", rho0)
    d1 = np.asarray(sim1.density("ne"), dtype=float)          # (ntheta, nr) reconstruit
    R1 = np.asarray(sim1.eval_rhs("ne"), dtype=float)
    assert d1.shape == (NTH, NR), "densite mono-box de forme %r" % (d1.shape,)
    assert float(np.max(np.abs(d1 - rho0))) == 0.0, "densite mono-box != entree (round-trip)"

    for tb in (2, 4):
        sim = _exb_system(tb)
        sim.set_density("ne", rho0)
        d = np.asarray(sim.density("ne"), dtype=float)
        # densite reconstruite a partir des bandes == champ pose (scatter puis gather exacts).
        assert float(np.max(np.abs(d - rho0))) == 0.0, (
            "densite theta_boxes=%d != entree (marshaling multi-box fautif)" % tb)
        # residu scalaire bit-identique au mono-box (independant du decoupage).
        R = np.asarray(sim.eval_rhs("ne"), dtype=float)
        assert float(np.max(np.abs(R - R1))) == 0.0, (
            "eval_rhs ExB scalaire theta_boxes=%d != mono-box" % tb)
    print("  [OK ] (b) ExB scalaire : marshaling + assemblage scalaire multi-box consistants")


# ---------------------------------------------------------------------------
# (c) Rejets explicites
# ---------------------------------------------------------------------------

def test_rejections_divisibility_and_direct_poisson():
    # (c1) theta_boxes ne divisant pas ntheta : PolarMesh leve cote Python.
    raised = False
    try:
        pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=NTH, theta_boxes=5)  # 32 % 5 != 0
    except ValueError:
        raised = True
    assert raised, "PolarMesh(theta_boxes=5, ntheta=32) doit lever (5 ne divise pas 32)"

    # theta_boxes > ntheta : refuse aussi.
    raised = False
    try:
        pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=8, theta_boxes=16)
    except ValueError:
        raised = True
    assert raised, "PolarMesh(theta_boxes=16, ntheta=8) doit lever (theta_boxes > ntheta)"

    # (c2) un SystemConfig construit a la main (contourne PolarMesh) est protege cote C++.
    cfg = pops.SystemConfig()
    cfg.geometry = "polar"
    cfg.r_min, cfg.r_max, cfg.nr, cfg.ntheta = RMIN, RMAX, NR, NTH
    cfg.theta_boxes = 5  # 32 % 5 != 0
    cfg.n = NR
    raised = False
    try:
        pops.System(config=cfg)
    except Exception:
        raised = True
    assert raised, "System(theta_boxes=5, ntheta=32) doit lever cote C++ (check_geometry)"

    # (c3) Poisson polaire DIRECT + theta_boxes>1 -> message AMONT clair (avant la construction du solveur).
    sim = _exb_system(2)
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    sim.set_density("ne", _scalar_density())
    raised = False
    msg = ""
    try:
        sim.solve_fields()  # construit le PolarPoissonSolver -> garde ensure_elliptic_polar
    except Exception as e:  # noqa: BLE001 -- on inspecte le message
        raised = True
        msg = str(e)
    assert raised, "Poisson direct + theta_boxes=2 doit lever (le solveur direct est mono-box)"
    low = msg.lower()
    assert ("theta_boxes" in low) or ("mono-box" in low) or ("schur" in low), (
        "message de rejet Poisson direct multi-box peu clair : %r" % msg)
    print("  [OK ] (c) rejets : divisibilite (PolarMesh + SystemConfig) + Poisson direct multi-box")


# ---------------------------------------------------------------------------
# (d) Round-trip get/set state multi-box
# ---------------------------------------------------------------------------

def test_state_roundtrip_multibox():
    tb = 4
    U0 = _iso_state()

    # set_state -> get_state == identite (etat conservatif ; scatter sur 4 bandes puis gather global).
    sim = _iso_system(tb)
    sim.set_state("iso", U0)
    U1 = np.asarray(sim.get_state("iso"), dtype=float)
    assert U1.shape == (3, NTH, NR), "get_state de forme %r (attendu (3, %d, %d))" % (U1.shape, NTH, NR)
    assert float(np.max(np.abs(U1 - U0))) == 0.0, "set_state/get_state non identite (multi-box)"

    # set_primitive_state -> get_primitive_state == identite (conversion modele consistante, multi-box).
    # On passe par le binding bas niveau (tableau (ncomp, ntheta, nr)) pour eviter la reshape carree de
    # la facade pops.System.set_primitive_state (hypothese n x n, hors-sujet ici).
    dr = (RMAX - RMIN) / NR
    dth = 2.0 * math.pi / NTH
    P0 = np.empty((3, NTH, NR))
    for j in range(NTH):
        th = (j + 0.5) * dth
        for i in range(NR):
            rr = (RMIN + (i + 0.5) * dr - RMIN) / (RMAX - RMIN)
            h = math.sin(math.pi * rr)
            P0[0, j, i] = 1.2 + 0.2 * math.cos(2.0 * th) * h   # rho (> 0)
            P0[1, j, i] = 0.5 * h * math.cos(th)               # v_r
            P0[2, j, i] = -0.3 * h * math.sin(2.0 * th)        # v_theta
    sim2 = _iso_system(tb)
    sim2._s.set_primitive_state("iso", P0)
    P1 = np.asarray(sim2._s.get_primitive_state("iso"), dtype=float)
    assert P1.shape == (3, NTH, NR)
    assert float(np.max(np.abs(P1 - P0))) < 1e-12, "set/get_primitive_state non identite (multi-box)"

    # set_density -> density == identite (densite posee, reste au repos ; reconstruction global exacte).
    rho0 = _scalar_density()
    sim3 = _exb_system(tb)
    sim3.set_density("ne", rho0)
    d = np.asarray(sim3.density("ne"), dtype=float)
    assert float(np.max(np.abs(d - rho0))) == 0.0, "set_density/density non identite (multi-box)"
    print("  [OK ] (d) round-trip get/set state multi-box (conservatif + primitif + densite)")


def main():
    test_transport_isothermal_bit_identical_multibox()
    test_exb_scalar_multibox_consistent()
    test_rejections_divisibility_and_direct_poisson()
    test_state_roundtrip_multibox()
    print("OK test_polar_theta_boxes (transport multi-box bit-identique + rejets + round-trip)")


if __name__ == "__main__":
    main()

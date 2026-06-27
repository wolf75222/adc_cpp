#!/usr/bin/env python3
"""Chantier T2 : masque de domaine DISQUE conservatif (CONTRAT, inerte par defaut).

Cote Python on valide le CONTRAT du scaffolding (l'API existe, est inerte par defaut, et le masque
materialise correspond bien au disque) :

  (a) BIT-IDENTITE / inertie : tant que set_disc_domain n'est PAS appele, disc_mask() est TOUT ACTIF
      (que des 1.0) et une advection step() est BYTE-IDENTIQUE a un run jumeau qui ne touche jamais
      l'API disque. Construire / interroger la machinerie disque (defaut) ne perturbe pas la
      trajectoire -- c'est l'invariant "inerte par defaut".

  (b) Le masque materialise par set_disc_domain est le DISQUE attendu : 0/1 cellule-centre coincidant
      AVEC le level set hypot(x-cx, y-cy) - R < 0 (meme convention que le mur conducteur du Poisson).
      Et les garde-fous du contrat sont actifs (R > 0, cartesien seulement).

ATTENTION : ce PR est un CONTRAT (geometrie option 2a). set_disc_domain CONSTRUIT le masque mais ne
BRANCHE PAS encore le transport mask-aware dans step() -> la trajectoire reste celle du chemin
cartesien plein. La conservation FV du sous-domaine actif est validee cote C++
(tests/test_disc_domain_mask.cpp, qui exerce assemble_rhs_masked).

Lance comme un simple script python3 (pas pytest : la CI lance ces tests directement). Se saute
proprement si le module _pops n'est pas importable (build absent).
"""

from pops.numerics.variables import Conservative
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import sys

import numpy as np

try:
    import pops
except ImportError as e:  # pragma: no cover - environnement sans build
    print("skip  module pops absent (PYTHONPATH ? build ?) : %s" % e)
    sys.exit(0)


fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def iso_model(cs2=1.0, alpha=3.0, n0=1.0):
    """Fluide isotherme NATIF (briques natives, aucun compilateur C++ : CI-safe, meme chemin que
    test_schur_conservation.py). Roles Density / MomentumX / MomentumY (3 var)."""
    return pops.Model(
        state=pops.FluidState(kind="isothermal", cs2=cs2),
        transport=pops.IsothermalFlux(),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=alpha, n0=n0),
    )


def ring(n, L, cx=0.5, cy=0.5):
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-(((X - cx * L) ** 2 + (Y - cy * L) ** 2) / (0.02 * L * L)))


# ---------------------------------------------------------------------------
# (a) inertie / bit-identite : masque tout actif par defaut, trajectoire inchangee
# ---------------------------------------------------------------------------

def _build(n, L):
    sim = pops.System(n=n, L=L, periodic=True)
    sim.set_poisson(bc="periodic")
    rho0 = ring(n, L)
    sim.add_equation("s", model=iso_model(n0=float(rho0.mean())),
                     spatial=pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov(),
                                              variables=Conservative()),
                     time=pops.Explicit())
    # Vitesse initiale CONSTANTE non nulle : le transport advecte la bosse (test non trivial).
    sim.set_primitive_state("s", rho=rho0, u=0.7 + 0.0 * rho0, v=-0.4 + 0.0 * rho0)
    return sim


def test_inert_default():
    n, L = 40, 1.0

    # disc_mask() est TOUT ACTIF tant que set_disc_domain n'est pas appele.
    sim = _build(n, L)
    mk = np.array(sim.disc_mask())
    chk(mk.shape == (n, n),
        "(a) disc_mask() a la forme (ny, nx) = (%d, %d) : recu %r" % (n, n, tuple(mk.shape)))
    chk(np.all(mk == 1.0),
        "(a) masque TOUT ACTIF par defaut (tous 1.0, sous-domaine = domaine entier) : min = %g, "
        "max = %g" % (float(mk.min()), float(mk.max())))

    # Run jumeau : l'un INTERROGE disc_mask() a chaque pas (machinerie disque, defaut tout-actif),
    # l'autre ne touche JAMAIS l'API disque. Trajectoires BYTE-IDENTIQUES.
    sim_a = _build(n, L)
    sim_b = _build(n, L)
    dt = 0.2 * (L / n) / np.hypot(0.7, 0.4)
    for _ in range(30):
        _ = sim_a.disc_mask()  # interroge la machinerie disque (defaut) a chaque pas
        sim_a.step(dt)
        sim_b.step(dt)
    ua = np.array(sim_a.density("s"))
    ub = np.array(sim_b.density("s"))
    max_diff = float(np.max(np.abs(ua - ub)))
    print("    [INERTIE] max|rho_avec_query - rho_sans_query| = %.3e (attendu 0)" % max_diff)
    chk(max_diff == 0.0,
        "(a) interroger la machinerie disque (defaut) ne change pas la trajectoire d'un bit "
        "(diff = 0) : invariant inerte par defaut")
    chk(np.max(np.abs(ua - 1.0)) > 1e-3,
        "(a) le transport a effectivement avance l'etat (test non trivial) : max dev = %.3e"
        % float(np.max(np.abs(ua - 1.0))))


# ---------------------------------------------------------------------------
# (b) le masque set_disc_domain est le DISQUE attendu (coincide avec le level set)
# ---------------------------------------------------------------------------

def test_disc_mask_matches_levelset():
    n, L = 48, 1.0
    cx, cy, R = 0.5, 0.5, 0.3
    sim = _build(n, L)
    sim.set_disc_domain(cx, cy, R)
    mk = np.array(sim.disc_mask())

    # Reference : level set hypot(x_cell - cx, y_cell - cy) - R < 0 au CENTRE des cellules.
    # disc_mask() est (ny, nx) row-major (j lent, i rapide) ; x = i rapide -> meshgrid indexing 'xy'.
    xc = (np.arange(n) + 0.5) * (L / n)
    yc = (np.arange(n) + 0.5) * (L / n)
    XI, YJ = np.meshgrid(xc, yc, indexing="xy")  # XI varie sur l'axe i (colonnes), YJ sur j (lignes)
    ls = np.hypot(XI - cx, YJ - cy) - R
    expected = (ls < 0.0).astype(np.float64)

    chk(mk.shape == (n, n), "(b) disc_mask() forme (ny, nx)")
    n_active = int(mk.sum())
    chk(0 < n_active < n * n,
        "(b) le disque partitionne la grille (actives ET inactives) : %d actives sur %d"
        % (n_active, n * n))
    chk(np.array_equal(mk, expected),
        "(b) masque == indicatrice du level set hypot(x-cx, y-cy) - R < 0 (meme convention que "
        "le mur conducteur) : %d cellules en desaccord"
        % int(np.sum(mk != expected)))
    chk(set(np.unique(mk).tolist()) <= {0.0, 1.0},
        "(b) masque strictement 0/1 (valeurs uniques %r)" % np.unique(mk).tolist())


# ---------------------------------------------------------------------------
# (c) garde-fous du contrat : R > 0, cartesien seulement
# ---------------------------------------------------------------------------

def test_guards():
    n, L = 32, 1.0
    sim = _build(n, L)
    raised = False
    try:
        sim.set_disc_domain(0.5, 0.5, 0.0)  # R <= 0 doit lever
    except Exception:
        raised = True
    chk(raised, "(c) set_disc_domain(R=0) leve (rayon R > 0 requis)")

    # Polaire : l'anneau est deja borne par ses parois radiales -> set_disc_domain doit lever.
    raised_polar = False
    try:
        simp = pops.System(mesh=pops.PolarMesh(nr=16, ntheta=16, r_min=0.2, r_max=1.0))
        simp.set_disc_domain(0.0, 0.0, 0.5)
    except Exception:
        raised_polar = True
    chk(raised_polar,
        "(c) set_disc_domain leve en geometrie polaire (anneau deja borne par ses parois radiales)")


def main():
    print("(a) inertie / bit-identite (masque tout actif par defaut)")
    test_inert_default()
    print("(b) masque set_disc_domain == indicatrice du level set disque")
    test_disc_mask_matches_levelset()
    print("(c) garde-fous du contrat (R > 0, cartesien seulement)")
    test_guards()

    if fails == 0:
        print("test_disc_domain_mask : tout est vert")
    else:
        raise SystemExit("test_disc_domain_mask : %d verification(s) en echec" % fails)


if __name__ == "__main__":
    main()

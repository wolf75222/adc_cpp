"""Critere d'arret MIXTE rel/abs du V-cycle GeometricMG, expose par set_poisson(abs_tol=...).

NE TOURNE QU'EN CI. Le binding set_poisson(abs_tol=...) est un parametre NOUVEAU, absent du module
_pops deja construit ; l'exercer exige une RECONSTRUCTION de l'extension. Le coeur C++ du correctif
(early-exit + plancher mixte) est valide LOCALEMENT par le harnais python/tests/gpu/mg_cold_tolerance.cpp
(compile contre les headers, avant/apres patch, ligne BASE bit-identique). Ce test verrouille le
BINDING bout en bout cote CI.

Couvre :
  (a) abs_tol > 0 : un solve_fields() repete sur ETAT INCHANGE sort sans cycler (early-exit) ->
      potential() BIT-IDENTIQUE d'un appel a l'autre (phi n'est pas re-touche) ;
  (b) NO-DEFAULT-CHANGE : abs_tol par defaut (omis) == abs_tol=0.0 explicite -> potential() identique
      (le parametre a defaut 0 est un vrai no-op, critere relatif historique) ;
  (c) garde-fou : abs_tol < 0 rejete.
"""
import numpy as np

import pops


def build(n=64, abs_tol=None):
    """System cartesien Dirichlet, un bloc de charge (ExB + fond), densite gaussienne -> phi non trivial.
    abs_tol=None : set_poisson par defaut (parametre omis) ; sinon passe abs_tol explicitement."""
    sim = pops.System(n=n, periodic=False)
    sim.add_block(
        "ne",
        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                        source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    if abs_tol is None:
        sim.set_poisson(bc="dirichlet")
    else:
        sim.set_poisson(bc="dirichlet", abs_tol=abs_tol)
    xs = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    dens = np.exp(-(X * X + Y * Y) / 0.02)
    sim.set_density("ne", dens)
    return sim


def check_early_exit():
    """(a) Plancher LARGE (1e-3) : tout etat converge a un residu bien sous le plancher -> chaque
    solve_fields() repete sur etat inchange early-exit (aucun V-cycle), donc phi reste BIT-IDENTIQUE."""
    sim = build(abs_tol=1e-3)
    sim.solve_fields()
    p1 = np.array(sim.potential(), copy=True)
    sim.solve_fields()  # etat inchange -> early-exit, phi intouche
    p2 = np.array(sim.potential(), copy=True)
    sim.solve_fields()
    p3 = np.array(sim.potential(), copy=True)
    assert np.all(np.isfinite(p1)), "phi non fini apres le 1er solve_fields"
    assert np.array_equal(p1, p2), "abs_tol>0 : le 2e solve_fields a modifie phi (early-exit absent)"
    assert np.array_equal(p1, p3), "abs_tol>0 : le 3e solve_fields a modifie phi (early-exit absent)"
    print("[OK] early-exit : solve_fields repete bit-identique sous abs_tol=1e-3")


def check_default_noop():
    """(b) abs_tol par defaut (omis) == abs_tol=0.0 explicite : potential() bit-identique apres un solve
    (le defaut 0 ne change RIEN au critere relatif historique)."""
    sim_def = build(abs_tol=None)
    sim_zero = build(abs_tol=0.0)
    sim_def.solve_fields()
    sim_zero.solve_fields()
    a = np.array(sim_def.potential())
    b = np.array(sim_zero.potential())
    assert np.array_equal(a, b), "abs_tol=0.0 explicite differe du defaut (pas un no-op)"
    print("[OK] no-default-change : abs_tol omis == abs_tol=0.0")


def check_reject_negative():
    """(c) abs_tol < 0 rejete a la configuration."""
    raised = False
    try:
        build(abs_tol=-1.0)
    except Exception:
        raised = True
    assert raised, "abs_tol < 0 aurait du etre rejete par set_poisson"
    print("[OK] abs_tol < 0 rejete")


def main():
    check_early_exit()
    check_default_noop()
    check_reject_negative()
    print("[OK] test_mg_cold_tolerance")


if __name__ == "__main__":
    main()

"""Facade Python AmrSystem MULTI-BLOCS (capstone PR1, docs/AMR_MULTIBLOCK_DESIGN.md).

Deux blocs EXPLICITES a SCHEMAS SPATIAUX DIFFERENTS (none/rusanov vs minmod/rusanov), co-localises
sur UNE hierarchie AMR PARTAGEE, avec un Poisson de SYSTEME a second membre SOMME co-localise
(Sum_b q_b n_b). On verifie cote Python :
  (a) les DEUX blocs sont enregistres (n_blocks == 2) et evoluent (densite changee par le transport) ;
  (b) la MASSE de CHAQUE bloc est conservee a ~machine (reflux + average_down, PAR BLOC) ;
  (c) le potentiel de systeme est non trivial (le Poisson somme co-localise produit un phi) ;
  (d) le chemin MONO-BLOC reste deterministe / bit-identique (run x2 -> dmax == 0, chemin AmrCouplerMP) ;
  (e) multi-blocs + regrid_every > 0 est ACCEPTE (deverrouillage Phase 2, C.6 : regrid d'union des tags).

Test PUR Python (aucune compilation .so) : ne gate sur rien, toujours execute.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import numpy as np

import pops


def _bump(n, amp):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    r = 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    return r + (1.0 - r.mean())  # offset moyen nul -> Sum q n solvable en periodique


def _scalar_charge(q, B0=1.0):
    return pops.Model(pops.Scalar(), pops.ExB(B0=B0), pops.NoSource(), pops.ChargeDensity(charge=q))


def _build(n=32, regrid_every=0):
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=regrid_every)
    sim.add_block("ions", _scalar_charge(+1.0),
                  spatial=pops.Spatial(limiter=FirstOrder(), flux=Rusanov()))
    sim.add_block("electrons", _scalar_charge(-1.0),
                  spatial=pops.Spatial(limiter=Minmod(), flux=Rusanov()))  # SCHEMA DIFFERENT
    sim.set_poisson(bc="periodic")
    sim.set_density("ions", _bump(n, 0.40))
    sim.set_density("electrons", _bump(n, 0.20))
    return sim


def main():
    n = 32

    # (a)(b)(c) deux blocs, schemas differents, hierarchie partagee, Poisson somme co-localise.
    sim = _build(n=n, regrid_every=0)
    assert sim.n_blocks() == 2, "n_blocks != 2"

    d0i = np.asarray(sim.density("ions"))
    d0e = np.asarray(sim.density("electrons"))
    m0i, m0e = sim.mass("ions"), sim.mass("electrons")

    sim.advance(0.001, 10)

    d1i = np.asarray(sim.density("ions"))
    d1e = np.asarray(sim.density("electrons"))
    m1i, m1e = sim.mass("ions"), sim.mass("electrons")
    phi = np.asarray(sim.potential())

    # (a) les DEUX blocs ont evolue (transport E x B a deplace la densite).
    assert float(np.abs(d1i - d0i).max()) > 1e-6, "bloc ions non avance"
    assert float(np.abs(d1e - d0e).max()) > 1e-6, "bloc electrons non avance"
    # (b) masse de CHAQUE bloc conservee a ~machine (par bloc, reflux + average_down).
    assert abs(m1i - m0i) < 1e-9, "masse ions non conservee (dm=%.2e)" % abs(m1i - m0i)
    assert abs(m1e - m0e) < 1e-9, "masse electrons non conservee (dm=%.2e)" % abs(m1e - m0e)
    # (c) potentiel de systeme non trivial (Poisson somme co-localise q0 n0 + q1 n1).
    assert float(np.abs(phi).max()) > 1e-8, "potentiel trivial (Poisson somme inactif)"
    print("OK  AmrSystem multi-blocs : 2 blocs schemas differents, masse par bloc conservee, "
          "Poisson somme co-localise")

    # (d) MONO-BLOC deterministe (chemin AmrCouplerMP intouche) : run x2 -> dmax == 0.
    def run_mono():
        s = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
        s.add_block("ne", _scalar_charge(+1.0), spatial=pops.Spatial(limiter=FirstOrder(), flux=Rusanov()))
        s.set_poisson(bc="periodic")
        s.set_density("ne", _bump(n, 0.40))
        s.advance(0.001, 10)
        return np.asarray(s.density())

    a = run_mono()
    b = run_mono()
    assert float(np.abs(a - b).max()) == 0.0, "mono-bloc non bit-identique (dmax != 0)"
    print("OK  mono-bloc bit-identique (dmax == 0, chemin AmrCouplerMP intouche)")

    # (e) multi-blocs + regrid_every > 0 ACCEPTE (deverrouillage Phase 2, C.6 : regrid d'union des tags).
    # L'ancien refus (hierarchie figee) est leve : la grille re-grille a partir de l'union des tags de
    # tous les blocs. On verifie que le build paresseux + l'avance NE LEVENT PLUS et que la hierarchie
    # reste valide (au moins un patch fin). Le mouvement effectif de la grille est verrouille en C++
    # (test_amr_multiblock_regrid_union) ; ici on assure la non-regression de la facade Python.
    s = _build(n=n, regrid_every=2)  # regrid_every > 0 en multi-blocs : DESORMAIS supporte
    s.set_refinement(1.05)  # seuil bas -> l'union des tags des deux bumps raffine effectivement
    s.advance(0.001, 6)     # declenche le build paresseux + plusieurs regrids
    assert s.n_patches() >= 1, "hierarchie sans patch fin apres regrid d'union"
    assert np.isfinite(np.asarray(s.density("ions"))).all(), "etat ions non fini apres regrid"
    print("OK  multi-blocs + regrid_every > 0 accepte (regrid d'union des tags, deverrouillage Phase 2)")

    print("OK test_amr_multiblock")


if __name__ == "__main__":
    main()

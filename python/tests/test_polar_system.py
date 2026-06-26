"""Phase 2b grille polaire : chemin POLAIRE complet a travers pops.System(mesh=pops.PolarMesh(...)).

Valide le LIVRABLE Python : un System polaire se construit (anneau global r x theta), recoit un bloc
ExB scalaire + un Poisson polaire, une densite annulaire, puis avance par step() ET step_cfl() en
conservant la masse FV polaire (Sum n r dr dtheta) a ~machine. C'est le pendant Python (chemin reel
System::step branche) du test C++ test_polar_system_step (qui exerce la meme numerique sans Python).

Verifications :
  (1) construction + step()/step_cfl() ne levent pas et restent FINIS (pas de nan : c'etait le bug du
      gradient de phi lu hors domaine -> blow-up) ;
  (2) la densite EVOLUE reellement (champ non gele : Poisson + aux + transport actifs) ;
  (3) la masse FV polaire est conservee a ~machine sur K pas (paroi radiale solide, wall_radial).

Mono-rang (le Poisson polaire direct refuse MPI) : ce test n'est pas un test MPI.
"""
import math

import numpy as np

import pops


def _annular_density(nr, nth, rmin, rmax):
    # Layout attendu par System::set_density (polaire) : axe lent = theta (j), axe rapide = r (i),
    # i.e. flat[j * nr + i]. Profil annulaire lisse, strictement positif, module en theta (asymetrie
    # -> Poisson non trivial -> grad_theta != 0 -> vitesse radiale non nulle a l'interieur).
    dr = (rmax - rmin) / nr
    dth = 2.0 * math.pi / nth
    rho = []
    for j in range(nth):
        th = (j + 0.5) * dth
        for i in range(nr):
            r = rmin + (i + 0.5) * dr
            rr = (r - rmin) / (rmax - rmin)
            rho.append(1.0 + 0.3 * math.cos(2.0 * th) * math.sin(math.pi * rr))
    return rho


def _flat(field):
    return np.asarray(field, dtype=float).ravel()


def test_polar_system_step_and_cfl_conserve_mass():
    rmin, rmax, nr, nth = 0.30, 1.00, 48, 48
    sim = pops.System(mesh=pops.PolarMesh(r_min=rmin, r_max=rmax, nr=nr, ntheta=nth))
    sim.add_block(
        "ne",
        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                        source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0)),
        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    sim.set_density("ne", _annular_density(nr, nth, rmin, rmax))

    m0 = sim.mass("ne")
    rho0 = _flat(sim.density("ne"))
    assert math.isfinite(m0) and m0 > 0.0
    assert np.all(np.isfinite(rho0))

    # (A) step_cfl : exerce le pas physique min polaire min(dr, r_min*dtheta) + solve_fields_polar.
    dt = sim.step_cfl(0.3)
    assert math.isfinite(dt) and dt > 0.0
    rho1 = _flat(sim.density("ne"))
    assert np.all(np.isfinite(rho1)), "densite non finie apres step_cfl (blow-up : gradient phi hors domaine ?)"
    dmax = float(np.max(np.abs(rho1 - rho0)))
    assert dmax > 1e-9, "le pas couple ne modifie pas la densite (Poisson/aux/transport inertes ?)"
    assert dmax < 1.0, "variation > 1.0 apres 1 pas = instabilite (densite initiale ~1)"

    # (B) step(dt) explicite + suite de step_cfl : la masse FV polaire reste conservee a ~machine.
    for _ in range(15):
        sim.step(dt)
    for _ in range(15):
        sim.step_cfl(0.3)

    m1 = sim.mass("ne")
    rho2 = _flat(sim.density("ne"))
    assert np.all(np.isfinite(rho2)) and math.isfinite(m1), "masse/densite non finie apres K pas (blow-up)"
    assert float(np.min(rho2)) > 0.0, "densite devenue negative (pas couple instable)"
    rel = abs(m1 - m0) / abs(m0)
    assert rel < 1e-11, "masse FV polaire non conservee : ecart relatif %.3e" % rel


def test_polar_rejects_nr_below_3():
    # nr < 3 lirait phi hors bornes (stencil radial decentre ordre 2, phi sans ghost) : doit etre
    # REFUSE explicitement, aux DEUX niveaux. (1) PolarMesh leve a la construction du maillage.
    raised = False
    try:
        pops.PolarMesh(r_min=0.3, r_max=1.0, nr=2, ntheta=8)
    except ValueError:
        raised = True
    assert raised, "PolarMesh(nr=2) doit lever (nr >= 3 requis)"

    # (2) Un appelant qui construit le SystemConfig a la main (contourne PolarMesh) doit aussi etre
    # protege par check_geometry cote C++.
    cfg = pops.SystemConfig()
    cfg.geometry = "polar"
    cfg.r_min, cfg.r_max, cfg.nr, cfg.ntheta = 0.3, 1.0, 2, 8
    raised = False
    try:
        pops.System(config=cfg)
    except Exception:
        raised = True
    assert raised, "System(geometry='polar', nr=2) doit lever cote C++ (check_geometry)"


if __name__ == "__main__":
    test_polar_system_step_and_cfl_conserve_mass()
    test_polar_rejects_nr_below_3()
    print("OK test_polar_system")

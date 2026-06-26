"""Regression SCIENTIFIQUE du chemin polaire (Polaire 2b) : l'instabilite diocotron est CAPTUREE.

Motivation de Polaire 2b : la grille cartesienne diffuse le gradient RADIAL d'un anneau de charge et
ETOUFFE l'instabilite diocotron ; la grille polaire (axe radial aligne sur le gradient) la preserve.
Ce test verrouille cette propriete : sur un anneau de charge CREUX (bord interieur libre -> diocotron
instable) avec un mode azimutal l seme, l'amplitude du mode l doit CROITRE de facon monotone (croissance
exponentielle de l'instabilite), tout en conservant la masse FV polaire a ~machine et en restant fini.

Config rapide (48x48, ~200 pas, < 2 s ; WENO5/SSPRK3) calibree pour une croissance nette et robuste
(observe x4.4). La mesure QUANTITATIVE du taux (l=3/4/5, O5, n=384/512 vs theorie) est le run ROMEO ;
ici on verrouille seulement le SIGNE de l'effet (le polaire capture l'instabilite, pas du bruit).
Mono-rang (le Poisson polaire direct refuse MPI) : ce n'est pas un test MPI.
"""
import math

import numpy as np

import pops

RMIN, RMAX, NR, NTH = 0.30, 1.00, 48, 48
L_MODE = 4
EPS = 0.02
RA, RB = 0.50, 0.80          # anneau de charge creux : bord interieur libre a RA > RMIN
NSTEPS = 200


def _hollow_ring_density():
    # Profil radial lisse (sin^2) dans [RA, RB], nul ailleurs, module par eps*cos(L theta).
    # Layout attendu par set_density (polaire) : flat[j*NR + i] (theta lent, r rapide).
    dr = (RMAX - RMIN) / NR
    dth = 2.0 * math.pi / NTH
    rho = []
    for j in range(NTH):
        pert = 1.0 + EPS * math.cos(L_MODE * (j + 0.5) * dth)
        for i in range(NR):
            r = RMIN + (i + 0.5) * dr
            base = math.sin(math.pi * (r - RA) / (RB - RA)) ** 2 if RA <= r <= RB else 0.0
            rho.append(max(1e-6, base * pert))  # plancher strictement positif (ExB scalaire)
    return rho


def _mode_amplitude(sim, l):
    a = np.asarray(sim.density("ne"), float).ravel().reshape(NTH, NR)  # [theta, r]
    return float(np.sum(np.abs(np.fft.fft(a, axis=0)[l, :])))          # |mode l|, integre en r


def test_polar_diocotron_mode_grows_and_conserves():
    sim = pops.System(mesh=pops.PolarMesh(r_min=RMIN, r_max=RMAX, nr=NR, ntheta=NTH))
    sim.add_block(
        "ne",
        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                        source=pops.NoSource(), elliptic=pops.ChargeDensity(charge=1.0)),
        spatial=pops.Spatial(weno5=True), time=pops.Explicit(method="ssprk3"))
    sim.set_poisson(rhs="charge_density", solver="polar", bc="dirichlet")
    sim.set_density("ne", _hollow_ring_density())

    m0 = sim.mass("ne")
    a0 = _mode_amplitude(sim, L_MODE)
    assert math.isfinite(m0) and m0 > 0.0 and a0 > 0.0
    # Le mode seme l=4 doit DOMINER a t=0 (sanity layout/FFT) : les autres modes sont ~0.
    others = max(_mode_amplitude(sim, l) for l in (1, 2, 3, 5, 6))
    assert a0 > 50.0 * (others + 1e-30), "le mode seme l=%d ne domine pas a t=0" % L_MODE

    dt = sim.step_cfl(0.3)
    assert math.isfinite(dt) and dt > 0.0
    samples = [a0]
    for s in range(1, NSTEPS + 1):
        sim.step(dt)
        if s % 25 == 0:
            samples.append(_mode_amplitude(sim, L_MODE))

    m1 = sim.mass("ne")
    rho = np.asarray(sim.density("ne"), float)
    assert np.all(np.isfinite(rho)), "densite non finie (blow-up)"
    assert float(np.min(rho)) > -1e-9, "densite nettement negative (instable)"
    assert abs(m1 - m0) / abs(m0) < 1e-9, "masse FV polaire non conservee"

    # Croissance MONOTONE (tolerance bruit) du mode l : signature de l'instabilite, pas de la diffusion.
    for i in range(len(samples) - 1):
        assert samples[i + 1] >= samples[i] * 0.99, "amplitude mode l non monotone (echantillon %d)" % i
    growth = samples[-1] / a0
    assert growth > 2.0, "mode l=%d insuffisamment amplifie (x%.2f) : instabilite non capturee" % (L_MODE, growth)


if __name__ == "__main__":
    test_polar_diocotron_mode_grows_and_conserves()
    print("OK test_polar_diocotron")

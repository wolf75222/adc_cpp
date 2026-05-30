#!/usr/bin/env python3
"""Tutoriel exécutable : diocotron en anneau (couplage hyperbolique-elliptique en image).

Une couche de charge annulaire au repos est instable : la dérive E x B la déforme en
`l` lobes (ici `ring_mode = 3`) qui s'enroulent. L'animation montre les DEUX champs côte
à côte : la densité n_e (transportée, hyperbolique) et le potentiel phi (résolu par
Poisson à chaque pas, elliptique). C'est le couplage rendu visible : phi suit n, et n
dérive selon grad phi.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/diocotron_ring.py [n] [nsteps]
Sortie : docs/tut_diocotron_ring.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import adc

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 480

    cfg = adc.DiocotronConfig()
    cfg.n = n
    cfg.ic = adc.DiocotronIC.Ring
    cfg.n_i0 = 0.0          # pas de fond ionique pour l'anneau
    cfg.ring_r0, cfg.ring_r1 = 0.18, 0.26
    cfg.ring_mode = 3       # 3 lobes azimutaux
    cfg.ring_delta = 0.12
    sim = adc.DiocotronSolver(cfg)

    m0 = sim.mass()
    dens, phis, snap = [], [], max(1, nsteps // 50)
    for s in range(nsteps + 1):
        if s % snap == 0:
            dens.append(sim.density().copy())
            phis.append(sim.potential().copy())
        if s < nsteps:
            sim.step_cfl(0.4)
    rel = abs(sim.mass() - m0) / m0
    print(f"diocotron anneau n={n}, {nsteps} pas : {len(dens)} frames, "
          f"dérive masse relative = {rel:.2e}")

    nvmax = max(f.max() for f in dens)
    pmax = max(np.abs(p).max() for p in phis)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 4.2), constrained_layout=True)
    fig.patch.set_alpha(0)

    def update(k):
        for a in (a1, a2):
            a.clear(); a.set_xticks([]); a.set_yticks([])
        a1.imshow(dens[k], origin="lower", cmap="inferno", vmin=0, vmax=nvmax,
                  extent=[0, 1, 0, 1])
        a1.set_title("densité $n_e$ (transport E x B)")
        a2.imshow(phis[k], origin="lower", cmap="RdBu_r", vmin=-pmax, vmax=pmax,
                  extent=[0, 1, 0, 1])
        a2.set_title("potentiel $\\phi$ (Poisson)")
        fig.suptitle(f"diocotron en anneau, pas {k * snap}")
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(dens), blit=False)
    out = ROOT / "docs" / "tut_diocotron_ring.gif"
    anim.save(out, writer=animation.PillowWriter(fps=12))
    print(f"écrit {out}  ({len(dens)} frames)")
    # Le fin anneau s'enroule en filaments sous la grille : le limiteur MUSCL aux bords
    # raides fait deriver la masse de ~1e-4 (relatif) sur tout le run nonlineaire. La
    # bande lisse (diocotron.py), elle, conserve a l'arrondi.
    assert rel < 5e-4, "la masse doit rester conservee a ~1e-4 pres"
    return 0


if __name__ == "__main__":
    sys.exit(main())

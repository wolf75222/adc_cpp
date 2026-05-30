#!/usr/bin/env python3
"""Tutoriel exécutable : instabilité diocotron pilotée depuis les bindings Python.

Construit un DiocotronSolver (condition initiale en bande perturbée), l'avance au
CFL de la dérive E x B, capture la densité à intervalles réguliers et en fait une
animation GIF. Démontre que tout le solveur couplé (transport E x B + Poisson) se
pilote en quelques lignes de Python, et que la masse est conservée.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/diocotron.py [n] [nsteps]
Sortie : docs/tut_diocotron_py.gif
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
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    cfg = adc.DiocotronConfig()
    cfg.n = n
    cfg.ic = adc.DiocotronIC.Band     # bande de charge perturbée
    cfg.band_mode = 4                 # 4 lobes azimutaux
    cfg.band_amp = 0.5
    sim = adc.DiocotronSolver(cfg)

    m0 = sim.mass()
    frames, snap = [], max(1, nsteps // 40)
    for s in range(nsteps + 1):
        if s % snap == 0:
            frames.append(sim.density().copy())
        if s < nsteps:
            sim.step_cfl(0.4)
    drift = abs(sim.mass() - m0)
    print(f"diocotron Python n={n}, {nsteps} pas : {len(frames)} frames, "
          f"dérive masse = {drift:.2e}")

    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)
    fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
    fig.patch.set_alpha(0)

    def update(k):
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"diocotron (bindings Python) : densité n_e, pas {k * snap}")
        ax.imshow(frames[k], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax,
                  extent=[0, 1, 0, 1], aspect="equal")
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = ROOT / "docs" / "tut_diocotron_py.gif"
    anim.save(out, writer=animation.PillowWriter(fps=12))
    print(f"écrit {out}  ({len(frames)} frames)")
    assert drift < 1e-9, "la masse doit être conservée"
    return 0


if __name__ == "__main__":
    sys.exit(main())

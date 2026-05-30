#!/usr/bin/env python3
"""Tutoriel exécutable : la vie d'une instabilité diocotron (planche temporelle).

Un seul run de bande perturbée (mode 2), photographié à six instants : la perturbation
linéaire enfle, la bande s'enroule, deux vortex se forment puis fusionnent. Une figure
statique (idéale pour un rapport, là où le GIF ne s'insère pas), qui raconte le passage du
régime linéaire au régime non linéaire.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/diocotron_sequence.py [n]
Sortie : docs/tut_diocotron_sequence.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import adc

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    shots = [40, 120, 220, 320, 420, 520]   # pas où l'on photographie

    cfg = adc.DiocotronConfig()
    cfg.n = n
    cfg.ic = adc.DiocotronIC.Band
    cfg.band_mode = 2
    cfg.band_amp = 0.5
    sim = adc.DiocotronSolver(cfg)

    m0 = sim.mass()
    frames = {}
    for s in range(max(shots) + 1):
        if s in shots:
            frames[s] = sim.density().copy()
        sim.step_cfl(0.4)
    drift = abs(sim.mass() - m0)
    vmax = max(f.max() for f in frames.values())
    print(f"séquence diocotron n={n} : {len(shots)} instantanés, "
          f"dérive masse = {drift:.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4), constrained_layout=True)
    fig.patch.set_alpha(0)
    for ax, s in zip(axes.ravel(), shots):
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(frames[s], origin="lower", cmap="inferno", vmin=0, vmax=vmax,
                  extent=[0, 1, 0, 1])
        ax.set_title(f"pas {s}")
    fig.suptitle("Instabilité diocotron : du régime linéaire aux vortex non linéaires",
                 fontsize=13)
    out = ROOT / "docs" / "tut_diocotron_sequence.png"
    fig.savefig(out, dpi=120)
    print(f"écrit {out}")
    assert drift < 1e-9, "la masse doit être conservée (bande lisse)"
    return 0


if __name__ == "__main__":
    sys.exit(main())

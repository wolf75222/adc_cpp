#!/usr/bin/env python3
"""Tutoriel exécutable : multigrille vs FFT, le même Laplacien par deux chemins.

Le couplage Euler-Poisson peut résoudre Poisson par multigrille géométrique (itératif,
tout n, GPU-ready) OU par FFT spectrale (directe, n puissance de 2). Les deux inversent le
MÊME Laplacien 5 points : on lance la même simulation avec chaque backend et on montre les
deux champs de densité plus leur écart, au niveau de l'arrondi. Une figure, trois vignettes.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/poisson_backends.py [n] [nsteps]
Sortie : docs/tut_poisson_backends.png
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


def run(use_fft: bool, n: int, nsteps: int):
    cfg = adc.EulerPoissonConfig()
    cfg.n, cfg.use_fft, cfg.eps = n, use_fft, 5e-2
    cfg.four_pi_G = 60.0          # champ marqué : la densité développe de la structure
    s = adc.EulerPoissonSolver(cfg)
    for _ in range(nsteps):
        s.step(2e-3)
    return np.asarray(s.density())


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128   # puissance de 2 pour le FFT
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    mg = run(False, n, nsteps)
    fft = run(True, n, nsteps)
    diff = np.abs(mg - fft)
    rel = diff.max() / np.abs(mg).max()
    print(f"MG vs FFT (n={n}, {nsteps} pas) : max|MG - FFT| = {diff.max():.2e} "
          f"(relatif {rel:.2e})")

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), constrained_layout=True)
    fig.patch.set_alpha(0)
    vmin, vmax = mg.min(), mg.max()
    for ax, field, title in ((axes[0], mg, "multigrille"), (axes[1], fft, "FFT")):
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(field, origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
                       extent=[0, 1, 0, 1])
        ax.set_title(f"densité ({title})")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    dmax = max(diff.max(), 1e-16)
    imd = axes[2].imshow(diff, origin="lower", cmap="cividis", extent=[0, 1, 0, 1])
    axes[2].set_title(f"|MG - FFT| (max {diff.max():.1e})")
    fig.colorbar(imd, ax=axes[2], fraction=0.046)
    out = ROOT / "docs" / "tut_poisson_backends.png"
    fig.savefig(out, dpi=130)
    print(f"écrit {out}")

    assert rel < 1e-3, "les deux backends inversent le même Laplacien : ils doivent coïncider"
    return 0


if __name__ == "__main__":
    sys.exit(main())

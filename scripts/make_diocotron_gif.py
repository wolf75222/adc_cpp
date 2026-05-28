"""GIF anime de l'enroulement (cat's eyes) de l'instabilite diocotron.

Lit les instantanes dens_XXXX.txt (grilles n x n ecrites par
examples/diocotron.cpp) et produit un GIF de la densite n_e(x, y, t).

Usage :
  ./build/bin/diocotron /tmp/dio out 256 700
  python scripts/make_diocotron_gif.py /tmp/dio docs/anim_diocotron.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    gif = Path(sys.argv[2])
    gif.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(out_dir.glob("dens_*.txt"))
    if not frames:
        print(f"aucun instantane dans {out_dir}", file=sys.stderr)
        return 1
    grids = [np.loadtxt(f) for f in frames]
    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
    fig.patch.set_alpha(0)
    im = ax.imshow(grids[0], origin="lower", extent=[0, 1, 0, 1],
                   vmin=vmin, vmax=vmax, cmap="inferno", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"diocotron : densite $n_e$")

    def update(k):
        im.set_data(grids[k])
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(grids), blit=True)
    anim.save(gif, writer=animation.PillowWriter(fps=10))
    print(f"wrote {gif}  ({len(grids)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

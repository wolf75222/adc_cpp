"""GIF anime du diocotron sur AMR 2 niveaux MULTI-PATCH (Berger-Rigoutsos dynamique).

Lit les instantanes ecrits par examples/diocotron_multipatch.cpp :
  dens_XXXX.txt  : densite niveau 0 (grossier, nc x nc)
  boxes_XXXX.txt : extents physiques [xlo ylo xhi yhi] des patchs fins (un par ligne,
                   nombre variable d'une frame a l'autre car re-cluster par BR)
et trace la densite grossiere avec, par-dessus, les cadres des patchs fins (jaune)
re-decoupes a la volee autour des zones de fort gradient.

Usage :
  ./build/bin/diocotron_multipatch /tmp/dmp 96 400
  python scripts/make_diocotron_multipatch_gif.py /tmp/dmp docs/anim_diocotron_multipatch.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    out_dir = Path(sys.argv[1])
    gif = Path(sys.argv[2])
    gif.parent.mkdir(parents=True, exist_ok=True)

    dframes = sorted(out_dir.glob("dens_*.txt"))
    if not dframes:
        print(f"aucun instantane dans {out_dir}", file=sys.stderr)
        return 1
    coarse = [np.loadtxt(f) for f in dframes]

    # boxes_XXXX.txt : nombre de patchs variable -> on lit chaque frame separement.
    boxes = []
    for k in range(len(dframes)):
        bf = out_dir / f"boxes_{k:04d}.txt"
        rows = np.loadtxt(bf, ndmin=2) if bf.exists() and bf.stat().st_size else np.empty((0, 4))
        boxes.append(rows)

    vmin = min(g.min() for g in coarse)
    vmax = max(g.max() for g in coarse)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
    fig.patch.set_alpha(0)
    kw = dict(origin="lower", vmin=vmin, vmax=vmax, cmap="inferno", aspect="equal",
              extent=[0, 1, 0, 1])

    def update(k):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        npatch = len(boxes[k])
        ax.set_title(f"diocotron AMR multi-patch : {npatch} patchs (Berger-Rigoutsos)")
        ax.imshow(coarse[k], zorder=0, **kw)
        for xlo, ylo, xhi, yhi in boxes[k]:
            ax.add_patch(patches.Rectangle(
                (xlo, ylo), xhi - xlo, yhi - ylo, fill=False,
                edgecolor="yellow", lw=1.1, zorder=3))
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(coarse), blit=False)
    anim.save(gif, writer=animation.PillowWriter(fps=10))
    print(f"wrote {gif}  ({len(coarse)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

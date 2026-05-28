"""GIF anime du diocotron sur AMR a 3 niveaux emboites (montage composite).

Lit les instantanes ecrits par examples/diocotron_amr3.cpp :
  dens_c_XXXX.txt : densite niveau 0 (grossier, nc x nc, fond)
  dens_1_XXXX.txt : densite niveau 1 (raffine x2), placee a son extent physique
  dens_2_XXXX.txt : densite niveau 2 (raffine x4), idem
  boxes.csv       : extents physiques [xlo,xhi,ylo,yhi] des niveaux 1 et 2 par frame
et compose les trois resolutions dans une seule vue (zorder croissant), avec les
cadres imbriques (niveau 1 cyan, niveau 2 lime).

Usage :
  ./build/bin/diocotron_amr3 /tmp/dio3 128 500
  python scripts/make_diocotron_amr3_gif.py /tmp/dio3 docs/anim_diocotron_amr3.gif
"""

from __future__ import annotations

import csv
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

    cframes = sorted(out_dir.glob("dens_c_*.txt"))
    if not cframes:
        print(f"aucun instantane dans {out_dir}", file=sys.stderr)
        return 1
    coarse = [np.loadtxt(f) for f in cframes]
    fine1 = [np.loadtxt(out_dir / f"dens_1_{k:04d}.txt") for k in range(len(cframes))]
    fine2 = [np.loadtxt(out_dir / f"dens_2_{k:04d}.txt") for k in range(len(cframes))]

    ext = {}
    with open(out_dir / "boxes.csv") as fh:
        for row in csv.DictReader(fh):
            k = int(row["frame"])
            ext[k] = (
                [float(row["x1lo"]), float(row["x1hi"]), float(row["y1lo"]), float(row["y1hi"])],
                [float(row["x2lo"]), float(row["x2hi"]), float(row["y2lo"]), float(row["y2hi"])],
            )

    vmin = min(g.min() for g in coarse)
    vmax = max(g.max() for g in coarse)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
    fig.patch.set_alpha(0)
    kw = dict(origin="lower", vmin=vmin, vmax=vmax, cmap="inferno", aspect="equal")

    def update(k):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r"diocotron AMR 3 niveaux : densite $n_e$")
        ax.imshow(coarse[k], extent=[0, 1, 0, 1], zorder=0, **kw)
        e1, e2 = ext[k]
        ax.imshow(fine1[k], extent=e1, zorder=1, **kw)
        ax.imshow(fine2[k], extent=e2, zorder=2, **kw)
        ax.add_patch(patches.Rectangle(
            (e1[0], e1[2]), e1[1] - e1[0], e1[3] - e1[2], fill=False,
            edgecolor="cyan", lw=1.1, zorder=3))
        ax.add_patch(patches.Rectangle(
            (e2[0], e2[2]), e2[1] - e2[0], e2[3] - e2[2], fill=False,
            edgecolor="lime", lw=1.1, zorder=4))
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(coarse), blit=False)
    anim.save(gif, writer=animation.PillowWriter(fps=10))
    print(f"wrote {gif}  ({len(coarse)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

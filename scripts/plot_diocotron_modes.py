"""Montage des modes diocotron l = 3, 4, 5 (l tourbillons en anneau).

Compare visuellement aux figures 5.1, 5.2, 5.3 de Hoffart-Maier-Shadid-Tomas
(arXiv:2510.11808). Prend un instantane (par defaut a mi-parcours) de chaque
repertoire de sortie de examples/diocotron_column.cpp.

Usage :
  for L in 3 4 5; do ./build/bin/diocotron_column /tmp/col$L 256 1500 $L; done
  python scripts/plot_diocotron_modes.py docs/fig_diocotron_modes.png /tmp/col3 /tmp/col4 /tmp/col5
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    out = Path(sys.argv[1])
    dirs = [Path(d) for d in sys.argv[2:]]
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, len(dirs), figsize=(3.4 * len(dirs), 3.6),
                            constrained_layout=True)
    if len(dirs) == 1:
        axs = [axs]
    fig.patch.set_alpha(0)
    for ax, d in zip(axs, dirs):
        frames = sorted(d.glob("dens_*.txt"))
        g = np.loadtxt(frames[len(frames) // 2])  # mi-parcours
        ax.imshow(g, origin="lower", extent=[0, 1, 0, 1], cmap="inferno")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(d.name)
    fig.savefig(out, dpi=120, transparent=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

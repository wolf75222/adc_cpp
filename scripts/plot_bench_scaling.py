"""Courbe de scaling OpenMP du banc bench_amr (deux-fluides AP mono-grille).

Lit le CSV produit par le balayage (n, threads, M mailles-MAJ/s) et trace le debit en
fonction du nombre de threads, une courbe par taille de grille, avec la droite de scaling
ideal (lineaire depuis 1 thread) en pointilles. Montre que le scaling depend de la taille
et plafonne aux 4 coeurs performants du M2.

Usage :
  bash scripts/bench_scaling.sh > /tmp/scaling.csv   # ou le one-liner du bench
  python scripts/plot_bench_scaling.py /tmp/scaling.csv docs/fig_openmp_scaling.png
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    csv_path, png = Path(sys.argv[1]), Path(sys.argv[2])
    png.parent.mkdir(parents=True, exist_ok=True)

    data: dict[int, dict[int, float]] = defaultdict(dict)
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            try:
                data[int(row["n"])][int(row["threads"])] = float(row["mcells_per_s"])
            except (ValueError, KeyError):
                continue

    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    fig.patch.set_alpha(0)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for c, (n, series) in zip(colors, sorted(data.items())):
        th = sorted(series)
        y = [series[t] for t in th]
        ax.plot(th, y, "o-", color=c, lw=1.8, label=f"n={n}")
        ideal = [y[0] * t for t in th]  # scaling lineaire depuis 1 thread
        ax.plot(th, ideal, "--", color=c, lw=0.9, alpha=0.5)

    ax.axvline(4, color="gray", ls=":", lw=0.9)
    ax.text(4.05, ax.get_ylim()[1] * 0.05, "4 coeurs perf (M2)", color="gray", fontsize=8)
    ax.set_xlabel("threads OpenMP")
    ax.set_ylabel("debit (M mailles-MAJ / s)")
    ax.set_title("Scaling OpenMP : deux-fluides AP mono-grille (M2)")
    ax.set_xticks([1, 2, 4, 8])
    ax.legend(title="grille", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

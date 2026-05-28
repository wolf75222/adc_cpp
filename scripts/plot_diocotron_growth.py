"""Taux de croissance de l'instabilite diocotron.

Lit diocotron_amp.csv (ecrit par examples/diocotron.cpp) et trace l'amplitude
du mode en echelle semi-log. Ajuste une exponentielle exp(gamma t) sur la phase
lineaire (croissance reguliere avant saturation) et reporte gamma.

Usage :
  ./build/bin/diocotron /tmp/dio out
  python scripts/plot_diocotron_growth.py /tmp/dio/diocotron_amp.csv docs/fig_diocotron_growth.png
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
    csv = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(csv, delimiter=",", skiprows=2)
    t, amp = data[:, 0], data[:, 1]

    # phase lineaire : de 1.1*amp0 jusqu'a 80% du max (avant saturation)
    a0, amax = amp[0], amp.max()
    mask = (amp > 1.1 * a0) & (amp < 0.8 * amax) & (t < t[np.argmax(amp)])
    gamma = np.nan
    if mask.sum() >= 3:
        coef = np.polyfit(t[mask], np.log(amp[mask]), 1)
        gamma = coef[0]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.semilogy(t, amp, "o", ms=3, color="C0", label="amplitude du mode (simulation)")
    if np.isfinite(gamma):
        tw = t[mask]
        fit = np.exp(coef[1]) * np.exp(gamma * tw)
        ax.semilogy(tw, fit, "-", color="crimson", lw=2,
                    label=rf"$\propto e^{{\gamma t}}$, $\gamma={gamma:.3f}$")
        ax.axvspan(tw[0], tw[-1], color="crimson", alpha=0.08,
                   label="fenetre lineaire")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|\,n_e - \langle n_e\rangle_x\,\|_2$")
    ax.set_title("Instabilite diocotron : croissance lineaire puis saturation")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out, dpi=130, transparent=True)
    print(f"wrote {out}")
    print(f"gamma (fit) = {gamma:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

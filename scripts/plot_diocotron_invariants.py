#!/usr/bin/env python3
"""Figure des INVARIANTS PHYSIQUES diocotron (verification de fidelite, au-dela du taux de
croissance). Trace, normalises a leur valeur initiale, vs le temps : masse (doit rester a 1,
forme flux), moment angulaire et energie de champ (~conserves), enstrophie (Casimir : sa
DECROISSANCE mesure la diffusion numerique du schema), et rho_max (principe du maximum).

Usage : python scripts/plot_diocotron_invariants.py invariants.csv out.png [--label TXT]
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cols = {}
    rows = []
    with open(args.csv) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("t,"):
                names = ln.split(",")
                continue
            parts = ln.split(",")
            if len(parts) != len(names):
                continue  # ligne partielle (fichier en cours d'ecriture)
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    d = np.asarray(rows)
    t = d[:, 0]
    idx = {n: k for k, n in enumerate(names)}

    fig, ax = plt.subplots(figsize=(7.5, 5))
    series = [("mass", "masse $\\int\\rho$"), ("angmom", "moment angulaire $\\int\\rho r^2$"),
              ("energy", "energie $\\frac{1}{2}\\int|\\nabla\\phi|^2$"),
              ("enstrophy", "enstrophie $\\int\\rho^2$ (diffusion)"),
              ("rho_max", r"$\rho_{\max}$ (principe du max)")]
    for key, lab in series:
        if key not in idx:
            continue
        y = d[:, idx[key]]
        y0 = y[0] if abs(y[0]) > 1e-300 else 1.0
        ax.plot(t, y / y0, lw=2, label=lab)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("temps physique $t$")
    ax.set_ylabel("invariant / valeur initiale")
    ttl = "Invariants physiques diocotron (transport E x B incompressible)"
    if args.label:
        ttl += f" - {args.label}"
    ax.set_title(ttl, fontsize=11)
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print("ecrit", args.out)


if __name__ == "__main__":
    sys.exit(main())

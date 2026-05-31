#!/usr/bin/env python3
"""Figure haute precision diocotron : taux de croissance instantane (fenetre glissante) du mode l
en fonction du temps, pour differents ordres de schema (NoSlope ordre 1, VanLeer ordre 2, WENO5
ordre 5), tous en SSPRK3, vs le taux analytique. Montre comment l'ordre eleve fait passer le taux
de la sous-evaluation diffusive (NoSlope) au voisinage de l'analytique (overshoot ~+8 %), et la
sensibilite a la fenetre de mesure (le papier ajuste une fenetre etroite apres le transitoire).

Usage : python scripts/plot_diocotron_highorder.py out.png CSV:label CSV:label ...
        [--mode 4] [--analytic 0.911] [--window-lo 4.2 --window-hi 5.2] [--width 1.0]
"""
import argparse
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OMD = 0.9 / (2 * math.pi)


def read(path):
    t, a = [], []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln[0] == "#" or ln.startswith("t,"):
            continue
        x, y = ln.split(",")
        t.append(float(x)); a.append(float(y))
    return np.asarray(t), np.asarray(a)


def sliding_gamma(t, a, width):
    """gamma_norm(t_center) ajuste sur [t_c - w/2, t_c + w/2]."""
    tc, g = [], []
    lo = t[0]
    while lo + width <= t[-1]:
        m = (t >= lo) & (t <= lo + width) & (a > 0)
        if m.sum() >= 4:
            coef = np.polyfit(t[m], np.log(a[m]), 1)
            tc.append(lo + width / 2); g.append(coef[0] / OMD)
        lo += width / 8
    return np.asarray(tc), np.asarray(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("series", nargs="+", help="CSV:label")
    ap.add_argument("--mode", type=int, default=4)
    ap.add_argument("--analytic", type=float, default=0.911)
    ap.add_argument("--window-lo", type=float, default=4.2)
    ap.add_argument("--window-hi", type=float, default=5.2)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(args.analytic, color="k", ls="--", lw=1.4,
               label=f"analytique $\\gamma_{{{args.mode}}}$ = {args.analytic}")
    ax.axvspan(args.window_lo, args.window_hi, color="0.85", alpha=0.6,
               label="fenetre de mesure (papier)")
    for s in args.series:
        path, lab = s.split(":", 1)
        t, a = read(path)
        tc, g = sliding_gamma(t, a, args.width)
        ax.plot(tc, g, lw=2, label=lab)
    ax.set_xlabel("temps physique $t$")
    ax.set_ylabel(r"$\gamma_{\mathrm{norm}}(t)$ (fenetre glissante)")
    ax.set_title(f"Diocotron mode {args.mode} : taux instantane vs ordre du schema (SSPRK3)")
    ax.set_ylim(0.3, 1.15)
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print("ecrit", args.out)


if __name__ == "__main__":
    sys.exit(main())

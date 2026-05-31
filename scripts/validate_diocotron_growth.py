#!/usr/bin/env python3
"""Validation du taux de croissance diocotron : numerique (simu) vs analytique (papier).

Reproduit la Section 5 de Hoffart-Maier-Shadid-Tomas (arXiv:2510.11808) avec le solveur
volumes-finis : on lance l'instabilite d'une colonne creuse (mode azimutal l), on ajuste une
exponentielle exp(gamma t) sur la PHASE LINEAIRE (montee reguliere avant saturation) de
l'amplitude du mode l du potentiel, puis on NORMALISE par la frequence diocotron
omega_D = rhobar / (2 pi) (convention du papier ; rhobar = densite moyenne de l'anneau =
1 - delta). Le taux normalise doit converger, en raffinant le maillage, vers le taux
analytique (probleme aux valeurs propres de Petri/Davidson-Felice, cf. diocotron_theory).

Usage :
  python scripts/validate_diocotron_growth.py CSV [--rhobar R] [--target G] [--label TXT]
  # plusieurs CSV (etude de resolution) : passer plusieurs chemins.
"""
import argparse
import math
import sys

import numpy as np


def read_amp(path):
    t, a = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("t,"):
                continue
            ts, as_ = line.split(",")
            t.append(float(ts))
            a.append(float(as_))
    return np.asarray(t), np.asarray(a)


def fit_fixed_window(t, a, t0, t1):
    """Ajuste log(a) ~ gamma t sur une FENETRE PHYSIQUE FIXE [t0, t1].

    Plus robuste que la detection de pic pour COMPARER des resolutions : la fenetre
    saturation-relative (0.85 du pic) deplace ses bornes quand le pic se decale avec la
    resolution, ce qui BIAISE la pente a haute resolution (le rollover de saturation est
    capte differemment). Une fenetre commune, choisie dans la phase exponentielle propre
    (avant tout debut de saturation), mesure le MEME regime pour toutes les resolutions.
    """
    m = (t >= t0) & (t <= t1) & (a > 0)
    if m.sum() < 4:
        return float("nan"), (t0, t1)
    coef = np.polyfit(t[m], np.log(a[m]), 1)
    return coef[0], (t0, t1)


def fit_linear_phase(t, a):
    """Ajuste log(a) ~ gamma t sur la phase de croissance avant saturation."""
    if len(a) < 8:
        return float("nan"), (0, 0)
    ipk = int(np.argmax(a))           # pic = fin de la phase lineaire
    a0 = a[0] if a[0] > 0 else a[a > 0][0]
    # debut : des que l'amplitude a depasse 1.3 x le seuil initial ; fin : 85% du pic
    lo = np.searchsorted(a[:ipk + 1], 1.3 * a0) if ipk > 0 else 0
    hi = ipk
    while hi > lo and a[hi] > 0.85 * a[ipk]:
        hi -= 1
    hi = max(hi, lo + 4)
    hi = min(hi, ipk)
    if hi - lo < 4:
        lo, hi = max(1, ipk // 4), ipk
    sl = slice(lo, hi + 1)
    coef = np.polyfit(t[sl], np.log(a[sl]), 1)
    return coef[0], (t[lo], t[hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--rhobar", type=float, default=0.9, help="densite moyenne anneau (1-delta)")
    ap.add_argument("--target", type=float, default=0.911, help="taux analytique vise (mode)")
    ap.add_argument("--labels", default="", help="etiquettes separees par des virgules")
    ap.add_argument("--window", default="", help="fenetre physique fixe t0,t1 (phase lineaire) "
                    "au lieu de la detection de pic ; recommande pour comparer des resolutions")
    args = ap.parse_args()

    win = None
    if args.window:
        w = args.window.split(",")
        win = (float(w[0]), float(w[1]))

    omega_d = args.rhobar / (2.0 * math.pi)
    labels = args.labels.split(",") if args.labels else [c for c in args.csv]
    print(f"omega_D = rhobar/(2 pi) = {args.rhobar}/(2 pi) = {omega_d:.5f}")
    if win:
        print(f"fenetre lineaire fixe : t in [{win[0]}, {win[1]}]")
    print(f"taux analytique vise (normalise) : {args.target:.4f}\n")
    print(f"{'cas':<22} {'gamma_phys':>11} {'gamma_norm':>11} {'ratio':>8} {'fenetre fit':>16}")
    for path, lab in zip(args.csv, labels):
        t, a = read_amp(path)
        g_phys, (t0, t1) = fit_fixed_window(t, a, *win) if win else fit_linear_phase(t, a)
        g_norm = g_phys / omega_d
        ratio = g_norm / args.target if args.target else float("nan")
        print(f"{lab[:22]:<22} {g_phys:>11.4f} {g_norm:>11.4f} {ratio:>8.3f} "
              f"  [{t0:.1f},{t1:.1f}]")


if __name__ == "__main__":
    sys.exit(main())

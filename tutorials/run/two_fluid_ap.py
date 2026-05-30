#!/usr/bin/env python3
"""Tutoriel exécutable : le schéma deux-fluides asymptotic-preserving (AP).

Régime raide (omega_pe = 1e3, dt*omega_pe = 5). Le schéma AP (stabilize=True) reste
borné et quasi-neutre ; le même schéma sans la reformulation de Poisson (stabilize=False)
explose. On trace max|n_e - 1| en fonction du pas pour les deux, sur le MÊME pas de temps :
c'est la démonstration directe de l'intérêt du schéma AP.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/two_fluid_ap.py
Sortie : docs/tut_tfap_ap.png
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


def run(stabilize: bool, nsteps: int, dt: float):
    cfg = adc.TwoFluidAPConfig()
    cfg.n = 64
    cfg.omega_pe = 1e3
    cfg.omega_pi = 20.0
    cfg.stabilize = stabilize
    ts = adc.TwoFluidAPSolver(cfg)
    dev = []
    for _ in range(nsteps):
        ts.step(dt)
        d = ts.max_dev()
        dev.append(d if np.isfinite(d) else np.nan)
        if not np.isfinite(d) or d > 1e3:
            break          # non stabilisé : a explosé
    return np.array(dev)


def main() -> int:
    dt = 5.0 / 1e3          # dt*omega_pe = 5 : un explicite naïf exigerait dt ~ 1e-3
    nsteps = 120
    ap = run(True, nsteps, dt)
    ex = run(False, nsteps, dt)
    print(f"AP (stabilisé) : max|dne| final = {ap[-1]:.2e} (borné)")
    print(f"non stabilisé  : explose au pas {len(ex)} (max|dne| = {ex[-1]:.2e})")

    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.semilogy(np.arange(len(ap)) * dt * 1e3, np.clip(ap, 1e-9, None),
                "C2-", lw=1.8, label="AP (stabilize=True)")
    ax.semilogy(np.arange(len(ex)) * dt * 1e3, np.clip(ex, 1e-9, None),
                "C3--", lw=1.8, label="sans AP (explose)")
    ax.set_xlabel(r"temps $\times\,\omega_{pe}$")
    ax.set_ylabel(r"max$|n_e - 1|$")
    ax.set_title(r"Schéma asymptotic-preserving : $\omega_{pe}\,\Delta t = 5$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    out = ROOT / "docs" / "tut_tfap_ap.png"
    fig.savefig(out, dpi=130)
    print(f"écrit {out}")

    assert ap[-1] < 0.1, "le schéma AP doit rester borné"
    assert len(ex) < nsteps or ex[-1] > 1.0, "le non-stabilisé doit exploser"
    return 0


if __name__ == "__main__":
    sys.exit(main())

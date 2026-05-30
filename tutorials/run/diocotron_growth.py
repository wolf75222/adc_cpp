#!/usr/bin/env python3
"""Tutoriel exécutable : taux de croissance de l'instabilité diocotron.

Condition initiale en bande perturbée. Pendant la phase linéaire, la perturbation
croît exponentiellement avant de saturer (enroulement). On suit l'écart RMS à l'état
initial, on le trace en échelle log, et on ajuste la pente de la phase linéaire : c'est
le taux de croissance gamma. Validation physique directe du couplage E x B + Poisson.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/diocotron_growth.py
Sortie : docs/tut_diocotron_growth.png
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


def main() -> int:
    cfg = adc.DiocotronConfig()
    cfg.n = 128
    cfg.ic = adc.DiocotronIC.Band
    cfg.band_mode = 4
    cfg.band_amp = 0.3
    cfg.band_disp = 1e-3          # petite perturbation -> longue phase linéaire
    sim = adc.DiocotronSolver(cfg)

    rho0 = sim.density().copy()
    dx = sim.dx()
    t, amp = [], []
    nsteps = 700
    for s in range(nsteps):
        sim.step_cfl(0.3)
        if s % 5 == 0:
            d = sim.density() - rho0
            t.append(sim.time())
            amp.append(np.sqrt(np.mean(d * d)))     # écart RMS à l'initial
    t, amp = np.array(t), np.array(amp)

    # ajuste la pente exponentielle sur la fenêtre linéaire (entre 1% et 30% du max)
    lo, hi = 0.01 * amp.max(), 0.3 * amp.max()
    mask = (amp > lo) & (amp < hi)
    gamma = np.polyfit(t[mask], np.log(amp[mask]), 1)[0] if mask.sum() > 3 else float("nan")
    print(f"diocotron mode {cfg.band_mode} : taux de croissance ajusté gamma = {gamma:.3f}")

    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.semilogy(t, amp, "C0-", lw=1.6, label="écart RMS à l'initial")
    if np.isfinite(gamma):
        ax.semilogy(t[mask], np.exp(np.polyval(np.polyfit(t[mask], np.log(amp[mask]), 1), t[mask])),
                    "C3--", lw=1.4, label=fr"ajustement $e^{{\gamma t}}$, $\gamma={gamma:.2f}$")
    ax.set(xlabel="temps", ylabel="amplitude de la perturbation",
           title=f"Instabilité diocotron (mode {cfg.band_mode}) : croissance exponentielle")
    ax.legend(frameon=False); ax.grid(True, which="both", alpha=0.25)
    out = ROOT / "docs" / "tut_diocotron_growth.png"
    fig.savefig(out, dpi=130)
    print(f"écrit {out}")

    assert np.isfinite(gamma) and gamma > 0, "la perturbation doit croître (instabilité)"
    return 0


if __name__ == "__main__":
    sys.exit(main())

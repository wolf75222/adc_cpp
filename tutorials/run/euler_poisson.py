#!/usr/bin/env python3
"""Tutoriel exécutable : Euler-Poisson auto-gravitant depuis les bindings Python.

Un gaz compressible couplé à sa propre gravité (Poisson). On l'avance et on suit la
densité maximale, la masse et la quantité de mouvement totale. Invariants attendus :
masse conservée à l'arrondi, quantité de mouvement totale nulle (le système est
symétrique, aucune force extérieure nette). Backend elliptique au choix (FFT direct ou
multigrille) via use_fft.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/euler_poisson.py
Sortie : docs/tut_euler_poisson.png
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
    cfg = adc.EulerPoissonConfig()
    cfg.n = 64
    cfg.use_fft = True          # Poisson spectral direct (périodique, n puissance de 2)
    es = adc.EulerPoissonSolver(cfg)

    m0 = es.mass()
    nsteps, dt = 120, 0.004
    t, rho_max, px, drift = [], [], [], []
    for s in range(nsteps):
        es.step(dt)
        t.append(s * dt)
        rho_max.append(es.density().max())
        px.append(abs(es.total_momentum(0)) + abs(es.total_momentum(1)))
        drift.append(abs(es.mass() - m0))
    print(f"Euler-Poisson n={cfg.n} : rho_max {rho_max[0]:.3f} -> {rho_max[-1]:.3f}, "
          f"|p|_max = {max(px):.2e}, dérive masse_max = {max(drift):.2e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
    fig.patch.set_alpha(0)
    a1.plot(t, rho_max, "C0-", lw=1.8)
    a1.set(xlabel="temps", ylabel=r"$\rho_{\max}$", title="densité maximale (auto-gravité)")
    a1.grid(True, alpha=0.25)
    a2.semilogy(t, np.clip(px, 1e-20, None), "C3-", lw=1.5, label=r"$|p_x|+|p_y|$")
    a2.semilogy(t, np.clip(drift, 1e-20, None), "C2--", lw=1.5, label="dérive masse")
    a2.set(xlabel="temps", ylabel="invariant", title="conservation (doit rester ~ arrondi)")
    a2.legend(frameon=False); a2.grid(True, which="both", alpha=0.25)
    out = ROOT / "docs" / "tut_euler_poisson.png"
    fig.savefig(out, dpi=130)
    print(f"écrit {out}")

    assert max(px) < 1e-9, "la quantité de mouvement totale doit rester nulle"
    assert max(drift) < 1e-9, "la masse doit être conservée"
    return 0


if __name__ == "__main__":
    sys.exit(main())

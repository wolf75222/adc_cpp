#!/usr/bin/env python3
"""Tutoriel exécutable : effondrement de Jeans vs stabilité du plasma, profil en image.

Même perturbation acoustique `delta_rho = eps rho0 cos(kx)`, champ fort (four_pi_G = 120,
omega_p > c_s k). En GRAVITÉ (attractif) elle est instable : les sur-densités s'effondrent
en pics (formation de structure). En PLASMA (répulsif) elle oscille, bornée. On anime le
profil rho(x, t) des deux côtés, chacun avec sa propre échelle.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/euler_poisson_collapse.py [n] [nsteps]
Sortie : docs/tut_ep_collapse.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import adc

ROOT = Path(__file__).resolve().parents[2]
FOUR_PI_G, EPS = 120.0, 2e-2


def profiles(kind, n, nsteps, dt, snap):
    cfg = adc.EulerPoissonConfig()
    cfg.n, cfg.four_pi_G, cfg.eps = n, FOUR_PI_G, EPS
    cfg.interaction, cfg.use_fft = kind, False
    s = adc.EulerPoissonSolver(cfg)
    out, ts = [], []
    for k in range(nsteps + 1):
        if k % snap == 0:
            out.append(np.asarray(s.density()).mean(axis=0).copy())  # rho(x) = moyenne_y
            ts.append(s.time())
        if k < nsteps:
            s.step(dt)
    return np.array(ts), out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    cs2 = (5.0 / 3.0)
    dt = 0.3 * (1.0 / n) / (np.sqrt(cs2) + 0.1)
    snap = max(1, nsteps // 60)
    x = (np.arange(n) + 0.5) / n

    tg, grav = profiles(adc.InteractionKind.Gravity, n, nsteps, dt, snap)
    tp, plas = profiles(adc.InteractionKind.Plasma, n, nsteps, dt, snap)
    gmax = max(f.max() for f in grav)
    pdev = max(np.abs(f - 1).max() for f in plas)
    print(f"effondrement : gravité pic {grav[0].max():.3f} -> {grav[-1].max():.3f} "
          f"(croît) ; plasma dévie de {pdev:.3f} (borné)")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), constrained_layout=True)
    fig.patch.set_alpha(0)

    def update(k):
        a1.clear(); a2.clear()
        a1.plot(x, grav[k], "C3-", lw=2)
        a1.set(ylim=(0.0, 1.1 * gmax), ylabel=r"$\rho(x)$",
               title=f"gravité : effondrement de Jeans (t = {tg[k]:.3f})")
        a1.fill_between(x, 1.0, grav[k], color="C3", alpha=0.15)
        a2.plot(x, plas[k], "C0-", lw=2)
        a2.set(ylim=(1 - 1.4 * pdev, 1 + 1.4 * pdev), xlabel="x", ylabel=r"$\rho(x)$",
               title="plasma : oscillation bornée (toujours stable)")
        a2.axhline(1.0, color="k", lw=0.6, alpha=0.4)
        for a in (a1, a2):
            a.grid(True, alpha=0.25)
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(grav), blit=False)
    out = ROOT / "docs" / "tut_ep_collapse.gif"
    anim.save(out, writer=animation.PillowWriter(fps=12))
    print(f"écrit {out}  ({len(grav)} frames)")
    assert grav[-1].max() > 1.3 * grav[0].max(), "la gravité doit s'effondrer (pic croît)"
    assert pdev < 0.3, "le plasma doit rester borné"
    return 0


if __name__ == "__main__":
    sys.exit(main())

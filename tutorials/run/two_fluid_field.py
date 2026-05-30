#!/usr/bin/env python3
"""Tutoriel exécutable : champ de séparation de charge du deux-fluides AP (animation).

Le schéma asymptotic-preserving ne se résume pas à une courbe : il transporte deux
densités (électrons, ions) couplées par Poisson. On anime la densité électronique n_e et
la séparation de charge n_e - n_i, qui oscille à la fréquence plasma tout en restant
quasi-neutre (max|n_e - n_i| petit). Régime modérément raide pour que le motif soit lisible.

Usage :
  PYTHONPATH=build-py/python python3 tutorials/run/two_fluid_field.py [n] [nsteps]
Sortie : docs/tut_tfap_field.gif
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


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 240

    cfg = adc.TwoFluidAPConfig()
    cfg.n = n
    cfg.L = 1.0
    cfg.omega_pe, cfg.omega_pi = 40.0, 6.5
    cfg.eps = 0.1
    ts = adc.TwoFluidAPSolver(cfg)

    # Le pas est borne par la CFL de TRANSPORT (advection au son cse), pas par omega_pe :
    # la raideur plasma est traitee implicitement par le schema AP, le transport non.
    cse = 1.0  # vitesse du son electronique (defaut cse2 = 1)
    dt = 0.3 * (cfg.L / n) / (cse + 0.1)
    me, charge, snap = [], [], max(1, nsteps // 60)
    m0 = ts.mass_e()
    for s in range(nsteps + 1):
        if s % snap == 0:
            ne = np.asarray(ts.density_e())
            ni = np.asarray(ts.density_i())
            me.append(ne.copy()); charge.append((ne - ni).copy())
        if s < nsteps:
            ts.step(dt)
    drift = abs(ts.mass_e() - m0)
    qmax = max(np.abs(c).max() for c in charge)
    evmin = min(f.min() for f in me); evmax = max(f.max() for f in me)
    print(f"deux-fluides champ n={n}, {nsteps} pas : max|charge|={qmax:.3f}, "
          f"dérive masse_e={drift:.2e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 4.2), constrained_layout=True)
    fig.patch.set_alpha(0)

    def update(k):
        for a in (a1, a2):
            a.clear(); a.set_xticks([]); a.set_yticks([])
        a1.imshow(me[k], origin="lower", cmap="viridis", vmin=evmin, vmax=evmax,
                  extent=[0, 1, 0, 1])
        a1.set_title("densité électronique $n_e$")
        a2.imshow(charge[k], origin="lower", cmap="seismic", vmin=-qmax, vmax=qmax,
                  extent=[0, 1, 0, 1])
        a2.set_title("séparation de charge $n_e - n_i$")
        fig.suptitle(f"deux-fluides AP, pas {k * snap}")
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(me), blit=False)
    out = ROOT / "docs" / "tut_tfap_field.gif"
    anim.save(out, writer=animation.PillowWriter(fps=12))
    print(f"écrit {out}  ({len(me)} frames)")
    assert drift < 1e-6, "la masse électronique doit être conservée"
    assert qmax < 1.0, "le plasma doit rester quasi-neutre (charge bornée)"
    return 0


if __name__ == "__main__":
    sys.exit(main())

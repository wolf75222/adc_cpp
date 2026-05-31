#!/usr/bin/env python3
# GIF du diocotron AMR pilote DEPUIS PYTHON (binding DiocotronAmrSolver). Montre que la
# facade AMR compilee est utilisable sans une ligne de C++ : config, boucle step_cfl,
# densite en numpy a chaque frame. Le nombre de patchs evolue par regrid Berger-Rigoutsos.
#
#   PYTHONPATH=<dir du .so> python3 scripts/make_python_amr_gif.py docs/anim_python_amr.gif
#
# Sans argument : ecrit docs/anim_python_amr.gif.

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import adc  # binding pybind11 (libadc)

out = sys.argv[1] if len(sys.argv) > 1 else "docs/anim_python_amr.gif"

cfg = adc.DiocotronAmrConfig()
cfg.n = 128
cfg.band_amp = 1.0
cfg.band_mode = 2
cfg.refine_frac = 0.15
cfg.regrid_every = 15
sim = adc.DiocotronAmrSolver(cfg)
m0 = sim.mass()

frames, patches = [], []
nframes, sub = 40, 12
for f in range(nframes):
    for _ in range(sub):
        sim.step_cfl(0.4)
    frames.append(sim.density().copy())
    patches.append(sim.n_patches())

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(frames[0], origin="lower", cmap="magma", vmin=0.8, vmax=2.2)
ax.set_xticks([]); ax.set_yticks([])
ttl = ax.set_title("")

def upd(k):
    im.set_data(frames[k])
    ttl.set_text(f"diocotron AMR (Python) - frame {k} - {patches[k]} patchs")
    return im, ttl

ani = animation.FuncAnimation(fig, upd, frames=len(frames), blit=False)
ani.save(out, writer=animation.PillowWriter(fps=8))
print(f"ecrit {out} ({len(frames)} frames) ; mass drift = {abs(sim.mass()-m0):.2e} ; "
      f"patchs {min(patches)}..{max(patches)}")

#!/usr/bin/env python3
# Courbes de croissance diocotron a partir des ring_amp.csv REELS du hero-run ROMEO
# (job 613961, WENO5-Z + SSPRK3). Chaque CSV : amplitude du mode vs temps. La phase
# lineaire = droite en echelle log-amplitude ; sa pente = taux de croissance.
#
#   python3 scripts/plot_romeo_growth.py [dir=romeo/runs/613961_growth]
# -> docs/romeo_growth_mode4.png
#
# Robuste aux lignes hors-ordre / dupliquees (tri par t, dedup) : un dump peut etre
# desordonne, le fit log-lineaire ne depend que de la relation (t, amplitude).

import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt

root = sys.argv[1] if len(sys.argv) > 1 else "romeo/runs/613961_growth"

def load(tag):
    f = os.path.join(root, tag, "ring_amp.csv")
    if not os.path.exists(f):
        return None
    rows = []
    for ln in open(f):
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("t,"):
            continue
        a, b = ln.split(",")
        rows.append((float(a), float(b)))
    d = np.array(rows)
    t, amp = d[:, 0], d[:, 1]
    order = np.argsort(t, kind="stable")           # tri par t (robuste au desordre)
    t, amp = t[order], amp[order]
    keep = np.concatenate(([True], np.diff(t) > 0))  # dedup t
    t, amp = t[keep], amp[keep]
    amp = np.abs(amp)
    return t, amp

def gamma_fit(t, amp, lo, hi):
    m = (t >= lo) & (t <= hi) & (amp > 0)
    if m.sum() < 3:
        return None
    s = np.polyfit(t[m], np.log(amp[m]), 1)[0]  # pente de ln(amp) = taux brut
    return s

# --- mode 4 (mode du papier, cible 0.911), 3 resolutions effectives ---
colors = {256: "#9ecae1", 512: "#3182bd", 1024: "#08519c"}
fig, ax = plt.subplots(figsize=(6.6, 4.4))
omega_D = 0.9 / (2 * np.pi)   # normalisation rho_bar/(2 pi), rho_bar = 0.9 (cf. journal)
for eff in (256, 512, 1024):
    r = load(f"m4_e{eff}")
    if r is None:
        print(f"m4_e{eff} : absent"); continue
    t, amp = r
    ax.semilogy(t, amp, "-", color=colors[eff], lw=1.8, label=f"eff {eff}")
    g = gamma_fit(t, amp, 4.2, 5.2)
    info = "" if g is None else f"  (gamma_norm~{g/omega_D:.3f})"
    print(f"m4_e{eff}: n={len(t)} t=[{t.min():.2f},{t.max():.2f}] amp=[{amp.min():.2e},{amp.max():.2e}]{info}")
ax.set_xlabel("temps")
ax.set_ylabel("amplitude du mode (log)")
ax.set_title("Diocotron mode 4 : croissance mesuree sur ROMEO (job 613961, WENO5-Z)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(title="resolution effective", fontsize=9)
fig.tight_layout()
fig.savefig("docs/romeo_growth_mode4.png", dpi=130, transparent=True)
print("ecrit docs/romeo_growth_mode4.png")

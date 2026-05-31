#!/usr/bin/env python3
# Plots de convergence du taux de croissance diocotron, a partir des resultats ROMEO
# (GH200/EPYC, journal romeo/HERO_RESULTS.md). Les nombres sont les gamma_norm mesures
# sur les hero-runs ; les sorties brutes du scratch ROMEO ne sont pas conservees, ces
# tables EN sont le resultat distille.
#
#   python3 scripts/plot_romeo_convergence.py
#
# -> docs/romeo_highorder_convergence.png , docs/romeo_amr_efficiency.png

import matplotlib.pyplot as plt

# --- Job 613961 : WENO5-Z + SSPRK3, gamma_norm dans la fenetre du papier ---
eff = [256, 512, 1024]
analytic = {3: 0.772, 4: 0.911, 5: 0.683}            # cible analytique par mode
gamma = {3: [0.838, 0.850, 0.853],
         4: [0.985, 0.988, 0.987],
         5: [0.730, 0.731, 0.729]}
colors = {3: "#1f77b4", 4: "#d62728", 5: "#2ca02c"}

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for m in (3, 4, 5):
    ax.plot(eff, gamma[m], "o-", color=colors[m], lw=2, ms=6, label=f"mesure, mode {m}")
    ax.axhline(analytic[m], color=colors[m], ls="--", lw=1, alpha=0.7)
    ax.text(1024, analytic[m], f"  analytique {analytic[m]}", color=colors[m],
            va="center", ha="left", fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xticks(eff); ax.set_xticklabels(eff)
ax.set_xlabel("resolution effective (cellules par cote)")
ax.set_ylabel(r"$\gamma_{\mathrm{norm}}$ (fenetre du papier)")
ax.set_title("Diocotron WENO5-Z + SSPRK3 sur ROMEO (job 613961)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="center right")
ax.margins(x=0.18)
fig.tight_layout()
fig.savefig("docs/romeo_highorder_convergence.png", dpi=130, transparent=True)
print("ecrit docs/romeo_highorder_convergence.png")

# --- Job 613945 : colonne, AMR multi-niveau vs uniforme (gamma_norm lin) ---
# (cellules, gamma_lin) par cas ; AMR atteint le meme taux a bien moins de cellules.
unif = {512: (262_144, 0.753), 1024: (1_048_576, 0.748)}     # uniforme VanLeer
amr = {512: (104_632, 0.762), 1024: (409_008, 0.747)}        # AMR multi-niveau VanLeer

fig2, ax2 = plt.subplots(figsize=(6.4, 4.2))
ux = [unif[e][0] for e in (512, 1024)]; uy = [unif[e][1] for e in (512, 1024)]
ax_ = [amr[e][0] for e in (512, 1024)]; ay = [amr[e][1] for e in (512, 1024)]
ax2.plot(ux, uy, "s-", color="#444", lw=2, ms=8, label="uniforme VanLeer")
ax2.plot(ax_, ay, "o-", color="#d62728", lw=2, ms=8, label="AMR multi-niveau VanLeer")
for e in (512, 1024):
    frac = 100 * amr[e][0] / unif[e][0]
    ax2.annotate(f"eff {e}\n{frac:.0f} % des cellules", (amr[e][0], amr[e][1]),
                 textcoords="offset points", xytext=(8, -22), fontsize=8, color="#d62728")
    ax2.annotate(f"eff {e}", (unif[e][0], unif[e][1]),
                 textcoords="offset points", xytext=(8, 6), fontsize=8, color="#444")
ax2.set_xscale("log")
ax2.set_xlabel("nombre de cellules avancees")
ax2.set_ylabel(r"$\gamma_{\mathrm{norm}}$ (fenetre lineaire)")
ax2.set_title("Efficacite AMR : meme taux, moins de cellules (ROMEO job 613945)")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, loc="lower right")
ax2.margins(x=0.25, y=0.2)
fig2.tight_layout()
fig2.savefig("docs/romeo_amr_efficiency.png", dpi=130, transparent=True)
print("ecrit docs/romeo_amr_efficiency.png")

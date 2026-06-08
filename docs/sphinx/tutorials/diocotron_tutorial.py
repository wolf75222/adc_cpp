#!/usr/bin/env python3
"""Tutoriel canonique adc_cpp : instabilite diocotron reduite (derive E x B), de bout en bout.

Ce script est la SOURCE de verite du tutoriel A->Z de la documentation Sphinx : la page
``getting_started/tutorial`` l'inclut par ``literalinclude`` (le code de la doc N'EST PAS recopie
a la main). Il est volontairement AUTONOME : il ne depend que de ``adc`` (le module compile),
``numpy`` et ``matplotlib`` -- aucune dependance au paquet applicatif ``adc_cases``.

Ce qu'il fait, dans l'ordre du tutoriel
---------------------------------------
1. importe ``adc`` et detecte le backend reellement execute ;
2. ECRIT LE MODELE EN FORMULES avec ``adc.dsl.Model`` (variable conservative, champs auxiliaires
   phi/grad, flux d'advection E x B, valeurs propres, second membre elliptique) ;
3. COMPILE le modele : essaie le backend ``production`` (chemin natif zero-copie, prefere) puis
   retombe sur ``aot`` (numeriquement identique, host-marshale) -- exactement comme les cas reels ;
4. construit un ``adc.System`` periodique, choisit le schema (MUSCL minmod + Rusanov, explicite),
   branche le Poisson de systeme (densite de charge, multigrille) et pose la condition initiale ;
5. integre en temps en capturant des diagnostics (amplitude de la perturbation, masse) ;
6. produit les figures : une COURBE (amplitude vs temps), une CARTE 2D (densite finale), un GIF, et
   une COMPARAISON uniforme/AMR (``adc.AmrSystem`` raffine vs grille uniforme) ;
7. ecrit un enregistrement de provenance (SHA adc_cpp, backend, resolution, commande) a cote des
   figures, pour que chaque asset soit reproductible.

Physique (modele REDUIT, pas le systeme Euler-Poisson complet)
--------------------------------------------------------------
Une seule variable conservative, la densite ``n``, advectee par la derive E x B
``v = (-d_y phi / B0, d_x phi / B0)`` (a divergence nulle), ou ``phi`` resout le Poisson de systeme
``-lap phi = alpha (n - n_i0)`` (fond ionique neutralisant ``n_i0``). Conventions ancrees dans le
coeur : ``include/adc/physics/hyperbolic.hpp`` (``ExBVelocity``) et ``.../elliptic.hpp``
(``BackgroundDensity``). C'est le benchmark de NORMALISATION du diocotron, pas une reproduction du
systeme complet (cf. ``adc_cases/diocotron`` et ``docs/HOFFART_FIDELITY.md``).

Usage
-----
    python diocotron_tutorial.py [--n 96] [--steps 60] [--outdir _assets] [--quick]

``--quick`` reduit la resolution et le nombre de pas pour un passage de fumee rapide (CI doc).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import adc
from adc import dsl

# Parametres physiques du modele reduit (doivent etre coherents entre les formules et le RHS Poisson).
B0 = 1.0      # champ magnetique de fond (porte la derive E x B)
ALPHA = 1.0   # facteur du second membre elliptique alpha (n - n_i0)

REPO_ROOT = Path(__file__).resolve().parents[3]   # tutorials -> sphinx -> docs -> racine du depot
ADC_INCLUDE = os.environ.get("ADC_INCLUDE", str(REPO_ROOT / "include"))


# --- 2. Le modele, ECRIT EN FORMULES (adc.dsl.Model) -----------------------------------------------
def diocotron_model(n_i0: float) -> "dsl.Model":
    """Modele diocotron reduit en formules symboliques, reproduisant les briques natives
    ``ExBVelocity`` (transport) et ``BackgroundDensity`` (elliptique). ``n_i0`` = fond ionique
    neutralisant (moyenne de la densite, pour la solubilite du Poisson periodique)."""
    m = dsl.Model("diocotron_tutorial")

    (n,) = m.conservative_vars("n")          # unique variable conservative : la densite (role Density)
    m.aux("phi")                             # champs auxiliaires fournis par le solveur (canal adc::Aux)
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    vx = (-grad_y) / B0                       # derive E x B : v = (-d_y phi / B0, d_x phi / B0)
    vy = grad_x / B0
    m.flux(x=[n * vx], y=[n * vy])            # flux d'advection f = n v(dir)
    m.eigenvalues(x=[vx], y=[vy])             # spectre : une onde, la vitesse de derive

    m.primitive_vars(n=n)                     # scalaire transporte : primitif = conservatif
    m.conservative_from([n])
    m.elliptic_rhs(ALPHA * (n - n_i0))        # couple le bloc au Poisson : rhs = alpha (n - n_i0)

    m.check()                                 # toute variable referencee doit etre declaree
    return m


# --- 2bis. Le MEME modele, compose de briques natives (l'autre front d'ecriture) --------------------
def native_diocotron_model(n_i0: float):
    """Le MEME diocotron, mais compose de briques natives au lieu de formules : Scalar (etat) +
    ExB (transport E x B) + NoSource + BackgroundDensity (second membre elliptique alpha (n - n_i0)).
    C'est l'autre facon d'ecrire un modele -- on prouve plus bas (``native_vs_dsl``) qu'elle produit
    un etat BIT-IDENTIQUE aux formules. Conventions C++ : ``ExBVelocity`` et ``BackgroundDensity``."""
    return adc.Model(state=adc.Scalar(),
                     transport=adc.ExB(B0=B0),
                     source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=ALPHA, n0=n_i0))


# --- 4. Condition initiale (bande de charge perturbee, mode azimutal) -------------------------------
def band_density(n: int, L: float = 1.0, amp: float = 1.0, width: float = 0.05,
                 mode: int = 2, disp: float = 0.02, floor: float = 1.0) -> np.ndarray:
    """Bande horizontale de charge perturbee sinusoidalement le long de x (convention ``ne[j, i]``).

        ne(x, y) = floor + amp exp(-(y - y0)^2 / width^2),  y0 = 0.5 L + disp cos(2 pi mode x / L).
    """
    xs = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(xs, xs)                # indexing 'xy' : X[j, i] = xs[i], Y[j, i] = xs[j]
    y0 = 0.5 * L + disp * np.cos(2.0 * np.pi * mode * X / L)
    ne = floor + amp * np.exp(-((Y - y0) ** 2) / (width ** 2))
    return np.ascontiguousarray(ne)


def perturbation_amplitude(density: np.ndarray) -> float:
    """Amplitude L2 de la perturbation = ecart a la moyenne le long de x (la bande non perturbee est
    uniforme en x ; ce qui en devie porte l'instabilite)."""
    base = density.mean(axis=1, keepdims=True)
    delta = density - base
    return float(np.sqrt(np.mean(delta * delta)))


# --- 3+4. Compilation + branchement : production (natif) si possible, sinon aot (identique) ----------
def compile_and_build(model: "dsl.Model", ne0: np.ndarray, L: float, outdir: Path):
    """Compile le modele DSL ET le branche dans un ``adc.System`` periodique.

    Essaie d'abord ``production`` (chemin natif zero-copie ``add_native_block``, la cible du plan),
    puis retombe sur ``aot`` (``add_compiled_block``, numeriquement identique, host-marshale) --
    exactement la strategie des cas reels (cf. ``adc_cases/diocotron_dsl``). Le chemin natif exige
    que le module ``_adc`` et le ``.so`` du modele aient ETE COMPILES AVEC LES MEMES EN-TETES adc
    (garde d'ABI) ; un module frais (construit comme dans getting_started/installation) prend le
    chemin natif, sinon le ``aot`` s'applique. Renvoie (sim, backend_retenu)."""
    last = None
    for backend in ("production", "aot"):
        try:
            compiled = model.compile(str(outdir / f"diocotron_{backend}.so"),
                                     ADC_INCLUDE, backend=backend)
            sim = adc.System(n=ne0.shape[0], L=L, periodic=True)
            sim.add_equation("ne", model=compiled,
                             spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                             time=adc.Explicit())
            sim.set_poisson(rhs="charge_density", solver="geometric_mg")
            sim.set_density("ne", ne0)
            return sim, backend
        except Exception as exc:  # noqa: BLE001 - diagnostic : on tente le backend suivant
            last = exc
            print(f"  backend {backend!r} indisponible ({type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:80]}), essai suivant")
    raise RuntimeError("aucun backend DSL n'a pu etre branche au System") from last


def run(sim, steps: int, cfl: float, capture_every: int):
    """Integre ``steps`` pas, capture des trames + l'amplitude au fil du temps."""
    frames = [np.asarray(sim.density("ne")).copy()]
    times = [sim.time()]
    amps = [perturbation_amplitude(frames[0])]
    for k in range(steps):
        sim.step_cfl(cfl)
        if (k + 1) % capture_every == 0:
            frame = np.asarray(sim.density("ne")).copy()
            frames.append(frame)
            times.append(sim.time())
            amps.append(perturbation_amplitude(frame))
    return frames, np.array(times), np.array(amps)


# --- 6. Figures : courbe, carte, GIF, comparaison uniforme/AMR --------------------------------------
def make_figures(frames, times, amps, ne0, L, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Courbe (amplitude vs temps) + carte 2D (densite finale), cote a cote.
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].semilogy(times, amps, "o-", color="#b5179e")
    ax[0].set_xlabel("temps"); ax[0].set_ylabel("amplitude perturbation (L2)")
    ax[0].set_title("croissance de l'instabilite diocotron")
    im = ax[1].imshow(frames[-1].T, origin="lower", cmap="magma",
                      extent=[0, L, 0, L])
    ax[1].set_title("densite finale"); ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / "diocotron_growth.png", dpi=120)
    # Image de couverture PNG (premiere trame du GIF) pour les exports statiques.
    fig_cov, ax_cov = plt.subplots(figsize=(3.8, 3.6))
    ax_cov.imshow(frames[-1].T, origin="lower", cmap="magma", extent=[0, L, 0, L])
    ax_cov.set_title("diocotron (densite)"); ax_cov.set_xlabel("x"); ax_cov.set_ylabel("y")
    fig_cov.tight_layout(); fig_cov.savefig(outdir / "diocotron_cover.png", dpi=120)
    plt.close(fig_cov)

    # GIF de l'evolution de la densite.
    figg, axg = plt.subplots(figsize=(3.8, 3.6))
    img = axg.imshow(frames[0].T, origin="lower", cmap="magma", extent=[0, L, 0, L], animated=True)
    axg.set_xlabel("x"); axg.set_ylabel("y")

    def _update(i):
        img.set_data(frames[i].T)
        img.set_clim(frames[i].min(), frames[i].max())
        axg.set_title(f"t = {times[i]:.2f}")
        return [img]

    anim = FuncAnimation(figg, _update, frames=len(frames), blit=False)
    anim.save(outdir / "diocotron.gif", writer=PillowWriter(fps=6))
    plt.close(figg); plt.close(fig)


# --- 6bis. Comparaison uniforme vs AMR --------------------------------------------------------------
def uniform_vs_amr(ne0, n_i0, L, steps, cfl, outdir: Path):
    """Rejoue la MEME physique sur une grille uniforme et sur ``adc.AmrSystem`` (raffinement adaptatif),
    et trace les deux densites finales cote a cote. On utilise la composition NATIVE de briques pour
    cette comparaison (les deux chemins, uniforme et AMR, partagent exactement le meme modele)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    su = adc.System(n=ne0.shape[0], L=L, periodic=True)
    su.add_block("ne", model=native_diocotron_model(n_i0), spatial=adc.Spatial(minmod=True),
                 time=adc.Explicit())
    su.set_poisson(rhs="charge_density", solver="geometric_mg"); su.set_density("ne", ne0)

    sa = adc.AmrSystem(n=ne0.shape[0], L=L, periodic=True)
    sa.add_block("ne", model=native_diocotron_model(n_i0), spatial=adc.Spatial(minmod=True),
                 time=adc.Explicit())
    sa.set_refinement(0.05)
    sa.set_poisson(rhs="charge_density", solver="geometric_mg"); sa.set_density("ne", ne0)

    for _ in range(steps):
        su.step_cfl(cfl); sa.step_cfl(cfl)
    du = np.asarray(su.density("ne")); da = np.asarray(sa.density("ne"))

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for a, d, t in ((ax[0], du, "grille uniforme"), (ax[1], da, "AMR (raffinee)")):
        im = a.imshow(d.T, origin="lower", cmap="magma", extent=[0, L, 0, L])
        a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"diocotron : uniforme vs AMR (max|delta| = {float(np.max(np.abs(da - du))):.2e})")
    fig.tight_layout(); fig.savefig(outdir / "diocotron_uniform_vs_amr.png", dpi=120)
    plt.close(fig)
    return float(np.max(np.abs(da - du)))


# --- 6ter. La meme physique, deux fronts : briques == DSL (bit-identique) ---------------------------
def native_vs_dsl(dsl_final: np.ndarray, ne0, n_i0, L, steps, cfl, outdir: Path):
    """Rejoue la MEME physique en BRIQUES natives (``native_diocotron_model``) sur la meme grille / le
    meme schema / le meme nombre de pas que le run DSL, et compare l'etat final aux FORMULES. Les deux
    fronts d'ecriture (briques et formules) produisent un noyau numerique IDENTIQUE : l'ecart est nul a
    la precision binaire. Trace les deux cartes cote a cote avec ``max|ecart|`` en titre, et renvoie
    cet ecart (asserte nul dans ``main``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sb = adc.System(n=ne0.shape[0], L=L, periodic=True)
    sb.add_block("ne", model=native_diocotron_model(n_i0), spatial=adc.Spatial(minmod=True),
                 time=adc.Explicit())
    sb.set_poisson(rhs="charge_density", solver="geometric_mg")
    sb.set_density("ne", ne0)
    for _ in range(steps):
        sb.step_cfl(cfl)
    native_final = np.asarray(sb.density("ne"))

    max_delta = float(np.max(np.abs(native_final - dsl_final)))
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for a, d, t in ((ax[0], native_final, "briques natives"), (ax[1], dsl_final, "formules (DSL)")):
        im = a.imshow(d.T, origin="lower", cmap="magma", extent=[0, L, 0, L])
        a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"diocotron : meme physique, deux fronts (max|briques - DSL| = {max_delta:.0e})")
    fig.tight_layout(); fig.savefig(outdir / "diocotron_native_vs_dsl.png", dpi=120)
    plt.close(fig)
    return max_delta


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def detect_backend_runtime() -> str:
    """Backend de parallelisme reellement compile dans le module (cf. getting_started/backend)."""
    for attr in ("backend", "parallel_backend", "build_info"):
        val = getattr(adc, attr, None)
        if callable(val):
            try:
                return str(val())
            except Exception:  # noqa: BLE001
                pass
        elif val is not None:
            return str(val)
    return "serial (defaut ; cf. getting_started/backend pour Kokkos/MPI)"


def main() -> None:
    p = argparse.ArgumentParser(description="Tutoriel diocotron adc_cpp (A->Z, reproductible).")
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--outdir", default=str(Path(__file__).parent / "_assets"))
    p.add_argument("--quick", action="store_true", help="resolution/pas reduits (fumee CI)")
    args = p.parse_args()
    if args.quick:
        args.n, args.steps = 48, 20

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    L = 1.0

    print(f"adc importe depuis : {adc.__file__}")
    print(f"backend de parallelisme : {detect_backend_runtime()}")

    ne0 = band_density(args.n, L, amp=1.0, width=0.05, mode=2, disp=0.02)
    n_i0 = float(ne0.mean())

    t0 = time.time()
    sim, backend = compile_and_build(diocotron_model(n_i0), ne0, L, outdir)
    print(f"modele DSL compile + branche (backend = {backend!r}) en {time.time() - t0:.1f} s")

    capture_every = max(1, args.steps // 15)
    frames, times, amps = run(sim, args.steps, args.cfl, capture_every)

    growth = amps[-1] / amps[0]
    mass_drift = abs(float(sim.mass("ne")) - float(ne0.sum())) / float(ne0.sum())
    print(f"amplitude {amps[0]:.3e} -> {amps[-1]:.3e} (x{growth:.2f}) ; derive masse = {mass_drift:.2e}")
    assert growth > 1.0, "l'instabilite diocotron n'a pas cru"
    assert mass_drift < 1e-9, "masse non conservee (transport advectif periodique)"

    make_figures(frames, times, amps, ne0, L, outdir)

    # La MEME physique en briques natives : on prouve que les deux fronts d'ecriture (formules DSL et
    # briques) produisent un etat BIT-IDENTIQUE (meme noyau numerique), puis on trace les deux cartes.
    nd_delta = native_vs_dsl(frames[-1], ne0, n_i0, L, args.steps, args.cfl, outdir)
    print(f"equivalence briques/DSL : max|delta| = {nd_delta:.3e}")
    assert nd_delta == 0.0, "briques et formules DSL devraient etre bit-identiques (max|delta| != 0)"

    amr_delta = uniform_vs_amr(ne0, n_i0, L, args.steps, args.cfl, outdir)
    print(f"comparaison uniforme/AMR : max|delta| = {amr_delta:.3e}")

    provenance = {
        "script": "docs/sphinx/tutorials/diocotron_tutorial.py",
        "command": f"python diocotron_tutorial.py --n {args.n} --steps {args.steps}",
        "adc_cpp_sha": git_sha(REPO_ROOT),
        "backend_compile": backend,
        "backend_runtime": detect_backend_runtime(),
        "resolution": f"{args.n}x{args.n}",
        "steps": args.steps,
        "cfl": args.cfl,
        "python": sys.version.split()[0],
        "adc_module": adc.__file__,
        "assets": ["diocotron_growth.png", "diocotron_cover.png", "diocotron.gif",
                   "diocotron_native_vs_dsl.png", "diocotron_uniform_vs_amr.png"],
        "growth_factor": growth,
        "mass_drift": mass_drift,
        "native_dsl_max_delta": nd_delta,
        "amr_uniform_max_delta": amr_delta,
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"figures + provenance ecrites dans {outdir}")
    print("OK tutoriel diocotron")


if __name__ == "__main__":
    main()

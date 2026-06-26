#!/usr/bin/env python3
"""Canonical adc_cpp tutorial: reduced diocotron instability (E x B drift), end to end.

This script is the SOURCE of truth for the A->Z tutorial in the Sphinx documentation: the
`getting-started/tutorial` page includes it via `literalinclude` (the doc code is NOT copied by
hand). It is deliberately SELF-CONTAINED: it depends only on `pops` (the compiled module), `numpy`
and `matplotlib` -- no dependency on the `adc_cases` application package.

What it does, in tutorial order
--------------------------------
1. imports `pops` and detects the running backend;
2. WRITES THE MODEL AS FORMULAS with `pops.dsl.Model` (conservative variable, auxiliary fields
   phi/grad, E x B advection flux, eigenvalues, elliptic right-hand side);
3. COMPILES the model: tries the `production` backend (zero-copy native path, preferred) then
   falls back to `aot` (numerically identical, host-marshalled) -- exactly like the cases;
4. builds a periodic `pops.System`, picks the scheme (MUSCL minmod + Rusanov, explicit), wires the
   system Poisson (charge density, multigrid) and sets the initial condition;
5. integrates in time while capturing diagnostics (perturbation amplitude, mass);
6. produces the figures: a CURVE (amplitude vs time), a 2D MAP (final density), a GIF, and a
   uniform/AMR COMPARISON (`pops.AmrSystem` refined vs uniform grid);
7. writes a provenance record (adc_cpp SHA, backend, resolution, command) next to the figures, so
   that each asset is reproducible.

Physics (REDUCED model, not the full Euler-Poisson system)
----------------------------------------------------------
A single conservative variable, the density `n`, advected by the E x B drift
`v = (-d_y phi / B0, d_x phi / B0)` (divergence-free), where `phi` solves the system Poisson
`-lap phi = alpha (n - n_i0)` (neutralizing ionic background `n_i0`). Conventions anchored in the
core: `include/pops/physics/hyperbolic.hpp` (`ExBVelocity`) and `.../elliptic.hpp`
(`BackgroundDensity`). This is the diocotron NORMALIZATION benchmark, not a reproduction of the
full system (cf. `adc_cases/diocotron` and `docs/HOFFART_FIDELITY.md`).

Usage
-----
    python diocotron_tutorial.py [--n 96] [--steps 60] [--outdir _assets] [--quick]

`--quick` reduces the resolution and the number of steps for a fast smoke pass (doc CI).
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

import pops
from pops import dsl

# Physical parameters of the reduced model (must be consistent between the formulas and the Poisson RHS).
B0 = 1.0      # background magnetic field (carries the E x B drift)
ALPHA = 1.0   # factor of the elliptic right-hand side alpha (n - n_i0)

REPO_ROOT = Path(__file__).resolve().parents[3]   # tutorials -> sphinx -> docs -> repo root
POPS_INCLUDE = os.environ.get("POPS_INCLUDE", str(REPO_ROOT / "include"))


# --- 2. The model, WRITTEN AS FORMULAS (pops.dsl.Model) ---------------------------------------------
def diocotron_model(n_i0: float) -> "dsl.Model":
    """Reduced diocotron model as symbolic formulas, reproducing the native bricks `ExBVelocity`
    (transport) and `BackgroundDensity` (elliptic). `n_i0` = neutralizing ionic background (mean of
    the density, for the solvability of the periodic Poisson)."""
    m = dsl.Model("diocotron_tutorial")

    (n,) = m.conservative_vars("n")          # single conservative variable: the density (Density role)
    m.aux("phi")                             # auxiliary fields provided by the solver (pops::Aux channel)
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    vx = (-grad_y) / B0                       # E x B drift: v = (-d_y phi / B0, d_x phi / B0)
    vy = grad_x / B0
    m.flux(x=[n * vx], y=[n * vy])            # advection flux f = n v(dir)
    m.eigenvalues(x=[vx], y=[vy])             # spectrum: one wave, the drift speed

    m.primitive_vars(n=n)                     # transported scalar: primitive = conservative
    m.conservative_from([n])
    m.elliptic_rhs(ALPHA * (n - n_i0))        # couples the block to Poisson: rhs = alpha (n - n_i0)

    m.check()                                 # every referenced variable must be declared
    return m


# --- 2bis. The SAME model, composed of native bricks (the other writing front) ---------------------
def native_diocotron_model(n_i0: float):
    """The SAME diocotron, but composed of native bricks instead of formulas: Scalar (state) +
    ExB (E x B transport) + NoSource + BackgroundDensity (elliptic right-hand side alpha (n - n_i0)).
    This is the other way to write a model -- we prove below (`native_vs_dsl`) that it produces a
    BIT-IDENTICAL state to the formulas. C++ conventions: `ExBVelocity` and `BackgroundDensity`."""
    return pops.Model(state=pops.Scalar(),
                     transport=pops.ExB(B0=B0),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=ALPHA, n0=n_i0))


# --- 4. Initial condition (perturbed charge band, azimuthal mode) ----------------------------------
def band_density(n: int, L: float = 1.0, amp: float = 1.0, width: float = 0.05,
                 mode: int = 2, disp: float = 0.02, floor: float = 1.0) -> np.ndarray:
    """Horizontal charge band perturbed sinusoidally along x (convention `ne[j, i]`).

        ne(x, y) = floor + amp exp(-(y - y0)^2 / width^2),  y0 = 0.5 L + disp cos(2 pi mode x / L).
    """
    xs = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(xs, xs)                # indexing 'xy': X[j, i] = xs[i], Y[j, i] = xs[j]
    y0 = 0.5 * L + disp * np.cos(2.0 * np.pi * mode * X / L)
    ne = floor + amp * np.exp(-((Y - y0) ** 2) / (width ** 2))
    return np.ascontiguousarray(ne)


def perturbation_amplitude(density: np.ndarray) -> float:
    """L2 amplitude of the perturbation = deviation from the mean along x (the unperturbed band is
    uniform in x; what deviates from it carries the instability)."""
    base = density.mean(axis=1, keepdims=True)
    delta = density - base
    return float(np.sqrt(np.mean(delta * delta)))


# --- 3+4. Compilation + wiring: production (native) if possible, else aot (identical) --------------
def compile_and_build(model: "dsl.Model", ne0: np.ndarray, L: float, outdir: Path):
    """Compile the DSL model AND wire it into a periodic `pops.System`.

    Tries `production` first (zero-copy native path `add_native_block`, the plan's target), then
    falls back to `aot` (`add_compiled_block`, numerically identical, host-marshalled) -- exactly the
    cases' strategy (cf. `adc_cases/diocotron_dsl`). The native path requires the `_pops` module and the
    model `.so` to have BEEN COMPILED WITH THE SAME pops headers (ABI guard); a fresh module (built as in
    getting-started/installation) takes the native path, otherwise `aot` applies. Returns (sim, backend)."""
    last = None
    for backend in ("production", "aot"):
        try:
            compiled = model.compile(str(outdir / f"diocotron_{backend}.so"),
                                     POPS_INCLUDE, backend=backend)
            sim = pops.System(n=ne0.shape[0], L=L, periodic=True)
            sim.add_equation("ne", model=compiled,
                             spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"),
                             time=pops.Explicit())
            sim.set_poisson(rhs="charge_density", solver="geometric_mg")
            sim.set_density("ne", ne0)
            return sim, backend
        except Exception as exc:  # noqa: BLE001 - diagnostic: try the next backend
            last = exc
            print(f"  backend {backend!r} unavailable ({type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:80]}), trying the next one")
    # Both backends failed. On a fresh install the usual cause is a missing/incompatible Kokkos root
    # for the DSL .so (POPS_KOKKOS_ROOT). Delegate the diagnosis + copy-paste fixes to pops.doctor()
    # rather than re-implementing the environment checks here.
    print("\nNo DSL backend could be wired. Running pops.doctor() for a diagnosis:\n")
    try:
        pops.doctor()
    except Exception:  # noqa: BLE001 - doctor is best-effort here
        pass
    raise RuntimeError(
        "no DSL backend could be wired to the System; see the pops.doctor() output above "
        "(typically: set POPS_KOKKOS_ROOT, cf. getting-started/installation.md)") from last


def run(sim, steps: int, cfl: float, capture_every: int):
    """Integrate `steps` steps, capturing frames + the amplitude over time."""
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


# --- 6. Figures: curve, map, GIF, uniform/AMR comparison -------------------------------------------
def make_figures(frames, times, amps, ne0, L, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Curve (amplitude vs time) + 2D map (final density), side by side.
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].semilogy(times, amps, "o-", color="#b5179e")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("perturbation amplitude (L2)")
    ax[0].set_title("diocotron instability growth")
    im = ax[1].imshow(frames[-1].T, origin="lower", cmap="magma",
                      extent=[0, L, 0, L])
    ax[1].set_title("final density"); ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / "diocotron_growth.png", dpi=120)
    # PNG cover image (first GIF frame) for static exports.
    fig_cov, ax_cov = plt.subplots(figsize=(3.8, 3.6))
    ax_cov.imshow(frames[-1].T, origin="lower", cmap="magma", extent=[0, L, 0, L])
    ax_cov.set_title("diocotron (density)"); ax_cov.set_xlabel("x"); ax_cov.set_ylabel("y")
    fig_cov.tight_layout(); fig_cov.savefig(outdir / "diocotron_cover.png", dpi=120)
    plt.close(fig_cov)

    # GIF of the density evolution.
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


# --- 6bis. Uniform vs AMR comparison ---------------------------------------------------------------
def uniform_vs_amr(ne0, n_i0, L, steps, cfl, outdir: Path):
    """Replays the SAME physics on a uniform grid and on `pops.AmrSystem` (adaptive refinement), and
    plots both final densities side by side. We use the NATIVE brick composition for this comparison
    (both paths, uniform and AMR, share exactly the same model)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    su = pops.System(n=ne0.shape[0], L=L, periodic=True)
    su.add_block("ne", model=native_diocotron_model(n_i0), spatial=pops.Spatial(minmod=True),
                 time=pops.Explicit())
    su.set_poisson(rhs="charge_density", solver="geometric_mg"); su.set_density("ne", ne0)

    sa = pops.AmrSystem(n=ne0.shape[0], L=L, periodic=True)
    sa.add_block("ne", model=native_diocotron_model(n_i0), spatial=pops.Spatial(minmod=True),
                 time=pops.Explicit())
    sa.set_refinement(0.05)
    sa.set_poisson(rhs="charge_density", solver="geometric_mg"); sa.set_density("ne", ne0)

    for _ in range(steps):
        su.step_cfl(cfl); sa.step_cfl(cfl)
    du = np.asarray(su.density("ne")); da = np.asarray(sa.density("ne"))

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for a, d, t in ((ax[0], du, "uniform grid"), (ax[1], da, "AMR (refined)")):
        im = a.imshow(d.T, origin="lower", cmap="magma", extent=[0, L, 0, L])
        a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"diocotron: uniform vs AMR (max|delta| = {float(np.max(np.abs(da - du))):.2e})")
    fig.tight_layout(); fig.savefig(outdir / "diocotron_uniform_vs_amr.png", dpi=120)
    plt.close(fig)
    return float(np.max(np.abs(da - du)))


# --- 6ter. The same physics, two fronts: bricks == DSL (bit-identical) -----------------------------
def native_vs_dsl(dsl_final: np.ndarray, ne0, n_i0, L, steps, cfl, outdir: Path):
    """Replays the SAME physics with native BRICKS (`native_diocotron_model`) on the same grid / scheme
    / number of steps as the DSL run, and compares the final state to the FORMULAS. The two writing
    fronts (bricks and formulas) produce an IDENTICAL numerical kernel: the difference is zero to binary
    precision. Plots both maps side by side with `max|delta|` in the title, and returns that difference
    (asserted zero in `main`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sb = pops.System(n=ne0.shape[0], L=L, periodic=True)
    sb.add_block("ne", model=native_diocotron_model(n_i0), spatial=pops.Spatial(minmod=True),
                 time=pops.Explicit())
    sb.set_poisson(rhs="charge_density", solver="geometric_mg")
    sb.set_density("ne", ne0)
    for _ in range(steps):
        sb.step_cfl(cfl)
    native_final = np.asarray(sb.density("ne"))

    max_delta = float(np.max(np.abs(native_final - dsl_final)))
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for a, d, t in ((ax[0], native_final, "native bricks"), (ax[1], dsl_final, "formulas (DSL)")):
        im = a.imshow(d.T, origin="lower", cmap="magma", extent=[0, L, 0, L])
        a.set_title(t); a.set_xlabel("x"); a.set_ylabel("y"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"diocotron: same physics, two fronts (max|bricks - DSL| = {max_delta:.0e})")
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
    """Parallelism backend compiled into the module (cf. getting-started/backend)."""
    for attr in ("backend", "parallel_backend", "build_info"):
        val = getattr(pops, attr, None)
        if callable(val):
            try:
                return str(val())
            except Exception:  # noqa: BLE001
                pass
        elif val is not None:
            return str(val)
    return "serial (default; cf. getting-started/backend for Kokkos/MPI)"


def main() -> None:
    p = argparse.ArgumentParser(description="adc_cpp diocotron tutorial (A->Z, reproducible).")
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--outdir", default=str(Path(__file__).parent / "_assets"))
    p.add_argument("--quick", action="store_true", help="reduced resolution/steps (CI smoke)")
    args = p.parse_args()
    if args.quick:
        args.n, args.steps = 48, 20

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    L = 1.0

    print(f"pops imported from: {pops.__file__}")
    print(f"parallelism backend: {detect_backend_runtime()}")

    ne0 = band_density(args.n, L, amp=1.0, width=0.05, mode=2, disp=0.02)
    n_i0 = float(ne0.mean())

    t0 = time.time()
    sim, backend = compile_and_build(diocotron_model(n_i0), ne0, L, outdir)
    print(f"DSL model compiled + wired (backend = {backend!r}) in {time.time() - t0:.1f} s")

    capture_every = max(1, args.steps // 15)
    frames, times, amps = run(sim, args.steps, args.cfl, capture_every)

    growth = amps[-1] / amps[0]
    mass_drift = abs(float(sim.mass("ne")) - float(ne0.sum())) / float(ne0.sum())
    print(f"amplitude {amps[0]:.3e} -> {amps[-1]:.3e} (x{growth:.2f}); mass drift = {mass_drift:.2e}")
    assert growth > 1.0, "the diocotron instability did not grow"
    assert mass_drift < 1e-9, "mass not conserved (periodic advective transport)"

    make_figures(frames, times, amps, ne0, L, outdir)

    # The SAME physics as native bricks: we prove that the two writing fronts (DSL formulas and bricks)
    # produce a BIT-IDENTICAL state (same numerical kernel), then plot both maps.
    nd_delta = native_vs_dsl(frames[-1], ne0, n_i0, L, args.steps, args.cfl, outdir)
    print(f"bricks/DSL equivalence: max|delta| = {nd_delta:.3e}")
    assert nd_delta == 0.0, "bricks and DSL formulas should be bit-identical (max|delta| != 0)"

    amr_delta = uniform_vs_amr(ne0, n_i0, L, args.steps, args.cfl, outdir)
    print(f"uniform/AMR comparison: max|delta| = {amr_delta:.3e}")

    provenance = {
        "script": "docs/sphinx/tutorials/diocotron_tutorial.py",
        "command": f"python diocotron_tutorial.py --n {args.n} --steps {args.steps}",
        "pops_cpp_sha": git_sha(REPO_ROOT),
        "backend_compile": backend,
        "backend_runtime": detect_backend_runtime(),
        "resolution": f"{args.n}x{args.n}",
        "steps": args.steps,
        "cfl": args.cfl,
        "python": sys.version.split()[0],
        "pops_module": pops.__file__,
        "assets": ["diocotron_growth.png", "diocotron_cover.png", "diocotron.gif",
                   "diocotron_native_vs_dsl.png", "diocotron_uniform_vs_amr.png"],
        "growth_factor": growth,
        "mass_drift": mass_drift,
        "native_dsl_max_delta": nd_delta,
        "amr_uniform_max_delta": amr_delta,
    }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"figures + provenance written to {outdir}")
    print("OK diocotron tutorial")


if __name__ == "__main__":
    main()

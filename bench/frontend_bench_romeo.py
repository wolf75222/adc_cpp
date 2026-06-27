#!/usr/bin/env python3
import csv
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import pops
from pops.ir.ops import sqrt
from pops.physics.facade import Model

GAMMA = 1.4


def initial_state(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x)
    rho = 1.0 + 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    u = 0.2 + 0.05 * np.sin(2 * np.pi * Y)
    v = -0.1 + 0.05 * np.cos(2 * np.pi * X)
    p = np.ones_like(rho)
    U = np.zeros((4, n, n))
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * v
    U[3] = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return U


def dsl_model():
    m = Model("frontend_euler_periodic")
    rho, rhou, rhov, E = m.conservative_vars(
        "rho",
        "rho_u",
        "rho_v",
        "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"],
    )
    g = m.param("gamma", GAMMA)
    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(g * p / rho)
    m.flux(
        x=[rhou, rhou * u + p, rhou * v, rho * H * u],
        y=[rhov, rhov * u, rhov * v + p, rho * H * v],
    )
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from(
        [prho, prho * pu, prho * pv, pp / (g - 1.0) + 0.5 * prho * (pu * pu + pv * pv)]
    )
    m.elliptic_rhs(0.0 * rho)
    m.check()
    return m


def build_bricks(n, U):
    t0 = time.perf_counter()
    sim = pops.System(n=n, L=1.0, periodic=True)
    spec = pops.Model(
        state=pops.FluidState("compressible", gamma=GAMMA),
        transport=pops.CompressibleFlux(),
        source=pops.NoSource(),
        elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0),
    )
    sim.add_equation(
        "gas",
        spec,
        spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"),
    )
    sim.set_poisson(rhs="charge_density", solver="fft")
    sim.set_state("gas", U.reshape(-1))
    return sim, (time.perf_counter() - t0) * 1000.0


def build_dsl(n, U, cache_dir):
    old = os.environ.get("POPS_CACHE_DIR")
    os.environ["POPS_CACHE_DIR"] = cache_dir
    try:
        m = dsl_model()
        c0 = time.perf_counter()
        cm = m.compile(backend="production")
        compile_ms = (time.perf_counter() - c0) * 1000.0
        if cm.backend != "production" or cm.adder != "add_native_block":
            raise RuntimeError("bad DSL backend/adder: %s %s" % (cm.backend, cm.adder))
        t0 = time.perf_counter()
        sim = pops.System(n=n, L=1.0, periodic=True)
        sim.add_equation(
            "gas",
            cm,
            spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"),
        )
        sim.set_poisson(rhs="charge_density", solver="fft")
        sim.set_state("gas", U.reshape(-1))
        setup_ms = (time.perf_counter() - t0) * 1000.0
        return sim, setup_ms, compile_ms
    finally:
        if old is None:
            os.environ.pop("POPS_CACHE_DIR", None)
        else:
            os.environ["POPS_CACHE_DIR"] = old


def time_sim(sim, dt, steps, warmup):
    for _ in range(warmup):
        sim.step(dt)
    t0 = time.perf_counter()
    sim.advance(dt, steps)
    advance_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    U = np.asarray(sim.get_state("gas"))
    extract_ms = (time.perf_counter() - t1) * 1000.0
    return advance_ms, extract_ms, float(U[: U.size // 4].sum())


def time_step_loop(build_fn, dt, steps, warmup):
    sim, _setup_ms, _compile_ms = build_fn()
    for _ in range(warmup):
        sim.step(dt)
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step(dt)
    return (time.perf_counter() - t0) * 1000.0


def write_row(
    path,
    commit,
    frontend,
    cache_state,
    setup_ms,
    compile_ms,
    advance_ms,
    step_loop_ms,
    extract_ms,
    notes,
):
    total = setup_ms + compile_ms + advance_ms + extract_ms
    with open(path, "a", newline="") as fh:
        csv.writer(fh).writerow(
            [
                commit,
                "frontend-euler-periodic",
                frontend,
                "kokkos-openmp",
                128,
                40,
                "1e-4",
                cache_state,
                setup_ms,
                compile_ms,
                advance_ms,
                step_loop_ms,
                extract_ms,
                total,
                "",
                notes,
            ]
        )


def main():
    out, commit, thr = sys.argv[1], sys.argv[2], sys.argv[3]
    n, steps, warmup, dt = 128, 40, 5, 1e-4
    U = initial_state(n)

    sim, setup_ms = build_bricks(n, U)
    adv, ext, mass = time_sim(sim, dt, steps, warmup)
    loop_ms = time_step_loop(lambda: (*build_bricks(n, U), 0.0), dt, steps, warmup)
    write_row(
        out,
        commit,
        "python-bricks",
        "n/a",
        setup_ms,
        0.0,
        adv,
        loop_ms,
        ext,
        "threads=%s mass=%.17e" % (thr, mass),
    )

    cache = tempfile.mkdtemp(prefix="pops_dsl_cache_")
    try:
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache, exist_ok=True)

        sim, setup_ms, compile_ms = build_dsl(n, U, cache)
        adv, ext, mass = time_sim(sim, dt, steps, warmup)
        loop_ms = time_step_loop(lambda: build_dsl(n, U, cache), dt, steps, warmup)
        write_row(
            out,
            commit,
            "python-dsl-production",
            "cold",
            setup_ms,
            compile_ms,
            adv,
            loop_ms,
            ext,
            "threads=%s mass=%.17e" % (thr, mass),
        )

        sim, setup_ms, compile_ms = build_dsl(n, U, cache)
        adv, ext, mass = time_sim(sim, dt, steps, warmup)
        loop_ms = time_step_loop(lambda: build_dsl(n, U, cache), dt, steps, warmup)
        write_row(
            out,
            commit,
            "python-dsl-production",
            "warm",
            setup_ms,
            compile_ms,
            adv,
            loop_ms,
            ext,
            "threads=%s mass=%.17e" % (thr, mass),
        )
    finally:
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()

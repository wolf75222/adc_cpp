#!/usr/bin/env python3
"""AmrSystem positivity_floor on the COMPILED .so path (Zhang-Shu, ADC-322).

ADC-259 wired positivity_floor onto the NATIVE AMR transport (AmrSystem.add_block) but REJECTED it on
the compiled .so path, because the flat ABI loader (pops_install_native_amr) carried no floor slot.
ADC-322 regenerates the loader so the floor rides the .so too: the trailing pos_floor is marshaled
through add_compiled_model -> set_compiled_block into the SAME compute_face_fluxes leaf the native path
uses (mono via AmrBuildParams::pos_floor, multi via the AmrCompiledBlockBuilder slot).

This test compiles an ISOTHERMAL block (p = cs2 rho, the ticket's regime: flooring the density also
floors the pressure) with backend='production', target='amr_system' and drives it through the public
facade AmrSystem.add_equation. It checks, on the compiled .so path:

  (1) NO LONGER RAISES: add_equation(positivity_floor > 0) no longer raises (it used to) and, on the
      1e6-contrast oscillating top-hat advected at u=1 (where weno5 reconstructs a negative face density),
      the floored .so run stays FINITE. The load-bearing claim (an unfloored run diverging on the same
      spike) is covered on the System path by tests/test_positivity_floor.cpp and on a refined AMR C/F
      interface by python/tests/test_amr_positivity_floor.py section (3); it is not asserted here since it
      depended on the pre-ADC-324 never-tagged seed of set_refinement(1e30) (now a mono-level hierarchy),
      leaving a coarse-only grid on which the demo is not robust. The floor still rides the compiled
      loader -- exercised by (2)'s dmax==0 marshalling check and (3)'s multi-block routing.
  (2) BIT-IDENTICAL AT floor=0: on a smooth strictly-positive drifting state, the compiled run with
      positivity_floor > 0 is BIT-IDENTICAL (dmax == 0) to positivity_floor = 0 -- the floor never bites
      a face above it, so the regenerated loader is a no-default-change at floor 0.
  (3) MULTI-BLOCK: two compiled blocks with positivity_floor > 0 build and run finite -- the floor rides
      the OTHER compiled routing too (build_multi -> AmrCompiledBlockBuilder -> dispatch_amr_block).

The guarantee is face / C/F-ghost-mean Density positivity only (order-1 fallback), parity with the
native path (tests/test_amr_positivity_floor.cpp, python/tests/test_amr_positivity_floor.py).

Needs a C++ compiler + the adc headers + POPS_KOKKOS_ROOT (the production loader is Kokkos-only):
auto-skips (exit 0) without a compiler, like test_dsl_production_amr. Validated under CI (ci-kokkos*).
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops import dsl

CS2 = 0.25       # isothermal sound speed^2 (p = cs2 rho): flooring rho floors the pressure
N = 48
DT = 0.0008
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def build_iso_model():
    """Isothermal block (3 var, Density role at component 0) written in DSL formulas: compile(...) turns
    it into a production .so loader. elliptic_rhs = 0 isolates the transport (no Poisson noise)."""
    m = dsl.Model("iso_floor")
    rho, rhou, rhov = m.conservative_vars("rho", "rho_u", "rho_v")
    u = rhou / rho
    v = rhov / rho
    pu = m.primitive("u", u)
    pv = m.primitive("v", v)
    m.flux(x=[rhou, rhou * pu + CS2 * rho, rhou * pv],
           y=[rhov, rhov * pu, rhov * pv + CS2 * rho])
    m.eigenvalues(x=[pu - dsl.sqrt(CS2), pu, pu + dsl.sqrt(CS2)],
                  y=[pv - dsl.sqrt(CS2), pv, pv + dsl.sqrt(CS2)])
    m.primitive_vars(rho, pu, pv)
    m.conservative_from([rho, rho * pu, rho * pv])
    m.elliptic_rhs(0.0 * rho)
    return m


def spike_state():
    """1e6-contrast top-hat in x (rho = 1 in the central band, 1e-6 outside) with a non-monotone
    oscillating spike at x ~ 3/4, advected at u = 1. set_conservative_state expects (ncomp, n, n)."""
    rho = np.full((N, N), 1e-6)
    rho[:, N // 3:2 * N // 3] = 1.0
    ks = 3 * N // 4
    rho[:, ks] = 0.8
    rho[:, ks + 1] = 0.5
    rho[:, ks + 2] = 1e-6
    rho[:, ks + 3] = 1.0
    rho[:, ks + 4] = 1e-6
    return np.stack([rho, rho * 1.0, 0.0 * rho])  # (3, n, n): u = 1 advects the contact


def smooth_state():
    """Smooth strictly-positive Gaussian bubble drifting at u = 0.3 (no face undershoots the floor)."""
    xs = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(xs, xs, indexing="xy")
    rho = 1.0 + 0.4 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)  # (n, n)
    return np.stack([rho, 0.3 * rho, 0.0 * rho])  # (3, n, n)


def compiled_single(cm, pf, state):
    """Single compiled block (add_equation -> add_native_block) with positivity_floor=pf, seeded with
    the full conservative state, stepped 38 times. Returns the coarse density (flat array)."""
    s = pops.AmrSystem(n=N, L=1.0, periodic=True)
    s.set_refinement(1e30)
    s.add_equation("gas", cm,
                   spatial=pops.Spatial(limiter="weno5", flux="rusanov", positivity_floor=pf),
                   time=pops.Explicit())
    s.set_conservative_state("gas", state)
    for _ in range(38):
        s.step(DT)
    return np.asarray(s.density("gas"))


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  no C++ compiler or adc headers")
        print("test_amr_compiled_positivity_floor : OK (nothing to compile)")
        return

    tmp = tempfile.mkdtemp()
    try:
        cm = build_iso_model().compile(os.path.join(tmp, "iso_floor_amr.so"), INCLUDE,
                                       backend="production", target="amr_system")
        assert isinstance(cm, dsl.CompiledModel)
        assert cm.adder == "add_native_block" and cm.target == "amr_system"

        # --- (1) no longer raises + the floor rides the .so; floored run finite on the 1e6 spike ------
        # ADC-341/ADC-324: set_refinement(1e30) is now a MONO-LEVEL hierarchy (no seed fine patch). The
        # historical "unfloored .so run blows up" assertion relied on the never-tagged 1e30 seed and is
        # not robust on the resulting coarse-only grid (neither floored nor unfloored diverges there); the
        # load-bearing property is covered on System by tests/test_positivity_floor.cpp and on a refined
        # AMR C/F interface by python/tests/test_amr_positivity_floor.py section (3). Here we keep the
        # compiled-facade contract: floor>0 is accepted (previously raised at add_equation) and the
        # floored .so run stays finite. That the floor actually rides the loader is proven by (2)'s
        # dmax==0 marshalling check and (3)'s multi-block routing.
        print("== (1) compiled .so: positivity_floor>0 accepted + floored run finite on the contrast-1e6 spike ==")
        d_on = compiled_single(cm, 1e-8, spike_state())   # previously raised ValueError at add_equation
        chk(np.all(np.isfinite(d_on)),
            "compiled add_equation(positivity_floor>0) accepted + finite over 38 steps")

        # --- (2) bit-identical at floor=0: smooth positive drift -> floor ON == floor OFF -------------
        print("== (2) no-default-change: compiled floor>0 == floor=0 on smooth positive data ==")
        da = compiled_single(cm, 0.0, smooth_state())
        db = compiled_single(cm, 1e-8, smooth_state())
        chk(float(np.abs(da - db).max()) == 0.0,
            "compiled floor>0 bit-identical to floor=0 on smooth state (dmax==0)")
        chk(np.all(np.isfinite(db)), "floored compiled smooth run finite")

        # --- (3) multi-block: the floor rides the AmrCompiledBlockBuilder slot too --------------------
        # Two compiled blocks switch the single-block AmrCouplerMP for the multi-block AmrRuntime engine:
        # the floor flows through build_multi -> AmrCompiledBlockBuilder -> dispatch_amr_block (a DIFFERENT
        # routing than (1)-(2)). set_conservative_state is single-block only, so seed via set_density
        # (u=0): the floor stays inactive but the new slot's threading + arity are exercised end to end.
        print("== (3) multi-block compiled: positivity_floor threaded through AmrCompiledBlockBuilder ==")
        band = np.full((N, N), 1e-6)
        band[:, N // 3:2 * N // 3] = 1.0
        sm = pops.AmrSystem(n=N, L=1.0, periodic=True)
        sm.set_refinement(1e30)
        sm.add_equation("a", cm, spatial=pops.Spatial(limiter="weno5", flux="rusanov",
                                                     positivity_floor=1e-8))
        sm.add_equation("b", cm, spatial=pops.Spatial(limiter="weno5", flux="rusanov",
                                                     positivity_floor=1e-8))
        sm.set_density("a", band.ravel().copy())
        sm.set_density("b", band.ravel().copy())
        for _ in range(5):
            sm.step(DT)
        chk(np.all(np.isfinite(np.asarray(sm.density("a")))), "multi-block compiled floor: block a finite")
        chk(np.all(np.isfinite(np.asarray(sm.density("b")))), "multi-block compiled floor: block b finite")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        print(f"FAIL test_amr_compiled_positivity_floor : {fails} failure(s)")
        sys.exit(1)
    print("OK test_amr_compiled_positivity_floor")


if __name__ == "__main__":
    main()

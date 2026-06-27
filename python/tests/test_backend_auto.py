#!/usr/bin/env python3
"""Backend par DEFAUT 'auto' (ADC-63) : production quand la parite toolchain avec le module _pops
est etablie, aot sinon -- et JAMAIS un choix muet (CompiledModel.backend dit ce qui a ete
construit, backend_auto_reason dit pourquoi).

  (1) parite OK (module charge + en-tetes du depot == build) : compile() sans backend ->
      backend == 'production', le bloc se branche par add_equation et tourne ;
  (2) parite CASSEE (en-tetes copies puis modifies -> signature differente) : auto -> 'aot'
      avec une raison explicite, et le bloc aot tourne aussi (fallback FONCTIONNEL) ;
  (3) backend EXPLICITE : inchange ('aot' demande = aot meme si la parite production existe).

Invariants par assert ; imprime "OK test_backend_auto" en cas de succes.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops.codegen.toolchain import resolve_auto_backend
from pops.ir.ops import sqrt
from pops.physics.facade import Model

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso3(name):
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    c = sqrt(0.5)
    m.flux(x=[mx, mx * u + 0.5 * rho, mx * v], y=[my, my * u, my * v + 0.5 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    return m


cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip test_backend_auto : compilateur ou en-tetes pops absents")
    sys.exit(0)

tmp = tempfile.mkdtemp()
try:
    print("== (1) auto -> production quand la parite toolchain est etablie ==")
    bk, reason = resolve_auto_backend(INCLUDE)
    chk(bk == "production", f"resolve_auto_backend = production (raison : {reason[:60]})")
    cm = iso3("auto_prod").compile(os.path.join(tmp, "auto_prod.so"), INCLUDE)
    chk(cm.backend == "production", f"compile() sans backend -> {cm.backend!r}")
    chk(cm.backend_auto_reason is not None and "toolchain" in cm.backend_auto_reason,
        f"raison posee : {str(cm.backend_auto_reason)[:60]}")
    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.set_poisson()
    sim.add_equation("f", model=cm, spatial=pops.FiniteVolume(limiter="minmod"),
                     time=pops.Explicit())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    z = np.zeros((n, n))
    sim.set_primitive_state("f", rho=1.0 + 0.3 * np.exp(-40 * ((X - .5) ** 2 + (Y - .5) ** 2)),
                            u=z, v=z)
    sim.step_cfl(0.3)
    chk(np.all(np.isfinite(np.asarray(sim.density("f")))), "bloc production (auto) tourne fini")

    print("== (2) parite cassee -> repli aot EXPLIQUE et fonctionnel ==")
    stale = os.path.join(tmp, "include_stale")
    shutil.copytree(INCLUDE, stale)
    with open(os.path.join(stale, "pops", "core", "types.hpp"), "a") as f:
        f.write("\n// drift de signature pour test_backend_auto (copie jetable)\n")
    bk2, reason2 = resolve_auto_backend(stale)
    chk(bk2 == "aot" and "module" in reason2 or "en-tetes" in reason2,
        f"resolve(stale) = {bk2} ({reason2[:60]})")
    cm2 = iso3("auto_aot").compile(os.path.join(tmp, "auto_aot.so"), stale)
    chk(cm2.backend == "aot", f"auto avec en-tetes derivants -> {cm2.backend!r}")
    chk(cm2.backend_auto_reason is not None, "raison du repli posee")

    print("== (3) backend explicite inchange ==")
    cm3 = iso3("explicit_aot").compile(os.path.join(tmp, "explicit_aot.so"), INCLUDE,
                                       backend="aot")
    chk(cm3.backend == "aot" and cm3.backend_auto_reason is None,
        "backend='aot' explicite : aot, sans raison auto (politique court-circuitee)")
    try:
        iso3("bad").compile(os.path.join(tmp, "bad.so"), INCLUDE, backend="bogus")
        chk(False, "backend inconnu aurait du lever")
    except ValueError as e:
        chk("auto" in str(e), f"backend inconnu rejete, message cite 'auto' : {str(e)[:60]}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_backend_auto : {fails} echec(s)")
    sys.exit(1)
print("OK test_backend_auto")

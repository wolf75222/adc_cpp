#!/usr/bin/env python3
"""m.enable_roe() (solde de l'audit, GENERICITY point 11) : la capability ROE est EMISE par le
DSL depuis les ROLES, et le solveur Roe-like generique du coeur (HasRoeDissipation) devient
disponible hors Euler 4 variables.

  (1) PARITE 4-var : un Euler DSL (roles + 'p' + enable_roe) avance avec riemann='roe' comme le
      bloc NATIF add_block(compressible, riemann='roe') -- l'emission transcrit l'algebre
      canonique du coeur (memes moyennes de Roe, meme gamma-1 deduit, meme entropy fix) ;
  (2) 3-var isotherme + enable_roe : riemann='roe' ACCEPTE (capability), pas finis, et un
      cisaillement stationnaire (rho const, u=0, v(x)) est preserve EXACTEMENT (la dissipation
      de Roe s'annule onde par onde : dp=du_n=drho=0 et |u_n|=0 sur l'onde de cisaillement) ;
  (3) rejets explicites : 3-var SANS capability -> ValueError ; enable_roe sans 'p' ou sans
      roles -> erreur a l'emission ; has_roe expose par CompiledModel.

Invariants par assert ; imprime "OK test_dsl_roe" en cas de succes.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops import dsl

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
GAMMA = 1.4


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def gaussian(n, amp=0.4):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + amp * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


def euler4_dsl(name, roe=False):
    m = dsl.Model(name)
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", GAMMA)
    u = rhou / rho
    v = rhov / rho
    p = (g - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    H = (E + p) / rho
    c = dsl.sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
           y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([prho, prho * pu, prho * pv,
                         pp / (g - 1.0) + 0.5 * prho * (pu * pu + pv * pv)])
    m.elliptic_rhs(0.0 * rho)
    if roe:
        m.enable_roe()
    return m


def iso3_dsl(name, roe=False, p_decl=True):
    m = dsl.Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    cs2 = 0.5
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    if p_decl:
        m.primitive("p", cs2 * rho)
    c = dsl.sqrt(cs2)
    m.flux(x=[mx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    if roe:
        m.enable_roe()
    return m


cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip test_dsl_roe : compilateur ou en-tetes pops absents")
    sys.exit(0)

tmp = tempfile.mkdtemp()
try:
    n = 24
    print("== (1) parite 4-var : DSL enable_roe == bloc natif compressible riemann='roe' ==")
    cm = euler4_dsl("euler_roe", roe=True).compile(os.path.join(tmp, "euler_roe.so"), INCLUDE,
                                                   backend="production")
    chk(getattr(cm, "has_roe", False), "CompiledModel.has_roe = True (capability emise)")
    rho0 = gaussian(n)
    z = np.zeros((n, n))
    p0 = 1.0 + 0.0 * rho0

    sd = pops.System(n=n, L=1.0, periodic=True)
    sd.set_poisson()
    sd.add_equation("gas", model=cm, spatial=pops.FiniteVolume(limiter="minmod", riemann="roe"),
                    time=pops.Explicit())
    sd.set_primitive_state("gas", rho=rho0, u=z + 0.1, v=z, p=p0)

    sn = pops.System(n=n, L=1.0, periodic=True)
    sn.set_poisson()
    sn.add_block("gas",
                 pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                           transport=pops.CompressibleFlux(), source=pops.NoSource(),
                           elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0)),
                 spatial=pops.FiniteVolume(limiter="minmod", riemann="roe"))
    sn.set_primitive_state("gas", rho=rho0, u=z + 0.1, v=z, p=p0)

    for _ in range(8):
        sd.step(2e-4)
        sn.step(2e-4)
    dd, dn = np.asarray(sd.density("gas")), np.asarray(sn.density("gas"))
    err = float(np.max(np.abs(dd - dn))) / float(np.max(np.abs(dn)))
    chk(err < 1e-12, f"8 pas roe : DSL generique == natif canonique (ecart rel {err:.2e})")
    chk(np.all(np.isfinite(dd)), "etat fini")

    print("== (2) 3-var isotherme + enable_roe : accepte, fini, cisaillement EXACT ==")
    cm3 = iso3_dsl("iso3_roe", roe=True).compile(os.path.join(tmp, "iso3_roe.so"), INCLUDE,
                                                 backend="production")
    s3 = pops.System(n=n, L=1.0, periodic=True)
    s3.set_poisson()
    s3.add_equation("f", model=cm3, spatial=pops.FiniteVolume(limiter="minmod", riemann="roe"),
                    time=pops.Explicit())
    x = (np.arange(n) + 0.5) / n
    vshear = np.tile(0.3 * np.sin(2 * np.pi * x), (n, 1))
    s3.set_primitive_state("f", rho=1.0 + z, u=z, v=vshear)
    before = np.asarray(s3.get_state("f")).copy()
    for _ in range(6):
        s3.step_cfl(0.3)
    after = np.asarray(s3.get_state("f"))
    dmax = float(np.max(np.abs(after - before)))
    chk(dmax == 0.0, f"cisaillement stationnaire preserve EXACTEMENT (dmax={dmax:.1e})")

    print("== (3) rejets explicites + garde-fous d'emission ==")
    cm_no = iso3_dsl("iso3_noroe").compile(os.path.join(tmp, "iso3_noroe.so"), INCLUDE,
                                           backend="production")
    try:
        s = pops.System(n=16, L=1.0, periodic=True)
        s.add_equation("f", model=cm_no, spatial=pops.FiniteVolume(limiter="minmod",
                                                                  riemann="roe"))
        chk(False, "roe sans capability sur 3-var aurait du lever")
    except (ValueError, RuntimeError) as e:
        chk("roe" in str(e), f"rejet sans capability : {str(e)[:70]}")
    try:
        iso3_dsl("iso3_nop", roe=True, p_decl=False).compile(
            os.path.join(tmp, "iso3_nop.so"), INCLUDE, backend="production")
        chk(False, "enable_roe sans primitive 'p' aurait du lever a l'emission")
    except ValueError as e:
        chk("'p'" in str(e) or "pression" in str(e), f"sans 'p' rejete : {str(e)[:70]}")
    try:
        mm = dsl.Model("noroles")
        a, b, c_ = mm.conservative_vars("a", "b", "c")
        mm.primitive("p", a)
        mm.flux(x=[b, a, c_], y=[c_, a, b])
        mm.eigenvalues(x=[a, a, a], y=[a, a, a])
        mm.primitive_vars(a, b, c_)
        mm.conservative_from([a, b, c_])
        mm.enable_roe()
        mm.compile(os.path.join(tmp, "noroles.so"), INCLUDE, backend="production")
        chk(False, "enable_roe sans roles fluides aurait du lever a l'emission")
    except ValueError as e:
        chk("roles" in str(e), f"sans roles rejete : {str(e)[:70]}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_dsl_roe : {fails} echec(s)")
    sys.exit(1)
print("OK test_dsl_roe")

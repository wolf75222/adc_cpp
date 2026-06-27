#!/usr/bin/env python3
"""Methode temporelle explicite "euler" (ForwardEuler, ordre 1) -- ADC-174.

Motivation : fidelite aux references au premier ordre (RIEMOM2D : split dimensionnel
additif + Euler == Euler non-splitte, algebriquement Mx+My-M = M+dt(Lx+Ly)) ; le seul
ecart de schema d'un replay vs ces references est l'etage 2 de ssprk2. "euler" est un
mode VALIDATION : ssprk2 reste le defaut (no-default-change verifie ici).

On verifie :
 (1) facade : Explicit() -> kind 'explicit' (defaut intact) ; Explicit(method='euler') ->
     kind 'euler' ; methode inconnue rejetee avec la liste a jour ;
 (2) IDENTITE DE SHU-OSHER, bit-exacte : un pas ssprk2 == 0.5 U0 + 0.5 euler(euler(U0)) --
     plus fort qu'un test d'ordre : prouve que 'euler' est EXACTEMENT l'operateur d'etage
     de ssprk2 (memes rhs, memes ghosts), donc ordre 1 par construction ;
 (3) no-default-change : un pas Explicit() == un pas Explicit(method='ssprk2') bit-exact ;
 (4) garde de discrimination : euler != ssprk2 sur le meme pas (le test (2) ne compare pas
     deux choses egales par accident) ;
 (5) AmrSystem : time='euler' rejete par le perimetre AMR existant (explicit|ssprk3|imex),
     message explicite -- pas d'ignore silencieux ;
 (6) [compilateur] chemins .so : backend='production' (add_native_block, gabarit
     add_compiled_model -> make_block) porte euler -- meme identite de Shu-Osher ; le chemin
     AOT (add_compiled_block, ABI extern C figee sur SSPRK2) le REJETTE explicitement en
     pointant production/natif (pas d'ignore silencieux).

Modele natif pur transport (isotherme sans source ni Poisson) pour (1)-(5) : aucun
compilateur requis ; (6) s'auto-saute sans compilateur ou sans Kokkos.
"""
import os
import sys
import tempfile

import numpy as np

import pops
from pops.codegen.toolchain import _default_cxx
from pops.physics.facade import Model

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn(); return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def transport_model():
    # NoSource + brique elliptique inerte (jamais resolue : pas de set_poisson) : l'avance
    # est un PUR transport, condition de l'identite de Shu-Osher du test (2).
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                     transport=pops.IsothermalFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


def make_sim(method):
    n = 24
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=pops.Explicit(method=method))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


print("== (1) facade ==")
chk(pops.Explicit().kind == "explicit", "Explicit() -> kind 'explicit' (defaut intact)")
chk(pops.Explicit(method="euler").kind == "euler", "Explicit(method='euler') -> kind 'euler'")
msg = err_msg(lambda: pops.Explicit(method="rk4"))
chk("'euler'" in msg, f"methode inconnue rejetee, liste a jour ({msg[:48]}...)")

print("== (2) identite de Shu-Osher : ssprk2 == 0.5 U0 + 0.5 euler(euler(.)) ==")
dt = 2e-3
s2 = make_sim("ssprk2")
se = make_sim("euler")
U0 = np.array(se.get_state("ions"))
s2.step(dt)
se.step(dt)
se.step(dt)
ref = 0.5 * U0 + 0.5 * np.array(se.get_state("ions"))
got = np.array(s2.get_state("ions"))
bit = np.array_equal(got, ref)
emax = np.abs(got - ref).max()
chk(bit or emax < 1e-15, f"identite bit-exacte (array_equal={bit}, err max {emax:.1e})")
if not bit:
    print("      NOTE : pas bit-exact mais < 1e-15 -- ordre des flottants de lincomb")

print("== (3) no-default-change : defaut == ssprk2 ==")
sd = make_sim("ssprk2")
s_def = pops.System(n=24, L=1.0, periodic=True)
s_def.add_block("ions", transport_model(),
                spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                time=pops.Explicit())
s_def.set_state("ions", np.array(sd.get_state("ions")))
sd.step(dt)
s_def.step(dt)
chk(np.array_equal(np.array(sd.get_state("ions")), np.array(s_def.get_state("ions"))),
    "Explicit() et Explicit(method='ssprk2') bit-identiques")

print("== (4) garde de discrimination ==")
s2b = make_sim("ssprk2")
seb = make_sim("euler")
s2b.step(dt)
seb.step(dt)
d = np.abs(np.array(s2b.get_state("ions")) - np.array(seb.get_state("ions"))).max()
chk(d > 1e-8, f"euler != ssprk2 sur un pas (ecart max {d:.2e})")

print("== (5) AMR : rejet explicite ==")
amr = pops.AmrSystem(n=32, L=1.0, periodic=True, regrid_every=0)
msg = err_msg(lambda: amr.add_block("ions", transport_model(),
                                    spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                                    time=pops.Explicit(method="euler")))
chk("euler" in msg or "explicit" in msg, f"AmrSystem rejette time='euler' ({msg[:60]}...)")

cxx = _default_cxx(None)
if not cxx:
    print("pas de compilateur C++ : test (6) saute")
    print("FAILS =", fails)
    sys.exit(1 if fails else 0)

print("== (6) chemins .so : production porte euler, AOT rejette ==")
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def adv_model():
    m = Model("eulprod")
    q1, q2 = m.conservative_vars("q1", "q2")
    m.flux(x=[1.5 * q1, -0.7 * q2], y=[-0.7 * q1, 1.5 * q2])
    z = 0.0 * q1  # les listes d'eigenvalues attendent des Expr (pas des flottants nus)
    m.eigenvalues(x=[z - 1.5, z + 1.5], y=[z - 1.5, z + 1.5])
    m.primitive_vars(q1, q2)
    m.conservative_from([q1, q2])
    return m


tmp = tempfile.mkdtemp(prefix="pops_euler_")
try:
    prod = adv_model().compile(os.path.join(tmp, "eulprod.so"), INCLUDE, backend="production")
except RuntimeError as ex:
    if "Kokkos" in str(ex):
        print("Kokkos introuvable : test (6) saute --", str(ex)[:60])
        print("FAILS =", fails)
        sys.exit(1 if fails else 0)
    raise


def make_prod_sim(method):
    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.add_equation("q", model=prod,
                     spatial=pops.FiniteVolume(limiter="none", riemann="rusanov"),
                     time=pops.Explicit(method=method))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    sim.set_state("q", np.stack([1.0 + 0.3 * np.sin(2 * np.pi * X),
                                 1.0 + 0.2 * np.cos(2 * np.pi * Y)]))
    return sim


p2 = make_prod_sim("ssprk2")
pe = make_prod_sim("euler")
U0p = np.array(pe.get_state("q"))
p2.step(dt)
pe.step(dt)
pe.step(dt)
refp = 0.5 * U0p + 0.5 * np.array(pe.get_state("q"))
gotp = np.array(p2.get_state("q"))
ep = np.abs(gotp - refp).max()
chk(np.array_equal(gotp, refp) or ep < 1e-15,
    f"production : identite de Shu-Osher (err max {ep:.1e})")

aot = adv_model().compile(os.path.join(tmp, "eulaot.so"), INCLUDE, backend="aot")
sim_a = pops.System(n=16, L=1.0, periodic=True)
msg = err_msg(lambda: sim_a.add_equation("q", model=aot,
                                         spatial=pops.FiniteVolume(limiter="none",
                                                                  riemann="rusanov"),
                                         time=pops.Explicit(method="euler")))
chk("production" in msg and "SSPRK2" in msg,
    f"AOT : euler rejete en pointant production/natif ({msg[:60]}...)")

print("FAILS =", fails)
sys.exit(1 if fails else 0)

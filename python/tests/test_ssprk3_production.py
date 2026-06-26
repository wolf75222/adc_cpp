#!/usr/bin/env python3
"""SSPRK3 expose sur le chemin de PRODUCTION (loader natif .so), pas seulement add_block.

Avant ce chantier, le schema RK explicite n'etait selectionnable que sur le chemin natif
add_block (modele compose pops.Model) : le chemin compile/production (add_native_block ->
pops_install_native -> add_compiled_model<ProdModel>) ne marshalait que "explicit"|"imex" et
RETOMBAIT SILENCIEUSEMENT sur SSPRK2 -- add_native_block rejetait meme "ssprk3". Le cas hoffart
(arXiv:2510.11808), qui compile en backend="production", restait donc bloque en SSPRK2.

Ce chantier threade le schema RK jusqu'au make_block du .so :
  - native_loader.hpp add_native_block : accepte time == "ssprk3" (au lieu de le rejeter) ;
  - dsl_block.hpp add_compiled_model : derive method = (time=="ssprk3")?"ssprk3":"ssprk2" et le
    passe a make_block (qui supportait deja AdvanceExplicit<..., SSPRK3Step>).

On verifie :
 (1) GARDE-FOU (sans compilateur) : le raw _s.add_native_block(..., time="ssprk3") sur un .so
     INEXISTANT echoue au dlopen -- PAS sur un rejet de schema temporel ; une chaine inconnue
     ("rk4") reste rejetee par la garde amont. Regression directe de la validation amont.
 (2) PARITE (avec compilateur) : un bloc production+SSPRK3 est BIT-IDENTIQUE au bloc natif
     add_block+SSPRK3 (meme make_block -> meme AdvanceExplicit<SSPRK3Step>), sur eval_rhs et apres
     plusieurs pas a dt fixe.
 (3) NON-TRIVIALITE : production+SSPRK3 DIFFERE de production+SSPRK2 -- preuve que ssprk3 est bien
     selectionne (pas un ssprk2 silencieux).
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from test_dsl_coupled import build_euler_poisson, GAMMA, INCLUDE

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn()
        return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def _native_spec():
    """Le MEME modele euler_poisson, version NATIVE composee par briques (reference de parite)."""
    return pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                     transport=pops.CompressibleFlux(),
                     source=pops.GravityForce(),
                     elliptic=pops.GravityCoupling(sign=-1.0, four_pi_G=1.0, rho0=1.0))


def _initial_state(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((4, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    U[3] = 1.0 / (GAMMA - 1.0)
    return U


# --- (1) GARDE-FOU amont : ssprk3 ACCEPTE par add_native_block (echec au dlopen, pas un rejet) -----
print("== (1) add_native_block(time='ssprk3') : accepte (dlopen fail), 'rk4' rejete ==")
ss = pops.System(n=16)._s  # facade compilee brute, pour viser add_native_block directement
msg_ok = err_msg(lambda: ss.add_native_block("x", "/inexistant.so", limiter="minmod",
                                             riemann="rusanov", recon="conservative", time="ssprk3"))
chk(msg_ok != "" and "dlopen" in msg_ok and "ssprk3' | 'imex'" not in msg_ok and "explicit' | 'imex'"
    not in msg_ok,
    "time='ssprk3' passe la garde schema -> echec au dlopen (pas un rejet de schema temporel)")
msg_bad = err_msg(lambda: ss.add_native_block("x", "/inexistant.so", limiter="minmod",
                                              riemann="rusanov", recon="conservative", time="rk4"))
chk(msg_bad != "" and ("explicit'" in msg_bad and "imex'" in msg_bad),
    "time='rk4' (inconnu) reste rejete par la garde amont (pas de dlopen)")

# --- (2)/(3) PARITE + NON-TRIVIALITE (necessite un compilateur + en-tetes pops) ---------------------
cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip  (2)/(3) : compilateur ou en-tetes pops absents")
    print("test_ssprk3_production : OK (garde amont verte, parite non compilee)"
          if fails == 0 else f"{fails} ECHEC(S)")
    sys.exit(0 if fails == 0 else 1)

n, L = 48, 1.0
U = _initial_state(n)
Uflat = U.reshape(-1).tolist()
spec = _native_spec()
e = build_euler_poisson()
tmp = tempfile.mkdtemp()
try:
    so = e.compile(os.path.join(tmp, "euler_poisson_native.so"), INCLUDE, backend="production")

    def build_prod(method):
        s = pops.System(n=n, L=L, periodic=True)
        s._s.add_native_block("gas", so, limiter="minmod", riemann="rusanov", recon="conservative",
                              time=method, gamma=GAMMA, substeps=1, evolve=True)
        s.set_poisson(rhs="charge_density", solver="geometric_mg")
        s.set_state("gas", Uflat)
        return s

    def build_ref_ssprk3():
        s = pops.System(n=n, L=L, periodic=True)
        s.add_block("gas", spec, spatial=pops.Spatial(minmod=True, flux="rusanov", recon="conservative"),
                    time=pops.Explicit(method="ssprk3"))
        s.set_poisson(rhs="charge_density", solver="geometric_mg")
        s.set_state("gas", Uflat)
        return s

    # (2a) eval_rhs : production+SSPRK3 == natif add_block+SSPRK3 (le RESIDU spatial ne depend pas du
    # RK -- mais on verifie que les DEUX chemins instancient le meme bloc avant toute avance).
    prod = build_prod("ssprk3"); prod.solve_fields()
    ref = build_ref_ssprk3(); ref.solve_fields()
    R_prod = np.array(prod.eval_rhs("gas")).reshape(4, n, n)
    R_ref = np.array(ref.eval_rhs("gas")).reshape(4, n, n)
    chk(float(np.max(np.abs(R_prod))) > 1e-3, "(2a) residu non trivial")
    chk(float(np.max(np.abs(R_prod - R_ref))) == 0.0,
        "(2a) eval_rhs production+SSPRK3 BIT-IDENTIQUE au natif add_block+SSPRK3")

    # (2b) avance SSPRK3 : etat final bit-identique au bloc natif sur 12 pas a dt fixe.
    prod = build_prod("ssprk3"); ref = build_ref_ssprk3()
    dt = 1e-3
    for _ in range(12):
        prod.step(dt); ref.step(dt)
    Up = np.array(prod.get_state("gas")).reshape(4, n, n)
    Ur = np.array(ref.get_state("gas")).reshape(4, n, n)
    chk(np.isfinite(Up).all() and Up[0].min() > 0, "(2b) etat production+SSPRK3 physique (fini, rho>0)")
    chk(float(np.max(np.abs(Up - Ur))) == 0.0,
        "(2b) 12 pas production+SSPRK3 BIT-IDENTIQUE au natif add_block+SSPRK3")

    # (3) NON-TRIVIALITE : production+SSPRK3 DIFFERE de production+SSPRK2 (ssprk3 bien selectionne).
    p2 = build_prod("explicit")  # "explicit" == SSPRK2 (defaut historique)
    p3 = build_prod("ssprk3")
    for _ in range(12):
        p2.step(dt); p3.step(dt)
    U2 = np.array(p2.get_state("gas")).reshape(4, n, n)
    U3 = np.array(p3.get_state("gas")).reshape(4, n, n)
    diff = float(np.max(np.abs(U2 - U3)))
    chk(diff > 0.0,
        "(3) production+SSPRK3 != production+SSPRK2 (ecart %.2e -> ssprk3 effectivement actif)" % diff)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("test_ssprk3_production : tout est vert" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)

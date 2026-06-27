#!/usr/bin/env python3
"""Flux HLL expose pour les modeles a vitesses d'onde signees, SANS exiger une pression (3-var).

HLLC/Roe sont gardes par `n_vars==4 && pressure` (transport compressible). HLL (2 ondes, Davis) ne
demande QUE `model.wave_speeds` (vitesses signees) -- pas de pression. Le DSL emet wave_speeds des
qu'une primitive 'p' est DECLAREE (meme si 'p' n'est pas dans primitive_vars, cas du modele isotherme
magnetise Hoffart : rho, m_x, m_y + 'p'=theta*rho declaree). HLL devient donc utilisable la ou hllc/roe
sont rejetes. Moins diffusif que Rusanov (dissipation ~ |sR-sL| signee au lieu de 2*max|v| symetrique).

On verifie :
 (1) Rusanov inchange (NO-DEFAULT-CHANGE implicite : aucune branche rusanov touchee).
 (2) HLL tourne end-to-end sur un Euler compressible natif (4-var) : etat fini, masse conservee,
     resultat DIFFERENT de Rusanov (moins diffusif -> branche HLL active).
 (3) [compilateur] HLL ACCEPTE sur un modele DSL 3-var isotherme qui DECLARE 'p' (wave_speeds emis) ;
     tourne et reste fini -- la capacite nouvelle (avant : "flux Riemann inconnu 'hll'").
 (4) [compilateur] HLL REJETE (erreur claire "vitesses d'onde") sur un modele DSL 3-var SANS 'p'
     (pas de wave_speeds emis) -- le requires-gate protege, pas d'echec de compilation.

S'auto-saute (exit 0) sans compilateur pour (3)/(4) ; (1)/(2) tournent toujours.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops.ir.ops import sqrt
from pops.physics.facade import Model

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


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


def gas():  # Euler compressible 4-var : a pressure() + wave_speeds()
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                     transport=pops.CompressibleFlux(), source=pops.NoSource(),
                     elliptic=pops.ChargeDensity(charge=0.0))


def smooth_rho(n):
    xx, yy = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="xy")
    return 1.0 + 0.3 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)


def run_gas(riemann, n=48, nsteps=10, cfl=0.2):
    s = pops.System(n=n, L=1.0, periodic=True)
    s.add_block("gas", model=gas(), spatial=pops.Spatial(weno5=True, flux=riemann),
                time=pops.Explicit())
    s.set_poisson()
    s.set_density("gas", smooth_rho(n))
    for _ in range(nsteps):
        s.step_cfl(cfl)
    return np.array(s.density("gas")), s.mass("gas")


# --- (1)/(2) HLL natif 4-var (sans compilateur) ----------------------------------------------------
print("== (1)/(2) HLL sur Euler compressible natif (4-var) ==")
d_rus, m_rus = run_gas("rusanov")
d_hll, m_hll = run_gas("hll")
m0 = float(smooth_rho(48).sum())
chk(np.isfinite(d_hll).all() and d_hll.min() > 0, "(2) HLL : etat fini, densite positive")
chk(abs(m_hll - m0) < 1e-7 * abs(m0), "(2) HLL : masse conservee (flux conservatif)")
chk(float(np.max(np.abs(d_hll - d_rus))) > 1e-9, "(2) HLL DIFFERE de Rusanov (branche HLL active, moins diffusif)")
# flux Riemann inconnu reste rejete
chk("godunov" in err_msg(lambda: run_gas("godunov")), "(1) flux inconnu 'godunov' toujours rejete")

# --- (3)/(4) capacite DSL 3-var isotherme (avec compilateur) ---------------------------------------
cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip  (3)/(4) : compilateur ou en-tetes pops absents")
    print("test_hll_isothermal : OK (HLL natif vert)" if fails == 0 else f"{fails} ECHEC(S)")
    sys.exit(0 if fails == 0 else 1)


def iso3(declare_p):
    """Isotherme magnetise 3-var (rho, mx, my). declare_p=True -> primitive 'p' declaree (wave_speeds
    emis, HLL dispo) ; False -> pas de 'p' (pas de wave_speeds, HLL doit etre rejete)."""
    m = Model("iso3_%s" % ("withp" if declare_p else "nop"))
    rho, mx, my = m.conservative_vars("rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"])
    cs2 = 0.5
    u = m.primitive("u", mx / rho); v = m.primitive("v", my / rho)
    if declare_p:
        m.primitive("p", cs2 * rho)  # declaree -> wave_speeds emis (meme hors primitive_vars)
    c = sqrt(cs2)
    m.flux(x=[mx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    return m


tmp = tempfile.mkdtemp()
try:
    cm_p = iso3(True).compile(os.path.join(tmp, "iso3_withp.so"), INCLUDE, backend="production")
    cm_np = iso3(False).compile(os.path.join(tmp, "iso3_nop.so"), INCLUDE, backend="production")
    chk("p" not in cm_p.prim_names, "(3) 'p' hors primitive_vars (hllc/roe le rejetteraient)")

    def build(cm, riem):
        s = pops.System(n=40, L=1.0, periodic=True)
        s.add_equation("f", model=cm, spatial=pops.FiniteVolume(limiter="weno5", riemann=riem,
                                                              variables="conservative"),
                       time=pops.Explicit(method="ssprk2"))
        s.set_poisson()
        z = np.zeros((40, 40)); r = 1.0 + 0.2 * smooth_rho(40) / smooth_rho(40).max()
        s.set_primitive_state("f", rho=r, u=z, v=z)
        return s

    # (3) HLL ACCEPTE sur le 3-var qui declare 'p' -> tourne fini.
    s_hll = build(cm_p, "hll")
    for _ in range(8):
        s_hll.step_cfl(0.2)
    chk(np.isfinite(np.array(s_hll.density("f"))).all(), "(3) HLL accepte sur DSL 3-var (p declaree) + tourne fini")

    # (4) HLL REJETE sur le 3-var SANS 'p' (pas de wave_speeds) -> erreur claire.
    msg = err_msg(lambda: build(cm_np, "hll"))
    chk("wave_speeds" in msg,
        "(4) HLL rejete sur DSL 3-var sans 'p' (message 'vitesses d'onde'): %r" % msg[:70])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("test_hll_isothermal : tout est vert" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)

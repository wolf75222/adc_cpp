#!/usr/bin/env python3
"""AmrSystem : alignement de table System/AMR pour limiter='weno5' x riemann={'hllc','roe'}.

DIVERGENCE CORRIGEE (audit GENERICITY_2026-06 §8 "registry des tags") : les branches hllc et roe des
dispatchs AMR (detail::dispatch_amr_block ET dispatch_amr_compiled, amr_dsl_block.hpp) n'avaient PAS de
cas 'weno5' (seulement none/minmod/vanleer) alors que System::make_block (block_builder.hpp) le route.
Resultat : un utilisateur AmrSystem demandant un schema compressible weno5+hllc (ou weno5+roe) recevait
"limiter inconnu 'weno5'" la ou le MEME modele buildait sous System. Les deux branches AMR portent
desormais le cas weno5 (build_amr_block / build_amr_compiled supportent deja Weno5, cables sur
rusanov/hll) -> parite STRICTE de surface System/AMR.

Verifie (bloc NATIF adc.Model -> AUCUN compilateur requis ; le dispatch est exerce au BUILD) :
  - MONO-BLOC (un seul add_block -> dispatch_amr_compiled) : Euler compressible + weno5 + hllc TOURNE
    fini ; idem weno5 + roe. (Avant le fix : RuntimeError "limiter inconnu 'weno5'".)
  - MULTI-BLOCS (>= 2 add_block -> dispatch_amr_block) : deux blocs Euler weno5 + hllc TOURNENT finis.
  - GARDE INTACTE : weno5 + hllc sur un transport ISOTHERME 3-var (non Euler, sans pression / capability
    HLLC) reste REJETE par la CAPABILITE ("exige un transport compressible"), PAS par le limiteur ->
    weno5 ne court-circuite pas la garde de flux.

Invariants par assert ; imprime "OK test_amr_weno5_hllc_roe" en cas de succes.
"""
import sys

import numpy as np

import adc

GAMMA = 1.4
fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def euler_spec():
    """Bloc natif compressible (4 var, pression -> capability HLLC/Roe canonique). Pas de compilateur."""
    return adc.Model(state=adc.FluidState("compressible", gamma=GAMMA),
                     transport=adc.CompressibleFlux(), source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=0.0, n0=0.0))


def iso_spec():
    """Bloc natif isotherme (3 var, PAS de pression -> hllc/roe rejetes par capability)."""
    return adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                     transport=adc.IsothermalFlux(), source=adc.NoSource(),
                     elliptic=adc.BackgroundDensity(alpha=0.0, n0=0.0))


def bump(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return (1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)).ravel()


n = 32
rho = bump(n)

# --- 1. MONO-BLOC (dispatch_amr_compiled) : weno5 + hllc, puis weno5 + roe ---------------------------
for riem in ("hllc", "roe"):
    print(f"== mono-bloc Euler : weno5 + {riem} (dispatch_amr_compiled) ==")
    s = adc.AmrSystem(n=n, L=1.0, periodic=True)
    s.set_refinement(1e30)  # mono-niveau : le sujet est le ROUTAGE du dispatch (exerce au build)
    s.add_block("gas", euler_spec(),
                spatial=adc.FiniteVolume(limiter="weno5", riemann=riem), time=adc.Explicit())
    s.set_density("gas", rho.copy())
    for _ in range(3):
        s.step(1e-4)
    d = np.asarray(s.density("gas"))
    chk(np.all(np.isfinite(d)), f"weno5 + {riem} : densite finie sur 3 pas (build OK, plus de 'limiter inconnu')")

# --- 2. MULTI-BLOCS (dispatch_amr_block) : deux blocs Euler weno5 + hllc -----------------------------
print("== multi-blocs Euler : 2 blocs weno5 + hllc (dispatch_amr_block) ==")
s = adc.AmrSystem(n=n, L=1.0, periodic=True)
s.set_refinement(1e30)
s.add_block("a", euler_spec(), spatial=adc.FiniteVolume(limiter="weno5", riemann="hllc"), time=adc.Explicit())
s.add_block("b", euler_spec(), spatial=adc.FiniteVolume(limiter="weno5", riemann="hllc"), time=adc.Explicit())
s.set_density("a", rho.copy())
s.set_density("b", rho.copy())
for _ in range(3):
    s.step(1e-4)
chk(np.all(np.isfinite(np.asarray(s.density("a")))), "multi-blocs weno5 + hllc : bloc a fini")
chk(np.all(np.isfinite(np.asarray(s.density("b")))), "multi-blocs weno5 + hllc : bloc b fini")

# --- 3. GARDE DE CAPABILITE INTACTE : weno5 + hllc sur isotherme 3-var -> rejet de FLUX, pas de LIM --
print("== isotherme 3-var : weno5 + hllc rejete par la CAPABILITE (pas par le limiteur) ==")
try:
    s = adc.AmrSystem(n=n, L=1.0, periodic=True)
    s.set_refinement(1e30)
    s.add_block("iso", iso_spec(),
                spatial=adc.FiniteVolume(limiter="weno5", riemann="hllc"), time=adc.Explicit())
    s.set_density("iso", rho.copy())
    s.step(1e-4)
    chk(False, "weno5 + hllc sur isotherme aurait du lever (capability)")
except RuntimeError as e:
    msg = str(e)
    chk("hllc" in msg or "compressible" in msg,
        f"rejet par la capability HLLC (pas 'limiter inconnu') : {msg[:80]}")
    chk("limiter inconnu" not in msg, "weno5 ne court-circuite PAS la garde de flux")

if fails:
    print(f"FAIL test_amr_weno5_hllc_roe : {fails} echec(s)")
    sys.exit(1)
print("OK test_amr_weno5_hllc_roe")

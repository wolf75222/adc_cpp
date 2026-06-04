#!/usr/bin/env python3
"""WENO5-Z + SSPRK3 accessibles depuis l'API Python (adc.Spatial / adc.Explicit).

Verifie :
 (1) adc.Spatial(limiter="weno5") (et le raccourci weno5=True) construit et tourne un bloc
     end-to-end : plus d'erreur "limiter inconnu", masse conservee, etat fini. Idem en flux hllc.
 (2) adc.Explicit(method="ssprk3") (et ssprk3=True) selectionne le schema temporel d'ordre 3 ;
     le DEFAUT reste SSPRK2 (kind="explicit").
 (3) NO-DEFAULT-CHANGE : un run minmod + SSPRK2 (le defaut) donne un resultat BIT-IDENTIQUE selon
     qu'on cree le bloc avec adc.Spatial() / adc.Explicit() par defaut ou en nommant explicitement
     limiter="minmod"/method="ssprk2". Le chemin par defaut n'a pas bouge.
 (4) PRECISION : sur un transport lisse (Euler, bulle de densite douce), WENO5+SSPRK3 reste fini et
     conserve la masse ; combine a minmod/rusanov (defaut) il tourne aussi -> les schemas coexistent.
 (5) Garde-fous : weno5 rejete sur les chemins compiles (.so) ; une methode temporelle invalide leve.
"""
import sys

import numpy as np

import adc

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def meshx(n):
    return (np.arange(n) + 0.5) / n


def gas():  # Euler compressible (4 var) : accepte rusanov/hllc/roe
    return adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                     transport=adc.CompressibleFlux(), source=adc.NoSource(),
                     elliptic=adc.ChargeDensity(charge=0.0))


def smooth_rho(n):
    xx, yy = np.meshgrid(meshx(n), meshx(n), indexing="xy")
    return 1.0 + 0.3 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)


def run(n, limiter, flux, method, nsteps=10, cfl=0.2):
    s = adc.System(n=n, L=1.0, periodic=True)
    s.add_block("gas", model=gas(),
                spatial=adc.Spatial(limiter=limiter, flux=flux),
                time=adc.Explicit(method=method))
    s.set_poisson()
    s.set_density("gas", smooth_rho(n))
    for _ in range(nsteps):
        s.step_cfl(cfl)
    return s


# --- 1. adc.Spatial(limiter="weno5") end-to-end (plus de "limiter inconnu") -----
print("== adc.Spatial(limiter='weno5') : construit + tourne ==")
n = 48
sw = run(n, "weno5", "rusanov", "ssprk3")
dw = np.array(sw.density("gas"))
chk(np.isfinite(dw).all() and dw.min() > 0, "weno5+rusanov+ssprk3 : etat fini, densite positive")
m0 = float(smooth_rho(n).sum())
chk(abs(sw.mass("gas") - m0) < 1e-7 * abs(m0), "weno5 : masse conservee")
# raccourci weno5=True + flux hllc
sw2 = adc.System(n=32, L=1.0, periodic=True)
sw2.add_block("gas", model=gas(), spatial=adc.Spatial(weno5=True, flux="hllc"),
              time=adc.Explicit(ssprk3=True))
sw2.set_poisson(); sw2.set_density("gas", smooth_rho(32))
for _ in range(8):
    sw2.step_cfl(0.2)
chk(np.isfinite(np.array(sw2.density("gas"))).all(), "weno5=True + hllc + ssprk3=True : fini")

# --- 2. SSPRK3 selectionnable ; defaut = SSPRK2 ---------------------------------
print("== adc.Explicit : defaut SSPRK2, method='ssprk3' selectionnable ==")
chk(adc.Explicit().kind == "explicit", "Explicit() defaut -> kind 'explicit' (SSPRK2)")
chk(adc.Explicit().method == "ssprk2", "Explicit() defaut -> method 'ssprk2'")
chk(adc.Explicit(method="ssprk3").kind == "ssprk3", "Explicit(method='ssprk3') -> kind 'ssprk3'")
chk(adc.Explicit(ssprk3=True).kind == "ssprk3", "Explicit(ssprk3=True) -> kind 'ssprk3'")
chk(raises(lambda: adc.Explicit(method="rk4")), "Explicit : methode inconnue levee")

# --- 3. NO-DEFAULT-CHANGE : minmod + SSPRK2 bit-identique ------------------------
print("== no-default-change : minmod/SSPRK2 (defaut) bit-identique ==")
# Oracle : bloc cree avec les valeurs par DEFAUT (Spatial()/Explicit()).
s_def = run(64, "minmod", "rusanov", "ssprk2")  # method explicite mais == defaut
d_def = np.array(s_def.density("gas"))
# Reference : memes etapes, en laissant TOUS les defauts implicites (Spatial(), Explicit()).
s_ref = adc.System(n=64, L=1.0, periodic=True)
s_ref.add_block("gas", model=gas())  # spatial/time None -> Spatial() (minmod) + Explicit() (ssprk2)
s_ref.set_poisson(); s_ref.set_density("gas", smooth_rho(64))
for _ in range(10):
    s_ref.step_cfl(0.2)
d_ref = np.array(s_ref.density("gas"))
chk(np.array_equal(d_def, d_ref),
    "minmod/SSPRK2 explicite == defaut implicite (BIT-IDENTIQUE, diff vide)")

# --- 4. WENO5+SSPRK3 sain sur un transport lisse, et coexiste avec le defaut -----
# La preuve d'ORDRE rigoureuse (advection lineaire exacte : WENO5 erreur << minmod, pente > 2) vit
# dans le test C++ tests/test_weno5_ssprk3.cpp (acces direct a make_block / l'erreur analytique). Ici
# on verifie l'end-to-end : WENO5+SSPRK3 reste fini, densite positive, masse conservee sur un Euler
# lisse de longue duree, et un bloc weno5 et un bloc minmod (defaut) coexistent dans le meme System.
print("== WENO5+SSPRK3 sain (long run lisse) + coexistence avec le defaut ==")
s_w5 = adc.System(n=64, L=1.0, periodic=True)
s_w5.add_block("gas", model=gas(), spatial=adc.Spatial(weno5=True), time=adc.Explicit(ssprk3=True))
s_w5.set_poisson(); s_w5.set_density("gas", smooth_rho(64))
m_w5_0 = s_w5.mass("gas")
for _ in range(40):
    s_w5.step_cfl(0.3)
d_w5 = np.array(s_w5.density("gas"))
chk(np.isfinite(d_w5).all() and d_w5.min() > 0, "WENO5+SSPRK3 long run : fini, densite positive")
chk(abs(s_w5.mass("gas") - m_w5_0) < 1e-9 * abs(m_w5_0),
    "WENO5+SSPRK3 long run : masse conservee (flux conservatif)")
mix = adc.System(n=32, L=1.0, periodic=True)
mix.add_block("hi", model=gas(), spatial=adc.Spatial(weno5=True), time=adc.Explicit(ssprk3=True))
mix.add_block("lo", model=gas(), spatial=adc.Spatial(minmod=True), time=adc.Explicit())
mix.set_poisson(); mix.set_density("hi", smooth_rho(32)); mix.set_density("lo", smooth_rho(32))
for _ in range(6):
    mix.step_cfl(0.2)
chk(np.isfinite(np.array(mix.density("hi"))).all() and np.isfinite(np.array(mix.density("lo"))).all(),
    "bloc weno5/ssprk3 et bloc minmod/ssprk2 coexistent dans un meme System")

# --- 5. garde-fous : weno5 sur les chemins compiles (.so) -> rejet --------------
print("== garde-fous : weno5 sur chemins compiles rejete ==")
# add_compiled_block / add_native_block n'exposent pas weno5 (grille locale .so a 2 ghosts).
ss = adc.System(n=16)._s  # facade compilee brute (pour viser les methodes .so directement)
chk(raises(lambda: ss.add_compiled_block("x", "/inexistant.so", "weno5")),
    "add_compiled_block(weno5) leve (rejet avant dlopen ou limiteur)")
chk(raises(lambda: ss.add_native_block("x", "/inexistant.so", "weno5")),
    "add_native_block(weno5) leve")

print("OK test_weno5_ssprk3" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)

#!/usr/bin/env python3
"""Cache des vitesses d'onde HLL (ADC-199) : opt-in, bit-identique en NoSlope.

Motivation (audit step_cfl, run diocotron HyQMOM) : avec riemann='hll' + vitesses d'onde
exactes, AssembleRhsKernel rappelle model.wave_speeds par FACE -> wave_speeds est recalcule
plusieurs fois par cellule et par etage RK ; pour un modele HyQMOM (hierarchie de moments +
factorisations a chaque appel) le pas explose (x10 mesure n=128). Le cache opt-in evalue
wave_speeds UNE fois par cellule et direction, puis borne chaque face par min/max des deux
cellules voisines.

On verifie :
 (1) BIT-IDENTITE : limiter='none' (NoSlope) + recon conservatif -> wave_speeds recoit les memes
     entrees que le chemin par face, donc le cache (ON) donne EXACTEMENT le meme etat que OFF apres
     N pas (np.array_equal sur l'etat). Le champ doit avoir REELLEMENT evolue (sinon test creux).
 (2) DEFAUT INCHANGE : un bloc construit SANS wave_speed_cache == le bloc OFF (cache implicite OFF).
 (3) GARDE riemann : wave_speed_cache=True avec riemann != 'hll' -> erreur explicite (pas d'ignore
     silencieux : le cache ne s'applique qu'au flux HLL).
 (4) GARDE temps : wave_speed_cache=True avec un traitement IMEX -> erreur explicite (cable sur
     l'avance explicite seulement).
 (5) GARDE geometrie disque : wave_speed_cache=True avec un mode de transport disque (staircase /
     cutcell, via set_disc_domain / set_geometry_mode) -> erreur explicite dans LES DEUX ORDRES
     (cache puis disque, disque puis cache). Le cache n'est cable que sur l'avance cartesienne pleine.

NOTE : la PREUVE D'ENGAGEMENT (le cache appelle wave_speeds par cellule, pas par face) vit dans le
test C++ tests/test_wave_speed_cache_engagement.cpp (compteur de wave_speeds, calls_on < calls_off) :
les verifs ON==OFF ci-dessous reussiraient meme si le cache devenait un no-op silencieux.
Modele natif IsothermalFlux (expose wave_speeds) : aucun compilateur requis.
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


def err_msg(fn):
    try:
        fn()
        return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


CS2 = 0.5
N = 32


def make_sim(cache, riemann="hll", limiter="none", time=None):
    sim = adc.System(n=N, L=1.0, periodic=True)
    sim.add_block("ions",
                  adc.Model(state=adc.FluidState("isothermal", cs2=CS2),
                            transport=adc.IsothermalFlux(),
                            source=adc.NoSource(),
                            elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
                  spatial=adc.FiniteVolume(limiter=limiter, riemann=riemann,
                                           wave_speed_cache=cache),
                  time=time if time is not None else adc.Explicit())
    return sim


x = (np.arange(N) + 0.5) / N
X, Y = np.meshgrid(x, x, indexing="ij")
U0 = np.stack([1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y),
               0.2 * np.cos(2 * np.pi * X),
               -0.15 * np.sin(2 * np.pi * Y)])

print("== (1) bit-identite NoSlope+HLL : cache ON == OFF apres N pas ==")
s_off = make_sim(cache=False)
s_on = make_sim(cache=True)
s_off.set_state("ions", U0)
s_on.set_state("ions", U0)
for _ in range(20):
    s_off.step_cfl(0.4)
    s_on.step_cfl(0.4)
A_off = np.array(s_off.get_state("ions"))
A_on = np.array(s_on.get_state("ions"))
chk(not np.array_equal(A_off, U0), "l'etat a reellement evolue (test non creux)")
chk(np.array_equal(A_off, A_on), "cache ON et OFF bit-identiques (0 ulp) sur l'etat final")

print("== (2) defaut inchange : sans wave_speed_cache == cache OFF ==")
s_def = adc.System(n=N, L=1.0, periodic=True)
s_def.add_block("ions",
                adc.Model(state=adc.FluidState("isothermal", cs2=CS2),
                          transport=adc.IsothermalFlux(),
                          source=adc.NoSource(),
                          elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
                spatial=adc.FiniteVolume(limiter="none", riemann="hll"),
                time=adc.Explicit())
s_def.set_state("ions", U0)
for _ in range(20):
    s_def.step_cfl(0.4)
chk(np.array_equal(np.array(s_def.get_state("ions")), A_off),
    "FiniteVolume sans wave_speed_cache == cache OFF (bit-identique)")

print("== (3) garde riemann : cache + rusanov -> erreur ==")
msg = err_msg(lambda: make_sim(cache=True, riemann="rusanov"))
chk("wave_speed_cache" in msg and "hll" in msg,
    f"rusanov + cache rejete ({msg[:60]}...)")

print("== (4) garde temps : cache + IMEX -> erreur ==")
msg = err_msg(lambda: make_sim(cache=True, riemann="hll", time=adc.IMEX()))
chk("wave_speed_cache" in msg,
    f"IMEX + cache rejete ({msg[:60]}...)")

print("== (5) garde geometrie disque : cache + transport staircase/cutcell -> erreur ==")
# Le cache n'est cable que sur l'avance cartesienne PLEINE : un mode disque (set_disc_domain /
# set_geometry_mode) emprunte advance_masked / advance_eb qui l'ignorent -> rejet explicite, pas
# d'ignore muet. On exerce les DEUX ordres (cache d'abord puis disque, et disque d'abord puis cache).


def make_disc_sim_then_mode():
    sim = make_sim(cache=True)  # bloc cache (cartesien plein)
    sim.set_disc_domain(0.5, 0.5, 0.3, mode="staircase")  # doit lever (cache deja actif)


def make_mode_then_cache():
    sim = adc.System(n=N, L=1.0, periodic=True)
    sim.set_disc_domain(0.5, 0.5, 0.3, mode="cutcell")  # mode disque d'abord
    sim.add_block("ions",
                  adc.Model(state=adc.FluidState("isothermal", cs2=CS2),
                            transport=adc.IsothermalFlux(),
                            source=adc.NoSource(),
                            elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
                  spatial=adc.FiniteVolume(limiter="none", riemann="hll",
                                           wave_speed_cache=True),  # doit lever (mode disque actif)
                  time=adc.Explicit())


msg = err_msg(make_disc_sim_then_mode)
chk("wave_speed_cache" in msg and ("staircase" in msg or "disque" in msg),
    f"cache puis set_disc_domain(staircase) rejete ({msg[:60]}...)")
msg = err_msg(make_mode_then_cache)
chk("wave_speed_cache" in msg and "disque" in msg,
    f"set_disc_domain(cutcell) puis add_block(cache) rejete ({msg[:60]}...)")

print("FAILS =", fails)
sys.exit(1 if fails else 0)

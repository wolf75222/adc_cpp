#!/usr/bin/env python3
"""Diagnostic dt_hotspot (ADC-182) : la cellule qui domine la borne CFL de transport.

Motivation (audit step_cfl, run diocotron HyQMOM) : un dt qui s'effondre etait dicte par UNE
cellule a moments non-realisables -- invisible depuis last_dt_bound() (nomme le bloc, pas la
cellule) ; le diagnostic avait exige un scan numpy externe. dt_hotspot() le rend natif,
generique (toute brique, tout modele), a la demande et HORS chemin chaud.

On verifie :
 (1) cellule chaude PLANTEE : etat isotherme au repos sauf une cellule (i0, j0) a grande
     vitesse -> dt_hotspot retourne exactement (i0, j0) et w == |u| + cs (analytique) ;
 (2) coherence stepper : w retourne == cfl*h/dt de step_cfl (meme reduction, bit-exact) ;
 (3) no-default-change : la presence du diagnostic ne change pas step_cfl (bit-identique
     entre deux Systems construits a l'identique, l'un interroge l'autre non) ;
 (4) etat uniforme : la cellule retournee est la premiere en ordre lexicographique (0, 0) --
     determinisme du depart d'egalite ;
 (5) garde : bloc inconnu -> erreur explicite.
Modele natif : aucun compilateur requis.
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
        fn(); return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


CS2 = 0.5


def make_sim(n=32):
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.add_block("ions",
                  adc.Model(state=adc.FluidState("isothermal", cs2=CS2),
                            transport=adc.IsothermalFlux(),
                            source=adc.NoSource(),
                            elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0)),
                  spatial=adc.FiniteVolume(limiter="none", riemann="rusanov"),
                  time=adc.Explicit())
    return sim


print("== (1) cellule chaude plantee ==")
n, i0, j0, u_hot = 32, 21, 9, 7.5
sim = make_sim(n)
rho = np.ones((n, n))
mx = np.zeros((n, n))
mx[j0, i0] = u_hot * rho[j0, i0]   # axe y = lignes, axe x = colonnes (layout des etats)
sim.set_state("ions", np.stack([rho, mx, np.zeros((n, n))]))
w, ih, jh = sim.dt_hotspot("ions")
w_ref = u_hot + np.sqrt(CS2)
chk(abs(w - w_ref) < 1e-12, f"w == |u| + cs analytique ({w:.6f} vs {w_ref:.6f})")
chk((int(ih), int(jh)) == (i0, j0), f"cellule ({int(ih)}, {int(jh)}) == plantee ({i0}, {j0})")

print("== (2) coherence stepper : w == cfl*h/dt ==")
dt = sim.step_cfl(0.4)
chk(abs(0.4 * (1.0 / n) / dt - w) < 1e-9 * w,
    f"cfl*h/dt = {0.4 * (1.0 / n) / dt:.6f} == w (meme reduction)")

print("== (3) no-default-change : interroger le diagnostic ne change pas le pas ==")
sa, sb = make_sim(), make_sim()
x = (np.arange(32) + 0.5) / 32
X, Y = np.meshgrid(x, x, indexing="ij")
U0 = np.stack([1.0 + 0.3 * np.sin(2 * np.pi * X), 0.4 * np.cos(2 * np.pi * Y),
               np.zeros((32, 32))])
sa.set_state("ions", U0)
sb.set_state("ions", U0)
_ = sa.dt_hotspot("ions")   # interroge AVANT le pas
dta = sa.step_cfl(0.4)
dtb = sb.step_cfl(0.4)
chk(dta == dtb and np.array_equal(np.array(sa.get_state("ions")),
                                  np.array(sb.get_state("ions"))),
    "step_cfl et etat bit-identiques avec/sans interrogation")

print("== (4) determinisme du depart d'egalite (etat uniforme) ==")
su = make_sim()
su.set_state("ions", np.stack([np.ones((32, 32)), 0.5 * np.ones((32, 32)),
                               np.zeros((32, 32))]))
w_u, iu, ju = su.dt_hotspot("ions")
chk((int(iu), int(ju)) == (0, 0), f"uniforme -> premiere cellule (0,0) (recu ({int(iu)}, {int(ju)}))")

print("== (5) garde ==")
msg = err_msg(lambda: sim.dt_hotspot("fantome"))
chk("fantome" in msg or "introuvable" in msg or "unknown" in msg.lower(),
    f"bloc inconnu rejete ({msg[:50]}...)")

print("FAILS =", fails)
sys.exit(1 if fails else 0)

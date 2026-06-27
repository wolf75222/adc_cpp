"""Horloge AMR (audit 2026-06, IO PR-IO-3 prerequis) : AmrSystem.macro_step() / set_clock(),
parite avec System. Prerequis du checkpoint AMR, mais UTILE SEUL (cadence stride + reprise
d'horloge).

CONTEXTE. La cadence stride d'un bloc AMR (hold-then-catch-up : un bloc stride=M est tenu aux
macro-pas 0..M-2 puis rattrape quand (macro_step+1) % M == 0) et la cadence de regrid
(macro_step % regrid_every) dependent du COMPTEUR DE MACRO-PAS, pas seulement du temps t. AmrSystem
n'exposait que time() ; macro_step() / set_clock() comblent ce trou (parite System).

VERROUILLE :
  T1 - macro_step() == nombre de macro-pas effectues (multi-blocs avec un bloc stride=2 : le compteur
       avance d'UN par macro-pas, INDEPENDAMMENT du stride des blocs).
  T2 - set_clock(t, ms) restaure EXACTEMENT (time() == t, macro_step() == ms).
  T3 - le compteur REPREND depuis la valeur restauree (set_clock(.., K) puis 2 pas -> K+2).
  T4 - set_clock(.., -1) leve (macro_step >= 0 exige).
  T5 - macro_step() == 0 avant tout pas (parite System).

LIMITE HONNETE. AmrSystem.checkpoint/restart reste NON cable (etats fins par patch absents de l'ABI,
cf. test docstring de AmrSystem.checkpoint). Ce test ne valide donc PAS une reprise bit-identique de
l'ETAT (impossible) ; il valide l'horloge + la cadence, qui sont independantes et utiles seules
(p.ex. pour piloter une sortie write a cadence fixe et reprendre la phase regrid/stride).
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import numpy as np

import pops


def _bump(n, amp):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    r = 1.0 + amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01)
    return r + (1.0 - r.mean())  # moyenne nulle -> Sum q n solvable en periodique


def _scalar_charge(q, B0=1.0):
    return pops.Model(pops.Scalar(), pops.ExB(B0=B0), pops.NoSource(), pops.ChargeDensity(charge=q))


def _build_stride(n=32):
    """AMR multi-blocs : un bloc a stride=2 (cadence hold-then-catch-up) -> la cadence depend du
    compteur de macro-pas, ce que macro_step()/set_clock() exposent et restaurent."""
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
    sim.add_block("ions", _scalar_charge(+1.0),
                  spatial=pops.Spatial(limiter=FirstOrder(), flux=Rusanov()))
    sim.add_block("slow", _scalar_charge(-1.0),
                  spatial=pops.Spatial(limiter=Minmod(), flux=Rusanov()),
                  time=pops.Explicit(stride=2))  # bloc lent : cadence stride=2
    sim.set_poisson(bc="periodic")
    sim.set_density("ions", _bump(n, 0.40))
    sim.set_density("slow", _bump(n, 0.20))
    return sim


def test_amr_macro_step_counts_macro_steps():
    """T1 + T5 : macro_step() == nombre de macro-pas (0 au depart, +1 par step, stride-independant)."""
    sim = _build_stride()
    assert sim.macro_step() == 0, "macro_step() != 0 avant tout pas"
    dt = 1e-3
    for k in range(1, 6):
        sim.step(dt)
        assert sim.macro_step() == k, "macro_step() = %d apres %d pas" % (sim.macro_step(), k)


def test_amr_set_clock_roundtrip():
    """T2 : set_clock(t, ms) restaure time() et macro_step()."""
    sim = _build_stride()
    sim.step(1e-3)
    sim.step(1e-3)
    sim.set_clock(0.25, 7)
    assert sim.macro_step() == 7, "macro_step() = %d apres set_clock(.., 7)" % sim.macro_step()
    assert abs(sim.time() - 0.25) < 1e-15, "time() = %r apres set_clock(0.25, ..)" % sim.time()


def test_amr_clock_resumes_from_restored():
    """T3 : apres set_clock(.., K) le compteur REPREND depuis K (K + 2 apres 2 pas)."""
    sim = _build_stride()
    sim.set_clock(0.0, 3)  # restauration AVANT le 1er pas (build paresseux : phase poussee au 1er step)
    sim.step(1e-3)
    sim.step(1e-3)
    assert sim.macro_step() == 5, "macro_step() = %d (attendu 3 + 2)" % sim.macro_step()


def test_amr_set_clock_rejects_negative():
    """T4 : set_clock(.., macro_step < 0) leve (parite System)."""
    sim = _build_stride()
    raised = False
    try:
        sim.set_clock(0.0, -1)
    except RuntimeError:
        raised = True
    assert raised, "set_clock(.., -1) aurait du lever"


if __name__ == "__main__":
    test_amr_macro_step_counts_macro_steps()
    print("OK T1/T5 : macro_step() compte les macro-pas")
    test_amr_set_clock_roundtrip()
    print("OK T2 : set_clock round-trip (t, macro_step)")
    test_amr_clock_resumes_from_restored()
    print("OK T3 : le compteur reprend depuis la valeur restauree")
    test_amr_set_clock_rejects_negative()
    print("OK T4 : set_clock(.., -1) rejete")
    print("test_amr_clock : OK")

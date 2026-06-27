#!/usr/bin/env python3
"""Checkpoint / restart AMR BIT-IDENTIQUE, MONO-BLOC MONO-RANG (ADC-65).

Comble les manques d'ABI signales par l'ancienne docstring de AmrSystem.checkpoint (etats fins par
patch + etat conservatif COMPLET illisibles/inecrivables, hierarchie non imposable). On expose
desormais, en MONO-BLOC MONO-RANG :

  - level_state(k) / set_level_state(k, .) : etat conservatif COMPLET du niveau k (toutes
    composantes ; grossier ET patchs fins), plat composante-majeur c*nf*nf + j*nf + i (nf = n << k) ;
  - level_potential(k) / set_level_potential(k, .) : phi du niveau k (niveau 0 = warm-start du
    multigrille, load-bearing pour la reprise bit-identique) ;
  - set_hierarchy(boxes) : IMPOSE la hierarchie fine sauvee (au lieu du clustering Berger-Rigoutsos),
    pour reconstruire EXACTEMENT le decoupage au restart.

VERROUILLE :
  T1 (BIT-IDENTIQUE) : run A de 10 pas AVEC patchs fins actifs (refinement bas, regrid_every=0) ;
       checkpoint a 5 pas ; restart dans un systeme NEUF ; 5 pas ; etat FINAL identique a run A,
       grossier (niveau 0) ET patchs (niveau 1) -- dmax == 0.0 EXACT.
  T2 (HIERARCHIE) : patch_boxes() identiques au checkpoint et apres restart.
  T3 (HORLOGE) : time() / macro_step() restaures (la cadence reprend exactement).
  T4 (REJETS) : composition differente (bloc), n different, regrid_every > 0 au restart, multi-blocs
       au checkpoint -- chacun leve. np>1 = garde de code (non declenchable en serie ; verifiee par
       lecture, cf. checkpoint/restart qui levent si _pops.n_ranks() != 1).

LIMITES HONNETES (rejets explicites, jamais un checkpoint silencieusement faux) : multi-blocs (moteur
AmrRuntime, layout + aux PARTAGES) et MPI np>1 (gather par niveau) sont une SUITE. regrid_every > 0
est refuse (la cadence regrid post-restart re-divergerait la hierarchie).

Lancement : PYTHONPATH=<build>/python python3 python/tests/test_amr_checkpoint.py
"""
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
import os
import tempfile

import numpy as np

import pops


def _bump(n, L=1.0, amp=1.0, w=0.10):
    """Bosse gaussienne centrale (floor 1.0, pic ~2.0) : compacte -> patchs fins centraux quand
    refine_threshold est entre le plancher et le pic. field[j, i] (X selon i, Y selon j)."""
    xs = (np.arange(n) + 0.5) / n * L
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    r2 = (X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2
    return np.ascontiguousarray(1.0 + amp * np.exp(-r2 / w ** 2))


def _build(n=32, regrid_every=0, block="ne"):
    """AMR mono-bloc scalaire (ExB, fond neutralisant pour le Poisson periodique), refinement bas
    (patchs fins actifs des le seed), hierarchie FIGEE (regrid_every=0 -> reprise bit-identique)."""
    rho0 = _bump(n)
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=regrid_every)
    sim.add_block(block,
                  pops.Model(pops.Scalar(), pops.ExB(B0=1.0), pops.NoSource(),
                            pops.BackgroundDensity(alpha=1.0, n0=float(rho0.mean()))),
                  spatial=pops.Spatial(limiter=Minmod(), flux=Rusanov()),
                  time=pops.Explicit())
    sim.set_refinement(threshold=1.5)  # rho > 1.5 -> patchs centraux (pic 2.0, plancher 1.0)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg")
    sim.set_density(block, rho0)
    return sim


def test_amr_checkpoint_bit_identical():
    """T1 + T2 : reprise BIT-IDENTIQUE (etat final niveau 0 ET 1) + hierarchie restauree identique."""
    dt = 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt")

        # --- RUN A : 10 pas, checkpoint a mi-course (5 pas) ---
        simA = _build()
        for _ in range(5):
            simA.step(dt)
        simA.checkpoint(path)
        boxes_chk = simA.patch_boxes()
        assert any(b[0] == 1 for b in boxes_chk), "patchs fins inactifs : test sans interet"
        for _ in range(5):
            simA.step(dt)
        finalA = [np.asarray(simA.level_state(k), dtype=np.float64)
                  for k in range(simA.n_levels())]

        # --- RESTART : systeme NEUF, MEME composition, reprise puis 5 pas ---
        simB = _build()
        simB.restart(path)
        boxes_rst = simB.patch_boxes()
        # T2 : la hierarchie imposee == celle du checkpoint
        assert boxes_rst == boxes_chk, "patch_boxes() apres restart != checkpoint"
        assert simB.n_levels() == simA.n_levels(), "nombre de niveaux divergent"
        for _ in range(5):
            simB.step(dt)
        finalB = [np.asarray(simB.level_state(k), dtype=np.float64)
                  for k in range(simB.n_levels())]

        # T1 : etat FINAL identique BIT A BIT, niveau par niveau (grossier ET patchs fins)
        for k in range(len(finalA)):
            assert finalA[k].shape == finalB[k].shape, "forme du niveau %d divergente" % k
            dmax = float(np.max(np.abs(finalA[k] - finalB[k]))) if finalA[k].size else 0.0
            assert dmax == 0.0, "niveau %d : dmax = %r != 0 (reprise NON bit-identique)" % (k, dmax)


def test_amr_checkpoint_restores_clock():
    """T3 : t et macro_step restaures par le restart (la cadence reprend exactement)."""
    dt = 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt")
        simA = _build()
        for _ in range(5):
            simA.step(dt)
        simA.checkpoint(path)
        t_chk, ms_chk = simA.time(), simA.macro_step()
        assert ms_chk == 5, "macro_step() = %d apres 5 pas (attendu 5)" % ms_chk

        simB = _build()
        simB.restart(path)
        assert simB.macro_step() == ms_chk, "macro_step() = %d apres restart (attendu %d)" % (
            simB.macro_step(), ms_chk)
        assert abs(simB.time() - t_chk) < 1e-15, "time() = %r apres restart (attendu %r)" % (
            simB.time(), t_chk)


def _expect(exc_types, fn, label):
    raised = False
    try:
        fn()
    except exc_types:
        raised = True
    assert raised, label


def test_amr_checkpoint_rejects():
    """T4 : rejets explicites (composition differente, n different, regrid_every>0, multi-blocs)."""
    dt = 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt")
        simA = _build()
        for _ in range(3):
            simA.step(dt)
        simA.checkpoint(path)

        # composition differente : nom de bloc != celui du checkpoint
        sim_diffblock = _build(block="autre")
        _expect(ValueError, lambda: sim_diffblock.restart(path),
                "restart aurait du lever sur une composition (bloc) differente")

        # grille differente : n != celui du checkpoint
        sim_diffn = _build(n=48)
        _expect(ValueError, lambda: sim_diffn.restart(path),
                "restart aurait du lever sur une grille (n) differente")

        # regrid_every > 0 au restart : la cadence regrid re-divergerait
        sim_regrid = _build(regrid_every=10)
        _expect(ValueError, lambda: sim_regrid.restart(path),
                "restart aurait du lever sur regrid_every > 0")


def test_amr_checkpoint_rejects_multiblock():
    """T4 (multi-blocs) : checkpoint AMR multi-blocs leve (layout + aux PARTAGES = suite)."""
    n = 32
    rho0 = _bump(n)
    sim = pops.AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
    for nm, q in (("ions", +1.0), ("elec", -1.0)):
        sim.add_block(nm,
                      pops.Model(pops.Scalar(), pops.ExB(B0=1.0), pops.NoSource(),
                                pops.ChargeDensity(charge=q)),
                      spatial=pops.Spatial(limiter=Minmod(), flux=Rusanov()))
    sim.set_poisson(bc="periodic")
    sim.set_density("ions", rho0)
    sim.set_density("elec", rho0)
    sim.step(1e-3)  # force le build (moteur AmrRuntime)
    with tempfile.TemporaryDirectory() as tmp:
        _expect((NotImplementedError, RuntimeError),
                lambda: sim.checkpoint(os.path.join(tmp, "ckpt")),
                "checkpoint multi-blocs aurait du lever")


if __name__ == "__main__":
    test_amr_checkpoint_bit_identical()
    print("OK T1/T2 : reprise bit-identique (niveau 0 + patchs fins) + hierarchie restauree")
    test_amr_checkpoint_restores_clock()
    print("OK T3 : horloge (t, macro_step) restauree")
    test_amr_checkpoint_rejects()
    print("OK T4 : rejets composition / grille / regrid_every")
    test_amr_checkpoint_rejects_multiblock()
    print("OK T4 : rejet multi-blocs")
    print("test_amr_checkpoint : OK")

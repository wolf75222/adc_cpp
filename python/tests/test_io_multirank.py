"""IO v1 multi-rangs (audit 2026-06, section 10 / PR-IO-2) : accesseurs GLOBAUX collectifs
(state_global / density_global / potential_global) + ecriture rang-0 de sim.write / sim.checkpoint.

CONTEXTE. Le System construit UNE box couvrant tout le domaine (mono-box ; cf. system.cpp ctor :
ba = {index_domain}, dm round-robin -> box 0 sur le rang 0). Sous MPI np>1, les accesseurs
non-globaux (density / get_state / potential) lisent fab(0) : valides sur le rang proprietaire, mais
HORS BORNES sur un rang sans box. Les variantes _global rassemblent le champ par all_reduce_sum
(chaque rang detient le champ complet) ; sim.write / sim.checkpoint les utilisent puis n'ecrivent le
fichier que sur le rang 0.

CE TEST tourne en MONO-RANG (la batterie pytest n'a pas de harnais MPI ; le cas np>1 -- gather
bit-identique a np=1/2/4 et aller-retour checkpoint/restart -- est couvert par le test C++
tests/test_mpi_system_io_gather.cpp, lance sous mpirun par le preset mpi/ci-mpi). Il verrouille
l'invariant CENTRAL :

  T1 - EQUIVALENCE GLOBAL == LOCAL en mono-rang : state_global == get_state, density_global ==
       density, potential_global == potential, BIT-IDENTIQUE (all_reduce = identite, box = domaine
       complet). C'est la garantie que la facade IO multi-rangs n'a RIEN change au mono-rang.
  T2 - ROUND-TRIP write npz via le chemin global : le fichier relu redonne exactement get_state.
  T3 - CHECKPOINT/RESTART bit-identique via le chemin global (mono-rang ; la semantique np>1 est la
       meme : tout l'etat vit sur le rang 0).
  T4 - my_rank / n_ranks exposes (0 / 1 en serie).
"""
import os
import tempfile

import numpy as np

import adc


def _build(n=16):
    sim = adc.System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_block("ions",
                  adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                            transport=adc.IsothermalFlux(),
                            source=adc.PotentialForce(charge=1.0),
                            elliptic=adc.ChargeDensity(charge=1.0)),
                  spatial=adc.FiniteVolume(limiter="minmod"), time=adc.Explicit())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    sim.set_density("ions", (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).ravel())
    return sim


def test_io_global_equals_local_mono_rank():
    """T1 : en mono-rang, les accesseurs GLOBAUX rendent EXACTEMENT les accesseurs locaux."""
    sim = _build()
    for _ in range(4):
        sim.step(2e-3)
    assert np.array_equal(np.asarray(sim.state_global("ions")),
                          np.asarray(sim.get_state("ions"))), "state_global != get_state (mono-rang)"
    assert np.array_equal(np.asarray(sim.density_global("ions")),
                          np.asarray(sim.density("ions"))), "density_global != density (mono-rang)"
    assert np.array_equal(np.asarray(sim.potential_global()),
                          np.asarray(sim.potential())), "potential_global != potential (mono-rang)"


def test_io_write_npz_roundtrip_global():
    """T2 : write(npz) (chemin global) relu redonne get_state."""
    sim = _build()
    for _ in range(3):
        sim.step(2e-3)
    tmp = tempfile.mkdtemp()
    p = sim.write(os.path.join(tmp, "out"), format="npz", step=2)
    d = np.load(p)
    ref = np.asarray(sim.get_state("ions")).reshape(3, sim.ny(), sim.nx())
    assert np.array_equal(d["state_ions"], ref), "npz (global) != get_state"
    assert int(d["macro_step"]) == 3, "macro_step absent/incoherent dans le npz"


def test_io_checkpoint_restart_global():
    """T3 : checkpoint (gather global) -> restart (scatter MPI-safe) bit-identique en mono-rang."""
    tmp = tempfile.mkdtemp()
    sim = _build()
    for _ in range(3):
        sim.step(2e-3)
    sim.checkpoint(os.path.join(tmp, "chk"))
    for _ in range(4):
        sim.step(2e-3)
    ref = np.asarray(sim.get_state("ions"))

    sim2 = _build()
    sim2.restart(os.path.join(tmp, "chk"))
    for _ in range(4):
        sim2.step(2e-3)
    assert np.array_equal(np.asarray(sim2.get_state("ions")), ref), "restart (global) non bit-identique"


def test_mpi_helpers_exposed():
    """T4 : my_rank / n_ranks exposes au module (0 / 1 en serie)."""
    from adc import _adc
    assert _adc.my_rank() == 0
    assert _adc.n_ranks() >= 1


if __name__ == "__main__":
    test_io_global_equals_local_mono_rank()
    print("OK T1 : global == local (mono-rang)")
    test_io_write_npz_roundtrip_global()
    print("OK T2 : write npz round-trip (global)")
    test_io_checkpoint_restart_global()
    print("OK T3 : checkpoint/restart bit-identique (global)")
    test_mpi_helpers_exposed()
    print("OK T4 : my_rank/n_ranks exposes")
    print("test_io_multirank : OK")

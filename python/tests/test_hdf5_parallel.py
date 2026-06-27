"""IO PR-IO-3 : ecriture HDF5 PARALLELE par hyperslabs, OPT-IN (ADC-66).

CONTEXTE. sim.write(format='hdf5') ecrit par defaut un fichier UNIQUE apres gather global
(state_global / potential_global) sur le rang 0 (chemin serie, PR-IO-2). L'option parallel=True
ouvre le fichier en collectif (h5py driver='mpio') et fait ECRIRE A CHAQUE RANG SES boites en
hyperslabs (accesseurs locaux System::local_boxes / local_state) -- pas de gather global. Le System
cartesien etant MONO-BOX (une box couvrant le domaine, sur le rang 0), le VRAI parallelisme par
hyperslabs n'apparait que sur une geometrie multi-box (cf. AMR, ADC-65) ; ce test verrouille la
CORRECTION de la mecanique en mono-rang.

CETTE BATTERIE tourne en MONO-RANG (pas de harnais MPI cote pytest). Elle verrouille :

  (a) EQUIVALENCE parallel=True == parallel=False (np=1) : les deux fichiers, relus CHAMP A CHAMP,
      portent les memes etats/phi/attributs (le chemin parallele ecrit des datasets CONTIGUS, le
      serie des datasets gzip : meme VALEURS, pas forcement meme octets HDF5). SKIP si h5py absent OU
      h5py sans MPI (messages distincts) OU module _pops sans accesseurs locaux (build anterieur).
  (b) ERREUR CLAIRE si h5py present mais SANS support MPI : parallel=True leve un RuntimeError avec
      remede (installer h5py compile MPI, ou parallel=False) -- jamais d'ecriture silencieuse.
  (c) REGRESSION : format='hdf5' par defaut (parallel=False) inchange -- relu, l'etat == state_global.
  (d) checkpoint(parallel=True) : rejet EXPLICITE (reste npz gather-rang-0).
  (e) write(parallel=True) sur un format != 'hdf5' : rejet EXPLICITE.

NP>1 (manuel, hors pytest) : sur une machine avec h5py compile MPI + mpi4py, lancer
  `mpirun -n 2 python -c "import test_hdf5_parallel as t; t.manual_np_check()"`
(depuis python/tests, PYTHONPATH sur le build). manual_np_check() ecrit en parallel=True et verifie,
sur le rang 0, l'egalite champ a champ avec un dump npz gather-rang-0 du MEME etat.
"""
from pops.numerics.reconstruction.limiters import Minmod
import numpy as np

import pops


class _Skip(Exception):
    pass


class _SkipModule:
    """Shim minimal du vocabulaire pytest utilise ci-dessous : la batterie du depot lance les
    tests en SCRIPTS PLATS (python3 fichier.py), pas sous pytest. skip -> exception locale geree
    par le runner ; raises -> contextmanager d'assertion ; importorskip -> import ou skip."""

    @staticmethod
    def skip(msg):
        raise _Skip(msg)

    @staticmethod
    def importorskip(mod, reason=None):
        import importlib
        try:
            return importlib.import_module(mod)
        except ImportError:
            raise _Skip(reason or ("module %s absent" % mod))

    class raises:
        def __init__(self, exc):
            self.exc = exc
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            assert et is not None and issubclass(et, self.exc), \
                "exception %r attendue" % self.exc
            self.value = ev
            return True


pytest = _SkipModule()


def _build(n=16):
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="periodic")
    sim.add_block("ions",
                  pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                            transport=pops.IsothermalFlux(),
                            source=pops.PotentialForce(charge=1.0),
                            elliptic=pops.ChargeDensity(charge=1.0)),
                  spatial=pops.FiniteVolume(limiter=Minmod()), time=pops.Explicit())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    sim.set_density("ions", (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).ravel())
    return sim


def _read_h5(path):
    """Relit un dump HDF5 (serie ou parallele) en un dict comparable CHAMP A CHAMP."""
    import h5py
    out = {"attrs": {}, "blocks": {}}
    with h5py.File(path, "r") as f:
        for k in ("t", "macro_step", "nx", "ny", "abi_key"):
            out["attrs"][k] = f.attrs[k]
        out["phi"] = np.asarray(f["phi"][...])
        for b in f:
            if b == "phi":
                continue
            g = f[b]
            out["blocks"][b] = {
                "state": np.asarray(g["state"][...]),
                "names": [bytes(s) for s in g.attrs["names"]],
                "roles": [bytes(s) for s in g.attrs["roles"]],
            }
    return out


def _assert_dumps_equal(a, b):
    assert set(a["attrs"]) == set(b["attrs"])
    assert a["attrs"]["t"] == b["attrs"]["t"]
    assert int(a["attrs"]["macro_step"]) == int(b["attrs"]["macro_step"])
    assert int(a["attrs"]["nx"]) == int(b["attrs"]["nx"])
    assert int(a["attrs"]["ny"]) == int(b["attrs"]["ny"])
    assert str(a["attrs"]["abi_key"]) == str(b["attrs"]["abi_key"])
    assert np.array_equal(a["phi"], b["phi"]), "phi differe (serie vs parallele)"
    assert set(a["blocks"]) == set(b["blocks"])
    for blk in a["blocks"]:
        assert np.array_equal(a["blocks"][blk]["state"], b["blocks"][blk]["state"]), \
            "etat '%s' differe (serie vs parallele)" % blk
        assert a["blocks"][blk]["names"] == b["blocks"][blk]["names"]
        assert a["blocks"][blk]["roles"] == b["blocks"][blk]["roles"]


def test_parallel_equals_serial_mono_rank(tmp_path):
    """(a) np=1 : parallel=True relu == parallel=False relu, champ a champ."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py absent : ecriture HDF5 (serie et parallele) non testable")
    if not h5py.get_config().mpi:
        pytest.skip("h5py present mais SANS support MPI (get_config().mpi == False) : "
                    "ecriture parallele par hyperslabs non testable")
    pytest.importorskip("mpi4py", reason="mpi4py absent : ouverture mpio non testable")
    sim = _build()
    for _ in range(3):
        sim.step(2e-3)
    if not hasattr(sim._s, "local_boxes"):
        pytest.skip("module _pops sans local_boxes/local_state (build anterieur a ADC-66) : "
                    "reconstruire adc_cpp pour exercer le chemin parallele")
    p_ser = sim.write(str(tmp_path / "ser"), format="hdf5", parallel=False)
    p_par = sim.write(str(tmp_path / "par"), format="hdf5", parallel=True)
    _assert_dumps_equal(_read_h5(p_ser), _read_h5(p_par))


def test_parallel_clear_error_when_h5py_without_mpi(tmp_path):
    """(b) h5py present SANS MPI : parallel=True leve un RuntimeError avec remede."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py absent : le cas 'h5py sans MPI' n'est pas reproductible ici")
    if h5py.get_config().mpi:
        pytest.skip("h5py compile AVEC MPI : le cas 'h5py sans MPI' n'est pas reproductible ici")
    sim = _build()
    with pytest.raises(RuntimeError) as exc:
        sim.write(str(tmp_path / "x"), format="hdf5", parallel=True)
    msg = str(exc.value)
    assert "MPI" in msg, "le message d'erreur doit pointer le defaut de support MPI"
    assert "parallel=False" in msg, "le message d'erreur doit proposer le remede parallel=False"


def test_serial_hdf5_default_unchanged(tmp_path):
    """(c) regression : format='hdf5' par defaut (gather rang-0) inchange et fidele a state_global."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py absent : sortie HDF5 serie non testable")
    sim = _build()
    for _ in range(3):
        sim.step(2e-3)
    p = sim.write(str(tmp_path / "ser"), format="hdf5")  # defaut parallel=False
    nv, ny, nx = sim._s.n_vars("ions"), sim._s.ny(), sim._s.nx()
    expected = np.asarray(sim.state_global("ions"), dtype=np.float64).reshape(nv, ny, nx)
    with h5py.File(p, "r") as f:
        assert int(f.attrs["nx"]) == nx and int(f.attrs["ny"]) == ny
        st = np.asarray(f["ions"]["state"][...])
        phi = np.asarray(f["phi"][...])
    assert np.array_equal(st, expected), "etat HDF5 serie != state_global (regression)"
    assert np.array_equal(
        phi, np.asarray(sim.potential_global(), dtype=np.float64).reshape(ny, nx))


def test_checkpoint_parallel_true_rejected():
    """(d) checkpoint(parallel=True) : rejet explicite (reste npz gather-rang-0), jamais silencieux."""
    sim = _build()
    with pytest.raises(NotImplementedError) as exc:
        sim.checkpoint("ignored_path", parallel=True)
    assert "parallel=False" in str(exc.value)


def test_parallel_rejected_for_non_hdf5(tmp_path):
    """(e) write(parallel=True) sur un format != 'hdf5' : rejet explicite."""
    sim = _build()
    with pytest.raises(ValueError) as exc:
        sim.write(str(tmp_path / "x"), format="npz", parallel=True)
    assert "hdf5" in str(exc.value)


def manual_np_check():  # pragma: no cover -- lance a la main sous mpirun -n>1 (hors pytest)
    """Verification MANUELLE np>1 (cf. docstring du module). Sur le rang 0, compare l'etat relu du
    dump parallele a un gather-rang-0 (state_global / potential_global) du MEME etat."""
    import os
    import tempfile
    import h5py
    from mpi4py import MPI

    assert h5py.get_config().mpi, "h5py doit etre compile MPI pour manual_np_check"
    comm = MPI.COMM_WORLD
    sim = _build(n=32)
    for _ in range(5):
        sim.step(2e-3)
    out = os.path.join(tempfile.gettempdir(), "pops_hdf5_par_check")
    path = sim.write(out, format="hdf5", parallel=True)
    nv, ny, nx = sim._s.n_vars("ions"), sim._s.ny(), sim._s.nx()
    ref_state = np.asarray(sim.state_global("ions"), dtype=np.float64).reshape(nv, ny, nx)
    ref_phi = np.asarray(sim.potential_global(), dtype=np.float64).reshape(ny, nx)
    comm.Barrier()
    if comm.Get_rank() == 0:
        with h5py.File(path, "r") as f:
            st = np.asarray(f["ions"]["state"][...])
            phi = np.asarray(f["phi"][...])
        assert np.array_equal(st, ref_state), "etat parallele != state_global (np>1)"
        assert np.array_equal(phi, ref_phi), "phi parallele != potential_global (np>1)"
        print("manual_np_check OK (np=%d) : %s" % (comm.Get_size(), path))
    comm.Barrier()


if __name__ == "__main__":  # pragma: no cover
    import tempfile
    import pathlib
    import sys as _sys
    if "--np-check" in _sys.argv:
        manual_np_check()  # usage manuel : mpirun -n 2 python test_hdf5_parallel.py --np-check
        _sys.exit(0)
    fails = 0
    tests = [test_parallel_equals_serial_mono_rank, test_parallel_clear_error_when_h5py_without_mpi,
             test_serial_hdf5_default_unchanged, test_checkpoint_parallel_true_rejected,
             test_parallel_rejected_for_non_hdf5]
    for t in tests:
        needs_tmp = "tmp_path" in t.__code__.co_varnames[:t.__code__.co_argcount]
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as d:
                    t(pathlib.Path(d))
            else:
                t()
            print("  [OK ] %s" % t.__name__)
        except _Skip as e:
            print("  [SKIP] %s : %s" % (t.__name__, e))
        except AssertionError as e:
            print("  [XX ] %s : %s" % (t.__name__, e))
            fails += 1
    if fails:
        print("FAIL test_hdf5_parallel : %d echec(s)" % fails)
        _sys.exit(1)
    print("OK test_hdf5_parallel")

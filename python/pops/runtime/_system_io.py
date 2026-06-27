"""System IO mixin (Spec-4 PR-F): outputs, checkpoint, restart.

VISUALIZATION OUTPUT (vtk/npz/hdf5, serial + parallel-hyperslab), restartable v1 checkpoint and
restart of :class:`pops.runtime.system.System`. Pure Python (zero change to the C++ hot path),
single-rank / rank-0 gather. ATOMIC write (.tmp + os.replace). Mixed into ``System`` via
inheritance; operates on ``self._s``. ``abi_key`` is the module ABI key, baked into the .so
metadata, re-exported through ``pops.runtime.bricks``.
"""

from pops.runtime.bricks import abi_key


class _SystemIO:
    """Output / checkpoint / restart methods of System."""

    # ------------------------------------------------------------------
    # OUTPUTS / CHECKPOINT / RESTART v1 (audit 2026-06, IO; cf. docs/IO_CHECKPOINT_PLAN.md).
    # Pure Python (zero change to the C++ hot path), single-rank; HDF5 aggregated/parallel and AMR =
    # PR-IO-3. ATOMIC write (.tmp file then os.replace: a crash mid-write never
    # corrupts a previous checkpoint).
    # ------------------------------------------------------------------
    def write(self, path, format="vtk", step=None, fields=None, parallel=False):
        """VISUALIZATION OUTPUT: writes the current state to an opened file (ParaView/numpy).

        - ``format="vtk"``: ImageData .vti ASCII (Cartesian; opened by ParaView / VisIt) -- one
          CellData per conservative variable of each block + the potential phi.
        - ``format="npz"``: compressed np.savez (any backend / any geometry) -- per-block states,
          names/roles, phi, t, macro_step, grid.
        - @p step: numbered suffix (path_000123.vti); None = raw path + extension.
        - @p fields: subset of blocks to write (None = all).
        - @p parallel: PARALLEL HDF5 write by hyperslabs (opt-in, format='hdf5' ONLY). Default
          False = rank-0 gather path below, STRICTLY unchanged. True = each rank writes ITS
          boxes into a single file via h5py(mpio) -- requires h5py built with MPI + mpi4py (otherwise a
          CLEAR error with a remedy, never a silent degraded write). Cf. _write_hdf5_parallel.
        - @return the written path.

        MULTI-RANK (MPI np>1): the fields are gathered via the GLOBAL collective accessors
        (state_global / potential_global -- every rank MUST therefore call write), then ONLY rank 0
        writes the file (a single file, identical to single-rank). The System being mono-box (one
        box covering the whole domain, on rank 0), the gather is exact. The other ranks return the
        path without I/O. PARALLEL HDF5 (per-rank hyperslabs): parallel=True (cf. _write_hdf5_parallel;
        real parallelism only in MULTI-BOX, the Cartesian System being mono-box)."""
        import os
        import numpy as np
        from pops import _pops
        if parallel and format != "hdf5":
            raise ValueError(
                "write: parallel=True is only supported for format='hdf5' (write by "
                "hyperslabs); format=%r goes through the rank-0 gather path (parallel=False)."
                % (format,))
        rank0 = (_pops.my_rank() == 0)
        blocks = [b for b in self._s.block_names() if fields is None or b in fields]
        suffix = ("_%06d" % int(step)) if step is not None else ""
        nxv, nyv = self._s.nx(), self._s.ny()
        if format == "npz":
            # COLLECTIVE gather (all ranks) BEFORE the rank-0 guard: state_global / potential_global
            # do an internal all_reduce and must be called by each rank.
            out = {"t": self._s.time(), "macro_step": self._s.macro_step(),
                   "nx": nxv, "ny": nyv, "blocks": np.array(blocks)}
            for b in blocks:
                nv = self._s.n_vars(b)
                out["state_" + b] = np.asarray(self._s.state_global(b), dtype=np.float64).reshape(
                    nv, nyv, nxv)
                out["names_" + b] = np.array(list(self._s.variable_names(b, "conservative")))
                out["roles_" + b] = np.array(list(self._s.variable_roles(b, "conservative")))
            out["phi"] = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
            target = path + suffix + ".npz"
            if not rank0:
                return target  # only rank 0 writes the file (gather already done collectively)
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **out)
            os.replace(tmp, target)
            return target
        if format == "vtk":
            target = path + suffix + ".vti"
            arrays, names = [], []
            for b in blocks:
                nv = self._s.n_vars(b)
                st = np.asarray(self._s.state_global(b), dtype=np.float64).reshape(nv, nyv, nxv)
                for c, nm in enumerate(self._s.variable_names(b, "conservative")):
                    arrays.append(st[c]); names.append("%s_%s" % (b, nm))
            arrays.append(np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv))
            names.append("phi")
            if not rank0:
                return target  # collective gather done above; only rank 0 writes
            lines = ['<?xml version="1.0"?>',
                     '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
                     '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="0 0 0" '
                     'Spacing="%.17g %.17g 1">' % (nxv, nyv, 1.0 / nxv, 1.0 / nyv),
                     '    <Piece Extent="0 %d 0 %d 0 0">' % (nxv, nyv),
                     '      <CellData>']
            for nm, arr in zip(names, arrays):
                lines.append('        <DataArray type="Float64" Name="%s" format="ascii">' % nm)
                lines.append("          " + " ".join("%.17g" % v for v in arr.ravel()))
                lines.append('        </DataArray>')
            lines += ['      </CellData>', '    </Piece>', '  </ImageData>', '</VTKFile>', '']
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(lines))
            os.replace(tmp, target)
            return target
        if format == "hdf5":
            if parallel:
                # PARALLEL HDF5 by hyperslabs (PR-IO-3, opt-in): each rank writes ITS boxes into
                # a single file (h5py mpio), no global gather. SEPARATE path -- the serial path
                # below stays STRICTLY unchanged.
                return self._write_hdf5_parallel(path + suffix + ".h5", blocks, nxv, nyv)
            # AGGREGATED HDF5 v1 (wave 3, plan PR-IO-2): a single file, one group per block,
            # attributes for the clock/grid. Multi-rank: collective gather (state_global /
            # potential_global) then rank-0 write (a single file). PARALLEL HDF5 (per-rank
            # hyperslabs) = parallel=True (branch above). h5py optional: absent -> clear error.
            # COLLECTIVE gather (all ranks) BEFORE the rank-0 guard.
            states = {b: np.asarray(self._s.state_global(b), dtype=np.float64).reshape(
                self._s.n_vars(b), nyv, nxv) for b in blocks}
            phi_g = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
            target = path + suffix + ".h5"
            if not rank0:
                return target  # only rank 0 writes the file
            try:
                import h5py
            except ImportError:
                raise RuntimeError(
                    "write(format='hdf5'): h5py missing (pip/conda install h5py); "
                    "use format='npz' (equivalent, no dependency) in the meantime.")
            tmp = target + ".tmp"
            with h5py.File(tmp, "w") as f:
                f.attrs["t"] = self._s.time()
                f.attrs["macro_step"] = self._s.macro_step()
                f.attrs["nx"] = nxv
                f.attrs["ny"] = nyv
                f.attrs["abi_key"] = abi_key()
                for b in blocks:
                    g = f.create_group(b)
                    g.create_dataset("state", data=states[b], compression="gzip")
                    g.attrs["names"] = [s.encode() for s in
                                        self._s.variable_names(b, "conservative")]
                    g.attrs["roles"] = [s.encode() for s in
                                        self._s.variable_roles(b, "conservative")]
                f.create_dataset("phi", data=phi_g, compression="gzip")
            os.replace(tmp, target)
            return target
        raise ValueError("write: format 'vtk' | 'npz' | 'hdf5' (received %r)" % (format,))

    def _write_hdf5_parallel(self, target, blocks, nxv, nyv):
        """PARALLEL HDF5 WRITE by hyperslabs (write(format='hdf5', parallel=True)) -- PR-IO-3.

        OPT-IN PATH, separate from the serial path (rank-0 gather) which stays untouched. Instead of
        gathering the whole field on rank 0, each rank WRITES ITS BOXES into a SINGLE file opened
        collectively (h5py driver='mpio'). The global datasets (ncomp, ny, nx) per block + phi (ny, nx)
        are created COLLECTIVELY, the metadata (t, macro_step, nx, ny, abi_key, names/roles) written
        collectively, then each rank writes its hyperslabs dset[:, jlo:jhi+1, ilo:ihi+1] in INDEPENDENT
        I/O (disjoint boxes ; a rank with no box writes nothing).

        TRUE PARALLELISM = MULTI-BOX only. The cartesian System is MONO-BOX (one box covering the
        domain, on rank 0) : under np>1 rank 0 writes the single box and the other ranks carry no
        box -- the hyperslab gain appears on a multi-box geometry (cf. AMR, ADC-65). The mechanics
        stay CORRECT in the general case (iteration over all local fabs). phi is solved/gathered
        COLLECTIVELY (potential_global, all_reduce) then written by rank 0 alone (full scalar field,
        contiguous dataset).

        CONTIGUOUS DATASETS (no gzip) : parallel HDF5 does not allow independent writes to
        chunk-filtered datasets. The serial path keeps gzip ; the re-read VALUES are identical field by
        field (parallel=True under np=1 == parallel=False, verified by test_hdf5_parallel).

        NEVER SILENT : h5py absent, h5py without MPI, or mpi4py absent -> RuntimeError with remedy
        (install h5py built with MPI + mpi4py, or parallel=False)."""
        import os
        import numpy as np
        from pops import _pops
        # h5py FIRST, THEN the MPI support test : an h5py present but WITHOUT MPI must give
        # the targeted error (remedy), independently of whether mpi4py is present.
        try:
            import h5py
        except ImportError:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : h5py absent. Remedy : install h5py built with "
                "MPI (parallel HDF5), or parallel=False (global gather + rank-0 write).")
        if not h5py.get_config().mpi:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : h5py present but WITHOUT MPI support "
                "(h5py.get_config().mpi == False). Remedy : install h5py built with MPI (parallel "
                "HDF5), or parallel=False (global gather + rank-0 write).")
        try:
            from mpi4py import MPI
        except ImportError:
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : mpi4py absent (required to open in mpio). "
                "Remedy : install mpi4py, or parallel=False (global gather + rank-0 write).")
        # Guard : _pops module built BEFORE the local accessors (build prior to ADC-66).
        if not hasattr(self._s, "local_boxes"):
            raise RuntimeError(
                "write(format='hdf5', parallel=True) : the loaded _pops module does not expose "
                "local_boxes/local_state (build prior to hyperslab writes). Remedy : "
                "rebuild adc_cpp, or parallel=False.")
        comm = MPI.COMM_WORLD
        rank0 = (_pops.my_rank() == 0)
        # phi : solved + gathered COLLECTIVELY (all ranks ; potential_global does the all_reduce),
        # then written by rank 0 alone (global scalar field, contiguous dataset).
        phi_g = np.asarray(self._s.potential_global(), dtype=np.float64).reshape(nyv, nxv)
        # Descriptors identical on all ranks (shared composition) : pre-computed for coherent
        # collective operations (create_dataset / attrs).
        ncomp = {b: self._s.n_vars(b) for b in blocks}
        names = {b: [s.encode() for s in self._s.variable_names(b, "conservative")] for b in blocks}
        roles = {b: [s.encode() for s in self._s.variable_roles(b, "conservative")] for b in blocks}
        tmp = target + ".tmp"
        # COLLECTIVE open (all ranks open the same file via mpio).
        f = h5py.File(tmp, "w", driver="mpio", comm=comm)
        try:
            # Collective metadata -- identical to the serial path.
            f.attrs["t"] = self._s.time()
            f.attrs["macro_step"] = self._s.macro_step()
            f.attrs["nx"] = nxv
            f.attrs["ny"] = nyv
            f.attrs["abi_key"] = abi_key()
            for b in blocks:
                g = f.create_group(b)  # collective
                # GLOBAL dataset (ncomp, ny, nx) CONTIGUOUS (no gzip : independent writes forbidden
                # on a chunk-filtered dataset in parallel).
                dset = g.create_dataset("state", shape=(ncomp[b], nyv, nxv), dtype="f8")  # collective
                g.attrs["names"] = names[b]
                g.attrs["roles"] = roles[b]
                # Each rank writes ITS local boxes as hyperslabs (independent I/O : disjoint
                # boxes, a rank with no box -> empty loop). local_state already returns (ncomp, bny, bnx).
                for li, (ilo, jlo, ihi, jhi) in enumerate(self._s.local_boxes(b)):
                    dset[:, jlo:jhi + 1, ilo:ihi + 1] = np.asarray(
                        self._s.local_state(b, li), dtype=np.float64)
            phi_d = f.create_dataset("phi", shape=(nyv, nxv), dtype="f8")  # collective
            if rank0:
                phi_d[...] = phi_g  # global field already gathered : written by rank 0 alone
        finally:
            f.close()  # collective
        comm.Barrier()  # all ranks have closed BEFORE the atomic rename
        if rank0:
            os.replace(tmp, target)
        comm.Barrier()  # the rename is visible (shared FS) before any return
        return target

    def checkpoint(self, path, parallel=False):
        """RESTARTABLE CHECKPOINT v1 (npz) : COMPLETE block state + clock (t, macro_step --
        MANDATORY for the stride cadence) + grid + provenance (abi_key). CONTRACT (cf.
        docs/IO_CHECKPOINT_PLAN.md) : restart does NOT rebuild the composition -- the user
        script replays its add_block/set_poisson/couplings then calls sim.restart(path), which
        VERIFIES the consistency (blocks, sizes) and raises an explicit error otherwise. @return the path.

        MULTI-RANK (MPI np>1) : the states are gathered by the collective GLOBAL accessors
        (state_global / potential_global -- all ranks MUST call checkpoint), then ONLY
        rank 0 writes the SINGLE file (identical to mono-rank). The checkpoint/restart pair stays
        bit-identical under np>1 (mono-box System : all the state lives on rank 0, exact gather).

        @p parallel : the v1 checkpoint is ALWAYS rank-0 gather (npz format, not HDF5). The hyperslab
        write (parallel=True) only applies to the visualization OUTPUT
        write(format='hdf5') : an npz checkpoint has neither HDF5 datasets nor box partitioning. Passing
        parallel=True therefore raises an EXPLICIT error (never a silently degraded write) : for a
        parallel output, use write(format='hdf5', parallel=True) ; a restartable parallel HDF5
        checkpoint is later work (PR-IO-3, cf. docs/IO_CHECKPOINT_PLAN.md)."""
        import os
        import numpy as np
        from pops import _pops
        if parallel:
            raise NotImplementedError(
                "checkpoint(parallel=True) : the v1 checkpoint is a rank-0 gather npz (non "
                "HDF5 format, no box partitioning). The hyperslab write only concerns "
                "write(format='hdf5', parallel=True) (visualization output). A restartable parallel "
                "HDF5 checkpoint remains to be done (PR-IO-3, docs/IO_CHECKPOINT_PLAN.md) ; for "
                "now : checkpoint(parallel=False).")
        blocks = list(self._s.block_names())
        out = {"pops_checkpoint_version": 1,
               "t": self._s.time(), "macro_step": self._s.macro_step(),
               "nx": self._s.nx(), "ny": self._s.ny(),
               "abi_key": abi_key(), "blocks": np.array(blocks)}
        # COLLECTIVE gather (all ranks) BEFORE the rank-0 guard.
        for b in blocks:
            nv = self._s.n_vars(b)
            out["ncomp_" + b] = nv
            out["state_" + b] = np.asarray(self._s.state_global(b), dtype=np.float64)
            out["names_" + b] = np.array(list(self._s.variable_names(b, "conservative")))
        # phi : multigrid warm start (BIT-IDENTICAL restart) ; physical STATE if
        # gauss_policy="evolve" (phi is no longer re-derived from rho there).
        out["phi"] = np.asarray(self._s.potential_global(), dtype=np.float64)
        # COMPILED-PROGRAM HISTORIES (ADC-406b): a compiled time Program with multistep histories (e.g.
        # Adams-Bashforth 2) carries the System-owned ring buffers across macro-steps. To make a
        # (run, checkpoint, restart, continue) run bit-identical to a continuous run, the rings (the
        # previous RHS R_{n-1}, ...) MUST survive the checkpoint -- else AB2 cold-starts again and
        # diverges. The program HASH is recorded too: a restart against a DIFFERENT compiled Program is
        # rejected (the buffers / cadence would be meaningless). Both groups of keys are OPTIONAL: a
        # checkpoint with no installed program / no history restarts exactly as before (back-compatible).
        prog_hash = ""
        if hasattr(self._s, "installed_program_hash"):
            prog_hash = self._s.installed_program_hash()
        if prog_hash:
            out["program_hash"] = prog_hash
        hist_names = list(self._s.history_names()) if hasattr(self._s, "history_names") else []
        if hist_names:
            out["history_names"] = np.array(hist_names)
            for hname in hist_names:
                depth = int(self._s.history_depth(hname))
                out["history_depth_" + hname] = depth
                out["history_ncomp_" + hname] = int(self._s.history_ncomp(hname))
                out["history_init_" + hname] = bool(self._s.history_initialized(hname))
                # COLLECTIVE gather of every slot (all ranks call), like state_global above.
                for k in range(depth):
                    out["history_%s_%d" % (hname, k)] = np.asarray(
                        self._s.history_global(hname, k), dtype=np.float64)
        # SCHEDULER VALUE CACHE (ADC-458, Spec 3 section 30): a compiled Program with a held schedule
        # (every(N).hold / accumulate_dt) caches the System aux / a scratch per node so the field solve
        # runs only when DUE. The cache lives in the System (program_cache, NOT the .so step closure),
        # so checkpointing it makes a (run, checkpoint, restart, continue) run bit-for-bit identical to
        # a continuous run -- else the first post-restart step would cold-start the held node off its
        # cadence. Keys are OPTIONAL (a program with no held schedule caches nothing): cache_nodes is
        # the list of cached IR node ids, and per node the cached value + bookkeeping. All ranks gather
        # (program_cache_global mirrors history_global), only rank 0 writes (below).
        cache_nodes = (list(self._s.program_cache_nodes())
                       if hasattr(self._s, "program_cache_nodes") else [])
        if cache_nodes:
            out["cache_nodes"] = np.array(cache_nodes, dtype=np.int64)
            out["cache_names"] = np.array([str(self._s.program_cache_name(nid))
                                           for nid in cache_nodes])
            for nid in cache_nodes:
                out["cache_ncomp_%d" % nid] = int(self._s.program_cache_ncomp(nid))
                out["cache_ngrow_%d" % nid] = int(self._s.program_cache_ngrow(nid))
                out["cache_last_update_%d" % nid] = int(
                    self._s.program_cache_last_update_step(nid))
                out["cache_accum_dt_%d" % nid] = float(
                    self._s.program_cache_accumulated_dt(nid))
                # COLLECTIVE gather of the cached value (all ranks call), like history_global above.
                out["cache_value_%d" % nid] = np.asarray(
                    self._s.program_cache_global(nid), dtype=np.float64)
        target = path if path.endswith(".npz") else path + ".npz"
        if _pops.my_rank() != 0:
            return target  # only rank 0 writes the checkpoint (gather already done)
        tmp = target + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(tmp, target)
        return target

    def restart(self, path):
        """RESUMES a v1 checkpoint : VERIFIES the composition (same blocks, same sizes -- explicit
        error otherwise, never a silently wrong resume), restores the state of each block
        then the clock (t, macro_step : the stride cadence resumes exactly). The COMPOSITION
        (add_block / set_poisson / set_magnetic_field / couplings) must have been replayed by the
        script BEFORE the call (v1 contract, cf. checkpoint).

        MULTI-RANK (MPI np>1) : all ranks read the file (shared file system) and
        call set_state / set_potential / set_clock. set_state / set_potential are MPI-safe (the
        owner rank -- rank 0, mono-box -- writes, the others are no-ops) ; set_clock sets
        the clock on each rank. The resume is therefore bit-identical under np>1."""
        import numpy as np
        target = path if path.endswith(".npz") else path + ".npz"
        d = np.load(target, allow_pickle=False)
        if int(d["pops_checkpoint_version"]) != 1:
            raise ValueError("restart : checkpoint version %r not supported (expected 1)"
                             % (d["pops_checkpoint_version"],))
        if int(d["nx"]) != self._s.nx() or int(d["ny"]) != self._s.ny():
            raise ValueError("restart : checkpoint grid (%d x %d) != system (%d x %d)"
                             % (int(d["nx"]), int(d["ny"]), self._s.nx(), self._s.ny()))
        chk_blocks = [str(b) for b in d["blocks"]]
        cur_blocks = list(self._s.block_names())
        if chk_blocks != cur_blocks:
            raise ValueError("restart : checkpoint blocks %r != current composition %r "
                             "(replay the SAME composition before restart)" % (chk_blocks, cur_blocks))
        for b in chk_blocks:
            if int(d["ncomp_" + b]) != self._s.n_vars(b):
                raise ValueError("restart : block '%s' has %d components in the checkpoint, %d here"
                                 % (b, int(d["ncomp_" + b]), self._s.n_vars(b)))
            self._s.set_state(b, np.asarray(d["state_" + b], dtype=np.float64))
        # phi BEFORE the clock : warm start of the restored solver (bit-identical restart ; physical
        # state in gauss_policy="evolve").
        if "phi" in d:
            self._s.set_potential(np.asarray(d["phi"], dtype=np.float64).ravel())
        # COMPILED-PROGRAM HASH GUARD (ADC-406b): if the checkpoint recorded an installed program hash,
        # the user must have RE-INSTALLED the SAME compiled Program before restart (the v1 replay
        # contract). A different Program (different IR hash) makes the restored histories / cadence
        # meaningless -> fail loud rather than silently continue with the wrong scheme.
        if "program_hash" in d:
            chk_hash = str(d["program_hash"])
            cur_hash = (self._s.installed_program_hash()
                        if hasattr(self._s, "installed_program_hash") else "")
            if cur_hash != chk_hash:
                raise RuntimeError("checkpoint was created with a different compiled Program hash")
        # COMPILED-PROGRAM HISTORIES (ADC-406b): restore each ring (the previous RHS, ...) so a
        # multistep scheme (AB2) resumes EXACTLY where it stopped -- continuous == (run, ckpt, restart,
        # continue) bit-for-bit. The program re-registers the rings on its first post-restart step;
        # restoring them here (before that step) seeds the slots and the initialized flag so the first
        # post-restart read sees the true R_{n-1} (no phantom cold-start re-fill).
        available = set(str(h) for h in d["history_names"]) if "history_names" in d else set()
        # MISSING-HISTORY GUARD (ADC-414, spec error 18): a history the CURRENT program already
        # registered (it stepped at least once before this restart, or was re-installed and run) but the
        # checkpoint never recorded cannot be restored -> the multistep scheme would silently cold-start.
        # Fail loud, distinct from the hash-mismatch message above. A fresh program that has not yet
        # registered any ring (the common install-then-restart flow) has no required history -> no false
        # positive; the rings it re-registers on its first post-restart step are restored below.
        if hasattr(self._s, "history_names"):
            for hname in self._s.history_names():
                if hname not in available:
                    raise RuntimeError(
                        "checkpoint does not contain required Program history '%s'" % hname)
        if available:
            for hname in (str(h) for h in d["history_names"]):
                depth = int(d["history_depth_" + hname])
                for k in range(depth):
                    self._s.restore_history(
                        hname, k, np.asarray(d["history_%s_%d" % (hname, k)], dtype=np.float64))
                self._s.set_history_initialized(hname, bool(d["history_init_" + hname]))
        # SCHEDULER VALUE CACHE (ADC-458, Spec 3 section 30): restore each held node's cached value +
        # bookkeeping so a held schedule resumes EXACTLY on its cadence -- continuous == (run, ckpt,
        # restart, continue) bit-for-bit. The cache is keyed by IR node id (re-keyed by the re-installed
        # Program, which uses the SAME ids -- guaranteed by the program-hash guard above). A checkpoint
        # that lists a cache node but lost its value array is a CORRUPT/TRUNCATED checkpoint: fail loud
        # with the verbatim spec message naming the node, distinct from the hash-mismatch above (the
        # held node would otherwise silently cold-start off its cadence). Back-compatible: a checkpoint
        # with no cache_nodes restores as before.
        if "cache_nodes" in d and hasattr(self._s, "restore_program_cache"):
            cache_names = ([str(nm) for nm in d["cache_names"]]
                           if "cache_names" in d else None)
            for idx, nid in enumerate(int(n) for n in d["cache_nodes"]):
                name = (cache_names[idx] if cache_names is not None and idx < len(cache_names)
                        else "node_%d" % nid)
                value_key = "cache_value_%d" % nid
                if value_key not in d:
                    raise RuntimeError(
                        "checkpoint missing cached value for scheduled node '%s'" % name)
                # ngrow defaults to 1 for a pre-ngrow-key checkpoint (the aux MVP width) so older
                # checkpoints still restore; a held scratch (ngrow 2) carries its own key.
                ngrow = int(d["cache_ngrow_%d" % nid]) if ("cache_ngrow_%d" % nid) in d else 1
                self._s.restore_program_cache(
                    nid, int(d["cache_ncomp_%d" % nid]), ngrow,
                    int(d["cache_last_update_%d" % nid]),
                    float(d["cache_accum_dt_%d" % nid]), name,
                    np.asarray(d[value_key], dtype=np.float64))
        self._s.set_clock(float(d["t"]), int(d["macro_step"]))

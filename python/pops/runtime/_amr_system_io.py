"""AmrSystem IO mixin (Spec-4 PR-F): AMR outputs, checkpoint, restart.

Visualization output (coarse fields + patch footprints) and the bit-identical single-block
single-rank v1 checkpoint/restart of :class:`pops.runtime.amr_system.AmrSystem`. Mixed in via
inheritance; operates on ``self._s``, ``self._L`` and ``self._regrid_every``. ``abi_key`` is the
module ABI key re-exported through ``pops.runtime.bricks``.
"""

from pops.runtime.bricks import abi_key


class _AmrSystemIO:
    """Output / checkpoint / restart methods of AmrSystem."""

    def write(self, path, format="npz", step=None):
        """AMR VISUALIZATION OUTPUT (wave 3) : COARSE fields per block + phi + footprints of the
        fine patches. format='npz' (per-block densities, phi, patch_rectangles, t) or 'vtk' (.vti of
        the COARSE : per-block density + phi -- the fine patches are provided in npz via their
        rectangles, the multi-resolution VTK = PR-IO-3). @p step : numbered suffix. @return path."""
        import os
        import numpy as np
        n = self._s.nx()
        suffix = ("_%06d" % int(step)) if step is not None else ""
        # EACH block, by its name (binding AmrSystem::block_names, parity with System) : in multi-block,
        # density() without a name would read ONLY block 0 and would lose the others SILENTLY.
        names = list(self._s.block_names())
        if not names:
            names = [""]
        if format == "npz":
            out = {"t": self._s.time(), "n": n,
                   "patch_rectangles": np.array(self.patch_rectangles(), dtype=np.float64)
                   if self.patch_rectangles() else np.zeros((0, 4))}
            for b in names:
                key = b if b else "block"
                out["density_" + key] = np.asarray(self.density(b) if b else self.density(),
                                                   dtype=np.float64)
            out["phi"] = np.asarray(self.potential(), dtype=np.float64)
            target = path + suffix + ".npz"
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **out)
            os.replace(tmp, target)
            return target
        if format == "vtk":
            target = path + suffix + ".vti"
            arrays, labels = [], []
            for b in names:
                key = b if b else "block"
                arrays.append(np.asarray(self.density(b) if b else self.density(),
                                         dtype=np.float64).reshape(n, n))
                labels.append("%s_density" % key)
            arrays.append(np.asarray(self.potential(), dtype=np.float64).reshape(n, n))
            labels.append("phi")
            lines = ['<?xml version="1.0"?>',
                     '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',
                     '  <ImageData WholeExtent="0 %d 0 %d 0 0" Origin="0 0 0" '
                     'Spacing="%.17g %.17g 1">' % (n, n, self._L / n, self._L / n),
                     '    <Piece Extent="0 %d 0 %d 0 0">' % (n, n),
                     '      <CellData>']
            for nm, arr in zip(labels, arrays):
                lines.append('        <DataArray type="Float64" Name="%s" format="ascii">' % nm)
                lines.append("          " + " ".join("%.17g" % v for v in arr.ravel()))
                lines.append('        </DataArray>')
            lines += ['      </CellData>', '    </Piece>', '  </ImageData>', '</VTKFile>', '']
            tmp = target + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(lines))
            os.replace(tmp, target)
            return target
        raise ValueError("AmrSystem.write : format 'npz' | 'vtk' (received %r)" % (format,))

    def checkpoint(self, path):
        """RESTARTABLE BIT-IDENTICAL AMR CHECKPOINT v1 (npz), SINGLE-BLOCK SINGLE-RANK (ADC-65). Writes
        the FULL CONSERVATIVE STATE of EACH level (all components ; the coarse AND the fine patches,
        valid cells), the phi of each level (level 0 = WARM-START of the multigrid,
        load-bearing for the bit-identical resume), the HIERARCHY (patch_boxes), the clock (t,
        macro_step) and the regrid cadence. CONTRACT (parity with System.checkpoint) : restart does NOT
        rebuild the composition -- the script replays its add_block/set_poisson/set_refinement/set_density
        then calls sim.restart(path), which CHECKS consistency and raises otherwise. @return the path.

        SCOPE (EXPLICIT rejections, never a silently wrong/partial checkpoint) :
          - SINGLE-BLOCK only : multi-block (AmrRuntime engine) shares layout AND aux between blocks
            and does not expose the per-level/block state -> follow-up (the C++ accessors also reject).
          - SINGLE-RANK (np == 1) : the level accessors read the LOCAL fabs without an MPI gather ;
            a per-level gather (BoxArray + DistributionMapping) is a follow-up.
          - regrid_every == 0 : a bit-identical resume requires a FROZEN hierarchy (otherwise the regrid
            would re-diverge after the restart). We reject at the checkpoint (early failure, clear message).

        Out-of-scope fallback : AmrSystem.write (visualization) or a single-level System."""
        import os
        import numpy as np
        from pops import _pops
        if _pops.n_ranks() != 1:
            raise NotImplementedError(
                "AmrSystem.checkpoint : MPI np>1 not wired (ADC-65 single-rank : the per-level states "
                "are read on the LOCAL fabs, the per-level gather = follow-up). Run single-rank, or "
                "use a single-level System (bit-identical checkpoint/restart including under MPI).")
        if self._s.n_blocks() != 1:
            raise NotImplementedError(
                "AmrSystem.checkpoint : multi-block not wired (ADC-65 single-block : the AmrRuntime engine "
                "shares layout AND aux between blocks and does not expose the per-level/block state = follow-up). "
                "Use a single add_block, or a single-level System (bit-identical checkpoint/restart).")
        if self._regrid_every != 0:
            raise ValueError(
                "AmrSystem.checkpoint : bit-identical resume wired for regrid_every == 0 only "
                "(frozen hierarchy) ; this system has regrid_every=%d (the post-restart regrid would re-diverge "
                "the hierarchy). Rebuild the system with regrid_every=0." % self._regrid_every)
        nlev = int(self._s.n_levels())
        pb = self._s.patch_boxes()  # (level, ilo, jlo, ihi, jhi) inclusive, index space of the level
        out = {"pops_amr_checkpoint_version": 1,
               "t": self._s.time(), "macro_step": self._s.macro_step(),
               "n": self._s.nx(), "L": self._L, "regrid_every": self._regrid_every,
               "abi_key": abi_key(), "blocks": np.array(list(self._s.block_names())),
               "n_vars": int(self._s.n_vars()), "n_levels": nlev,
               "patch_boxes": (np.asarray(pb, dtype=np.int64) if pb
                               else np.zeros((0, 5), dtype=np.int64))}
        for k in range(nlev):
            # FULL conservative state of level k (c*nf*nf + j*nf + i) + phi (nf*nf). Fine level : only
            # the patch cells are defined (0 elsewhere) ; the restart only rewrites those cells.
            out["state_%d" % k] = np.asarray(self._s.level_state(k), dtype=np.float64)
            out["phi_%d" % k] = np.asarray(self._s.level_potential(k), dtype=np.float64)
        target = path if path.endswith(".npz") else path + ".npz"
        tmp = target + ".tmp"  # ATOMIC write (.tmp + os.replace : a crash corrupts nothing)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(tmp, target)
        return target

    def restart(self, path):
        """RESUMES an AMR v1 checkpoint (BIT-IDENTICAL, SINGLE-BLOCK SINGLE-RANK, ADC-65). CHECKS
        consistency (version, grid, blocks, components, regrid_every == 0) then : (1) IMPOSES the
        saved fine hierarchy (set_hierarchy, instead of Berger-Rigoutsos clustering) ; (2) restores
        the FULL conservative state of each level AS-IS (no re-prolongation) ; (3) restores the
        phi of each level (level 0 = warm-start of the multigrid -> the 1st solve post-restart
        starts from the same guess) ; (4) restores the clock (t, macro_step). The COMPOSITION (add_block /
        set_poisson / set_refinement / set_density) must have been REPLAYED by the script BEFORE the call.

        ORDER : set_hierarchy BEFORE set_level_state (imposing the layout precedes restoring the
        valid cells) ; phi and clock after. The 1st step replays update() (sync_down + warm-start solve)
        then advance -- the ghosts (coarse AND fine) are remade by the step, exactly
        like after a regrid, hence the bit-identical resume without restoring any ghosts."""
        import numpy as np
        from pops import _pops
        if _pops.n_ranks() != 1:
            raise NotImplementedError(
                "AmrSystem.restart : MPI np>1 not wired (ADC-65 single-rank ; cf. checkpoint). Run "
                "single-rank, or use a single-level System.")
        if self._s.n_blocks() != 1:
            raise NotImplementedError(
                "AmrSystem.restart : multi-block not wired (ADC-65 single-block ; cf. checkpoint). Use "
                "a single add_block, or a single-level System.")
        if self._regrid_every != 0:
            raise ValueError(
                "AmrSystem.restart : requires regrid_every == 0 (frozen hierarchy ; otherwise the regrid "
                "post-restart would re-diverge the restored hierarchy). Rebuild the system with "
                "regrid_every=0 before restart. (current regrid_every = %d)" % self._regrid_every)
        target = path if path.endswith(".npz") else path + ".npz"
        d = np.load(target, allow_pickle=False)
        if int(d["pops_amr_checkpoint_version"]) != 1:
            raise ValueError("restart : AMR checkpoint version %r not supported (expected 1)"
                             % (d["pops_amr_checkpoint_version"],))
        if int(d["n"]) != self._s.nx():
            raise ValueError("restart : checkpoint grid (n=%d) != system (n=%d)"
                             % (int(d["n"]), self._s.nx()))
        if float(d["L"]) != self._L:
            raise ValueError("restart : checkpoint domain (L=%r) != system (L=%r) -- different dx"
                             % (float(d["L"]), self._L))
        if int(d["regrid_every"]) != 0:
            raise ValueError("restart : checkpoint taken with regrid_every=%d != 0 (bit-identical "
                             "resume impossible)" % int(d["regrid_every"]))
        chk_blocks = [str(b) for b in d["blocks"]]
        cur_blocks = list(self._s.block_names())
        if chk_blocks != cur_blocks:
            raise ValueError("restart : checkpoint blocks %r != current composition %r "
                             "(replay the SAME composition before restart)" % (chk_blocks, cur_blocks))
        if int(d["n_vars"]) != int(self._s.n_vars()):
            raise ValueError("restart : %d components in the checkpoint, %d here"
                             % (int(d["n_vars"]), int(self._s.n_vars())))
        nlev = int(d["n_levels"])
        if nlev != int(self._s.n_levels()):
            raise ValueError("restart : %d levels in the checkpoint, %d here (does the composition / the "
                             "refinement differ ?)" % (nlev, int(self._s.n_levels())))
        # (1) IMPOSE the saved fine hierarchy (the coupler filters level 1), except a
        # SINGLE-LEVEL hierarchy (n_levels == 1, e.g. amr-schur path with no fine patch) : nothing to impose then.
        boxes = [tuple(int(x) for x in row) for row in np.asarray(d["patch_boxes"], dtype=np.int64)]
        if nlev >= 2:
            if not any(b[0] == 1 for b in boxes):
                raise ValueError("restart : %d-level hierarchy but no fine patch (level 1) "
                                 "in the checkpoint (inconsistent)." % nlev)
            self._s.set_hierarchy(boxes)
        # (2) restore the FULL conservative state of each level AS-IS (no re-prolongation) ;
        # set_level_state flattens the array and only writes the valid cells (the patches).
        for k in range(nlev):
            self._s.set_level_state(k, np.asarray(d["state_%d" % k], dtype=np.float64))
        # (3) restore the phi (level 0 = warm-start of the multigrid : bit-identical resume).
        for k in range(nlev):
            self._s.set_level_potential(k, np.asarray(d["phi_%d" % k], dtype=np.float64).ravel())
        # (4) restore the clock AFTER the state (parity with System ; macro_step advances the cadence phase).
        self._s.set_clock(float(d["t"]), int(d["macro_step"]))

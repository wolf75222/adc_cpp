# Outputs / checkpoint / restart -- design plan (audit 2026-06, work item 7)

Status : **PLAN** (target API + HPC constraints + breakdown into PRs). Nothing is wired in this
document ; it fixes the contract BEFORE the implementation to avoid an ad hoc output API per case
(today each adc_cases case writes its .npy / .png by hand via `density()` / `get_state()`).

## Target user API

```python
sim.write("out/run", format="hdf5", step=k)     # un dump de visualisation (champ + metadonnees)
sim.write("out/run", format="vtk", step=k)      # idem, lisible par ParaView/VisIt sans h5py
sim.checkpoint("out/chk_000400")                # etat COMPLET redemarrable
sim.restart("out/chk_000400")                   # reprend exactement (memes blocs deja composes)
```

- `write` = VISUALIZATION OUTPUT : loss acceptable (subset of fields, possibly
  subsampled), open formats (HDF5/XDMF or VTK), ONE user cadence (`step=` names the
  file, the user drives the frequency).
- `checkpoint` / `restart` = EXACT RESUME : everything needed to resume the run
  bit-for-bit (modulo non-deterministic backend reductions), NO loss.

## Minimal content of a checkpoint

| Element | Current source | Note |
|---|---|---|
| time `t` | `System::time()` | exact double |
| counter `macro_step_` | Impl | REQUIRED : the stride cadence (hold-then-catch-up) depends on it |
| mesh (n, L, geometry, nr/ntheta, r_min/r_max, periodic) | SystemConfig | re-validated at restart (mismatch = explicit error) |
| blocks : name, ncomp, substeps, stride, evolve, gamma | BlockState | the insertion ORDER must be preserved (indexing) |
| state U of each block (all components, valid cells) | MultiFab | the ghosts are reconstructed (fill_boundary) |
| shared aux (phi, grad, B_z, T_e, width) | Impl::aux + bz_field_ | B_z is an INPUT (not re-derivable) ; phi re-derivable but saving it preserves the warm start (gauss_policy="evolve" : REQUIRED, phi is no longer re-derived !) |
| model parameters (ModelSpec / runtime params of the .so) | spec + block_params_ | the .so themselves are NOT embedded : we save model_hash + so_path for verification |
| scheduler value cache (held every(N).hold / accumulate_dt nodes) | System::program_cache (CacheManager) | per node the cached aux/scratch + last_update_step + accumulated_dt ; serialized like the history rings (ADC-458 section 30), so a held node resumes on its cadence ; a missing cached value names the node fail-loud |
| temporal policy (scheme lie/strang, gauss_policy) | stepper/fields | |
| Newton options / global dt bounds | add_block / add_dt_bound | the Python `add_dt_bound` callbacks are NOT serializable : document "to be re-set after restart" |

Assumed decision : `restart` DOES NOT REBUILD the composition (the `add_block` / `set_poisson` /
couplings remain the responsibility of the user script, which replays its composition then calls
`restart`). The checkpoint VERIFIES consistency (same blocks, same sizes, same model_hash if
available) and raises an explicit error otherwise. This is the simplest contract that avoids
serializing C++/Python closures.

## HPC constraints

1. **Not one file per process on large runs.** Three levels, from simplest to most
   scalable :
   - V1 (single-rank / small runs) : a single file written by rank 0 after gather
     (reuses the existing `copy_state` marshaling). Sufficient for the current local cases
     (System = one box).
   - V2 (medium MPI) : sequential HDF5 by AGGREGATION (rank 0 collects piece by piece, writes in
     streaming) -- no MPI-IO dependency, bounded memory.
   - V3 (production GPFS/Lustre, cf. ROMEO) : PARALLEL HDF5 (h5py-mpi / native HDF5) with a
     GLOBAL dataset per block, hyperslabs per rank. **DONE on the System side (write) : ADC-66 /
     PR-IO-3.** OPT-IN via `sim.write(format="hdf5", parallel=True)` : collective opening
     `h5py.File(driver="mpio", comm=COMM_WORLD)`, global datasets `(ncomp, ny, nx)` created
     collectively, each rank writes ITS boxes in hyperslabs. NO CMake flag nor C++ HDF5
     dependency : everything goes through h5py-mpi on the Python side (absent / without MPI -> CLEAR error with remedy,
     never a silent write) ; the only C++ addition is a pair of minimal NON-collective accessors
     `System::local_boxes` / `System::local_state`. `parallel=False` (default) = V1 path
     rank-0 gather UNCHANGED. **The cartesian System being SINGLE-BOX** (one box, rank 0), true
     hyperslab parallelism only appears on a MULTI-BOX geometry -- the AMR (one
     group/dataset per level + boxes) remains to be done (ADC-65).
2. **Layout** : datasets `(ncomp, ny, nx)` (component-major, consistent with `get_state`),
   HDF5 attributes for the metadata (t, macro_step, config, variable names/roles).
   AMR : one group per level + boxes (format inspired by AMReX plotfiles, but HDF5).
3. **Atomicity** : write into `<path>.tmp` then rename (a checkpoint corrupted by a crash during
   writing must not overwrite the previous one).
4. **VTK** : `.vti` format (ImageData) sufficient in uniform cartesian ; polar -> `.vts`
   (StructuredGrid r/theta). Written on the Python side (numpy -> binary), without a heavy dependency.

## Breakdown into PRs

1. **PR-IO-1** : `sim.write(path, format="vtk"|"npz", step=)` on the pure PYTHON side (reads get_state /
   potential / variable_names ; zero C++ change). Gives a uniform visualizable output right away
   to the adc_cases cases. `format="npz"` = de facto single-rank checkpoint-lite.
2. **PR-IO-2** : `sim.checkpoint(path)` / `sim.restart(path)` Python (npz or hdf5 if h5py present),
   verification contract above. REQUIRES exposing `macro_step()` on the bindings side (trivial) and a
   controlled `set_time(t, macro_step)` (restores the stride cadence).
3. **PR-IO-3** : aggregated/parallel HDF5 + AMR (multi-level) + large ROMEO runs. Aggregated HDF5 (V1) =
   DONE (PR-IO-2). PARALLEL HDF5 by hyperslabs on the System `write` side = **DONE (ADC-66)** :
   `sim.write(format="hdf5", parallel=True)` (h5py mpio + mpi4py, opt-in ; minimal C++ accessors
   `local_boxes`/`local_state`). REMAINS : multi-level AMR (ADC-65), and a restartable HDF5
   parallel CHECKPOINT (the checkpoint remains npz gather-rank-0 ; `checkpoint(parallel=True)` raises).

## Non-objectives (explicit)

- No serialization of the compositions (blocks/couplings) nor of the Python callbacks.
- No proprietary format : HDF5/VTK/NPZ only.
- Cross-version restart (different pops headers) is NOT guaranteed : the checkpoint embeds
  `abi_key`/`model_hash` to DETECT it, not to convert it.

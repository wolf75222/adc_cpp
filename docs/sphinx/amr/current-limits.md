# Current limits


What the AMR does not do yet.

- **Two levels only.** The hierarchy is coarse + one fine level (ratio 2). The regrid only
  rebuilds the finest level; beyond 2 levels, multi-level regrid does not exist
  yet, even in single-block.
- **Poisson "coarse + inject".** The Poisson is solved on the coarse then injected toward the
  fine, it is not a multi-level composite elliptic solve. This is sufficient for
  the diocotron observable (which lives on a median circle resolved by the coarse) but worth knowing.
- **No global Schur source stage on AMR.** The Schur-condensed source splitting (`adc.Split`,
  `CondensedSchur`) has no AMR counterpart: `AmrSystem.add_block` / `add_equation`
  reject it explicitly. For this stage, use a non-refined `System`.
- **Multirate via the compiled path: restricted.** On the "production" DSL path (`.so`),
  `add_equation` explicitly rejects `stride > 1` and the partial IMEX mask
  (`implicit_vars` / `implicit_roles`): the flat ABI of the loader does not carry them, and they
  would silently be taken at their default values. For a multirate or partial-IMEX-mask `.so`,
  go through native `add_block` (`adc.Model(...)`), which exposes them.
- **Elliptic solver.** On AMR, the solver is always the geometric multigrid
  (`geometric_mg`); no FFT. The right-hand side is the sum of the elliptic bricks of the blocks.
- **Validation: what is tested vs ROMEO only.** The multi-block AMR is covered by the
  CPU tests (Serial / OpenMP) and the MPI parity np=1/2/4 in this repository. The GPU validation
  (GH200) of the AMR paths is done manually on ROMEO (the path is device-clean by
  construction, named functors); see [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) for the
  test / backend cross-reference line by line.
- **Cut-cell / polar out of AMR scope.** The cut-cell walls and the polar geometry are
  worksites of the `System` (single-level); they are not carried on the AMR hierarchy.

Design frontier (Phase 2 / Phase 3): per-block refinement criteria, multi-level composite
elliptic solve, and (much further out) distinct hierarchies per species
with conservative projections. Detail:
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) section 7.

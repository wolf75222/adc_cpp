# Multi-block AMR


`AmrSystem` is a multi-block facade: you call `add_block` (native bricks) or
`add_equation` (compiled DSL model) once per species, exactly as on `System`. All the
blocks share the hierarchy, the aux and the coarse Poisson (co-located summed right-hand side),
but each keeps its own spatial scheme (limiter x flux x reconstruction), its temporal
treatment (`explicit` or `imex`), and its multirate (`substeps` / `stride`). The runtime engine is
`AmrRuntime` (type-erased registry by block name), refined counterpart of the multi-block
single-level engine of `System`.

```python
sim = adc.AmrSystem(n=96, L=1.0, periodic=True)

electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0),
                      elliptic=adc.ChargeDensity(charge=-1.0))
ions = adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                 transport=adc.IsothermalFlux(),
                 source=adc.PotentialForce(charge=+1.0),
                 elliptic=adc.ChargeDensity(charge=+1.0))

sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))
sim.add_block("ions", model=ions,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_refinement(0.05)
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne0)
sim.set_density("ions", np.ones((96, 96)))

sim.advance(0.001, 8)
print("blocs :", sim.n_blocks(), "| masse e- :", sim.mass("electrons"))
```

Inter-species coupled sources (ionization, collisions) can be wired in via
`add_coupled_source`: they are read cell by cell (same `(i, j)`, no interpolation
between species, thanks to the shared hierarchy) and, built with exactly opposite
contributions, conserve the pair mass to machine precision. The multi-block is validated by a
battery of tests called "capstone": two blocks with different schemes (`test_amr_system_twoblock`),
production multi-block DSL (`test_amr_multiblock_compiled`), IMEX (`test_amr_multiblock_imex`),
coupled sources (`test_amr_multiblock_coupled_source`), substeps (`test_amr_multiblock_substeps`),
union regrid (`test_amr_multiblock_regrid_union`) and MPI parity np=1/2/4
(`test_mpi_amr_twoblock_parity`). Detail: [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) section 8 and
the banner "STATUT : implemente" at the top of [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

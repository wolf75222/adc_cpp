# AMR (adaptive refinement)

`adc.AmrSystem` is the refined counterpart of `adc.System`: one or more blocks (species)
carried on a block-structured AMR hierarchy (with rectangular boxes, AMReX /
FLASH / SAMRAI style). The mesh is refined where the solution requires it, and only there. This
page summarizes how to drive the AMR from Python; for design details see
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) (section 8), [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
(sections 13-15) and the design notes
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) /
[AMR_REGRID_UNION_TAGS_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_REGRID_UNION_TAGS_DESIGN.md).

The API is identical to that of `System` (same `add_block` / `add_equation` /
`set_poisson` / `set_density` / `step_cfl`): you refine an existing case by changing
`adc.System(...)` into `adc.AmrSystem(...)` and adding a refinement criterion. The
A->Z tutorial moreover compares the two paths on the same physics (cf.
[tutorials/diocotron_tutorial.py](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py), function
`uniform_vs_amr`).

```{toctree}
:maxdepth: 1

shared-hierarchy
tagging-regrid
prolongation-restriction
reflux
multi-block-amr
current-limits
```

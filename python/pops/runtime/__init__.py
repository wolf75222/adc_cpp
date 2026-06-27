"""pops.runtime : the runtime layer of the PoPS bindings (Spec-4 PR-F).

This is the TOP layer and the ONLY layer permitted to import the ``_pops`` extension (via
``pops._bootstrap``). It owns the user-facing runtime objects:

- the systems ``System`` / ``AmrSystem`` (compose blocks, share a Poisson, advance the whole) ;
- the composable BRICKS (state / transport / source / elliptic) and the spatial + time policies ;
- the geometry MESH objects (CartesianMesh / PolarMesh / AuxHalo) ;
- the parallelism knobs (set_threads / has_kokkos / parallel_info) ;
- the environment doctor / capability matrix ;
- the host PythonFlux prototyping backend.

The lower layers (ir / model / physics / time / lib / codegen) are numpy-free until first use
and never import this package. The runtime methods that need the codegen / physics / dsl layers
import them LAZILY in-method, both to avoid a cycle and to keep ``import pops`` numpy-free.

Example::

    import pops
    sim = pops.System(n=192, periodic=False)
    sim.add_block(
        "ne",
        model=pops.Model(state=pops.Scalar(), transport=pops.ExB(B0=1.0),
                        source=pops.NoSource(), elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0)),
        spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)
    sim.set_density("ne", ne_numpy)
    sim.step_cfl(0.4)
"""

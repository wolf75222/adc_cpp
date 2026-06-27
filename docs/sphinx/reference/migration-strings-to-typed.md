# Migration: strings to typed objects

Spec 5 replaces string algorithm selectors with typed objects. A string still names a user
object (a field, a parameter, a block); a typed object now chooses the algorithm, layout,
backend, or policy. For the rule see {doc}`../concepts/typed-api`; for the per-package surface
see {doc}`spec5-packages`.

A string selector is no longer silently accepted: it raises with the typed replacement spelled
out (`pops.descriptors.reject_string_selector`):

```python
from pops.descriptors import reject_string_selector

reject_string_selector("hll", "riemann", "pops.numerics.riemann.HLL()")
# TypeError: String algorithm selector rejected: riemann='hll'.
#            Use pops.numerics.riemann.HLL().
```

## Conversion table

Old string form on the left, the typed object on the right. Import the typed forms from the
package named in the last column.

| Old (string) | New (typed) | Package |
| --- | --- | --- |
| `riemann="rusanov"` | `Rusanov()` | `pops.numerics.riemann` |
| `riemann="hll"` | `HLL()` | `pops.numerics.riemann` |
| `riemann="hllc"` | `HLLC()` | `pops.numerics.riemann` |
| `riemann="roe"` | `Roe()` | `pops.numerics.riemann` |
| `reconstruction="first_order"` | `FirstOrder()` | `pops.numerics.reconstruction` |
| `reconstruction="muscl", limiter="minmod"` | `MUSCL(limiter=Minmod())` | `pops.numerics.reconstruction` |
| `reconstruction="muscl", limiter="vanleer"` | `MUSCL(limiter=VanLeer())` | `pops.numerics.reconstruction` |
| `reconstruction="weno5z"` | `WENO5Z()` | `pops.numerics.reconstruction` |
| `set_refinement(variable="density", above=0.5)` | `Refine.on("density").above(0.5)` | `pops.mesh.amr` |
| `layout="amr", levels=2, ratio=2` | `AMR(mesh, max_levels=2, ratio=2)` | `pops.mesh.layouts` |
| `layout="uniform"` | `Uniform(mesh)` | `pops.mesh.layouts` |
| `regrid_every=8` | `RegridEvery(8)` | `pops.mesh.amr` |
| `output(format="hdf5")` | `OutputPolicy(format=HDF5())` | `pops.output` |
| `output(format="plotfile", levels="coarse")` | `OutputPolicy(format=Plotfile(), levels=CoarseOnly())` | `pops.output` |
| `param("nu", 0.1, domain="positive")` | `RuntimeParam("nu", default=0.1, domain=Positive())` | `pops.params` |
| `param("cfl", 0.4, domain="0..1")` | `RuntimeParam("cfl", default=0.4, domain=Range(0.0, 1.0))` | `pops.params` |
| `compile(..., math="strict")` | `Optimization(math=StrictMath())` | `pops.codegen` |
| `compile(..., math="fast")` | `Optimization(math=FastMath())` | `pops.codegen` |
| `riemann="my_hllc"` (external) | `CompiledBrickRef(manifest, "my_hllc", expect_category="riemann")` | `pops.external` |

## What stays a string

Strings still name objects the user invents -- they are labels, not algorithm choices:

```python
from pops.params import RuntimeParam, Positive
from pops.fields import PoissonProblem
from pops.mesh.amr import Refine

RuntimeParam("nu", default=0.1, domain=Positive())   # "nu" is the user's parameter name
PoissonProblem(name="phi", unknown="phi")            # "phi" is the user's field name
Refine.on("density").above(0.5)                       # "density" names the tagged field
```

The rule of thumb: if the string is something you named, keep it; if the string selected a
library algorithm, backend, or layout, replace it with the typed object above.

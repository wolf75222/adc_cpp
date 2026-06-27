# Spatial schemes


The spatial scheme = reconstruction (limiter) + numerical Riemann flux + reconstructed
variables. Two equivalent facades describe it.

Every scheme is chosen with a TYPED `pops.numerics` descriptor; a bare string raises a
`TypeError` that names the typed alternative (Spec 5 sec.7). `pops.Spatial(limiter=, flux=,
recon=)` is the direct facade, with boolean shortcuts (`minmod=True`, `vanleer=True`,
`weno5=True`, `none=True`, `primitive=True`):

```python
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction import WENO5

pops.Spatial(minmod=True)                       # MUSCL minmod, Rusanov, conservative variables
pops.Spatial(vanleer=True, flux=HLLC())         # MUSCL Van Leer, HLLC
pops.Spatial(weno5=True, primitive=True)        # WENO5-Z, primitive reconstruction
```

`pops.FiniteVolume(limiter=, riemann=, variables=)` is the same thing, but the numerical
Riemann flux is called `riemann` (and not `flux`, reserved for the physical flux of a DSL model).
The reconstruction slot also accepts the Spec 5 sec.14.1 alias `reconstruction=`:

```python
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.variables import Conservative

pops.FiniteVolume(limiter=Minmod(), riemann=Rusanov(), variables=Conservative())
```

The typed descriptors:

- limiter / reconstruction (`pops.numerics.reconstruction`): `FirstOrder()` (first-order Godunov),
  `.limiters.Minmod()`, `.limiters.VanLeer()` (second-order MUSCL, 2 ghosts),
  `WENO5()` / `WENO5Z()` (WENO5-Z, order 5 in a smooth zone, 5-point stencil / 3 ghosts,
  oscillation-free capture near a front). WENO5 is exposed only by the native `add_block` path and
  the compiled `aot`/`production` backends (the `prototype` JIT path rejects it);
- Riemann flux (`pops.numerics.riemann`): `Rusanov()` (the most stable, default for scalar
  transport), `HLL()` (generic signed-wave, requires `model.wave_speeds`), `HLLC()`, `Roe()`. HLLC
  and Roe run on the canonical Euler 2D layout (4 variables + pressure), or generically on any model
  that supplies the capability hooks (`HasHLLCStructure` / `HasRoeDissipation`, emitted in the DSL
  with `m.enable_hllc()` / `m.enable_roe()`);
- variables (`pops.numerics.variables`): `Conservative()` or `Primitive()`. The primitive set is
  more stable for Euler (positivity of `rho` and `p`).

On the C++ side, the limiters are policies in `numerics/reconstruction.hpp` (`NoSlope`,
`Minmod`, `VanLeer`, `Weno5`), the fluxes in `numerics/numerical_flux.hpp` (`RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux`). Detail and formulas: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
sections 2 and 3.

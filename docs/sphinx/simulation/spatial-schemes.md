# Spatial schemes


The spatial scheme = reconstruction (limiter) + numerical Riemann flux + reconstructed
variables. Two equivalent facades describe it.

`adc.Spatial(limiter=, flux=, recon=)` is the direct facade, with boolean shortcuts
(`minmod=True`, `vanleer=True`, `weno5=True`, `none=True`, `primitive=True`):

```python
adc.Spatial(minmod=True)                       # MUSCL minmod, Rusanov, conservative variables
adc.Spatial(vanleer=True, flux="hllc")         # MUSCL Van Leer, HLLC
adc.Spatial(weno5=True, primitive=True)        # WENO5-Z, primitive reconstruction
```

`adc.FiniteVolume(limiter=, riemann=, variables=)` is the same thing, but the numerical
Riemann flux is called `riemann` (and not `flux`, reserved for the physical flux of a DSL model):

```python
adc.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")
```

The possible values:

- limiter: `"none"` (first-order Godunov), `"minmod"`, `"vanleer"` (second-order MUSCL, 2 ghosts),
  `"weno5"` (WENO5-Z, order 5 in a smooth zone, 5-point stencil / 3 ghosts, oscillation-free
  capture near a front). `weno5` is exposed only by the native `add_block` path and the
  compiled `aot`/`production` backends (the `prototype` JIT path rejects it);
- Riemann flux: `"rusanov"` (the most stable, default for scalar transport), `"hllc"`,
  `"roe"`. HLLC and Roe require a compressible transport (4 variables + pressure);
- reconstruction: `"conservative"` or `"primitive"`. The primitive is more stable for Euler
  (positivity of `rho` and `p`).

On the C++ side, the limiters are policies in `numerics/reconstruction.hpp` (`NoSlope`,
`Minmod`, `VanLeer`, `Weno5`), the fluxes in `numerics/numerical_flux.hpp` (`RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux`). Detail and formulas: [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
sections 2 and 3.

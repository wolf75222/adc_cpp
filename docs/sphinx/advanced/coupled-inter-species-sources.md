# Coupled inter-species sources


Beyond transport and a block's local source, one can describe an inter-species
coupling (ionization, collisions, thermal exchange) in formulas, without writing any
C++ and without a per-cell Python callback. The DSL `pops.dsl.CoupledSource` carries the
formula as stack-machine bytecode, interpreted on the C++ side in a device `for_each_cell`
(so MPI-safe and GPU-clean). The stage is applied by explicit splitting, after the
transport.

The canonical example is a three-species ionization
(`d_t n_e = +k n_e n_g`, `d_t n_i = +k n_e n_g`, `d_t n_g = -k n_e n_g`):

```python
import pops
from pops import dsl

src = dsl.CoupledSource("ionization")
ne = src.block("electrons").role("density")
ni = src.block("ions").role("density")
ng = src.block("neutrals").role("density")
kp = src.param("Kiz", 0.7)
src.add("electrons", role="density", expr=+kp * ne * ng)
src.add("ions",      role="density", expr=+kp * ne * ng)
src.add("neutrals",  role="density", expr=-kp * ne * ng)
compiled = src.compile(backend="production")

sim.add_coupling(compiled)   # branche l'etage sur System.add_coupled_source
```

`sim.add_coupling(...)` also accepts the named couplings `pops.Ionization` /
`pops.Collision` / `pops.ThermalExchange` (fixed formula). Without a call to `add_coupling`, the
`System` stays bit-identical (the stage is inert by default).

The compilation produces a flat ABI (`in_blocks`, `in_roles`, `consts`, `out_blocks`,
`out_roles`, `prog_ops`, `prog_args`, `prog_lens`): bytecode, never a Python
callback. The test checks that the trajectory follows bit-for-bit a NumPy forward-Euler
reference of the same ODE, and that the expected invariants hold (`n_i + n_g`
conserved, `n_e - n_i` constant: each ionization creates an e/i pair).

## Going further

- Public / internal / deprecated classification of the coupling classes (including the concept
  `CoupledSourceFor` and the bytecode evaluator `CoupledSourceProgram`):
  [COUPLING_SURFACE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLING_SURFACE.md).
- Reference test: `python/tests/test_dsl_coupled_source.py` (and the conservation
  variant `test_dsl_coupled_source_conservation.py`).

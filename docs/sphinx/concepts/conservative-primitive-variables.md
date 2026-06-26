# Conservative and primitive variables

A hyperbolic model in `adc` carries its state in two complementary layouts: the conservative
variables `U`, which the scheme evolves and conserves, and the primitive variables `P`, which the
closures and the wave speeds read. This page explains why both exist and where each is used.

## Two views of the same state

The conservative variables are the quantities the conservation law evolves in time. For a
compressible fluid they are density `rho`, momenta `rho*u`, `rho*v`, and total energy `E`. They
are the natural variables of the divergence form $\partial_t U + \nabla\cdot F(U, \mathrm{aux}) = S(U, \mathrm{aux})$:
the finite-volume update moves `U` from one cell to its neighbor, and what one cell loses at a face
the next cell gains. That balance is exact only because the state is held in conserved form.

The primitive variables are the quantities the physics is written with: density, velocity `u`,
`v`, and pressure `p`. The flux formula, the pressure closure, and the eigenvalues (characteristic
speeds) are expressed cleanly in primitives. A velocity is `rho*u` divided by `rho`; you do not
want to redo that division everywhere the model needs a velocity.

So the two layouts are not redundant: each is the convenient frame for a different job. `U` is what
is transported and conserved; `P` is what closures and spectra are read from.

## The two conversions

A model that evolves more than one variable ties the two layouts together with a pair of pointwise
conversions, `cons_to_prim` (`to_primitive`) and `prim_to_cons` (`to_conservative`). The first reads
`U` and returns `P`; the second reads `P` and returns `U`. They are inverse maps of each other.

For a scalar transported density there is nothing to convert: the primitive equals the conservative,
and both conversions are the identity. The distinction only earns its keep once a variable like
momentum has to be divided by density to recover a velocity.

A model that satisfies the full hyperbolic contract (`pops::HyperbolicPhysicalModel`) carries the
variable names, the roles, and these two conversions together with its flux, because the flux is
written for one specific variable layout. Variables, conversions, and flux are physically linked and
travel as one brick, whether you write it natively or in the DSL. See the
[models section](../models/index.md).

## Why the scheme reconstructs in primitives

At order two and above, the spatial operator does not use cell averages directly; it reconstructs a
left and a right state at each face before calling the Riemann solver. That reconstruction can run on
the conserved variables or on the primitives, and the choice matters.

Reconstructing the conserved state can push the reconstructed values out of the admissible domain at
a strong shock: a limited slope on `rho*u` and `E` does not guarantee a positive density or pressure
on either side of the face. Reconstructing in primitives limits `rho`, `u`, and `p` directly, which
keeps positivity of density and pressure and is more stable for Euler with shocks.

This is why a fluid block typically selects primitive reconstruction. The choice is exposed through
the `variables="primitive"` option of `pops.FiniteVolume`; the operator then calls `to_primitive`
before reconstructing and `to_conservative` after, and the conserved update at the end stays exact.

## Roles, not indices

Because couplings between species need to find a quantity by meaning and not by a literal slot, each
variable carries a physical role (`Density`, `MomentumX`, `Energy`, and so on). Canonical names map
to roles automatically (`rho` or `n` to `Density`, `E` to `Energy`); an unrecognized name stays
`Custom`. A coupling that needs the density of another block asks for the role, so it keeps working
when two models order their variables differently.

```python
# fluid block: conservative U, primitives P, and the inverse map
(rho, mx, my, E) = m.conservative_vars("rho", "mx", "my", "E")
u = m.primitive("u", mx / rho)            # cons -> prim, pointwise
v = m.primitive("v", my / rho)
m.primitive_vars(rho=rho, u=u, v=v, p=p)  # fixes the ordered P layout
m.conservative_from([rho, rho * u, rho * v, E])  # prim -> cons (you give the inverse)
```

The DSL cannot invert your primitives symbolically, so you state the inverse explicitly with
`conservative_from`; it generates `to_conservative`. For the discretization that consumes these
variables (reconstruction, Riemann flux), see
[ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

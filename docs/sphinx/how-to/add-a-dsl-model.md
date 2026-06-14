# Add a DSL model

Express a model's physics as symbolic formulas with `adc.dsl.Model`, compile it into a `.so`,
and attach it to a `System`. Use this when you want to declare the flux, eigenvalues, source and
elliptic right-hand side directly, instead of composing native bricks. This page assumes you can
already build the Python module and run a case. For the syntax of every declarator, see the
[DSL reference](../reference/dsl_reference.md); for the concepts, see
[fluxes, sources and eigenvalues](../concepts/fluxes-sources-eigenvalues.md).

## Before you start

Set the environment variables the DSL needs to find the adc headers and cache the generated
`.so`. Replace `REPO` with the path to your `adc_cpp` checkout.

```bash
export ADC_INCLUDE=REPO/include
```

```bash
export ADC_CACHE_DIR=REPO/.adc_cache
```

`ADC_INCLUDE` points at the headers the `.so` compiles against; `ADC_CACHE_DIR` caches the
generated `.so` between runs (default `~/.cache/adc/dsl`).

## Steps

1. Declare the conservative state and any parameters. Pass `roles=` so the `System` can resolve
   couplings by role.

   ```python
   from adc import dsl

   m = dsl.Model("euler_poisson")
   rho, rhou, rhov, E = m.conservative_vars(
       "rho", "rho_u", "rho_v", "E",
       roles=["Density", "MomentumX", "MomentumY", "Energy"])
   g = m.param("gamma", 1.4)
   ```

2. Define the primitives in dependency order with `primitive`.

   ```python
   u = m.primitive("u", rhou / rho)
   v = m.primitive("v", rhov / rho)
   p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
   ```

3. Declare the flux and the eigenvalues, one `Expr` per component and direction. Use
   `dsl.sqrt` for the sound speed; it is the only named math function in the algebra.

   ```python
   H = (E + p) / rho
   c = dsl.sqrt(g * p / rho)
   m.flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u],
          y=[rhov, rhov*u, rhov*v + p, rho*H*v])
   m.eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])
   ```

4. Add the source and the elliptic right-hand side if the model needs them. Read the potential
   gradient through the fixed `aux` channels `grad_x` and `grad_y`.

   ```python
   gx, gy = m.aux("grad_x"), m.aux("grad_y")
   m.source([0.0, -rho*gx, -rho*gy, -(rhou*gx + rhov*gy)])
   m.elliptic_rhs(-1.0 * (rho - 1.0))
   ```

5. Set the primitive layout, supply the inverse `Prim -> U` by hand, then validate. The DSL does
   not invert the primitives for you.

   ```python
   prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
   m.conservative_from([prho, prho*pu, prho*pv,
                        pp/(g-1.0) + 0.5*prho*(pu*pu + pv*pv)])
   m.check()
   ```

6. Compile the model. `production` is the native zero-copy path; it requires that `_adc` and the
   `.so` share the same headers, compiler and C++ standard.

   ```python
   compiled = m.compile(backend="production")
   ```

7. Attach the compiled model to a `System` and choose the scheme. HLLC and Roe require a
   primitive named `p`.

   ```python
   import adc

   s = adc.System(n=32, L=1.0, periodic=True)
   s.add_equation("gas", compiled,
                  spatial=adc.FiniteVolume(limiter="minmod", riemann="hllc",
                                           variables="primitive"))
   s.set_poisson(rhs="charge_density", solver="geometric_mg")
   ```

## Next steps

- Pick the backend for your run (`prototype`, `aot`, `production`) in the
  [DSL reference](../reference/dsl_reference.md).
- Mix native bricks with DSL bricks in one model with the
  [brick reference](../reference/bricks_reference.md).
- Run the model on a refined hierarchy in the [simulation guide](../simulation/index.md).

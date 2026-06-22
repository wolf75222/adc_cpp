# Extend with a C++ brick


The core is model-agnostic: it names no scenario. A physical model is a
composition of generic bricks (state, transport, source, elliptic), and the per-cell
computation stays compiled C++.

To write a new native brick, one satisfies the `PhysicalModel` concept
(`include/adc/core/model/physical_model.hpp`). The minimal contract:

```cpp
template <class M>
concept PhysicalModel =
    requires(const M m, const typename M::State u, const typename M::Aux a, int dir) {
      typename M::State;                                   // type d'etat conservatif
      typename M::Aux;                                     // == adc::Aux
      { M::n_vars } -> std::convertible_to<int>;           // nombre de composantes
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;       // flux directionnel
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;   // CFL
      { m.source(u, a) } -> std::same_as<typename M::State>;          // source LOCALE
      { m.elliptic_rhs(u) } -> std::convertible_to<Real>;             // second membre Poisson
    };
```

All these methods must be `ADC_HD` (host/device) if they are called in
kernels. The optional extension `HasPrimitiveVars` adds `to_primitive` / `to_conservative`
(reconstruction in primitive variables, more stable for Euler: positivity of rho and p),
and `HyperbolicPhysicalModel` adds the variable descriptor (`conservative_vars()` /
`primitive_vars()`). Once the brick is compliant, it composes into a `CompositeModel`
and is exposed at runtime like the existing bricks.

## Going further

- The five orthogonal layers, the map of the modules, the elliptic stage
  (problem / operator / solver / post-processing): [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).
- The design choices (concepts + policies, `for_each_cell` seam, `EllipticSolver`):
  [CHOICES.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md).
- The concept and its extensions: `include/adc/core/model/physical_model.hpp`;
  the reference composition: `include/adc/physics/composition/composite.hpp`.

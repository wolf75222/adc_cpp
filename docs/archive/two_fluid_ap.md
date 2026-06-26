# Isothermal two-fluid asymptotic-preserving scheme

Method note on the two-fluid solver: model, stiffness, AP scheme, transport
discretization, and measured robustness envelope. It is a PURPOSE-BUILT integrator, not composable
brick by brick like `System`; it is therefore not a generic brick but a SCENARIO. It has
left the `adc_cpp` core and now lives in `adc_cases/two_fluid_ap/`: the GPU-portable core
is in `adc_cases/two_fluid_ap/two_fluid_ap.hpp`, exposed via the C ABI of
`adc_cases/two_fluid_ap/_two_fluid_ap.cpp`. The latter is compiled on the fly against the
GENERIC headers of `adc_cpp` (mesh, elliptic, parallel) then loaded into Python
(ctypes) by `adc_cases/two_fluid_ap/run.py`. Physical reference: Hoffart, arXiv:2510.11808.

## 1. Model

Two species s (electrons e, ions i), density `n_s`, momentum `m_s = n_s u_s`,
isothermal sound speed `c_s`. Charge `z_e = -1`, `z_i = +1`. Electrostatic field
`E = -grad phi`, plasma frequencies `omega_ps`. For each species:

```
d_t n_s + div(m_s)                              = 0
d_t m_s + div(m_s o m_s / n_s + c_s^2 n_s I)    = z_s omega_ps^2 n_s E
```

coupled by the Poisson `lap(phi) = z_e n_e + z_i n_i = n_i - n_e` (n_0 = 1).

## 2. The stiffness

The plasma frequency `omega_pe` is the frequency of Langmuir oscillations. An explicit
scheme is stable only if `omega_pe * dt < O(1)`. In the quasi-neutral regime
`omega_pe` is large: resolving the oscillation forces a tiny step while the
dynamics of interest (transport) is slow. An explicit scheme "blows up" as soon as
`omega_pe * dt` exceeds order 1 (verified by the test: the non-stabilized mode diverges).

## 3. Asymptotic-preserving scheme

The Lorentz term is treated implicitly. By carrying the implicit momentum
`m^{n+1} = m* + dt z omega_ps^2 E` into the continuity then into the Poisson, the
field obeys a reformulated elliptic equation. At constant `beta0` (n_0 = 1):

```
beta0 = dt^2 (omega_pe^2 + omega_pi^2)
lap(phi) = (n_e* - n_i*) / (1 + beta0)
```

The factor `1 / (1 + beta0)` is what makes the scheme AP: when `omega_pe -> inf`
(`beta0 -> inf`), the RHS tends to 0, so `n_e -> n_i` (quasi-neutrality) and the step stays
stable independently of `omega_pe * dt`. This is the `stabilize` option (default).

One step (split operator):

```
1. m*   = m  - dt div(F_mom)         flux Rusanov (tfap_mstar)
2. n*   = n  - dt div(m*)            continuite (centree ou upwind)   -> RHS Poisson
3. phi  : lap(phi) = (n_e* - n_i*) / (1 + beta0)   multigrille (GeometricMG)
4. E    = -grad phi
5. m^{n+1} = m* + dt z omega_ps^2 E  Lorentz implicite
6. n^{n+1} = n  - dt div(m^{n+1})    continuite
```

The elliptic part is delegated to an `EllipticSolver` template (`PoissonFFTSolver` on CPU,
`GeometricMG` entirely on-device for the GPU GH200).

## 4. Transport discretization

The momentum (step 1) uses a dimensionally split **Rusanov** flux
(wave speed `a = |u| + c_s`), order 1. The continuity (steps 2 and 6) offers two
schemes, selected by `upwind_continuity`:

- **centered** (default): `div(m)` by central differences. It is a pure central mass flux
  `F = 0.5 (m_i + m_{i+1})`, without dissipation. Order 2, but dispersive on steep
  fronts.
- **upwind MUSCL**: Rusanov mass flux consistent with the momentum,
  `F = 0.5 (m_i + m_{i+1}) - 0.5 a (n_R - n_L)`, where `n_L, n_R` are reconstructed at the face
  with a **minmod** slope. The dissipation is `O(dx^2)` in the smooth regime (no
  over-diffusion) and drops back to order 1 at extrema (monotone). Requires 2 ghosts on the
  density. It is an order-2 upwind.

## 5. Measured robustness envelope

The AP invariants (bound, quasi-neutrality, mass conservation) are verified by the
case `adc_cases/two_fluid_ap/run.py`.

**Smooth perturbation** (cosine mode, stiff AP regime `dt * omega_pe = 5`): the scheme is
robust up to a large amplitude. At `eps = 0.8` the density stays positive
(`min n_e = 0.89`), bounded, quasi-neutral, mass conserved to roundoff. This is the design
regime of the scheme.

**Steep front** (narrow acoustic bump, weak coupling): the centered continuity
shows an undershoot of 17.6%. Important point verified by the upwind MUSCL: this
undershoot is almost **identical** (17.0%) with the accurate MUSCL scheme. The two
accurate schemes agree, so this undershoot is mostly the **physical acoustic
rarefaction**, not a numerical Gibbs oscillation. A 1st-order Rusanov mass flux
reduced it to 13.4%, but that was over-diffusion (loss of accuracy), not
a correction. On a smooth well-resolved bump, MUSCL loses only 0.4% of the peak relative
to the centered scheme: it is an accurate order-2 upwind, that is its real contribution.

Practical conclusion: the centered continuity suffices for the targeted smooth AP regime; the
upwind MUSCL option is an accurate order-2 transport for cases with strong gradients, without
over-diffusion.

## 6. Magnetic field (Hoffart extension)

A uniform out-of-plane magnetic field `B = B_z z` adds the magnetic Lorentz force
`z (m x B)` to the momentum. With `B` out-of-plane it only ROTATES
`(m_x, m_y)` at the cyclotron frequency `wc_s = |q_s B_z / m_s|`, without changing `|m|` nor `n`.
The exact rotation of angle `theta_s = z_s wc_s dt` is unconditionally stable (no
`wc * dt` limit). It is composed with the electrostatic step by Strang splitting:
`R(theta/2) o pas-ES o R(theta/2)`, order 2.

Opt-in via `wce`, `wci` (cyclotron frequencies per species; 0 = no field, electrostatic
behavior unchanged). Validated analytically by
[`tests/test_two_fluid_cyclotron.cpp`](../tests/test_two_fluid_cyclotron.cpp): on a uniform
plasma (zero charge, `E = 0`, inert transport) the momentum rotates
at `wce` to 0.00% deviation, `|m|` conserved to roundoff (`~1e-15`). Exposed up to Python
(`TwoFluidAPConfig.omega_ce/omega_ci`).

It is the first brick toward the full magnetized Hoffart model: it remains to couple the
field to the inhomogeneous dynamics (the `E x B` and the diamagnetic entering the transport),
beyond the pure rotation validated here.

## 7. Validation

- Isotropic dispersion on diagonal mode: theory/measurement deviation 3.1%.
- AP bound and quasi-neutrality at `omega_pe = 1e3`, `dt * omega_pe = 5`, where the explicit
  blows up.
- Mass conservation per species to roundoff (`~1e-11`), centered and upwind.
- GPU-portable GH200 (same kernels `for_each_cell` + `POPS_HD`, on-device multigrid),
  bit-identical to CPU.
- Drivable from Python by the scenario `adc_cases/two_fluid_ap/`: the `TwoFluidAP` class
  of `run.py` (config `n`, `omega_pe`, `upwind_continuity`, ...) loads the C ABI compiled on the
  fly; outside the public API of the core (purpose-built scenario, not composable).

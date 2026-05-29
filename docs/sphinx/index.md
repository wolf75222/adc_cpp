# adc_cpp

Solveur C++23 maison (mini-AMReX écrit *from scratch*) pour les systèmes
hyperbolique-elliptique couplés sur AMR : seam de dispatch unique série /
OpenMP / Kokkos (GPU GH200) / MPI, pile MultiFab + BoxArray + Geometry,
AMR block-structured multi-niveaux et multi-patch (Berger-Rigoutsos, reflux
coverage-aware), Poisson multigrille ET FFT spectrale, couplage diocotron
(dérive E x B), Euler-Poisson auto-gravitant et deux-fluides isotherme
asymptotic-preserving. Bindings Python via pybind11.

```{toctree}
:maxdepth: 2
:caption: Guide

installation
quickstart
examples
api
```

```{toctree}
:maxdepth: 1
:caption: Tutoriels

tutorials_index
```

```{toctree}
:maxdepth: 1
:caption: Référence C++

C++ API (Doxygen) <https://wolf75222.github.io/adc_cpp/cpp/>
```

## En bref

Trois axes orthogonaux (concept `PhysicalModel`, policy `NumericalFlux`, concept
`EllipticSolver`) et un seam de parallélisme unique :

- `DiocotronSolver` (dérive E x B), `EulerPoissonSolver` (auto-gravitant),
  `TwoFluidAPSolver` (deux-fluides isotherme AP)
- flux `RusanovFlux` / `HLLFlux` / `HLLCFlux`, reconstruction MUSCL (Minmod / VanLeer)
- `GeometricMG` (multigrille V-cycle GS red-black) / `PoissonFFTSolver` (spectral direct)
- AMR : `amr_step_multilevel_mf` (N niveaux mono-box), `amr_step_multilevel_multipatch`
  (multi-box, reflux coverage-aware), `AmrCoupler` / `AmrCouplerMP` (regrid Berger-Rigoutsos)
- `for_each_cell` : série / OpenMP / Kokkos ; `comm.hpp` : collectives MPI

Détail des algorithmes :
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).
Architecture et choix de design :
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md),
[CHOICES.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md).

## Liens

- Code source : <https://github.com/wolf75222/adc_cpp>
- Référence C++ Doxygen : [/cpp/](https://wolf75222.github.io/adc_cpp/cpp/)
- Solveurs sœurs (sur `pde_core_cpp`) : [euler_cpp](https://github.com/wolf75222/euler_cpp),
  [advection_cpp](https://github.com/wolf75222/advection_cpp)

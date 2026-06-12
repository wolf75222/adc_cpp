# adc_cpp

Solveur C++23 pour les systemes hyperbolique-elliptique couples sur AMR (pile mesh ecrite
*from scratch*) : seam de dispatch unique serie / OpenMP / Kokkos (GPU GH200) / MPI, pile
MultiFab + BoxArray + Geometry, AMR block-structured multi-niveaux et multi-patch
(Berger-Rigoutsos, reflux coverage-aware), Poisson multigrille et FFT spectrale, couplage
diocotron (derive E x B), Euler-Poisson auto-gravitant et deux-fluides isotherme
asymptotic-preserving. Bindings Python via pybind11.

Cette documentation est le guide utilisateur. Les documents de conception detailles
([ARCHITECTURE](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md),
[ALGORITHMS](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md),
[BACKEND_COVERAGE](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md)...)
restent la reference contributeur et sont lies la ou c'est utile.

```{toctree}
:maxdepth: 2
:caption: Prise en main

getting_started/index
```

```{toctree}
:maxdepth: 2
:caption: Ecrire un modele

models/index
```

```{toctree}
:maxdepth: 2
:caption: Simuler

simulation/index
amr/index
```

```{toctree}
:maxdepth: 2
:caption: Executer

backends/index
advanced/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 1
:caption: Reference C++

API C++ embarquee <doxygen/index>
C++ API (Doxygen) <https://wolf75222.github.io/adc_cpp/cpp/>
```

## En bref

Trois axes orthogonaux (concept `PhysicalModel`, policy `NumericalFlux`, concept
`EllipticSolver`) et un seam de parallelisme unique :

- composition generique `System` : un bloc par modele, ou un modele est une composition de
  briques generiques (`adc.Model(state, transport, source, elliptic)`) ; le coeur ne nomme aucun
  scenario (les noms diocotron, euler_poisson... vivent cote `adc_cases`). Poisson de systeme
  partage ; cote Python via `adc.System`. Trois facons d'ecrire un modele : composition de
  briques natives, modele symbolique `adc.dsl.Model`, ou composition hybride (voir
  [Modeles](models/index.md)).
- flux `RusanovFlux` / `HLLCFlux` / `RoeFlux`, reconstruction MUSCL (Minmod / VanLeer) + WENO5-Z ;
- `GeometricMG` (multigrille V-cycle GS red-black) / `PoissonFFTSolver` (spectral direct) ;
- AMR : `AmrSystem` mono- et multi-bloc, multi-patch N niveaux, reflux coverage-aware,
  `AmrCouplerMP` (regrid Berger-Rigoutsos), voir [AMR](amr/index.md) ;
- `for_each_cell` : serie / OpenMP / Kokkos ; `comm.hpp` : collectives MPI, voir
  [Backends paralleles](backends/index.md).

Nouveau venu ? Commencez par la [presentation](getting_started/presentation.md),
[installez](getting_started/installation.md), puis suivez le
[tutoriel A->Z](getting_started/tutorial.md).

## Liens

- Code source : <https://github.com/wolf75222/adc_cpp>
- Reference C++ Doxygen : [/cpp/](https://wolf75222.github.io/adc_cpp/cpp/)
- Solveurs soeurs sur le socle [pde_core_cpp](https://github.com/wolf75222/pde_core_cpp) :
  euler_cpp et advection_cpp (depots prives)
- Scenarios applicatifs : [adc_cases](https://github.com/wolf75222/adc_cases)

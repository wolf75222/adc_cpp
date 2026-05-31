# Tutoriels

Parcours guidés d'`adc_cpp` en C++ et en Python en parallèle. Chaque tutoriel prend un
problème, le pose des deux côtés, le fait tourner et explique la sortie. Les figures et
animations sont pré-générées sous `docs/` ; voir `00_install.md` pour les reproduire.

## Index

| # | Titre | Sujet |
|---|---|---|
| 00 | [Installer et compiler](00_install.md) | CMake, options de backend, module Python, disposition des artefacts |
| 01 | [L'instabilité diocotron](01_diocotron.md) | dérive E x B, couplage hyperbolique-elliptique, taux de croissance vs théorie |
| 02 | [Poisson : multigrille vs FFT](02_poisson.md) | concept `EllipticSolver`, V-cycle GS red-black, solveur spectral direct, quand l'un bat l'autre |
| 03 | [L'API Python](03_python_api.md) | interop numpy, configs/solveurs, rejouer les expériences depuis Python |
| 04 | [AMR multi-niveaux](04_amr_multilevel.md) | sous-cyclage Berger-Oliger, reflux FluxRegister, récursion N niveaux, conservation bit-identique |
| 05 | [AMR multi-patch + Berger-Rigoutsos](05_amr_multipatch.md) | plusieurs patchs par niveau, reflux coverage-aware, clustering, regrid dynamique, coupleur `AmrCouplerMP` |
| 06 | [Deux-fluides isotherme AP](06_two_fluid_ap.md) | raideur plasma, schéma asymptotic-preserving, dispersion, continuité upwind MUSCL |
| 07 | [Champ magnétique : rotation cyclotron](07_magnetic.md) | force de Lorentz magnétique, rotation exacte, splitting de Strang, fréquence cyclotron |
| 08 | [Backends : OpenMP, MPI, GPU](08_backends.md) | le seam `for_each_cell`, déterminisme thread-count, bit-identité MPI, portage Kokkos GH200 |
| 09 | [Euler-Poisson : gravité ou plasma](09_euler_poisson.md) | couplage hyperbolique-elliptique, dualité de signe, Jeans vs Bohm-Gross, effondrement vs explosion de Coulomb |
| 10 | [Reproduire le papier diocotron](10_diocotron_reproduction.md) | objectif de stage (arXiv:2510.11808), M1 à M2b, Poisson multi-niveau, convergence AMR vs uniforme (~41-44 % des cellules) |

## Tutoriels exécutables

Les pages ci-dessus expliquent ; [run/](run/README.md) fait tourner. Des scripts Python
(bindings `adc`) et C++ qui produisent vraiment un plot ou un GIF, chacun vérifié par un
`assert` sur l'invariant physique, plus les jobs MPI / GPU sur ROMEO. Sortie sous `docs/`.

## Conventions

Les blocs C++ renvoient à des fichiers sous `examples/` ; les blocs Python importent le
module compilé `adc` (alias par défaut `adc`). Les figures citées vivent sous `docs/`
(pré-générées). Le banc de mesure est `examples/bench_amr.cpp` (voir `08_backends.md`).

L'écosystème : `adc_cpp` porte sa propre pile AMR *from scratch* (MultiFab + seam),
contrairement à `euler_cpp` / `advection_cpp` qui dépendent de `pde_core_cpp`. Le
rationale est dans le README et `docs/ARCHITECTURE.md`.

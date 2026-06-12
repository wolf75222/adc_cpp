# API C++

La reference C++ complete d'`adc_cpp` (toutes les classes, concepts et fonctions, avec les
signatures et les commentaires d'en-tete) est generee par **Doxygen** depuis les en-tetes
`include/adc/**`. Elle est publiee en ligne :

- **Site Doxygen :** <https://wolf75222.github.io/adc_cpp/cpp/>

La meme reference est aussi embarquee dans ce site (entree "API C++ embarquee" de la barre
laterale, ou <https://wolf75222.github.io/adc_cpp/doxygen/>) : pages generees par doxysphinx,
integrees a la navigation et a la recherche Sphinx. Le site Doxygen brut reste publie sous
`/cpp/` a l'identique.

La configuration Doxygen vit dans le depot : [`docs/Doxyfile`](../../Doxyfile). La regenerer
localement produit le HTML sous le repertoire de sortie configure :

```bash
doxygen docs/Doxyfile
```

La bibliotheque est header-only (templates et concepts C++23) : il n'y a pas d'objet `.a`/`.so`
a lier pour le coeur, on inclut les en-tetes. Les trois axes de conception sont orthogonaux
(modele physique x flux numerique x solveur elliptique), composes par un seam de parallelisme
unique (`for_each_cell` : serie / OpenMP / Kokkos ; `comm.hpp` : collectives MPI). Voir
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) et [CHOICES.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/CHOICES.md).

## Concepts et classes principaux

| Symbole | Role |
|---------|------|
| `PhysicalModel` | Concept de modele physique : etat, flux physique, valeurs propres, conversions conservatif <-> primitif, second membre elliptique. C'est l'axe "equation". |
| `NumericalFlux` | Policy de flux numerique de Riemann a la face (Rusanov / HLL / HLLC / Roe). C'est l'axe "flux", choisi independamment du modele. |
| `EllipticSolver` | Concept de solveur elliptique (interface `solve()`) modele par `GeometricMG`, `PoissonFFTSolver`, les solveurs Krylov tensoriels... C'est l'axe "Poisson". |
| `System` | Coupleur runtime : compose des blocs (un modele par bloc), partage un Poisson de systeme, avance le tout. Expose au binding Python `adc.System`. |
| `AmrSystem` | Pendant raffine de `System` : un ou plusieurs blocs sur une hierarchie AMR block-structured (regrid, reflux conservatif). Expose `adc.AmrSystem`. |
| `GeometricMG` | Solveur Poisson multigrille (V-cycle, lisseur Gauss-Seidel rouge-noir) ; gere paroi conductrice et cut-cell. Tout cas (dont non-periodique). |
| `PoissonFFTSolver` | Solveur Poisson spectral direct (FFT) ; domaine periodique, `n = 2^k`. Mono-rang par design (refuse MPI). |
| `AmrCouplerMP` | Coupleur AMR multi-patch : regrid Berger-Rigoutsos, sous-cyclage en temps des niveaux fins, reflux coverage-aware a l'interface grossier-fin. |

(Les definitions canoniques de ces symboles vivent respectivement dans
`include/adc/core/physical_model.hpp`, `include/adc/numerics/numerical_flux.hpp`,
`include/adc/numerics/elliptic/elliptic_solver.hpp`, `include/adc/runtime/system.hpp`,
`include/adc/runtime/amr_system.hpp`, `include/adc/numerics/elliptic/geometric_mg.hpp`,
`include/adc/numerics/elliptic/poisson_fft_solver.hpp`,
`include/adc/coupling/amr_coupler_mp.hpp`.)

Pour la couverture de tests de ces composants par backend, voir
[matrice des backends](backend_matrix.md).

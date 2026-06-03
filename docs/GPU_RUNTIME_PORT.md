# Portage de la pile runtime sur GPU (feuille de route)

Le DSL et le coeur de calcul sont verifies jusqu'au GPU GH200 (flux, brique generee, CAS Euler complet
via le seam Kokkos `for_each_cell` ; cf. docs/GPU_ROMEO.md). Ce qui reste pour une PRODUCTION GPU de
bout en bout, c'est porter la PILE RUNTIME entiere (System / MultiFab / Poisson / AMR / MPI) sur device.
C'est un chantier d'integration MAJEUR (semaines), pas un tour. Ce document le decoupe en phases, avec
ce qui est deja acquis et ce que chaque phase exige.

## Atout de conception : le seam ne change pas les sites d'appel

`adc/mesh/for_each.hpp` (`for_each_cell`, `for_each_cell_reduce_*`) bascule CPU <-> GPU a la
COMPILATION sans toucher les operateurs. `ADC_HD` rend tout le coeur device-callable. Donc le portage
GPU est surtout un travail de RESIDENCE des donnees sur device + de portage des etapes encore hote,
pas une reecriture des noyaux de calcul.

## Deja device-ready (verifie sur GH200)

- `for_each_cell` / reductions -> `Kokkos::parallel_for` / `parallel_reduce` (espace `Cuda`).
- `ADC_HD`, `StateVec`, `Aux` : device-callable ; flux + `eigenvalues` + briques (Euler/iso/ExB).
- Brique generee par le DSL : compile nvcc `sm_90`, == `adc::Euler` au bit (CUDA) / a 1 ULP (Kokkos).
- Cas Euler 2D complet (80 pas, CFL, Rusanov, periodique) sur GH200 : masse conservee, == CPU.

## Phases du portage runtime (par dependance croissante)

1. **MultiFab device-resident.** ✅ FAIT (verifie GH200). Constat : `fab_allocator` sous
   `ADC_HAS_KOKKOS + __CUDACC__` est `ManagedAllocator` (cudaMallocManaged) -> les Fab sont en MEMOIRE
   UNIFIEE, donc deja device-accessibles, et `assemble_rhs` (via `for_each_cell` -> Kokkos) tourne sur
   le device par CONSTRUCTION. Un transport Euler COMPLET (80 pas, fill_boundary + assemble_rhs +
   maj SSPRK/FE sur la VRAIE pile adc) donne sur GH200 un resultat BIT-IDENTIQUE au CPU
   (`python/tests/gpu/phase1_transport.cpp`, masse 4096 / energie identiques). A leve + corrige un bug
   nvcc dans `numerics/spatial_operator.hpp` (capture `dx`/`dy` en contexte `constexpr-if`, interdite
   pour une lambda `__host__ __device__` etendue). Optimisation restante : eviter le va-et-vient hote
   de `fill_boundary` (phase 2) ; la memoire unifiee assure la correction mais pas la perf optimale.
2. **Conditions aux limites sur device.** ✅ FAIT (verifie GH200). `copy_shifted` (ghosts
   PERIODIQUES, `mesh/fill_boundary.hpp`) etait deja `for_each` -> device. `fill_physical_bc`
   (Foextrap / Dirichlet, `mesh/physical_bc.hpp`) porte des boucles HOTE vers `for_each_cell` ;
   `device_fence` interne supprime (les kernels BC s'ordonnent apres `copy_shifted`, et les faces y
   apres les faces x, sur le meme flux). Un transport NON-periodique (sortie Foextrap) sur GH200 donne
   un resultat BIT-IDENTIQUE au CPU (`python/tests/gpu/phase2_transport.cpp` ; la masse decroit comme
   il se doit, sortie libre). 49 ctests (dont test_physical_bc, poisson_disc, cut_cell) verts.
3. **Poisson sur device.** ✅ FAIT (verifie GH200) -- et SANS modification de code. Toute la boucle
   V-cycle de `GeometricMG` etait DEJA en `for_each` -> device : smoother red-black GS, residu, Laplacien
   (`poisson_operator.hpp`), restriction `average_down` + prolongation `interpolate` (`mesh/refinement.hpp`),
   norme via `for_each_cell_reduce_max` (`mf_arith.hpp`). Seul le SETUP (masque + coefs cut-cell depuis
   des `std::function`) reste hote (one-shot, ecrit la memoire unifiee). Un solve Poisson Dirichlet
   (n=128) sur GH200 donne un resultat BIT-IDENTIQUE au CPU : cycles=9, sum/max(phi) identiques
   (`python/tests/gpu/phase3_poisson.cpp`). Le "gros morceau" etait deja porte par le seam for_each.
   (Le solveur a COEFFICIENTS VARIABLES eps(x) reste un ajout numerique a part, non requis ici.)
4. **Couplages inter-especes sur device.** ✅ FAIT (verifie GH200). Les 3 couplages
   (ionisation, collision, echange thermique, `system.cpp`) portes de boucles HOTE vers `for_each_cell`
   (kernels device lisant/ecrivant PLUSIEURS blocs au meme point) ; `device_fence` prealable de
   `apply_couplings` supprime (kernels ordonnes apres le transport). Le kernel d'ionisation sur GH200
   donne un resultat BIT-IDENTIQUE au CPU, n_i + n_g conserve (`python/tests/gpu/phase4_coupling.cpp`).
   Host : `test_bindings` (conservation des 3 couplages) vert. => jalon 1->2->4 atteint : TRANSPORT
   MULTI-ESPECES complet (transport + BCs + couplages) sur GPU, sans Poisson.
5. **AMR sur device.** regrid, prolongation/restriction de niveau, reflux : kernels device + gestion
   de la hierarchie cote hote (orchestration) avec donnees device. Effort : eleve.
6. **MPI CUDA-aware.** Echange de halos device-to-device (GPUDirect), `fill_boundary` distribue sans
   detour par l'hote ; FFT distribuee device. Effort : eleve.
7. **Validation bout-en-bout.** Un cas plasma multi-especes complet (transport + Poisson + couplages
   [+ AMR]) sur GH200, compare bit-a-bit (ou a tolerance FP documentee) au meme cas CPU/serie.

## Strategie suggeree

- Avancer phase par phase, chacune validee CPU == GPU (le seam autorise a basculer une etape a la
  fois). Demarrer par 1 -> 2 -> 4 (transport pur multi-blocs sur GPU, sans Poisson) : deja un jalon
  utile et testable. Poisson device (3) ensuite, puis AMR (5) et MPI (6).
- Outillage ROMEO : `module load cuda/12.6`, build Kokkos+CUDA (`Kokkos_ARCH_HOPPER90`), `srun
  --account=<compte> -p instant --constraint=armgpu --gres=gpu:1` (cf. docs/GPU_ROMEO.md). nvcc ne
  s'execute QUE sur le noeud GPU (aarch64), pas sur le login (x86).
- Orchestration : un workflow multi-agent par backend (Kokkos-Serial / OpenMP / CUDA) + verificateurs
  adverses comparant a l'oracle CPU est le bon outil pour la phase de validation (cf. sessions DSL).

## Etat

Le reste de la vision (DSL symbolique : interprete, codegen flux/brique/source/elliptique, CSE, JIT
.so, dispatch type-erased dans le System ; flux Roe ; VariableRole ; eps constant ; reorg physics/
numerics) est COMPLET au niveau prototype, teste (49 ctests C++ + 13 tests Python) et verifie jusqu'au
GH200. Ce portage runtime est le seul gros morceau ouvert.

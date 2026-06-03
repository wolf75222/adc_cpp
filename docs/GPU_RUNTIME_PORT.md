# Portage de la pile runtime sur GPU (feuille de route)

Le DSL et le coeur de calcul sont verifies jusqu'au GPU GH200 (flux, brique generee, CAS Euler complet
via le seam Kokkos `for_each_cell` ; cf. docs/GPU_ROMEO.md). Ce qui reste pour une PRODUCTION GPU de
bout en bout, c'est porter la PILE RUNTIME entiere (System / MultiFab / Poisson / AMR / MPI) sur device.
ETAT (juin 2026) : les composants sont valides SEPAREMENT sur GH200, pas en pile integree. Le
solveur MONO-GRILLE complet (transport + BCs + couplages + Poisson + pas de temps, orchestre par
le System) tourne sur GH200, verifie == CPU (phases 1-5, 7) ; les ops de champ AMR (reflux,
transferts) tournent sur device (phase 5) ; l'echange de halos MULTI-GPU est valide (phase 6,
np=1/2/4 bit-identiques) ; le backend AOT .so d'un modele genere par le DSL tourne sur device
(phase 8, avec marshaling hote). Ce qui RESTE : la VALIDATION INTEGREE AmrSystem + MPI + GPU
(jamais executee ensemble), la perf full-device, et la parite AOT zero-copie sur device. Ce
document decoupe en phases.

## Modele d'execution : MPI + Kokkos (PAS de CUDA ecrit a la main)

L'architecture est **MPI + Kokkos**, pas "trois couches Kokkos+CUDA+MPI". MPI distribue les
sous-domaines entre rangs (un GPU par rang) ; Kokkos parallelise le calcul LOCAL et abstrait le
materiel via son ExecutionSpace : backend **Cuda** pour GPU NVIDIA, **Serial/OpenMP** pour CPU. Le MEME
code source (`for_each_cell`, `assemble_rhs`, les briques) cible donc CPU et GPU selon le backend choisi
A LA COMPILATION ; on n'ecrit AUCUN kernel CUDA a la main. Dans les sorties ci-dessous, `exec=Cuda` est
simplement le backend Kokkos actif ; les memes .cpp passent en `exec=Serial`/`OpenMP` sur CPU (c'est ce
que verifie la CI MPI cote hote). `nvcc_wrapper` n'est que le compilateur exige par le backend Cuda de
Kokkos. Le coeur est desormais **100% Kokkos** : l'allocateur unifie utilise
`Kokkos::kokkos_malloc<Kokkos::SharedSpace>` + `Kokkos::fence` (et non plus `cudaMallocManaged` /
`cudaDeviceSynchronize`), et `ADC_HD` delegue a `KOKKOS_FUNCTION`. Plus AUCUNE API CUDA ecrite a la
main ; seul subsiste un repli `__host__ __device__` dans la branche HORS Kokkos de `ADC_HD` (inerte
chez nous). `SharedSpace` etant un alias portable (`CudaUVMSpace` / `HIPManagedSpace` /
`SYCLSharedUSMSpace` / `HostSpace`), le meme coeur ciblerait aussi AMD/Intel via les backends Kokkos.

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
   il se doit, sortie libre). ~53 ctests (dont test_physical_bc, poisson_disc, cut_cell) verts.
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
5. **AMR (ops de champ) sur device.** ✅ FAIT (verifie GH200) -- sans modif de code. Les tests
   self-checking `test_flux_register` (registre de flux / reflux 2-niveaux, conservation) et
   `test_amr_diffusion` (transport multi-niveaux) PASSENT sur GH200 : transferts `average_down` /
   `interpolate` + reflux + transport sont des `for_each` -> device. Le CLUSTERING / la hierarchie
   (`cluster.hpp`, `regrid.hpp`) est de la METADATA hote (box lists, predicats `std::function`) : reste
   hote (correct, infrequent ; nvcc ne la compile pas, ce qui est normal -- ce n'est pas un kernel).
   Les registres de flux gardent des boucles hote (corrects via memoire unifiee ; full-device reflux =
   perf follow-up). python/tests/gpu/{amr_CMakeLists.txt, romeo_amr_build.sh}.
6. **MPI multi-GPU (backend Kokkos Cuda + OpenMPI CUDA-aware).** ✅ FAIT (verifie GH200). `fill_boundary`
   distribue (echange de halos cross-rang) tourne sur 1/2/4 GH200, un GPU par rang, avec OpenMPI
   `+cuda` (UCX). Le MEME `mpi6_fillboundary.cpp` (= `tests/test_mpi_fillboundary.cpp` + init Kokkos +
   `device_fence` avant lecture hote) donne `gfails=0` en np=1/2/4 (`exec=Cuda`) : ghosts distants
   bit-identiques a la valeur periodique attendue, donc transfert device-to-device correct. La memoire
   unifiee + le `device_fence` suffisent ; aucune modif de `fill_boundary`. `python/tests/gpu/mpi6_fillboundary.cpp`.
   RESTE (perf) : halos device sans rebond hote (GPUDirect direct), FFT distribuee device.
7. **Validation bout-en-bout via le System.** ✅ FAIT (verifie GH200) -- sans modif de code.
   `system.cpp` ENTIER (dispatch des modeles, transport HLLC, source de gravite, solve Poisson a CHAQUE
   pas, pas de temps CFL) compile sous nvcc, et le cas `euler_poisson` complet tourne sur GH200 : `max|phi|`
   et `sum(phi)` bit-identiques au CPU, masse a ~1.7e-15 relatif (FMA dans la reduction CFL). Les
   correctifs nvcc des phases 1/2 + le design for_each suffisent. `python/tests/gpu/phase7_system.cpp`.
   RESTE pour la prod multi-GPU : phase 6 (MPI CUDA-aware) + perf (full-device reflux, eviter les
   sync hote des reductions par pas).
8. **Backend AOT sur device (modele genere par le DSL).** ✅ FAIT pour le chemin .so avec
   MARSHALING HOTE (verifie GH200). Un modele `euler_poisson` ECRIT EN FORMULES, compile AOT
   (`compile_or_jit(mode="compile")`) en .so via `compiled_block_abi.hpp` (`add_compiled_block`),
   execute le chemin de production (`assemble_rhs<Minmod, HLLCFlux>` recon primitif + SSPRK2) sur
   le GH200 : residu, masse, quantite de mouvement et energie BIT-IDENTIQUES au build serie hote
   (`sim_aot/`). A leve + corrige un vrai bug : `extract()` lisait la memoire unifiee AVANT la fin
   du kernel async ; ajout d'un `device_fence()`. Le marshaling de tableaux plats traverse le
   dlopen sans partage d'objet C++ (ABI propre) ; ce chemin .so ne porte ni AMR ni MPI.
   RESTE (non fait) : la variante ZERO-COPIE NATIVE (`add_compiled_model` / `dsl_block.hpp`, modele
   compile dans le meme binaire que le System, sans marshaling) est validee BIT-IDENTIQUE a
   `add_block` sur CPU/Serial (`test_compiled_model_parity`), mais sur GPU (backend Cuda) la variante
   a LAMBDAS ETENDUES butait sur une limite nvcc : une lambda etendue `__host__ __device__` instanciee
   dans une TU EXTERNE n'etait pas acceptee. Cette limite a ete levee par le chemin a FONCTEURS NOMMES
   (cf. point 9).
9. **Parite multi-box MPI du chemin compile a foncteurs nommes.** ✅ FAIT (verifie GH200).
   Le residu des fermetures de `make_block` (`block_builder.hpp` ; la machinerie exacte
   d'`add_compiled_model`, instanciee depuis une UNITE DE TRADUCTION EXTERNE via des foncteurs NOMMES,
   le chemin device-clean qui contourne la limite nvcc des lambdas etendues du point 8) doit etre
   invariant au decoupage du domaine en boites ET au nombre de rangs. Le test compare une decomposition
   16-boites distribuee par SFC a une reference mono-boite : `max|R|` BIT-IDENTIQUE (`dmax = 0`), L2 a
   l'arrondi pres de l'ordre de sommation. Porte en test de regression header-only `tests/test_mpi_mbox_parity.cpp`
   (job CI MPI, np=1/2/4, Kokkos Serial sur CPU) ET execute sur GH200 (Kokkos Cuda) : le MEME source,
   compile par nvcc_wrapper et lance par `srun -n {1,2,4} --gpus-per-task=1` (OpenMPI 4.1.7 CUDA-aware,
   noeud armgpu), donne `dmax = 0.00e+00` aux trois comptes de rangs, `maxK`/`max1`/`L2` identiques au
   run CPU. Un `Kokkos::fence()` (garde `ADC_HAS_KOKKOS`) precede la lecture HOTE du residu (kernels
   `for_each_cell` async sous Cuda). Ce que cela valide HONNETEMENT sur device : le chemin
   `make_block`/`add_compiled_model` (foncteurs nommes) + `fill_ghosts` multi-box intra-rang ET
   cross-rang MPI multi-GPU, pour le residu d'un pas. Ce que cela ne valide PAS : l'integration AMR
   dans le meme run, ni la perf full-device (le test lit le residu cote hote, donc fence par pas).

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
.so, dispatch type-erased dans le System, AOT bloc compile a parite native sur CPU/Serial ; flux Roe ;
VariableRole present mais pas encore cable ; eps(x) variable cote coeur, cablage System/Python a faire ;
reorg physics/ numerics/) est COMPLET au niveau PROTOTYPE, teste (~53 ctests C++ + ~16 tests Python) et
verifie jusqu'au GH200 PAR COMPOSANTS SEPARES. Restent : la VALIDATION INTEGREE AmrSystem + MPI + GPU
(seuls des composants separes ont ete valides), la perf full-device (reflux sans rebond hote, halos
GPUDirect device-direct, FFT distribuee device) et la parite AOT zero-copie sur device (limite nvcc,
phase 8). Ce portage runtime reste le gros morceau ouvert.

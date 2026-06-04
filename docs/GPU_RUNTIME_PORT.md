# Portage de la pile runtime sur GPU (feuille de route)

Le DSL et le coeur de calcul sont verifies jusqu'au GPU GH200 (flux, brique generee, CAS Euler complet
via le seam Kokkos `for_each_cell` ; cf. docs/GPU_ROMEO.md). Ce qui reste pour une PRODUCTION GPU de
bout en bout, c'est porter la PILE RUNTIME entiere (System / MultiFab / Poisson / AMR / MPI) sur device.
ETAT (juin 2026) : le solveur MONO-GRILLE complet (transport + BCs + couplages + Poisson + pas de
temps, orchestre par le System) tourne sur GH200, verifie == CPU (phases 1-5, 7) ; les ops de champ
AMR (reflux, transferts) tournent sur device (phase 5) ; l'echange de halos MULTI-GPU est valide
(phase 6, np=1/2/4 bit-identiques) ; le backend AOT .so d'un modele genere par le DSL tourne sur
device (phase 8, avec marshaling hote). La VALIDATION INTEGREE AmrSystem + MPI + GPU (les trois axes
ensemble dans UN SEUL run) est desormais FAITE sur GH200 (phase 10) : np=1/2/4 BIT-IDENTIQUES (dmax=0)
et masse conservee a 0. Ce qui RESTE : la perf full-device (le run integre ne scale pas, le grossier
etant replique -- voir phase 10) et la parite AOT zero-copie sur device. Ce document decoupe en phases.

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
   il se doit, sortie libre). Les ctests coeur (dont test_physical_bc, poisson_disc, cut_cell) verts.
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
10. **Validation INTEGREE AmrSystem + MPI + GPU (les trois axes dans UN SEUL run).** ✅ FAIT (verifie
   GH200). C'etait le dernier verrou : phases 5/6/9 validaient l'AMR, le MPI multi-GPU et le chemin
   compile SEPAREMENT, jamais ensemble. Un harness branche un modele euler_poisson COMPILE via
   `add_compiled_model(AmrSystem, ...)` (chemin `runtime/amr_dsl_block.hpp`, PR #45) sur une vraie
   hierarchie AMR (`AmrSystem` : grossier replique 128^2 + niveau fin 256^2 multi-patch suivi par
   regrid Berger-Rigoutsos, reflux conservatif, Poisson grossier a chaque pas), et DISTRIBUE les
   patchs fins sur `n_ranks()` GH200 (un GPU par rang, halos cross-rang via `fill_boundary`, reflux
   et masse reduits par `all_reduce`). Le MEME source tourne en `srun -n {1,2,4} --gpus-per-task=1`
   (OpenMPI 4.1.7 CUDA-aware, noeud armgpu) sous Kokkos Cuda (`exec=Cuda`), 4 patchs fins, 40 pas
   apres warmup. Resultat (`AMRMPI np=...`) :
   - **BIT-IDENTIQUE au nombre de rangs** : `mass`, `csum`, `csumsq`, `cmax` de la densite grossiere
     IDENTIQUES aux 17 chiffres a np=1 (oracle mono-GPU), np=2 et np=4. `PARITE dmax = 0.00e+00`.
   - **grossier bit-identique cross-rang** : `crossrank_spread = 0.00e+00` (le niveau 0 replique est
     le meme champ sur chaque GPU, donc halos/reflux/injection distants corrects).
   - **masse conservee a 0** : `dm = |mass - m0| = 0.00e+00` (reflux conservatif exact sur device).
   - perf : `per_step_ms` ~221 (np=1), ~266 (np=2), ~272 (np=4) sur un noeud GH200. Le run NE SCALE
     PAS : c'est ATTENDU et HONNETE. Le grossier est REPLIQUE (defaut `replicated_coarse=true`), donc
     le Poisson grossier + le transport grossier sont REDONDANTS sur chaque GPU (compute O(NX*NY) x
     nrangs, zero communication) ; seuls les patchs fins se repartissent (4 patchs -> 2/GPU a np=2,
     1/GPU a np=4). A cette taille le grossier domine -> ajouter des GPU n'accelere pas, ajoute juste
     le cout des halos fins cross-rang. Le mode SCALABLE (`replicated_coarse=false`, grossier reparti)
     existe dans `AmrCouplerMP` mais degrade le MG geometrique (cf. son commentaire) et n'est pas
     cable dans `AmrSystem` : c'est le vrai chantier perf restant pour le strong-scaling AMR.
   Un BUG LATENT a ete corrige au passage : `add_compiled_model(AmrSystem)` ET le chemin natif
   `AmrSystem::build` construisaient le grossier mono-box en `DistributionMapping(1, n_ranks())`
   (round-robin) -> la box ne vivait que sur le rang 0, et `coarse().fab(0)` segfaultait sur les
   autres rangs des le premier write/inject/read sous np>1. Or `AmrCouplerMP` (et `GeometricMG`)
   attendent un grossier REPLIQUE (`DistributionMapping(vector<int>(ba.size(), my_rank()))`). Le
   chemin AMR runtime n'avait simplement jamais ete exerce sous MPI. Corrige (grossier replique) ;
   en serie `my_rank()=0` -> identique bit a bit a l'historique. Porte en test de regression
   header-only `tests/test_mpi_amr_compiled_parity.cpp` (job CI MPI, np=1/2/4, Kokkos Serial sur CPU,
   les memes invariants : `crossrank_spread=0`, conservation, parite au nb de rangs) ; harness GPU
   `python/tests/gpu/{amrmpi_integrated.cpp, amrmpi_CMakeLists.txt, amrmpi_romeo_build.sh}`.

11. **Strong-scaling AMR : grossier REPARTI cable dans AmrSystem.** ⚠️ FAIT (cable + correct sur
    GH200) mais NE SCALE PAS (resultat NEGATIF, chiffre, honnete). Le verrou perf de la phase 10
    etait que le grossier REPLIQUE rend le Poisson + le transport grossiers redondants sur chaque
    GPU. Le mode SCALABLE (`replicated_coarse=false`, grossier multi-box reparti) existait dans
    `AmrCouplerMP` mais n'etait pas CABLE dans `AmrSystem`. Cable ici : `AmrSystemConfig::distribute_coarse`
    (+ `coarse_max_grid`) -> les deux chemins de build (natif `AmrSystem::Impl::build` et compile
    `amr_dsl_block::build_amr_compiled`) construisent le niveau 0 en `BoxArray::from_domain(dom, n/2)`
    (2x2) REPARTI round-robin et passent `replicated_coarse=false` au coupleur, au `GeometricMG` et a
    `advance_amr`. Helpers de grossier centralises (`detail::coupler_{make_coarse_layout,write_coarse,
    read_coarse,inject_coarse_to_fine_mb}`, amr_coupler_mp.hpp) multi-box + distribution-aware
    (lecture/ecriture par boites GLOBALES, reconstruction `density()` n*n par `all_reduce_sum_inplace`
    sur les boites disjointes). Le regrid Berger-Rigoutsos a ete rendu MPI-correct pour un grossier
    reparti : `tag_cells` ne voit que les boites locales -> OU global des tags (`all_reduce_or_inplace`,
    nouveau collectif) avant le clustering, sinon la BoxArray fine differerait par rang ; et le
    remplissage des nouveaux patchs depuis le parent passe par `parallel_copy` quand le parent est
    reparti (au lieu de `mf_find_box`, qui ne voit pas les boites distantes).
    - **CORRECTION (host CI, Kokkos Serial) : grossier reparti == replique BIT A BIT.** Test de
      regression `tests/test_mpi_amr_distributed_coarse.cpp` (np=1/2/4) : meme cas 4 bulles
      euler_poisson, le reparti compare a l'oracle replique dans le MEME binaire ->
      `dist_vs_repl_dmax = 0.00e+00`, `cmax_crossrank_spread = 0`, masse conservee a ~1e-15. Le MG
      CONVERGE sur le grossier 2x2 (champ fini, non trivial, identique au mono-box).
    - **CONVERGENCE MG mesuree (mono-box vs multi-box, MMS).** Diagnostic local (Dirichlet + pic
      gaussien decentre, critere 1e-9) : 2x2 converge en AUTANT de cycles que le mono-box (7-8 a
      n=64/128/256, residus identiques), 4x4 +0/+1 cycle, 8x8 degrade nettement (~13-14 cycles,
      ~1.75x). DONC le 2x2 (defaut, et le seul decoupage utile jusqu'a 4 rangs) NE degrade PAS le
      multigrille ; la degenerescence annoncee n'apparait qu'a decoupage agressif (>=8x8). C'est
      pourquoi `coarse_max_grid` defaut = n/2.
    - **CORRECTION (GH200, Kokkos Cuda, srun -n 1/2/4 --gpus-per-task=1).** np=2 et np=4 reparti :
      `csum`/`csumsq`/`cmax` BIT-IDENTIQUES au replique, masse conservee a 2.2e-16, `cmax` bit-identique
      cross-rang. Un BUG DEVICE a ete trouve+corrige au passage : `parallel_copy` lance des kernels de
      copie ASYNC sous Cuda et, a np=1, RETOURNE SANS fence (la barriere interne n'est que sur le
      chemin np>1) ; le remplissage parent du regrid lisait alors `parloc` (memoire device fraiche)
      avant la fin de la copie -> NaN a np=1 reparti UNIQUEMENT sur Cuda (host Serial: `dmax=0`).
      Corrige par un `device_fence()` apres ce `parallel_copy` (amr_regrid_coupler.hpp).
    - **SCALING (per_step_ms, max sur les rangs, n=128, 40 pas mesures, 1 noeud GH200) -- NEGATIF :**

      | np | REPLIQUE (defaut) | REPARTI (2x2) |
      |----|-------------------|---------------|
      | 1  | 222 ms            | 705 ms        |
      | 2  | 269 ms            | 999 ms        |
      | 4  | 278 ms            | 1403 ms       |

      Run complet refait apres le fix device-fence (np=1 reparti mesure : pas de NaN, dm=2.2e-16,
      cmax bit-identique cross-rang aux 6 points). Le grossier reparti est ~3.7x (np=2) a ~5.0x
      (np=4) PLUS LENT que le replique, et EMPIRE avec le nombre de rangs (705 -> 999 -> 1403 ms).
      Le strong-scaling N'EST PAS atteint. RAISON, honnete : a
      cette taille (grossier 128^2) le compute grossier est trivial, mais le `GeometricMG` multi-box
      echange des halos `fill_boundary` entre boites grossieres a CHAQUE niveau de chaque V-cycle (~7
      niveaux), CROSS-RANG sous MPI, et le pas AMR ajoute `parallel_copy` (inject aux, reflux, regrid)
      device-to-device via UCX. Ce trafic de LATENCE domine largement le compute economise. Distribuer
      un grossier deja petit ECHANGE du compute redondant bon marche contre de la communication chere.
    - **RECO (honnete).** Le cablage est PROPRE, correct et bit-identique au replique (mergeable comme
      OPTION, defaut inchange), mais le strong-scaling AMR par grossier reparti n'est PAS rentable a
      cette echelle. Chemins realistes pour un vrai scaling, non faits ici : (a) grossier reparti
      seulement quand sa MEMOIRE est le verrou (tres grand NX*NY), pas son temps ; (b) MG HYBRIDE :
      multigrille distribue sur les niveaux fins du grossier mais bottom-solve sur un grossier
      RASSEMBLE (gather sur 1 rang) au lieu d'un GS multi-box cross-rang par niveau ; (c) reduire le
      trafic (halos GPUDirect, agglomeration des `parallel_copy`). En l'etat, GARDER le replique par
      defaut (rapide, bit-identique, valide phase 10) et n'activer `distribute_coarse` que comme
      echappatoire memoire. Cable + test de regression CI (`test_mpi_amr_distributed_coarse`, np=1/2/4
      Serial) + harness GH200 (`amrmpi_integrated` mesure replique ET reparti) ; resultat de perf
      documente ici comme NEGATIF chiffre.

## Validation device des features post-#48 (round 2)

La CI ne joue que Release / Python / MPI / Kokkos SERIAL (CPU). Plusieurs briques fusionnees sur
master APRES #48 ont un CHEMIN DEVICE mais n'avaient ete exercees que CPU. On les a confirmees sur
GH200 (noeud `armgpu`, `module load cuda/12.6`, Kokkos 4.4.01 `Kokkos_ARCH_HOPPER90`, `nvcc_wrapper`),
chacune par la MEME logique compilee en `exec=Cuda` (backend Kokkos Cuda, `srun -n 1 --gpus-per-task=1`)
ET en oracle `exec=Serial` (g++, `ADC_HAS_KOKKOS` off), avec comparaison BIT-A-BIT cellule par cellule
(`diff_bin`, `dmax = max|cuda - serial|`). `for_each_cell` est ASYNC sous Cuda : chaque harness fait
`device_fence()` avant la lecture hote / le dump. Harness versionnes (hors CI, gardes par `srun`/sbatch) :
`python/tests/gpu/{gpu_aux_validate,gpu_epm_validate,gpu_amr_bz_validate,diff_bin}.cpp`,
`gpuval2_CMakeLists.txt`, `romeo_gpuval2_build.sh`. Resultats REELS (job sbatch GH200) :

- **T_e lu via `load_aux<5>` (composante aux 4) (#50/#51).** ✅ VALIDE DEVICE. Le portage precedent
  n'avait valide que `load_aux<4>` (B_z, comp 3) ; on ajoute la comp 4 (T_e). Un modele jouet `n_aux=5`
  (flux nul, source `S = T_e u`) lit `a.T_e = a(i,j,4)` dans `assemble_rhs` -> `load_aux<5>` (fonceur
  nomme `AssembleRhsKernel`, `for_each_cell` ADC_HD) sur device. Profil NON CONSTANT `T_e = 1 + x + 2y`.
  exec=Cuda : `R = T_e u` dans [2.1875, 7.8125] (lecture par cellule), `max|R - T_e u| = 0`.
  **`dmax = 0.000e+00`** vs Serial (256 cellules). Bit-identique.
  NOTE D'HONNETETE : on valide ICI le chemin device REEL de la lecture de T_e (`assemble_rhs`, fonceurs
  nommes), PAS le chemin `System::add_compiled_model`. Ce dernier instancie des lambdas etendues
  `__host__ __device__` dans la TU appelante (limite nvcc connue, documentee dans
  `runtime/dsl_block.hpp` et `tests/test_compiled_model_parity.cpp`) et SEGFAUTE a l'execution sur Cuda
  -- independamment de T_e (un harness System+`add_compiled_model`+`eval_rhs` a bien crashe sur GH200,
  `compute-sanitizer memcheck` = 0 erreur device, donc crash cote hote/lambda etendue). Le marshaling
  T_e du chemin System (`apply_te`, `copy_state` comp 4) reste donc couvert uniquement en CI Serial.

- **EPM ECRANTE / Helmholtz `div(eps grad phi) - kappa phi = f` (#44, `GeometricMG::set_reaction`).**
  ✅ VALIDE DEVICE. Le terme `kappa` vit dans les `for_each_cell` ADC_HD du smoother red-black, du
  residu et de l'apply (`numerics/elliptic/poisson_operator.hpp`) -> device sous Cuda. MMS `eps=1+0.5x`
  + `kappa=50`, Dirichlet exact, V-cycles avec le meme critere que `tests/test_screened_poisson.cpp`.
  exec=Cuda : cycles 8/9/9 (IDENTIQUES a Serial), convergence ordre 2 (ratios Linf 3.69 / 3.85).
  **`dmax = 0.000e+00`** vs Serial sur phi (n=64, 4096 cellules). Memes cycles, meme phi au bit pres.

- **EPM ANISOTROPE `div(diag(eps_x, eps_y) grad phi) = f` (#52/#56, `set_epsilon_anisotropic`).**
  ✅ VALIDE DEVICE. Le second champ `eps_y` (faces normales a y) est lu dans les memes
  `for_each_cell` ADC_HD que ci-dessus. MMS `eps_x=1+0.5x`, `eps_y=1+0.3y` (cf.
  `tests/test_anisotropic_epsilon.cpp`). exec=Cuda : cycles 9/10/11 (IDENTIQUES a Serial), ordre 2
  (ratios Linf 4.00 / 4.00). **`dmax = 0.000e+00`** vs Serial sur phi (n=64, 4096 cellules).

- **B_z par niveau dans le chemin AMR (#53, `AmrSystemCoupler::fill_bz`).** ✅ VALIDE DEVICE. B_z(x,y)
  est pose aux centres DE CHAQUE NIVEAU (`geom.refine(1<<k)`, dx = dx_coarse / 2^k) sur la comp
  `kAuxBaseComps` du canal aux partage ; le modele le lit `load_aux<4>` dans le noyau source AMR
  (`for_each_cell` ADC_HD) niveau par niveau. Profil NON CONSTANT `B_z = 1 + sin(2 pi x) cos(2 pi y)`
  pour distinguer les niveaux. exec=Cuda : `B_z` relu = 0.80865828 au niveau 0 (centre (4,4)),
  0.90245484 au niveau 1 (centre (8,8)) -- VALEURS DISTINCTES, chacune == son centre de niveau ; la
  source consomme le bon B_z par niveau (grossier et fin evoluent avec LEUR B_z). **`dmax = 0.000e+00`**
  vs Serial sur U grossier+fin (512 cellules, 2 niveaux), conservation respectee.
  Validation initiale par `advance_amr` (HEADER-ONLY, le moteur que `AmrSystemCoupler` appelle niveau
  par niveau). La FACADE `AmrSystemCoupler` ENTIERE est desormais validee sous nvcc elle aussi (limite
  device (b) LEVEE) : le concept `CoupledSystemLike` sondait `for_each_block` avec une LAMBDA GENERIQUE
  en contexte non evalue (`requires s.for_each_block([](auto&){})`), que le frontend nvcc/EDG refusait
  -> `CoupledSystemLike<CoupledSystem<...>>` faux sous Cuda -> CTAD du coupleur impossible. La sonde
  est passee a un FONCTEUR NOMME `detail::ForEachBlockProbe` (meme recette que les foncteurs nommes,
  point 8). Harness `python/tests/gpu/gpu_amrsys_facade_validate.cpp` (instancie la facade entiere :
  CoupledSystem 2 blocs + Poisson de systeme + `solve_fields` + `step` sur AMR 2 niveaux) : GH200
  `CUDA_BUILD_OK`, `exec=Cuda` OK, U(grossier+fin, 2 blocs) **`dmax = 0.000e+00`** vs Serial,
  avant/apres confirme (sonde lambda remise -> nvcc echoue sur la CTAD).

- **B_z multi-box AMR distribue sur plusieurs GPU (#59).** ⚠️ FONCTIONNEL DEVICE multi-GPU, mais PAS
  bit-identique au sens strict sur les sommes globales. #59 a fusionne sur master la couverture
  multi-box (mono-rang + MPI np=2/4, CI Kokkos Serial). Sur GH200 (np=1/2/4, un GH200 par rang,
  exec=Cuda, OpenMPI CUDA-aware, grossier 2x2 boites + 2 patchs fins disjoints repartis SFC,
  `coarse_replicated=false`) : B_z est correctement echantillonne PAR NIVEAU et PAR BOITE locale
  (`bz_bad = 0` a chaque np) et la source `S = B_z u` le lit par boite sur le device. `cmax`
  (reduction max, insensible a l'ordre) est BIT-IDENTIQUE aux trois np (`dcmax = 0`). En revanche les
  invariants ADDITIFS globaux (mass/csum/csumsq, `all_reduce_sum` sur les boites locales) DIFFERENT au
  niveau de l'arrondi entre np : `dmass ~ 1e-15`, `dcsum ~ 3e-13`, `PARITE dmax = 9.1e-13` (np=2/4 vs
  oracle np=1) -- effet d'ORDRE DE REDUCTION FMA (le grossier multi-box est genuinement reparti, donc
  la somme partielle change d'ordre selon le decoupage par rang). Ce n'est pas un bug : le calcul
  device par cellule est correct et le max est exact ; seules les sommes flottantes dependent de
  l'ordre. Contraste avec phase 10 (`amrmpi_integrated`, dmax=0) ou le grossier est REPLIQUE -> chaque
  rang somme le MEME domaine entier -> reductions identiques. Honnetement : multi-GPU FONCTIONNELLEMENT
  CORRECT (B_z par boite/niveau lu sur device, conservation a l'arrondi) mais bit-identite stricte NON
  atteinte sur les quantites sommees quand le grossier est reparti. Harness
  `python/tests/gpu/{gpu_amr_bz_mpi_validate.cpp, gpuval2_mpi_CMakeLists.txt, romeo_gpuval2_mpi_build.sh}`.

Bilan round 2 : les 4 features mono-GPU a chemin device (T_e, EPM Helmholtz, EPM anisotrope, B_z par
niveau AMR) sont confirmees BIT-IDENTIQUES (dmax=0) sur GH200 en exec=Cuda vs Serial ; pour chaque
elliptique, memes cycles MG ; conservation respectee. Le B_z multi-box distribue multi-GPU (#59) est
fonctionnellement correct (B_z par boite/niveau lu sur device, `bz_bad=0`, `dcmax=0`) mais les sommes
globales ne sont pas bit-identiques entre np (ordre de reduction, dmax ~ 9e-13). Reserve honnete : le
chemin `System::add_compiled_model` sur Cuda reste limite par les lambdas etendues cross-TU (suivi
existant phase 8) -- la lecture des champs aux (B_z, T_e) a ete validee device via `assemble_rhs` /
`advance_amr` (fonceurs nommes), qui sont les chemins device reels.

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
reorg physics/ numerics/) est COMPLET au niveau PROTOTYPE, teste (71 ctests C++ coeur, +21 entrees
ctest MPI, 26 tests Python) et verifie jusqu'au GH200. La VALIDATION INTEGREE AmrSystem + MPI + GPU (les trois axes dans un seul run)
est FAITE (phase 10, GH200, dmax=0, masse conservee a 0). Les briques a chemin device fusionnees
APRES #48 (T_e via `load_aux<5>`, EPM ecrante/Helmholtz #44, EPM anisotrope #52/#56, B_z par niveau
AMR #53) sont confirmees BIT-IDENTIQUES sur GH200 (round 2, dmax=0, memes cycles MG, conservation).
Restent, cote PERF et non plus correction : le strong-scaling AMR full-device (grossier reparti
`replicated_coarse=false` cable dans `AmrSystem`, reflux sans rebond hote, halos GPUDirect
device-direct, FFT distribuee device) et la parite AOT zero-copie sur device (limite nvcc, phase 8 ;
c'est aussi pourquoi T_e a ete valide device via `assemble_rhs` et non `add_compiled_model`).

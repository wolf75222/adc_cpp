# Audit de conformite aux standards C++ de `adc_cpp`

Date : 2026-06-12.
Base relue : `origin/master` / `ffb9022`.
Perimetre : l'integralite du depot, soit 283 fichiers et ~56 700 lignes, decoupes en 14 unites
d'audit : coeur/AMR/parallele/physique, maillage, numerique (3 lots), couplage (2 lots), runtime
(3 lots), bindings Python, tests (2 lots) et bench/CMake/scripts. Sont relus tous les
`include/adc/**/*.hpp`, `python/*.cpp`, `tests/**/*.cpp`, `python/tests/**/*.cpp`, `bench/*.cpp`,
`CMakeLists.txt`, `CMakePresets.json`, `.clang-format`, `.clang-tidy` et les scripts de build.

Methode : relecture par sous-systeme, une unite a la fois. Chaque constat
non cosmetique a ete verifie sur piece (lecture du `file:line` cite, du site jumeau et de la
convention de fait du depot avant classification). Les constats qui se sont reveles infondes ou
survendus a la verification ont ete soit retires, soit retrogrades en severite : 6 constats ont
ete rejetes purement (cf. note de cloture en fin de section 6), et une vingtaine de "important" initiaux ont ete ramenes a
"cosmetique" faute de trigger ou d'impact reel. Le document ne reporte que les ecarts INTERNES aux
conventions du depot ou les regles de fond reellement enfreintes ; les regles-guides
deliberement non suivies (snake_case, `#pragma once`, pas de `[[nodiscard]]`...) sont listees une
fois en section 3 et NON comptees comme ecarts.

Ce document est un audit de style et de conformite, pas une roadmap scientifique ni un audit de
maintenabilite. Il complete :
- [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md) : audit de maintenabilite et de responsabilites.
- `CODE_DOCUMENTATION_CONVENTION.md` : convention de commentaires/Doxygen. ATTENTION : ce fichier
  est reference par `CODEBASE_AUDIT.md` mais n'a jamais ete commite : il n'existe que dans l'arbre
  de travail principal, donc le lien est mort pour tout clone frais de `master`. A committer
  (suivi via ADC-125).
- `CODING_STANDARDS_DECISIONS.md` : registre des arbitrages entre Google / Core Guidelines / LLVM
  pour le profil cible. C'est la que doivent etre actes les choix de la section 3 ; cet audit en
  suppose le contenu et le designe comme la source de verite des conventions.

## 1. Lecture globale

Le depot est de tres bonne facture et, surtout, COHERENT avec lui-meme. Il ne suit pas un guide
externe unique : il applique une convention maison stable (snake_case, header-only `#pragma once`,
structs POD device-copyables, idiome `ADC_HD`, foncteurs nommes plutot que lambdas etendues sous
`nvcc`, exceptions `std::runtime_error` prefixees, casts toujours explicites). Le verdict global de
conformite n'est donc pas "suit Google" ou "suit Core Guidelines", mais : le depot suit
majoritairement et fidelement une convention interne coherente, et les ecarts releves sont des
ecarts a CETTE convention, plus quelques regles de fond (securite/UB/portabilite) que les trois
guides partagent.

Deux constats seulement sont bloquants, tous deux conditionnels au chantier de portage Windows
(epic ADC-90) ou au chemin device :
- `include/adc/mesh/box_hash.hpp:71-73` : `static_cast<long>(bx) << 32` suppose `long >= 64` bits ;
  sur ABI LLP64 (Windows natif) `long` fait 32 bits, le decalage est un comportement indefini et la
  cle perd `bx` -> hash spatial casse. Verifie sur piece.
- `include/adc/numerics/elliptic/elliptic_problem.hpp:104` : `field_postprocess` dispatche via une
  lambda etendue `[=] ADC_HD(int i, int j)` sur un chemin device, exactement le motif que le reste
  du repertoire elliptique bannit (segfault `nvcc` cross-TU, doctrine #93). Verifie : c'est le seul
  ADC_HD-lambda de tout `elliptic/`.

Aucun autre UB certain ni bug dur n'a ete trouve. Les autres risques (membres POD non initialises,
pointeurs/references membres qui aliasent des stores sans regle des cinq, gardes `assert` perdues
sous NDEBUG) sont LATENTS : neutralises par l'usage actuel mais a durcir.

## 2. Synthese chiffree

180 constats retenus apres verification (6 rejetes en plus, cf. note de cloture en fin de section 6).
Le tableau ci-dessous fait foi pour le decompte ; les sections 4.x enumerent les constats
representatifs par sous-systeme et regroupent les ecarts repetitifs, si bien que le comptage litteral
des puces y derive de un a deux items par section.

| Sous-systeme | Fichiers | Lignes | Bloquant | Important | Cosmetique | Total |
|---|---|---|---|---|---|---|
| coeur/AMR/parallele/physique | 23 | 2861 | 0 | 2 | 13 | 15 |
| maillage | 13 | 1912 | 1 | 5 | 7 | 13 |
| numerique-1 (elliptique) | 12 | 3252 | 1 | 5 | 14 | 20 |
| numerique-2 (flux/operateurs) | 10 | 3119 | 0 | 2 | 12 | 14 |
| numerique-3 (temps/AMR) | 13 | 2198 | 0 | 1 | 14 | 15 |
| couplage-1 | 10 | 2499 | 0 | 0 | 14 | 14 |
| couplage-2 | 7 | 1850 | 0 | 3 | 4 | 7 |
| runtime-1 (DSL/AMR system) | 5 | 3163 | 0 | 1 | 10 | 11 |
| runtime-2 (loader/ABI) | 13 | 2855 | 0 | 1 | 15 | 16 |
| runtime-3 (stepper/store) | 4 | 1380 | 0 | 1 | 7 | 8 |
| bindings Python | 3 | 3654 | 0 | 4 | 14 | 18 |
| tests-a | 72 | 12888 | 0 | 1 | 7 | 8 |
| tests-b | 86 | 13293 | 0 | 2 | 5 | 7 |
| bench/CMake/scripts | 12 | 1734 | 0 | 2 | 12 | 14 |
| TOTAL | 283 | 56658 | 2 | 30 | 148 | 180 |

Repartition par dimension (importants et bloquants) : organisation/DRY domine (duplication de
cascades de dispatch, de stencils, de pack/unpack, de harnais de test), devant securite (UB latent,
gardes `assert` non durcies, modulo par zero non garde), types (regle des cinq incomplete, membres
non initialises) et idiomes device (lambdas etendues residuelles).

Verdict global : conforme a la convention interne, avec une dette de duplication structurelle
concentree sur les chemins AMR/Schur/DSL et un point de portabilite reel (`long` 32 bits sur
LLP64) qui devient bloquant des que le port Windows avance.

## 3. Cadrage : pourquoi les trois guides ne sont pas superposables

La cible est une bibliotheque C++23 header-only, HPC, Kokkos/CUDA (`nvcc`, chemin device) avec
bindings pybind11. Sur ce profil, Google C++ Style, C++ Core Guidelines et LLVM Coding Standards
divergent sur des axes mutuellement exclusifs : on ne peut pas satisfaire les trois a la fois.

- Casse des fonctions : `UpperCamelCase` (Google) vs `camelBack` (LLVM) vs `snake_case` (CG) :
  trois styles incompatibles. Le depot tranche pour `snake_case` (CG).
- Casse variables/membres et constantes : `snake_case` + `kCamelCase` (Google/CG) vs
  `UpperCamelCase` (LLVM). Le depot suit `snake_case` pour les locaux, suffixe `_` pour les membres
  prives, `kCamelCase` pour les magic-numbers.
- Include guard : `#ifndef` impose (Google/LLVM) vs `#pragma once` tolere (CG SF.8). En header-only
  le depot prend `#pragma once`, ergonomique.
- Ordre des includes : std tot (Google) vs system en dernier (LLVM), ordres opposes. Le depot
  impose son propre ordre : bloc `<adc/...>` d'abord, ligne vide, puis STL (avec `SortIncludes:false`).
- Exceptions et RTTI : interdits (Google/LLVM) vs recommandes E.2/E.3 et `dynamic_cast` C.146 (CG).
  Le code `__device__` ne peut ni `throw` ni `typeid` -> la position no-except/no-RTTI est imposee
  de facto sur le chemin device ; les exceptions vivent cote hote uniquement.
- Contrats : `assert`/`CHECK` (Google/LLVM, device avec reserves) vs `Expects()/Ensures()` GSL (CG,
  non device-callable). Le depot utilise `assert` hote + `static_assert` de modelisation par concepts.
- `[[nodiscard]]` : sous-specifie par les trois. Le depot le DESACTIVE volontairement
  (`modernize-use-nodiscard` retire dans `.clang-tidy`).

Consequence methodologique : ces choix sont coherents dans tout le depot et relevent d'un
arbitrage, pas d'un defaut. Ils ne sont PAS comptes comme ecarts. Les arbitrages eux-memes doivent
etre actes dans `CODING_STANDARDS_DECISIONS.md` ; le present audit ne juge que la conformite a la
convention interne ainsi fixee, et les regles de fond communes aux trois guides (init des objets,
pas de division par zero, regle des cinq, pas d'UB de decalage).

Regles-guides deliberement non suivies, valides dans tout le depot (rappel unique, NON reportees
comme constats) : `snake_case` (vs CamelCase Google) ; `struct` a membres publics pour les
POD/policies/foncteurs ; `#pragma once` ; absence de `[[nodiscard]]` ; commentaires en francais sans
accents ; `Real(litteral)` idiomatique ; tableaux C `Real v[N]` (device-clean) ; out-params par
reference pour les retours multiples device (F.21) ; capture du modele PAR VALEUR dans les foncteurs ;
`using namespace adc;` au scope fichier dans les TU de test/bench/bindings (feuilles, pas d'en-tete).

## 4. Constats par sous-systeme

### 4.1 coeur/AMR/parallele/physique (23 fichiers)

Tres bonne facture : architecture concept-driven (`PhysicalModel`/`HyperbolicPhysicalModel`/
`EquationBlockLike`), invariant device-clean rigoureux, AMR Berger-Rigoutsos solide, documentation
dense. Aucun bug dur ; les deux risques reels sont latents.

Important :
- `include/adc/core/variables.hpp:32-44` (C.48/ES.20) : `struct Variable` laisse `role`/`component`
  et `struct VariableSet` laisse `kind`/`size` sans initialiseur in-class, alors que `ArenaStats`/
  `ClusterParams`/`Aux` en posent. Type public default-constructible -> lecture = UB. Fix trivial
  sans casser l'agregat `{kind, names, size}`.
- `include/adc/physics/hyperbolic.hpp:81-114` (DRY) : `ExBVelocityPolar` duplique octet pour octet
  `ExBVelocity` (l.27-59) ; seule la semantique (cartesien vs base locale polaire) differe, portee
  par les commentaires. Le meme fichier resout le cas jumeau par heritage (`IsothermalFluxPolar :
  IsothermalFlux`). Aliaser ou factoriser.

Cosmetique (13) : `equation_block.hpp:63` (`string_view` membre potentiellement pendant, latent) ;
usage pervasif de `long` pour les comptes de cellules -> 32 bits sur LLP64 ; `regrid_level` tague
seulement les fabs locaux sans le `all_reduce_or` documente (correct en serie) ; `cluster_rec`
imbrique 3-4 niveaux ; conversions `double(x)` ; locaux capitalises `Sx`/`Sy`/`D` ; placement non
uniforme `@file`/`#pragma once` ; `comm.hpp`/`load_balance.hpp` sans en-tete Doxygen ; double
documentation Doxygen+prose ; ilot `/** */` en `physics/` ; commentaire "16 bits" obsolete dans
`part1by1` ; `ManagedArena::stats()` non-const ; `VariableSet::size` denormalise vs `names.size()`.

### 4.2 maillage (13 fichiers)

Propre et coherent : `#pragma once` unanime, structs POD device-copyables, regle-de-0 respectee,
const-correctness et `explicit` corrects.

Bloquant :
- `include/adc/mesh/box_hash.hpp:71-73` (Integer Types/ES.101) : `BoxHash::key()` fait
  `static_cast<long>(bx) << 32` et stocke dans `unordered_map<long, ...>`. Sur LLP64 le `<< 32` est
  un decalage >= largeur du type = UB, et `bx` est perdu -> collisions massives, hash casse. Passer
  a `std::int64_t`.

Important :
- `fab2d.hpp:45,95,123-127` + `box2d.hpp:54` + `box_array.hpp:52` : `long` natif pour
  strides/tailles/offsets memoire ; overflow silencieux > 2^31 sur LLP64, compounde le bloquant.
- `refinement.hpp:44-47` / `box2d.hpp:111-114` / `box_hash.hpp:68` : la division-plancher entiere
  (gestion du negatif) est ECRITE TROIS FOIS, corps identiques. Factoriser un `floor_div` ADC_HD.
- `refinement.hpp:179,209` + `multifab.hpp:137` : lambdas etendues `[=] ADC_HD` sur le chemin chaud
  V-cycle MG, alors que `fill_boundary`/`mf_arith`/`physical_bc` ont migre vers des foncteurs nommes
  (meme mode de panne `nvcc` cross-TU documente).
- `patch_box.hpp:16-20` (C.48/C.49) : 5 membres `level/ilo/jlo/ihi/jhi` sans initialiseur alors que
  tous les POD freres (`Box2D`, `Array4`, `Geometry`, `BCRec`) initialisent ; type exporte a Python.
- `refinement.hpp:121-157` : `parallel_copy` emballe/deballe les messages MPI en boucles hote
  4-imbriquees dupliquees envoi/reception avec `std::vector` `std::allocator`, alors que
  `fill_boundary` fait le meme pack/unpack en `for_each` device + `comm_allocator` epingles.

Cosmetique (7) : ordre d'includes inverse + `<cstdlib>/<cstdio>` inutilises dans `refinement.hpp` ;
`fill_boundary_begin`/`parallel_copy` trop longues ; methodes POD `Box2D`/`Geometry` non `constexpr`
(174 `constexpr` ailleurs) ; locaux camelCase ; doubles en-tetes Doxygen+prose ; `patch_box.hpp`
sans `@file` ; divisions par `r` non gardees dans `coarsen`/`refine`.

### 4.3 numerique-1 elliptique (12 fichiers)

Sain et homogene : C++23, header-only, RAII, conception concept-driven, aucun cast C, divisions
gardees, vrai soin device (foncteurs nommes, tampons sur la pile O(N^2)).

Bloquant :
- `elliptic_problem.hpp:104` : seul kernel de l'unite a dispatcher via lambda etendue `[=] ADC_HD`
  sur un chemin device (phi -> aux grad consomme par le coupler), motif banni par
  `poisson_operator.hpp:127-132`. Extraire un foncteur `detail::FieldPostprocessKernel`.

Important :
- `poisson_fft_solver.hpp:114` : garde `Ny % n_ranks() == 0` par `assert` (perdu sous NDEBUG) alors
  que `nyl_` est deja tronque dans la liste d'init -> slabs mal dimensionnees, OOB en Release. Le
  jumeau `PoissonFFTSolver` a justement migre vers `throw`. Durcir en `throw`.
- `poisson_operator.hpp:150-166 / 203-219 / 332-349` : le bloc permittivite de face harmonique
  (+ branche cut-cell) recopie a l'identique dans les trois foncteurs. Factoriser `face_weights(...)`.
- `polar_poisson_solver.hpp:260-316` : `residual()` reconstruit a la main l'operateur deja assemble
  par `solve()` (coeffs radiaux, valeur propre azimutale, repli BC, FFT par ligne). Risque de derive
  solve/residu.
- `geometric_mg.hpp:189-360` : 6 surcharges `set_*` repetant init-hote + restriction + ghosts ; le
  repli `kappa` omet `fill_ghosts` (risque de bug silencieux). Un template prive `sample_per_level`.
- `composite_fac_poisson.hpp:377-480` : `composite_coarse_residual()` ~103 lignes, residu base + 4
  blocs C-F quasi symetriques (imbrication 4-5 niveaux) + norme inf. Decouper par direction.

Cosmetique (14) : ctor convertisseur implicite `DistributedFFTSolver` ; `TensorKrylovSolver` a deux
references-membres (lifetime) ; `hqr_minmax()` ~133 lignes (portage EISPACK verbatim) ; 8 fichiers
sans `@file` ; ordre includes inverse `dense_eig.hpp` ; locaux matriciels MAJUSCULE ; `kPi_` litteral
vs `std::numbers::pi` ; `std::function` par valeur sans `move` ; accesseurs `op_*()` non-const ;
`x,y,z,w,s` non initialises ; init positionnel fragile de `MGLevel` ; helpers FFT dans `adc::` au
lieu de `detail::` ; seuils numeriques magiques ; prefixes d'exception heterogenes.

### 4.4 numerique-2 flux/operateurs (10 fichiers)

Mature et tres documente, discipline device forte (foncteurs nommes, gardes `if constexpr`,
divisions protegees). Aucun UB franc dans les kernels.

Important :
- `amr_flux_helpers.hpp:85-90` : `mf_apply_source` (Model-template, chemin Euler par defaut) lance un
  kernel via lambda etendue `[=] ADC_HD` capturant le modele et appelant `m.source(...)`, alors que
  `AmrSspRhsKernel` juste au-dessus et toute l'unite imposent le foncteur nomme.
- `polar_tensor_operator.hpp:726-727` : `PolarTensorKrylovSolver` porte des pointeurs `a_rr_`/`a_tt_`
  qui aliasent ses propres stores `a_rr_store_`/`a_tt_store_`, sans copy/move/dtor (regle des cinq) ;
  `MultiFab`/`Fab2D` etant copiables ET movables, une copie/move fait pendre les pointeurs (UB).
  `= delete` la copie/move ou re-pointer dans un move ecrit a la main.

Cosmetique (12) : `std::numeric_limits` sans `#include <limits>` (compile par transitivite) ;
`assert(a_rr && a_tt)` au lieu de `throw` (deref nullptr en NDEBUG) ; helpers de stencil a 9-11
parametres homogenes interchangeables ; sous-indentation `else` (auto-fixable) ; 2 fichiers sans
`@file` ; `@file` apres includes ; documentation dupliquee 3x ; `__is_trivially_copyable`
(identifiant reserve) ; out-params `Real&` (idiome device) ; locaux notation mathematique ;
duplication structurelle des 4 assembleurs RHS.

### 4.5 numerique-3 temps/AMR (13 fichiers)

Deux familles : fichiers socle courts et propres (steppers, splitting, imex, ssprk) ; moteurs AMR et
Newton implicite, corrects mais lourds. Securite globalement bonne (indexation `static_cast<size_t>`,
encodage cellule borne).

Important :
- `implicit_stepper.hpp:234-257 / 297-320` (F.3/DRY) : l'assemblage de la jacobienne Newton
  (`if constexpr (HasSourceJacobian) {...} else {fd...}`) duplique ~24 lignes mot pour mot entre le
  chemin 2a et le chemin 2b, censes etre bit-identiques -> risque de divergence silencieuse.
  Extraire `assemble_newton_jacobian(...)`.

Cosmetique (14) : `fab(bL/bR/bB/bT)` non garde sur `mf_find_box==-1` (protege par invariant
parent-replique) ; constantes d'encodage `1048576`/`16` dupliquees encode/decode ; `fail_policy` en
`int`+`kFail*` vs `enum class` ; `subcycle_level_mp` ~205 lignes ; `struct Reg` local == `RegMP` ;
motif des 4 faces recopie ~8 fois (dont oracles volontaires) ; 13 fichiers sans `@file` ; `double`
brut au lieu de `Real` dans les oracles ; `TimeTreatment::Explicit` nu vs prefixe `k` ;
sous-indentation `else` ; pointeurs bruts non-detenant `aux`/`pOld`/`pNew` ; callables par valeur vs
`Step&&` ; reuse de `r`/`L`/`m` ambigus ; `char msg[256]` snprintf au lieu de `std::string`.

### 4.6 couplage-1 (10 fichiers)

Solide et coherent : `#pragma once`, ordre includes respecte, `ADC_HD` + foncteurs nommes,
concepts/`requires`, `enum class`, exceptions, casts explicites, sink par valeur+`move`. Aucun ecart
bloquant, aucun important apres verification (les 6 candidats importants ont ete retrogrades).

Cosmetique (14) : triple duplication du mapping `BCRec->Foextrap` (`coeff_bc` x2 +
`detail::derive_aux_bc`) alors que `aux_fill.hpp` centralise deja ; `shared_ptr` la ou `unique_ptr`
suffit (incoherent avec la classe soeur) ; `CsProgram::eval` depile sans garde de pile (protege
seulement cote Python) ; validation `n*n` en `int` debordable (la soeur evite en `size_t`) ;
`max_wave_speed`/`level_state`/`level_potential` non-const ; `AmrCouplerMP` 626 lignes
orchestration+marshaling ; `step_multilevel` dense ; locaux SCREAMING `PNX`/`PNY` ; doubles en-tetes
Doxygen+prose ; prefixes d'exception heterogenes ; `same_box` reimplemente `operator==` ; tableau C
`MultiFab* saved[]` ; `(void)dom` vs `[[maybe_unused]]` ; membre `bcPhi_` camelCase.

### 4.7 couplage-2 (7 fichiers)

Code soigne, device-clean, aucun constat bloquant.

Important :
- `system_coupler.hpp:61-70` (C.21) : `ScopedBlockState` scope-guard a un dtor mutateur actif sans
  `= delete` copie/move -> double-restauration possible si copie.
- `system_coupler.hpp:244,292` (ES.45) : `Real(1e-30)` en dur alors que `kCflSpeedFloor` existe
  (`core/types.hpp:49`) et est documente comme remplacant du "1e-30 disperse".
- `system_coupler.hpp:143-147` (DRY) : `SystemAssembler::derive_aux` re-encode la convention
  `FieldPostProcess{Plus,true}` inline au lieu du helper `detail::coupler_grad_phi` deja utilise par
  `Coupler::derive_aux` -> risque de derive de convention.

Cosmetique (4) : `step()` polaire ~118 lignes ; precondition `Ny/np_` non verifiee (classe
DEPRECATED, code mort) ; locaux majuscules + `cmath` inutile + `@file` avant `#pragma once` ; double
en-tete + forwarding de callback incoherent + phases polaires mal numerotees.

### 4.8 runtime-1 DSL/AMR system (5 fichiers)

Bonne tenue : `@file`/`@brief` partout, `#pragma once`, C++ moderne (concepts, `if constexpr`,
PIMPL forward-declare), surete ABI (`abi_key.hpp`) et device-clean exemplaires.

Important :
- `amr_dsl_block.hpp:596-703 / 707-778` + `block_builder.hpp:443-528` (DRY) : la cascade de dispatch
  riemann x limiteur (rusanov/hll/hllc/roe x none/minmod/vanleer/weno5) avec ses gardes `if constexpr`
  est repliquee TROIS fois. Les commentaires (l.665-667, 750-752) actent une divergence de table
  deja survenue (hllc AMR sans weno5). Extraire un gabarit de dispatch parametre par l'action
  terminale.

Cosmetique (10) : `add_block` ~18 params dont 6 scalaires Newton a plat alors que `NewtonOptions`
existe (les fonctions internes l'utilisent deja) ; locaux MAJUSCULE `I0/I1/J0/J1`/`PNX/PNY` ;
`SourceNewtonReport` 8 membres non initialises ; macros d'aide `abi_key` non `#undef` ; lignes de
code >100 col non reformatees (auto-fixable) ; `build_amr_compiled` ~250 lignes ; refs non-const vers
l'etat interne ; prefixe d'exception classe-qualifie vs `adc (...)` ; `mutable` superflu sur
`solve_count_` (auto-fixable) ; copie non `= delete` explicitement sur `AmrSystem` move-only.

### 4.9 runtime-2 loader/ABI (13 fichiers)

Sain et tres documente, pas de bug/UB franc, marshaling host/device prudent (`static_cast`
systematiques, gardes `local_size()==0`, `device_fence()` avant lecture hote). `dynlib.hpp`/
`export.hpp`/`runtime_params.hpp`/`model_spec.hpp` sont des couches propres.

Important :
- `system.hpp:122-134` (I.23) : `add_block` ~19 params (8 `newton_*`), `set_source_stage` ~11,
  `add_coupled_source` ~12 ; cote ABI `residual` 13 et `advance` 15. Parametres de memes types
  adjacents = footgun d'ordre. Regrouper en POD (`NewtonOptions`, `SourceStageOptions`) comme
  `ModelSpec` deja present.

Cosmetique (15) : fonctions extraites VERBATIM de `system.cpp` sans redecoupage
(`add_compiled_block` ~222 l.) ; `3` magique vs `kAuxBaseComps` ; `IModel<NV>` base polymorphe sans
suppression de copie (abstraite, risque pratique nul) ; prefixes de message `System::` vs nu ;
include `adc` intercale dans le bloc STL ; lignes >100 col (auto-fixable) ; sinks par valeur sans
`move` ; variable morte `nn` (auto-fixable) ; SFINAE `void_t` vs `requires` ; locaux mono-lettre +
max ecrit a la main ; entiers stockes en `double` dans `SourceNewtonReport` ; recon MUSCL en `int`
magique 0/1/2 ; `reinterpret_cast` dlsym (~20 sites, a confiner) ; const-correctness `potential()`.

### 4.10 runtime-3 stepper/store (4 fichiers)

Sain et de bonne facture : C++ moderne maitrise, iteration MPI-safe systematique sur les fabs
locaux, `@file`/`@brief` partout, `#pragma once`, messages d'erreur coherents.

Important :
- `system_stepper.hpp:241-517` (DRY) : les 4 macro-pas (`step`/`step_strang`/`step_cfl`/
  `step_adaptive`) repliquent verbatim la boucle d'avance de bloc, le calcul du pas physique `h`, la
  boucle des bornes globales `dt_bounds_` (qui porte la semantique MPI `all_reduce_min` dont la
  desync = deadlock) et le calcul `n_b`. Extraire `cfl_grid_h()`, `apply_global_dt_bounds()`,
  `for_each_due_block()`.

Cosmetique (7) : `BlockState` membres `ncomp/substeps/evolve/gamma` non initialises (UB hypothetique,
tous les sites construisent par agregat) ; sentinelle `1e30` au lieu de `numeric_limits` ; `reg[]`
non zero-init (auto-fixable) ; `poisson_bc`/`wall_active` non-const ; surcharges `find()` dupliquees ;
donnees publiques exposees via `class` + back-pointer brut (delibere, documente) ; bloc STL scinde
en deux.

### 4.11 bindings Python (3 fichiers)

Sain et tres defensif (gardes de taille avant memcpy, erreurs prefixees, lambdas device capturant
par valeur, sink `std::function`/`BlockClosures` par valeur+`move`, aucun cast C, `reinterpret_cast`
confine au dlopen). Aucun bug bloquant certain.

Important :
- `amr_system.cpp:849-852` (ES.105) : `set_conservative_state` calcule `nn = cfg.n * cfg.n` puis
  `U.size() % nn` ; `cfg.n==0` (settable depuis Python, ctor ne valide pas) -> modulo par zero = UB
  atteignable. Valider `cfg.n >= 1` a la construction.
- `system.cpp:57-93` + `amr_system.cpp:29-61` (DRY) : `resolve_implicit_components` copie quasi a
  l'identique entre les deux TU (copie admise au commentaire). Extraire dans un header partage.
- `system.cpp` (~12 sites, l.392/407/427/905/1038/1492/1716/1744/1759/1783/1823...) : la boucle de
  marshaling `out[(c*gny+j)*gnx+i]` est repetee ~12 fois quasi identique. Helpers `gather_fab`/
  `scatter_fab`.
- `system.cpp:502-558` + `amr_system.cpp:498-518` (DRY) : validation des options Newton dupliquee, +
  rapport Newton triple (lambda `py::dict` x2 dans `bindings.cpp`, conversion `SourceNewtonReport`
  x2). Un `parse_newton_options(...)` partage + une fabrique unique du `py::dict`.

Cosmetique (14) : `using namespace adc;` au scope fichier dans `bindings.cpp` ; fonctions tres
longues (`add_block`, `add_coupled_source`, `build_multi`, `add_native_block`) ; `PYBIND11_MODULE`
monolithique ~570 lignes ; `else` sous-indente (auto-fixable) ; include `adc` intercale dans la STL ;
egalites flottantes exactes de detection de defaut ; helpers `static` vs `namespace {}` ; suffixe `_`
de membres pimpl incoherent ; narrowing `size_t->int` ; `time_method` en `int` vs `enum class` ;
accesseurs non-const ; `(void)ncomp` vs `[[maybe_unused]]` ; `reinterpret_cast` dlsym (a confiner) ;
commentaire trompeur sur l'ordre d'inclusion.

### 4.12 tests-a (72 fichiers)

Qualite elevee et homogene : chaque test autonome (`int main` 72/72), device-clean, sans new/delete,
sans cast C, tests negatifs propres (catch type). Aucun UB declenche, scan printf propre.

Important :
- `test_condensed_schur_source_stepper.cpp:490,530` (ES.50/Type.3) : `const_cast<Setup&>(S)` retire
  la const d'un objet genuinement `const Setup S` pour le passer a un membre `Setup& S` de
  `RefIntegrator` qui ne fait que lire. Pas d'UB aujourd'hui, fragile. Declarer `const Setup& S`.

Cosmetique (7) : harnais de test triplique (lambda `chk`+`fails` redeclare, `raises`/`close_rel`/
`checksum` copies, aucun en-tete `test_support.hpp`) ; `reinterpret_cast` void*->T* au lieu de
`static_cast` dans 4 loaders natifs (auto-fixable) ; `std::system()` pour compiler un `.so` (4
fichiers, entrees build-controlees) ; `main()` 180-290 lignes ; casts `double()` (28, auto-fixable) ;
constantes ALL_CAPS `NC`/`KAPPA` ; compteur global mutable `static int fails`.

### 4.13 tests-b (86 fichiers)

Sain et conscient des contraintes device (helpers `ADC_HD`, foncteurs nommes). Includes coherents,
`comm_init` toujours apparie a `comm_finalize`, accesseurs de taille `int` (pas de bug de format).

Important :
- `test_geometry.cpp:12` (+ ~118 fichiers, DRY) : aucun framework de test ; chaque `.cpp`
  reimplemente le meme harnais (lambda `chk` x50+, compteur `fails`, variante inline). Un changement
  de format de rapport touche 86 fichiers. Extraire un `tests/test_harness.hpp`.
- `test_polar_condensed_schur_source_stepper.cpp:449,485` (ES.50/Type.3) : `const_cast<Setup&>` sur
  un `const Setup S` lu seulement (les solveurs polaires prennent deja des `const&`). Eliminable.

Cosmetique (5) : teardown Kokkos manuel `initialize/finalize` non exception-safe et incoherent avec
`ScopeGuard` ; casts C-style/fonction `(double)x`/`double(x)` (~50, auto-fixable) ; constante `kPi`
redupliquee (16 declarations + 28 litteraux) ; `main()` polaires 150-200 lignes ; `std::exit(2)` dans
un helper `load()` avec fuite de `FILE*`.

### 4.14 bench/CMake/scripts (12 fichiers)

Soigne : structure claire, `CMakeLists` racine moderne (cible INTERFACE `adc` isolee, FetchContent
Kokkos avec verif SHA256, `adc_dev_options` en PRIVATE), scripts `set -euo pipefail` + quoting. Outils
hors CI execute, pas l'API.

Important :
- `frontend_cpp.cpp:251` + sites bench (DRY) : `percentile`/`timed`/`PhaseTimers`/`eat` byte-identiques
  sur 4-5 fichiers, sans `bench/common.hpp`. Une divergence de fix (interpolation, semantique fence)
  casse la coherence des mesures.
- `profile_step.cpp:188-228` + `frontend_cpp.cpp` + `scaling_step.cpp` (device) : 3 benches ecrivent
  les `Array4` par boucles HOTE brutes sans `sync_host`/`sync_device`, alors que
  `profile_transport_mbox.cpp` montre le bon idiome (`for_each_cell`+`ADC_HD`+sync). Portables
  uniquement par memoire host-accessible (UVM) ; casseraient sur backend device non-UVM, et `bench/`
  n'etant que compile-teste l'ecart ne serait pas detecte.

Cosmetique (12) : `new`/`delete` proprietaire `fft_storage` ; `atoi`/`atof` sans rapport d'erreur ;
signatures 11-14 params avec out-params `double&` ; argument inconnu ignore silencieusement ;
`volatile Real s` comme barriere anti-elision (CP.200) ; casts `double()` (auto-fixable) ; `using
namespace adc;` global ; `namespace {}` vs `static` non uniforme ; standard C++ pose deux fois dans
`CMakeLists.txt` ; commentaire "cible 3.20" perime (3.21) ; `kPi` mort + `wall` non emis ; `printf`
~34 args hors couverture `-Wformat=2`.

## 5. Auto-corrigeable vs jugement

Auto-corrigeable (outillage, a router vers le milestone "Qualite de code & CI durcie"). Le depot a
deja `.clang-format` (`BasedOnStyle: Google`, `ReflowComments:false`, `SortIncludes:false`) et
`.clang-tidy` (familles larges, `modernize-use-nodiscard`/`magic-numbers`/`identifier-length`
desactives), tous deux INFORMATIFS (le job `format`/`tidy` ne fait que signaler) :
- `clang-format` reglerait : sous-indentation des `else` de `if constexpr`
  (`numerical_flux.hpp:207`, `implicit_stepper.hpp:243/306`, `system.cpp:605`), lignes de CODE >100
  col (`amr_dsl_block.hpp:615+`, `block_builder_polar.hpp:299`, `compiled_block_abi`).
- `clang-tidy google-readability-casting` : tous les `double(x)`/`(double)x`/`(int)b` (tests-a ~28,
  tests-b ~50, bench ~20, coeur, python). Distinguer de `Real(litteral)`, idiome conserve.
- Edits triviaux : `/** */` -> `///` (`physics/`), initialiseurs in-class manquants
  (`patch_box.hpp`, `Variable`/`VariableSet`, `BlockState`, `SourceNewtonReport`), `reg[] = {}`,
  retrait de `mutable solve_count_`, variable morte `nn`, `reinterpret_cast` void*->T* -> `static_cast`
  dans les loaders de test.

Ces corrections cosmetiques pourraient devenir bloquantes en CI une fois la convention actee dans
`CODING_STANDARDS_DECISIONS.md` (passer `format`/certaines familles `tidy` de informatif a `WarningsAsErrors`).

Jugement / refactor (ne pas confier a un outil) : les bloquants (`box_hash` int64, `field_postprocess`
foncteur), tous les DRY structurels (cascades de dispatch x3, stencils x3, `set_*` x6, marshaling x12,
jacobien Newton x2, macro-pas x4, harnais de test, `bench/common.hpp`), la regle des cinq
(`PolarTensorKrylovSolver`, `ScopedBlockState`), les gardes a durcir (`assert`->`throw`,
`cfg.n>=1`, garde de pile `CsProgram::eval`), le passage `long`->`int64_t` pour le port Windows, et
les regroupements de parametres en POD.

## 6. Suites a donner (priorisees)

Bloquants d'abord, puis dette structurelle a fort effet de levier, puis cosmetique outille. Ces
suites sont tracees dans le milestone *Revue & audit qualite du code* : ADC-209 (1), ADC-210 (2 et 3),
ADC-211 (4 et 6), ADC-212 (5), ADC-213 (7), ADC-214 (8), ADC-215 (9), ADC-216 (10), ADC-217 (11),
ADC-219 (12) ; le critere d'acceptation #5 (ecarts bloquants corriges ou traces) est ainsi clos.

1. `box_hash.hpp:71-73` : passer la cle et `bins_` a `std::int64_t`, caster `bx` avant le decalage.
   Gate du port Windows natif. A tracer en issue (lie a l'epic ADC-90).
2. `elliptic_problem.hpp:104` : extraire `FieldPostprocessKernel` (foncteur nomme device-clean).
   Risque segfault `nvcc` Release. A tracer en issue.
3. `amr_flux_helpers.hpp:85` et `refinement.hpp:179/209` + `multifab.hpp:137` : convertir les lambdas
   etendues residuelles en foncteurs nommes (meme classe de risque que 2). A tracer en issue.
4. `amr_system.cpp:852` (`cfg.n==0` modulo par zero) : valider `n>=1` au ctor de `System`/`AmrSystem`.
5. Regle des cinq : `= delete` ou implementer copie/move de `PolarTensorKrylovSolver`
   (`polar_tensor_operator.hpp:726`) et `ScopedBlockState` (`system_coupler.hpp:61`).
6. Durcir les gardes : `Ny%np` `assert`->`throw` (`poisson_fft_solver.hpp:114`), garde de pile
   `CsProgram::eval` (mode debug), `mf_find_box==-1` (`amr_subcycling.hpp:523`).
7. DRY a fort levier (chacun a tracer en issue) : cascade de dispatch x3 (runtime-1), stencil
   eps/cut-cell x3 et `set_*` x6 (numerique-1), jacobien Newton x2 (numerique-3), 4 macro-pas du
   stepper (runtime-3), marshaling x12 + Newton + `resolve_implicit_components` (bindings),
   `floor_div` x3 + `parallel_copy` (maillage).
8. Regrouper les longues listes de parametres en POD (`NewtonOptions`/`SourceStageOptions`) :
   `system.hpp:122`, `amr_system.hpp:232`, ABI `residual`/`advance`.
9. Harnais de test partage `tests/test_harness.hpp` (`chk`/`raises`/`close_rel`/`kPi`) et
   `bench/common.hpp` (`timed`/`percentile`/`PhaseTimers`/`eat`). A tracer en issue.
10. Portabilite `long`->`std::int64_t`/`std::size_t`/`std::ptrdiff_t` pour les tailles/offsets
    memoire (maillage, coeur). Gros perimetre, a tracer en issue dans l'epic Windows.
11. En-tetes Doxygen manquantes (`comm.hpp`, `load_balance.hpp`, `patch_box.hpp`, 13 fichiers
    numerique-3, 8 numerique-1, 2 numerique-2) et `CODE_DOCUMENTATION_CONVENTION.md` jamais
    commite, a committer pour reparer le lien depuis `CODEBASE_AUDIT.md` (suivi via ADC-125).
12. Passe outillee unique (section 5) une fois la convention actee, puis durcir le job `quality.yml`.

Note honnetete methodologique : 6 constats ont ete rejetes purement a la verification (un par
unite : coeur, couplage-1, couplage-2, runtime-1, runtime-3, bench), et une vingtaine de candidats
"important" ont ete retrogrades faute de trigger reel ou d'impact (notamment `string_view` membre
sans site declencheur, usage `long` borne par la box, `regrid_level` serie-only documente,
`cluster_rec`/`hqr_minmax` cohesifs et de portage). Le decompte de 180 constats reflete cet etat
apres verification croisee, pas la liste brute initiale.

# Conception : AMR multi-blocs sur une hierarchie partagee (capstone multi-especes)

DESIGN-ONLY. Aucune implementation. Ce document est une SPEC raisonnee, honnete sur ses
limites, de la migration de l'AMR `adc` d'UN seul bloc explicite vers PLUSIEURS blocs
(especes) sur la MEME hierarchie AMR partagee, conservativement couplee, calquee sur la
facon dont la grille mono-niveau porte deja plusieurs blocs. C'est le capstone : electrons,
ions et neutres sur les MEMES niveaux/patchs, un seul Poisson grossier dont le second membre
est la SOMME des contributions elliptiques co-localisees, des sources couplees lues cellule a
cellule, un regrid pilote par l'UNION des criteres.

Le modele cible est celui d'AMReX / FLASH / SAMRAI : UNE hierarchie commune portant plusieurs
champs, raffinement par l'union des criteres, jamais une hierarchie par espece.


## 0. Note d'honnetete liminaire (etat reel du code a ce head)

Le code a ete relu directement. Deux faits structurent toute la suite.

FAIT 1 : le MOTEUR multi-blocs AMR EXISTE DEJA, au niveau template C++, sous le nom
`AmrSystemCoupler` (`include/adc/coupling/amr_system_coupler.hpp`). Il porte :
- une hierarchie PARTAGEE par bloc (`std::vector<std::vector<AmrLevelMP>> block_levels_`,
  meme `BoxArray` par niveau verifie au ctor, lignes 118-128) ;
- un aux PARTAGE par niveau (`std::vector<MultiFab> aux_`, un par niveau, recable vers chaque
  bloc, lignes 136-141) ;
- un Poisson de SYSTEME a second membre SOMME (`rhs_assembler_(system_, mg_.rhs())` dans
  `solve_fields`, ligne 183 ; l'assembleur est `ChargeDensityRhs`, cf. ci-dessous) ;
- un schema spatial, des sous-pas et une cadence PAR BLOC (`block_time_treatment_v`,
  `block_substeps_v`, `block_stride_v`, `step`, lignes 211-250) ;
- l'IMEX / implicite par callback (`AmrImplicitSourceStepper`, lignes 361-369) ;
- des sources couplees NIVEAU PAR NIVEAU (`coupled_source_step`, lignes 266-283) ;
- un B_z partage par niveau (`fill_bz`, lignes 329-340).

Ce que l'owner appelle "extraire un moteur multi-blocs" est donc en grande partie un travail
de RECONNAISSANCE et de COMPLEMENT, pas de creation ex nihilo. C'est une bonne nouvelle a dire
franchement : la cible Phase 1 est plus proche qu'il n'y parait.

FAIT 2 : la FACADE RUNTIME (celle exposee a Python) reste MONO-BLOC. `AmrSystem`
(`include/adc/runtime/amr_system.hpp`, impl `python/amr_system.cpp`) REFUSE explicitement un
2e bloc :
```
// python/amr_system.cpp:129-130
if (p_->has_block || p_->has_compiled)
  throw std::runtime_error("AmrSystem : un seul bloc (AMR mono-modele)");
// idem set_compiled_block, ligne 152-153
```
Elle enveloppe un seul `AmrCouplerMP<Model>` (`include/adc/coupling/amr_coupler_mp.hpp`),
materialise par `detail::dispatch_amr_compiled` / `build_amr_compiled`
(`include/adc/runtime/amr_dsl_block.hpp`). Le `AmrSystemCoupler` multi-blocs n'est PAS branche
a cette facade.

Les DEUX vraies lacunes pour la cible Phase 1 sont donc :
1. la FACADE RUNTIME multi-blocs et son binding Python (exactement l'ecart deja franchi entre
   `SystemCoupler` compile-time et la facade runtime `System` : registre type-erase de blocs,
   `add_block` repete, pas multirate avec `stride` / `evolve` / `substeps`, somme du Poisson) ;
2. le REGRID multi-blocs : `AmrSystemCoupler` n'a AUCUNE methode `regrid` (verifie : grep
   `regrid` dans `amr_system_coupler.hpp` -> rien), contrairement a `AmrCouplerMP::regrid`
   (lignes 321-325) qui delegue a `amr_regrid_finest`. Sa hierarchie est FIGEE a la
   construction. Le regrid d'UNION des tags qui reconstruit la grille UNE fois et
   prolonge/restreint TOUS les blocs est a ECRIRE.

Tout le reste de ce document s'appuie sur ces noms reels.


## 1. Decision d'architecture et pourquoi

DECISION. Une seule hierarchie AMR PARTAGEE par toutes les especes. Chaque espece est un
`AmrBlock` : une pile de niveaux `std::vector<AmrLevelMP>` posee sur les MEMES `BoxArray` par
niveau et le MEME `DistributionMapping`, plus son modele, son schema spatial, sa politique
temporelle. TOUS les blocs vivent sur TOUS les patchs : un bloc n'est JAMAIS absent d'un
patch, meme si son critere ne l'a pas declenche, parce qu'il est couple aux autres. Tous
partagent le MEME aux par niveau (phi, grad phi, [B_z, T_e]) et le MEME Poisson grossier ; le
second membre elliptique est la SOMME des contributions par bloc, assemblees a partir de
champs CO-LOCALISES (memes cellules de la meme hierarchie). Le raffinement est l'UNION des
criteres par bloc (plus les tags utilisateur et les tags de phi).

POURQUOI une hierarchie partagee, conservativement couplee, et NON une par espece :

- POISSON. Le couplage electrostatique est GLOBAL : `lap phi = Sum_s q_s n_s`. Sur une
  hierarchie partagee, `solve_fields` lit toutes les densites au MEME index de cellule et
  ecrit un seul phi. C'est exactement ce que fait deja `AmrSystemCoupler::solve_fields`
  (lignes 177-206) : `rhs_assembler_(system_, mg_.rhs())` lit tous les blocs sur le grossier,
  puis une seule injection coarse->fine de l'aux. Des grilles par espece exigeraient
  d'interpoler chaque densite vers une grille commune A CHAQUE solve, et de redistribuer phi
  vers chaque grille : un assemblage a partir de grilles NON CONFORMES, ce que la cible
  proscrit.

- SOURCES COUPLEES. Ionisation / collision / echange thermique lisent les etats de PLUSIEURS
  especes dans la MEME cellule (cf. la formule cible `d_t n_e = +k n_e n_g`,
  `d_t n_i = +k n_e n_g`, `d_t n_g = -k n_e n_g`). Sur une hierarchie partagee, la lecture est
  LOCALE (meme `(i,j)`, meme fab), AUCUNE interpolation inter-especes. C'est le contrat de
  `AmrSystemCoupler::coupled_source_step` (lignes 266-283) : a chaque niveau k, chaque bloc est
  temporairement repointe vers son niveau k et la source lit tous les blocs + `aux_[k]`. Des
  hierarchies disjointes rendraient ce splitting impossible sans projection conservative.

- CONSERVATION ET REFLUX. Le reflux est BLOC PAR BLOC (chaque bloc a ses propres registres de
  flux, `FluxRegister` dans `amr_reflux_mf.hpp`). La source reste CELLULE-LOCALE, appliquee
  apres le transport (`mf_apply_source_treatment`, lignes 75-82), JAMAIS dans les registres de
  reflux : c'est exactement la propriete du Gap2 IMEX-on-AMR deja merge. Une source a
  contributions exactement opposees dans la MEME cellule conserve la masse de paire a la
  precision machine, ce qui ne tient que si les especes partagent la cellule.

- MPI / KOKKOS. Une seule hierarchie = un seul plan de distribution
  (`DistributionMapping(ba.size(), n_ranks())`), un seul jeu de halos, une seule reduction de
  reflux. Des hierarchies disjointes multiplieraient les plans et les collectives.

CE QU'ON N'OUVRE PAS (et pourquoi le dire). Pas de hierarchie par espece. Pas de niveaux
differents par espece (invariant verifie au ctor d'`AmrSystemCoupler`, lignes 118-128 : meme
nombre de niveaux, meme `BoxArray` par niveau). Pas d'allocation partielle "patch actif pour
une seule espece". Ces restrictions sont volontaires : elles preservent l'aux unique et le
Poisson unique, raison d'etre du partage, et la conservation cellule a cellule des sources.


## 2. Le moteur multi-blocs : API concrete (ce qui existe, ce qui se nomme)

L'engine multi-blocs s'articule autour des types CONCRETS suivants. Quand ils existent deja,
on les cite par leur nom de fichier reel ; quand ils sont a extraire, on indique le code dont
ils promeuvent un role.

### 2.1 La couche de blocs compile-time (existe)

- `EquationBlock<Model, Spatial, Time>` (`include/adc/core/equation_block.hpp`) : porte
  `Model`, `Spatial` (limiter + flux), `Time` (politique), `MultiFab* state`, `BCRec bc`.
- `CoupledSystem<Blocks...>` (`include/adc/core/coupled_system.hpp`) : tuple de blocs avec
  `n_blocks`, `block<I>()`, `for_each_block(f)`. C'est le "registre" compile-time.
- `AmrLevelMP { MultiFab U; const MultiFab* aux; Real dx, dy; }`
  (`amr_reflux_mf.hpp`, lignes 791-795) : un niveau de la hierarchie multi-patch.
- `LevelHierarchy { std::vector<AmrLevelMP> levels; Box2D base_dom; Periodicity base_per;
  bool coarse_replicated; bool recon_prim; bool imex; }` (`amr_reflux_mf.hpp`, lignes
  1012-1019) : la hierarchie comme OBJET nomme. C'est le grain du futur `AmrHierarchyLayout`
  (cf. 2.2).
- `AmrLevelStack<Level>` (`include/adc/coupling/amr_level_storage.hpp`) : detient la pile de
  niveaux ET la pile d'aux parallele, avec l'INVARIANT D'ADRESSES (aux dimensionne une fois,
  `reattach_aux(k)` remplace en place). C'est la brique de stockage qui rend le recablage
  `levels[k].aux = &aux_[k]` sur.

### 2.2 `AmrRuntime` : l'engine multi-blocs (a extraire, partiellement present)

L'engine cible regroupe sept roles. Cinq sont DEJA portes par `AmrSystemCoupler` ; deux sont a
ajouter. On donne des signatures C++ qui collent aux types existants.

- `AmrHierarchyLayout` : per-level `BoxArray` + `DistributionMapping` + `dx` + ratio de
  raffinement (=2 ici, `SubcyclingSchedule(2)`). Aujourd'hui implicite : chaque `AmrLevelMP`
  porte son `U.box_array()` / `U.dmap()` / `dx,dy`, et le ctor d'`AmrSystemCoupler` IMPOSE que
  tous les blocs aient le meme `BoxArray` par niveau. A PROMOUVOIR en type explicite, source
  unique de verite sur la grille, partage par tous les blocs :
  ```cpp
  struct AmrHierarchyLayout {
    std::vector<BoxArray>            ba;     // [niveau]
    std::vector<DistributionMapping> dm;     // [niveau], parallele a ba
    std::vector<Real>                dx, dy; // [niveau] = dx_coarse / 2^k
    Box2D base_dom; Periodicity base_per{true,true};
    bool coarse_replicated = true; int ref_ratio = 2;
  };
  ```

- `AmrBlock` : la pile de niveaux d'UN bloc sur le layout partage, plus son identite
  numerique. Promotion de la paire `(block_levels_[b], system_.block<b>())` :
  ```cpp
  template <class Model, class Spatial, class Time>
  struct AmrBlock {
    std::string_view name;
    Model model; BCRec bc;
    std::vector<AmrLevelMP> levels;   // U par niveau sur AmrHierarchyLayout
    VariableSet cons_vars, prim_vars; // noms + ROLES (cf. variables.hpp)
    bool recon_prim = false;
    // Spatial = {Limiter, NumericalFlux} ; Time = politique (treatment/substeps/stride/evolve)
  };
  ```
  Invariant : `levels[k].U.box_array() == layout.ba[k]` pour tout k (deja verifie au ctor).

- `AmrBlockRegistry` : la collection de blocs. En compile-time c'est `CoupledSystem<Blocks...>`
  (deja la). En runtime ce sera un `std::vector<AmrSpecies>` type-erase calque sur
  `System::Impl::sp` (`python/system.cpp:270-301`, la struct `Species`).

- `AmrFieldSolver` : le Poisson de SYSTEME a second membre SOMME, co-localise. C'est
  `solve_fields` d'`AmrSystemCoupler` (lignes 177-206) cable sur `ChargeDensityRhs`
  (`include/adc/coupling/elliptic_rhs.hpp`, lignes 117-133), qui EXIGE une `SpeciesCharge` par
  bloc (`species.size() != System::n_blocks` -> erreur), somme `q_s n_s` via
  `add_scaled_component`, puis injecte l'aux coarse->fine (`coupler_inject_aux_mb`) une seule
  fois. Co-localise par construction : `rhs` est sur `block_levels_[0][k].U.box_array()`, la
  grille partagee.

- `AmrScheduler` : honore `treatment` / `substeps` / `stride` / `evolve` par bloc. C'est
  `AmrSystemCoupler::step` (lignes 211-250) plus la semantique stride de
  `advance_subcycled` (`include/adc/numerics/time/scheduler.hpp`) et de `System` runtime
  (`stride_due`, `python/system.cpp:327`). Le contrat cible (cf. 4.iv) :
  - `Explicit` -> transport AMR par `advance_amr<Limiter,NumericalFlux>` ;
  - `IMEX` -> transport explicite (`SourceFreeModel<Model>`) + source implicite par le
    callback (reutilise `mf_apply_source_treatment` / `backward_euler_source` du Gap2) ;
  - `Implicit` -> REJET tant qu'un vrai stepper global n'existe pas ;
  - `evolve=false` -> bloc GELE (non avance, ne contraint pas la CFL) mais TOUJOURS lu par le
    second membre du Poisson comme fond fixe ;
  - `substeps=N` -> N sous-pas dans le pas du bloc ; `stride=M` -> tenu M-1 macro-pas puis
    rattrapage d'un pas effectif M*dt.

- `AmrCouplingRegistry` : les sources couplees inter-especes appliquees niveau par niveau,
  cellule par cellule, contributions exactement opposees. C'est `coupled_source_step`
  (lignes 266-283) cable sur une `CoupledSource` (concept `CoupledSourceFor`,
  `include/adc/coupling/coupled_source.hpp`). Le kernel device de production est
  `CoupledSourceKernel` (`include/adc/coupling/coupled_source_program.hpp`, P5 #131) : POD
  capture par valeur, ecritures ADDITIVES `out[t](i,j,c) += dt*S_t` ; deux termes opposes
  (`+k n_e n_g` sur ion, `-k n_e n_g` sur neutre) y conservent la masse de paire exactement.

- `AmrRegridPolicy` : l'union des tags + rebuild + prolong/restrict de TOUS les blocs. C'est la
  PIECE MANQUANTE (cf. 5). La brique mono-bloc existe : `amr_regrid_finest`
  (`include/adc/coupling/amr_regrid_coupler.hpp`) ; il faut un orchestrateur multi-blocs qui
  tague l'UNION puis reconstruit la grille UNE fois pour tous.

### 2.3 Pourquoi un engine et non un n-ieme coupleur

`AmrSystemCoupler` melange deja "assemble" (Poisson + aux) et "avance" (step + reflux), comme
le note son propre commentaire (lignes 371-375 : alias `AmrSystemDriver`). La scission
Assembler/Driver est faite cote mono-niveau (`SystemAssembler` / `SystemDriver`,
`include/adc/coupling/system_coupler.hpp`). L'engine `AmrRuntime` formalise la meme separation
cote AMR, mais c'est un raffinement COSMETIQUE et reporte : la classe unifiee est deja validee.
La priorite reste la FACADE RUNTIME et le REGRID, pas la jolie scission.


## 3. Strategie COMPAT-FACADE : preserver le mono-bloc BIT-IDENTIQUE

L'objectif est que l'`AmrSystem` mono-bloc actuel et ses tests restent BIT-IDENTIQUES. La
regle est de NE PAS bricoler la classe mono-bloc en place, mais de la transformer en FACADE.

PLAN. `AmrSystem` (runtime) devient une facade au-dessus d'un engine multi-blocs runtime
`AmrRuntime` ; le chemin mono-bloc passe par un `AmrSingleBlockSystem` qui n'installe qu'UN
bloc. Concretement :

1. L'`AmrSystem::Impl` actuel construit un `AmrCouplerMP<Model>` unique via
   `build_amr_compiled`. Le chemin mono-bloc DOIT rester ce chemin tant que le registre ne
   contient qu'un bloc : `AmrCouplerMP<Model>` et `AmrSystemCoupler<CoupledSystem<Block>, ...>`
   ne sont PAS le meme code. Pour garantir le bit-identique, la facade route :
   - `n_blocks == 1` (et pas de couplage inter-especes, pas de bloc gele) -> chemin
     `AmrCouplerMP<Model>` ACTUEL, INCHANGE -> tests mono-bloc bit-identiques par construction
     (aucune ligne du chemin n'est touchee) ;
   - `n_blocks >= 2` OU couplage demande -> nouveau chemin `AmrSystemCoupler`.
2. Ce routage interne ne change PAS la signature publique d'`AmrSystem`. `add_block` cesse de
   throw au 2e appel ; il enregistre une spec de bloc supplementaire (comme
   `System::add_block` empile une `Species`). Le 1er `add_block` seul, suivi de `step` /
   `step_cfl` / `mass` / `density` / `potential`, doit produire EXACTEMENT les memes octets
   qu'aujourd'hui.
3. Le critere de bit-identite est verifie par un test de NON-REGRESSION : un cas mono-bloc
   (un des `test_amr_*` ou `test_dsl_production_amr.py`) doit donner `maxdiff == 0` entre la
   facade refactorisee et le binaire actuel. Tant que ce test n'est pas vert, la PR (i) ne
   merge pas.

POURQUOI ne pas faire passer le mono-bloc par `AmrSystemCoupler` tout de suite. Parce que
`AmrSystemCoupler` et `AmrCouplerMP` different sur des details qui CASSENT le bit-identique :
- `AmrCouplerMP::compute_aux` ecrit grad phi inline (lignes 283-293) et appelle
  `coupler_eval_rhs` (`f = model.elliptic_rhs(U)`), alors qu'`AmrSystemCoupler::solve_fields`
  passe par `field_postprocess` + `ChargeDensityRhs` (somme). Les deux sont mathematiquement le
  meme calcul a une espece de charge `q=1`, mais l'ordre des operations flottantes differe.
- `AmrCouplerMP` porte le regrid periodique (via la fermeture `h.step`,
  `amr_dsl_block.hpp:99-104`) ; `AmrSystemCoupler` n'en a pas.

Donc : le mono-bloc reste sur `AmrCouplerMP` (intouche), et `AmrSystemCoupler` recoit le
regrid + le routage facade. Quand `AmrSystemCoupler` aura prouve un mono-bloc strictement
bit-identique a `AmrCouplerMP` (test maxdiff=0), on POURRA fusionner les deux chemins ; jusque
la, ils coexistent. C'est exactement la prudence du commentaire `replicated_coarse`
d'`AmrCouplerMP` (lignes 234-246) : la suppression d'un chemin est reportee tant que l'autre
n'est pas strictement superieur.


## 4. Migration Phase 1 : decoupage en PR ordonnees

Chaque PR liste son WRITE-SET (fichiers), son test d'acceptation, et son verrou de
bit-identite / conservation. L'ordre est strict : chaque PR laisse l'arbre vert.

### PR (i) -- Introduire `AmrBlock` + registre, AUCUN changement de physique
WRITE-SET :
- `include/adc/coupling/amr_system_coupler.hpp` : extraire `AmrHierarchyLayout` (promotion des
  `BoxArray`/`DistributionMapping`/`dx` deja imposes identiques), faire porter a chaque bloc un
  `AmrBlock` (nom + levels + cons/prim VariableSet). Pas de nouveau comportement.
- `include/adc/coupling/amr_level_storage.hpp` : reutilise tel quel (invariant d'adresses).
ACCEPTATION : un cas 1 bloc explicite (`test_amr_compiled_model.cpp` style) tourne identique.
BIT-IDENTITE : le `step` ne change pas de corps -> `maxdiff == 0` vs head actuel sur ce cas.

### PR (ii) -- Deux blocs explicites, schemas DIFFERENTS, sans source couplee
WRITE-SET :
- `include/adc/coupling/amr_system_coupler.hpp` : deja N-blocs ; ajouter un test instanciant
  `CoupledSystem<BlockA, BlockB>` ou `BlockA::Spatial != BlockB::Spatial` (p.ex. Minmod/Rusanov
  vs VanLeer/HLLC) sur la MEME hierarchie 2 niveaux.
- `tests/CMakeLists.txt` + `tests/test_amr_system_twoblock.cpp` (nouveau test).
ACCEPTATION : deux blocs AMR a schemas differents, stables sur N pas ; masse de chaque bloc
conservee au reflux (`AmrSystemCoupler::mass(b)`, lignes 287-297).
CONSERVATION : `mass(0)` et `mass(1)` constants a la tolerance de reflux (reflux bloc par bloc,
chaque bloc a ses `FluxRegister`).

### PR (iii) -- Poisson de SYSTEME a second membre SOMME (co-localise)
WRITE-SET :
- `include/adc/coupling/elliptic_rhs.hpp` : `ChargeDensityRhs` (deja la) ; verifier la garde
  `species.size() == n_blocks`.
- test : deux especes de charges opposees, `lap phi = q_i n_i + q_e n_e`, compare a un Poisson
  mono-niveau assemble main.
ACCEPTATION : `solve_fields` lit les deux blocs co-localises et resout UN phi. La somme du RHS
egale (a la precision machine) la somme assemblee separement -> co-localisation prouvee.
CONSERVATION : la charge totale integree sur le grossier reste la somme attendue.

### PR (iv) -- substeps / stride / evolve + step_cfl substeps-aware
WRITE-SET :
- `include/adc/coupling/amr_system_coupler.hpp` : ajouter `step_cfl(cfl)` calque sur
  `System::step_cfl` (`python/system.cpp:1663-1693`), substeps-aware :
  `dt <= cfl * h * substeps_b / (stride_b * w_b)`, min sur les blocs evolutifs ; un bloc
  `evolve=false` NE contraint PAS le pas mais reste dans le RHS Poisson.
- la semantique stride DOIT etre celle de `System` (HOLD-THEN-CATCH-UP, `stride_due(macro,M) =
  (macro+1)%M==0`, `python/system.cpp:320-327`), PAS celle de `advance_subcycled`
  (`macro%M==0`, debut de fenetre). Aujourd'hui `AmrSystemCoupler::step` utilise
  `macro_step_ % stride != 0` (ligne 225) : a HARMONISER sur `stride_due` pour la coherence
  temporelle du couplage (sinon un bloc lent est "dans le futur" au 1er pas, couplage faux).
ACCEPTATION : electrons IMEX(substeps=10) + ions Explicit(substeps=1) stable, ET l'inverse ;
neutres stride=20 toujours lus par la source et le Poisson entre deux rattrapages ;
`evolve=False` present dans le RHS elliptique comme fond fixe.
VERROU : `step_cfl` doit respecter A LA FOIS stride et substeps ; un dt calcule sur `w_max`
seul puis multiplie par M violerait la CFL d'un facteur M (note explicite de `System::step_cfl`).

### PR (v) -- DSL production multi-bloc (INSTALL d'un bloc NOMME)
WRITE-SET :
- `include/adc/runtime/amr_dsl_block.hpp` : `add_compiled_model(AmrSystem&, name, Model{}, ...)`
  doit INSTALLER UN BLOC NOMME (et non remplacer le bloc unique) -> symetrique de
  `add_compiled_model(System&)` (`include/adc/runtime/dsl_block.hpp` + `block_builder.hpp`).
- `include/adc/runtime/amr_system.hpp` + `python/amr_system.cpp` : `set_compiled_block` /
  `add_native_block` cessent de throw au 2e appel (cf. 3) et empilent une spec.
- `python/bindings.cpp` : exposer le 2e `add_block` (deja branche, `bindings.cpp:239`), valider
  `dsl.Model(...).compile(target="amr_system", backend="production")` pour DEUX blocs.
ACCEPTATION : `test_dsl_production_amr.py` etendu a deux blocs compiles natifs sur la meme
hierarchie ; ABI key verifiee (`adc_native_abi_key`, `amr_system.cpp:218-235`).
VERROU : foncteurs NOMMES device-clean (cf. `BlockRhsEval`/`AdvanceExplicit` de
`block_builder.hpp` et `ForEachBlockProbe` de `coupled_system.hpp:59-63`) ; aucune lambda
etendue cross-TU (recette #64/#97).

### PR (vi) -- Sources couplees sur AMR (meme cellule, contributions opposees)
WRITE-SET :
- `include/adc/coupling/amr_system_coupler.hpp` : `coupled_source_step` (deja la) cable sur les
  couplages nommes (ionisation/collision/echange) ET sur `CoupledSourceKernel`
  (`coupled_source_program.hpp`).
- `python/amr_system.cpp` + `bindings.cpp` : `sim.add_coupling(adc.Ionization(...))` calque sur
  `System::add_ionization` / `add_coupled_source` (`include/adc/runtime/system.hpp:266-329`).
ACCEPTATION : ionisation `+k n_e n_g` / `-k n_e n_g` sur 3 blocs co-localises, niveau par
niveau, sur tous les patchs.
CONSERVATION : invariant de creation de paire et de masse lourde a la PRECISION MACHINE
(`n_i + n_g` constant) ; verifie sur le grossier ET sur chaque patch fin (memes cellules).

### PR (vii) -- IMEX local sur AMR
WRITE-SET :
- `include/adc/coupling/amr_system_coupler.hpp` : le callback IMEX reutilise
  `mf_apply_source_treatment(m, U, aux, dt, /*imex=*/true)` (`amr_reflux_mf.hpp:75-82`) ->
  `backward_euler_source` (foncteur device nomme `BackwardEulerSourceKernel`). La source reste
  cellule-locale, snapshotee, HORS reflux.
ACCEPTATION : un bloc IMEX raide sur AMR stable la ou l'explicite diverge ; masse conservee.
CONSERVATION : le split implicite ne touche pas les registres de reflux (source hors flux) ->
conservation aux interfaces coarse-fine intacte (propriete Gap2).

### PR (viii) -- SEULEMENT ensuite : Schur / vrai implicite global / repro papier
WRITE-SET : hors scope Phase 1. S'appuie sur `CondensedSchurSourceStepper`
(`include/adc/coupling/condensed_schur_source_stepper.hpp`) et `schur_condensation.hpp`. Le
`treatment == Implicit` reste REJETE par l'`AmrScheduler` jusqu'a ce qu'un vrai stepper global
existe.


## 5. Algorithme de regrid (union des tags, rebuild une fois, prolong/restrict de TOUS les blocs)

C'est la piece manquante centrale. La brique mono-bloc est `amr_regrid_finest`
(`include/adc/coupling/amr_regrid_coupler.hpp`) : tag du parent -> `grow_tags` -> `all_reduce_or`
si grossier reparti -> `berger_rigoutsos` -> clamp nesting -> nouveau `BoxArray` fin ->
report des donnees fines + interp parent -> realloc aux. Elle opere sur UN bloc.

ALGORITHME MULTI-BLOCS (a ecrire, `AmrRegridPolicy` / `AmrSystemCoupler::regrid`) :

1. `solve_fields()` une fois (aux a jour, pour le critere de gradient de phi).
2. UNION DES TAGS sur le niveau parent : pour chaque bloc, `tag_cells(block.levels[pk].U,
   pdom, crit_block)` (`include/adc/amr/regrid.hpp:30-42`), puis OU logique des `TagBox` :
   ```
   tags = tags_electrons OR tags_ions OR tags_neutrals OR tags_phi OR tags_user
   ```
   `tags_phi` se tague sur `aux_[pk]` (composante 0 = phi, ou son gradient). `TagBox` est une
   grille de char ; l'OU est un `|=` cellule a cellule. `grow_tags` ensuite (nesting + marge).
3. Si grossier reparti : `all_reduce_or_inplace` sur les tags UNIS (meme garde MPI-safe que
   `amr_regrid_finest:59-60`) pour que tous les rangs partent de la MEME grille de tags ->
   `berger_rigoutsos` produit des patchs IDENTIQUES par rang (sinon les dmaps divergent).
4. REBUILD DU LAYOUT UNE SEULE FOIS : `berger_rigoutsos(grown)` -> clamp nesting -> nouveau
   `BoxArray` fin + un seul `DistributionMapping(nfine, n_ranks())`. Ce layout est PARTAGE par
   tous les blocs (c'est la regle d'or : un rebuild, pas un par espece).
5. PROLONG / RESTRICT COHERENT DE TOUS LES BLOCS sur ce nouveau layout : pour chaque bloc,
   reconstruire `levels[fk].U` sur le nouveau `BoxArray` (largeur de ghost heritee de
   `n_grow()`, cf. `amr_regrid_finest:73`), remplir par REPORT des donnees fines la ou
   l'ancien patch couvre + INTERP depuis le parent ailleurs (exactement le corps de
   `amr_regrid_finest:94-121`). TOUS les blocs utilisent le MEME `BoxArray`/`dmap` : aucun bloc
   absent d'un patch.
6. REBUILD AUX / PHI / RHS : realloc l'aux PARTAGE par niveau sur le nouveau layout
   (`AmrLevelStack::reattach_aux` preserve l'adresse), recabler `levels[k].aux = &aux_[k]`,
   re-poser B_z (`fill_bz`), re-`solve_fields`.
7. VERIFICATION PAR BLOC : la masse de chaque bloc (composante 0 integree) doit etre conservee
   par le regrid (le report + interp conservatif redistribue sans creer/detruire). Test :
   `mass(b)` avant == `mass(b)` apres regrid, a la tolerance.

DIFFICULTE REELLE FLAGGEE. `amr_regrid_finest` est une free function qui ne regrid QUE le
niveau le plus fin (`L.back()`), pas une hierarchie a N niveaux arbitraire. Pour Phase 1
(grossier + 1 fin, ce que materialise `build_amr_compiled:71-78`) c'est suffisant. Au-dela de 2
niveaux, le regrid multi-niveaux n'existe pas encore meme en mono-bloc : a noter comme limite,
pas a resoudre ici.


## 6. Registre des risques (avec de-risking)

- MIXTE stride/substeps a travers les blocs dans `step_cfl`. RISQUE : un dt calcule sur `w_max`
  seul, multiplie par stride, viole la CFL d'un facteur M. DE-RISK : copier LITTERALEMENT la
  formule par bloc de `System::step_cfl` (`python/system.cpp:1667-1680`),
  `dt = min_b cfl*h*substeps_b/(stride_b*w_b)`, blocs geles exclus. Test : un bloc stride=4
  reste sous la CFL effective.

- PRECISION ELLIPTIQUE MULTI-NIVEAUX. RISQUE : le Poisson de Phase 1 est resolu sur le GROSSIER
  puis injecte coarse->fine (`coupler_inject_aux_mb`), ce n'est PAS un vrai solve multi-niveaux.
  DE-RISK : Phase 1 assume "coarse + inject" (l'instabilite diocotron observable vit sur un
  cercle median resolu par le grossier, cf. `AmrSystem::potential()`,
  `amr_system.hpp:175-179`). Le vrai multi-niveaux (composite solve) est Phase 2, flagge comme
  tel ; ne pas le promettre en Phase 1.

- COHERENCE DU REGRID + CONSERVATION. RISQUE : reconstruire les blocs sur des layouts
  legerement differents (ordre des boites, dmap) casse la co-localisation et la conservation.
  DE-RISK : UN SEUL `BoxArray`/`DistributionMapping` calcule (etape 4), reutilise tel quel par
  tous les blocs ; test `mass(b)` invariant au regrid, par bloc.

- SOURCE COUPLEE : contributions exactement opposees, meme cellule. RISQUE : un decalage
  d'index ou une lecture d'un etat deja modifie casse la conservation de paire. DE-RISK :
  `CoupledSourceKernel` (`coupled_source_program.hpp:98-108`) evalue TOUS les termes sur l'etat
  GELE au debut du pas (`reg` fige) avant d'ecrire ; l'ordre des `.add` n'importe pas au 1er
  ordre. Sur hierarchie partagee, aucune interpolation inter-especes (meme `(i,j)`). Test :
  `n_i + n_g` constant a la precision machine, par niveau.

- INVARIANCE PAR RANG MPI. RISQUE : `fab(0)` sans garde `local_size()`, ou un grossier reparti
  mal reduit. DE-RISK : iteration sur `local_size()` partout (cf. `mass`,
  `amr_system_coupler.hpp:291`) ; `DistributionMapping(ba.size(), n_ranks())` ;
  `all_reduce_or_inplace` sur les tags unis avant clustering (regrid reparti) ;
  `FluxRegister::gather` = `all_reduce_sum_inplace` (identite en serie -> bit-identique np=1).
  Test : np=1/2/4 bit-identiques (calque de `test_mpi_amr_multipatch`).

- CONFORMITE nvcc DES FONCTEURS NOMMES. RISQUE : une lambda generique en contexte non evalue
  (concept) ou une lambda etendue cross-TU fait buter nvcc (Heisenbug device). DE-RISK : la
  recette est deja appliquee (`ForEachBlockProbe`, `coupled_system.hpp:59-63` ;
  `BlockRhsEval`/`AdvanceExplicit`/`MaxSpeed`/`PoissonRhs`, `block_builder.hpp` ;
  `CoupledSourceKernel`, `BackwardEulerSourceKernel`). Toute nouvelle premiere-instanciation
  depuis une TU externe passe par un FONCTEUR NOMME, jamais une lambda (recette #64/#97).

- PROPRETE DEVICE GH200. RISQUE (lecon recente) : un accesseur `Geometry`/`Box2D` NON `ADC_HD`
  utilise dans un kernel device casse silencieusement la numerique device. DE-RISK : tout
  nouvel accesseur lu dans un kernel device DOIT etre `ADC_HD` (cf.
  `for_each_cell_reduce_sum` dans `mass`, `amr_system_coupler.hpp:293-294`). Valider une fois
  un cas multi-bloc production sur GH200 (calque de `gpu_amrsys_facade_validate.cpp`, qui
  instancie deja `CoupledSystem` 2 blocs + AMR 2 niveaux + Poisson + step).


## 7. Frontiere Phase 2 / Phase 3 (verrouillee)

PHASE 1 (la cible v1) : une hierarchie commune ; N `AmrBlock` chacun sur TOUS les patchs avec
son modele / schema / temps / substeps / stride / evolve ; Poisson somme co-localise ; sources
couplees meme-cellule a contributions opposees ; regrid par union des tags (incl. tags_phi)
avec prolong/restrict coherent de TOUS les blocs ; reflux / conservation bloc par bloc ;
substeps / stride / evolve honores par le scheduler ET par `step_cfl`.

PHASE 2 (plus tard, RESTE CONSERVATIF) : criteres de raffinement PAR BLOC (l'union reste, mais
chaque bloc declare son critere) ; poids de cout PAR BLOC pour l'equilibrage de charge ;
exploitation d'evolve/stride/substeps pour un saut TEMPOREL UNIQUEMENT, c'est-a-dire un saut
GLOBAL A UN BLOC dans le temps (stride / evolve=false au niveau du bloc), JAMAIS une absence
spatiale locale d'un bloc sur un patch. EXPLICITEMENT : ne PAS sauter l'avance d'un bloc sur
certains patchs ; un bloc est toujours present et conservatif partout. Vrai solve elliptique
multi-niveaux (composite) ici.

PHASE 3 (seulement si un besoin scientifique reel emerge) : hierarchies DISTINCTES par espece
AVEC projections conservatives entre hierarchies. Beaucoup plus dur, explicitement PAS le
premier objectif.

INVARIANT TRANSVERSE : toute optimisation future reste CONSERVATIVE et TEMPORELLE ; jamais une
absence spatiale locale d'un bloc sur un patch.


## 8. Checklist de tests d'acceptation (Phase 1)

- [ ] Deux blocs AMR explicites a schemas DIFFERENTS, stables sur N pas.
- [ ] electrons IMEX(substeps=10) + ions Explicit(substeps=1), stable ; ET l'inverse.
- [ ] neutres stride=20 toujours lus par les sources et le Poisson entre rattrapages.
- [ ] `evolve=False` present comme fond fixe dans le second membre elliptique.
- [ ] regrid conserve la masse de CHAQUE bloc (`mass(b)` avant == apres, par bloc).
- [ ] `dsl.Model(...).compile(target="amr_system", backend="production")` pour DEUX blocs
      (bloc NOMME installe, pas remplace), ABI key verifiee.
- [ ] source couplee `+k n_e n_g` / `-k n_e n_g` : `n_i + n_g` constant a la precision machine,
      par niveau, sur tous les patchs.
- [ ] MPI np=1/2/4 bit-identiques (reflux par bloc + tags unis reduits).
- [ ] Kokkos Serial vert.
- [ ] un cas multi-bloc production valide sur GH200 (instanciation device complete).
- [ ] NON-REGRESSION : un cas mono-bloc reste BIT-IDENTIQUE (maxdiff=0) a travers la facade.


## 9. References de code (toutes verifiees a ce head)

- `include/adc/coupling/amr_system_coupler.hpp` : engine multi-blocs AMR (existe) ; PAS de
  regrid.
- `include/adc/coupling/amr_coupler_mp.hpp` : coupleur AMR MONO-BLOC + `regrid` (delegue a
  `amr_regrid_finest`).
- `include/adc/coupling/amr_regrid_coupler.hpp` : `amr_regrid_finest` (Berger-Rigoutsos, niveau
  le plus fin).
- `include/adc/coupling/elliptic_rhs.hpp` : `ChargeDensityRhs` (somme `Sum_s q_s n_s`,
  une `SpeciesCharge` par bloc).
- `include/adc/coupling/coupled_source.hpp` : concept `CoupledSourceFor`, `NoCoupledSource`.
- `include/adc/coupling/coupled_source_program.hpp` : `CoupledSourceKernel` (P5 #131, POD
  device-clean, ecritures additives opposees).
- `include/adc/coupling/amr_level_storage.hpp` : `AmrLevelStack` (invariant d'adresses aux).
- `include/adc/core/coupled_system.hpp` : `CoupledSystem<Blocks...>`, `for_each_block`,
  `ForEachBlockProbe` (foncteur nomme device-clean).
- `include/adc/core/equation_block.hpp` : `EquationBlock<Model, Spatial, Time>`.
- `include/adc/numerics/time/amr_reflux_mf.hpp` : `AmrLevelMP`, `LevelHierarchy`, `advance_amr`,
  `FluxRegister`, `CoverageMask`, `CoarseFineInterface`, `mf_apply_source_treatment`,
  `mf_average_down_mb`.
- `include/adc/numerics/time/scheduler.hpp` : `advance_subcycled`, `block_substeps_v`,
  `block_stride_v`, `block_time_treatment_v`.
- `include/adc/coupling/system_coupler.hpp` : `SystemAssembler` / `SystemDriver` (scission
  mono-niveau), `step_cfl` substeps-aware (`cfl*h*substeps/(stride*w)`).
- `include/adc/runtime/system.hpp` + `python/system.cpp` : facade RUNTIME multi-blocs (modele a
  imiter), `Species`, `stride_due` (HOLD-THEN-CATCH-UP), `step_cfl` (lignes 1663-1693).
- `include/adc/runtime/amr_system.hpp` + `python/amr_system.cpp` : facade RUNTIME AMR MONO-BLOC
  (refus du 2e bloc, lignes 129-130 et 152-153).
- `include/adc/runtime/amr_dsl_block.hpp` : `add_compiled_model(AmrSystem&)` (un seul bloc).
- `include/adc/runtime/block_builder.hpp` : `make_block` / `make_max_speed` /
  `make_poisson_rhs`, foncteurs nommes device-clean.
- `docs/COUPLER_HIERARCHY.md`, `docs/SCHUR_CONDENSATION_DESIGN.md`, `docs/GPU_RUNTIME_PORT.md`
  (recette device-clean), `docs/PAPER_ROADMAP.md`.

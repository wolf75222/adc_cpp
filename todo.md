# TODO - adc_cpp

> Liste de travail vivante. Synthese de (1) l'objectif initial du chantier (canal `aux` extensible
> + parite AMR + cablage runtime / Python / DSL), (2) ce que `docs/archive/ROADMAP.md` (ARCHIVE) marque "en file",
> (3) ce que les agents ont explicitement note comme "reste a faire".
> Convention : `[x]` fait et sur `master`, `[~]` partiel, `[ ]` a faire.

## ETAT COURANT (session juin 2026, master = #154)

Synthese a date apres la pile Schur + pipeline System + polaire + revue adversariale + run nuit. Detail
dans les sections 10 a 15 (nouvelles).

**Merge run nuit (#150-#166) :** #150/#152/#158 fix device TEST-SIDE (foncteurs hote dans kernels) ;
#151 extraction NativeLoader de system.cpp ; #155 CI Kokkos OpenMP (91/91) ; #156 surface couplage ;
#157 validation MPI+Kokkos Cuda multi-GPU rank-invariant ; #159 neutraliser mentions diocotron ;
#160/#161/#162 corrections doc (BACKEND_COVERAGE 7/7, ARCHITECTURE A.2, findings A2/A3/A4) ;
#163 concepts C++20 elliptiques (EllipticOperator/LinearSolver/FieldPostProcessor) ; #164 supprimer
amr_coupler.hpp deprecated ; #165 harnais profiling (Poisson MG domine 96-99.9%) ; #166 refresh todo.

**Merge sessions precedentes :** #130 Poisson polaire (2a) ; #131 source couplee DSL (P5) ; #132 IMEX
AMR (Gap 2) ; #133 foncteurs nvcc ; #134 krylov CL homogenes ; #135 fix device Geometry/Box2D+CFL ;
#136 CI split fast/full ; #137 honnetete API ; #138 step_cfl substeps-aware ; #139 doc AMR multi-blocs ;
#141 garde-fou layout AMR ; #154 capstone PR1 (facade runtime multi-blocs AmrSystem, hierarchie partagee).

**Merge session courante (post-coupure agents nuit) :** #167 A2 (helper `add_pair` conservatif DSL +
verif opt-in) ; #170 C.1 (decoupe `amr_reflux_mf.hpp` en 5 sous-entetes + umbrella, verbatim, ctest 93/93) ;
#169 A3 (average_down trailing covered cells AMR + tests discriminants). **En vol : #168 Polaire 2b.**

**PROCESS (nouveau, integrateur) :** chaque vague de PR passe une REVUE ADVERSARIALE (workflow multi-
lentilles : correctness / bit-identity / test-rigor / hygiene-scope, chaque finding verifie par un
skeptique independant) AVANT merge, EN PLUS de ci-full. Merge seulement si ci-full verte ET zero blocker
confirme. Gain mesure : la revue a attrape sur #168 un blocker que la CI NE voyait PAS (precondition
nr>=3 de `derive_aux_polar` non imposee -> lecture phi hors bornes pour `PolarMesh(nr=2)`) -> corrige.

Findings de revue : **1-7 sur master** ; 8 differe au portage MPI.

**VERDICT device GH200 : 7/7 device-clean Kokkos Cuda single-GPU + MPI+Kokkos Cuda multi-GPU
rank-invariant (10 tests, dmax=0, #157) + Kokkos OpenMP CI 91/91 (#155) + MPI+Kokkos OpenMP rank-invariant
(perf caveat : 3 tests lourds trop lents a np>1). Tous les echecs device initiaux etaient TEST-SIDE
(#150/#152/#158) ; le LIBRARY elliptique/Schur/polaire est device-correct.**

CI (depuis #136) : PR de routine = ci-fast (Release + Python). MPI + Kokkos via push master / nightly /
`workflow_dispatch` / label `ci-full`. REGLE : poser le label `ci-full` sur toute PR risquee (MPI /
Kokkos / device / Schur / AMR) AVANT merge pour la validation complete.

**En vol :**
- [~] **#168 Polaire 2b (PRIORITE, CE PR)** : transport + Poisson polaire branches dans `System.step`
  (anneau GLOBAL, PAS de couplage cart<->polaire ; cartesien defaut bit-identique). WIP de l'agent nuit
  recupere + 2 BUGS CORRIGES : (1) `phi` sans ghost -> derive de l'aux lisait phi hors allocation -> nan
  masque (test passait FAUX : `nan>tol` faux en C++) ; fix `derive_aux_polar` (radial DECENTRE ordre 2
  aux parois, theta ENROULE periodique). (2) revue adversariale : precondition nr>=3 non imposee -> OOB
  pour `PolarMesh(nr=2)` ; garde ajoutee (check_geometry C++ + PolarMesh Python) + test de rejet.
  Validation : C++ conservation masse 3.4e-15 ; Python `System(PolarMesh)` step+step_cfl <1e-11 ;
  cartesien 4/4 PASS. ci-full re-run en cours -> merge des verte.

**Prochaines etapes (sequencees, ETAT PRECIS post-scoping) :**
1. **Schur PR6** (apres merge #168) : mesure diocotron-Schur sur le chemin polaire desormais dispo ;
   isole l'effet TEMPOREL du Schur ; ne pretend PAS corriger l'erreur geometrique. Mesure + doc, pas une
   modif coeur. Write-set : adc_cases / harnais + doc.
2. **AMR capstone (cf. section 15, ROADMAP PRECIS)** : #154 a livre (ii) = facade 2 blocs explicites
   fonctionnelle (hierarchie partagee, schemas differents, Poisson somme co-localisee, conservation par
   bloc, MPI np=1/2/4). Le moteur compile-time `AmrSystemCoupler` sait DEJA substeps/stride/IMEX/sources
   couplees ; c'est la FACADE RUNTIME `AmrRuntime` qui ne les honore pas encore. Reste, par ordre :
   (iii) test validation Poisson-somme [trivial] ; **(iv) substeps/stride/evolve + step_cfl substeps-aware
   [IMMEDIAT, deverrouille le multirate ; write-set amr_runtime.hpp + amr_system.cpp, AUCUN conflit]** ;
   (v) DSL production multi-bloc [//] ; (vi) sources couplees runtime [//, A3 deja fait] ; (vii) IMEX
   multi-bloc runtime [apres vi] ; (viii) regrid union-tags [Phase 2]. stride_due deja correct (#140).
3. **Lot B restant** (apres merge #168, qui touche `system.cpp`) : SystemFieldSolver, SystemBlockStore,
   SystemStepper -- extractions de `python/system.cpp` (NativeLoader deja extrait #151), bit-identiques.
4. **Lot A.5** : convention de commentaires par dossier (CODE_DOCUMENTATION_CONVENTION.md), progressif,
   dossier par dossier, sans churn automatique. Basse priorite.
5. **A4** : couvert par le test C++/Python #168 (conservation masse avec flux radial interieur non nul) ;
   ajouter au besoin un cas MMS polaire avec v_r != 0 dedie.
6. **Hoffart re-drive** : build Kokkos PIC corrige (echec -fPIC du module Python sur aarch64), smoke ->
   n=384 -> n=512 GH200. Le chemin polaire annulaire est maintenant dispo dans System (apres #168).
7. **Perf (apres Polaire 2b, cf. #165)** : Poisson MG V-cycle domine 96-99.9% et REGRESSE sous Kokkos
   OpenMP (parallel_for par lissage jusqu'aux grilles 2x2 ; le chemin serie garde n_cells>=4096, pas le
   chemin Kokkos). UNE optim par PR, profil avant/apres. Pas avant que Polaire 2b soit merge.
8. **BORD D'ANNEAU = discontinuite de densite transportee, PAS une paroi** : NE PAS refaire paroi-transport
   dessus (physiquement faux) ; leviers valides = polaire / haut ordre / AMR. Differes : finding 8
   (`fab(0)`) au portage MPI ; `/(2*dx)->*cx` au dernier bit ; P7 (implicit-total + params runtime DSL).

## 0. API publique RECOMMANDEE (point d'entree utilisateur)

Deux facons d'ecrire un modele, toutes deux executees en C++ natif (zero boucle cellule par cellule
en Python sur le chemin performant) :

- **Composer des briques natives** : `adc.Model(state, transport, source, elliptic)` -> assemble un
  `ModelSpec` a partir de briques d'etat / transport / source / elliptique deja compilees dans la lib.
- **Ecrire le modele en formules** : `adc.dsl.Model(...)` (DSL symbolique) puis `m.compile(...)`. Le
  backend RECOMMANDE est **`production`** (loader natif zero-copie `add_native_block`, objectif
  MPI/GPU/AMR) ; le marquer comme defaut conseille du DSL.

Chemins AVANCES / LEGACY / TEST (PAS le chemin utilisateur principal) :
- `m.compile(backend='prototype')` (JIT NumPy/hote), `m.compile(backend='aot')` (.so ABI plate) :
  chemins de developpement / portabilite, doubles par `production`.
- `System.add_dynamic_block` (JIT) et `System.add_compiled_block` (AOT) : adders bas niveau ; preferer
  `add_native_block` (production) via la facade DSL.
- `adc.PythonFlux` : backend de PROTOTYPAGE, chemin HOTE pur (numpy, ordre 1 Rusanov periodique), HORS
  hot path GPU/MPI. Pour tester rapidement un flux inedit sans recompiler, PAS pour la production.

## 1. Chantier "Aux extensible" (champs auxiliaires au-dela de phi / grad)

Objectif : un modele declare/lit des champs aux SUPPLEMENTAIRES (B_z magnetique, T_e electronique)
sans casser l'existant, en retro-compat bit-exacte (`n_aux` defaut = 3 -> strictement identique).

- [x] **Inc. 1 - Lecture** : `adc::Aux` + `B_z` (comp 3), `kAuxBaseComps=3`, `aux_comps<Model>()`,
      `load_aux<NComp>`. Les foncteurs nommes lisent `load_aux<aux_comps<Model>()>`. (#24)
- [x] **Inc. 2 - Peuplement Coupler** mono-bloc : `fill_bz`, aux alloue a `aux_comps<Model>()`. (#24)
- [x] **Inc. 3 - Peuplement SystemAssembler** multi-blocs (aux = max sur les blocs). (#25)
- [x] **Inc. 4 - `CompositeModel::n_aux`** = max des briques ; `aux_comps` deplace dans
      `physical_model.hpp` (header contrat). (#26)
- [x] **Inc. 5 - runtime `System`** : `ensure_aux_width` + `set_magnetic_field` (binding Python
      calque sur `set_epsilon_field`). Chemin natif `add_compiled_model` complet. (#29)
- [x] **Inc. 6 - DSL** : emet `n_aux` quand une formule lit `aux('B_z')` (`AUX_CANONICAL`). (#30)
- [x] **Inc. 7 - chemin JIT** `add_dynamic_block` : `IModel::n_aux()` virtuel + marshaling
      `aux_ncomp_` -> B_z transporte, Python end-to-end. (#32)
- [x] **Inc. 8 - chemin AOT compile** `add_compiled_block` : l'ABI `compiled_block_abi.hpp`
      transporte desormais la largeur aux (B_z/T_e), symetrique de l'inc. 7 cote ABI `extern "C"` ;
      modele DSL B_z pilote 100% depuis Python via `compile_aot`. (#46)
- [x] **T_e - 2e champ extra DERIVE** : T = p/rho calculee par le `System` depuis un bloc fluide
      designe a chaque solve (comp 4, `set_electron_temperature_from`, recalcule dans `solve_fields`,
      pas user-fourni comme B_z). Valide la generalisation a 2 champs aux. (#35) ; lu sur les TROIS
      chemins dynamiques : natif (#35), AOT (test #50), JIT (marshaling complete #51, etait a 0).
- [x] **AMR / implicite** : `load_aux` width-aware sur `advance_amr` et le stepper implicite
      (canal extensible sur le chemin AMR) ; bit-identique pour un modele de base. (#42)
      + B_z peuple PAR NIVEAU dans le coupleur AMR de systeme (chaque niveau recoit son B_z). (#53)

## 2. Chantier "EPM elliptique generique" (operateur elliptique composable)

- [x] Permittivite variable `eps(x)` : `GeometricMG::set_epsilon` + `System::set_epsilon_field`
      (binding Python) sur `master`.
- [x] `EllipticProblem` / `FieldPostProcess` nommes (coeff, CL, nullspace, convention `E = -grad phi`).
- [x] Second membre de Poisson de systeme GENERIQUE (somme des `elliptic_rhs` des briques par bloc). (#43)
- [x] Operateur elliptique ECRANTE / Helmholtz `div(eps grad phi) - kappa phi = f` (GeometricMG + binding). (#44)
- [x] Operateur elliptique ANISOTROPE `div(diag(eps_x, eps_y) grad phi)` : coeur GeometricMG (#52),
      eps_x(x)/eps_y(x) exposes au runtime System + Python (#56), test cut-cell + anisotrope MMS ordre 2 (#55).
- [ ] Recabler les sites en forme `/(2*dx)` vers la forme multiplicative `*cx` (`amr_coupler`,
      `amr_coupler_mp`, `spectral_coupler`) - differe au dernier bit, donc hors perimetre tant
      qu'on veut le bit-identique.

## 3. Durcissement de l'architecture (`docs/archive/ROADMAP.md`, ARCHIVE -- plus une doc active)

- [~] **Moteur AMR unifie** : `advance_amr(LevelHierarchy&)`, `FluxRegister`, `CoverageMask`,
      `RegridPolicy` (`amr_regrid_finest`) promus en vrais types. `PatchRange` promu (empreinte
      grossiere `[I0..I1]x[J0..J1]` d'un patch fin ; dedup de 6 calculs d'empreinte inlines, arithmetique
      `(hi-1)/2` historique preservee donc bit-identique). (#80) RESTE : le routage bordant de
      `CoarseFineInterface` et `SubcyclingSchedule` (encore inlines dans la recursion `subcycle_level_mp`)
      et replier la famille `amr_step_*` (`amr_step_2level_multipatch` replique-grossier vs
      `amr_step_multilevel_multipatch` recursif : invariants distincts, non fusionnes pour garder le
      bit-identique ; `advance_amr` sert deja de facade unifiee).
- [x] **API memoire explicite** : `for_each_cell_reduce_{sum,max}`, `sum`/`norm_inf` faits.
      `sync_host()` / `sync_device()` explicites poses sur le seam `for_each.hpp` + methodes
      `MultiFab` : encodent l'intention de residence. Sous memoire unifiee (`Kokkos::SharedSpace`)
      `sync_host()` est un `device_fence()` cible, `sync_device()` un no-op (bit-identique).
      Scaffolding pour le futur chemin NON unifie (buffers separes + deep_copy).
- [x] **Familles de ghosts** : `fill_physical_bc` / `fill_boundary` / `mf_fill_fine_ghosts` separes ;
      le coarse-fine remonte en helper nomme de premier niveau `fill_cf_ghost_cell` (interp constante
      en espace + lineaire en temps, factorise les 3 corps de `mf_fill_fine_ghosts_t/_multi/_mb`). (#80)
- [x] **VariableRole** : couplages inter-especes par role (#18) + la brique generee par le DSL
      declare ses VariableRole et le runtime resout les variables par role avec fallback indices. (#40)
      `block_names()` lit desormais le registre de blocs C++ (voit JIT/AOT, plus seulement
      `add_block`) (#72) ; les blocs `.so` (JIT `add_dynamic_block` + AOT `add_compiled_block`)
      transportent leurs noms/roles/gamma via symboles ABI OPTIONNELS, avec fallback pour les `.so`
      anciens et garde-fou sur la longueur de `names=`. (#75)
- [x] AMR multi-patch distribue MPI (2 et N niveaux), `CouplingPolicy` mince, suite de validation
      numerique coeur, decoupage elliptique (operateur / solveur / probleme).

**Ordre de parite `AmrSystem` -> `System`** (combler les ecarts du chemin AMR dans cet ordre) :
- [x] **Gap 1 - flip facade** : rejet FACADE Python de HLLC/Roe + reconstruction primitive leve sur
  `AmrSystem.add_equation` (le moteur C++ `add_compiled_model(AmrSystem&)` les supportait deja ; rejet
  PUREMENT facade). FAIT.
- [x] **Gap 2 - IMEX sur AMR** : `mf_apply_source_treatment` selectionne forward-Euler vs
  `backward_euler_source` (foncteur nomme `BackwardEulerSourceKernel`) via un bool runtime ; parite
  dmax=0, conservation 1e-15, defaut explicite bit-identique, tests stiff. (#132)
- [~] **Gap 3 - multi-box natif** + **Gap 4 - multi-espece (capstone)** : FUSIONNES dans le capstone AMR
  multi-blocs (section 15, doc #139). Constat : le moteur multi-blocs EXISTE DEJA (`AmrSystemCoupler`) ;
  reste a exposer N blocs via la facade runtime + ajouter `regrid` au coupleur. EN ATTENTE DE GO.

## 4. GPU (GH200) - integration

- [x] Composants valides SEPAREMENT et bit-identiques au CPU sur GH200 : System mono-grille, ops de
      champ AMR, halos MPI multi-GPU, backend AOT d'un modele DSL, `load_aux<4>` (B_z device).
- [x] Parite multi-box MPI du chemin compile (`add_compiled_model` / `make_block`, np=1/2/4
      bit-identique) PERENNISEE comme test de regression dans le depot. (#39)
- [x] `add_compiled_model` cable cote `AmrSystem` (pendant multi-niveau du chemin compile). (#45)
- [x] **Validation INTEGREE** `AmrSystem` + MPI + GPU en un seul run : FAITE sur GH200 (`exec=Cuda`,
      euler_poisson compile sur AMR 128->256 multi-patch, patchs fins repartis 1 GPU/rang). np=1/2/4
      BIT-IDENTIQUES (csum identique a 17 chiffres, dmax=0), `crossrank_spread=0`, masse conservee
      (dm=0). Test de regression CPU/MPI `test_mpi_amr_compiled_parity` (CI) + harness GPU
      `python/tests/gpu/amrmpi_integrated.cpp`. Doc `docs/GPU_RUNTIME_PORT.md` (phase 10). (#48)
- [x] **Validation device round 2** des features post-#48 (GH200, `exec=Cuda` vs oracle Serial,
      `dmax` bit-a-bit) : T_e `load_aux<5>`, EPM Helmholtz, EPM anisotrope, B_z par niveau AMR ->
      tous **`dmax=0`** ; harness `python/tests/gpu/gpu_{aux,epm,amr_bz}_validate.cpp`. (#61)
- [~] **B_z multi-box distribue multi-GPU (#59)** : FONCTIONNEL sur device (B_z par boite/niveau lu,
      `bz_bad=0`, `cmax` bit-identique `dcmax=0`) mais PAS bit-identique sur les sommes globales
      (`dmass~1e-15`, `dcsum~3e-13`) -- effet d'ORDRE DE REDUCTION FMA (grossier reparti). Pas un bug.
- [x] **Limite device (a) LEVEE** : `System::add_compiled_model` est device-clean sur Cuda. Les
      fermetures d'enveloppe de `block_builder.hpp` (`BlockRhsEval`, `Advance*`, `RhsInto`, `MaxSpeed`,
      `PoissonRhs`) sont passees de lambdas etendues a des FONCTEURS NOMMES (meme recette que les
      kernels) -> plus de segfault Cuda, parite A==B bit-identique sur GH200 (`dres=0`, exit 0 ;
      avant : SIGSEGV exit 139). Modeles DSL compiles utilisables sur GPU via la facade `System`. (#64)
- [x] **Limite device (b) LEVEE** : la facade `AmrSystemCoupler` s'instancie + compile desormais sous
      nvcc. La sonde du concept `CoupledSystemLike` (`s.for_each_block([](auto&){})`, lambda generique
      en contexte non evalue que le frontend nvcc/EDG refusait -> CTAD du coupleur impossible) est
      passee a un FONCTEUR NOMME `detail::ForEachBlockProbe` (meme recette que (a) / #64). Valide GH200
      (job 637927) : `CUDA_BUILD_OK`, `exec=Cuda` OK, U(grossier+fin, 2 blocs) bit-identique au Serial
      (`dmax=0`), avant/apres confirme (sonde lambde remise -> echec nvcc). Harness perenne
      `python/tests/gpu/gpu_amrsys_facade_validate.cpp` (+ gpuval2 CMake). Hote inchange (72/72 ctest).
- [ ] **Perf full-device** : le run integre NE SCALE PAS (grossier REPLIQUE -> Poisson/transport
      grossier redondants par GPU ; seuls les patchs fins se repartissent). Mode `replicated_coarse=false`
      (grossier reparti) existe dans `AmrCouplerMP` mais degrade le MG et n'est pas cable dans `AmrSystem` :
      vrai chantier strong-scaling AMR. + parite AOT zero-copie sur device (sans rebond hote).

## 5. Physique magnetisee

- [x] Push de Boris E+B combine (`tfap_boris`, cyclotron exact, derive ExB sans croissance seculaire).
- [ ] Reformulation AP tensorielle sous champ fort.

## 6. Reproduction Hoffart (arXiv:2510.11808) - APPLICATIF, cote `adc_cases`

- [~] **M1** : taux de croissance numerique vs analytique (diocotron). Pipeline valide ; `gamma_norm`
      croit vers 0.911 mais limite par la diffusion numerique du bord d'anneau (-> motive l'AMR).
- [x] **M2 / M2b** : AMR sur le bord d'anneau (triple le taux a base egale) + Poisson multi-niveau.
- [ ] Montee en resolution / convergence vers le taux analytique ; integration SAMRAI ulterieure.

## 7. Nettoyage / consolidation (vague P1/P2, juin 2026)

Passe de durcissement APRES le chantier aux/EPM/GPU : pas de feature papier (SAMRAI, domaine
disque, Schur EPM, AMR multi-bloc, repro Hoffart) -- toutes DIFFEREES. Une PR par bloc, CI verte.

- [x] **Garde-fou ctest** : les 10 tests Python des chantiers aux/EPM/roles enregistres en ctest
      avant le nettoyage (filet de regression). (#69)
- [x] **AmrSystem** : refus EXPLICITE des parametres non cables + dedup des deux chemins de build AMR. (#71)
- [x] **Docs honnetes** : README / ARCHITECTURE / ALGORITHMS alignes sur l'etat reel, notes hors-nav
      archivees (sans toucher le backlog vivant `todo.md` ni la `ROADMAP`). (#70)
- [x] **Headers morts** : 6 en-tetes a 0 include / 0 test documentes comme API non cablee (0 suppression). (#73)
- [x] **Dedup helpers aux** : `derive_aux_bc`, `fill_bz`, `wall_active` factorises
      (`coupling/aux_fill.hpp`, `runtime/wall_predicate.hpp`). (#74)
- [x] **`block_names()` depuis le registre C++** (voit JIT/AOT, plus seulement `add_block`). (#72)
- [x] **ABI bloc `.so`** : noms / roles / gamma transportes par symboles ABI OPTIONNELS sur les deux
      chemins (JIT + AOT), fallback pour les `.so` anciens, garde-fou sur la longueur de `names=`. (#75)
- [x] **adc_cases** : package + manifest CI, validations renforcees (pas de validation importante
      masquee par le manifest). (adc_cases #2, #3)

### Vague parallele (juin 2026, 4 chantiers en PR separees, write-sets disjoints)

- [x] **adc_cases package maintenable** : `recipes.py` (recettes systeme separees des modeles
      mono-espece), `common/native.py` (compile JIT hors source + cle d'ABI + erreurs de symbole
      explicites), preambule d'import garde (plus de `sys.path.insert` inconditionnel), `two_fluid_ap`
      build dans `out/<cas>/build/` (git-ignore). (adc_cases #4)
- [x] **Facade DSL `m.compile(backend=prototype|aot|production)`** : aiguillage par intention couple a
      l'adder System correct ; preserve noms/roles/gamma/n_aux/B_z/T_e ; `require_metadata=True` leve une
      erreur explicite au lieu du fallback muet. PUR-PYTHON (aucune modif binding). A #79, `production`
      etait encore un alias de `aot` ; depuis #85 c'est un backend natif zero-copie DISTINCT
      (`add_native_block`), cf. "Suite identifiee (faite)" ci-dessous. (#79)
- [x] **Moteur AMR durci** : `PatchRange` + helper coarse-fine `fill_cf_ghost_cell` promus,
      bit-identique (Serial 73/73, MPI 94/94 np=1/2/4). (#80) Voir section 3.
- [x] **Audit feuille de route papier** : `docs/PAPER_ROADMAP.md` (4 paniers : API actuelle / DSL
      production / domaine disque FV / AMR multi-bloc + EPM avance). Verrou = bord d'anneau cartesien
      (cut-cell ne sert que Poisson, pas le transport). Aucune implementation. (#78)

### Suite identifiee (faite)

- [x] **Routage `CoarseFineInterface` + `SubcyclingSchedule`** extraits de `subcycle_level_mp` en types
      nommes, bit-identique (Serial 74/74, MPI 95/95 np=1/2/4). (#82)
- [x] **Backend DSL `production` distinct** : loader natif zero-copie `add_native_block` (inline
      `add_compiled_model<ProdModel>`, ABI-key gate, symboles `ADC_EXPORT`), `production` ne pointe plus
      sur `aot`. Portabilite ELF (promotion `_adc` en portee globale). Parite CPU bit-identique a
      `add_block`. (#85)

## 8. Plan Ideal ADC : ecrire le modele en Python, executer en C++ natif

Objectif : l'utilisateur ecrit les equations en Python (DSL symbolique), ADC genere/compile une brique
C++ NATIVE branchee dans `adc_cpp` comme un modele ecrit a la main ; aucune boucle cellule par cellule
en Python sur le chemin performant. Trois backends : `prototype` (NumPy/hote), `aot` (.so ABI plate),
`production` (natif zero-copie, objectif MPI/GPU/AMR).

- [x] **Etape 1 - API `dsl.Model` stable** au-dessus de `HyperbolicModel` (facade ; `m.flux` declarateur
      / `m.eval_flux` evaluateur ; `m.primitive_vars(**kwargs)`). (#89)
- [x] **Etape 2 - `param` + `CompiledModel` + erreurs propres** : `Param` nomme compile-time (runtime =
      phase E) ; `CompiledModel` (backend/adder/so_path/noms/roles/gamma/n_aux/params/cle ABI) ;
      `add_equation` (dispatch `ModelSpec` vs `CompiledModel`), `FiniteVolume(riemann=)`, `run`. Erreurs
      explicites (role/param/backend/flux). (#89, #90 fix substeps ModelSpec)
- [x] **Etape 3 - `production` reel pour `System`** : loader natif zero-copie. (#85)
- [x] **Etape 4 - cas demonstrateurs `adc_cases`** : `diocotron_dsl` (ExB en formules, == `models.diocotron`
      BIT-IDENTIQUE) + `two_species_dsl` (electrons+ions, temps par bloc, Poisson `sum q_s n_s`), backend
      `production`, validation CI legere (adc_cases #7) + `magnetic_isothermal_dsl` (isotherme magnetise,
      B_z pilote depuis Python via `set_magnetic_field`, oracle Lorentz, `aot`==`production` bit-identique en
      CI Linux, adc_cases #9). Les 3 demonstrateurs DSL tournent en CI.
- [x] **Etape 5 - `production` -> `AmrSystem`** (Phase D) : `AmrSystem::add_native_block` +
      `target="amr_system"` ; parite bit-identique a `add_compiled_model(AmrSystem&)`. VALIDE CPU/CI
      (test_amr_native_loader dlopen, Release+MPI+Kokkos verts). (#92) **WENO5/Rusanov/conservatif** cable
      sur le chemin natif AMR (parite `add_native_block`==`add_compiled_model`==`add_block`, dmax=0, #105).
      Limites restantes (cf. ordre de parite AmrSystem, section 3 + capstone section 15) : AMR mono-bloc /
      pas multi-espece ; **IMEX source LOCALE OK (Gap 2 #132)** mais pas de Schur GLOBAL sur AMR ; multi-box
      natif non cable cote facade. (HLLC/Roe/reconstruction primitive : **Gap 1 LEVE cote facade**.)
- [~] **Etape 6 - validation MPI/GPU du chemin `production`** : **np=1 GPU VALIDE sur GH200.** Le crash
      device dans `solve_fields()` etait du a des lambdas `ADC_HD` etendues inline (noyaux elliptiques/mesh
      `copy_shifted`/`fill_boundary`/MG, premiere instanciation cross-TU -> stub kernel nvcc nul en
      Release sans `-g`) ; converties en FONCTEURS NOMMES (#97, meme recette que #64). Preuve GH200 (job
      ROMEO 640236, Release sans `-g`) : `geometric_mg` ET `fft` Cuda np=1 exit 0 (etaient 139),
      compute-sanitizer 0 erreur, `dmax_abs` Cuda-vs-Serial = 5.0e-13 (MG) / 1.3e-15 (FFT) -- DANS LA
      TOLERANCE (reassociation FMA des reductions Kokkos), CPU bit-identique. **MPI `solve_fields`
      np=1/2/4 VALIDE CPU/CI** : le bug mono-box / `fab(0)` appele sans test `local_size()` (rang sans
      box locale -> segfault hote) est corrige par une garde `local_size()` (no-op sur rang sans box,
      np=1 bit-identique) + test `test_mpi_system_solve_fields_np{1,2,4}`. (#99) **Device-MPI production
      VALIDE sur GH200** (job ROMEO 641249, harness #93) pour `geometric_mg` : np=1/2/4 exit 0, pas de
      deadlock, bit-identique cross-np, compute-sanitizer 0 erreur, `dmax_abs` Cuda-vs-Serial dans la
      tolerance ; `fft` np=1 OK. **`fft` np>1 sous System = REFUSE proprement (#106)** : `set_poisson("fft")`
      leve si n_ranks()>1, et l'`assert(n_ranks()==1)` compile-out devient un garde-fou DUR (throw actif en
      Release) dans `PoissonFFTSolver`. fft direct = np=1 seulement ; `DistributedFFTSolver` existe (teste a
      part, `test_mpi_fft_distributed`) mais non route dans System (layout bandes vs box unique).
- [ ] **Etape 7 - domaine disque FV / Polaire Phase 2b** (prochain livrable scientifique) :
      DECISION DU PROPRIETAIRE SCIENTIFIQUE -- le bord d'anneau du diocotron est une DISCONTINUITE DE
      DENSITE TRANSPORTEE, PAS une paroi physique. NE PAS refaire "paroi-transport" (masque / cut-cell fixe)
      sur le bord d'anneau : ce serait physiquement FAUX. Le cut-cell reste pertinent pour le CONDUCTEUR
      EXTERNE uniquement, mais l'experience #109 a montre qu'il NE CORRIGE PAS le taux de croissance.
      Pour le bord d'anneau, les seuls leviers valides sont la GEOMETRIE POLAIRE, le HAUT ORDRE (WENO5/SSPRK3),
      et l'AMR. Le prochain livrable scientifique est le CHEMIN POLAIRE COMPLET (Polaire Phase 2b) :
      cabler transport+Poisson polaire dans `System.step` + couplage cartesien<->polaire + RUN diocotron
      annulaire. Reproduction papier quantitative subordonnee (cf. section 6, confirmation haute resolution
      du plateau l=4).

**STATUT HONNETE (ne PAS presenter "Plan Ideal termine")** : System production CPU = OK ; AmrSystem
production CPU = OK (WENO5/Rusanov/conservatif, #105) ; demonstrateurs DSL = OK (diocotron_dsl,
two_species_dsl, magnetic_isothermal_dsl, tous en CI) ; **production GPU np=1 = OK (GH200, #97)** ;
**`solve_fields` MPI np=1/2/4 = OK cote CPU/CI (#99)** ; **device-MPI production `geometric_mg` = VALIDE
(GH200, #93, np=1/2/4)** ; **`fft` np>1 sous System = refuse proprement (#106), plus de segfault** (DistributedFFTSolver non route, layout) ; `set_density`/`get_state` multi-rang
= hors scope ; WENO5 sur `CompiledModel` (.so AOT ET production) = SUPPORTE (3 ghosts via `set_block_ghosts`, #102),
WENO5 cable aussi sur le chemin natif AMR (#105) ; `m.compile()` ergonomique (auto-detect include + cache `so_path`, #103) ;
`AmrSystem.potential()` = binding EXISTE et expose (`python/bindings.cpp:272`) ; **Schur/polaire device** :
6/6 device-clean sur Kokkos Cuda single-GPU (cf. VERDICT en tete) ; les 3 echecs initiaux de la validation
GH200 etaient TEST-SIDE (foncteurs hote dans des kernels device), fixes #150+#152, library device-correct ;
`PAPER_ROADMAP.md` = a NE PAS reecrire automatiquement
(attend la validation humaine du sweep O5) ; **prochain livrable scientifique = CHEMIN POLAIRE COMPLET (Polaire
Phase 2b). DECISION SCIENTIFIQUE : le bord d'anneau est une discontinuite de densite transportee, PAS une paroi
physique ; NE PAS refaire paroi-transport sur ce bord (physiquement faux). Cut-cell pertinent pour le conducteur
externe uniquement (#109 n'a pas corrige le taux). Leviers valides pour le bord d'anneau : geometrie polaire,
haut ordre (WENO5/SSPRK3), AMR.**

## 9. Mesure diocotron haut ordre (PR-0 + O5, cote `adc_cases`)

- [x] Balayage ordre x resolution O1/O2 (PR-0) puis O5 = WENO5-Z + SSPRK3 (atteignable depuis Python
      via #88) : `diocotron/SWEEP_RESULTS.md`. l=3 part majoritairement diffuse ; l=4 passe de ~-12% (O2)
      a ~-4% sur deux points propres (n=128/256) a O5 (n=192 = artefact de fenetre de fit, trace) ; l=5
      a la cible. Conclusion PRUDENTE : l'hypothese PR-0 d'un plateau structurel ~12% est AFFAIBLIE (sans
      etre refutee). (#5, #6)
- [ ] **Confirmation haute resolution n=384 / n=512 (incl. O5) sur ROMEO/GH200** AVANT toute reecriture
      de la roadmap papier (`PAPER_ROADMAP.md`). Sur accord.

## 10. Condensation de Schur (EPM implicite condense, theta-schema)

Algorithme NUMERIQUE (pas une physique) : eliminer la vitesse pour condenser sur le potentiel ->
pas implicite stable a grand dt. Modele = physique ; SchurCondensation = algo. Operateur condense
`-Lap phi - theta^2 dt^2 alpha div(rho B^-1 grad phi)` (NON symetrique) -> Krylov.

- [x] **PR0 - doc** `docs/SCHUR_CONDENSATION_DESIGN.md` : convention de signe verrouillee
      `A_op = I + theta^2 dt^2 alpha rho B^-1`, A non symetrique -> Krylov necessaire. (#119)
- [x] **PR2 - LorentzEliminator** : `B`, `B^-1` analytique (`apply_Binv`, `binv_ij`), POD/`ADC_HD`. (#118)
- [x] **PR1 - TensorEllipticOperator** : `-div(A grad phi) + kappa phi`, termes croises (stencil 9 points,
      foncteur nomme `cross_div`), `set_cross_terms`. A=I/diag bit-identique. (#120)
- [x] **PR3 - Krylov BiCGStab matrice-libre** non symetrique, preconditionne par MG sur la partie
      symetrique (MG seul stagne/diverge), dots collectifs MPI-safe. (#122)
- [x] **PR3 - batisseur** `schur_condensation.hpp` : coefficients de `A_op` + RHS condense, generique
      sur les roles. (#124)
- [x] **PR4 - etage source** `CondensedSchurSourceStepper` : build -> Krylov -> reconstruit v ->
      energie -> extrapole U^{n+1} -> ghosts. Relation implicite 1e-15, stable a 8x le dt explicite. (#126)
- [x] **PR5 - binding Python** : `adc.Split(hyperbolic, source=adc.CondensedSchur(...))` + `adc.Role` +
      routage `set_source_stage` (no-op si nullptr, apres transport). (#128)
- [x] **Fix revue (findings 1+2)** : precond ET matvec a CL HOMOGENES sous Dirichlet non nul (la matvec
      de r0 garde l'affine ; matvec en boucle + precond linearises en retranchant leur offset) ;
      `op_`!=`precond_` impose. Bit-identique a Dirichlet nul. (#134)
- [~] **Device-clean GPU** : le stack Schur etait FAUX EN SILENCE sur Cuda (accesseurs Geometry/Box2D
      non `ADC_HD` -> RHS/reductions faux, BiCGStab "0 iters rel=0 puis NaN"). Fix dans #135 (en vol,
      attend validation GH200). Host = correct, CI verte.
- [ ] **PR6 - mesure diocotron-Schur** : DIFFERE jusqu'a la geometrie polaire (le Schur stabilise le
      TEMPS ; il n'adresse PAS le gap de taux de croissance, qui est GEOMETRIQUE).
- [ ] **PR7/PR8 - portage GPU/MPI + AMR** : apres #135 ; bit-invariance au nombre de rangs.

## 11. Pipeline ergonomique System (P1-P7)

- [x] **P1 - stride par bloc** : `Explicit/IMEX(stride=M)` (hold-then-catch-up), CFL
      `dt=cfl*h*substeps/(stride*w)`. (#121) Honnetete du wording -> finding revue 3 (#138).
- [x] **P2 - clarifier Implicit/IMEX** : renommage `SourceImplicit`, deprecation de `Implicit`. (#123)
- [x] **P3 - masque `implicit_vars`** sur la politique temporelle / le bloc (PAS sur le modele). (#125)
- [x] **P4 - `set/get_primitive_state`** (init/diagnostic en variables primitives). (#127)
- [x] **P5 - CoupledSource DSL** : `adc.dsl.CoupledSource` -> bytecode postfixe interprete par un
      foncteur device nomme dans `apply_couplings` ; generique inter-especes ; 3 especes + conservation,
      MPI np=1/2/4 bit-identique. (#131)
- [ ] **P6 - AMR multi-blocs** = capstone Gap 4, voir section 15.
- [ ] **P7 - implicit-total + params runtime DSL** : DIFFERE.

## 12. Geometrie polaire / annulaire (le vrai verrou du taux de croissance diocotron)

Conclusion scientifique : le verrou du taux de croissance est l'advection CARTESIENNE du gradient
radial de l'anneau ; la geometrie POLAIRE le preserve (proto : ratio 73 vs 1.0 cartesien). Le Schur
n'adresse PAS ce gap (il stabilise le temps). Donc le levier = mettre la geometrie polaire.

- [x] **Abstraction Mesh** : `adc.PolarMesh` / `adc.CartesianMesh` -> `System(mesh=)` (PAS
      `FiniteVolume(geometry=)`). `PolarGeometry`, divergence polaire `(1/r)d_r(r F_r)+(1/r)d_th F_th`,
      `ExBVelocityPolar` `v_r=-(1/(Br))d_th phi`, `v_th=(1/B)d_r phi`. (#116)
- [x] **Phase 2a - Poisson polaire direct** sur anneau : FFT-en-theta + tridiagonal-en-r (robuste,
      evite le MG-en-polaire qui stagne). (#130)
- [~] **Device** : flux/metrique polaire FAUX en silence sur Cuda (meme cause #135 : `r_cell`/`theta_cell`
      non `ADC_HD`). Fix #135 (en vol).
- [~] **Phase 2b (#168, en vol)** : transport + Poisson polaire branches dans `System.step` sur un anneau
      GLOBAL (PAS de couplage cartesien<->polaire en v1 ; cartesien defaut bit-identique). `derive_aux_polar`
      (aux en base locale e_r/e_theta), paroi radiale solide (wall_radial), `mass`/`step_cfl`/`step_adaptive`
      polaires, garde nr>=3. Tests C++ (conservation 3.4e-15) + Python `System(PolarMesh)` (<1e-11). RESTE
      apres merge : le RUN diocotron annulaire quantitatif (mesure du taux) = livrable scientifique suivant.

## 13. Revue adversariale + durcissement (findings)

Revue adversariale exhaustive du travail merge (Schur + pipeline System + polaire) : 8 findings
confirmes reels sur 12. Disposition :

- [x] **F1 - precond CL non homogenes** (krylov) : operateur affine -> BiCGStab faux pour Dirichlet
      non nul. Corrige #134.
- [x] **F2 - aliasing `op_`==`precond_`** : commentaire "toujours valable" faux. Corrige #134.
- [ ] **F3 - step_cfl substeps>1 non bit-identique** : DECISION = garder la CFL substeps-aware
      (correcte), corriger le wording + tests. En vol #138.
- [ ] **F4 - `evolve=False` ignore en silence sur prototype/aot** : rejet explicite. En vol #137.
- [ ] **F5 - `adc.Split` perdu en silence sur AmrSystem** : rejet explicite. En vol #137.
- [ ] **F6 - descripteurs `CondensedSchur` jamais transmis** (code mort API) : rejet non-defaut. En vol #137.
- [ ] **F7 - `max_wave_speed_mf` non collectif** (dt divergent par rang MPI) : `all_reduce_max`. Dans #135.
- [ ] **F8 - `fab(0)` sans garde `local_size()`** dans le marshaling : DIFFERE au portage MPI (un demi-fix
      ferait un faux-silencieux au lieu d'un crash franc ; a faire avec gather/scatter).

## 14. Acceleration CI (garder tous les tests, aller plus vite)

- [x] **#136 (FAIT, sur master)** : cache du Kokkos INSTALLE (gros gain : plus de rebuild a chaque run)
      + ccache (`CMAKE_CXX_COMPILER_LAUNCHER`) + Ninja + `ctest --parallel 2` (PAS plus : TU AMR/Eigen
      1-2 Go) + `setup-python@v6` + auto-decouverte `python/tests/test_*.py` +
      `concurrency: cancel-in-progress` + split **ci-fast** (PR) / **ci-full** (push master + nightly +
      `workflow_dispatch` + label `ci-full`). Aucun test supprime ; label `ci-full` pour exiger
      MPI/Kokkos sur une PR risquee.
      RESULTAT MESURE : **~25 min -> ~5 min a chaud** (Release 4.8 min, MPI 2 min, Kokkos 1.7 min) ;
      **45 tests Python** auto-decouverts (superset des 40 ; +5 jamais cables avant). Les PR d'agents
      suivantes en beneficient des maintenant (CI plus rapide = iteration plus rapide).

## 15. AMR multi-blocs sur hierarchie partagee (capstone Gap 4 / P6)

Reference : `docs/AMR_MULTIBLOCK_DESIGN.md` (#139). Modele AMReX/FLASH/SAMRAI : UNE hierarchie AMR
commune portant N blocs co-localises ; regrid pilote par l'UNION des criteres.

Cadrage STRICT (verrouille) : meme hierarchie, memes cellules, champs co-localises ; TOUS les blocs
evolues sur TOUS les patchs (jamais d'absence spatiale d'une espece) ; souplesse uniquement par-bloc
(modele/spatial/time/substeps/stride/evolve) ; Poisson `rhs[level]=somme des elliptic_rhs` co-localises ;
sources couplees meme-cellule a contributions EXACTEMENT opposees ; reflux par bloc ; tags = e OR i OR n
OR phi OR user ; optimisations futures TEMPORELLES seulement (stride / evolve global au bloc), JAMAIS
spatiales locales. PAS de hierarchie par espece en v1 (Phase 3 seulement si besoin scientifique reel).

CONSTAT CLE (confirme par scoping juin 2026 sur master = #154) : le moteur multi-blocs EXISTE DEJA =
`AmrSystemCoupler` (compile-time : hierarchie partagee, Poisson somme, scheme/substeps/stride par bloc,
IMEX-callback, sources niveau-par-niveau + average_down trailing #169, B_z partage ; `same_layout_or_throw`).
La FACADE RUNTIME `AmrRuntime` (#154, `include/adc/runtime/amr_runtime.hpp`) expose un registre type-erase
par nom et fait DEJA tourner 2 blocs explicites, mais ne TRANSMET pas encore substeps/stride/IMEX/sources
couplees/regrid de la facade vers le moteur. Donc COMPLETION cote facade, pas creation. Le bug `stride_due`
est DEJA corrige (#140 : cadence `(macro_step_+1) % stride`, hold-then-catch-up). Mono-bloc passe toujours
par `AmrCouplerMP` (jamais `AmrRuntime`) -> baseline bit-identique. Multi-bloc + regrid_every>0 REFUSE.

Decoupe PR (Phase 1, write-sets) - ETAT PRECIS (scoping) :
- [x] **(i)** extraire `AmrBlock` / `AmrHierarchyLayout` (bit-identique mono-bloc). FAIT #141.
- [x] **(ii)** 2 blocs explicites, schemas differents (facade AmrRuntime + Poisson somme co-localisee +
      conservation par bloc + MPI np=1/2/4). FAIT #154. Tests test_amr_system_twoblock + mpi_twoblock_parity.
- [ ] **(iii)** test de VALIDATION Poisson somme co-localise [trivial, validation seule ; le moteur le fait deja].
- [ ] **(iv) IMMEDIAT** substeps/stride/evolve + step_cfl substeps-aware : `AmrRuntime::step` doit honorer
      `substeps_`/`stride` par bloc (actuellement tous explicites a 1 sous-pas) ; `AmrSystem::step_cfl` doit
      devenir substeps-aware (`dt=cfl*h*min(sub_b)/max(stride_b*w_b)`). Write-set : amr_runtime.hpp +
      amr_system.cpp + 2 tests. AUCUN conflit avec system.cpp/dsl.py. stride_due deja correct (#140).
- [ ] **(v)** DSL production multi-bloc : `add_native_block`/`add_compiled_model(AmrSystem&)` ne doit plus
      lever sur le 2e bloc (file d'attente + build a `ensure_built`). Write-set amr_dsl_block.hpp + amr_system.cpp.
- [ ] **(vi)** sources couplees AMR meme-cellule / opposees : exposer `coupled_source_step` du moteur via le
      registre runtime ; A3 (average_down covered cells) DEJA fait (#169). Write-set amr_runtime.hpp + amr_system.cpp.
- [ ] **(vii)** IMEX local AMR runtime : la facade honore `time="imex"` multi-bloc (le moteur a deja le callback).
- [ ] **(viii)** Phase 2 : regrid union-tags (deverrouille multi-bloc + regrid_every>0) ; puis Schur / implicite
      global / repro papier.

Tests d'acceptation : 2 blocs explicites schemas differents ; e- IMEX(substeps=10) + ions
Explicit(substeps=1) ET l'inverse ; neutres stride=20 nourrissant sources/Poisson ; evolve=False fond
fixe dans le RHS elliptique ; regrid conserve la masse par bloc ; DSL production 2 blocs ; MPI np=1/2/4 ;
Kokkos Serial ; 1 cas multi-bloc production sur GH200.

## 16. Findings conservation (revue)

- [x] **A3 [BUG conservation] FAIT #169** : `AmrSystemCoupler::coupled_source_step` et `AmrImplicitSourceStepper` appliquaient la source aux cellules grossieres COUVERTES sans average_down trailing. Fix : cascade `mf_average_down_mb` fin->grossier apres la boucle de niveaux (no-op strict mono-niveau, bit-identique). Tests `test_amr_source_covered_cells` + `test_amr_composite_source_conservation` (discriminants : echouent sans le fix). Revu adversarialement (MERGE-SAFE).
- [x] **A2 [RISK conservation] FAIT #167** : helper `add_pair(block_a, block_b, role, expr)` emet +expr / -expr depuis le MEME sous-arbre (conservatif par construction) + mode `compile(verify_conservation=True)` opt-in. Test `test_dsl_coupled_source_conservation`. Revu adversarialement (MERGE-SAFE).
- [~] **A4 [GAP test]** : couvert en pratique par les tests #168 (conservation masse polaire avec flux radial INTERIEUR non nul, profil module en theta -> v_r != 0). RESTE optionnel : un cas MMS polaire dedie avec v_r != 0 analytique (au-dela de la conservation globale).
- Tests a ajouter : `test_dsl_coupled_source_conservation`, `test_amr_composite_source_conservation`, `test_amr_source_covered_cells`, `test_polar_conservation_radial_flux`.

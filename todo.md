# TODO - adc_cpp

> Liste de travail vivante. Synthese de (1) l'objectif initial du chantier (canal `aux` extensible
> + parite AMR + cablage runtime / Python / DSL), (2) ce que `docs/ROADMAP.md` marque "en file",
> (3) ce que les agents ont explicitement note comme "reste a faire".
> Convention : `[x]` fait et sur `master`, `[~]` partiel, `[ ]` a faire.

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

## 3. Durcissement de l'architecture (`docs/ROADMAP.md` "en file")

- [~] **Moteur AMR unifie** : `advance_amr(LevelHierarchy&)`, `FluxRegister`, `CoverageMask` promus
      en vrais types. RESTE : promouvoir `PatchRange`, le routage bordant de `CoarseFineInterface`,
      `SubcyclingSchedule`, `RegridPolicy` (encore inlines dans `subcycle_level_mp`) et y replier la
      famille `amr_step_*` (qui encode le cas dans le nom).
- [x] **API memoire explicite** : `for_each_cell_reduce_{sum,max}`, `sum`/`norm_inf` faits.
      `sync_host()` / `sync_device()` explicites poses sur le seam `for_each.hpp` + methodes
      `MultiFab` : encodent l'intention de residence. Sous memoire unifiee (`Kokkos::SharedSpace`)
      `sync_host()` est un `device_fence()` cible, `sync_device()` un no-op (bit-identique).
      Scaffolding pour le futur chemin NON unifie (buffers separes + deep_copy).
- [~] **Familles de ghosts** : `fill_physical_bc` / `fill_boundary` / `mf_fill_fine_ghosts` separes.
      RESTE : remonter le coarse-fine en helper nomme de premier niveau.
- [x] **VariableRole** : couplages inter-especes par role (#18) + la brique generee par le DSL
      declare ses VariableRole et le runtime resout les variables par role avec fallback indices. (#40)
      `block_names()` lit desormais le registre de blocs C++ (voit JIT/AOT, plus seulement
      `add_block`) (#72) ; les blocs `.so` (JIT `add_dynamic_block` + AOT `add_compiled_block`)
      transportent leurs noms/roles/gamma via symboles ABI OPTIONNELS, avec fallback pour les `.so`
      anciens et garde-fou sur la longueur de `names=`. (#75)
- [x] AMR multi-patch distribue MPI (2 et N niveaux), `CouplingPolicy` mince, suite de validation
      numerique coeur, decoupage elliptique (operateur / solveur / probleme).

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
- [ ] **Limite device (b)** : la facade `AmrSystemCoupler` ne s'instancie pas sous nvcc (concept
      `requires for_each_block`) -> chemin device de B_z-AMR valide via `advance_amr`, facade en CI
      Serial. (memes foncteurs nommes a appliquer a la facade, comme (a))
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

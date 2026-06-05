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
      erreur explicite au lieu du fallback muet. PUR-PYTHON (aucune modif binding). `production` =
      alias HONNETE de `aot` aujourd'hui (pas encore un backend zero-copie device distinct -> suivi). (#79)
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
      `production`, validation CI legere. (adc_cases #7) `magnetic_isothermal_dsl` saute (pas d'oracle natif).
- [x] **Etape 5 - `production` -> `AmrSystem`** (Phase D) : `AmrSystem::add_native_block` +
      `target="amr_system"` ; parite bit-identique a `add_compiled_model(AmrSystem&)`. VALIDE CPU/CI
      (test_amr_native_loader dlopen, Release+MPI+Kokkos verts). (#92) Limites AMR (mono-bloc, explicite,
      pas de recon primitive/Roe/weno5) rejetees explicitement.
- [~] **Etape 6 - validation MPI/GPU du chemin `production`** : **np=1 GPU VALIDE sur GH200.** Le crash
      device dans `solve_fields()` etait du a des lambdas `ADC_HD` etendues inline (noyaux elliptiques/mesh
      `copy_shifted`/`fill_boundary`/MG, premiere instanciation cross-TU -> stub kernel nvcc nul en
      Release sans `-g`) ; converties en FONCTEURS NOMMES (#97, meme recette que #64). Preuve GH200 (job
      ROMEO 640236, Release sans `-g`) : `geometric_mg` ET `fft` Cuda np=1 exit 0 (etaient 139),
      compute-sanitizer 0 erreur, `dmax_abs` Cuda-vs-Serial = 5.0e-13 (MG) / 1.3e-15 (FFT) -- DANS LA
      TOLERANCE (reassociation FMA des reductions Kokkos), CPU bit-identique. **MPI `solve_fields`
      np=1/2/4 VALIDE CPU/CI** : le bug mono-box / `fab(0)` appele sans test `local_size()` (rang sans
      box locale -> segfault hote) est corrige par une garde `local_size()` (no-op sur rang sans box,
      np=1 bit-identique) + test `test_mpi_system_solve_fields_np{1,2,4}`. (#99) RESTE : valider la
      production **device-MPI** (GPU + multi-rang) sur GH200 -- chantier SEPARE a venir.
- [ ] **Etape 7 - DIFFERE** : domaine disque FV / paroi transport + reproduction papier quantitative
      (cf. section 6 ; subordonne a la confirmation haute resolution du plateau l=4).

**STATUT HONNETE (ne PAS presenter "Plan Ideal termine")** : System production CPU = OK ; AmrSystem
production CPU = OK ; demonstrateurs DSL Python = OK ; **production GPU np=1 = OK (GH200, #97)** ;
**`solve_fields` MPI np=1/2/4 = OK cote CPU/CI (#99)** ; production **device-MPI** (GPU + multi-rang)
= reste a valider separement (GH200) ; `set_density`/`get_state` multi-rang = hors scope ; WENO5 sur
`CompiledModel` = encore limite (2 ghosts) ; `PAPER_ROADMAP.md` = a NE PAS reecrire automatiquement
(attend la validation humaine du sweep O5).

## 9. Mesure diocotron haut ordre (PR-0 + O5, cote `adc_cases`)

- [x] Balayage ordre x resolution O1/O2 (PR-0) puis O5 = WENO5-Z + SSPRK3 (atteignable depuis Python
      via #88) : `diocotron/SWEEP_RESULTS.md`. l=3 part majoritairement diffuse ; l=4 passe de ~-12% (O2)
      a ~-4% sur deux points propres (n=128/256) a O5 (n=192 = artefact de fenetre de fit, trace) ; l=5
      a la cible. Conclusion PRUDENTE : l'hypothese PR-0 d'un plateau structurel ~12% est AFFAIBLIE (sans
      etre refutee). (#5, #6)
- [ ] **Confirmation haute resolution n=384 / n=512 (incl. O5) sur ROMEO/GH200** AVANT toute reecriture
      de la roadmap papier (`PAPER_ROADMAP.md`). Sur accord.

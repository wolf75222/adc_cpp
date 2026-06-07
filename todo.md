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
#169 A3 (average_down trailing covered cells AMR + tests discriminants). #168 Polaire 2b MERGE.

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

**Polaire 2b MERGE (#168, master = 43a9160) :** transport + Poisson polaire dans `System.step` (anneau
GLOBAL, PAS de couplage cart<->polaire ; cartesien bit-identique). 2 bugs trouves+corriges : (1) `phi`
sans ghost -> derive de l'aux lisait phi hors allocation -> nan masque (test passait FAUX : `nan>tol`
faux en C++) ; fix `derive_aux_polar` (radial DECENTRE ordre 2 aux parois, theta ENROULE periodique).
(2) revue adversariale : precondition nr>=3 non imposee -> OOB pour `PolarMesh(nr=2)` ; garde
(check_geometry C++ + PolarMesh Python) + test de rejet. Validation : C++ conservation masse 3.4e-15 ;
Python `System(PolarMesh)` step+step_cfl <1e-11 ; cartesien 4/4 PASS.

**MILESTONE SCIENTIFIQUE (#174) :** le chemin polaire CAPTURE l'instabilite diocotron. Sanity locale
(anneau de charge creux, mode l=4 seme, WENO5/SSPRK3) : l'amplitude du mode CROIT proprement (x8.8 a
96x96/600 pas, taux ajuste gamma~0.195), masse conservee a 2.2e-14, densite finie/positive. Verrouille
en regression CI rapide (`python/tests/test_polar_diocotron.py`, 48x48, x>2). La mesure QUANTITATIVE du
taux vs theorie (l=3/4/5, O5, n=384/512) reste le run ROMEO.

**Vague parallele MERGEE (revue adversariale + ci-full) :**
- [x] **AMR capstone (iv) #175** : facade runtime honore substeps/stride par bloc + step_cfl substeps-aware
  (mirror de `AmrSystemCoupler::step` ; mono-bloc bit-identique via routage AmrCouplerMP ; test multirate
  avec neutralize-and-fail). MERGE-SAFE (revue 4 lentilles, 0 blocker).
- [x] **Lot A.5 core/ #173** : convention de commentaires sur `include/adc/core/` (commentaires seuls,
  no-code-change verifie, ctest 96/96).
- [x] **Doc honnetete #172** : docstring DSL (paragraphe CoupledSource/add_pair) + classification des 3
  briques physics (AdvectionDiffusion/LangmuirMode/TwoFluidLinear = TEST/VALIDATION, non utilisees par
  adc_cases ; correction d'un commentaire "deprecated" faux).
- [x] **Test diocotron polaire #174** (cf. milestone ci-dessus).

**Vague 2 MERGEE (revue adversariale + ci-full) :**
- [x] **Lot B SystemFieldSolver #176** : extraction de `system.cpp` (elliptique + derive de champ) vers
  `system_field_solver.hpp` ; 1470->1155 lignes ; md5 bit-identite (5 chemins) ; ctest 133/133.
- [x] **AMR capstone (vi) #179** : sources couplees inter-especes sur la facade runtime (+k/-k meme cellule,
  average_down covered cells #169) ; conservation par-cellule+globale ; disable-and-fail. MERGE-SAFE.
- [x] **Lot B SystemStepper #180** : extraction de step/advance/step_cfl/step_adaptive/stride_due/
  run_source_stage/apply_couplings ; 1155->1044 ; md5 bit-identite MATCH exact ; ctest 134/134.
- [x] **Lot E rejets explicites #178** : tests verrouillant les rejets du chemin polaire (transport/riemann/
  imex/eps-variable) + roles DSL. A REVELE le gap role-fallback -> corrige #181.
- [x] **Fix role-fallback #181** : `add_coupled_source` (chemin DSL resolve) STRICT -> throw si le bloc
  n'expose pas le role (avant : repli silencieux comp=0). role_index + couplages nommes inchanges.
- [x] **Couverture Schur-via-System #182** : test natif `System -> run_source_stage -> Schur` (gap revue #180).

**MILESTONE SCIENTIFIQUE 2 (probe locale, PRELIMINAIRE) :** taux gamma(l) polaire (96x96, anneau creux) :
l=3 gamma~0.233 (x7.0), l=4 ~0.210 (x6.4), l=5 ~0.168 (x4.7) ; croissance exponentielle propre, positive,
l-dependante (decroit avec l), masse conservee ~1e-14 partout. QUALITATIF ; vs theorie = run ROMEO.

NOTE INFRA verifiee : les tests Python tournent en CI via un find-glob `python/tests/test_*.py` dans le job
Release (ci.yml l.148-150) ; `python/CMakeLists.txt` (foreach) est l'enregistrement ctest SEPARE (jobs
MPI/Kokkos). DETTE tracee : segfault de teardown sur profils polaires INSTABLES (PRE-EXISTANT, hors #176).

**Vague 3 MERGEE :**
- [x] **AMR runtime IMEX par bloc #184** : la facade runtime honore time=imex par bloc (source implicite
  locale, mirror du callback AmrImplicitSourceStepper) ; stiff stable, disable-and-fail (explicite ->
  non-fini) ; all-explicit/mono-bloc bit-identique. (3 findings reels NON-bloquants -> traites par #185.)
- [x] **IMEX-hardening #185** : DIVERGENCE IMEX substeps>1 documentee et assumee -- la facade runtime
  SOUS-CYCLE l'IMEX (substeps=K : K applications de dt/K, sain et plus CFL-safe) alors que le moteur
  compile-time `AmrSystemCoupler` IGNORE substeps sur sa branche IMEX (un seul pas de bdt). Les deux
  trajectoires DIFFERENT pour substeps>1 ; ce n'est PAS un mirror bit-identique a substeps>1 (claim
  "mirror fidele" corrige en commentaire). Test substeps>1 (DIFFERS load-bearing), mx/my/E observes
  directement. ctest 99/99.

**CADRAGE AMR (etat reel, a NE PAS surestimer) : l'AMR multi-bloc runtime est PHASE 1 A HIERARCHIE
FIGEE, complete** -- substeps/stride (#175), sources couplees (#179), IMEX local (#184/#185), sur UNE
hierarchie commune NON adaptative (le regrid runtime est refuse pour multi-bloc). RESTENT ABSENTS :
le **regrid union-tags** (hierarchie adaptative = Phase 2), le **DSL natif multi-bloc**, le **Schur
GLOBAL sur AMR**. Ne PAS ecrire "capstone AMR complet".

**En vol (vague parallele, branches isolees, juin 2026) :** P0 fix role-fallback AMR (mirror #181) ;
design-only regrid union-tags ; test A4 flux radial ; fix segfault teardown polaire instable ; A.5
commentaires physics/. (Plus : A.5 commentaires `amr/` deja en WIP local non commite, a preserver.)

**=== RESTE A FAIRE (synthese todo + CODEBASE_AUDIT, juin 2026) ===**

AUDIT COMPLET (juin 2026) -- TOUS les lots du plan section 10 de docs/CODEBASE_AUDIT.md sont faits ou sont des
invariants maintenus. Aucun lot d'audit restant.
- Lot A (doc + verite API) : A.1 dsl.py #172, A.2 ARCHITECTURE table #161, A.3 SourceImplicit/CondensedSchur #194,
  A.4 mentions applicatives neutralisees #159, A.5 convention commentaires TOUS dossiers (#173 core, #189 physics,
  #193 numerics, #196 mesh+coupling, #198 runtime, #200 amr).
- Lot B (runtime System) : B.1 NativeLoader #151, B.2 SystemFieldSolver #176, B.2bis SystemStepper #180,
  B.3 SystemBlockStore #197, B.4 System::Impl orchestrateur mince = resultat de B.1-B.3 (god-class P0 fermee).
- Lot C (AMR) : C.1 amr_reflux_mf decoupe #170, C.2 amr_coupler retire #164, C.3 garde layout #141, C.4 stride #140,
  C.5 decision = runtime multi-bloc, C.6 regrid union-tags #199 => CAPSTONE AMR COMPLET (Phase 1 figee + Phase 2 regrid).
- Lot D (Schur/elliptique) : D.1 EllipticOperator #163, D.2 TensorKrylovSolver = solveur PUR (prend GeometricMG& op
  en argument, ne possede pas la physique -- INVARIANT verifie), D.3 CondensedSchurSourceStepper = etage temps/couplage
  via Split/set_source_stage, PAS model.source (INVARIANT verifie + doc #194), D.4 Schur device-clean 7/7.
- Lot E (validation) : E.1 tests stride, E.2 roles #178/#181, E.3 PolarMesh erreurs #168/#178, E.4 noms backend #201,
  E.5 CI auto-decouverte verifiee (find-glob).
NB : D.2 et D.3 sont des INVARIANTS A MAINTENIR (ne pas laisser TensorKrylovSolver posseder la physique ni
CondensedSchurSourceStepper devenir un model.source), pas des taches a faire.

RESTE (audit) :
- [x] **Lot B.3 SystemBlockStore -- FAIT #197** : `class SystemBlockStore` (include/adc/runtime/system_block_
  store.hpp, 165 l) OWN le registre `std::vector<BlockState>` (ex-Species) + index/find/copy_comp0/copy_state/
  write_state ; Impl delegue. Decision CONSERVATRICE : alias `using Species=...` + `std::vector<Species>& sp=
  blocks_.blocks` pour NE PAS churner les gabarits deja extraits (SystemFieldSolver/SystemStepper/native_loader).
  Bit-identique (helpers verbatim, ordre d'insertion preserve, ordre membres == init aggregat -- verifie par
  revue adversariale 4 lentilles : 0 finding). system.cpp 1065->1015. ctest 140/140 inchange. CI full verte.
- [x] **Lot C.6 / AMR (viii) regrid union-tags -- FAIT #199 (CAPSTONE AMR COMPLET)** : `AmrRuntime::regrid()`
  pilote par l'UNION des tags (predicat par bloc D1 + |grad phi| D4) -> tag_union (OU) -> all_reduce_or (MPI) ->
  Berger-Rigoutsos -> UN layout fin partage -> prolongation de TOUS les blocs (dont stride-tenus D3) + aux ->
  same_layout_or_throw. regrid AVANT step (D2), 2 niveaux (D5). amr_regrid_finest refactore en briques
  reutilisables (mono-bloc bit-identique). Facade deverrouillee (multi-bloc+regrid_every>0 ne leve plus).
  regrid_every=0 BIT-IDENTIQUE. Test test_amr_multiblock_regrid_union (24 assertions a-e + conservation masse).
  CI full verte (dont MPI). Revue adversariale 4 lentilles : 2 findings majeurs SOULEVES puis REFUTES (phi opt-in
  = limite scope documentee, et le defaut densite capture deja le bord d'anneau = discontinuite de densite ;
  chemin all_reduce exerce par le job CI MPI). => PHASE 1 + PHASE 2 = CAPSTONE AMR COMPLET.
  SUIVI -- FAIT #205 : (1) predicat phi |grad phi| cable depuis la facade Python (AmrSystem.set_phi_refinement,
  defaut <=0 = desactive/bit-identique) ; (2) test de parite MPI np=1/2/4 du regrid (test_amr_regrid_mpi_parity).
- [x] **AMR (v) DSL production multi-bloc -- FAIT #195** : `add_compiled_model(AmrSystem&)`/`set_compiled_block`
  ne levent plus au 2e bloc compile ; file de specs + build paresseux a `ensure_built` (miroir du chemin
  natif). 1 bloc route TOUJOURS par AmrCouplerMP (mono-bloc bit-identique, dmax==0) ; N blocs via AmrRuntime.
  Footgun corrige (revue adversariale BLOCKER) : `AmrSystem.add_equation` REJETTE stride>1 / masque IMEX sur
  le chemin compile .so (ABI plate ne les transporte pas) et les FORWARDE sur le chemin ModelSpec natif.
  Tests : test_amr_multiblock_compiled (cas F = IMEX stride=2 + masque par composante), test_amr_production_
  stride_reject.py. ctest 140/140. => PHASE 1 MULTI-BLOC A HIERARCHIE FIGEE COMPLETE (substeps/stride #175,
  sources couplees #179/#191, IMEX #184/#185, multi-bloc DSL/compile #195). Reste Phase 2 = regrid (C.6).
- [x] **Lot A.5 convention de commentaires -- COMPLET** : core/ #173, physics/ #189, numerics/ #193,
  mesh/+coupling/ #196, runtime/ non-amr #198, amr/ primitives #200. (runtime/amr_*.hpp deja commentes par
  #185/#195/#199.) NOTE residuelle : 2 coquilles non-ASCII pre-existantes (abi_key.hpp "batis", system_field_
  solver.hpp "ind+ependants" = em-dash casse) a corriger en passe ASCII dediee.
- [x] **Lot A.3 -- FAIT #194** : note SourceImplicit (local, par cellule) vs CondensedSchur (global, Schur)
  dans les docstrings python/adc/__init__.py + docs/SCHUR_CONDENSATION_DESIGN.md (pas de dossier examples/
  dans adc_cpp ; les exemples runnables sont dans adc_cases). Doc-only, 28 insertions.
- [x] **Lot E.4 -- FAIT #201** : audit couverture par backend. VERDICT : la couverture par backend-path etait
  DEJA complete (tout test C++ hors-bloc-MPI tourne Serial/Kokkos-Serial/Kokkos-OpenMP/MPI-np1 ; tests MPI dans
  le job MPI ; noms deja explicites test_mpi_*/_npN). Seul trou = DOC perimee (13 C++ + 10 py manquants au
  tableau) -> docs/BACKEND_COVERAGE.md remis a jour (section capstone AMR + bilan recalcule), 0 churn de test.
  Nouveau gap #6 note : capstone multi-bloc valide 4 backends CPU, pas encore de harness ROMEO device.
- [x] **Fix role-fallback cote AMR -- FAIT #191** : `AmrRuntime::add_coupled_source` (include/adc/runtime/
  amr_runtime.hpp, lambda resolve) durci strict (miroir #181) : bloc inconnu/role non-canonique/role canonique
  non expose -> throw nommant bloc+role, plus de repli silencieux comp=0. Test tests/test_amr_coupled_source_
  role_strict.cpp (ctest #94) : role valide/absent/inconnu. Build 214/214, ctest 100/100. (verifie 06/06)

RESTE (scientifique / hors audit) :
- [x] **Run Hoffart haute resolution (ROMEO/GH200) -- n=384 ET n=512 FAITS** : build -fPIC resolu (Kokkos PIC
  /scratch_p/rmdraux/kokkos-install-pic) ; module _adc aarch64+CUDA bati (adc_gpu_hires/adc_cpp/build-gpu-py).
  O5 WENO5/SSPRK3 sur 1 GH200 (Kokkos Cuda np=1) :
    n=384 (job 644126) : l=3 -2.3%, l=4 -4.9%, l=5 +11.2%
    n=512 (job 647010) : l=3 -0.38%, l=4 -8.4%, l=5 +16.0%
  l=3 CONVERGE proprement vers le papier (-0.38% a n=512) = confirmation forte de la normalisation 2pi/rhobar.
  l=4/l=5 NON monotones (l=4 -4.9->-8.4, l=5 +11->+16) : sensibilite a la FENETRE DE FIT pour les modes rapides
  (la fenetre auto de sweep.py est longue [~5,~38] et mord dans la saturation pour les modes a forte croissance).
  SUIVI (mesure, pas physique) : re-fitter l=4/l=5 sur une fenetre exponentielle PRECOCE dediee (probe_fit.py
  existe deja sur ROMEO). Le coeur scientifique (normalisation, l=3 exact) est ETABLI.
- [x] **Schur PR6 -- MESURE (adc_cases, branche feat/normalization-and-schur-measurement)** : effet TEMPOREL
  du Schur sur un fluide magnetise CARTESIEN raide mesure. cas schur_magnetized_cartesian/ : dt stable explicite
  = 3.16e-4 (dt*omega_c=0.32, borne cyclotronique) vs CondensedSchur theta=0.5 = 1.78e-1 (gain 562x), theta=1.0
  = 3.16e-1 (gain 1000x). Le Schur retire la borne dt*omega_c<O(1) ; le pas approche le pas de transport.
  set_magnetic_field cartesien OK ; etage condense via set_source_stage (meme C++ CondensedSchurSourceStepper
  #126). (chemin polaire reste explicite-only ; "Schur polaire" = feature ulterieure). A REVOIR par le proprietaire.
- [x] **Normalisation diocotron CONSOLIDEE (adc_cases, meme branche)** : NORMALIZATION.md + diag/diag_polar_omega.py
  (gamma_norm = gamma_raw*2pi/rhobar ; l=4 exact n=128/192). Cas hoffart_euler_poisson_dsl de Codex inclus.
- [x] **Perf : Poisson MG small-box sous Kokkos -- FAIT (PR #206)** : `for_each_cell` execute en SERIE (boucle
  hote) si box < 4096 cellules SOUS backend Kokkos a execution HOTE (Serial/OpenMP) ; chemin device Cuda/GH200
  strictement INCHANGE (garde `if constexpr DefaultExecutionSpace==DefaultHostExecutionSpace`). Seuil 4096
  (surchargeable ADC_FOREACH_SERIAL_THRESHOLD). PROFIL ROMEO (Kokkos OpenMP, phase poisson ms/pas) : n=128/16th
  -91%, n=256/16th -80%, n=512/16th -55%, et 1-thread a/sous baseline partout (le MG EMPIRAIT avec les threads
  avant). BIT-IDENTIQUE PROUVE : hash FNV du phi final identique seuil=0 vs 4096 sur n=64..512 a 1/8/16 threads
  (for_each_cell sans dependance inter-iteration : GS rouge-noir colore ; reduce NON touche). 7 tests elliptiques OK.

DETTE / DIFFERES (ne pas oublier) :
- [x] segfault de teardown sur profils polaires INSTABLES (PRE-EXISTANT, hors #176) -> CORRIGE #192 :
  bug get/set_primitive_state + accesseurs (to_2d/to_3d) supposaient un domaine CARRE (n*n) ; passes a
  (ny,nx) -> polaire nr!=ntheta correct, cartesien bit-identique (ny==nx==n), AMR garde carre. CI full verte.
- [x] A4 -- FAIT #202 : test_polar_mms_vr.cpp, MMS polaire DEDIE v_r != 0 (champ (v_r=0.35, v_theta=0.6),
  source manufacturee polaire (1/r)d_r(r rho v_r)+(1/r)d_theta(rho v_theta)), ordre de convergence 2.00 (limite
  par la ponderation de face radiale). + #209 : MMS fluide 3-var (v_r != 0) ordre 2.00. (A4 TERMINE. Un MMS polaire
  TRANSITOIRE v_r!=0 reste un durcissement OPTIONNEL, NON prioritaire : ne doit pas retarder Schur polaire
  ni la reproduction scientifique.) AUDIT MMS (workflow juin 2026) : verdict COUVERTURE SUFFISANTE -- ordre-2
  GENUINE/structurel (ponderation metrique radiale r_{i+/-1/2}, casse le telescopage haut-ordre ; PAS un bug) ;
  gap transitoire deja couvert par la validation temporelle CARTESIENNE (meme integrateur SSPRK3 reutilise en
  polaire) ; AUCUN nouveau test requis (2 notes doc optionnelles non bloquantes).
- [x] **finding 8 -- FAIT #204** : `fab(0)` sans garde `local_size` corrige (6 fermetures rhs_into/max_speed/
  advance dans native_loader.hpp ; garde `if (U.local_size()==0) return` ; collectifs hors fermetures -> pas
  d'interblocage) + test MPI np=1/2/4 (test_mpi_system_gather_scatter). CI job MPI verte. RESTE : `/(2*dx)->*cx`
  au dernier bit (NON bit-identique, differe) ; P7
  (implicit-total + params runtime DSL).

**Prochaines etapes :** voir la section "RESTE A FAIRE" ci-dessus (source de verite unique). Invariants
verrouilles a NE PAS violer :
- **Schur PR6 = CARTESIEN seulement** (mesure de l'effet TEMPOREL sur un fluide magnetise raide). Le
  chemin POLAIRE est EXPLICITE-ONLY : le stage Schur n'y est PAS branche. NE PAS pretendre que "Schur
  polaire" fonctionne ; le brancher serait une feature ulterieure. (Le PAPIER Hoffart, lui, fait le
  systeme Euler-Poisson COMPLET raide avec son complement de Schur ; la pile Schur d'adc #118-128 en est
  l'analogue FV -- reproduire la METHODE du papier = un chantier separe, pas Schur PR6.)
- **BORD D'ANNEAU = discontinuite de densite TRANSPORTEE, PAS une paroi.** NE PAS poser de "paroi" de
  transport dessus (physiquement faux) ; leviers valides = polaire / haut ordre / AMR.
- **Perf : AUCUNE optimisation sans profil AVANT/APRES.** Cible identifiee #165 (Poisson MG small-box
  sous Kokkos), une optim par PR.
- **finding 8 (`fab(0)`)** : a faire AVEC le gather/scatter du portage MPI, PAS en demi-fix (un demi-fix
  transforme un crash franc en faux-silencieux). `/(2*dx)->*cx` : NON bit-identique, dernier bit. P7 differe.

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
- [x] cx (`/(2*dx)` -> `*cx`) -- CLOS NON RENTABLE (evalue juin 2026). Les 3 sites divisifs restants
      (amr_coupler_mp:307-308, spectral_coupler:113-114, system_field_solver:429-430) sont des boucles CPU
      HORS HOTSPOT (1x par solve, apres le V-cycle MG qui domine 96-99% du temps #165/#206), memory-bound,
      et la division par CONSTANTE est deja optimisable par le compilo. Casser la bit-identite IEEE-754 pour
      un gain sub-ms NON mesurable n'est pas justifie. Les sites GPU (field_postprocess, condensed_schur,
      schur_condensation) sont DEJA en `*cx`. Decision : laisser les 3 sites en l'etat. Pas de PR.

## 3. Durcissement de l'architecture (`docs/archive/ROADMAP.md`, ARCHIVE -- plus une doc active)

- [x] **Moteur AMR unifie** : `advance_amr(LevelHierarchy&)`, `FluxRegister`, `CoverageMask`,
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
- [x] **Gap 3 - multi-box natif** + **Gap 4 - multi-espece (capstone)** : FUSIONNES dans le capstone AMR
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
- [x] **B_z multi-box distribue multi-GPU (#59)** : FONCTIONNEL sur device (B_z par boite/niveau lu,
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
- [ ] **Perf full-device -- RECHERCHE/BESOIN-ROMEO (scope juin 2026, cf docs/RESEARCH_BACKLOG.md)** : la
      replication grossiere est un CHOIX (le mode distribue mesure 3-5x PLUS LENT : MG halo latency-bound sur
      petites boites). Prochain pas = profil ROMEO multi-GPU par niveau de V-cycle ; DECISION GATE : latence
      >50% -> MG hybride ; sinon replication = bon compromis -> clore. NON auto-completable. [le run integre NE SCALE PAS (grossier REPLIQUE -> Poisson/transport
      grossier redondants par GPU ; seuls les patchs fins se repartissent). Mode `replicated_coarse=false`
      (grossier reparti) existe dans `AmrCouplerMP` mais degrade le MG et n'est pas cable dans `AmrSystem` :
      vrai chantier strong-scaling AMR. + parite AOT zero-copie sur device (sans rebond hote).

## 5. Physique magnetisee

- [x] Push de Boris E+B combine (`tfap_boris`, cyclotron exact, derive ExB sans croissance seculaire).
- [ ] **Reformulation AP tensorielle sous champ fort -- RECHERCHE (scope juin 2026, cf docs/RESEARCH_BACKLOG.md)**
      : le Schur condense gere DEJA le champ fort inconditionnellement (stable) ; l'AP serait un gain
      d'EFFICACITE seulement. Prochain pas = etude math (expansion asymptotique + toy 1D), pas de code. NO-GO
      si operateur non-local incompatible roles/DSL. NON auto-completable.

## 6. Reproduction Hoffart (arXiv:2510.11808) - APPLICATIF, cote `adc_cases`

- [x] **M1 -- NORMALISATION TROUVEE (juin 2026)** : `gamma_norm = gamma_raw * 2pi/rhobar` (facteur du
      projet). Verifie sur le chemin POLAIRE ExB (echelle papier 6:8:16, top-hat, WENO5/SSPRK3, n=128) :
      l=4 EXACT (0.913 vs 0.911), l=3 +26% (0.97), l=5 -29% (0.48). Le scatter +-29% est la sensibilite a
      la fenetre de fit (intrinseque a une mesure numerique de taux diocotron, cf. sweep cartesien projet).
      DIAGNOSTIC CLEF : `gamma_raw(sim) ~ Im(omega)_eigenmode` directement (0.155 vs 0.123) -> le sim
      polaire tourne en unites de temps ExB-naturelles, AUCUN re-scaling beta necessaire. La rotation au
      bord interne r0 est ~0 (pas de charge enfermee dans r<r0) -> une normalisation "par rotation locale"
      ne marche PAS ; c'est bien le facteur GLOBAL 2pi/rhobar. POURQUOI Codex (cartesien-Schur) donne 0.035
      et pas 0.77 : (a) son runner OMET le facteur 2pi/rhobar (x2pi -> 0.22) ET (b) sa grille CARTESIENNE
      diffuse le bord d'anneau (gamma_raw polaire 0.155 ~ 4.4x son 0.035 a resolution comparable). Polaire
      + 2pi/rhobar reproduit le papier ; pas un bug de physique. ROBUSTE EN RESOLUTION : a n=192 les
      trois modes tombent dans [0.87,0.97] (l=4 EXACT aux deux resolutions ; l=5 passe de -29% n=128 a
      +27% n=192 quand sa fenetre se resserre -> le scatter est de la SENSIBILITE A LA FENETRE de fit,
      PAS un deficit de physique). Diag reproductible : `/tmp/diag_polar_omega.py`.
- [x] **MODELE COMPLET = DEJA OPERATIONNEL (chemin cartesien, juin 2026, investigation multi-agents)** :
      adc_cases/hoffart_euler_poisson_dsl tourne le systeme COMPLET (continuite + momentum + Lorentz +
      pression isotherme p=theta*rho + Gauss) via adc.Split(Explicit ssprk3, CondensedSchur) = pile Schur
      #118-128 sur grille CARTESIENNE. l=3 = -0.38% n=512 GH200. L'observable de taux (FFT-theta de phi sur
      un cercle) est polaire-propre MEME en cartesien (la diffusion de bord ne touche que le rendu de
      DENSITE, pas le taux). => VOIE B (cartesien) RECOMMANDEE ; Voie A (fluide polaire + Schur polaire) =
      RECHERCHE, optionnelle (PolarPoissonSolver direct scalaire incompatible avec le Schur tenseur croise ;
      dispatch_transport_polar rejette le fluide). Roadmap : docs/FULL_MODEL_VALIDATION_ROADMAP.md.
- [x] **Conservation discrete cartesien-fluide-Schur -- FAIT #207** : tests masse (machine, domaine ferme),
      symetrie momentum (machine), impulsion momentum (physique O(dt) convergente), E>0/p>0 (3 limiteurs).
      Note honnete FV-vs-FE. Decouverte : Dirichlet fuit la masse ~1e-2 par Foextrap (artefact CL, pas schema).
- [~] **RESTE Hoffart -- DECISIONS PRISES + EN COURS (juin 2026)** : le proprietaire a tranche : cible l=4/5
      a +-2% (=> re-fit + rejouer), figure 2D nette REQUISE (=> Voie A poursuivie), tout en parallele.
      EN VOL :
        - #208 doc CONSERVATION_SUMMARY (FV vs FE) -- CI.
        - #209 Voie A ETAPE 1 MERGEE : transport fluide isotherme polaire (IsothermalFluxPolar + source geometrique
          centrifuge S_geom verifiee EXACTE par derivation sympy independante ; ExB-polaire + cartesien
          bit-identiques via concept-gate ; equilibre rotatif ordre 1.99, MMS 2.00, masse 1.2e-15). Revue
          adversariale 4 lentilles = 0 vrai probleme (6 "findings" = confirmations positives). -- CI verte, MERGEE (675e587).
        - re-fit fenetre precoce l=4/l=5 sur ROMEO (job 647356) -- RESULTAT HONNETE : la fenetre precoce
          N'ATTEINT PAS +-2% et n'est PAS un fit-window pur. n=512 : l=3 +4.86% (la fenetre LONGUE etait
          meilleure : -0.38%), l=4 +5.09% (mieux que -8.4%), l=5 +18.10% (~ inchange vs +16%). Les modes
          preferent des fenetres DIFFERENTES ; l=5 reste +16-18% quelle que soit la fenetre => DEVIATION
          REELLE (resolution insuffisante pour le mode rapide a fort gradient, et/ou perturbation delta=0.1
          legerement non-lineaire), PAS un artefact de mesure. CONCLUSION : la normalisation 2pi/rhobar est
          PROUVEE par l=3 (-0.38% fenetre longue) ; +-2% sur l=4/l=5 exigerait n=768/1024 (plus de GPU)
          et/ou delta plus petit (regime plus lineaire). Donnees : /scratch_p/rmdraux/early_fit_647356/.
      ETAT HONNETE DES TAUX (a ne pas survendre) : l=3 VALIDE la normalisation (-0.38% n=512) ; l=4 SENSIBLE
      A LA FENETRE de fit ; l=5 DEVIATION ROBUSTE 16-18% quelle que soit la fenetre ; CAUSE PAS ENCORE
      IDENTIFIEE (non-linearite ? resolution/diffusion ? modele/geometrie/diagnostic ?). PAS de promesse +-2%
      avant resultats.
      CAMPAGNE DISCRIMINANTE FAITE (job ROMEO 647366, 12 runs) -- VERDICT : NON-LINEARITE dominante. Tendance
      robuste : gamma_num remonte vers l'analytique quand delta diminue (l=5 n=512 : delta=0.10 -71%, 0.05 -37%,
      delta=0.025 -> +0.17% QUASI-EXACT ; l=4 n=512 : -64/-39/-22%). => la deviation l=5 +16-18% etait un
      ARTEFACT NON-LINEAIRE (delta=0.10 trop grand). Levier = delta plus petit, PAS n=768 (resolution pas le
      verrou). CAVEAT METHODE : la campagne a delta=0.10 (l=5 n=512=0.20, fenetre longue dans la saturation) NE
      CORRESPOND PAS au sweep original (0.81) -> la campagne n'a pas pris la MEME fenetre ; la tendance delta est
      valide mais la comparaison absolue campagne<->original est faussee par la fenetre. RESTE : (1) reconcilier
      la fenetre (refaire avec l'observable/fenetre EXACTE de l'original) ; (2) investiguer le residu l=4 (-22%
      a delta=0.025, tendance n inversee). PAS de promesse +-2%. Donnees : /scratch_p/rmdraux/647366/. [ancienne note :] modes l=4,5 x n=256,512 x
      delta=0.10/0.05/0.025, MEMES fenetres + MEME observable (aucun ajustement opportuniste). Lecture :
      erreur DIMINUE avec delta -> non-linearite ; DIMINUE avec n -> resolution/diffusion ; STABLE ->
      probleme de modele/geometrie/diagnostic. n=768 ENSUITE, seulement pour les configs qui le justifient.
      SUITE : Voie A ETAPE 2a = operateur elliptique polaire TENSORIEL iteratif MERGE #210
      (BiCGStab + precond RadialLine, MMS ordre 2, revue clean) ; ETAPE 2b = etage source Schur polaire MERGE #212 (gain dt 2000x) ; ETAPE 2c = cablage facade -> PR6 (elliptique tenseur croise +
      stencils Schur polaires) ; cas demonstrateur diocotron fluide polaire (la figure 2D nette) ; table de
      validation finale ; (optionnel) Strang ordre 2. Roadmap detaillee : docs/FULL_MODEL_VALIDATION_ROADMAP.md.
      Question ouverte restante : preuve structure-preservation FE formelle requise, ou tests empiriques O(dt^2)
      suffisent (le schema est FV, momentum non exact par construction) ?
- [x] **M2 / M2b** : AMR sur le bord d'anneau (triple le taux a base egale) + Poisson multi-niveau.
- [~] Montee en resolution / convergence vers le taux analytique : FAITE (n=384/512 GH200, l=3 -0.38%).
      Integration SAMRAI = EXTERNE-GROS, DIFFEREE (l'AMR maison d'adc est capstone-complet et couvre le chemin
      science ; cf docs/RESEARCH_BACKLOG.md pour le critere de reouverture). NON auto-completable.

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
- [x] **Etape 6 - validation MPI/GPU du chemin `production`** : **np=1 GPU VALIDE sur GH200.** Le crash
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
- [~] **Etape 7 - domaine disque FV / Polaire Phase 2b** (EN COURS) : Voie A etape 1 = transport fluide
      polaire MERGE #209 ; etape 2a = operateur elliptique polaire TENSORIEL iteratif MERGE #210 (MMS ordre 2, BiCGStab+RadialLine, revue adversariale
      clean) ; etape 2b = etage source Schur polaire MERGE #212 (assemblage A polaire via #210, reconstruct
      polaire, gain dt STABLE jusqu'a 2000x, bit-identite Schur cartesien). RESTE etape 2c = CABLAGE FACADE
      (System.step/Python du stepper #212) -> PR6 (mesure diocotron-Schur polaire raide). Demonstrateur 2D
      explicit FAIT (#feat/diocotron-polar-fluid) :
      DEMONSTRATEUR 2D POLAIRE FAIT (adc_cases branche feat/diocotron-polar-fluid, build+run ROMEO) : fluide
      isotherme polaire, mode l=4 croit (x14.9, gamma_fit 0.589), masse machine (4.6e-14), BORD D'ANNEAU NET
      (4 lobes sharp, zero diffusion isotrope vs cartesien). Param cle = cs2 (0.1 instable / 0.5 relaxe).
      LIMITE honnete : pas de Lorentz magnetique NATIF sur le chemin polaire 3-var (dispatch_source<3> = none/
      potential/gravity ; pas de brique v x B_z polaire) -> derive electrostatique + centrifuge utilisee ;
      brique source magnetique polaire = raffinement futur. (Doc perimee a corriger : docstring PolarMesh dit
      encore 'ExB scalaire seulement', faux depuis #209.)
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
- [x] **Confirmation haute resolution n=384 / n=512 (incl. O5) sur ROMEO/GH200** AVANT toute reecriture
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
- [x] **Device-clean GPU** : le stack Schur etait FAUX EN SILENCE sur Cuda (accesseurs Geometry/Box2D
      non `ADC_HD` -> RHS/reductions faux, BiCGStab "0 iters rel=0 puis NaN"). Fix dans #135 (en vol,
      attend validation GH200). Host = correct, CI verte.
- [ ] **PR6 - mesure diocotron-Schur** : DIFFERE jusqu'a la geometrie polaire (le Schur stabilise le
      TEMPS ; il n'adresse PAS le gap de taux de croissance, qui est GEOMETRIQUE).
- [x] **PR7/PR8 - portage GPU/MPI + AMR** : apres #135 ; bit-invariance au nombre de rangs.

## 11. Pipeline ergonomique System (P1-P7)

- [x] **P1 - stride par bloc** : `Explicit/IMEX(stride=M)` (hold-then-catch-up), CFL
      `dt=cfl*h*substeps/(stride*w)`. (#121) Honnetete du wording -> finding revue 3 (#138).
- [x] **P2 - clarifier Implicit/IMEX** : renommage `SourceImplicit`, deprecation de `Implicit`. (#123)
- [x] **P3 - masque `implicit_vars`** sur la politique temporelle / le bloc (PAS sur le modele). (#125)
- [x] **P4 - `set/get_primitive_state`** (init/diagnostic en variables primitives). (#127)
- [x] **P5 - CoupledSource DSL** : `adc.dsl.CoupledSource` -> bytecode postfixe interprete par un
      foncteur device nomme dans `apply_couplings` ; generique inter-especes ; 3 especes + conservation,
      MPI np=1/2/4 bit-identique. (#131)
- [x] **P6 - AMR multi-blocs** = capstone Gap 4, voir section 15.
- [~] **P7** : P7-b params runtime DSL = FAIT #213 (m.param(kind=runtime) + System.set_block_params ; change un param sans recompiler le .so ; byte-identite des params const confirmee) ; P7-a implicit-total =
      RECHERCHE (schema totalement implicite, gros chantier ; cf docs/RESEARCH_BACKLOG.md), differe sauf si le
      splitting Lie ordre 1 devient limitant mesure (sinon Strang ordre 2 suffit).

## 12. Geometrie polaire / annulaire (le vrai verrou du taux de croissance diocotron)

Conclusion scientifique : le verrou du taux de croissance est l'advection CARTESIENNE du gradient
radial de l'anneau ; la geometrie POLAIRE le preserve (proto : ratio 73 vs 1.0 cartesien). Le Schur
n'adresse PAS ce gap (il stabilise le temps). Donc le levier = mettre la geometrie polaire.

- [x] **Abstraction Mesh** : `adc.PolarMesh` / `adc.CartesianMesh` -> `System(mesh=)` (PAS
      `FiniteVolume(geometry=)`). `PolarGeometry`, divergence polaire `(1/r)d_r(r F_r)+(1/r)d_th F_th`,
      `ExBVelocityPolar` `v_r=-(1/(Br))d_th phi`, `v_th=(1/B)d_r phi`. (#116)
- [x] **Phase 2a - Poisson polaire direct** sur anneau : FFT-en-theta + tridiagonal-en-r (robuste,
      evite le MG-en-polaire qui stagne). (#130)
- [x] **Device** : flux/metrique polaire FAUX en silence sur Cuda (meme cause #135 : `r_cell`/`theta_cell`
      non `ADC_HD`). Fix #135 (en vol).
- [x] **Phase 2b (#168, FAIT)** : transport + Poisson polaire branches dans `System.step` sur un anneau
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
- [x] **F3 - step_cfl substeps>1 non bit-identique** : DECISION = garder la CFL substeps-aware
      (correcte), corriger le wording + tests. En vol #138.
- [x] **F4 - `evolve=False` ignore en silence sur prototype/aot** : rejet explicite. FAIT #137.
- [x] **F5 - `adc.Split` perdu en silence sur AmrSystem** : rejet explicite. FAIT #137.
- [x] **F6 - descripteurs `CondensedSchur` jamais transmis** (code mort API) : rejet non-defaut. FAIT #137.
- [x] **F7 - `max_wave_speed_mf` non collectif** (dt divergent par rang MPI) : `all_reduce_max`. Dans #135.
- [x] **F8 - `fab(0)` sans garde `local_size()`** dans le marshaling : FAIT #204 (garde local_size posee ; pas de demi-fix car un demi-fix
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
- [x] **(iii)** test de VALIDATION Poisson somme co-localise [trivial, validation seule ; le moteur le fait deja].
- [x] **(iv) #175** substeps/stride par bloc + step_cfl substeps-aware : `AmrRuntime::step` honore
      substeps/stride (hold-then-catch-up, mirror `AmrSystemCoupler::step`) ; `AmrSystem::step_cfl` =
      `cfl*h*min_b(substeps_b/(stride_b*w_b))`. Mono-bloc bit-identique (routage AmrCouplerMP). MERGE.
- [x] **(v)** DSL production multi-bloc : `add_native_block`/`add_compiled_model(AmrSystem&)` ne doit plus
      lever sur le 2e bloc (file d'attente + build a `ensure_built`). Write-set amr_dsl_block.hpp + amr_system.cpp.
- [x] **(vi) #179** sources couplees AMR meme-cellule / opposees : `coupled_source_step` via le registre
      runtime + average_down covered cells (#169) ; conservation par-cellule+globale, disable-and-fail. MERGE.
- [x] **(vii) FAIT (#184/#185)** IMEX local AMR runtime : la facade honore `time="imex"` multi-bloc (le moteur a deja
      le callback AmrImplicitSourceStepper). Agent en cours.
- [x] **(viii)** Phase 2 : regrid union-tags (deverrouille multi-bloc + regrid_every>0) ; puis Schur / implicite
      global / repro papier.

Tests d'acceptation : 2 blocs explicites schemas differents ; e- IMEX(substeps=10) + ions
Explicit(substeps=1) ET l'inverse ; neutres stride=20 nourrissant sources/Poisson ; evolve=False fond
fixe dans le RHS elliptique ; regrid conserve la masse par bloc ; DSL production 2 blocs ; MPI np=1/2/4 ;
Kokkos Serial ; 1 cas multi-bloc production sur GH200.

## 16. Findings conservation (revue)

- [x] **A3 [BUG conservation] FAIT #169** : `AmrSystemCoupler::coupled_source_step` et `AmrImplicitSourceStepper` appliquaient la source aux cellules grossieres COUVERTES sans average_down trailing. Fix : cascade `mf_average_down_mb` fin->grossier apres la boucle de niveaux (no-op strict mono-niveau, bit-identique). Tests `test_amr_source_covered_cells` + `test_amr_composite_source_conservation` (discriminants : echouent sans le fix). Revu adversarialement (MERGE-SAFE).
- [x] **A2 [RISK conservation] FAIT #167** : helper `add_pair(block_a, block_b, role, expr)` emet +expr / -expr depuis le MEME sous-arbre (conservatif par construction) + mode `compile(verify_conservation=True)` opt-in. Test `test_dsl_coupled_source_conservation`. Revu adversarialement (MERGE-SAFE).
- [x] **A4 [GAP test]** : couvert en pratique par les tests #168 (conservation masse polaire avec flux radial INTERIEUR non nul, profil module en theta -> v_r != 0). RESTE optionnel : un cas MMS polaire dedie avec v_r != 0 analytique (au-dela de la conservation globale).
- Tests a ajouter : `test_dsl_coupled_source_conservation`, `test_amr_composite_source_conservation`, `test_amr_source_covered_cells`, `test_polar_conservation_radial_flux`.

# adc — Tâches : du `PhysicalModel` au système multi-espèces

Liste unique et actionnable de ce qui est **fait** et de ce qui **reste**, d'après la
discussion tuteur et les incréments. Pas un solveur de production : on pose un **squelette
testable**. `[x]` = fait et vert ; `[ ]` = à faire.

---

## 0. Posture (cadre du tuteur)

- Squelette d'abord, pas le solveur final ; tester le niveau d'abstraction sur des cas simples.
- Abstraction et architecture **avant** la structure de données (elle se change plus tard).
- Performance **après** stabilisation (« j'optimiserai quand le code sera propre et figé »).
- Validé **par un utilisateur** (Sacha peut-il décrire son cas sans toucher AMR/MPI/GPU ?), pas par le compilateur.
- `PhysicalModel` est validé : on **ajoute** le niveau au-dessus, on ne le remplace pas.
- Diffusion = un flux de plus, pas une nouvelle couche.
- Deux dépôts : `adc_cpp` = cœur générique, `adc_cases` = modèles / façades / Python / exemples.

---

## 1. Fait (état actuel, tout vert)

### Architecture / dépôts
- [x] Split cœur/applications : `adc_cpp` (moteur, zéro modèle) ↔ `adc_cases` (modèles, façades, Python) via FetchContent (`adc::adc`). Poussés.
- [x] `adc_cpp` HEAD ne contient **aucune** application (vérifié) ; README recentré « bibliothèque cœur ».

### Abstractions par bloc (le niveau manquant — squelette en place)
- [x] `core/equation_block.hpp` : `EquationBlock<Model, Spatial, Time>` = state + modèle + discrétisation spatiale + politique temps + **BC par bloc**.
- [x] `core/coupled_system.hpp` : `CoupledSystem<Blocks...>` (variadic), `for_each_block`, `block<I>()`. **→ design data = `tuple<Blocks...>` (MultiFab par bloc) retenu pour le squelette.**
- [x] `integrator/time_integrator.hpp` : `TimePolicy<Method, Treatment, Substeps>` + `ExplicitTime` / `ImplicitTime` / `IMEXTime` / `PrescribedTime` ; tags `SSPRK2`/`SSPRK3` ; `UserTimeIntegrator`.
- [x] `integrator/scheduler.hpp` : `advance_subcycled(system, dt, advance_block)` lit les sous-pas par bloc.
- [x] `coupling/elliptic_rhs.hpp` : `SingleModelEllipticRhs` (mono) + `TwoFieldChargeDensityRhs` / `TwoBlockChargeDensityRhs` (RHS = q0·n0 + q1·n1, lit plusieurs blocs).
- [x] `coupling/system_coupler.hpp` : `SystemCoupler` mono-niveau — RHS elliptique global, blocs explicites avancés par le cœur, **blocs implicites/IMEX délégués à un callback** (point de branchement Newton/linéaire/collisions), sous-pas par bloc.
- [x] Tests `test_system_abstraction`, `test_system_coupler` (électrons implicite délégué + ions SSPRK3, sous-pas, RHS système) verts.

### Briques sélectionnables
- [x] `SpatialDiscretisation<Limiter, NumericalFlux>` + `Coupler::step<Disc, TimePolicy>` (flux + intégrateur + sous-cyclage) sur grille uniforme.
- [x] `SpatialDiscretisation` câblée en **AMR** (`AmrCoupler/MP::step<Disc>`, MUSCL conservatif avec 2 ghosts).
- [x] Solveur elliptique choisi à l'exécution (façade diocotron MG/FFT, pattern `variant`).

### Corrections
- [x] **AMR applique `model.source`** (le chemin AMR l'ignorait — bug) + test `test_amr_source`.
- [x] Diffusion = flux : trait `DiffusiveModel` (`+ν∆U` dans `assemble_rhs`) + modèle `AdvectionDiffusion` (grille uniforme).
- [x] `SpectralCoupler` appelle `model.elliptic_rhs` au lieu de coder `alpha*(u−n_i0)` du diocotron en dur.

---

## 2. À faire — compléter le squelette multi-espèces

**Ordre conseillé** : (1) `ChargeDensityRhs` N-blocs ✅ → (2) `CoupledSource` ✅ → (3) interface
implicite `ImplicitBlockStepper` + défaut ✅ → (4) **exemple C++ minimal** (électrons
implicites + ions explicites, rhs = n_i − n_e, sans Python) ✅ → (5) `AmrSystemCoupler` ✅ →
(6) API Python dans `adc_cases` ✅. Chaque étape validée par un test d'intégration (tous verts :
adc_cpp 38/38, adc_cases 47/47 + bindings Python). **§2 et §3 entièrement faits** ; **§4** : 4.1
(diffusion AMR) et 4.2 (contrat `Aux`) faits, 4.3 (max_drift_speed, change le dt) et 4.4 (dédup
diagnostics) reportés avec justification. Reste surtout des **raffinements** (composition Python
générique, modèle utilisateur C++ exposé à Python, perf).

### 2.1 Couplage et RHS
- [x] **`ChargeDensityRhs` à N blocs** : `f = Σ_s q_s n_s` (somme sur `for_each_block`),
  **charge/composante/signe configurables** par espèce (`SpeciesCharge` + `add_scaled_component`),
  dans `coupling/elliptic_rhs.hpp`. `TwoBlockChargeDensityRhs` conservé (compat). Testé
  (`test_two_species_minimal`, `test_amr_system_coupler`).
- [x] **`CoupledSource` (sources inter-espèces)** : `coupling/coupled_source.hpp` — concept
  `CoupledSourceFor`, `NoCoupledSource`, et `SystemCoupler::coupled_source_step(src, dt)`
  (splitting forward-Euler) qui lit **plusieurs blocs + aux**. Distinct de `model.source`
  (local). Testé (`test_coupled_source` : échange linéaire conservatif).
- [x] **BC par bloc réellement appliquées** : `SystemCoupler::stage_rhs` remplit les halos
  avec `block.bc` (pas une BC globale). Vérifié par `test_system_two_explicit` (périodique vs
  outflow → champs divergents, même donnée initiale).

### 2.2 Temps : implicite / IMEX réellement exécutés
- [x] **Vraie interface implicite + défaut** : `integrator/implicit_stepper.hpp` — concept
  `ImplicitBlockStepper`, `backward_euler_source` (Newton local, jacobienne par différences
  finies : **inconditionnellement stable**, exact en 1 itération pour une relaxation linéaire,
  là où Picard divergerait dès `dt·raideur > 1`), et `ImplicitSourceStepper` (défaut prêt à
  l'emploi, **aucun Newton côté utilisateur**). Testé (`test_two_species_minimal` : `dt·k=100`).
- [x] **IMEX partiel** : trait `Model::is_implicit(c)` (concept `PartiallyImplicitModel`).
  `backward_euler_source` fait du forward-backward Euler : variables explicites en Euler avant,
  variables implicites par Newton sur le **sous-système réduit** (`solve_dense` de taille
  `m_impl ≤ n_vars`). Sans le trait → tout implicite (compat). Testé (`test_imex_partial` :
  var raide implicite bornée, var douce explicite, le masque change bien le traitement).
- [x] **Sous-cyclage temporel par espèce + cadence φ** : `substeps` exécuté (mono-niveau **et**
  AMR). La cadence de re-résolution de φ est désormais **explicite** : `PoissonCadence`
  (`OncePerStep` défaut / `PerSubstep`) dans `AmrSystemCoupler` ; le `SystemCoupler` mono-niveau
  re-résout déjà φ à chaque étage RK. Testé (`test_amr_system_coupler` part C : 1 vs 4 solves).

### 2.3 AMR pour le système : `AmrSystemCoupler`
- [x] `coupling/amr_system_coupler.hpp` : porte `CoupledSystem` sur AMR. Chaque bloc a **sa**
  hiérarchie (`std::vector<AmrLevelMP>`), toutes les espèces **partagent** la grille AMR,
  l'aux (φ, ∇φ) et le **Poisson grossier de système** (`f = Σ_s q_s n_s`). Orchestration :
  `sync_down` (par bloc) → Poisson grossier → aux + injection fine → chaque bloc avancé par
  `advance_amr<Disc_bloc>` (sous-cyclage Berger-Oliger + **reflux** + **average_down**), avec
  ses **sous-pas d'espèce** ; implicites/IMEX délégués (défaut `AmrImplicitSourceStepper`,
  backward-Euler par niveau). Testé (`test_amr_system_coupler` : conservation par bloc, RHS
  système, φ non nul, relaxation implicite grossier+fin). Réutilise le moteur `advance_amr`
  et les primitives `_mb` ; mono-box par niveau validé (comme `AmrCoupler`).

### 2.4 Cas de validation (squelette testable)
- [x] **Exemple C++ minimal SANS Python** : `test_two_species_minimal` — électrons implicites
  (relaxation raide) + ions explicites + `rhs Poisson = n_i − n_e` via `ChargeDensityRhs` +
  `ImplicitSourceStepper`. Le test « un utilisateur peut-il composer son cas ? » : il
  n'assemble que des briques (modèle, schéma, politique temps, charge), aucun solveur écrit.
- [x] **électrons Euler + ions Euler isothermes + Poisson** (cas canonique deux-espèces) —
  `adc_cases` : modèles `ChargedEuler` (4 var) + `ChargedEulerIsothermal` (3 var) dans
  `model/charged_fluid.hpp`, composés en blocs hétérogènes (`test_two_species_euler`).
- [x] **diocotron à ions mobiles** — `adc_cases/test_diocotron_mobile_ions` : réutilise le
  modèle `Diocotron` pour 2 blocs, `n_i0` figé devient un bloc transporté, Poisson `α(n_e − n_i)`.
- [x] Garde : masse conservée par bloc (`test_amr_system_coupler`), RHS = q_i n_i + q_e n_e
  correct (`test_two_species_minimal`, `test_amr_system_coupler`). Comparaison à une référence
  analytique (backward-Euler exact, production exacte).

### 2.5 Tests d'intégration à ajouter
- [x] `CoupledSystem` + **Poisson RHS non nul** (`ChargeDensityRhs`) — `test_two_species_minimal`.
- [x] `SystemCoupler` avec **deux blocs explicites différents** (schémas/sous-pas distincts) — `test_system_two_explicit`.
- [x] **Bloc implicite** branché et durci (relaxation raide `dt·k=100`, stabilité inconditionnelle) — `test_two_species_minimal`.
- [x] `AmrSystemCoupler` : conservation par bloc, reflux, RHS système, relaxation implicite multi-niveau — `test_amr_system_coupler`.

---

## 3. À faire — API Python de composition (dans `adc_cases`, APRÈS l'exemple C++)

But : Python **compose**, ne calcule pas cellule par cellule. Les chaînes sélectionnent
des briques C++ compilées. À ne faire qu'**après** l'exemple C++ minimal (§2.4) qui valide
l'architecture utilisateur.

- [x] **Modèles physiques prêts à composer** (dans `adc_cases`) : `ChargedEuler`,
  `ChargedEulerIsothermal`, `Diocotron`, `Euler`… servent de blocs (`EquationBlock`).
- [x] Façade compilée `MultiSpeciesSolver` (PIMPL) instanciant un `CoupledSystem` +
  `SystemCoupler` (`adc_cases/solver/multispecies_solver.{hpp,cpp}`), comme les autres façades.
  Cas concret deux fluides ; la composition entièrement générique `vector<SpeciesConfig>`
  (espèces arbitraires) reste un cap ultérieur (explosion combinatoire / type erasure).
- [x] Bindings pybind11 : `adc.MultiSpeciesConfig`, `adc.MultiSpeciesSolver`
  (`step/advance/density_e/density_i/potential/mass_e/mass_i/max_charge`), champs en numpy
  (`python/bindings.cpp`, testé `python/test_bindings.py`).
- [x] **Composition à l'exécution** : `adc.Simulation` (`solver/simulation.{hpp,cpp}`) —
  `add_species(name, charge)` ajoute N espèces à la volée, partageant un Poisson de système ;
  `set_density` (numpy), `step/advance`, `density/potential/mass`. Esprit `sim.add_equation(...)`
  du TODO, borné aux espèces de dérive (Diocotron, 1 var, CI simples) ; physique compilée,
  pas de callback Python dans le hot path. Testé (`test_simulation`, `test_bindings.py`).
  Étendre à Euler / IMEX = même patron + une CI par modèle.
- [x] `model`/`flux`/`time` ↔ tags C++ : la physique est en C++ compilé (schémas et politiques
  fixés à la compilation), aucun callback Python dans le hot path.
- [x] (avancé) exposer un `PhysicalModel` C++ utilisateur, composable depuis Python : point
  d'extension = le dispatch par tag de `Simulation::add_species(name, model, charge)`
  (`adc_cases/src/simulation.cpp`). L'utilisateur écrit son modèle (header), ajoute un tag +
  une fermeture d'avancée, recompile, compose depuis Python — physique compilée, pas de
  callback dans le hot path. Plugin runtime sans recompiler = choix de design (futur).

---

## 4. À faire — follow-ups cœur (nettoyages / cohérence)

- [x] **Diffusion sur AMR** : portée comme **flux de face diffusif** `−ν(u_R−u_L)/h` dans
  `compute_face_fluxes` (reçoit `dx/dy` du niveau), gardée par `DiffusiveModel` (chemin
  hyperbolique strictement bit-identique). Vue par le reflux → conservative aux interfaces
  coarse-fine. Testée (`test_amr_diffusion` : masse conservée à 1e-12 + lissage).
- [x] **Contrat `Aux`** : tranché — `Aux` est **fixe** (`adc::Aux`). Le concept `PhysicalModel`
  exige désormais `std::same_as<typename M::Aux, Aux>` : il promet exactement ce que
  `load_aux` fournit (généraliser à un `Model::Aux` quelconque reste possible plus tard).
- [x] **`SpectralCoupler::max_drift_speed`** : généralisé via `model.max_wave_speed` (plus de
  `/model_.B0` codé en dur → coupleur spectral non lié au diocotron). ⚠️ **Change le `dt`
  CFL** du diocotron (`max(|gx|,|gy|)/B0` par direction au lieu de `hypot(gx,gy)/B0`) : non
  bit-identique sur la trajectoire — **à re-valider côté physique** (croissance diocotron).
  Les tests MPI (identité inter-rangs) restent verts.
- [x] **Dédup diagnostics** : `amr_diagnostics.hpp` porte désormais l'implémentation **unique**
  multi-box (`amr_mass_mb`, `amr_max_drift_speed_mb`) ; les variantes mono-box s'y ramènent
  (cas dégénéré 1 fab, bit-identique) et `AmrCouplerMP::mass()`/`max_drift_speed()` les
  appellent (+ `all_reduce` selon l'ownership) au lieu de réimplémenter les boucles. Vérifié
  bit-identique (adc_cpp 38/38, adc_cases 48/48 dont diocotron AMR).
- [x] **`Coupler` fourre-tout** : l'orchestrateur est extrait — `SystemAssembler` (assemble) +
  `SystemDriver` (avance), `SystemCoupler`/`SystemDriver` aliases. Le `Coupler` legacy mono-
  modèle garde son nom historique (diocotron validé) ; le renommer globalement = churn pur,
  laissé tel quel (décision tableau).

---

## 5. Décision à acter au tableau (avec Sacha)

- **Structure de données du système** : le squelette a retenu `tuple<Blocks...>` (un
  `MultiFab` par bloc, `EquationBlock::state` = pointeur). Alternative : `StateVec<N_total>`
  empilé (un bloc mémoire contigu, offsets par espèce — meilleure localité, plus complexe).
  À confirmer/infléchir : c'est la décision qui pèse sur la perf (point 0 : structure de
  données = plus tard, mais la décision d'interface se prend maintenant).
  - **Retour d'expérience du remplissage** : `tuple<Blocks...>` a tenu sans friction pour
    tous les cas livrés, y compris des blocs **hétérogènes** (Euler 4 var + isotherme 3 var :
    un `StateVec<N_total>` empilé imposerait des espèces de même taille ou un layout AoS
    irrégulier). Le `for_each_block`/`block<I>()` reste l'interface ; la structure interne
    (un `MultiFab` par bloc) peut basculer en empilé plus tard **sans changer l'API** des
    coupleurs. Recommandation : **garder `tuple` par bloc** comme défaut, mesurer avant
    d'empiler ; n'empiler que si un cas homogène (mêmes `n_vars`) devient le goulot perf.

---

## 6. Lexique (pour les slides — définir avant de jeter les sigles)

| Terme | Sens court |
|---|---|
| `BoxArray` | découpage du domaine en blocs (boîtes) |
| `MultiFab` | les champs `U` stockés sur ces blocs (collection distribuée) |
| `BCRec` | conditions aux limites d'un champ |
| `aux` | variable auxiliaire transportée (`phi, grad phi`) |
| seam | couture où vit le parallélisme (`for_each_cell`, `comm`) |
| `EquationBlock` | state + modèle + méthode spatiale + politique temps + BC (un bloc) |
| `CoupledSystem` | plusieurs `EquationBlock` |
| `SystemCoupler` | orchestrateur mono-niveau : RHS elliptique global + avance chaque bloc |
| `AmrSystemCoupler` | le `SystemCoupler` porté sur AMR (Poisson grossier + reflux par bloc) |
| `ChargeDensityRhs` | second membre de Poisson à N espèces : `f = Σ_s q_s n_s` |
| `CoupledSource` | source inter-espèces (lit plusieurs blocs + φ), distincte de `model.source` |
| `ImplicitSourceStepper` | défaut implicite : backward-Euler (Newton) sur la source, sans Newton utilisateur |
| `is_implicit(c)` | trait IMEX partiel : quelles variables d'un bloc sont implicites |
| `PoissonCadence` | fréquence de re-résolution de φ entre sous-pas (`OncePerStep`/`PerSubstep`) |
| `MultiSpeciesSolver` / `Simulation` | façades `adc_cases` : composition multi-espèces (compilée / à l'exécution) |

---

## 7. Synthèse (phrase tableau)

> `adc` sait prendre une **loi physique locale** (`PhysicalModel`) et la faire tourner sur
> un maillage avec Poisson, AMR, MPI et GPU. Le niveau d'**assemblage multi-blocs** est
> désormais **rempli** (squelette testable, tous tests verts) : `EquationBlock` (state +
> modèle + schéma spatial + politique temps + BC), `CoupledSystem` (plusieurs blocs),
> `SystemCoupler` mono-niveau **et** `AmrSystemCoupler` sur AMR (RHS elliptique de système
> `Σ_s q_s n_s`, explicite avancé par le cœur, implicite/IMEX délégué avec un défaut
> backward-Euler sans Newton utilisateur, source de couplage inter-espèces, sous-pas par
> bloc, conservation par reflux). Le squelette est **rempli ET exercé par de la vraie
> physique** : modèles deux-fluides canoniques (électrons Euler + ions isothermes) et
> diocotron à ions mobiles dans `adc_cases`, façade `MultiSpeciesSolver` + bindings Python
> qui composent un `CoupledSystem` depuis Python sans callback dans le hot path ; diffusion
> rendue conservative sur AMR (flux de face). Le `PhysicalModel` décrit une équation, le
> `CoupledSystem` un système, le `Scheduler`/coupleur l'ordre d'exécution ; le cœur garantit
> AMR / MPI / GPU. Restent des raffinements (composition Python générique, perf, §4.3/§4.4).

---

## 8. Retour tuteur (2026-06-01) — niveau d'assemblage : couvert vs à extraire

Discussion : coupler ions / électrons / **neutres**, chacun son `PhysicalModel` ; les
espèces interagissent dans le **second membre elliptique** `f` ET dans la **source `S`**
(jamais dans le flux `F`). On veut choisir **par espèce** : le modèle (isotherme / profil
constant donné / résolu ou non / pas à chaque pas), le **schéma spatial**, le **pas de
temps** (sous-cyclage : 10 pas électrons pour 1 pas ions), le **traitement** implicite /
explicite — et même **implicite sur une PARTIE seulement des variables** (coût). Côté
archi : extraire l'**intégrateur en temps** comme objet (`take_step`) donné au coupleur,
et **alléger le coupleur** qui « fait trop » (« avancer un coupleur » est philosophiquement
bancal : un coupleur *assemble*, un *driver* avance).

### 8.1 Déjà couvert (vérifié par test — réponses aux « est-ce qu'on peut ? »)
- [x] **N espèces hétérogènes, chacune son `PhysicalModel`** → `CoupledSystem<Blocks...>`
  (électrons Euler 4 var + ions isothermes 3 var, `test_two_species_euler`). Un **neutre**
  = un bloc de plus (charge 0 → ne contribue pas à `f`, mais interagit via `S`).
- [x] **Interaction dans `f`** (elliptique) : `f = Σ_s q_s n_s` → `ChargeDensityRhs`.
- [x] **Interaction dans `S`** (collisions, échange ; pas dans `F`) → `CoupledSource`
  (`coupled_source.hpp` ; `SystemCoupler::coupled_source_step` lit tous les blocs + φ).
- [x] **Schéma spatial différent par espèce** (électrons MUSCL, ions ordre 1, limiteurs
  distincts) → `SpatialDiscretisation` par bloc (`test_system_two_explicit`).
- [x] **Sous-cyclage par espèce** (10 pas électrons : 1 pas ions) → `substeps` +
  scheduler (`test_system_abstraction` : ne=10, ni=1).
- [x] **Électrons implicites + ions explicites** → `TimeTreatment` par bloc
  (`test_two_species_minimal`, `test_amr_system_coupler` part B).
- [x] **Implicite sur une PARTIE des variables** (pas tout le modèle) → trait
  `Model::is_implicit(c)` (`test_imex_partial`) = « dire quelles variables on step ».
- [x] **Espèce en profil constant / non avancée** → `PrescribedTime` (le scheduler saute
  les blocs `Prescribed`, qui contribuent quand même à `f`). *[à illustrer par un test]*
- [x] **Résoudre l'elliptique tous les pas ou pas** → `PoissonCadence` (`OncePerStep`/`PerSubstep`).
- [x] « numerical method » est **déjà** nommé `SpatialDiscretisation` (≠ time integrator).
- [x] **Le « `PhysicalModel` plus grand » qui englobe tout** = `EquationBlock`
  (`{Model, SpatialDiscretisation, TimePolicy, BC}`) ; `CoupledSystem` = plusieurs de ces
  bundles. C'est déjà le « state physical model » composable décrit (chaque fluide a son
  jeu d'équations : densité seule, ou tous les moments).

### 8.2 À extraire (les vrais manques d'abstraction) — plan détaillé

**A. `TimeIntegrator` objet de premier plan (priorité 1).** Aujourd'hui SSPRK est codé en
dur dans `SystemCoupler::advance_explicit_ssprk2/3` ET recopié dans `ssprk.hpp` et `Coupler` ;
`Time` n'est qu'un *tag* template, `UserTimeIntegrator` n'a pas de `take_step`. Objectif :
donner `{PhysicalModel, SpatialDiscretisation, TimeIntegrator}` au coupleur comme un trio,
le `TimeIntegrator` étant un objet du cœur (ou fourni par l'utilisateur).
- [x] **A1. Contrat** : concept `TimeStepper` exposant `take_step(rhs_eval, U, dt)`, où
  `rhs_eval(U_stage, R)` remplit `R = −div F + S`. L'intégrateur ne voit QUE `rhs_eval` + les
  ops MultiFab. (`integrator/time_steppers.hpp`.)
- [x] **A2. Impls du cœur** : `ForwardEuler`, `SSPRK2Step`, `SSPRK3Step` comme objets
  `take_step` génériques. `ssprk.hpp::advance_ssprk2` délègue désormais à `SSPRK2Step`.
- [x] **A3. Câbler l'intégrateur utilisateur** : le `Method` d'un bloc explicite peut être
  un tag du cœur (`SSPRK2`/`SSPRK3`) **ou un type `TimeStepper` écrit par l'utilisateur** —
  le coupleur l'instancie et appelle son `take_step` (`test_user_time_integrator`). Depuis
  Python : sélection par tag (pas de callback hot path). *[BYO-stepper Python = futur]*
- [x] **A4. Délégation** : `SystemCoupler::advance_explicit_block`, `ssprk.hpp`, **ET le
  `Coupler` legacy mono-modèle** (`advance`/`advance_ssprk3`) délèguent désormais aux objets
  `SSPRK2Step`/`SSPRK3Step` — plus aucune copie SSPRK inline. Le legacy gardait un
  `recompute_aux` par étage (`true` à l'étage 0, `per` ensuite) : reproduit par un `rhs_eval`
  qui compte les étages. **Bit-identique** (diocotron : adc_cases 48/48).
- [x] **A5. Exemple** : `test_user_time_integrator` (intégrateur utilisateur) + tout le reste
  tout-cœur. 
- [x] Garde : **bit-identique** aux SSPRK actuels (adc_cpp 41/41, adc_cases 48/48).

**B. Scinder le coupleur — Assembleur vs Driver (priorité 2 ; §9.6).** Un coupleur assemble
l'elliptique + résout Poisson + dérive aux + calcule le RHS spatial + intègre + sous-cycle :
trop. « Avancer un coupleur » est bancal — un coupleur *assemble*, un *driver* *avance*.
- [x] **B1/B2. Extraction en DEUX classes** (`system_coupler.hpp`) : `SystemAssembler`
  (`solve_fields` + `block_residual` = l'évaluateur de résidu ; phi/aux/geom/ba/dm ; AUCUN
  pas) et `SystemDriver` (schedule, `advance_explicit_block` via `take_step`, IMEX,
  `coupled_source_step`) qui **possède** un `SystemAssembler` (composition). Bit-identique
  (adc_cpp 42/42, adc_cases 48/48). Testé `test_assembler_driver` (assembleur seul + driver).
- [x] **B3. Nom « qui avance »** : `SystemCoupler` est désormais l'**alias** de `SystemDriver`
  (le nom « qui avance ») ; `AmrSystemDriver` alias pour l'AMR. Compat préservée (tests, façades).
- [x] **B4.** La façade compilée `MultiSpeciesSolver` (`adc_cases`) repose déjà sur
  `SystemDriver` (via l'alias `SystemCoupler`). La `Simulation` runtime garde sa propre
  orchestration (type-erased) par conception (espèces composées à l'exécution).

**C. Multirate — `dt` propre par modèle (priorité 3).** `substeps` couvre le « plus souvent »
(10 sous-pas électrons / 1 ion).
- [x] **Espèce « résolue pas à chaque pas »** (every-N) : `TimePolicy` gagne un `Stride`
  (`ExplicitTime<Method, Substeps, Stride>`). Un bloc de cadence N n'avance qu'1 macro-pas
  sur N, alors d'un pas effectif `N·dt` (il rattrape le temps : total `M·dt` au bout de M
  macro-pas, mais calculé N fois moins souvent — le « gaz pas résolu tous les pas »). Géré
  par `block_stride_v` + `advance_subcycled(system, dt, macro_step, …)` ; `SystemDriver` et
  `AmrSystemCoupler` portent `macro_step_`. `Stride=1` = historique bit-identique.
  Testé (`test_multirate_stride` : rapide stride 1 + lent stride 3, synchronisés à M=3).
- [x] **Pas macro choisi par CFL** : `SystemDriver::step_cfl(cfl)` / `cfl_dt(cfl)` calculent
  `dt = cfl·min(dx,dy) / w_max`, `w_max` = plus grande vitesse d'onde **sur toutes les
  espèces** (réduction `max_wave_speed_mf` par espèce, via `model.max_wave_speed`) — l'espèce
  la plus rapide contraint le pas. Combiné au `Stride` d'une espèce lente, cela donne le
  **multirate pratique** (pas macro fixé par les rapides, lente avancée 1 fois sur stride).
  Testé `test_cfl_dt`.
- [x] **Multirate pleinement adaptatif** : `SystemDriver::step_adaptive(cfl)` — pas macro
  fixé par l'espèce la plus rapide (CFL), et `stride` de CHAQUE espèce dérivé **au runtime**
  du ratio `w_max / w_s` (`stride_s = max(1, ⌊w_max/w_s⌋)`). Une espèce N× plus lente avance
  automatiquement 1 fois sur N, par un pas N× plus grand (= son dt stable). Le dispatch par
  bloc est factorisé (`advance_block_dispatch`) et partagé avec `step` (pas de duplication).
  Testé `test_adaptive_multirate` (rapide a=4 → stride 1 ; lent a=1 → stride 4, dérivés seuls).

---

## 9. Durcissement — revue Codex (2026-06-01) — TRAITÉ (2026-06-02)

Revue indépendante de `multispecies-fill`. Verdict initial : squelette **fonctionnel + testé**
mais points à durcir. **Les 6 points ont été traités** (adc_cpp 41/41, adc_cases 48/48, refactors
bit-identiques sur l'existant). État final ci-dessous.

- [x] **9.1 Vrai IMEX** *(recoupe §8.2 A)*. `SystemCoupler::step` ET `AmrSystemCoupler::step`
  font, pour un bloc `IMEX` : **transport explicite par le cœur** (−div F via `SourceFreeModel`
  + Euler avant / `advance_amr` source-free), **puis** source implicite par le callback.
  Implicite pur : pas de transport. (`test_imex_transport` ; avant : champ figé.) Limite connue :
  un bloc IMEX **diffusif** perd le flux Fickien au demi-pas explicite (`SourceFreeModel`
  n'expose pas `diffusivity()`) — raffinement à part.
- [x] **9.2 `ChargeDensityRhs` : défaut de charge** → `SpeciesCharge{}` vaut **`q = 0`** (un
  neutre / un bloc oublié ne pollue plus Poisson) ET `operator()` exige `species.size() ==
  System::n_blocks` (throw sinon : une charge par bloc, neutre déclaré explicitement `q=0`).
  (`test_system_hardening`.)
- [x] **9.3 `AmrSystemCoupler` garde-fous de construction** → le ctor **throw** si
  `block_levels.size() != n_blocks`, si un bloc n'a pas le bon `nlev`, ou si les grilles par
  niveau diffèrent (nombre de boîtes ≠ bloc 0). Plus d'out-of-bounds silencieux.
  (`test_system_hardening`.)
- [x] **9.4 BC `phi`/`aux` AMR** → `AmrSystemCoupler::solve_fields` dérive aux par le **même
  chemin** que le mono-niveau : `fill_ghosts(phi, bcPhi)` → `field_postprocess` →
  `fill_ghosts(aux, aux_bc)`, `aux_bc` dérivé de `bcPhi` (Periodic→Periodic, sinon Foextrap).
  Gère le non-périodique au lieu d'un `fill_boundary` périodique en dur.
- [x] **9.5 `CoupledSource` sur AMR** → `AmrSystemCoupler::coupled_source_step` (splitting
  **par niveau** : repointe chaque bloc vers son niveau k, lit tous les blocs + aux[k]).
  (`test_amr_system_coupler` Partie D : échange conservatif.)
- [x] **9.6 Nom `Coupler`** *(cosmétique)* → alias `SystemDriver` / `AmrSystemDriver` (le nom
  « qui avance ») + doc des deux rôles (assembleur = `solve_fields` ; driver = `step`). La
  scission en deux classes reste reportée (§8.2 B1/B2 : cosmétique sur code validé).

**Nuance (point TimeIntegrator)** : `ssprk.hpp` ne portait qu'`advance_ssprk2` générique ;
SSPRK3 était recodé dans les coupleurs (duplication réelle). C'est résolu (§8.2 A) :
`SystemCoupler` **délègue** SSPRK2/3 aux objets `SSPRK2Step`/`SSPRK3Step` du cœur,
`ssprk.hpp` délègue aussi. Reste le `Coupler` legacy mono-modèle (non migré, diocotron validé,
§8.2 A4).

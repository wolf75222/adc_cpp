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
(6) API Python dans `adc_cases` (reste à faire). Chaque étape validée par un test d'intégration
(§2.5, tous verts). **Reste dans le cœur** : §2.2 IMEX partiel + cadence φ ; §4 nettoyages.
**Hors cœur** : §2.4 modèles canoniques + §3 API Python (dans `adc_cases`).

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
- [ ] **IMEX partiel** : traiter implicitement un **sous-ensemble** des variables d'un bloc (trait `which_implicit()` sur le modèle), pas tout le bloc. (Le défaut actuel traite toute la source en implicite.)
- [ ] **Sous-cyclage temporel par espèce** : `substeps` exécuté (mono-niveau **et** AMR). Reste à trancher la **cadence de re-résolution de φ** entre sous-pas d'espèce (aujourd'hui φ figé sur le pas dans `AmrSystemCoupler`, re-solve par étage RK en mono-niveau).

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
- [ ] **électrons Euler + ions Euler isothermes + Poisson** (cas canonique deux-espèces) — modèles dans `adc_cases`.
- [ ] **diocotron à ions mobiles** (les ions deviennent un 2e bloc ; réutiliser le diocotron existant pour les électrons).
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

- [ ] **Modèles physiques prêts à composer** (dans `adc_cases`) : `ElectronEuler`, `IonEuler`,
  `Diocotron`, `TwoFluid`… exposés pour servir de blocs.
- [ ] Façade compilée `MultiSpeciesSolver(vector<SpeciesConfig>)` instanciant un `CoupledSystem` + `SystemCoupler` (PIMPL, comme les solveurs existants).
- [ ] Bindings pybind11 : `adc.Mesh2D(...)`, `adc.Simulation(mesh, backend=...)`, `sim.add_equation(name, model, spatial, time, bc)`, `sim.add_poisson(rhs, solver)`, `sim.run(t_end, cfl, output)`.
- [ ] `model="electron_euler"`, `flux="hllc"`, `time="imex"` ↔ tags C++ ; jamais de callback `flux(u)` Python dans le hot path.
- [ ] (avancé) exposer un `PhysicalModel` écrit en C++ par l'utilisateur, puis composable depuis Python.

---

## 4. À faire — follow-ups cœur (nettoyages / cohérence)

- [ ] **Diffusion sur AMR** : la porter comme **flux de face diffusif** (pas un Laplacien direct), sinon le reflux AMR ne la voit pas → non conservatif aux interfaces coarse-fine. (`compute_face_fluxes` doit recevoir `geom` et ajouter `−ν(u_R−u_L)/h`.)
- [ ] **Contrat `Aux`** : trancher. Soit `Aux` est fixe (`phi, grad_x, grad_y`) et on retire `typename M::Aux` du concept ; soit on généralise et `load_aux<Model>` construit `typename Model::Aux`. Aujourd'hui le contrat promet plus que le code ne donne.
- [ ] **`SpectralCoupler::max_drift_speed`** : encore `/model_.B0` (diagnostic). Le généraliser via `model.max_wave_speed` (attention : change le `dt`, donc la trajectoire → re-valider, pas bit-identique).
- [ ] **`Coupler` fourre-tout** : extraire l'orchestrateur (le `SystemCoupler` va dans ce sens) ; dédupliquer les diagnostics `AmrCouplerMP::mass()`/`max_drift_speed()` qui réimplémentent `amr_diagnostics.hpp`. Nom : `Coupler` est historique = *Assembler/Simulation Driver* (à acter, pas urgent).

---

## 5. Décision à acter au tableau (avec Sacha)

- **Structure de données du système** : le squelette a retenu `tuple<Blocks...>` (un
  `MultiFab` par bloc, `EquationBlock::state` = pointeur). Alternative : `StateVec<N_total>`
  empilé (un bloc mémoire contigu, offsets par espèce — meilleure localité, plus complexe).
  À confirmer/infléchir : c'est la décision qui pèse sur la perf (point 0 : structure de
  données = plus tard, mais la décision d'interface se prend maintenant).

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
| `SystemCoupler` | orchestrateur : RHS elliptique global + avance chaque bloc |

---

## 7. Synthèse (phrase tableau)

> `adc` sait prendre une **loi physique locale** (`PhysicalModel`) et la faire tourner sur
> un maillage avec Poisson, AMR, MPI et GPU. Le niveau d'**assemblage multi-blocs** est
> désormais **rempli** (squelette testable, tous tests verts) : `EquationBlock` (state +
> modèle + schéma spatial + politique temps + BC), `CoupledSystem` (plusieurs blocs),
> `SystemCoupler` mono-niveau **et** `AmrSystemCoupler` sur AMR (RHS elliptique de système
> `Σ_s q_s n_s`, explicite avancé par le cœur, implicite/IMEX délégué avec un défaut
> backward-Euler sans Newton utilisateur, source de couplage inter-espèces, sous-pas par
> bloc, conservation par reflux). Reste : IMEX partiel + cadence φ (§2.2), nettoyages cœur
> (§4), puis — dans `adc_cases` — les modèles canoniques deux-espèces (§2.4) et l'API Python
> de composition (§3). Le `PhysicalModel` décrit une équation, le `CoupledSystem` un système,
> le `Scheduler`/coupleur l'ordre d'exécution ; le cœur garantit AMR / MPI / GPU.

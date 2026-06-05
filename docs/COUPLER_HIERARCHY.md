# Hierarchie des coupleurs hyperbolique-elliptique

Ce document decrit chaque classe de couplage presente dans `include/adc/coupling/`,
sa responsabilite, ce qu'elle assemble ou avance, et quand la choisir.
Il est un complement de reference a `ARCHITECTURE.md` (section 5 et 8) et ne
duplique pas ce qui y est deja decrit.

---

## 1. Vue d'ensemble : arbre des coupleurs

```
Coupler<Model, Elliptic>                  -- mono-modele, mono-niveau
    |
    +-- SystemAssembler<System, RhsAsm, Elliptic>   -- multi-especes, mono-niveau (ASSEMBLE)
    |        |
    |        +-- SystemDriver<System, RhsAsm, Elliptic>    -- idem, AVANCE (= SystemCoupler)
    |
    +-- AmrCoupler<Model, Elliptic>       -- mono-modele, AMR mono-box  (DEPRECIE)
    |
    +-- AmrCouplerMP<Model, Elliptic>     -- mono-modele, AMR multi-box + regrid BR
    |
    +-- AmrSystemCoupler<System,RhsAsm,Elliptic>   -- multi-especes, AMR (= AmrSystemDriver)
```

Le backend elliptique `Elliptic` est parametre partout via le concept `EllipticSolver`
(fichier `numerics/elliptic/elliptic_solver.hpp`). La valeur par defaut est
`GeometricMG` (multigrille geometrique V-cycle).

---

## 2. Coupler -- mono-modele, mono-niveau

**Fichier :** `include/adc/coupling/coupler.hpp`

**Instanciation :** `adc::Coupler<Model, Elliptic = GeometricMG>`

### Role

Ferme la boucle Poisson -> aux -> advance pour un seul modele physique sur un unique
niveau de grille uniforme. C'est le coupleur le plus simple ; il est utilise directement
dans les cas tests unitaires et dans les tutoriels mono-espece.

### Ce qu'il assemble

A chaque etage de l'integrateur (SSPRK2 ou SSPRK3) :

1. Second membre elliptique : `f = SingleModelEllipticRhs(model, U)`, c'est-a-dire
   `model.elliptic_rhs(U)` cellule par cellule
   (`coupler.hpp:52-54`, delegue a `elliptic_rhs.hpp:27-40`).
2. Resolution de `lap(phi) = f` par le backend elliptique (`GeometricMG::solve`).
3. Derivation `aux = (phi, d_x phi, d_y phi)` par differences centrees, stocke avec
   signe `+grad phi` ; la convention `GradSign::Plus` est documentee dans
   `elliptic_problem.hpp` (`coupler.hpp:61-65`).
4. Si le modele declare `n_aux > 3` (champ `B_z` ou `T_e` supplementaire), peuplement
   de la composante extra depuis le callback `bz` fourni au constructeur.

### Ce qu'il avance

Les methodes `advance` (SSPRK2), `advance_ssprk3` (SSPRK3) et `step` (point d'entree
unifie avec sous-cyclage) appellent `SSPRK2Step::take_step` /
`SSPRK3Step::take_step` en leur passant l'evaluateur de residu
(`coupler.hpp:115-162`). Le coupleur ne contient aucune logique d'integrateur propre.

### Parametres de template notables

| Parametre | Role |
|---|---|
| `Limiter` | Reconstruction d'interface (NoSlope / Minmod / VanLeer / MC) |
| `Policy` | Frequence du solve elliptique : `PerStageCoupling` (phi recalcule a chaque etage RK) ou `OncePerStepCoupling` (phi gele pendant le pas) |
| `NumericalFlux` | Flux de Riemann (Rusanov / HLL / HLLC / Roe) |

### Quand l'utiliser

- Un seul modele physique, un seul champ, un seul niveau.
- Cas test unitaire, tutorial, validation rapide.
- Le modele multi-especes doit passer par `SystemAssembler` / `SystemDriver`.

---

## 3. SystemAssembler / SystemDriver (alias SystemCoupler) -- multi-especes, mono-niveau

**Fichier :** `include/adc/coupling/system_coupler.hpp`

**Instanciations :**
- `adc::SystemAssembler<System, RhsAssembler, Elliptic = GeometricMG>`
- `adc::SystemDriver<System, RhsAssembler, Elliptic = GeometricMG>`
- `adc::SystemCoupler` est un alias de `SystemDriver` (compat historique).

### Separation des responsabilites

Le retour de conception (commentaire introductif de `system_coupler.hpp:29-40`)
explique la scission en deux classes :

> "SystemAssembler : ASSEMBLE. [...] Il ne fait AUCUN pas de temps.
> SystemDriver : AVANCE. [...] Avancer un assembleur n'a pas de sens.
> Le Driver POSSEDE un Assembleur."

**SystemAssembler (lignes 63-167) :**

- Tient le `CoupledSystem` (tuple d'`EquationBlock`), le backend elliptique, le canal
  `aux` partage (un `MultiFab` commun a TOUS les blocs).
- `solve_fields()` : assemble `f = Sum_s q_s n_s` via le `RhsAssembler` fourni
  (typiquement `ChargeDensityRhs`), resout Poisson, derive `aux = (phi, grad phi)`.
- `block_residual<Limiter, NumericalFlux>(block, state, R, recompute_aux)` :
  evaluateur de residu d'un bloc a un etage (appele par le Driver via le TimeStepper).
- Largeur du canal `aux` = maximum de `aux_comps<Model>` sur tous les blocs
  (`system_coupler.hpp:136-143`) : un bloc qui demande `B_z` (n_aux = 4) voit la
  composante extra, un bloc de base (n_aux = 3) ne la voit pas ; bit-identique a
  l'historique si aucun bloc ne demande d'extra.

**SystemDriver (lignes 169-361) :**

- Possede un `SystemAssembler`.
- `step(dt, [implicit_callback])` : sous-cyclage par bloc selon `block_stride_v<Block>`,
  avance chaque bloc selon son `TimeTreatment` :
  - `Explicit` : appel de `SSPRK2Step` / `SSPRK3Step` (ou un `TimeStepper` utilisateur)
    en passant l'evaluateur de residu de l'assembleur.
  - `Implicit` / `IMEX` : resolution des champs, transport explicite si IMEX, puis
    deleguee au callback `implicit_advance(*this, block, dt, ...)`.
- `step_adaptive(cfl)` : pas macro adaptatif ; chaque bloc calcule son propre `stride`
  au runtime a partir du ratio des vitesses d'onde (`system_coupler.hpp:208-232`).
- `step_cfl(cfl)` : `dt = cfl * min(dx,dy) / w_max` puis `step(dt)`.
- `coupled_source_step(src, dt)` : etape de source de couplage inter-especes (splitting
  forward-Euler), lecture de tous les blocs + aux (`system_coupler.hpp:276-283`).

### Ce que RhsAssembler doit exposer

Le `RhsAssembler` est un callable `operator()(const System&, MultiFab& rhs)`.
Le header `elliptic_rhs.hpp` fournit :

- `SingleModelEllipticRhs<Model>` : `f = model.elliptic_rhs(U)` (mono-bloc).
- `TwoBlockChargeDensityRhs` : `f = q0 * n0 + q1 * n1` (deux blocs, compat).
- `ChargeDensityRhs` (recommande) : `f = Sum_s q_s n_s` generique a N blocs, exige
  une entree `SpeciesCharge` par bloc (`elliptic_rhs.hpp:117-133`).

### Quand l'utiliser

- Plusieurs especes sur un maillage uniforme.
- Transport explicite (SSPRK) ou IMEX, avec ou sans sous-cyclage par espece.
- Utiliser `SystemAssembler` directement si l'on veut ecrire son propre ordonnanceur.
- Utiliser `SystemDriver` (ou son alias `SystemCoupler`) dans tous les cas standard.

---

## 4. AmrCoupler -- mono-modele, AMR mono-box (DEPRECIE)

**Fichier :** `include/adc/coupling/amr_coupler.hpp`

**Instanciation :** `adc::AmrCoupler<Model, Elliptic = GeometricMG>`

**Statut :** DEPRECIE. Le commentaire d'en-tete (`amr_coupler.hpp:1-7`) indique :
"Aucun #include dans le coeur, les tests ou les bindings Python ; remplace en
production par AmrCouplerMP, dont le mono-box est le cas degenere bit-identique."
Conserve pour compatibilite documentaire ; a retirer apres migration complete.

### Ce qu'il fait

Meme sequence qu'`AmrCouplerMP` mais sans support multi-box et sans regrid :
`sync_down -> compute_aux -> advance_amr`. La hierarchie des niveaux est portee par
`AmrLevelStack<AmrLevelMP>`, les diagnostics (masse, derive) par `amr_diagnostics.hpp`.

---

## 5. AmrCouplerMP -- mono-modele, AMR multi-patch

**Fichier :** `include/adc/coupling/amr_coupler_mp.hpp`

**Instanciation :** `adc::AmrCouplerMP<Model, Elliptic = GeometricMG>`

### Role

Coupleur AMR E x B mono-modele, multi-patch a chaque niveau. Remplace `AmrCoupler`
en production. Le mono-box est le cas degenere bit-identique (garde de validation
`test_amr_multilevel_multipatch`).

### Sequence d'un pas (`step`)

1. `update()` = `sync_down()` + `compute_aux()` :
   - `sync_down` : moyenne conservatrice fin -> grossier sur toute la hierarchie
     (`mf_average_down_mb`).
   - Poisson grossier : `model.elliptic_rhs(U)` sur le niveau 0, solve MG.
   - `aux(0)` = `(phi, d_x phi, d_y phi)` par differences centrees sur le grossier.
   - Injection piecewise-constante `aux(k-1) -> aux(k)` pour chaque niveau fin via
     `detail::coupler_inject_aux_mb` (`amr_coupler_mp.hpp:56-96`).
2. `advance_amr<Disc>(model, levels, domain, dt, ...)` : pas AMR conservatif
   (Berger-Oliger + reflux coverage-aware, sous-cyclage par niveau).

### Parametres de construction notables

- `replicated_coarse` (defaut `true`) : le niveau 0 est replique sur chaque rang
  (une box, mono-rang). Passer `false` pour un grossier multi-box reparti
  (scalabilite memoire ; penalise le MG si trop de boxes). Equivalence bit-a-bit
  prouvee (`test_mpi_decoarse`, `maxdiff = 0`).
- `active` : predicat optionnel "cellule de conducteur" (paroi circulaire du diocotron).

### Regrid

`regrid(crit, grow, margin)` delegue a `amr_regrid_coupler.hpp::amr_regrid_finest`
(Berger-Rigoutsos). L'appelant fournit le critere de raffinement ; le coupleur
resynchronise `aux` apres regrid.

### Diagnostics

`mass()` et `max_drift_speed()` / `max_wave_speed()` delegues a `amr_diagnostics.hpp`.

### Quand l'utiliser

- Un seul modele, AMR adaptatif avec ou sans multi-patch.
- Pour un systeme multi-especes sur AMR, utiliser `AmrSystemCoupler`.

---

## 6. AmrSystemCoupler (alias AmrSystemDriver) -- multi-especes, AMR

**Fichier :** `include/adc/coupling/amr_system_coupler.hpp`

**Instanciation :** `adc::AmrSystemCoupler<System, RhsAssembler, Elliptic = GeometricMG>`

`AmrSystemDriver` est un alias de `AmrSystemCoupler` (note de conception
`amr_system_coupler.hpp:371-375` : scission cosmétique reportée, classe unifiée
validée bit-identique).

### Role

Porte un `CoupledSystem` sur une hierarchie AMR. Toutes les especes partagent la
MEME grille AMR par niveau (hypothese structurelle) et donc le MEME canal `aux`.
Le Poisson ne se resout que sur le niveau grossier.

### Structure interne

- `block_levels_[b][k]` : `AmrLevelMP` du bloc b au niveau k (tableau 2D).
- `aux_[k]` : un `MultiFab` commun a tous les blocs au niveau k
  (`amr_system_coupler.hpp:130-139`).
- Largeur du canal `aux` partage = max de `aux_comps<Model>` sur les blocs (calque
  exact de `SystemAssembler::system_aux_comps`, `amr_system_coupler.hpp:308-315`).

### Sequence de `solve_fields()`

1. `sync_down` par bloc (fin -> grossier, `mf_average_down_mb`).
2. `rhs_assembler_(system_, mg_.rhs())` : RHS de systeme (p. ex. `ChargeDensityRhs`).
3. `mg_.solve()` : Poisson sur le grossier.
4. `aux_[0]` = `(phi, grad phi)` via le meme chemin que `SystemAssembler::derive_aux`
   (ghosts `bcPhi_`, `field_postprocess`, ghosts `aux_bc_`) :
   `amr_system_coupler.hpp:187-197`.
5. Injection `aux_[k-1] -> aux_[k]` par `coupler_inject_aux_mb`.
6. Re-peuplement eventuel de `B_z` par niveau si `bz_` fourni (apres injection, pour
   preserver la resolution spatiale du champ : `amr_system_coupler.hpp:199-205`).

### Cadence Poisson (`PoissonCadence`)

- `OncePerStep` (defaut) : `phi` resolu une seule fois en tete de pas, gele pendant
  l'avance des blocs. Le moins cher.
- `PerSubstep` : `phi` re-resolu avant chaque sous-pas d'espece. Plus fidele pour un
  transport fortement pilote par le champ.

Note : le `SystemDriver` mono-niveau re-resout phi a CHAQUE etage RK
(`recompute_aux = true`) car c'est la cadence maximale par construction.

### Avance des blocs (`step`)

Pour chaque bloc (selon son `stride` compile-time) :

- `Explicit` : `advance_amr<Disc::Limiter, Disc::NumericalFlux>(model, levels, dt)`
  pour chaque sous-pas ; re-solve Poisson avant chaque sous-pas si `PerSubstep`.
- `IMEX` : transport AMR sur `SourceFreeModel<Block::Model>` (operateur -div F seul)
  puis callback `implicit_advance(*this, block, levels, dt)`.
- `Implicit` : tout au callback.

### Source de couplage inter-especes

`coupled_source_step(src, dt)` : re-pointe chaque bloc vers son niveau k pour chaque
niveau, puis appelle `src.apply(system, aux[k], dt)` : parite exacte avec
`SystemDriver::coupled_source_step` (`amr_system_coupler.hpp:266-283`).

### Defaut implicite : `AmrImplicitSourceStepper`

Backward-Euler (Newton) sur la source, applique niveau par niveau
(`amr_system_coupler.hpp:361-370`). Aucun solveur cote utilisateur requis.

### Quand l'utiliser

- Plusieurs especes, AMR adaptatif.
- Meme grille AMR pour toutes les especes (hypothese centrale).

---

## 7. RhsAssemblers : `elliptic_rhs.hpp`

**Fichier :** `include/adc/coupling/elliptic_rhs.hpp`

Ces types sont passes comme `RhsAssembler` aux coupleurs systeme. Ils ne sont pas
des coupleurs eux-memes mais des briques de composition.

| Type | Signature | Usage |
|---|---|---|
| `SingleModelEllipticRhs<Model>` | `operator()(const MultiFab& state, MultiFab& rhs)` | Mono-bloc : `f = model.elliptic_rhs(U)` |
| `TwoFieldChargeDensityRhs` | `operator()(U0, U1, rhs)` | Deux champs decouples, compat legacy |
| `TwoBlockChargeDensityRhs` | `operator()(system, rhs)` | Deux blocs d'un CoupledSystem |
| `ChargeDensityRhs` | `operator()(system, rhs)` | N blocs : `f = Sum_s q_s n_s` ; **recommande** |

`ChargeDensityRhs` exige une entree `SpeciesCharge{.charge, .comp}` par bloc :
si une espece est neutre, la declarer avec `charge = 0` est obligatoire
(`elliptic_rhs.hpp:122-125`).

---

## 8. CoupledSource : `coupled_source.hpp`

**Fichier :** `include/adc/coupling/coupled_source.hpp`

Une `CoupledSource` modelise un terme source inter-especes qui depend de plusieurs
blocs ET du potentiel. Le concept exige `apply(system, aux, dt)`.

- `NoCoupledSource` : no-op, zero cout, cas mono-espece ou couplage uniquement par
  Poisson (`coupled_source.hpp:37-40`).
- Les sources concretes (collisions, friction, echange de charge) vivent dans
  `adc_cases` ou les tests du coeur, pas dans le coeur lui-meme.

---

## 9. CondensedSchurSourceStepper -- etage source implicite Schur

**Fichier :** `include/adc/coupling/condensed_schur_source_stepper.hpp`

Ajoute en PR #126 (branche `feat/schur-pr4-stepper`).

### Role

Etage source AUTONOME (transport gele) resolvant implicitement le couplage
potentiel / vitesse / Lorentz d'un bloc fluide magnetise (Hoffart et al.,
arXiv:2510.11808). Il NE remplace PAS et NE s'integre PAS encore dans le chemin
`System::step` -- le cablage facade est prevu en PR5
(`condensed_schur_source_stepper.hpp:23`).

### Ce qu'il compose

1. `ElectrostaticLorentzCondensation` (`schur_condensation.hpp`, PR #124) : assemble
   l'operateur condense `A_op = I + c rho B^{-1}` (eps_x, eps_y diagonaux, a_xy/a_yx
   croises) et le second membre condense.
2. `TensorKrylovSolver` (`krylov_solver.hpp`, PR #122) : BiCGStab matrice-libre
   preconditionne par un V-cycle `GeometricMG` sur la partie symetrique (l'operateur
   est non auto-adjoint des que `B_z != 0`).
3. `LorentzEliminator` (`lorentz_eliminator.hpp`, PR #118) : elimination `B^{-1}` 2x2
   fermee pour la reconstruction de la vitesse.

### Sequence d'un pas (`step`)

1. Geler `phi^n` ; extraire `v^n = (mx, my) / rho` ; copier `B_z` dans un tampon interne.
2. Assembler `A_op` et `rhs_schur`.
3. Resoudre `L_int(phi) = -rhs_schur` par BiCGStab (convention de signe documentee
   dans l'en-tete : `L_schur = -L_int`, d'ou la negation du RHS).
4. Reconstruire `v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta})`.
5. Extrapoler `phi` et `v` du theta-stage au pas plein.
6. Mettre a jour l'energie cinetique si le role `Energy` est present.
7. Remplir les ghosts de l'etat et du potentiel.

### Caracteristiques

- Generique : lit les roles `Density / MomentumX / MomentumY` d'un `VariableSet`.
  Tout bloc fluide qui les expose est eligible sans code Schur supplementaire.
- Device-clean : tous les kernels sont des foncteurs nommes (pas de lambda etendue,
  compatible nvcc/GH200).
- Tampons alloues une seule fois au constructeur, reutilises a chaque `step()`.
- MPI-propre : les boucles iterent sur `local_size()`, le solve Krylov est collectif.

### Quand l'utiliser

- Source raide couplee potentiel / vitesse / Lorentz, traitement implicite requis.
- Necessite un bloc fluide avec roles `Density / MomentumX / MomentumY`.
- Non connecte a la facade `System::step` pour l'instant (PR5 a venir).

---

## 10. Canal aux partage : canal extensible

Le canal `aux` transporte au minimum trois composantes : `phi`, `d_x phi`, `d_y phi`
(constante `kAuxBaseComps = 3`). Les briques extensibles ajoutent :

- composante 3 : `B_z(x, y)` (champ magnetique hors-plan, statique, fourni par `bz_`).
- composante 4 : `T_e` (temperature electronique, derivee de `p/rho`).

La largeur est determinee par `aux_comps<Model>()`. Dans les coupleurs systeme, elle
est le MAX sur tous les blocs : un bloc de base (n_aux = 3) reste bit-identique a
l'historique. Le mecanisme est identique dans `SystemAssembler`, `AmrSystemCoupler`
et `AmrCouplerMP` (voir respectivement `system_coupler.hpp:136`, `amr_system_coupler.hpp:308`,
et la construction de `AmrLevelStack`).

---

## 11. Tableau de synthese

| Coupleur | Modeles | Niveaux | ASSEMBLE | AVANCE | Statut |
|---|---|---|---|---|---|
| `Coupler<M,E>` | 1 | 1 (uniforme) | oui (dans le coupleur) | oui | stable |
| `SystemAssembler<Sys,Rhs,E>` | N | 1 (uniforme) | oui (Poisson systeme + aux) | NON | stable |
| `SystemDriver<Sys,Rhs,E>` (= `SystemCoupler`) | N | 1 (uniforme) | via `SystemAssembler` possede | oui | stable |
| `AmrCoupler<M,E>` | 1 | N (mono-box) | oui | oui | DEPRECIE |
| `AmrCouplerMP<M,E>` | 1 | N (multi-box) | oui | oui | stable |
| `AmrSystemCoupler<Sys,Rhs,E>` (= `AmrSystemDriver`) | N | N (multi-box) | oui | oui | stable |
| `CondensedSchurSourceStepper` | 1 bloc fluide | 1 (uniforme) | operateur Schur | etage source seul | experimental (PR #126) |

La distinction ASSEMBLE / AVANCE prend tout son sens au niveau de `SystemAssembler` vs
`SystemDriver` : `SystemAssembler` peut etre reutilise dans un ordonnanceur
specialise (Newton externe, integrateur AP) sans emporter l'ordonnancement interne
de `SystemDriver`. `AmrSystemCoupler` n'a pas (encore) subi cette scission ; la note
de conception (`amr_system_coupler.hpp:371-375`) la signale comme reportee.

---

## 12. References dans le code source

| Symbole | Fichier | Ligne(s) |
|---|---|---|
| `Coupler<Model,Elliptic>` | `coupling/coupler.hpp` | 68 |
| `detail::coupler_eval_rhs` | `coupling/coupler.hpp` | 51 |
| `detail::coupler_grad_phi` | `coupling/coupler.hpp` | 61 |
| `SystemAssembler` | `coupling/system_coupler.hpp` | 63 |
| `SystemDriver` / `SystemCoupler` | `coupling/system_coupler.hpp` | 169, 360 |
| `AmrCoupler` (DEPRECIE) | `coupling/amr_coupler.hpp` | 65 |
| `AmrCouplerMP` | `coupling/amr_coupler_mp.hpp` | 225 |
| `detail::coupler_inject_aux_mb` | `coupling/amr_coupler_mp.hpp` | 56 |
| `AmrSystemCoupler` / `AmrSystemDriver` | `coupling/amr_system_coupler.hpp` | 70, 374 |
| `PoissonCadence` | `coupling/amr_system_coupler.hpp` | 63 |
| `AmrImplicitSourceStepper` | `coupling/amr_system_coupler.hpp` | 361 |
| `SingleModelEllipticRhs` | `coupling/elliptic_rhs.hpp` | 26 |
| `ChargeDensityRhs` | `coupling/elliptic_rhs.hpp` | 117 |
| `SpeciesCharge` | `coupling/elliptic_rhs.hpp` | 87 |
| `CoupledSourceFor` (concept) | `coupling/coupled_source.hpp` | 29 |
| `NoCoupledSource` | `coupling/coupled_source.hpp` | 37 |
| `CondensedSchurSourceStepper` | `coupling/condensed_schur_source_stepper.hpp` | 189 |
| `ElectrostaticLorentzCondensation` | `coupling/schur_condensation.hpp` | -- |

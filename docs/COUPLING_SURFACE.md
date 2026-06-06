# Surface de couplage : PUBLIC, INTERNE, DEPRECIE

Ce fichier est la source de verite pour la classification des classes de
`include/adc/coupling/`. Il repond a la question : "Quelle classe est
publique, interne ou depreciée ?"

---

## Entree utilisateur (PUBLIC Python)

L'entree utilisateur n'est PAS une classe de couplage C++ : c'est le couple
Python `adc.System` / `adc.AmrSystem` + le DSL (`adc.dsl.HyperbolicModel`,
`m.compile(...)`). Les classes ci-dessous sont des facades C++ internes ; elles
ne font pas partie de l'API publique documentee.

---

## Facades C++ internes (INTERNE)

### `coupler.hpp` -- `adc::Coupler<Model, Elliptic>`

Facade mono-modele, mono-niveau : ferme la boucle Poisson -> aux -> advance
pour un seul `PhysicalModel` sur un unique niveau uniforme. Utilisee dans les
tests unitaires et tutoriels mono-espece ; pas l'entree utilisateur.
**Classification : INTERNE**

### `system_coupler.hpp` -- `adc::SystemAssembler`, `adc::SystemDriver`, alias `adc::SystemCoupler`

`SystemAssembler` assemble le RHS multi-especes (Poisson systeme, aux, residus
par bloc) sans faire de pas de temps. `SystemDriver` avance (sous-cyclage
par espece, IMEX). `SystemCoupler` est l'alias de compat vers `SystemDriver`.
Portees par `adc.System` cote Python ; internes au C++ du noyau.
**Classification : INTERNE**

### `amr_coupler_mp.hpp` -- `adc::AmrCouplerMP<Model, Elliptic>`

Coupleur AMR E x B multi-patch : meme role qu'`AmrCoupler` mais hierarchie
multi-box par niveau (moteur `advance_amr`, reflux coverage-aware, regrid
Berger-Rigoutsos). Chemin de production mono-modele AMR.
**Classification : INTERNE**

### `amr_system_coupler.hpp` -- `adc::AmrSystemCoupler<System, RhsAssembler, Elliptic>`, alias `adc::AmrSystemDriver`

Coupleur de systeme sur AMR : porte un `CoupledSystem` sur une hierarchie AMR
partagee par toutes les especes. Cable par `adc.AmrSystem` Python (#92/#105).
`AmrSystemDriver` est l'alias "qui avance" (retour tuteur ss8.2 B).
**Classification : INTERNE**

### `coupling_policy.hpp` -- `adc::PerStageCoupling`, `adc::OncePerStepCoupling`

Tag types pour la politique de couplage temporel hyperbolique-elliptique
(frequence de resolution elliptique). Selectionnes au site d'appel par template.
**Classification : INTERNE**

### `elliptic_rhs.hpp` -- `adc::SingleModelEllipticRhs`, `adc::SystemEllipticRhs`

Assemblage du second membre elliptique : isole la responsabilite du `Coupler`.
`SingleModelEllipticRhs` pour un modele seul, `SystemEllipticRhs` pour plusieurs
especes.
**Classification : INTERNE**

### `coupled_source.hpp` -- concept `adc::CoupledSourceFor`

Concept C++20 definissant le contrat d'une source de couplage inter-especes
(collisions, echange thermique). Les sources concretes vivent dans `adc_cases`.
**Classification : INTERNE**

### `coupled_source_program.hpp` -- bytecode `CoupledSourceProgram`

Evaluateur de source couplee generique par bytecode postfixe (POD device-clean).
Permet d'executer des sources symboliques Python (`adc.dsl.CoupledSource`) dans
un `for_each_cell` device sans callback Python par cellule.
**Classification : INTERNE**

### `aux_fill.hpp` -- `adc::detail::derive_aux_bc`, `adc::detail::fill_bz_box`

Helpers partages par les trois coupleurs (Coupler, SystemAssembler,
AmrSystemCoupler) pour le canal aux (conditions aux limites de phi -> aux,
peuplement de B_z). Extraction pure, corps bit-identiques.
**Classification : INTERNE**

### `amr_level_storage.hpp` -- `adc::AmrLevelStack<Level>`

Stockage de la hierarchie AMR (pile de niveaux + aux). Extrait des coupleurs
AMR pour separer stockage et orchestration.
**Classification : INTERNE**

### `amr_diagnostics.hpp` -- `adc::amr_mass`, `adc::amr_max_drift_speed`

Diagnostics extraits des coupleurs AMR (masse integree, vitesse de derive max).
Free functions de namespace, seam GPU.
**Classification : INTERNE**

### `amr_regrid_coupler.hpp` -- `adc::amr_regrid_finest`

Regrid Berger-Rigoutsos extrait d'`AmrCouplerMP` (responsabilite b). Reconstruit
le niveau fin a la volee depuis un critere de raffinement fourni par l'appelant.
**Classification : INTERNE**

### `schur_condensation.hpp` -- `adc::ElectrostaticLorentzCondensation`

Batisseur de l'etage source condense par Schur (couplage potentiel/vitesse/
Lorentz, Hoffart et al. arXiv:2510.11808). Assemble l'operateur elliptique
tensoriel A_op et le second membre ; ne resout pas.
**Classification : INTERNE**

### `condensed_schur_source_stepper.hpp` -- `adc::CondensedSchurSourceStepper`

Etage source condense par Schur complet : compose `ElectrostaticLorentzCondensation`
et `TensorKrylovSolver`. Etage source autonome (transport gele) ; cablage
facade vers `System::step` prevu en PR5.
**Classification : INTERNE**

---

## Classe depreciée (DEPRECIE)

### `amr_coupler.hpp` -- `adc::AmrCoupler<Model, Elliptic>`

**DEPRECIE.** Coupleur AMR E x B mono-box (mono-modele). Aucun `#include`
dans le noyau, les tests ou les bindings Python. Remplace en production par
`AmrCouplerMP` (`amr_coupler_mp.hpp`), dont le mono-box est le cas degenere
bit-identique. Conserve pour reference historique (documente dans
`docs/ARCHITECTURE.md`) ; a retirer apres migration complete.
**Remplacement recommande : `adc::AmrCouplerMP` (`amr_coupler_mp.hpp`)**

### `spectral_coupler.hpp` -- `adc::SpectralCoupler<Model>`

**DEPRECIE.** Coupleur periodique distribue (FFT par bandes). Aucun `#include`
dans le noyau, les tests ou les bindings Python. Le role est repris par
`Coupler<Model, DistributedFFTSolver>` (le solveur FFT distribue est devenu un
`EllipticSolver` autonome, cf. `poisson_fft_solver.hpp`). Conserve pour
reference historique (documente dans `docs/ARCHITECTURE.md`).
**Remplacement recommande : `adc::Coupler<Model, DistributedFFTSolver>` (`coupler.hpp`)**

---

*Derniere mise a jour : 2026-06-06 (P0.3 du CODEBASE_AUDIT.md).*

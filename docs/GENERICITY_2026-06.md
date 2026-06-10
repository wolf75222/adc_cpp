# Genericite — decisions de design et limites restantes (audit 2026-06)

Reponse aux trois audits du 2026-06-09 (MEETING_AUDIT / HARDCODED_GENERIC_AUDIT /
CPP_FULL_REPO_AUDIT) : ce document resume CE QUI A ETE CHANGE, POURQUOI sous cette forme, et ce qui
reste explicitement NON generalise. Regle directrice partout : **comportement par defaut
strictement bit-identique** (les capacites nouvelles sont des opt-in declares, jamais des
changements silencieux).

## 1. step_cfl / step_adaptive : politique de pas a bornes agregees

- **Traits OPTIONNELS du contrat modele** (core/physical_model.hpp) :
  `stability_speed(U, aux, dir)` (lambda* remplacant max_wave_speed dans la CFL — stabilite !=
  precision, les solveurs de Riemann gardent max_wave_speed), `source_frequency(U, aux)` (mu en
  1/s, borne SANS h : la "deuxieme CFL" du meeting), `stability_dt(U, aux)` (pas admissible direct,
  cfl NON applique). Reductions device (max / max / min-via-inverse) dans spatial_operator.hpp,
  fermetures dans block_builder.hpp, forwarding conditionnel dans CompositeModel.
- **Sous-pas effectif** : toutes les bornes s'appliquent a `stride*dt/substeps` (memes facteurs que
  la borne transport historique).
- **Bornes GLOBALES** : `System::add_dt_bound(label, fn)` — une evaluation HOTE par pas (callback
  Python acceptable ICI, jamais par cellule), pour couplage multi-blocs / Schur / scheduler / rampe
  utilisateur. `step_adaptive` les honore aussi (clameur du macro-pas, conservative).
- **Diagnostic** : `System::last_dt_bound()` nomme la borne ACTIVE du dernier step_cfl
  ("transport:<bloc>" / "source_frequency:<bloc>" / "stability_dt:<bloc>" / "global:<label>" /
  "degenerate").
- **DSL** : `m.stability_speed(expr)` / `m.stability_dt(expr)` COMPILES comme flux/source
  (production GPU/MPI, pas de callback par cellule). Fallback strict : `max(abs(eigenvalues))`.
- **Constantes** : plancher 1e-30 nomme `kCflSpeedFloor` (core/types.hpp), partage System/AMR.

## 2. IMEX -> SourceImplicitBE : options et diagnostics Newton

- Le chemin reste EXACTEMENT ForwardEuler(transport sans source) + backward-Euler LOCAL sur la
  source (Newton par cellule). La documentation (IMEX/SourceImplicit) le NOMME desormais
  explicitement et nie la famille IMEX-RK ; alias `adc.SourceImplicitBE = adc.SourceImplicit`.
- `NewtonOptions {max_iters, rel_tol, abs_tol, fd_eps}` (implicit_stepper.hpp) : defauts = les
  constantes historiques (2 / 0 / 0 / 1e-7) -> chemin (2a) bit-identique, ZERO cout. Tolerances
  actives -> chemin instrumente (arret par cellule sur ||F||_inf <= abs_tol + rel_tol*||F0||_inf,
  detection residu non fini et pivot degenere/non fini via solve_dense -> bool).
- Diagnostics OPT-IN (`newton_diagnostics=True`) : scratch par cellule + reductions + all_reduce ->
  `sim.newton_report(bloc)` = {enabled, converged, max_residual, max_iters_used, n_failed}, agrege
  sur les sous-pas de la derniere avance. La cellule fautive (i, j) n'est PAS localisee (une
  reduction arg-max device serait un chantier a part ; le rapport donne le pire residu et le compte).
- Plomberie : adc.IMEX / adc.SourceImplicit (kwargs newton_*) -> System::add_block -> build_block ->
  AdvanceImex*. Options non-defaut HORS imex ou sur un backend .so (ABI non transportee) -> rejet
  explicite, jamais d'ignore silencieux.

## 3. Flux : domaine de validite explicite, HLL aligne

- HLLC/Roe documentes EULER 2D SEULEMENT (+ alias `EulerHLLCFlux2D` / `EulerRoeFlux2D`) ; l'entropy
  fix de Roe est la constante NOMMEE `kRoeEntropyFixFraction = 0.1`, documentee Euler/Roe-specifique.
- `hll` (generique a ondes signees, requires wave_speeds) est desormais route aussi par l'AMR
  (dispatch_amr_block + dispatch_amr_compiled, meme requires-gate que System) et documente partout
  (Spatial / FiniteVolume / system.hpp / Sphinx). Test visible :
  `adc.FiniteVolume(limiter="minmod", riemann="hll", variables="primitive")` sur l'isotherme 3-var
  (test_fv_hll_minmod, System + AmrSystem + rejets explicites).

## 4. Couplages nommes : multi-box/MPI-safe

- add_ionization / add_collision / add_thermal_exchange iterent `local_size()` (patron
  add_coupled_source) au lieu de `fab(0)/box(0)` : no-op sur un rang sans box, prets pour un System
  multi-box. La resolution par ROLE avec fallback indices canoniques est conservee (couplages
  nommes = sucre "layout fluide canonique", documente).

## 5. Briques magnetiques exposees + bug latent corrige

- `adc.MagneticLorentzForce(charge)` et `adc.PotentialMagneticForce(charge)` exposees dans
  adc.Model(...) (factory C++ "magnetic"/"potential_magnetic" deja prete) et dans le chemin hybride
  (_native_to_brick, y compris CompositeSource via champs imbriques a.qom/b.qom).
- **Bug latent corrige** : System::add_block (chemin natif) n'appelait JAMAIS
  `ensure_aux_width(aux_comps<M>())` — un modele natif a n_aux=4 (brique magnetique) lisait B_z
  HORS BORNES d'un aux a 3 composantes (valeurs poubelle silencieuses). Le test quantitatif
  (residu == q*B*m exactement sur etat uniforme) verrouille le cablage.

## 6. check_model : garde-fous generiques

- `dsl.Model.check_model(...)` (formules, avant compilation) : flux/source/elliptic finis,
  valeurs propres finies/reelles, coherence wave_speeds <-> max_wave_speed, round-trip
  to_conservative(to_primitive(U)) ~= U, positivite Density / 'p', echantillons reproductibles.
- `System.check_model(bloc)` (runtime, tout backend) : U fini, residu -div F + S fini, positivite
  par roles, round-trip des conversions du modele (etat sauve/restaure).

## 7. Constantes numeriques sorties des noyaux

- `kCflSpeedFloor` (1e-30, types.hpp), `kRoeEntropyFixFraction` (0.1, numerical_flux.hpp),
  `NewtonOptions.fd_eps` (1e-7), iters IMEX -> `NewtonOptions.max_iters`,
  tolerances Schur : `set_krylov(tol, max_iters)` sur les steppers condenses cartesien (1e-10/400)
  et polaire (1e-10/600), exposees via `adc.CondensedSchur(krylov_tol=, krylov_max_iters=)`.

## 8. Sorties / checkpoint

- PLAN seul (docs/IO_CHECKPOINT_PLAN.md) : API cible write/checkpoint/restart, contenu minimal du
  checkpoint (t, macro_step, grille, blocs, U, aux/B_z, params, stride/substeps), contraintes HPC
  (pas un fichier/processus ; HDF5 agrege puis parallele), decoupage en 3 PR.

## Vague 2 (correction d'intention : GENERALISER, pas seulement marquer)

La premiere vague traitait plusieurs chemins non couverts par REJET explicite ; la vague 2 les
CABLE (le rejet ne reste que la ou l'architecture n'est pas prete, et il est liste plus bas).

1. **StabilityPolicy sur AMR** : AmrRuntimeBlock et les hooks mono-bloc portent
   source_frequency/stability_dt ; build_amr_block/_compiled routent la vitesse de CFL via
   stability_speed (trait) ; AmrRuntime::step_cfl ET le step_cfl mono-bloc agregent les bornes
   (formules substeps/stride identiques a System) ; AmrSystem::add_dt_bound / last_dt_bound
   (parite System, all_reduce_min). Un modele DSL m.stability_dt(...) compile
   target='amr_system' contraint donc le pas AMR (test B4).
2. **Capabilities Riemann** : HLLC et Roe ne sont plus des algorithmes Euler-only deguises --
   `HasHLLCStructure` (pressure + wave_speeds + contact_speed + hllc_star_state) et
   `HasRoeDissipation` (d = |A_roe| dU, entropy fix inclus) permettent a UN MODELE de fournir sa
   structure ; le coeur applique l'algorithme generique (F* = F_k + s_k (U*_k - U_k) ;
   F = 1/2(F_L+F_R) - 1/2 d). Le chemin canonique Euler 2D reste l'implementation historique
   bit-identique. Gates System ET AMR elargies. Preuves : hooks-Euler == canonique a 1e-13 ;
   HLLC isotherme 3-var preserve EXACTEMENT un cisaillement stationnaire la ou HLL le diffuse.
3. **Newton** : damping (0,1], fail_policy none|warn|throw (hote, apres reductions ; observateur
   pur), cellule fautive (i, j) + composante via reduction encodee (rapport + message du throw).
   Preuve non-Euler : relaxation non lineaire 3 variables converge sous tolerances ; pathologie
   NaN sur UNE cellule -> throw avec (i, j) exacts.
4. **DSL m.source_frequency(expr)** : emise comme `frequency(U, aux)` sur la brique de SOURCE
   generee (le contrat optionnel de physics/source.hpp), forwardee par CompositeModel, agregee
   par step_cfl ("source_frequency:<bloc>"). Exige m.source (erreur explicite sinon).
5. **Schur roles dans l'ABI** : set_source_stage transporte density/momentum_x/momentum_y/energy
   (nom de role stable OU nom de variable) + bz_aux_component ; le stepper cartesien gagne un
   constructeur a composantes explicites (le ctor canonique y DELEGUE, bit-identique) ;
   adc.CondensedSchur(density=..., momentum=(...), magnetic_field=...) forwarde au lieu de
   rejeter. Defauts = roles canoniques, bit-identique.
6. **IO v1** : sim.write(path, format='vtk'|'npz', step=), sim.checkpoint / sim.restart (npz,
   ecriture atomique) ; bindings macro_step() / set_clock() / set_potential() -- la reprise est
   BIT-IDENTIQUE (y compris cadence stride via macro_step, et warm start MG via phi restaure).
7. **adc.capabilities()** : matrice de verite unique (riemann x facade, time, stability_policy,
   poisson, schur, backends DSL, io) ; docstrings perimees corrigees (PolarMesh, etage Schur
   polaire "ne se branche pas", perimetre Phase 3c du Schur AMR, CondensedSchur "mono-rang").

## Vague 3 (solde : polaire, couplages, DSL HLLC/Jacobien, AMR Newton, Schur polaire/AMR, IO, multi-blocs)

Confirmation au tableau (tuteur) : `dt <= dx/|lambda_max|  <=>  CFL = dt*lambda*/dx` — c'est
exactement le trait `stability_speed` (lambda* = vitesse de STABILITE, distincte de max_wave_speed
qui reste la vitesse du solveur de Riemann). La vague 3 etend cette politique partout ou elle
manquait encore.

1. **Polaire : StabilityPolicy cablee** (block_builder_polar.hpp) : fabriques
   make_cfl_speed_polar / make_source_frequency_polar / make_stability_dt_polar (foncteurs nommes
   PolarStabilitySpeed/PolarSourceFreq/PolarStabilityDt, device-clean) ; la branche polaire de
   System::add_block les installe. Un modele polaire declarant stability_speed / stability_dt /
   source_frequency borne donc le pas exactement comme en cartesien (memes formules
   substeps/stride, meme last_dt_bound). Defaut sans trait : max_wave_speed historique,
   bit-identique.
2. **CoupledSource.frequency(mu)** : une source couplee DECLARE sa frequence (1/s) ; la borne
   `dt <= cfl/mu` s'applique au MACRO-pas (les couplages sont appliques une fois par macro-pas,
   pas par sous-pas) ; raison "coupled_source:<label>". Plomberie System ET AmrSystem
   (add_coupled_source(frequency=, label=) -> coupled_freqs_ ; AmrRuntime::add_coupled_frequency ;
   step_cfl mono-bloc agrege aussi). DSL : `dsl.CoupledSource(...).frequency(mu)` transporte par
   CompiledCoupledSource. Defaut sans frequency : aucun changement.
3. **DSL emet les hooks HLLC** : `m.enable_hllc()` genere contact_speed + hllc_star_state DEPUIS
   LES ROLES (Density/MomentumX/MomentumY[/Energy] requis ; les variables hors roles fluides sont
   traitees en scalaires passifs advectes a la vitesse de contact — generalisation, pas une
   hypothese Euler). Exige 'p' declare (pression/pseudo-pression). CompiledModel.has_hllc ;
   CompositeModel forwarde contact_speed/hllc_star_state/roe_dissipation (concept-gates) ;
   riemann='hllc' accepte sur un 3-var NON Euler via la capability, rejet explicite sans elle.
   Preuve : HLLC 3-var isotherme tourne fini (test_v3_features D1).
4. **Jacobien analytique de la source** : trait `HasSourceJacobian` (jacobian(U, aux, J),
   J[r][c] = dS_r/dU_c) utilise par les DEUX chemins Newton (historique et instrumente) a la place
   des differences finies (if constexpr, zero cout sans trait). DSL : `m.source_jacobian(rows)`
   (n x n d'expressions) emis sur la brique source ; sucre `m.implicit_source(source, jacobian=)`.
   Preuve : meme racine que les FD a 2.8e-17 (C++), meme trajectoire IMEX a 1e-9 (Python).
5. **Options Newton TRANSPORTEES sur AMR multi-blocs natif** : AmrSystem::add_block accepte les
   kwargs newton_* ; BlockSpec porte NewtonOptions ; build_amr_block capture les options dans
   imex_advance (l'iters=2 fige a disparu des fermetures multi-blocs). Restent REJETES
   explicitement : mono-bloc AMR (fermetures du coupleur historique), loaders .so (ABI), et
   newton_diagnostics (le rapport agrege reste System-only).
6. **Schur : roles + krylov configurables sur polaire ET amr-schur** : constructeurs a composantes
   explicites (le ctor canonique DELEGUE, bit-identique) + set_krylov(tol, iters) sur
   PolarCondensedSchurSourceStepper (1e-10/600) et AmrCondensedSchurSourceStepper (grossier
   seulement) ; set_source_stage(density=, momentum_x=, momentum_y=, energy=, bz_aux_component=)
   resolu role-OU-nom sur les trois facades. AMR : magnetic_field != B_z reste rejete (le
   composite fin lit B_z par contrat Aux).
7. **set_conservative_state MULTI-BLOCS** : cable pour les modeles natifs (l'etat complet seede le
   niveau grossier via coupler_write_coarse_state ; preuve : la quantite de mouvement seedee
   advecte des le 1er pas). Loaders .so : rejet explicite (pas de chemin d'etat dans l'ABI v1).
8. **IO etendu** : `sim.write(format='hdf5')` (h5py OPTIONNEL, erreur claire sinon) ;
   `AmrSystem.write` npz/vtk (champs GROSSIERS de CHAQUE bloc, par nom, + phi + empreintes des
   patchs fins ; les champs multi-niveaux = PR-IO-3) ; checkpoint/restart AMR = rejet explicite
   pointant le plan (PR-IO-3).

## Points encore NON generalises (explicites, mis a jour vague 3)

1. **AMR** : pas de `ssprk3` ; coarse/fine suppose ratio 2 ; `set_poisson` limite a geometric_mg +
   rhs charge_density|composite ; options Newton mono-bloc AMR et loaders .so = rejet explicite
   (fermetures du coupleur historique / ABI) ; newton_diagnostics/newton_report = System seulement
   (la reduction arg-max encodee n'est cablee que sur le chemin System).
2. **AMR Schur Phase 4** : composite limite a 2 niveaux + UN patch fin mono-box + mono-rang ;
   multi-patch, > 2 niveaux, MPI, multi-blocs = Phase 4 (rejet explicite, perimetre documente
   dans l'en-tete du stepper). set_krylov AMR ne pilote que l'etage grossier.
3. **Polaire** : flux Rusanov seulement (pas de HLL/HLLC/Roe polaire) ; Poisson direct
   mono-rang/mono-box (Schur tensoriel = multi-box) ; decoupage theta non expose par la facade.
4. **Aux** : toujours extensible PAR LISTE CANONIQUE (ADC_AUX_FIELDS + AUX_CANONICAL miroir
   Python), pas d'auxiliaire arbitraire par modele.
5. **Briques natives layout fluide** : source.hpp / elliptic.hpp lisent toujours u[0]/u[1]/u[2]
   (documente "layout fluide canonique") ; pas de variantes role-aware.
6. **IMEX-RK** : aucune famille ARK/IMEX-RK ; SourceImplicitBE est le seul schema implicite local
   (le Jacobien analytique vague 3 en ameliore la robustesse, pas l'ordre).
7. **CoupledSource** : toujours explicite forward-Euler additif, capacites fixes (kCsMaxReg=32...) ;
   frequency(mu) est une CONSTANTE declaree, pas une expression par cellule (une frequence
   bytecode par cellule = suivi).
8. **Backends** : `compile(backend="aot")` reste le defaut (decision utilisateur a trancher ;
   adc.capabilities() publie la matrice) ; registry C++ des tags strings non factorisee
   (les tables make_block / dispatch_amr_* / polar restent paralleles, alignees par tests).
9. **check_model sur CompiledModel** : porte sur les FORMULES (dsl.Model) et le BLOC INSTALLE
   (System) ; un CompiledModel seul (sans son Model d'origine) n'est pas re-verifiable.
10. **IO** : System mono-rang (npz/vtk/hdf5 via h5py) ; HDF5 agrege/PARALLELE multi-rangs,
    checkpoint AMR et champs externes (B_z dans le checkpoint) = PR-IO-3 du plan.
11. **Roe cote DSL** : `m.enable_hllc()` emet les hooks HLLC ; l'equivalent Roe
    (m.roe_dissipation(...) ou une linearisation de Roe generee depuis les flux) reste a faire —
    le contrat C++ HasRoeDissipation est pret.

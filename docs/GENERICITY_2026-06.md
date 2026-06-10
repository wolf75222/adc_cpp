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

## Points encore NON generalises (explicites)

1. **AMR** : pas de `ssprk3` ; coarse/fine suppose ratio 2 ; `set_conservative_state` multi-blocs
   non cable ; `set_poisson` limite a geometric_mg + rhs charge_density|composite ; les OPTIONS
   NEWTON (adc.IMEX) et krylov_tol/max_iters + descripteurs de roles (adc.CondensedSchur) restent
   REJETES explicitement par les facades AmrSystem (iters=2 fige dans les fermetures AMR ; le
   forwarding complet = plomberie binding + AmrRuntimeBlock + imex_advance, suivi dedie).
2. **AMR Schur Phase 4** : composite limite a 2 niveaux + UN patch fin mono-box + mono-rang ;
   multi-patch, > 2 niveaux, MPI, multi-blocs = Phase 4 (rejet explicite, perimetre documente
   dans l'en-tete du stepper).
3. **Polaire** : bornes de pas optionnelles NON cablees (transport max_wave_speed seul) ; flux
   Rusanov seulement ; overrides de roles Schur rejetes ; Poisson direct mono-rang/mono-box ;
   decoupage theta non expose par la facade.
4. **Aux** : toujours extensible PAR LISTE CANONIQUE (ADC_AUX_FIELDS + AUX_CANONICAL miroir
   Python), pas d'auxiliaire arbitraire par modele.
5. **Briques natives layout fluide** : source.hpp / elliptic.hpp lisent toujours u[0]/u[1]/u[2]
   (documente "layout fluide canonique") ; pas de variantes role-aware.
6. **IMEX-RK** : aucune famille ARK/IMEX-RK ; SourceImplicitBE est le seul schema implicite local.
   Pas de Jacobien ANALYTIQUE fourni par le modele (differences finies seulement ; le hook DSL
   m.implicit_source(residual=, jacobian=) reste a faire).
7. **CoupledSource** : toujours explicite forward-Euler additif, capacites fixes (kCsMaxReg=32...),
   pas de frequency()/dt_bound() sur les sources couplees DSL (contournement : add_dt_bound).
8. **Backends** : `compile(backend="aot")` reste le defaut (decision utilisateur a trancher ;
   adc.capabilities() publie desormais la matrice) ; registry C++ des tags strings non factorisee
   (les tables make_block / dispatch_amr_* / polar restent paralleles, alignees par tests).
9. **check_model sur CompiledModel** : porte sur les FORMULES (dsl.Model) et le BLOC INSTALLE
   (System) ; un CompiledModel seul (sans son Model d'origine) n'est pas re-verifiable.
10. **IO** : v1 npz mono-rang System (write vtk/npz + checkpoint/restart bit-identique) ; HDF5
    agrege/parallele, AMR et champs externes (B_z dans le checkpoint) = PR-IO-3 du plan.
11. **HLLC/Roe capability cote DSL** : les hooks (contact_speed / hllc_star_state /
    roe_dissipation) sont un contrat C++ ; le DSL ne les emet pas encore (m.contact_speed(...) /
    m.roe_dissipation(...) = suivi).

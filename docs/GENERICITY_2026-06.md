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

## Points encore NON generalises (explicites)

1. **AMR** : step_cfl AMR reste transport-only (pas de bornes source/stability/globales) ; mono-bloc
   AMR garde la formule historique `cfl*h/w_max` ; pas de `ssprk3` AMR ; coarse/fine suppose ratio 2.
   Le `hll` est le seul alignement AMR fait ici. CONVENTION "pas d'ignore silencieux" appliquee
   (revue adverse) : un modele declarant stability_speed/source_frequency/stability_dt est REJETE
   explicitement par les dispatchers AMR ; les options Newton (adc.IMEX) et krylov_tol/max_iters
   (adc.CondensedSchur) sont REJETEES par les facades AmrSystem (non transportees).
2. **Polaire** : les bornes de pas optionnelles ne sont pas cablees sur le chemin polaire
   (make_block_polar ne fabrique pas source_frequency/stability_dt ; fallback transport historique).
   Polar reste Rusanov-only cote flux.
3. **Aux** : toujours extensible PAR LISTE CANONIQUE (ADC_AUX_FIELDS + AUX_CANONICAL miroir Python),
   pas d'auxiliaire arbitraire par modele.
4. **Briques natives layout fluide** : source.hpp / elliptic.hpp lisent toujours u[0]/u[1]/u[2]
   (documente "layout fluide canonique") ; pas de variantes role-aware.
5. **IMEX-RK** : aucune famille ARK/IMEX-RK ; SourceImplicitBE est le seul schema implicite local.
   Pas de Jacobien analytique fourni par le modele (differences finies seulement).
6. **CoupledSource** : toujours explicite forward-Euler additif, capacites fixes (kCsMaxReg=32...),
   pas de frequency()/dt_bound() sur les sources couplees DSL (la borne GLOBALE add_dt_bound sert
   de contournement manuel).
7. **Newton AMR / coupleurs** : les chemins AMR (amr_system_coupler, amr_subcycling) gardent
   iters=2 via la surcharge de compatibilite ; options/diagnostics non plomb es sur AMR.
8. **Backends** : la matrice de capacites par backend (prototype/aot/production x System/AMR) reste
   a publier ; `compile(backend="aot")` reste le defaut (decision a trancher : passer a
   "production" est un changement de comportement utilisateur).
9. **check_model sur CompiledModel** : la verification porte sur les FORMULES (dsl.Model) et le
   BLOC INSTALLE (System) ; un CompiledModel seul (sans son Model d'origine) n'est pas re-verifiable.
10. **Sorties/checkpoint** : non implementees (plan seulement).
11. **Roe generique** : volontairement NON tente (exigerait un contrat d'eigenstructure complet :
    roe_average, eigenvectors, entropy fix par modele).

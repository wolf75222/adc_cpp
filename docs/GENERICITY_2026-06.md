# Genericite -- decisions de design et limites restantes (audit 2026-06)

Reponse aux trois audits du 2026-06-09 (MEETING_AUDIT / HARDCODED_GENERIC_AUDIT /
CPP_FULL_REPO_AUDIT) : ce document resume CE QUI A ETE CHANGE, POURQUOI sous cette forme, et ce qui
reste explicitement NON generalise. Regle directrice partout : **comportement par defaut
strictement bit-identique** (les capacites nouvelles sont des opt-in declares, jamais des
changements silencieux).

## 1. step_cfl / step_adaptive : politique de pas a bornes agregees

- **Traits OPTIONNELS du contrat modele** (core/physical_model.hpp) :
  `stability_speed(U, aux, dir)` (lambda* remplacant max_wave_speed dans la CFL -- stabilite !=
  precision, les solveurs de Riemann gardent max_wave_speed), `source_frequency(U, aux)` (mu en
  1/s, borne SANS h : la "deuxieme CFL" du meeting), `stability_dt(U, aux)` (pas admissible direct,
  cfl NON applique). Reductions device (max / max / min-via-inverse) dans spatial_operator.hpp,
  fermetures dans block_builder.hpp, forwarding conditionnel dans CompositeModel.
- **Sous-pas effectif** : toutes les bornes s'appliquent a `stride*dt/substeps` (memes facteurs que
  la borne transport historique).
- **Bornes GLOBALES** : `System::add_dt_bound(label, fn)` -- une evaluation HOTE par pas (callback
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
  `ensure_aux_width(aux_comps<M>())` -- un modele natif a n_aux=4 (brique magnetique) lisait B_z
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

Confirmation au tableau (tuteur) : `dt <= dx/|lambda_max|  <=>  CFL = dt*lambda*/dx` -- c'est
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
   CompiledCoupledSource. Defaut sans frequency : aucun changement. RAFFINEMENT (sec. 7) : mu accepte
   AUSSI une Expr (memes champs block().role() + param() que les termes) -> frequence PAR CELLULE
   mu(U), emise en bytecode (freq_prog_ops/args) et reduite (MAX) par cellule a chaque pas
   (CoupledFreqKernel, foncteur device-clean ; all_reduce_max global ; meme raison). AMR : evaluee sur
   le NIVEAU GROSSIER des blocs d'entree. Constante = chemin historique, bit-identique.
3. **DSL emet les hooks HLLC** : `m.enable_hllc()` genere contact_speed + hllc_star_state DEPUIS
   LES ROLES (Density/MomentumX/MomentumY[/Energy] requis ; les variables hors roles fluides sont
   traitees en scalaires passifs advectes a la vitesse de contact -- generalisation, pas une
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
   **SOLDE (vague 3, points NON generalises #1)** : les OPTIONS sont desormais aussi cablees en
   MONO-BLOC (threadees sur le coupleur AmrCouplerMP : AmrBuildParams.newton_options ->
   build_amr_compiled -> cpl->step -> advance_amr -> subcycle_level_mp -> mf_apply_source_treatment ->
   backward_euler_source ; defaut {} bit-identique). newton_diagnostics/newton_report est cable en
   MULTI-BLOCS natif (NewtonReport par AmrRuntimeBlock en shared_ptr, reset en tete d'avance dans
   AmrRuntime::step, agregation max/somme + all_reduce identique a System::AdvanceImex ; binding
   AmrSystem.newton_report(name) -> dict, forme exacte du binding System). Restent rejetes : le
   RAPPORT en mono-bloc (rejet au build, threader le report dans le subcyclage du coupleur serait
   invasif) et les loaders .so (options ET diagnostics : ABI plate, rejet a la facade Python).
   Preuves : test_amr_newton_full (mono options finies, multi newton_report coherent, no-default-change
   dmax==0, rejets .so) + test_v3_features section (B).
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

## Vague 4 (polaire HLL, IO multi-rangs, horloge AMR) -- audit 2026-06, chantiers POLAIRE + IO

Solde des chantiers POLAIRE (section 3) et IO (section 10) du plan, a **perimetre honnete** (ce qui
est cable l'est reellement ; ce qui ne l'est pas est documente avec fichier:ligne, jamais masque).

1. **HLL POLAIRE cable** (`include/adc/runtime/block_builder_polar.hpp`, `make_block_polar`). Le RHS
   polaire `assemble_rhs_polar<Limiter, NumericalFlux, Model>` portait DEJA le flux numerique en
   PARAMETRE DE TEMPLATE (point d'injection identique au cartesien `build_block<Limiter, Flux>`) et
   appelle `nflux(model, L, aux_L, R, aux_R, dir)` -- exactement la signature de `HLLFlux`. Brancher
   HLL etait donc un petit cablage : `make_block_polar` route `riemann='hll'` vers
   `build_block_polar<Limiter, HLLFlux>`, GATE sur `model.wave_speeds` (foncteur nomme, device-clean,
   meme `requires` que `block_builder.hpp` make_block). Le fluide isotherme polaire
   (`IsothermalFluxPolar : IsothermalFlux`) herite `wave_speeds` -> eligible ; l'ExB scalaire
   (`ExBVelocityPolar`, pas de `wave_speeds`) -> rejet CLAIR. **Defaut `rusanov` strictement
   bit-identique** (branche separee, intouchee). HLLC/Roe restent rejetes (Euler 4-var, pas de brique
   flux d'energie polaire). Facade : `adc.PolarMesh` + `adc.FiniteVolume(riemann='hll')` ;
   `adc.capabilities()['riemann']['system_polar'] = ['rusanov', 'hll']`. Test :
   `python/tests/test_polar_hll.py` (rusanov reproductible, hll fini ET distinct de rusanov) +
   `test_polar_rejections.test_polar_rejects_hll_on_scalar_exb`.

2. **Decoupage theta polaire : EXPOSE (ADC-67, met a jour la decision "NON expose" ci-dessus).**
   `adc.PolarMesh(..., theta_boxes=N)` decoupe l'anneau en N BANDES theta (chaque boite couvre tout le
   rayon `[0, nr-1]` et une bande azimutale ; `theta_boxes` doit DIVISER `ntheta` et rester
   `<= ntheta`). `theta_boxes=1` (defaut) = mono-box, STRICTEMENT bit-identique. Plomberie : `system.cpp`
   `Impl::index_boxarray` construit le BoxArray en bandes (reutilise la decoupe `theta_split` du test
   `test_polar_schur_multibox`) + `DistributionMapping(ba.size(), n_ranks())` round-robin ;
   `SystemConfig.theta_boxes` (append-only) + binding ; validation aux DEUX niveaux (`PolarMesh` cote
   Python, `check_geometry` cote C++). MATRICE multi-box (honnete) :
   - **TRANSPORT polaire multi-box OK** : `assemble_rhs_polar` itere `local_size()` et le residu de bloc
     (`PolarBlockRhsEval`) remplit les ghosts par `fill_ghosts` COLLECTIF (halos inter-boites + theta
     periodique + r physique) AVANT l'assemblage -- pas de raccourci mono-box (deja valide multi-box par
     `test_polar_schur_multibox`, theta-split 8 boites). Le marshaling hote (`copy_state`/`copy_comp0`/
     `write_state` cote `Impl`, et `set_density`) reconstruit/scatter l'anneau GLOBAL quand
     `local_size()>1` (place chaque boite a ses indices globaux, comme `state_global`) -- le store reste
     VERBATIM (chemin `local_size()<=1` delegue tel quel, bit-identique cartesien + polaire mono-box).
   - **Poisson polaire DIRECT mono-box only** : `ensure_elliptic_polar` leve une erreur AMONT claire si
     `ba.size()!=1` (avant la construction du `PolarPoissonSolver`, des le 1er `solve_fields`/`step`/
     `potential`). Le solveur direct (FFT-en-theta + tridiag-en-r) exige lignes theta + colonnes r
     completes sur une box. Message : pointer vers `theta_boxes=1` OU l'etage Schur tensoriel.
   - **etage Schur tensoriel polaire multi-box** : le solveur C++ est deja multi-box ; `theta_boxes`
     pilote desormais le decoupage cote facade.
   Mono-rang (le Poisson direct refuse MPI). Tests : `python/tests/test_polar_theta_boxes.py`
   (transport isotherme bit-identique theta_boxes=1/2/4 ; ExB scalaire ; rejets divisibilite + Poisson
   direct multi-box ; round-trip get/set state multi-box). `adc.capabilities()['geometry']`.

3. **IO System MULTI-RANGS** (`python/system.cpp` + `include/adc/runtime/system.hpp` + bindings +
   `python/adc/__init__.py`). Constat : `copy_state` / `copy_comp0` / `potential` lisaient `fab(0)`
   (valable sur le rang proprietaire -- mono-box, box 0 sur rang 0 -- mais HORS BORNES sur un rang
   sans box). Ajoute : accesseurs GLOBAUX collectifs `density_global` / `state_global` /
   `potential_global` (tampon global rempli en indices GLOBAUX depuis les fabs locaux, puis
   `all_reduce_sum_inplace` -> chaque rang detient le champ complet ; pattern du reflux AMR). Les
   marshalings d'ecriture/lecture (`write_state` / `set_potential` / `copy_*`) sont desormais
   gardes contre `local_size()==0` (no-op / vide au lieu d'un UB). Facade : `sim.write` /
   `sim.checkpoint` font le gather collectif (tous rangs) puis n'ecrivent le fichier que sur le rang 0
   (`_adc.my_rank()`/`n_ranks()` exposes) ; `sim.restart` lit le fichier (FS partage) et appelle
   `set_state` / `set_potential` MPI-safe (rang 0 ecrit, autres no-op) + `set_clock`. **Mono-rang
   bit-identique** (`state_global == get_state`, all_reduce = identite, box = domaine complet) :
   test `python/tests/test_io_multirank.py`. SEMANTIQUE GARANTIE : sous MPI np>1, `write`/`checkpoint`
   produisent UN fichier identique au mono-rang (System mono-box), checkpoint/restart bit-identiques.
   HDF5 PARALLELE (hyperslabs) = PR-IO-3. (Pas de harnais MPI cote pytest -> couverture np>1 a valider
   en central ; le gather reutilise le pattern deja valide par `test_krylov_solver_np*` /
   `test_schur_condensation_np*`.)

4. **Horloge AMR : macro_step() / set_clock()** (`include/adc/runtime/amr_system.hpp` +
   `python/amr_system.cpp` + `amr_runtime.hpp` + `amr_dsl_block.hpp` + bindings). Parite System :
   `AmrSystem::Impl` porte un compteur de macro-pas AUTORITAIRE (incremente par step/step_cfl),
   `macro_step()` le rend, `set_clock(t, ms)` le restaure ET le pousse au compteur de CADENCE du
   moteur (regrid/stride) : `AmrRuntime::set_macro_step` (multi-blocs) OU hook `set_macro_step` du
   coupleur mono-bloc (additif en queue de `AmrCompiledHooks`, abi_key auto-bumpe via ADC_HEADER_SIG).
   Prerequis PR-IO-3, **utile seul** (cadence stride + reprise d'horloge). Test :
   `python/tests/test_amr_clock.py`.

5. **Checkpoint AMR : rejet AMELIORE** (`python/adc/__init__.py`). Un checkpoint AMR bit-identique est
   IMPOSSIBLE avec l'ABI actuelle ; un repli "grossier seul" serait LOSSY (density() = comp0 seul, la
   quantite de mouvement/energie multi-var non lisible) ET non bit-identique (set_conservative_state
   seede le grossier + prolonge, ne restaure pas les patchs fins). Le rejet `NotImplementedError` est
   conserve mais son message liste DESORMAIS PRECISEMENT les 4 manques ABI (lecture etats fins par
   patch ; lecture etat conservatif complet ; serialisation hierarchie+ownership ; ecriture etats fins
   par patch). Pas de coarse-restart opt-in : il ne serait pas propre (lossy + re-prolongation).

## Points encore NON generalises (explicites, mis a jour vague 3)

1. **AMR** : pas de `ssprk3` ; coarse/fine suppose ratio 2 ; `set_poisson` limite a geometric_mg +
   rhs charge_density|composite. Options Newton : CABLEES en mono-bloc (threadees sur le coupleur
   AmrCouplerMP : step -> advance_amr -> subcycle_level_mp -> mf_apply_source_treatment ->
   backward_euler_source, defaut {} bit-identique) ET en multi-blocs ; les loaders .so les rejettent
   encore (ABI plate). newton_diagnostics/newton_report : CABLE en MULTI-BLOCS natif (NewtonReport par
   AmrRuntimeBlock, reset en tete d'avance dans AmrRuntime::step, agregation max/somme + all_reduce
   identique a System) ; mono-bloc AMR et loaders .so = rejet explicite (le rapport n'est pas threade
   dans le subcyclage du coupleur ni transporte par l'ABI).
2. **AMR Schur Phase 4** : composite limite a 2 niveaux + UN patch fin mono-box + mono-rang ;
   multi-patch, > 2 niveaux, MPI, multi-blocs = Phase 4 (rejet explicite, perimetre documente
   dans l'en-tete du stepper). set_krylov AMR ne pilote que l'etage grossier.
3. **Polaire** : flux **Rusanov ET HLL** (HLL cable depuis la vague 4, cf. ci-dessous), mais
   **HLLC/Roe restent NON cables** (supposent n_vars==4 Euler avec energie, sans brique flux
   d'energie polaire -> rejet explicite, make_block_polar). Poisson direct mono-rang/mono-box (Schur
   tensoriel = multi-box). **Decoupage theta NON expose par la facade** (decision documentee
   ci-dessous) : le transport polaire (System.step) est lui-meme MONO-BOX
   (`python/system.cpp` ctor Impl : `ba(std::vector<Box2D>{index_domain(c)})`, une seule box ;
   `set_source_stage` L.~1095-1103 : la facade construit UNE box couvrant l'anneau, le decoupage
   theta n'est pilotable qu'au niveau de l'API C++ PolarCondensedSchurSourceStepper). Exposer un
   parametre `theta_boxes` sur `adc.PolarMesh` serait une **facade mensongere** : le solveur Schur
   tensoriel sait decouper theta, mais le System (transport + Poisson direct) non, et il n'existe
   AUCUNE plomberie (BoxArray decoupe + DistributionMap + halos fill_boundary du transport polaire)
   pour le rendre multi-box. Rien n'est donc expose ; le verrou est documente, pas masque.
4. **Aux** : extensible par LISTE CANONIQUE (ADC_AUX_FIELDS + AUX_CANONICAL miroir Python : phi/grad/
   B_z/T_e) **ET, depuis ADC-70 (phase 1), par champ NOMME declare par le modele** : `m.aux_field("nom")`
   reserve une composante du canal aux a partir de `kAuxNamedBase = 5` (apres T_e), lue en C++ via
   `aux.extra_field(k)` (tableau POD `Real extra[kAuxMaxExtra]`, device-clean ; `load_aux<NComp>` la
   charge sous `if constexpr (NComp > kAuxNamedBase)` -> zero codegen au defaut, bit-identique). Cote
   facade : `System.set_aux_field(bloc, nom, array)` / `aux_field(bloc, nom)` (resolution nom -> comp
   dans la facade Python a partir de `CompiledModel.aux_extra_names`, le C++ ne manipule que des indices
   via `set_aux_field_component`). Champs STATIQUES persistants (re-appliques apres `ensure_aux_width`,
   comme B_z) ; au plus `kAuxMaxExtra = 4` par modele. B_z / T_e restent sur leurs chemins dedies (rejet
   explicite redirigeant dans `set_aux_field`). PERIMETRE phase 1 = **System CARTESIEN** ; le `.so`
   exporte un symbole optionnel `adc_compiled_aux_extra_names` (auto-description). SUIVI (phase 2) : AMR
   (canal aux par niveau + regrid), polaire (validation), halos custom par champ, et table nom -> comp
   cote C++ `System::Impl` (resolution sans la facade Python).
5. **Briques natives ROLE-AWARE (fait)** : source.hpp (PotentialForce / GravityForce /
   MagneticLorentzForce) et elliptic.hpp (ChargeDensity / BackgroundDensity / GravityCoupling)
   portent desormais des MEMBRES d'indices (c_rho / c_mx / c_my / c_E, entiers POD -> device-clean),
   resolus A LA CONSTRUCTION (hote) par model_factory.hpp (bind_variable_roles) via
   TR::conservative_vars().index_of(role). Resolution AUTOMATIQUE et transparente (aucun parametre
   utilisateur nouveau). Defauts = layout fluide canonique -> pour tout transport NATIF (Euler /
   Isothermal / ExB, roles canoniques) les indices resolus == les defauts -> STRICTEMENT bit-identique
   (cable aussi en polaire, dispatch_model_polar). LIMITE : depuis l'API publique les briques natives
   ne se composent qu'avec ces transports CANONIQUES ; un layout permute ne rencontre une brique native
   que via un chemin C++ direct (verrou : detection `requires` du binder + registre des roles).
6. **IMEX-RK (fait, ARS(2,2,2))** : la famille IMEX-RK EXISTE desormais -- `adc.IMEXRK(scheme="ars222")`
   cable le schema d'Ascher-Ruuth-Spiteri (1997), **ordre 2** (transport explicite L = -div F couple a
   la source raide implicite par un tableau a etages). gamma = 1 - 1/sqrt(2), delta = 1 - 1/(2 gamma) ;
   tableaux stiffly accurate (b == derniere ligne de A) -> U^{n+1} = dernier etage. Cote C++ :
   `detail::AdvanceImexRkArs222` (block_builder.hpp), avance PARALLELE a `AdvanceImex` -- il REUTILISE
   `BlockRhsEval<SourceFreeModel>` (transport), `backward_euler_source` (solve implicite local) et
   saxpy/lincomb, AUCUN nouveau kernel device ; la contribution de source d'etage 2 est recuperee par la
   relation de coherence `dt*gamma*S^(2) = U^(2) - base2` (pas de noyau de source en plus). PERIMETRE =
   **System cartesien** : `time="imexrk_ars222"` est rejete explicitement sur AMR, polaire, les loaders
   .so (prototype/aot/production) et les splittings Strang/Schur (hyperbolique != Explicit). Le defaut
   `adc.IMEX` (= SourceImplicitBE, backward-Euler local, ordre 1) reste le seul schema implicite local
   et est **INCHANGE / bit-identique** (kind "imex" != "imexrk_ars222", chemins C++ distincts). LIMITE :
   la source IMEX-RK est PLEINEMENT implicite (la relation de coherence d'etage suppose un solve
   homogene) -> incompatible avec un masque partiel `implicit_vars`/`implicit_roles` (rejet explicite ;
   pour un IMEX partiel par composante, rester sur `adc.IMEX`). Le Jacobien analytique
   (`m.source_jacobian`, vague 3) ameliore aussi les solves d'etage IMEX-RK.
7. **CoupledSource** : toujours explicite forward-Euler additif, capacites fixes (kCsMaxReg=32...).
   frequency(mu) accepte desormais une CONSTANTE (chemin historique) OU une Expr -> frequence PAR
   CELLULE mu(U) en bytecode, reduite (MAX) a chaque pas (cf. sec. 2 ; AMR : borne sur le grossier).
   RESTE : la borne AMR ne voit pas une sous-estimation locale de mu sous un patch fin (evaluee sur
   le grossier, choix assume).
8. **Backends** : DECISION ACTEE (ADC-63) -- le defaut devient `backend="auto"` : PRODUCTION
   (loader natif zero-copie) quand la parite toolchain avec le module _adc est etablie (module
   chargeable + compilateur bake + signature d'en-tetes concordante), AOT sinon (defaut
   historique sur, sans module). Jamais muet : CompiledModel.backend dit ce qui a ete construit,
   CompiledModel.backend_auto_reason dit pourquoi ; un backend explicite court-circuite la
   politique (inchange). Perimetre : Model.compile / HyperbolicModel.compile (modeles de BLOC) ;
   les briques hybrides gardent leur defaut aot (pipeline distinct, suivi). **Registry des tags FACTORISE (fait)** : la VALIDATION des
   tags (limiteurs + flux de Riemann) et les n_ghost sont desormais une SOURCE UNIQUE
   (include/adc/runtime/dispatch_tags.hpp : kLimiters / kRiemanns + validate_limiter / validate_riemann
   / limiter_n_ghost). make_block, dispatch_amr_block, dispatch_amr_compiled et make_block_polar
   valident D'ABORD via ce registre (messages historiques preserves, par contexte) ; leurs throws de
   tag finaux deviennent une garde d'incoherence registry/dispatch. RESIDU : le DISPATCH lui-meme
   reste un template if/else par call-site (les types Limiter / Flux sont compile-time, non tabulables
   sans X-macro lourde) ; la table ne porte donc que les chaines + n_ghost, pas l'aiguillage de type.
   Bug de divergence corrige au passage : les branches hllc / roe AMR ont gagne le cas weno5 (parite
   stricte avec System).
9. **check_model sur CompiledModel** : FAIT (solde) -- `CompiledModel.check_runtime(n=, state=)`
   installe le .so dans un System EPHEMERE et delegue a System.check_model (etat/residu finis,
   positivite par roles, round-trip des conversions) ; etat de fumee par ROLES par defaut,
   state= pour un regime precis. Reste : les FORMULES d'un .so sans son dsl.Model d'origine ne
   sont pas re-derivables (le source symbolique n'est pas embarque dans le .so -- assume).
10. **IO** : System `write` / `checkpoint` / `restart` sont **MULTI-RANGS** (vague 4) -- gather GLOBAL
    collectif (all_reduce_sum_inplace) + ecriture rang-0 + scatter MPI-safe (System MONO-BOX : tout
    l'etat vit sur le rang 0, gather exact ; bit-identique au mono-rang). **HDF5 PARALLELE par
    hyperslabs : FAIT cote System `write` (ADC-66 / PR-IO-3)** -- `sim.write(format="hdf5",
    parallel=True)` OPT-IN : datasets globaux `(ncomp, ny, nx)` crees collectivement via h5py(mpio),
    chaque rang ecrit SES boites en hyperslabs (accesseurs C++ minimaux NON collectifs
    `System::local_boxes` / `System::local_state` ; `python/system.cpp` + `system.hpp` + bindings).
    `parallel=False` (defaut) STRICTEMENT inchange ; h5py absent / sans MPI / mpi4py absent ->
    RuntimeError CLAIR avec remede (jamais d'ecriture silencieuse). Le System cartesien etant MONO-BOX,
    le VRAI parallelisme par hyperslabs n'apparait qu'en MULTI-BOX (documente honnetement). Test :
    `python/tests/test_hdf5_parallel.py` (np=1 : equivalence parallel True==False champ a champ ;
    erreur claire si h5py sans MPI ; regression du chemin serie). RESTE = PR-IO-3 : **HDF5 PARALLELE
    AMR** (un groupe/dataset par niveau + boites, ADC-65), **checkpoint redemarrable HDF5 parallele**
    (le checkpoint reste npz gather-rang-0 ; `checkpoint(parallel=True)` leve), **checkpoint AMR**
    (rejet explicite, ABI des etats fins par patch manquante), champs externes (B_z dans le checkpoint).
11. **Roe cote DSL** : FAIT (solde) -- DEUX voies complementaires pour le hook `roe_dissipation`
    (trait `HasRoeDissipation`, le coeur faisant F = 1/2(FL+FR) - 1/2 d).
    - **Voie ROLES** -- `m.enable_roe()` emet `roe_dissipation` depuis les ROLES : avec Energy =
      transcription exacte de l'algebre canonique Euler du coeur (parite BIT-EXACTE constatee sur
      8 pas), sans Energy = meme decomposition avec c = sqrt(p/rho) moyenne a la Roe, composantes
      hors roles fluides = scalaires passifs sur l'onde entropique (test_dsl_roe : cisaillement
      stationnaire preserve exactement en 3-var).
    - **Voie FOURNIE** -- `m.roe_dissipation(x=rows_x, y=rows_y)` : pour un modele ARBITRAIRE (hors
      familles a roles fluides), l'utilisateur fournit lui-meme les n lignes d_i de son
      eigenstructure (meme esprit que `m.source_jacobian`), ecrites en `dsl.left(...)`/`dsl.right(...)`
      des deux etats UL/UR (une variable nue, sans marqueur, leve). L'**autodiff symbolique** du DSL
      assiste cette ecriture : `dsl.diff(expr, var)` derive l'arbre Expr (linearite, produit,
      quotient, chaine pow/sqrt ; primitives developpees par leur definition ; noeud inconnu ->
      NotImplementedError), et `m.flux_jacobian(dir)` en deduit le Jacobien de flux A = dF/dU. La
      DIAGONALISATION symbolique automatique de A (eigenstructure generale) reste hors perimetre --
      hors de portee d'une emission generique honnete : la voie fournie la delegue a l'utilisateur.

    Les deux voies sont EXCLUSIVES (un seul fournisseur du hook ; les declarer ensemble leve) et
    `CompiledModel.has_roe` couvre les deux (test_dsl_autodiff_roe : dsl.diff sur cas analytiques,
    flux_jacobian de l'isotherme 3-var, m.roe_dissipation reproduisant le Roe isotherme a la main
    == enable_roe a ~1e-12, rejets).

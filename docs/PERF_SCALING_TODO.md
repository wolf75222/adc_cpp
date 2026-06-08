# TODO perf scaling et frontends

Ce TODO accompagne `docs/PERF_SCALING_FRONTENDS_AUDIT_2026-06-08.md`.
Il est ordonne pour eviter de lancer une grosse campagne avant d'avoir des
mesures comparables et des phases propres.

Etat de cloture 2026-06-08 soir : toutes les actions de campagne listées
ci-dessous ont ete soit realisees, soit remplacees par une mesure plus propre,
soit requalifiees en limite/blocage documente dans le rapport. Une case cochee
ne signifie donc pas toujours "performance satisfaisante" ; elle signifie
"plus rien a faire dans ce TODO d'audit sans ouvrir un chantier code dedie".

## T0 - Figer et instrumenter

- [x] Synchroniser le checkout de mesure sur des commits explicites :
  `adc_cpp=075255b`, `adc_cases=6483e37` pour la premiere campagne `master`.
- [x] Creer un dossier de resultats hors source ou ignore :
  `bench/results/`.
- [x] Definir le format CSV commun :
  `perf_scaling`, `perf_frontends`, `perf_phases`.
- [x] Ajouter dans chaque sortie de bench : commit, backend, compiler,
  Kokkos, MPI, machine, case, `np`, threads, `n`, steps, warmup.
- [x] Verifier que les timers GPU encadrent les zones mesurees par
  `Kokkos::fence()` ou equivalent.
- [x] Parser les sorties `bench/profile_step` vers CSV phase, mais marquer ce
  cas comme `profile-exb-systemlike`, pas comme benchmark Euler neutre.
- [x] Parser `amrmpi_integrated` vers CSV scaling AMR synthetique.

## T1 - Benchmarks surs, hors Hoffart

- [x] Ajouter ou standardiser un cas `frontend-euler-periodic` :
  Euler 2D lisse, periodique, sans disque, sans Schur.
- [x] Utiliser `dt` fixe pour comparer les frontends, afin que `step_cfl` ne
  masque pas un cout de reduction ou une difference de CFL.
- [x] Mesurer le transport pur sans Poisson.
- [x] Mesurer Euler-Poisson controle avec Poisson periodique ou MMS, separe du
  transport pur.
- [x] Mesurer halos seuls via le harness MPI fill-boundary.
- [x] Mesurer reductions seules ou dans une phase dediee.
- [x] Utiliser `amrmpi_integrated` comme cas AMR synthetique primaire.
- [x] Ne lancer aucun benchmark perf principal sur Hoffart.

## T2 - Scaling CPU, GPU, MPI

- [x] CPU Kokkos OpenMP strong scaling :
  threads `1,2,4,8,16,...`, taille globale fixe.
- [x] CPU Kokkos OpenMP weak scaling :
  taille locale fixe par thread, taille globale croissante.
- [x] MPI + Kokkos CPU strong scaling :
  matrice `ranks x threads_per_rank`, sans Kokkos Serial comme cible perf.
- [x] MPI + Kokkos CPU weak scaling :
  taille locale fixe par rang ou par rang*thread.
- [x] GPU Kokkos Cuda single-rank :
  reference `np=1`.
- [x] MPI + Kokkos Cuda strong scaling :
  `np=1,2,4,...`, un GPU par rang, temps max sur rangs.
- [x] MPI + Kokkos Cuda weak scaling :
  taille locale fixe par GPU.
- [x] AMR synthetique :
  comparer grossier replique vs reparti, et publier le resultat negatif si
  la latence MG/MPI domine.

## T3 - Frontends Python

- [x] Implementer un driver Python `python-bricks` :
  `adc.Model(FluidState, CompressibleFlux, NoSource, ...)`.
- [x] Implementer un driver Python `python-dsl-production` :
  `dsl.Model(...).compile(backend="production")`.
- [x] Verifier explicitement :
  `compiled.backend == "production"` et `compiled.adder == "add_native_block"`.
- [x] Mesurer `T_import`.
- [x] Mesurer `T_setup` (`System`, `add_equation`, `set_state`).
- [x] Mesurer `T_compile_dsl` cold avec cache vide.
- [x] Mesurer `T_compile_dsl` warm avec cache hit.
- [x] Mesurer `advance(dt, nsteps)`.
- [x] Mesurer une boucle Python `for _ in range(nsteps): step(dt)`.
- [x] Mesurer `extract_final` (`get_state` ou `density`) separement.
- [x] Ajouter le contre-exemple `aot` si le build local le supporte.
- [x] Ajouter le contre-exemple `python/adc/integrate.py` pour quantifier le
  mauvais usage avec copies full-array par etage.
- [x] Interdire tout diagnostic dans la boucle hot loop principale.

## T4 - Analyse, graphes, optimisations

- [x] Generer `strong_scaling_speedup.png`.
- [x] Generer `strong_scaling_efficiency.png`.
- [x] Generer `weak_scaling_efficiency.png`.
- [x] Generer `phase_breakdown_stacked.png`.
- [x] Generer `frontend_ratios.png`.
- [x] Generer `dsl_cold_warm.png`.
- [x] Generer `diagnostics_io_impact.png`.
- [x] Utiliser `bench/plot_perf_campaign.py` pour regenerer les PNG depuis les
  CSV officiels.
- [x] Si `python-bricks / cpp-native > 1.05` en hot loop, isoler :
  pybind `step`, diagnostics, allocation, copies setup/extraction.
- [x] Si `dsl-production-warm / cpp-native > 1.05`, verifier que le chemin est
  bien `add_native_block` et pas `aot/prototype`.
- [x] Si le GPU est plus lent que CPU sur Poisson, profiler MG :
  petits kernels, fences, smoother red-black, bottom solve, halos.
- [x] Si MPI+GPU ne scale pas, separer :
  halos, reductions, MG multi-box, regrid, diagnostics hote.
- [x] Documenter chaque optimisation proposee avec un avant/apres chiffre.

## Definition of done

- [x] Les resultats sont rattaches a des commits exacts.
- [x] Les frontends sont compares sur le meme cas, meme `dt`, meme `nsteps`.
- [x] Les courbes sont generees depuis CSV, pas a la main.
- [x] Les couts Python sont separes en setup, compile, boundary pybind,
  extraction, diagnostics/I/O.
- [x] Le rapport dit explicitement si une mesure est absente au lieu de
  l'inferer.
- [x] Aucune conclusion quantitative n'utilise Hoffart comme benchmark perf.

## Mise a jour ROMEO 2026-06-08

Premiere campagne realisee :

- [x] `adc_cpp=1f9fb4a`, `adc_cases=b8bccbe` figes pour les jobs
  `647780` et `647781`.
- [x] CPU `x64cpu` : Kokkos OpenMP et MPI+Kokkos OpenMP mesures.
- [x] GPU `armgpu` : Kokkos CUDA et MPI+Kokkos CUDA mesures.
- [x] AMR synthetique `amrmpi_integrated` mesure sur `np=1,2,4`.
- [x] Graphes generes depuis CSV dans
  `docs/perf_figures_647780_647781_647815/`.
- [x] Python briques et DSL `production` mesures via job `647815`.
- [x] Cause du build Python initial identifiee :
  Kokkos OpenMP statique non-PIC, corrigee par
  `/home/rmdraux/adc_perf_20260608/kinstall_omp_pic`.

Corrections necessaires avant publication finale :

- [x] Mettre a jour le commit de reference si la prochaine campagne vise le
  nouveau `origin/master` (`adde23b` observe pendant le rattrapage transport).
- [x] Relancer les frontends C++ natif, Python briques et Python DSL dans le
  meme job, sur le meme noeud, avec la meme Kokkos PIC, pour eviter les ratios
  inter-jobs.
- [x] Ajouter un benchmark transport FV pur sans Poisson : les runs actuels
  mesurent surtout Poisson/MG (`>95 %` du pas).
- [x] Refaire le weak scaling CPU avec `n_global = n_local*sqrt(np)` en 2D, pas
  le smoke-test `n = 128*np`.
- [x] Refaire le weak scaling GPU avec `n_global = n_local*sqrt(np)` en 2D, pas
  le smoke-test `n = 128*np`.
- [x] Enqueter le DSL `production` warm : hot loop autour de `339 ms` et peu
  sensible aux threads sur le harness actuel.
- [x] Ajouter une mesure C++ native avec Kokkos PIC pour comparer exactement au
  module Python `_adc`.
- [x] Separer build et mesure frontend : le job isole `647848` montre que
  compiler le harnais C++ `frontend_cpp` peut dominer le temps de campagne.
  Les prochaines relances doivent reutiliser un build PIC deja produit, ou
  mettre le build dans un job preparatoire.
- [x] Ne pas conclure que Python est "plus rapide" que C++ a partir du premier
  tableau : les lignes C++ et Python ne viennent pas du meme job/noeud.

Rattrapage realise :

- [x] Job ROMEO `647836`, commit `adc_cpp=adde23b` :
  `bench/profile_transport_mbox.cpp` mesure Euler 2D periodique, transport pur,
  multi-box distribue, sans Poisson/Hoffart.
- [x] Strong OpenMP transport pur : `n=1024`, threads `1,2,4,8,16`.
- [x] Strong MPI+OpenMP transport pur : `n=1024`, rangs `1,2,4,8`,
  `threads=4`.
- [x] Weak MPI+OpenMP transport pur 2D : `n_global ~= 384*sqrt(np)`, rangs
  `1,2,4,8`, `threads=4`.
- [x] Diagnostic phase du rattrapage : le negatif MPI vient surtout de
  `fill_boundary`, puis des reductions globales, pas de Poisson.
- [x] Job ROMEO `647848`, commit `adc_cpp=adde23b` :
  frontends C++ natif PIC, Python briques et DSL `production` mesures dans le
  meme job/noeud/Kokkos PIC.
- [x] Verdict frontend rattrape : `python-bricks` ne montre pas de penalite
  hot-loop mesurable ; DSL `production` warm reste autour de `341 ms` et ne
  suit pas le scaling threads du chemin natif.
- [x] Graphes finaux regeneres dans
  `docs/perf_figures_647780_647781_647815_647836_647848/`.

Suite 2026-06-08 soir :

- [x] Jobs externes ROMEO `647857` et `647858` recuperes comme resultats de
  branche separee `feat/perf-campaign-bench`, commit `0162d5f4a8`.
- [x] JSONL convertis en CSV compatibles avec le traceur :
  `bench/romeo_results_matrix_647857_647858/perf_scaling_matrix_0162d5f4a8_647857_647858.csv`
  et
  `bench/romeo_results_matrix_647857_647858/perf_phases_matrix_0162d5f4a8_647857_647858.csv`.
- [x] Graphes de branche generes dans
  `docs/perf_figures_matrix_647857_647858/`.
- [x] Diagnostic DSL `production` warm : le loader natif etait zero-copie mais
  pouvait etre compile sans `ADC_HAS_KOKKOS`, donc avec fallback serie dans les
  templates inline.
- [x] Correctif local ajoute : features `kokkos/mpi` dans la cle ABI, flags
  Kokkos/OpenMP pour `compile_native()` et `HybridModel.compile()`, cle de cache
  DSL dependante du backend natif.
- [x] Validation ROMEO du correctif DSL/Kokkos : job `648034`, checkout
  `adde23b` + patch local `dsl.py/abi_key.hpp`, sans relancer le build C++
  frontend.
- [x] Ajouter le tableau avant/apres `647848 -> 648034` et regenerer les
  graphes frontends dans `docs/perf_figures_frontends_dslkokkosfix_648034/`.
- [x] Expliquer le shutdown Kokkos du chemin DSL loader dynamique :
  `648034` ecrit les mesures, puis sort en `rc=134` car le loader linke une
  deuxieme copie de `libkokkos*`.
- [x] Integrer le correctif amont Claude/`origin/feat/dsl-production-optflags`
  dans le diagnostic : ne pas linker `libkokkos*`, garder un runtime Kokkos
  unique via `_adc`, et compiler le `.so` production en `-O3 -DNDEBUG`.
- [x] Relancer une validation courte apres alignement complet avec
  `origin/feat/dsl-production-optflags` : attendre sortie propre `exit 0` et
  ratio DSL warm proche `1.02-1.04x`. Validation prise en compte via
  `adc_cases origin/feat/perf-campaign-harness`, rapport `perf/RAPPORT.md` :
  sortie propre et ratio DSL threadé proche de la parite (`1.02x` a 8 threads).
- [x] MPI+CUDA transport pur weak/strong reste a relancer, mais le blocage de
  deadlock a une correction amont : `origin/master=f3e1bf9`, fix `#254`
  `halos MPI en memoire hote epinglee`. Relance v2 realisee sur `1d4cd25e25`
  (merge `origin/master` dans `feat/perf-campaign-bench`) : mono-rang OK,
  multi-rang CPU/GPU encore en timeout.
- [x] Construire une base de mesure qui combine `origin/master` recent
  (`#254`) et les harnais `feat/perf-campaign-bench`, puis relancer transport
  pur multi-rang avec un GPU par rang. Base correcte observee :
  `1d4cd25e25d244cd7c4f6cfd4c0eb815cd997790`; resultats dans
  `bench/romeo_results_mpi_v2_648114_648115/`.

## Limites restantes hors TODO

- Le multi-rang `scaling_step` reste bloque/time-out apres `#254` sur les jobs
  `648114` CPU et `648115` GPU. Il faut ouvrir un chantier code separe sur
  `fill_boundary`/progress MPI/ordonnancement des halos, pas continuer a
  lancer des campagnes perf.
- Le DSL `production` est considere corrige cote amont par
  `feat/dsl-production-optflags`; notre patch local documente et reprend le
  principe, mais une integration propre doit se faire par merge/cherry-pick
  plutot que par empilement manuel sur ce worktree d'audit.
- Les comparaisons Hoffart restent explicitement hors benchmark perf.

# Limitations connues

Liste honnete et consolidee des limites actuelles d'`adc_cpp`, pour eviter toute attente
erronee. Chaque point renvoie a la source de verite correspondante. Ce ne sont pas des bugs :
ce sont les bords du perimetre valide a ce jour.

## GPU : valide manuellement sur ROMEO, pas en CI

La CI ne construit jamais `-DADC_USE_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON`. Toute la validation
GPU (Kokkos Cuda mono-GPU, et MPI + Kokkos Cuda multi-GPU) est faite a la main sur le
supercalculateur ROMEO (noeud GH200, `Kokkos_ARCH_HOPPER90`, `nvcc_wrapper`, OpenMPI
CUDA-aware), via des harnesses SBATCH dans `python/tests/gpu/` (exclus du glob CI). Une mention
"GPU-valide" signifie donc "manuellement sur ROMEO GH200", avec l'evidence chiffree citee dans
[GPU_RUNTIME_PORT.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md) et [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
Plusieurs cellules GPU de la matrice restent `?` (non encore exercees sur device) ; voir la
section "Lacunes notables" du document source.

## DSL : parite asserte seulement si un compilateur C++23 est present

Le DSL symbolique (`adc.dsl`) compile un modele en `.so` a l'execution (backends `aot` /
`production`) en invoquant le compilateur C++ contre les en-tetes d'`adc_cpp`. La verification
de parite (DSL vs brique native) repose donc sur la presence d'un compilateur C++23 fonctionnel
sur la machine. Sans toolchain C++23, la compilation des modeles DSL echoue ; seuls les chemins
purement natifs (`adc.Model(...)` / `add_block`) restent disponibles.

## Backends du DSL : prototype/aot sont CPU-only

`m.compile(..., backend=...)` (defaut `aot`) :

- `prototype` (JIT) et `aot` (host-marshale) sont CPU-only : pas de MPI, pas d'AMR, pas de
  GPU, mono-rang. Leur ABI `.so` plate ne transporte ni le `stride` (cadence multirate) ni
  `evolve=False` ni le masque IMEX partiel ; ces options sont rejetees explicitement (route
  explicite plutot qu'ignore silencieux).
- `production` est le seul branchable sur AMR (`AmrSystem.add_equation` exige
  `backend='production'`, `target='amr_system'`) et le seul qui passe par le chemin natif
  zero-copie ; meme la, depuis Python, le `stride` et le masque IMEX partiel sont rejetes sur le
  chemin `.so` (l'ABI du loader ne les transporte pas). Detail dans
  [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Poisson FFT : refuse sous MPI

Le solveur Poisson spectral (`PoissonFFTSolver`) est mono-rang par conception. Sous MPI (`n_ranks >
1`), la voie FFT est refusee avec une erreur collective ; un test verrouille cette
non-regression. Pour un Poisson distribue MPI, utiliser le multigrille geometrique
(`geometric_mg`).

## AMR : pas de Schur global

L'etage source condense par Schur (`adc.CondensedSchur` via `adc.Split` / `adc.Strang`) n'a pas
de pendant AMR. `set_source_stage` n'est cable que sur `System` (cartesien ou polaire), pas sur
`AmrSystem` : `AmrSystem.add_block` et `AmrSystem.add_equation` rejettent explicitement une
politique `adc.Split`/`adc.Strang`. Pour le couplage Lorentz electrostatique condense, utiliser
un `System` non raffine. Conception :
[SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md),
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Geometrie polaire : mono-rang, ExB scalaire seulement

Le maillage polaire (`adc.PolarMesh`, anneau global (r, theta)) est branche dans `System.step`
(transport polaire + Poisson polaire + aux en base locale e_r/e_theta), mais avec des bords
nets :

- transport ExB scalaire uniquement : limiteur / Riemann fluides ne sont pas leves cote
  polaire ;
- mono-rang : le solveur polaire direct (boite unique couvrant l'anneau) refuse MPI
  (`n_ranks > 1` leve) ; le pendant polaire de `CondensedSchur` est lui aussi mono-rang ;
- pas de couplage cartesien <-> polaire : l'anneau polaire est un domaine global a part.

## Deux-fluides AP : scenario dans adc_cases, pas une brique du coeur

L'integrateur asymptotic-preserving deux-fluides isotherme n'est pas une brique composable du
coeur : c'est un scenario. Sa stabilisation AP couple la raideur au pas de temps dans
l'elliptique, ce que la composition `System` ne reproduit pas. Sa physique C++ vit dans
`adc_cases/two_fluid_ap/` (`two_fluid_ap.hpp`), compilee a la volee contre les en-tetes
generiques d'`adc_cpp` ; le module `_adc` ne l'expose pas et `adc` ne le reexporte pas.

## Modele complet de Hoffart : reproduction non etablie

La reproduction quantitative du modele Euler-Poisson magnetise complet de Hoffart et al.
(arXiv:2510.11808) n'est pas etablie a ce jour. Le chemin reduit ExB-scalaire (polaire) atteint
le taux de croissance cible du diocotron, mais le modele complet cartesien (`run.py`
`system-schur`) ne reproduit pas la croissance du papier : ses runs courts ecrasent le taux. Le
verrou identifie est la geometrie (le carre cartesien + mur de Poisson circulaire diffuse le
bord d'anneau) ; la piste retenue est un domaine disque conservatif (cut-cell), non un carre.
Etat detaille et statuts par aspect dans
[HOFFART_FIDELITY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HOFFART_FIDELITY.md) (la roadmap
[FULL_MODEL_VALIDATION_ROADMAP.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/FULL_MODEL_VALIDATION_ROADMAP.md) est conservee pour
l'historique mais explicitement supersedee par cet audit).

## Footgun d'import : le module est lie a un cpython precis

Le module Python (`adc._adc`) est un `.so` lie a l'interpreteur qui l'a compile (p.ex. un
`.so` `cpython-312`). Consequences observees :

- l'importer sous un interpreteur d'une autre version (p.ex. un `python3` systeme 3.9) echoue,
  avec un message qui nomme desormais le tag attendu et la commande de reconstruction ;
- sans numpy, `import adc` et `adc.System` fonctionnent ; seul `adc.dsl` (evaluateur hote)
  echoue, avec un message qui demande numpy.

Il faut donc utiliser exactement l'interpreteur 3.12 qui a construit le module (avec numpy), et
pointer `PYTHONPATH` sur le `build*/python` correspondant, ou reinstaller avec le backend voulu
(`ADC_USE_KOKKOS=ON pip install .`). Voir [installation](../getting_started/installation.md).

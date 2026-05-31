# Conception : hero-run AMR distribué (objectif B)

But : faire tourner un diocotron à AMR dynamique à très grande échelle sur ROMEO
(GH200, MPI + Kokkos/CUDA), pour atteindre le taux analytique `0.911` au mode 4
avec bien moins de cellules que l'uniforme (M2b a chiffré le gain : même physique
pour ~43 % des cellules à résolution effective égale). Ce document est le PLAN ;
aucun code n'est écrit ici.

Convention de validation (héritée du dépôt, non négociable) : chaque étape est
vérifiée BIT À BIT identique `np=1/2/4` contre une référence (rang 0 / état
répliqué), masse conservée à l'arrondi, et le portage GPU bit-identique au CPU.

## État distribué actuel (ce qui MARCHE déjà)

- **Reflux multi-patch distribué, 2 et N niveaux** : `amr_step_2level_multipatch`
  et `subcycle_level_mp` / `amr_step_multilevel_multipatch` tournent réellement
  distribués (`test_mpi_amr_multipatch`, `test_mpi_amr_multipatch3`, `np=1/2/4`
  bit-identiques). Niveau 0 répliqué, niveaux > 0 répartis ; `average_down` et
  reflux par buffer grossier global + `all_reduce_sum_inplace` ; registre
  RESTREINT à l'interface coarse-fine (`O(interface)`, plus `O(NX*NY)`).
  (`include/adc/integrator/amr_reflux_mf.hpp`.)
- **Coupleur AMR distribué** : `AmrCouplerMP` ordonne le tout ; son injection d'aux
  `coupler_inject_aux_mb` passe par `parallel_copy` quand le parent est réparti
  (`test_mpi_coupler_inject`). (`include/adc/coupling/amr_coupler_mp.hpp`.)
- **Équilibrage SFC** : `make_sfc_distribution` (Z-order de Morton) BRANCHÉ et
  vérifié sous AMR distribué (`maxdiff = 0` vs rang 0, `np=1/2/4`).
- **Seam d'exécution** : `for_each_cell` choisit UN backend à la compilation
  (série / OpenMP / Kokkos). Sous Kokkos les noyaux partent sur le GPU.
  (`include/adc/mesh/for_each.hpp`.)
- **GPU uniforme** : `examples/gpu/coupled_kokkos.cpp` fait le pas Euler-Poisson
  COMPLET sur GPU (uniforme, mono-niveau), bit-identique au CPU.
- **MG géométrique** : `GeometricMG` applique des stencils PAR FAB, restriction et
  prolongation par noyaux de box, sans gather niveau-à-niveau : structurellement
  prêt pour un grossier multi-box réparti, MAIS jamais exercé tel quel (le
  diocotron garde le grossier répliqué). (`include/adc/elliptic/geometric_mg.hpp`.)

## Les trois verrous pour le hero-run

1. **Grossier répliqué** (niveau 0 sur chaque rang, `amr_coupler_mp.hpp:137`) :
   mémoire `O(NX*NY*nranks)`. Tolérable pour un grossier modéré sur quelques GH200,
   IMPOSSIBLE à grossier élevé ou sur toute la machine.
2. **Regrid des niveaux répartis (gather-tags)** : le tagging est local par rang,
   le clustering Berger-Rigoutsos tourne sur un `TagBox` LOCAL ; rassembler les
   tags répartis avant clustering n'est PAS implémenté (`include/adc/amr/tag_box.hpp:11`).
   `comm.hpp` n'a pas de `gather`/`broadcast`, seulement `all_reduce`.
3. **GPU + AMR jamais combinés** : le GPU ne tourne qu'en uniforme. `parallel_copy`,
   le gather du reflux, et le regrid (tagging + clustering hôte) ont une logique
   côté hôte à auditer pour le GPU (mémoire device, CUDA-aware MPI, copies
   device <-> hôte aux points de communication et de clustering).

## L'insight qui ordonne le plan

**Un diocotron à 2 NIVEAUX (grossier répliqué + un niveau fin réparti) ne touche
AUCUN des deux verrous durs.** Le regrid tague le niveau 0, qui est RÉPLIQUÉ :
chaque rang voit les mêmes tags, le clustering local est donc cohérent entre rangs
sans gather. Le gather-tags ne mord que pour 3+ niveaux où l'on tague un niveau
INTERMÉDIAIRE réparti. Et la dé-réplication du grossier n'est requise que lorsque
le grossier devient trop gros pour être répliqué.

Donc on peut livrer un hero-run AMR distribué utile AVANT de toucher la
dé-réplication ou le gather-tags, en s'appuyant sur le reflux distribué déjà prouvé.
Le risque est repoussé en fin de plan, quand il devient nécessaire.

## Plan étagé

### Étape 0 : diocotron AMR distribué de bout en bout (CPU), grossier répliqué (FAIT)

`test_mpi_diocotron_amr` : le diocotron 2 niveaux sur `AmrCouplerMP` (Poisson
grossier + injection d'aux + pas multi-patch + regrid Berger-Rigoutsos) tourne
distribué, `np=1/2/4` BIT À BIT identique (`max|Uc_dist - Uc_ref| = 0`) et masse
conservée à l'arrondi (`2e-15`) sous regrid dynamique réparti.

GAP TROUVÉ ET CORRIGÉ au passage : le multigrille de Poisson du coupleur n'était
PAS répliqué sous MPI. `GeometricMG` distribuait son grossier mono-box en
round-robin (`DistributionMapping(1, n_ranks())` -> box 0 sur le seul rang 0),
alors que `compute_aux` lit `mg_.phi().fab(0)` et injecte avec
`replicated_parent=true` sur CHAQUE rang : sous MPI, les rangs > 0 n'avaient pas
de grossier à lire. Aucun test MPI n'exerçait le `AmrCouplerMP.step()` complet (ils
testaient le reflux nu et la primitive d'injection seuls), d'où le trou. Correctif :
option `replicated` sur `GeometricMG` (chaque rang détient + résout le même Poisson
grossier, sans communication : V-cycle par-fab, `fill_boundary` local sur une box
couvrant le domaine, résidu par `norm_inf` = `all_reduce_MAX` idempotent) ; option
`replicated_coarse=true` sur `AmrCouplerMP` qui la lui passe. En série c'est
bit-identique au round-robin (60/60 inchangés).

- **Livrable** : `AmrCouplerMP` réellement MPI-correct de bout en bout (gate ci-dessus).
- **Risque réel** : ce n'était PAS du pur assemblage : un vrai trou de réplication
  du Poisson a été révélé et corrigé. D'où l'intérêt de la vérification.

### Étape 1 : porter l'étape 0 sur GPU (Kokkos), grossier répliqué

Faire tourner ce même pilote sous Kokkos/CUDA. Audit des points hôte/device de
l'AMR distribué : (a) `parallel_copy` entre fabs device de rangs différents
(CUDA-aware MPI, `openmpi +cuda`) ; (b) le gather du reflux (`all_reduce` sur un
buffer : copie device -> hôte -> all_reduce -> hôte -> device, avec `device_fence`) ;
(c) le regrid (le tagging lit des données device ; choix : noyau de tag device +
clustering hôte sur un `TagBox` rapatrié, le clustering étant bon marché).

VALIDÉ SUR GH200 (FAIT). `examples/gpu/diocotron_amr_kokkos.cpp` exécute tout le pas
AMR (coupleur : Poisson grossier + injection d'aux + reflux multi-patch
coverage-aware + regrid Berger-Rigoutsos) SOUS LE SEAM KOKKOS. Lancé sur un GH200
d'`armgpu` via `romeo/diocotron_amr_gpu.sbatch` : espace d'exécution `Cuda`, le pas
AMR tourne sur le GPU et le résultat est BIT À BIT IDENTIQUE au CPU
(`checksum = 4394594.404318` identique à la voie série et Kokkos-OpenMP locale ;
4 patchs ; `derive_masse ~ 2.9e-15`, l'écart au CPU `2.2e-15` venant de la somme de
masse réassociée par CUDA, pas de l'état). La référence uniforme `coupled_kokkos`
tourne aussi en `exec=Cuda`, `dmasse = 0`. Le verrou « GPU + AMR jamais combinés »
est donc levé : le pas AMR complet tourne sur GH200, bit-identique au CPU.

Note ROMEO : Kokkos n'est PAS fourni par spack/module (la recette `spack load
kokkos` était erronée) ; le `.sbatch` le compile depuis les sources (Serial + CUDA,
Hopper sm_90, nvcc_wrapper), une fois, en cache sur `/scratch_p`. Compte : `r250127`.

MULTI-GPU VALIDÉ (FAIT). `examples/gpu/diocotron_amr_mpi_kokkos.cpp` lance le diocotron
AMR distribué sur 4 GH200 d'un nœud `armgpu` (1 rang MPI par GPU), via
`romeo/diocotron_amr_mpi_gpu.sbatch` (build MPI + Kokkos/CUDA, module
`openmpi/aarch64/4.1.7-cuda`). Le `parallel_copy` et les `all_reduce` du reflux
déplacent les données entre fabs DEVICE de rangs différents (MPI CUDA-aware). Gate
d'invariance à la distribution : `exec=Cuda, np=4, max|Uc_dist - Uc_ref| = 0.000e+00`,
BIT À BIT identique entre patchs round-robin (DIST) et tous-sur-rang-0 (REF). Le pas
AMR distribué tourne donc réellement multi-GPU et le résultat ne dépend pas de la
répartition des patchs.

- **Livrable** : pas AMR distribué validé bit-identique sur 1 ET 4 GH200 réels. Le
  verrou « GPU + AMR jamais combinés » est entièrement levé (mono ET multi-GPU).
- **Reste (perf, pas correction)** : binding GPU optimal, multi-nœud (Infiniband),
  et l'échelle. La CORRECTION multi-GPU est acquise.

PERFORMANCE (constat HONNETE, mesuré). `concurrency()` confirme le parallélisme RÉEL :
`270336` threads sur le GH200 (espace Cuda), `1`/`8` sur CPU OpenMP selon
`OMP_NUM_THREADS` (donc PAS du monothread déguisé). MAIS parallèle != rapide ici : le
pas AMR (80 pas) prend `29 s` à nc=64 et `174 s` à nc=1024 sur GPU, contre `1.85 s`
(nc=64) sur UN thread CPU. Le GPU est donc plus lent que 1 cœur CPU à ces tailles, et
le CPU OpenMP est lui-même plus lent à 8 threads qu'à 1 (`210 s` vs `1.85 s` à nc=64).
Cause : le pas AMR lance une myriade de PETITS kernels (chaque V-cycle = dizaines de
`gs_rb_sweep` x `fill_ghosts` x `device_fence`) et le regrid Berger-Rigoutsos est
CÔTÉ HÔTE (série) ; à ces tailles la latence de lancement + les syncs device<->hôte
dominent le calcul. Le code est donc CORRECT en parallèle (bit-identique, 270k threads)
mais PAS optimisé. L'accélération demanderait fusion de kernels, regrid résident GPU,
moins de fences : un chantier de PERF distinct de la correction, hors périmètre du
hero-run AMR (qui vise d'abord la correction distribuée et l'échelle mémoire).

### Étape 2 : dé-réplication du grossier (objectif B proprement dit)

Requise seulement quand le grossier doit croître au-delà de ce que la réplication
permet. État : le coeur 2a/2b FONCTIONNE, prouvé bit-identique np=1/2/4 (bug de désync du
un bug `parallel_copy` à np=4 et le gather-tags 2c.

- **2a. Grossier multi-box réparti (FAIT, np=1/2/4).** `AmrCouplerMP` accepte un grossier
  multi-box réparti (`replicated_coarse=false`) : `compute_aux` boucle sur les fabs
  grossiers LOCAUX (au lieu de `fab(0)`), `mass()` fait un `all_reduce` quand le grossier
  est réparti (somme locale sinon), `max_drift_speed()` un `all_reduce_max`, et
  l'injection d'aux passe par `parallel_copy` (chemin `replicated_parent=false`). Tout
  reste BIT À BIT identique au chemin répliqué (série 60/60, AMR répliqué np=1/2/4
  `maxdiff=0`). `test_mpi_decoarse` prouve qu'un grossier multi-box 2x2 réparti donne le
  MÊME grossier que le mono-box répliqué, bit à bit, à np=1 et np=2.
- **2b. MG de Poisson sur grossier réparti (DEJA LA).** `GeometricMG` solveur multi-box
  réparti marche tel quel : `gs_rb_sweep` appelle `fill_ghosts` -> `fill_boundary`
  (échange de halos inter-box DISTRIBUÉ) entre balayages rouge/noir, et le GS red-black
  est indépendant de la décomposition -> `phi` bit-identique au mono-box. Pas de solveur
  de fond séparé nécessaire ici (le bottom smoother fait aussi `fill_ghosts`).
- **BUG TROUVÉ ET CORRIGÉ (np=4).** La de-réplication est maintenant bit-identique np=1/2/4
  (`test_mpi_decoarse` lancé à np=4, `maxdiff=0`). Racine, trouvée par dump des séquences
  d'appels `fill_boundary` par rang : `GeometricMG::current_residual()` rendait `norm_inf`
  du résidu SANS le réduire entre rangs (max LOCAL). Sur un grossier multi-box réparti,
  chaque rang voyait donc un résidu différent, et le critère d'arrêt du V-cycle se
  déclenchait à une itération différente selon le rang -> nombre de V-cycles (donc d'appels
  `fill_boundary`) DIFFÉRENT entre rangs -> les flux de messages tag-0 se désynchronisaient
  (l'échange de l'aux nc=3 d'un rang apparié à l'échange du grossier nc=1 d'un autre) ->
  `MPI_ERR_TRUNCATE`. (Le grossier RÉPLIQUÉ marchait car max local = max global.) Le commentaire
  de `norm_inf` notait d'ailleurs l'all-reduce comme « plus tard, non ajouté ici ». FIX :
  `all_reduce_max` sur le résidu dans `current_residual()` (une ligne) -> tous les rangs
  s'accordent sur le résidu -> même nombre de V-cycles -> séquences `fill_boundary`
  synchronisées. Idempotent sous réplication et identité en série : série 60/60 et les 13
  tests MPI restent verts, bit-identique au comportement historique.
- **SECOND BUG TROUVÉ ET CORRIGÉ (np=4).** Distinct de la désync du résidu : un segfault
  dès qu'un rang détient PLUSIEURS boxes grossières, ou qu'une empreinte fine borde une box
  grossière DISTANTE. Racine : le reflux multi-patch `subcycle_level_mp` codait en dur
  `replicated_parent = (lev == 0)`, donc au niveau 0 il échantillonnait le flux grossier
  bordant par `mf_find_box`. À la de-réplication le niveau 0 est RÉPARTI : une cellule
  grossière bordante peut appartenir à un rang DISTANT, `mf_find_box` rendait alors -1 puis
  `fx.fab(-1)` -> segfault (adresse fautive 0x40/0x80, null+offset). Le cas 4 boxes (1 par
  rang à np=4) survivait par alignement round-robin fortuit (chaque empreinte fine tombait
  dans la box grossière du même rang). FIX : on propage `replicated_coarse` du coupleur
  jusqu'à `subcycle_level_mp`, `replicated_parent = (lev == 0) && coarse_replicated`. Le
  niveau 0 de-répliqué emprunte alors le chemin `parallel_copy` (empreinte grossière par
  enfant) déjà utilisé pour les niveaux fins, MPI-correct. Régression `test_mpi_decoarse` :
  un patch fin CENTRÉ chevauchant les 4 boxes grossières (3 distantes à np=4) reproduit
  exactement l'ancien segfault et passe désormais bit-identique (`maxdiff=0`, np=1/2/4).
  Idempotent sous réplication : série 60/60 et MPI 73/73 restent verts. NB : une découpe 4x4
  (16 boxes) dégénère le fond du multigrille géométrique (16 boxes ne pavent pas la grille la
  plus grossière 2x2) et converge à un point distinct à la tolérance près, non déterministe ;
  la box distante se teste donc par le patch centré sur un grossier 2x2 propre, pas par 16 boxes.
- **2c. Gather-tags pour le regrid d'un niveau réparti (RESTE).** Ajouter à `comm.hpp` un
  `gather` (ou un `all_reduce` du `TagBox` indexé global), rassembler les tags répartis
  avant le clustering Berger-Rigoutsos (cf. `tag_box.hpp:11`). Non nécessaire à 2 niveaux
  (le niveau 0, même réparti, est tagué globalement via une réduction) mais requis pour
  raffiner un niveau intermédiaire réparti.

- **Risque** : le coeur (grossier multi-box + MG réparti) est acquis bit-identique
  np=1/2/4. Reste 2c (gather-tags) pour raffiner un niveau intermédiaire réparti.

### Étape 3 : run de production + science

`romeo/diocotron_amr_hero.sbatch` (armgpu, MPI + Kokkos/CUDA), montée en base,
mesure du taux de croissance par `validate_diocotron_growth.py`, comparaison
cellules/temps au hero-run uniforme (`diocotron_mpi`, la référence chiffrée déjà
prête). Cible scientifique : `0.911` au mode 4, pour ~43 % des cellules de
l'uniforme équivalent (la promesse de M2b, à l'échelle).

- **Gate** : taux convergé vers `0.911`, masse conservée sur tout le run, scaling
  fort et faible tracés.

- **Résultats réels (GH200, ROMEO `armgpu`).** Le pipeline tourne bout en bout sur H100 :
  `diocotron_column_amr` (rendu Kokkos-compatible, init `Kokkos::ScopeGuard`) se construit sous
  Kokkos/CUDA (`nvcc_wrapper`, sm_90) et s'exécute sur le GPU. Reproduction EXACTE de la table
  M2b-conv sur matériel réel, taux extrait par `validate_diocotron_growth.py` (rhobar=0.9,
  omega_D=0.143) :

  | cas (1 GPU GH200 H100) | cellules | gamma_norm | table M2b |
  |---|---|---|---|
  | uniforme eff 448 (nc=448) | 200 704 | 0.577 | 0.577 |
  | AMR `ml` eff 448 (nc=224) | 82 808  | 0.592 | 0.592 |

- **Barrière vers `0.911` LEVÉE : le multigrille divergeait au bord embedded.** Au-delà de eff 448
  la simu partait en `nan` dès les premiers pas, aux DEUX schémas (uniforme et AMR `ml`). Diagnostic
  complet, par instrumentation du résidu MG et de `vmax` localisé :
  - Ce n'est NI le pas de temps (déjà plafonné à `dt0`, la dérive E x B de l'anneau étant
    ~constante), NI le plancher de densité : pendant la divergence la densité RESTE bornée dans
    `[1e-3, 1]`, seul `phi` explose. Le `vmax` partait du BORD CONDUCTEUR (`r = 0.398 = Rwall`), pas
    du bord d'anneau.
  - Cause RÉELLE : le V-cycle géométrique DIVERGE près de la paroi conductrice sur grille fine. Le
    coarsening est NON-Galerkin et le masque du cercle est re-évalué à chaque niveau, donc la
    correction grossière devient incohérente avec le bord fin ; le lissage `nu1=nu2=2` ne la domine
    plus et le rayon spectral du cycle passe > 1. C'est ERRATIQUE en résolution (selon l'alignement
    du cercle sur la hiérarchie : eff 640/1280/2048 divergent, 512/896 stagnent seulement). Le warm
    start propage la divergence d'un pas à l'autre -> `phi` puis le champ en `nan`. Mesure : à
    nc=640 le résidu MG monte (`ratio = r_fin / r_0 = 2.7e2`) là où nc=224 converge (`5.7e-9`).
  - Correctif : `GeometricMG::solve_robust` (`include/adc/elliptic/geometric_mg.hpp`). Il lance le
    V-cycle standard, EXACTEMENT comme `solve()` (donc BIT-IDENTIQUE quand il converge ou stagne) ;
    SEULEMENT si le résidu final EXCÈDE le résidu initial (vraie divergence, pas une stagnation
    `ratio < 1` qu'on garde telle quelle) il durcit le lissage GS de façon STICKY (nu double,
    conservé pour les pas suivants) et repart à froid (`phi = 0`) jusqu'à redevenir contractant.
    Plus de lissage rend le cycle contractant (le GS domine la correction grossière incohérente :
    `nu = 2` diverge à nc=640, `nu >= 4` converge). L'exemple appelle `solve_robust` pour les deux
    Poisson (`mg` grossier et `fmg` multi-niveau).
  - **Vérification.** Les 8 runs enregistrés de la table M2b-conv (uniforme eff 192-448, AMR `ml`
    eff 192-448) restent BIT À BIT identiques (`diff` des `ring_amp.csv`), car aucun ne divergeait :
    `solve_robust` n'y déclenche jamais le durcissement. La suite elliptique reste verte
    (`test_geometric_mg`, `test_poisson_convergence` ordre 2.00, `test_elliptic_operator` MG=FFT,
    `test_gauss_law` ordre 2.00). Et le balayage monte désormais sans `nan` jusqu'à eff 1024
    (uniforme ET AMR `ml`, y compris le cas AMR `ml` base 320 à 66 patchs ; masse conservée `~1e-14`).
  - **Science débloquée.** La table M2b-conv est prolongée jusqu'à eff 1024 (voir `docs/ROADMAP.md`,
    M2b-conv-HR). En phase linéaire (mesure robuste `--window 5,14`) le taux CONTINUE sa montée
    monotone vers `0.911` : eff 448 -> 1024 donne `0.63 -> 0.65 -> 0.67 -> 0.70 -> 0.71` (uniforme
    comme AMR), l'AMR `ml` suivant l'uniforme à ~40 % des cellules. La mesure historique relative au
    pic plafonne, elle, à ~0.58 (biais de fenêtre : le rollover de saturation se raidit avec la
    résolution). Atteindre `0.911` reste affaire de résolution encore plus haute, désormais
    accessible sans blocage numérique (objet du hero-run distribué).

## Décisions ouvertes (à trancher avant de coder)

1. **Jusqu'où va-t-on ?** Étapes 0-1 (AMR distribué GPU à grossier répliqué)
   donnent déjà un hero-run AMR qui bat l'uniforme à base modérée. L'étape 2
   (dé-réplication) n'est nécessaire que pour un grossier élevé / toute la machine.
   On peut s'arrêter après 1 et mesurer, ou pousser jusqu'à 2.
2. **2 niveaux ou plus ?** À 2 niveaux, le gather-tags (2c) est inutile (le niveau
   0 répliqué porte les tags). Rester à 2 niveaux simplifie fortement ; passer à
   3+ niveaux distribués impose 2c.
3. **Limite de dérive (Diocotron) ou système complet (M3) au hero-run ?** Le
   Diocotron passe par `AmrCouplerMP` (aux par le flux) ; le magnétique complet
   (`MagneticEulerPoissonCoupler`) n'a PAS encore de variante AMR multi-patch (il
   enveloppe `Coupler<EulerPoisson>`, mono-niveau). Faire l'AMR magnétique demande
   d'envelopper `AmrCouplerMP<EulerPoisson>` dans le splitting de Strang : faisable
   mais c'est un chantier en soi, à placer après l'étape 1.

## Chemin recommandé

Étape 0 -> 1 d'abord (risque faible/moyen, livre un vrai hero-run AMR GPU à
grossier répliqué, 2 niveaux, limite de dérive), mesurer le gain vs l'uniforme à
l'échelle ; puis décider de l'étape 2 (dé-réplication) selon la base atteignable et
de l'AMR magnétique selon le temps. Le risque dur (solveur de fond réparti,
gather-tags) n'est engagé que s'il devient nécessaire, et chaque étape reste
vérifiable au standard bit-identique du dépôt.

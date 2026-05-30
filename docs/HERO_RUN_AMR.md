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

DE-RISK LOCAL FAIT (sans matériel CUDA) : `examples/gpu/diocotron_amr_kokkos.cpp`
exécute tout le pas AMR (coupleur : Poisson grossier + injection d'aux + reflux
multi-patch coverage-aware + regrid Berger-Rigoutsos) SOUS LE SEAM KOKKOS, espace
d'exécution hôte OpenMP. Le `for_each_cell` qui partira sur le GPU (espace Cuda sur
GH200) est donc déjà exercé. Résultat BIT À BIT IDENTIQUE à la voie série (même
`checksum`, même `derive_masse = 2.2e-15`, mêmes 4 patchs) : les noyaux AMR sont
des `parallel_for` par cellule sans réduction de somme dans le chemin chaud (les
réductions sont des `norm_inf` = max, exactement associatif), donc l'ordre
d'exécution ne change pas le résultat. Reste STRICTEMENT ROMEO : la mémoire device
réelle et le `parallel_copy` CUDA-aware entre fabs device de rangs différents.

- **Livrable** : algorithme AMR prouvé correct sous le seam Kokkos (de-risk local
  fait) ; reste le run CUDA réel + MPI CUDA-aware sur GH200.
- **Gate ROMEO** : GPU bit-identique au CPU (checksum), masse conservée ; un run
  réel sur 1 puis 4 GH200 d'un nœud `armgpu`.
- **Risque** : MOYEN, abaissé par le de-risk local (le pas AMR sous le seam est
  validé ; ne restent que les coutures device/CUDA-MPI).

### Étape 2 : dé-réplication du grossier (objectif B proprement dit)

Requise seulement quand le grossier doit croître au-delà de ce que la réplication
permet. Trois sous-chantiers, chacun avec son gate bit-identique :

- **2a. Grossier multi-box réparti** : remplacer le niveau 0 mono-box répliqué par
  une `BoxArray` multi-box + `DistributionMapping(ba.size(), n_ranks())`. Adapter
  `AmrCouplerMP` (l'injection d'aux a déjà le chemin `replicated_parent=false` par
  `parallel_copy`) et la mesure de masse (réduction globale au lieu de locale).
- **2b. MG de Poisson sur grossier réparti** : `GeometricMG` applique déjà des
  stencils par fab ; ajouter (i) l'échange de halos inter-box à chaque niveau du
  V-cycle (déjà via `fill_boundary` réparti), (ii) un SOLVEUR DE FOND réparti
  (rassembler le niveau le plus grossier sur un rang, résoudre, rediffuser ; OU
  continuer le coarsening jusqu'à une box mono-rang). Gate : `div(grad phi)=source`
  ordre 2 + bit-identique `np=1/2/4` sur un grossier multi-box.
- **2c. Gather-tags pour le regrid d'un niveau réparti** : ajouter à `comm.hpp` un
  `gather` (ou un `all_reduce` du `TagBox` indexé global, même schéma que le
  registre de flux), rassembler les tags répartis sur la grille de tag avant le
  clustering Berger-Rigoutsos (cf. `tag_box.hpp:11`). Gate : regrid d'un niveau
  intermédiaire réparti bit-identique `np=1/2/4` (étendre `test_mpi_amr_multipatch3`).

- **Risque** : ÉLEVÉ (2b le solveur de fond réparti surtout). C'est le coeur de
  l'objectif B, déclaré NO-GO pour un PETIT grossier mais REQUIS ici.

### Étape 3 : run de production + science

`romeo/diocotron_amr_hero.sbatch` (armgpu, MPI + Kokkos/CUDA), montée en base,
mesure du taux de croissance par `validate_diocotron_growth.py`, comparaison
cellules/temps au hero-run uniforme (`diocotron_mpi`, la référence chiffrée déjà
prête). Cible scientifique : `0.911` au mode 4, pour ~43 % des cellules de
l'uniforme équivalent (la promesse de M2b, à l'échelle).

- **Gate** : taux convergé vers `0.911`, masse conservée sur tout le run, scaling
  fort et faible tracés.

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

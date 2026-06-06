# Conception : regrid par UNION DES TAGS sur la hierarchie multi-blocs partagee (Phase 2)

DESIGN-ONLY. Aucune implementation. Ce document est une SPEC raisonnee, a VALIDER PAR L'OWNER
AVANT toute ligne de code. Il decrit l'algorithme qui rend ADAPTATIVE la hierarchie AMR
multi-blocs aujourd'hui FIGEE, et n'introduit aucun fichier de code.

CADRE DU PROJET. Le runtime multi-blocs livre jusqu'ici (`AmrRuntime`,
`include/adc/runtime/amr_runtime.hpp`, pendant runtime de `AmrSystemCoupler`,
`include/adc/coupling/amr_system_coupler.hpp`) est la "Phase 1 multi-bloc a hierarchie FIGEE" :
N blocs (electrons, ions, neutres, ...) co-localises sur UNE hierarchie AMR partagee, un seul
Poisson grossier a second membre SOMME, sources couplees cellule a cellule, multirate par bloc
(substeps / stride / evolve), mais une grille qui NE BOUGE JAMAIS apres la construction. Ni
`AmrSystemCoupler` ni `AmrRuntime` n'ont de methode `regrid` (verifie : grep `regrid` dans les
deux fichiers ne rend rien). La facade Python REFUSE donc explicitement la combinaison
multi-blocs + `regrid_every > 0` (`python/amr_system.cpp:246-251`).

OBJET DE CE DOCUMENT (Phase 2, la finale du capstone). Specifier le `regrid` par UNION DES TAGS
qui transforme cette hierarchie figee en hierarchie ADAPTATIVE : un seul critere collectif
(l'union de tous les blocs), un seul clustering Berger-Rigoutsos, un seul nouveau layout PARTAGE,
puis prolongation / restriction / reflux PAR BLOC sur ce layout. Ce document DEVERROUILLE
explicitement multi-blocs + `regrid_every > 0`, aujourd'hui refuse.

Le modele cible reste celui d'AMReX / FLASH / SAMRAI deja pose par
`docs/AMR_MULTIBLOCK_DESIGN.md` (section 5) : UNE hierarchie commune raffinee par l'UNION des
criteres de tous les champs, JAMAIS une hierarchie par espece.


## 0. Note d'honnetete liminaire (etat reel du code a ce head)

Le code a ete relu directement. Quatre faits structurent toute la suite.

FAIT 1 : la BRIQUE DE REGRID MONO-BLOC EXISTE et est eprouvee. `amr_regrid_finest`
(`include/adc/coupling/amr_regrid_coupler.hpp`) fait, pour UN bloc : tag du parent
(`tag_cells`) -> `grow_tags` (nesting + marge) -> `all_reduce_or_inplace` si grossier reparti
-> `berger_rigoutsos` -> clamp de nesting (margin) -> nouveau `BoxArray` fin + un seul
`DistributionMapping(nfine, n_ranks())` -> report des donnees fines existantes + interpolation
depuis le parent ailleurs -> realloc de l'aux du niveau fin (adresse re-cablee). C'est exactement
le squelette dont la version multi-blocs a besoin ; il manque uniquement l'orchestration "un seul
layout pour tous les blocs".

FAIT 2 : le REGRID MONO-BLOC EST DEJA CABLE A LA FACADE, mais SEULEMENT pour le chemin mono-bloc.
`AmrCouplerMP::regrid` (`include/adc/coupling/amr_coupler_mp.hpp:321-325`) delegue a
`amr_regrid_finest`, et la fermeture `h.step` du chemin mono-bloc l'appelle periodiquement
(`include/adc/runtime/amr_dsl_block.hpp:101-104` :
`if (regrid_every > 0 && *step_state % regrid_every == 0) cpl->regrid(crit);`). Le chemin
multi-blocs, lui, passe par `AmrRuntime` qui n'a aucun `regrid` : sa hierarchie est figee.

FAIT 3 : le REFUS multi-blocs + `regrid_every > 0` EST DEJA EN PLACE, par conception (correction
owner, `docs/AMR_MULTIBLOCK_DESIGN.md` section 4). Il est cable a `ensure_built` de la facade
(`python/amr_system.cpp:246-251`) : autoriser silencieusement `regrid_every > 0` en multi-blocs
ferait PRETENDRE a l'API qu'elle fait de l'AMR dynamique alors que la grille ne bouge jamais
(illusion dangereuse). LEVER ce refus est precisement ce que CE DESIGN autorise, une fois
l'algorithme ci-dessous implemente et teste.

FAIT 4 : le GARDE-FOU DE LAYOUT EXISTE et sera le filet de securite naturel de l'apres-regrid.
`detail::same_layout_or_throw` (`include/adc/coupling/amr_system_coupler.hpp:122-140`, reutilise
par `AmrRuntime` au ctor, `amr_runtime.hpp:213-218`) compare EXACTEMENT, entre tous les blocs :
nombre de niveaux, puis par niveau `BoxArray` (boites ET ordre, via `ba.boxes() ==`),
`DistributionMapping` (rang par boite, via `dm.ranks() ==`) et `dx`/`dy` (au bit pres). C'est la
PRECONDITION exacte de l'aux partage : tous les blocs vivent sur EXACTEMENT le meme layout par
niveau. Le regrid doit RE-ETABLIR cet invariant apres avoir bouge la grille (cf. section 4,
verification (V3)). Le reutiliser comme assertion post-regrid coute une ligne et attrape toute
incoherence de reconstruction.

CONSEQUENCE. Le travail de Phase 2 n'est PAS d'inventer un regrid, mais d'ORCHESTRER la brique
mono-bloc existante autour d'un critere COLLECTIF (union des tags) et d'un layout UNIQUE applique
a TOUS les blocs, puis de lever le refus de la facade et d'ajouter les tests de changement de
layout. Bonne nouvelle a dire franchement : la cible est plus proche qu'il n'y parait.


## 1. Decision d'architecture et pourquoi

DECISION (verrouillee par l'owner). UNE seule hierarchie AMR PARTAGEE pour TOUS les blocs, JAMAIS
une hierarchie par espece en v1. Le regrid est pilote par l'UNION COLLECTIVE des tags de tous les
blocs (tags = electrons OU ions OU neutres OU phi OU utilisateur), calculee a travers tous les
blocs ET tous les rangs MPI, suivie d'UN SEUL clustering Berger-Rigoutsos qui produit UN SEUL
nouveau layout partage. Prolongation / restriction / reflux sont appliques PAR BLOC sur ce
nouveau layout partage : chaque bloc est re-grille sur le layout d'union ; TOUS les blocs evolues
sont presents sur TOUS les patchs, JAMAIS spatialement absents.

POURQUOI un regrid d'UNION sur une hierarchie partagee, et NON un regrid par espece :

- AUX PARTAGE ET POISSON UNIQUE. L'aux (phi, grad phi, [B_z, T_e]) est PARTAGE par niveau
  (`AmrRuntime::aux_`, un `MultiFab` par niveau, `amr_runtime.hpp:233-238` ; idem
  `AmrSystemCoupler::aux_`). Le pointeur aux de chaque `AmrLevelMP` de chaque bloc pointe vers ce
  meme `MultiFab` partage. Cela N'A DE SENS que si tous les blocs vivent sur EXACTEMENT le meme
  layout par niveau. Un regrid qui produirait des layouts par espece casserait immediatement
  l'aux unique et le Poisson unique : il faudrait interpoler chaque densite vers une grille
  commune a chaque solve. L'union des tags garantit un layout unique par construction.

- CO-LOCALISATION DES SOURCES COUPLEES. Les sources inter-especes lisent plusieurs especes dans
  la MEME cellule (`AmrRuntime::coupled_source_step`, `amr_runtime.hpp:372-407` : `kern.in[c]` et
  `kern.out[t]` indexent le MEME `(i,j)`, meme `fab(li)`, par niveau). Cette lecture cellule-locale
  exige que tous les blocs partagent le layout. Un regrid d'union le preserve ; un regrid par
  espece le detruirait.

- CONSERVATION ET REFLUX PAR BLOC. Le reflux est deja bloc par bloc (chaque bloc a ses propres
  registres de flux dans `advance_amr`, fermeture `AmrRuntimeBlock::advance`). Le regrid doit
  conserver la masse de CHAQUE bloc independamment, ce qui suppose que la grille soit la meme pour
  tous (sinon les interfaces coarse-fine differeraient par bloc et le reflux ne serait plus
  comparable). Cf. section 3, etape (R5) et verification (V1).

- MPI / KOKKOS : un seul plan de distribution. Une hierarchie partagee = un seul
  `DistributionMapping` par niveau, donc une seule reduction collective des tags et un seul jeu de
  halos. Le tag-union doit donc etre une COLLECTIVE cross-rang (cf. section 3, etape (R3), et
  section 6) : tous les rangs doivent partir de la MEME grille de tags pour que Berger-Rigoutsos
  produise des patchs IDENTIQUES par rang, sinon les `DistributionMapping` divergent et MPI
  desynchronise. C'est exactement la garde MPI-safe deja presente dans `amr_regrid_finest:59-60`,
  qu'il faut hisser au niveau de l'union.

CE QU'ON N'OUVRE PAS (et pourquoi le dire). Pas de hierarchie par espece (Phase 3,
`docs/AMR_MULTIBLOCK_DESIGN.md` section 7). Pas de critere par bloc qui produirait des grilles
distinctes : en v1 le critere reste l'UNION, un seul layout. Pas d'absence spatiale locale d'un
bloc sur un patch : un bloc est TOUJOURS present et conservatif partout, meme si son propre
critere ne l'a pas declenche, parce qu'il est couple aux autres. Ces restrictions sont
volontaires : elles preservent l'aux unique, le Poisson unique et la conservation cellule a
cellule des sources, raison d'etre du partage.


## 2. Perimetre : ce que la Phase 2 livre, et ce qu'elle ne livre pas

LIVRE (cible v1 du regrid d'union) :
- un regrid pilote par l'UNION des tags de tous les blocs + tags de phi + tags utilisateur ;
- UN clustering, UN nouveau layout partage applique a TOUS les blocs ;
- prolong / restrict / reflux par bloc sur le nouveau layout ;
- conservation de masse PAR BLOC a travers le regrid (assertion d'invariant par bloc) ;
- correction backend (MPI + Kokkos CPU et GPU) : union des tags collective cross-rang, layout
  consistant sur chaque rang ;
- la levee du refus facade : multi-blocs + `regrid_every > 0` devient SUPPORTE.

PAS LIVRE (reste conforme aux frontieres de `docs/AMR_MULTIBLOCK_DESIGN.md` section 7) :
- le regrid MULTI-NIVEAUX (> 2 niveaux). Comme `amr_regrid_finest`, le regrid d'union v1 ne
  reconstruit QUE le niveau le plus fin (grossier + 1 fin), ce que materialise le layout partage
  `make_shared_amr_layout` (`amr_dsl_block.hpp`, grossier + 1 patch fin central FIXE). Au-dela de
  2 niveaux, le regrid multi-niveaux n'existe pas encore meme en mono-bloc : limite a noter, pas a
  resoudre ici.
- les CRITERES PAR BLOC (Phase 2 etendue / Phase 3 : chaque bloc declare son critere, l'union
  reste). En v1 le critere d'union peut etre un critere unique applique a chaque champ ; le
  raffinement reste l'union.
- le vrai SOLVE elliptique multi-niveaux (composite). Le Poisson reste "coarse + inject"
  (`coupler_inject_aux_mb`), comme en Phase 1.


## 3. Algorithme de regrid d'union, etape par etape

Cible : une methode `AmrRuntime::regrid(...)` (et son pendant compile-time
`AmrSystemCoupler::regrid(...)`), appelee periodiquement par la facade (cf. section 5). On note
`pk` le niveau parent (grossier, `pk = 0` en v1 a 2 niveaux), `fk = pk + 1` le niveau fin a
reconstruire, `nlev_` le nombre de niveaux.

(R0) PRECONDITION. Champs a jour : appeler `solve_fields()` une fois (aux par niveau a jour, pour
le critere de gradient de phi, exactement comme `amr_runtime.hpp:413`). Snapshot des masses par
bloc `mass(b)` AVANT regrid (pour la verification (V1)).

(R1) TAGS PAR BLOC SUR LE PARENT. Pour chaque bloc `b`, calculer une `TagBox` sur le niveau parent
via `tag_cells((*blocks_[b].levels)[pk].U, pdom, crit_b)` (`include/adc/amr/regrid.hpp:36-47`).
`crit_b` est un predicat `(ConstArray4 a, int i, int j) -> bool` sur la densite du bloc (composante
0) ou un gradient. En v1 le critere peut etre commun a tous les blocs ; l'UNION ci-dessous reste
le contrat. `TagBox` est une grille dense de `char` 0/1 sur `pdom` (`tag_box.hpp`).

(R2) TAGS DE PHI ET TAGS UTILISATEUR. Calculer une `TagBox` sur le canal aux du parent `aux_[pk]`
(composante 0 = phi, ou son gradient discret) : `tag_cells(aux_[pk], pdom, crit_phi)`. Ajouter les
tags utilisateur optionnels (predicat fourni par l'appelant).

(R3) UNION DES TAGS (cellule a cellule). Composer toutes les `TagBox` de (R1)+(R2) en UNE `TagBox`
d'union par OU logique cellule a cellule :

    tags_union = tags_e OU tags_i OU tags_n OU tags_phi OU tags_user

Toutes les `TagBox` partagent la meme boite `pdom`, donc l'union est un `|=` par indice. NOUVEAU
CODE : un helper `tag_union(span<const TagBox>) -> TagBox` (quelques lignes, pas de dependance
physique). Puis `grow_tags(tags_union, grow, pdom)` (`regrid.hpp:52-64`) pour le nesting et la
marge.

(R4) REDUCTION COLLECTIVE CROSS-RANG (MPI). Si le grossier est REPARTI (`pk == 0 &&
!replicated_coarse_`), chaque rang n'a tague que ses boites LOCALES (`tag_cells` ne parcourt que
`mf.local_size()`, `regrid.hpp:38`). Reduire les tags UNIS par `all_reduce_or_inplace` sur le
buffer `grown.t.data()` (MEME garde MPI-safe que `amr_regrid_finest:59-60`, hissee au niveau de
l'union) pour que TOUS les rangs partent de la MEME grille de tags. Replique : la grille de tags
est deja complete sur chaque rang -> `all_reduce_or` serait l'identite (no-op, on l'evite). Cette
reduction est INDISPENSABLE a la consistance du layout cross-rang : sinon Berger-Rigoutsos
produirait des patchs differents par rang et les `DistributionMapping` divergeraient.

(R5) CLUSTERING UNIQUE -> LAYOUT PARTAGE. UN SEUL `berger_rigoutsos(grown, ClusterParams{})`
(`include/adc/amr/cluster.hpp:171-181`) sur les tags d'union reduits. Appliquer le clamp de nesting
(`margin`) et la conversion coords parent -> coords fin (parent x2) EXACTEMENT comme
`amr_regrid_finest:62-68`. Construire UN SEUL `BoxArray fb` fin et UN SEUL
`DistributionMapping((int)fb.size(), n_ranks())`. C'est LA REGLE D'OR : un rebuild, pas un par
espece. Si `fb` est vide (rien a raffiner), retourner sans toucher la grille (no-op, comme
`amr_regrid_finest:69`).

(R6) PROLONG / RESTRICT COHERENT DE TOUS LES BLOCS. Pour CHAQUE bloc `b`, reconstruire
`(*blocks_[b].levels)[fk].U` sur le MEME `BoxArray fb` et le MEME `dmap`, avec la largeur de ghost
HERITEE de `(*blocks_[b].levels)[fk].U.n_grow()` (un bloc MUSCL ordre 2 porte 2 ghosts, cf.
`amr_regrid_finest:73`) et la largeur de composantes `U.ncomp()` du bloc. Remplir par :
  (a) INTERP depuis le parent du bloc la ou le nouveau patch n'est pas couvert par l'ancien fin
      (chemin replique : `mf_find_box` ; chemin reparti : `parallel_copy` vers une grille
      enfant-coarsen LOCALE puis `device_fence`, EXACTEMENT `amr_regrid_finest:84-112`) ;
  (b) REPORT des donnees fines existantes la ou l'ancien patch couvre le nouveau (intersection
      `nb.intersect(old.box(ol))`, `amr_regrid_finest:113-120`).
C'est, par bloc, le CORPS de `amr_regrid_finest`, mais sur un layout `fb`/`dmap` IMPOSE de
l'exterieur (le meme pour tous les blocs) au lieu d'etre recalcule par bloc. Tous les blocs
utilisent le MEME `fb`/`dmap` : aucun bloc absent d'un patch.

(R7) REBUILD DE L'AUX PARTAGE + RE-CABLAGE. Reallouer l'aux PARTAGE du niveau fin sur le nouveau
layout : `aux_[fk] = MultiFab(fb, dmap, aux_ncomp_, 1)` (largeur `aux_ncomp_` = max des aux_comps
des blocs, `amr_runtime.hpp:226-228`). Re-cabler le pointeur aux de CHAQUE bloc :
`(*blocks_[b].levels)[fk].aux = &aux_[fk]` (l'adresse `&aux_[fk]` reste stable apres reallocation
en place du `MultiFab` dans le `std::vector` existant). Re-poser B_z par niveau si un bloc le lit
(`fill_bz` cote `AmrSystemCoupler` ; l'`AmrRuntime` ne peuple pas B_z multi-bloc en v1, cf.
`amr_runtime.hpp:222-225`). Puis re-`solve_fields()` pour que phi / grad phi soient coherents avec
la nouvelle grille.

(R8) RESTAURATION DE L'INVARIANT DE COUVERTURE. Comme apres une source ou un transport, restaurer
la coherence des cellules grossieres couvertes par une cascade fin -> grossier
(`mf_average_down_mb`, deja faite dans `solve_fields`, `amr_runtime.hpp:416-419`). Le re-solve de
(R7) la declenche deja ; le noter explicitement evite qu'un diagnostic de masse (somme du seul
grossier) compte une valeur grossiere fantome sous le patch.

INVARIANT CENTRAL. Les etapes (R5) et (R6) garantissent que `(*blocks_[b].levels)[fk].U.box_array()
== fb` et `... .dmap().ranks() == dmap.ranks()` pour TOUT bloc `b`. C'est exactement ce que
`detail::same_layout_or_throw` verifie ; l'appeler en assertion post-regrid (verification (V3))
attrape toute reconstruction incoherente.


## 4. Verifications post-regrid (ce qu'on assert quand le layout change)

Trois invariants doivent etre verifies a CHAQUE regrid (en debug, et par les tests d'acceptation
de la section 7) :

(V1) CONSERVATION PAR BLOC. La masse de chaque bloc (composante 0 integree sur le grossier,
`AmrRuntime::mass(b)`, `amr_runtime.hpp:251`) est conservee a travers le regrid :
`mass(b)` AVANT == `mass(b)` APRES, a la tolerance du report + interp conservatif. Le report des
donnees fines + l'interp depuis le parent redistribuent sans creer ni detruire de masse. A tester
POUR CHAQUE bloc independamment (le reflux et la conservation sont bloc par bloc).

(V2) CONSISTANCE DES GHOSTS. Apres le re-`solve_fields()` de (R7), les ghosts du nouveau niveau fin
(de `U` par bloc et de l'aux partage) sont coherents : injection coarse->fine de l'aux
(`coupler_inject_aux_mb`, `amr_runtime.hpp:437-439`) et remplissage des ghosts de transport par
`advance_amr`. Un patch reconstruit en MUSCL ordre 2 doit porter `n_grow() >= 2` (verification de
la largeur de ghost heritee, (R6)) sinon la reconstruction lirait hors bornes au pas suivant.

(V3) CONSISTANCE DE LAYOUT COLLECTIVE (cross-bloc ET cross-rang). Appeler
`detail::same_layout_or_throw` sur la pile de niveaux reconstruite : tous les blocs partagent
EXACTEMENT le meme `BoxArray` (boites ET ordre), `DistributionMapping` (rang par boite) et `dx`/`dy`
par niveau. Cross-rang : la reduction collective (R4) garantit que chaque rang a calcule le MEME
`fb` ; un test MPI np=1/2/4 doit donner des trajectoires bit-identiques (calque de
`test_mpi_amr_twoblock_parity` cite dans `docs/AMR_MULTIBLOCK_DESIGN.md` section 8).


## 5. Composition avec le chemin courant (cadence, multirate, hierarchie figee)

CADENCE `regrid_every`. La facade porte deja `regrid_every` (`amr_system.hpp:40,63` ; defaut 20).
Le chemin mono-bloc l'utilise via la fermeture `h.step`
(`amr_dsl_block.hpp:101-104`). Le chemin multi-blocs doit l'utiliser de facon analogue : appeler
`runtime->regrid(...)` tous les `regrid_every` macro-pas, AVANT (ou apres) le `step(dt)` du
macro-pas, selon la convention retenue (cf. decision ouverte (D2)). Le compteur de macro-pas
existe deja (`AmrRuntime::macro_step_`, `amr_runtime.hpp:583`).

INTERACTION AVEC LE MULTIRATE (substeps / stride). Le regrid agit a la granularite du MACRO-PAS,
PAS du sous-pas : il se place entre deux macro-pas, jamais au milieu d'une boucle de substeps
(`amr_runtime.hpp:488-489`) ni entre deux rattrapages stride. Un bloc TENU par son stride (hors fin
de fenetre, `(macro_step_+1) % stride != 0`, `amr_runtime.hpp:475`) est NEANMOINS re-grille comme
les autres : il vit sur tous les patchs et contribue au Poisson avec son etat FIGE ; le regrid
deplace sa grille mais ne l'avance pas. Le critere d'union doit donc tager AUSSI les blocs tenus
(leur etat fige reste physiquement present). Interaction a tester explicitement (cf. (D3)).

COMPOSITION AVEC LA HIERARCHIE FIGEE (transition de Phase 1 a Phase 2). Tant que `regrid_every ==
0`, le chemin multi-blocs reste STRICTEMENT celui d'aujourd'hui (hierarchie figee, bit-identique) :
le regrid n'est jamais appele. Le DEVERROUILLAGE consiste a REMPLACER le refus
`python/amr_system.cpp:246-251` par l'activation de la cadence : multi-blocs + `regrid_every > 0`
cesse de throw et cable la fermeture de regrid periodique. Le mono-bloc garde son chemin
(`AmrCouplerMP`, intouche). Critere de non-regression : un cas multi-blocs avec `regrid_every == 0`
doit rester BIT-IDENTIQUE a avant cette PR (le regrid d'union ne s'active que pour
`regrid_every > 0`).


## 6. Carte REUTILISATION vs CODE NOUVEAU

Pour chaque etape de la section 3, ce qui est REUTILISE tel quel vs ce qui est NOUVEAU.

| Etape | Reutilise (existant, verifie) | Nouveau code |
|-------|-------------------------------|--------------|
| (R0) solve + snapshot | `AmrRuntime::solve_fields` (`amr_runtime.hpp:413`), `mass(b)` (`:251`) | snapshot des masses par bloc |
| (R1) tags par bloc | `tag_cells` (`amr/regrid.hpp:36`) | predicat `crit_b` par bloc (en v1 commun) |
| (R2) tags phi / user | `tag_cells` sur `aux_[pk]` | `crit_phi`, predicat utilisateur optionnel |
| (R3) union des tags | `TagBox` (`amr/tag_box.hpp`), `grow_tags` (`amr/regrid.hpp:52`) | helper `tag_union(span<TagBox>)` (OU cellule a cellule) |
| (R4) reduction MPI | `all_reduce_or_inplace` (deja dans `amr_regrid_finest:59-60`) | hisser la garde au niveau de l'UNION (sur `tags_union`, pas par bloc) |
| (R5) clustering unique | `berger_rigoutsos` (`amr/cluster.hpp:171`), `ClusterParams`, clamp nesting (`amr_regrid_coupler.hpp:62-68`) | calculer `fb`/`dmap` UNE fois, partager entre blocs |
| (R6) prolong/restrict par bloc | CORPS de `amr_regrid_finest:73-121` (interp parent replique/reparti, `parallel_copy`, `device_fence`, report fin) | refactor : `amr_regrid_finest` prend un layout IMPOSE (`fb`/`dmap`) au lieu de le recalculer ; boucle sur les blocs |
| (R7) rebuild aux + re-cablage | realloc aux + re-cablage (`amr_regrid_coupler.hpp:123-124` ; pattern aux partage `amr_runtime.hpp:233-238`), `coupler_inject_aux_mb`, `fill_bz` (cote `AmrSystemCoupler`) | orchestration : aux PARTAGE (un seul) re-cable vers TOUS les blocs |
| (R8) cascade couverture | `mf_average_down_mb` (deja dans `solve_fields`) | aucun (declenche par le re-solve) |
| (V1)-(V3) verifs | `mass(b)`, `same_layout_or_throw` (`amr_system_coupler.hpp:122`) | asserts post-regrid + tests (section 7) |

REFACTOR CLE (le seul changement non trivial). `amr_regrid_finest` calcule AUJOURD'HUI son propre
`BoxArray`/`dmap` a partir des tags d'un seul bloc (`amr_regrid_coupler.hpp:52-74`). Pour l'union,
il faut SCINDER cette fonction en deux responsabilites :
  1. CALCUL DU LAYOUT : tags -> `grow` -> `all_reduce_or` -> `berger_rigoutsos` -> clamp ->
     `(fb, dmap)`. C'est ce qu'on fait UNE fois sur l'union des tags (R3)-(R5).
  2. RE-GRILLE D'UN CHAMP SUR UN LAYOUT DONNE : prend `(fb, dmap, ngf, ncomp)` en parametre et
     reconstruit `U` (interp parent + report fin). C'est le corps `amr_regrid_coupler.hpp:73-124`
     SANS le calcul du layout. On l'appelle PAR BLOC (R6).
La fonction mono-bloc actuelle reste l'enchainement des deux (1 puis 2 sur un seul bloc), donc le
chemin mono-bloc `AmrCouplerMP::regrid` reste BIT-IDENTIQUE (il appelle l'enchainement complet).
Cette scission est interne a `amr_regrid_coupler.hpp` : pas de nouveau fichier, et le contrat
mono-bloc est preserve.

NOUVELLE METHODE FACADE-SIDE. `AmrRuntime::regrid(crit, grow, margin, ...)` (et le pendant
`AmrSystemCoupler::regrid`) orchestre (R0)-(R8). C'est le SEUL point d'entree multi-blocs ajoute.
Le deverrouillage facade (`python/amr_system.cpp`) remplace le throw par le cablage de la cadence.


## 7. Tests d'acceptation

Chaque test laisse l'arbre vert ; les invariants (V1)-(V3) sont verifies a chaque regrid.

(T1) UNION DES TAGS. Deux blocs dont les structures fines sont DISJOINTES spatialement (le bloc A
tague une region, le bloc B une autre) : le layout d'union COUVRE les deux regions. Verifier que le
`fb` contient des patchs sur les deux zones (pas seulement celle d'un bloc). Cas degenere : un seul
bloc tague -> meme `fb` que le regrid mono-bloc de ce bloc (parite avec `amr_regrid_finest`).

(T2) CONSERVATION PAR BLOC (V1). Sur N regrids successifs (un cas diocotron multi-blocs ou la
structure se deplace), `mass(b)` reste constant a la tolerance POUR CHAQUE bloc. Un bloc dont le
critere ne se declenche jamais conserve aussi sa masse (il est re-grille comme fond, present
partout).

(T3) CONSISTANCE DE LAYOUT (V3) + GHOSTS (V2). Apres regrid, `same_layout_or_throw` passe (tous les
blocs sur le meme `fb`/`dmap`) ; la largeur de ghost du nouveau fin est >= celle requise par le
schema le plus exigeant (MUSCL ordre 2 -> 2 ghosts). Un pas de transport apres regrid ne lit pas
hors bornes (sanitizer / assertion).

(T4) MPI np=1/2/4 BIT-IDENTIQUES. Calque de `test_mpi_amr_twoblock_parity`
(`docs/AMR_MULTIBLOCK_DESIGN.md` section 8) : grossier reparti round-robin, regrid d'union avec
`all_reduce_or_inplace` sur les tags unis -> `fb` IDENTIQUE par rang, trajectoires np=1/2/4
bit-identiques apres plusieurs regrids. Le spread cross-rang du `BoxArray` fin est nul.

(T5) KOKKOS Serial / OpenMP vert (multi-blocs + regrid actif). Backend-correct : les boucles de
reconstruction (R6) et `coupler_inject_aux_mb` passent par les primitives `_mb` deja portees
device. Un cas GPU (GH200) valide une fois l'instanciation device complete avec regrid
(`parallel_copy` + `device_fence` du chemin reparti, deja dans `amr_regrid_finest:89-92`).

(T6) NON-REGRESSION FIGEE. Multi-blocs avec `regrid_every == 0` reste BIT-IDENTIQUE a la Phase 1
(le regrid n'est jamais appele). Mono-bloc reste BIT-IDENTIQUE (chemin `AmrCouplerMP` intouche,
enchainement complet de `amr_regrid_finest`).

(T7) DEVERROUILLAGE. Multi-blocs + `regrid_every > 0` ne throw PLUS (l'ancien refus
`python/amr_system.cpp:246-251` est leve) et produit une hierarchie qui BOUGE effectivement entre
deux pas (le `BoxArray` fin change quand la structure se deplace).


## 8. Risques et decisions ouvertes (a faire valider par l'owner)

RISQUES.

- (X1) REDUCTION DES TAGS CROSS-RANG. Une union calculee localement puis reduite par
  `all_reduce_or_inplace` doit l'etre sur la grille de tags COMPLETE (taille `pdom.num_cells()`),
  pas par fab. Si le grossier est replique (`replicated_coarse_ == true`), la reduction est
  l'identite et doit etre EVITEE (sinon cout MPI inutile). De-risk : copier la garde exacte de
  `amr_regrid_finest:59-60` (`if (pk == 0 && !coarse_replicated) all_reduce_or_inplace(...)`),
  hissee sur `tags_union`.

- (X2) CONSISTANCE DE `fb` PAR RANG. Si deux rangs partent de grilles de tags differentes,
  `berger_rigoutsos` (deterministe mais data-dependant) produit des `fb` differents -> dmaps
  incompatibles -> MPI desynchronise (deadlock ou corruption silencieuse). De-risk : la reduction
  (R4) AVANT le clustering est la condition NECESSAIRE et SUFFISANTE ; test (T4) np=1/2/4
  bit-identique comme garde permanente.

- (X3) MASSE NON CONSERVEE PAR INTERP NON CONSERVATIF. Le report fin est exact ; l'interp depuis le
  parent doit etre CONSERVATIVE (la moyenne des enfants reconstruits egale le parent). Le chemin
  actuel de `amr_regrid_finest` fait une injection piecewise-constante (chaque enfant = parent),
  qui CONSERVE la masse au sens integral (4 enfants x valeur parente x dV_fin = valeur parente x
  dV_parent). De-risk : test (T2) par bloc ; si un interp d'ordre superieur est introduit plus
  tard, il devra rester conservatif (limiteur de pente borne).

- (X4) GHOST HERITE TROP ETROIT. Si la largeur de ghost du nouveau patch n'herite pas du
  `n_grow()` du niveau remplace, un schema ordre 2 lit hors bornes. De-risk : `ngf =
  (*blocks_[b].levels)[fk].U.n_grow()` PAR BLOC (un bloc Minmod et un bloc VanLeer peuvent differer ;
  l'aux partage prend la largeur max). Verification (V2) + test (T3).

- (X5) DOUBLE COMPTAGE COUVERTURE APRES REGRID. Le re-`solve_fields()` (R7) declenche la cascade
  `mf_average_down_mb` qui restaure les cellules grossieres couvertes. Si on omet ce re-solve, le
  diagnostic de masse (somme du seul grossier) compterait une valeur grossiere fantome sous le
  nouveau patch. De-risk : (R8) explicite ; test (T2) qui somme la masse APRES regrid ET re-solve.

DECISIONS OUVERTES (signature owner requise).

- (D1) CRITERE D'UNION : COMMUN ou PAR BLOC en v1 ? Le design suppose un critere d'UNION (un seul
  layout). Reste a trancher si chaque bloc fournit son propre predicat `crit_b` (puis OU des tags)
  ou si un critere unique est applique a chaque champ. La structure (R1)-(R3) supporte les deux ;
  la difference est l'API d'enregistrement du critere. RECOMMANDATION : predicat par bloc des v1
  (l'union des tags est alors naturelle), critere de phi separe.

- (D2) CADENCE : regrid AVANT ou APRES le `step(dt)` du macro-pas ? Le mono-bloc regrid au DEBUT du
  `h.step` (`amr_dsl_block.hpp:104`, avant l'avance). Pour la coherence, le multi-bloc devrait
  faire de meme (regrid puis step). A confirmer (impacte la phase du compteur `macro_step_`).

- (D3) INTERACTION STRIDE / SUBSTEPS. Un bloc TENU par son stride est-il tague et re-grille au
  macro-pas de regrid meme s'il n'avance pas ? RECOMMANDATION : OUI (son etat fige est present
  partout et contribue au Poisson ; ne pas le re-griller le laisserait sur l'ancien layout et
  casserait `same_layout_or_throw`). A valider : le regrid se place HORS des boucles de substeps et
  des fenetres de stride (granularite macro-pas seulement).

- (D4) TAGS DE PHI : sur phi ou sur grad phi ? Le critere physique du diocotron suit le gradient du
  potentiel (bord d'anneau). RECOMMANDATION : tag sur |grad phi| (composantes 1,2 de l'aux), pas
  sur phi (composante 0). A confirmer selon l'observable cible.

- (D5) MULTI-NIVEAUX : v1 reste a 2 niveaux (grossier + 1 fin), comme `amr_regrid_finest`.
  Confirmer que > 2 niveaux reste HORS scope Phase 2 (limite heritee du mono-bloc, cf. section 2).


## 9. References de code (toutes verifiees a ce head)

- `include/adc/coupling/amr_system_coupler.hpp` : engine multi-blocs AMR compile-time (hierarchie
  FIGEE, PAS de regrid) ; `AmrHierarchyLayout`, `detail::same_layout_or_throw` (garde de layout).
- `include/adc/runtime/amr_runtime.hpp` : moteur multi-blocs RUNTIME (registre type-erase par nom,
  aux partage, Poisson somme, sources couplees, multirate) ; PAS de regrid (cible de ce design).
- `include/adc/coupling/amr_coupler_mp.hpp` : coupleur AMR MONO-BLOC + `regrid` (`:321-325`,
  delegue a `amr_regrid_finest`) ; chemin mono-bloc INTOUCHE.
- `include/adc/coupling/amr_regrid_coupler.hpp` : `amr_regrid_finest` (Berger-Rigoutsos, niveau le
  plus fin) ; brique a SCINDER en "calcul du layout" + "re-grille d'un champ sur un layout donne".
- `include/adc/amr/tag_box.hpp` : `TagBox` (grille dense de tags 0/1 ; union = OU cellule a cellule).
- `include/adc/amr/regrid.hpp` : `tag_cells`, `grow_tags`, `regrid_level` (briques generiques).
- `include/adc/amr/cluster.hpp` : `berger_rigoutsos`, `ClusterParams` (clustering geometrique).
- `include/adc/amr/amr_hierarchy.hpp` : `AmrHierarchy` (conteneur de niveaux ; note : "le futur AMR
  multi-blocs conservatif devra partager une hierarchie commune", l. 31-32).
- `include/adc/runtime/amr_system.hpp` + `python/amr_system.cpp` : facade RUNTIME (`regrid_every` ;
  REFUS multi-blocs + `regrid_every > 0` a `amr_system.cpp:246-251`, leve par ce design).
- `include/adc/runtime/amr_dsl_block.hpp` : cablage du regrid mono-bloc (`:101-104`), layout
  partage 2 niveaux FIGE (`make_shared_amr_layout`) et allocation par bloc (`build_amr_block`).
- `include/adc/parallel/comm.hpp` : `all_reduce_or_inplace`, `n_ranks`, `my_rank` (collectives MPI).
- `docs/AMR_MULTIBLOCK_DESIGN.md` : capstone Phase 1 (engine multi-blocs, layout guard, frontiere
  Phase 2 / Phase 3) ; ce document en est la Phase 2.

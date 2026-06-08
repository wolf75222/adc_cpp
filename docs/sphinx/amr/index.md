# AMR (raffinement adaptatif)

`adc.AmrSystem` est le pendant raffine de `adc.System` : un ou plusieurs blocs (especes)
portes sur une hierarchie AMR block-structured (a boites rectangulaires, type AMReX /
FLASH / SAMRAI). La grille est raffinee la ou la solution le demande, et seulement la. Cette
page resume comment piloter l'AMR depuis Python ; pour les details de conception voir
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) (section 8), [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
(sections 13-15) et les notes de design
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) /
[AMR_REGRID_UNION_TAGS_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_REGRID_UNION_TAGS_DESIGN.md).

L'API est identique a celle de `System` (memes `add_block` / `add_equation` /
`set_poisson` / `set_density` / `step_cfl`) : on raffine un cas existant en changeant
`adc.System(...)` en `adc.AmrSystem(...)` et en ajoutant un critere de raffinement. Le
tutoriel A->Z compare d'ailleurs les deux chemins sur la meme physique (cf.
[tutorials/diocotron_tutorial.py](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py), fonction
`uniform_vs_amr`).

## Hierarchie partagee

Tous les blocs vivent sur une seule hierarchie AMR : memes boites, meme repartition MPI
(`DistributionMapping`), memes pas d'espace par niveau. C'est le modele "une hierarchie
commune portant plusieurs champs", jamais une hierarchie par espece. La version courante
porte deux niveaux (ratio de raffinement 2 : le niveau fin a un pas `dx/2`).

- **Mono-bloc** (un seul `add_block`) : chemin historique `AmrCouplerMP`, avec regrid dynamique
  et reflux conservatif. Bit-identique a ce qu'il a toujours produit.
- **Multi-blocs** (deux `add_block` ou plus) : N blocs co-localises sur la hierarchie
  partagee (moteur `AmrRuntime`). Un seul canal auxiliaire par niveau (`phi`, `grad phi`)
  et un seul Poisson grossier dont le second membre est la somme co-localisee des briques
  elliptiques des blocs (`f = somme_b q_b n_b`, lues aux memes cellules). La conservation est
  assuree par bloc (reflux + average-down). En multi-blocs, le nom du bloc indexe
  `set_density(name)`, `mass(name)` et `density(name)`.

Un garde-fou (`same_layout_or_throw`) verifie a la construction que tous les blocs partagent
exactement le meme layout par niveau (boites, ordre, repartition, `dx`/`dy`) : c'est la
precondition de l'aux unique et du Poisson unique. Detail : [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
section 8, [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) sections 1-2, et le coeur
`include/adc/runtime/amr_system.hpp`.

```python
import numpy as np
import adc

n, L = 96, 1.0
ne0 = np.ones((n, n))                 # densite initiale (n, n), row-major

sim = adc.AmrSystem(n=n, L=L, periodic=True)
model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_refinement(0.05)              # raffine la ou la densite depasse le seuil
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)

for _ in range(60):
    sim.step_cfl(0.4)                 # CFL sur le pas du niveau grossier

print("patchs fins :", sim.n_patches(), "| masse :", sim.mass("ne"))
rho = sim.density("ne")               # densite grossiere (n, n)
```

`adc.AmrSystem(n=, L=, periodic=)` est un raccourci : on peut aussi passer un
`adc.AmrSystemConfig` (champs `n`, `L`, `periodic`, `regrid_every`, `distribute_coarse`,
`coarse_max_grid`) si l'on veut regler la cadence de regrid ou la repartition du grossier.

## Tagging / regrid

Le raffinement est pilote par des criteres de tag evalues sur le niveau parent. La grille
fine est ensuite reconstruite par clustering Berger-Rigoutsos : etant donne les cellules
marquees, l'algorithme trouve un petit nombre de boites rectangulaires qui les couvrent sans
trop de gaspillage (coupe recursive sur la signature des marques). La cadence est portee par
`regrid_every` (re-raffinement tous les N macro-pas ; `0` = jamais apres l'initialisation).

Deux criteres sont exposes et se composent (OU cellule a cellule, "union des tags") :

- `set_refinement(threshold)` : densite par bloc. Raffine la ou la densite (composante 0)
  d'un bloc depasse `threshold`. Critere de base, valable mono- et multi-blocs.
- `set_phi_refinement(grad_threshold)` : gradient du potentiel `|grad phi|`. Raffine la ou
  la norme du gradient du potentiel electrostatique depasse `grad_threshold` (criteres physique
  du diocotron : le bord d'anneau suit le gradient du potentiel, pas la densite seule).
  Multi-blocs uniquement ; desactive par defaut (`grad_threshold <= 0`). A appeler avant
  le premier pas.

En multi-blocs, la hierarchie partagee est re-grillee a partir de l'union des tags de tous
les blocs (plus le tag de `phi` s'il est actif) : un seul critere collectif, un seul clustering,
un seul nouveau layout partage, puis prolongation / restriction / reflux par bloc sur ce
layout. La masse de chaque bloc est conservee a travers le regrid. Avec `regrid_every == 0` la
hierarchie multi-blocs reste figee (regrid jamais appele, bit-identique a une hierarchie
statique). Sous MPI, l'union des tags est reduite cross-rang (`all_reduce_or`) avant le
clustering, sinon les boites fines differeraient d'un rang a l'autre.

Algorithme detaille : [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 15 (clustering + regrid) et
[AMR_REGRID_UNION_TAGS_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_REGRID_UNION_TAGS_DESIGN.md) (etapes R0-R8 de
l'union des tags).

```python
sim = adc.AmrSystem(n=128, L=1.0, periodic=True)
sim.add_block("electrons", model=elec, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.add_block("ions",      model=ions, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_refinement(0.05)         # union des tags de densite (electrons OU ions)
sim.set_phi_refinement(0.5)      # + |grad phi| (multi-blocs ; bord d'anneau)
```

> La cadence de regrid (`regrid_every`) se regle via le `AmrSystemConfig` :
> `sim = adc.AmrSystem(adc.AmrSystemConfig())` puis `config.regrid_every = 20`, ou en passant
> `regrid_every=20` au constructeur (kwargs de config). Avec un seuil de densite a sa valeur par
> defaut (aucun tag), la grille reste inchangee.

## Prolongation / restriction

Les transferts entre niveaux sont les deux operateurs conservatifs classiques :

- **Restriction** (`average_down`, fin -> grossier) : chaque cellule grossiere recoit la
  moyenne des cellules fines qu'elle recouvre. C'est ce qui garde le grossier coherent sous
  un patch fin (et ce qui evite qu'un diagnostic de masse sur le seul grossier compte une valeur
  fantome sous le patch).
- **Prolongation / injection** (`interpolate`, grossier -> fin) : un nouveau patch fin est
  rempli par interpolation depuis le parent la ou il n'existait pas encore, et par report
  des donnees fines existantes la ou l'ancien patch couvre le nouveau. Cette injection est
  conservative (la moyenne des enfants reconstruits egale le parent).

A l'avance en temps, le grossier fait un pas `dt` pendant que le niveau fin fait `r` sous-pas de
`dt/r` (sous-cyclage Berger-Oliger), chacun respectant sa propre CFL. Les ghosts coarse-fine du
niveau fin sont remplis par interpolation espace-temps depuis le grossier. Code coeur :
`mesh/refinement.hpp` (`average_down` / `interpolate`), moteur `numerics/time/amr_reflux_mf.hpp`
(`advance_amr`). Detail : [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 13.

## Reflux

A l'interface entre un niveau fin et le grossier, les deux niveaux calculent des flux de face
differents : la conservation serait cassee si on n'y prenait garde. Le reflux corrige la
maille grossiere par la difference entre le flux fin (integre sur les `r` sous-pas) et le flux
grossier :

$$U_c \mathrel{-}= \frac{1}{\Delta x_c}\Big(\textstyle\sum_s \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)$$

Le reflux est coverage-aware : un masque de couverture (`CoverageMask`) bati sur le BoxArray
global evite la double correction au joint entre deux patchs fins voisins (interface fin-fin, ou
il ne faut pas refluxer) et dirige la correction vers la bonne boite parente quand le grossier
est lui-meme multi-box. En multi-blocs, chaque bloc a ses propres registres de flux : la
conservation est verifiee bloc par bloc. Sous MPI, le registre de flux est rassemble par
`all_reduce_sum` et le reflux distant passe par `parallel_copy`. Roles promus en types :
`FluxRegister` (accumulation des flux de face), `CoverageMask` (cellules couvertes). Detail :
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) sections 13-14, [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)
section 8.

## Multi-blocs

`AmrSystem` est une facade multi-blocs : on appelle `add_block` (briques natives) ou
`add_equation` (modele DSL compile) une fois par espece, exactement comme sur `System`. Tous les
blocs partagent la hierarchie, l'aux et le Poisson grossier (second membre somme co-localise),
mais chacun garde son propre schema spatial (limiter x flux x reconstruction), son traitement
temporel (`explicit` ou `imex`), et son multirate (`substeps` / `stride`). Le moteur runtime est
`AmrRuntime` (registre type-erase par nom de bloc), pendant raffine du moteur multi-blocs
mono-niveau de `System`.

```python
sim = adc.AmrSystem(n=96, L=1.0, periodic=True)

electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0),
                      elliptic=adc.ChargeDensity(charge=-1.0))
ions = adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                 transport=adc.IsothermalFlux(),
                 source=adc.PotentialForce(charge=+1.0),
                 elliptic=adc.ChargeDensity(charge=+1.0))

sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))
sim.add_block("ions", model=ions,
              spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_refinement(0.05)
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne0)
sim.set_density("ions", np.ones((96, 96)))

sim.advance(0.001, 8)
print("blocs :", sim.n_blocks(), "| masse e- :", sim.mass("electrons"))
```

Des sources couplees inter-especes (ionisation, collisions) peuvent etre branchees via
`add_coupled_source` : elles sont lues cellule a cellule (meme `(i, j)`, aucune interpolation
inter-especes, grace a la hierarchie partagee) et, construites a contributions exactement
opposees, conservent la masse de paire a la precision machine. Le multi-blocs est valide par une
batterie de tests dits "capstone" : deux blocs a schemas differents (`test_amr_system_twoblock`),
DSL production multi-bloc (`test_amr_multiblock_compiled`), IMEX (`test_amr_multiblock_imex`),
sources couplees (`test_amr_multiblock_coupled_source`), substeps (`test_amr_multiblock_substeps`),
regrid d'union (`test_amr_multiblock_regrid_union`) et parite MPI np=1/2/4
(`test_mpi_amr_twoblock_parity`). Detail : [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md) section 8 et
le bandeau "STATUT : implemente" en tete de [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Limites actuelles

Ce que l'AMR ne fait pas encore.

- **Deux niveaux seulement.** La hierarchie est grossier + un niveau fin (ratio 2). Le regrid ne
  reconstruit que le niveau le plus fin ; au-dela de 2 niveaux, le regrid multi-niveaux n'existe
  pas encore, meme en mono-bloc.
- **Poisson "coarse + inject".** Le Poisson est resolu sur le grossier puis injecte vers le
  fin, ce n'est pas un solve elliptique composite multi-niveaux. C'est suffisant pour
  l'observable diocotron (qui vit sur un cercle median resolu par le grossier) mais a connaitre.
- **Pas d'etage Schur global sur AMR.** Le splitting source condense par Schur (`adc.Split`,
  `CondensedSchur`) n'a pas de pendant AMR : `AmrSystem.add_block` / `add_equation` le
  rejettent explicitement. Pour cet etage, utiliser un `System` non raffine.
- **Multirate par le chemin compile : restreint.** Sur le chemin DSL "production" (`.so`),
  `add_equation` rejette explicitement `stride > 1` et le masque IMEX partiel
  (`implicit_vars` / `implicit_roles`) : l'ABI plate du loader ne les transporte pas, et ils
  seraient pris a leurs valeurs par defaut en silence. Pour un `.so` multirate ou a masque IMEX
  partiel, passer par `add_block` natif (`adc.Model(...)`), qui les expose.
- **Solveur elliptique.** Sur AMR, le solveur est toujours le multigrille geometrique
  (`geometric_mg`) ; pas de FFT. Le second membre est la somme des briques elliptiques des blocs.
- **Validation : ce qui est teste vs ROMEO seulement.** L'AMR multi-blocs est couvert par les
  tests CPU (Serial / OpenMP) et la parite MPI np=1/2/4 dans ce depot. La validation GPU
  (GH200) des chemins AMR est faite manuellement sur ROMEO (le chemin est device-clean par
  construction, foncteurs nommes) ; voir [BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) pour le
  croisement test / backend ligne par ligne.
- **Cut-cell / polaire hors scope AMR.** Les parois cut-cell et la geometrie polaire sont des
  chantiers du `System` (mono-niveau) ; elles ne sont pas portees sur la hierarchie AMR.

Frontiere de conception (Phase 2 / Phase 3) : criteres de raffinement par bloc, solve
elliptique composite multi-niveaux, et (beaucoup plus loin) hierarchies distinctes par espece
avec projections conservatives. Detail :
[AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md) section 7.

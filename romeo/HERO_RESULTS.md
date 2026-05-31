# Hero runs ROMEO : resultats et journal

Journal des calculs lances sur ROMEO 2025 (URCA) pour la reproduction du taux de croissance
diocotron (papier Hoffart-Maier-Shadid-Tomas, arXiv:2510.11808). Analyse scientifique consolidee
dans [docs/DIOCOTRON_GROWTH_RATE.md](../docs/DIOCOTRON_GROWTH_RATE.md). Sorties brutes sur le
scratch ROMEO (`/scratch_p/$USER/<jobid>/out`, NON sauvegarde, a recopier).

## Environnement et lecons de build

- Partition **x64cpu** (AMD EPYC 9654, OpenMP), compte `r250127`, user `rmdraux`. Le pas
  diocotron (Poisson + flux a petits noyaux) tourne mieux sur CPU multi-coeur que sur GPU
  (cf. docs/HERO_RUN_AMR.md, etape 1 : le GPU est plus lent a ces tailles).
- **Le depot est prive** : `git clone` echoue sur ROMEO (pas d'auth). On synchronise le source
  par `rsync -az --exclude .git --exclude 'build*' ./ romeo:~/adc_cpp/` depuis le Mac.
- **Le cmake de Spack est casse sur les noeuds de CALCUL** : `cmake@3.31.8` y depend de
  `libmd.so.0`, presente sur le LOGIN mais ABSENTE des noeuds de calcul (`cannot open shared
  object file`). Les exemples etant header-only, on **compile en direct** :
  `spack load gcc@14.2.0` puis `g++ -std=c++23 -O3 -fopenmp -I include examples/X.cpp -o bin`
  (le seam `for_each_cell` s'active sur `_OPENMP`). Binaire mis en cache dans un scratch stable,
  pre-constructible sur le login (meme archi x86_64). Determinisme OpenMP == serie (garanti par le depot).

## Job 613961 : convergence haute precision (WENO5-Z + SSPRK3)

`romeo/diocotron_highorder_hero.sbatch`, `OMP_NUM_THREADS=96`. Modes l = 3, 4, 5 x resolution
effective 256, 512, 1024. Reconstruction WENO5-Z (ordre 5), temps SSPRK3, Poisson re-resolu a
chaque etage RK, CFL 0.4. Termine (~1 h ; eff 1024 ~1750 s/mode). Tous stables, aucun `nan`.

Taux de croissance `gamma_norm` mesure dans la **fenetre du papier** (mode-specifique), ajustement
exponentiel propre (R^2 = 1.00), normalisation `omega_D = rho_bar/(2 pi)` :

| mode l | analytique | eff 256 | eff 512 | eff 1024 |
|---|---|---|---|---|
| 3 | 0.772 | 0.838 (+8 %) | 0.850 (+10 %) | 0.853 (+11 %) |
| 4 | 0.911 | 0.985 (+8 %) | 0.988 (+8 %) | 0.987 (+8 %) |
| 5 | 0.683 | 0.730 (+7 %) | 0.731 (+7 %) | 0.729 (+7 %) |

Constat clef : le sur-tir est **uniforme (~+7 a +11 %) et PLAT en resolution** (la resolution ne
referme PAS l'ecart). Voir l'analyse pour le diagnostic (cause geometrique, bord cartesien en escalier).

## Job 613945 : convergence colonne (NoSlope vs VanLeer vs AMR ml)

`romeo/diocotron_recon_hero.sbatch`, `OMP_NUM_THREADS=96`. eff 512 et 1024 termines (eff 2048 a
depasse le mur horaire de 2 h de la partition `short` ; mass conservee `~1e-13` partout). Taux
`gamma_norm` (`sat` = methode pic historique ; `lin` = fenetre lineaire fixe `--window 5,14`) :

| cas | eff | cellules | gamma_norm (lin / sat) |
|---|---|---|---|
| uniforme NoSlope | 512  | 262 144   | 0.650 / 0.583 |
| uniforme VanLeer | 512  | 262 144   | 0.753 / 0.575 |
| AMR `ml` VanLeer | 512  | 104 632   | 0.762 / 0.574 |
| uniforme NoSlope | 1024 | 1 048 576 | 0.706 / 0.578 |
| uniforme VanLeer | 1024 | 1 048 576 | 0.748 / 0.582 |
| AMR `ml` VanLeer | 1024 | 409 008   | 0.747 / 0.579 |

Constats : (1) VanLeer (lin ~0.75) DEPASSE largement NoSlope (lin 0.65 -> 0.71) : l'ordre de
reconstruction est le levier ; (2) l'**AMR `ml` SUIT l'uniforme** (0.762 ~ 0.753 a eff 512,
0.747 ~ 0.748 a eff 1024) pour ~40 % des cellules : la promesse M2b (meme physique, < moitie du
cout) tient a l'echelle ROMEO.

## Jobs intermediaires (debogage)

- 613943 (high-order) : ECHEC, `spack load cmake@3.31.8` ambigu (deux hashes) -> charge par hash.
- 613944 (colonne) : ECHEC, `libmd.so.0` absent des noeuds de calcul -> build direct g++.
- 613948 (high-order, eff 2048) : ANNULE, eff 2048 aurait depasse le mur horaire avant l'extraction
  -> re-cadre a eff 256/512/1024 (job 613961).

## Job 614089 : porte de correction GPU (compute-sanitizer)

`romeo/sanitizer.sbatch` sur 1 GH200 (armgpu). Build Kokkos/CUDA OK : les sommes de masse
routees par le seam reducteur (`for_each_cell_reduce_sum`) compilent sur nvcc. compute-sanitizer
sur les exemples GPU (qui initialisent Kokkos) :

- `coupled_kokkos` : memcheck / initcheck / synccheck = **0 erreur** chacun.
- `diocotron_amr_kokkos` (chemin AMR complet) : memcheck = **0 erreur**.

Detecteur reel de fence manquant = checksum bit-identique CPU vs GPU : `diocotron_amr_kokkos`
rend `checksum=4394594.404318` EXACTEMENT egal au CPU, derive de masse `2.22e-16`, npatch=4.
Les diagnostics routes par le seam restent donc bit-identiques CPU/GPU, sans fence oublie.
Sortie brute recopiee : `romeo/runs/adc_sanitizer.614089.out`.

## Reproduction

Depuis la racine du depot sur ROMEO (apres `rsync` du source) :
```
sbatch romeo/diocotron_highorder_hero.sbatch      # convergence WENO5+SSPRK3, modes 3/4/5
sbatch romeo/diocotron_recon_hero.sbatch          # convergence colonne NoSlope/VanLeer/AMR
```
Extraction du taux (recopier les CSV depuis le scratch, puis) :
```
python3 scripts/validate_diocotron_growth.py out/<cas>/ring_amp.csv --rhobar 0.9 --target 0.911 --window 4.2,5.2
```

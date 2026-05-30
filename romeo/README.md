# Hero run diocotron sur ROMEO (GH200)

Lancer le benchmark diocotron du papier Hoffart (arXiv:2510.11808) **à grande échelle** sur
ROMEO, en utilisant la machine GPU au complet, pour pousser la résolution assez haut que le
taux de croissance numérique atteigne l'analytique (0.911 au mode 4) et reproduise la
Section 5 du papier à l'échelle production.

## Ce que veut dire « toute la machine »

Le seam `for_each_cell` choisit **un seul** backend à la compilation (série / OpenMP /
Kokkos). On ne mélange donc pas OpenMP et Kokkos dans le même noyau. Le « full machine »
réel sur ROMEO `armgpu` est l'hybride **MPI + Kokkos/CUDA** :

- **MPI** distribue à travers les serveurs (Infiniband 800 Gb/s) et les 4 GH200 par serveur
  (NVLink/NVSWITCH). 1 rang MPI par GPU H100.
- **Kokkos/CUDA** sature chaque H100 : tous les noyaux (transport E x B, Poisson) passent par
  `for_each_cell` -> espace d'exécution Cuda. Validé bit-identique au CPU sur ROMEO.
- **OpenMP** est l'autre voie, séparée : partition `x64cpu` (AMD EPYC, 8448 cœurs), MPI +
  OpenMP sans GPU. C'est le mode « CPU pur ».

Donc deux capacités distinctes, chacune utilisant toute sa partie de la machine :

| Mode | Partition | Ce qui tourne | Couvre |
|---|---|---|---|
| `MPI + Kokkos/CUDA` | `armgpu` | 1 rang/GPU, noyaux sur H100 | les 232 GPU H100 |
| `MPI + OpenMP` | `x64cpu` | 1 rang/socket, threads OpenMP | les 8448 cœurs AMD |

Les deux sont déjà supportés (le backend est une propriété de la cible CMake, pas du code
physique). Le hero run vise `armgpu`.

## Build hybride (aarch64 : OBLIGATOIREMENT dans un job armgpu)

Le login est x86_64 : on ne peut pas compiler de l'ARM dessus. Soit on laisse le `.sbatch`
compiler (il le fait), soit on prépare en interactif :

```bash
salloc -t 1:00:00 --account=R00000 --constraint=armgpu --gpus-per-node=1 --mem=32G
romeo_load_armgpu_env
spack load cmake cuda openmpi+cuda
spack load kokkos +cuda +wrapper cuda_arch=90          # Hopper = sm_90
K=$(spack location -i kokkos)
cmake -S . -B build-gh200 -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
cmake --build build-gh200 -j --target diocotron_mpi
```

## Lancer

```bash
# depuis la racine du depot adc_cpp sur ROMEO
sbatch romeo/diocotron_hero.sbatch        # 8 noeuds x 4 GPU = 32 H100, 8192^2, 4000 pas
sbatch romeo/diocotron_scaling.sbatch     # scaling fort + faible (1 -> 32 GPU)
squeue --me                               # suivre
```

Avant de soumettre, éditer dans les `.sbatch` :
- `--account` : le vrai code projet (demander à Romain ; placeholder `R00000`).
- `--nodes` / `RES` / `STEPS` : ambition vs temps de file. `RES=8192` (67 M cellules) est un
  bon point de départ « ça calcule un certain temps » ; monter à 16384 pour saturer plus.
- `--time`, `--mem` (par SERVEUR) : voir la doc ROMEO.

## Ce hero-run est UNIFORME, pas AMR

Important : le binaire `diocotron_mpi` repose sur `SpectralCoupler`, une **grille uniforme**
périodique (Poisson spectral FFT, décomposition en bandes). **Aucun AMR, aucun regrid.** C'est
la version FORCE BRUTE : on pousse la résolution uniforme assez haut pour résoudre le bord
d'anneau. Elle utilise toute la machine (MPI + Kokkos/CUDA) mais pas l'adaptation de maillage.

Un hero-run **AMR dynamique** est un objectif distinct (la convergence de plusieurs briques) :
le reflux multi-patch distribué (fait, bit-identique np=1/2/4) + le coupleur AMR `AmrCouplerMP`
porté multi-GPU + le benchmark colonne (paroi conductrice, cf. M2 dans `docs/ROADMAP.md`). Tant
que ce n'est pas assemblé, le hero-run reste uniforme.

## Le payoff scientifique

Sur grille uniforme à `n <= 256` (Mac, M1), le taux mesuré plafonne à ~60 % de l'analytique
(`docs/fig_diocotron_reproduction.png`) : la diffusion numérique lisse le bord d'anneau fin.
À `n = 8192+` sur les H100, le bord reste net sur des dizaines de cellules -> le taux doit
converger vers `0.911` et reproduire fig 5.1-5.3 du papier à pleine résolution, par FORCE BRUTE.
C'est la démonstration « notre solveur reproduit le papier, à l'échelle ». Un run AMR atteindrait
le même taux pour bien moins de cellules : c'est l'intérêt, et le hero-run uniforme sert de
référence chiffrée (cellules, temps) pour mesurer ce que l'AMR fait gagner.

## Pièges ROMEO (rappel)

- 4 obligatoires : `--account`, `--time`, `--mem`, `--constraint`. `--mem` = par serveur.
- Cohérence `--constraint=armgpu` ↔ `romeo_load_armgpu_env` ↔ specs Spack ARM.
- Travailler dans `/scratch_p/$USER/$SLURM_JOBID` (le home : quota 20 Go). `/scratch_p` n'est
  PAS sauvegardé : recopier les sorties ailleurs.
- `srun` (pas `mpirun`) en batch : il lit la conf du job.
- MPI CUDA-aware (`openmpi +cuda`) pour le GPU-direct sur Infiniband.

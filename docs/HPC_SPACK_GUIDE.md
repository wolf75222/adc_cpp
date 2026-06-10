# Lancer adc_cpp sur un supercalculateur (Spack)

Guide generique pour compiler et executer adc_cpp (lib C++23 header-only +
module Python `_adc`) sur n'importe quel cluster via Spack, avec les
performances maximales. Backends optionnels : Kokkos (CPU OpenMP / GPU CUDA),
MPI (OpenMPI), HDF5 parallele.

Le pendant specifique a ROMEO est `Tools/machines/romeo/romeo_adc.profile.example` :
ce guide en est la version generique pour les autres clusters, et peut servir
de modele a recopier en profil de site.

## 0. Pre-requis

- CMake >= 3.21 et Ninja (peuvent venir de Spack, cf. section 2).
- Un compilateur C++23 (gcc >= 13 conseille, cf. section 3).
- Acces a un scratch rapide pour le cache des `.so` du DSL (section 5).

## 1. Choisir : Spack du site ou bootstrap perso

Beaucoup de centres fournissent deja Spack. Verifier d'abord :

```sh
module avail spack          # ou : which spack
module load spack           # si un module existe
spack --version
```

Si rien n'est disponible, bootstrap perso sans droits root :

```sh
git clone --depth=2 https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh     # a mettre dans ~/.bashrc
spack bootstrap now                     # construit les dependances internes
```

Preferer le Spack du site quand il existe : il connait deja les fabrics
reseau, les pilotes GPU et les compilateurs vendeur. Le bootstrap perso est le
recours quand le site n'en fournit pas ou que la version est trop ancienne.

## 2. Installer la pile

Choisir le compilateur d'abord (section 3), puis :

```sh
# CPU (OpenMP)
spack install kokkos +openmp

# GPU (CUDA) : nvcc_wrapper de Kokkos comme compilateur device
#   cuda_arch=80 -> A100, 90 -> H100 / GH200
spack install kokkos +cuda +wrapper cuda_arch=90

# MPI : prendre les fabrics du site si possible (cf. section 8)
spack install openmpi                  # fabrics=auto par defaut
# exemple oriente reseau du site :
# spack install openmpi fabrics=ucx +cuda    # a verifier sur votre site

# Outils de build
spack install cmake ninja

# HDF5 parallele (optionnel, sorties paralleles)
spack install hdf5 +mpi
```

Retrouver les prefixes installes (utile pour CMake / les env vars) :

```sh
spack location -i kokkos
spack location -i openmpi
spack location -i hdf5
```

`spack location -i <spec>` imprime le prefixe d'installation d'un paquet deja
installe ; on s'en sert pour `-DKokkos_ROOT`, `ADC_KOKKOS_ROOT`, etc.

## 3. Viser la microarchitecture (perf)

Lister ce que Spack reconnait :

```sh
spack arch                       # arch courante detectee
spack arch --known-targets       # toutes les cibles connues
```

Cibler la micro-archi des noeuds de calcul (pas du login s'ils different) :

```sh
spack install kokkos +openmp target=zen4            # AMD Genoa
spack install kokkos +openmp target=sapphirerapids  # Intel SPR
spack install kokkos +openmp target=neoverse_v2     # ARM Grace (GH200)
```

Compilateur recent (gcc >= 13 pour C++23 complet) :

```sh
spack compiler find                 # enregistre les compilos deja presents
spack install gcc@13                # si le site n'a pas de gcc recent
spack load gcc@13
spack compiler find                 # re-detecte gcc@13 fraichement installe
spack compiler list
```

Puis epingler le compilateur sur chaque spec avec `%` :

```sh
spack install kokkos +openmp target=zen4 %gcc@13
```

## 4. Compiler adc_cpp

Configurer sur le login si CMake/Ninja manquent sur les noeuds de calcul ;
les binaires resteront lances par sbatch. Les options ADC_* sont aussi
acceptees comme variables d'environnement.

Coeur + tests, CPU :

```sh
export KOKKOS_ROOT=$(spack location -i kokkos)
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_ROOT \
  -DADC_USE_MPI=ON \
  -DADC_USE_HDF5=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu
```

GPU : compiler avec le `nvcc_wrapper` de Kokkos comme compilateur C++ :

```sh
export KOKKOS_ROOT=$(spack location -i kokkos)   # build +cuda +wrapper
cmake -S . -B build-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$KOKKOS_ROOT/bin/nvcc_wrapper \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT=$KOKKOS_ROOT \
  -DADC_USE_MPI=ON
cmake --build build-gpu -j
```

Module Python `_adc` (scikit-build-core via pip) :

```sh
export ADC_USE_KOKKOS=ON
export Kokkos_ROOT=$(spack location -i kokkos)
export ADC_USE_MPI=ON
# GPU : ajouter
# export CMAKE_CXX_COMPILER=$Kokkos_ROOT/bin/nvcc_wrapper
pip install .                       # dans un venv ou conda env
```

Le compilateur du build est bake dans `_adc` (le DSL le reappelle au runtime,
cf. section 5) : il doit rester accessible sur les noeuds. Garder le `spack
load` du compilateur (ou son module) dans le script sbatch.

## 5. Variables runtime du DSL

Le module compile des `.so` au runtime. A exporter sur les noeuds :

```sh
# parite loader / module : meme Kokkos que celui du build
export ADC_KOKKOS_ROOT=$(spack location -i kokkos)

# cache des .so sur le scratch (rapide, partage entre jobs si meme micro-archi)
export ADC_CACHE_DIR=$SCRATCH/adc_dsl_cache

# flags d'optimisation des .so DSL (defaut "-O3 -DNDEBUG")
export ADC_DSL_OPTFLAGS="-O3 -DNDEBUG"
```

`-march=native` est possible dans `ADC_DSL_OPTFLAGS` pour gagner quelques %,
mais le cache `.so` n'est alors PAS partageable entre micro-architectures :
utiliser un `ADC_CACHE_DIR` distinct par type de noeud, ou s'en passer si les
noeuds sont heterogenes.

## 6. Lancer avec perf max (sbatch)

### CPU (OpenMP, 1 rang MPI par socket NUMA)

```sh
#!/bin/sh
#SBATCH --job-name=adc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # 1 rang par socket NUMA
#SBATCH --cpus-per-task=48           # coeurs par socket
#SBATCH --time=01:00:00

. ~/spack/share/spack/setup-env.sh   # ou module load spack
spack load gcc@13 kokkos openmpi     # meme stack que le build

export ADC_KOKKOS_ROOT=$(spack location -i kokkos)
export ADC_CACHE_DIR=$SCRATCH/adc_dsl_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun --cpu-bind=cores python run_case.py
```

En Python, `adc.set_threads(n)` pose `OMP_NUM_THREADS` et `KOKKOS_NUM_THREADS`
de maniere coherente ; le placement NUMA (`OMP_PROC_BIND` / `OMP_PLACES`)
reste a fixer dans l'environnement comme ci-dessus.

### GPU (CUDA, 1 rang MPI par GPU)

```sh
#!/bin/sh
#SBATCH --job-name=adc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 rang par GPU
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

. ~/spack/share/spack/setup-env.sh
spack load gcc@13 kokkos openmpi     # kokkos +cuda +wrapper

export ADC_KOKKOS_ROOT=$(spack location -i kokkos)
export ADC_CACHE_DIR=$SCRATCH/adc_dsl_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Piege OpenMPI + memoire UVM/managed : le transport CUDA-IPC (btl smcuda)
# deadlock entre GPU isoles par cgroup (--gpus-per-task=1). Quick-fix :
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0

srun --cpu-bind=cores python run_case.py
```

1 rang MPI par GPU est la regle ; ne pas sur-souscrire. Si le job hang dans
les halos multi-boites a np >= 2 sur GPU, le quick-fix ci-dessus contourne le
routage CUDA-IPC impossible entre GPU cgroup-isoles (le fix propre est dans
le code depuis la PR #254 ; le quick-fix reste utile sur un module ancien).

## 7. Reproductibilite : environnement Spack par machine

Figer la pile dans un environnement Spack versionne (un dossier par cluster) :

```sh
spack env create adc-clusterX        # cree l'env
spack env activate adc-clusterX
spack add kokkos +cuda +wrapper cuda_arch=90 target=neoverse_v2 %gcc@13
spack add openmpi cmake ninja hdf5 +mpi
spack concretize                     # ecrit spack.lock
spack install
```

Ou ecrire directement un `spack.yaml` puis `spack env create adc-clusterX spack.yaml` :

```yaml
spack:
  specs:
    - kokkos +cuda +wrapper cuda_arch=90 target=neoverse_v2
    - openmpi
    - cmake
    - ninja
    - hdf5 +mpi
  packages:
    all:
      compiler: [gcc@13]
      providers:
        mpi: [openmpi]
  view: true            # vue filesystem, ou un chemin : view: /chemin/vue
```

Committer `spack.yaml` ET `spack.lock` par machine (ex. sous
`Tools/machines/<cluster>/`) : `spack.lock` est l'etat concretise exact,
rejouable a l'identique.

## 8. Adapter a votre cluster

Inspecter le site avant de fixer le placement :

```sh
sinfo -o "%P %N %c %G %m"      # partitions, coeurs, GPU, RAM par noeud
module avail                    # modules du site (compilateurs, MPI, CUDA)
spack arch --known-targets      # cibles micro-archi reconnues
```

- Verifier `--cpus-per-task` contre le nombre de coeurs par socket de `sinfo`,
  et `--gpus-per-task` contre la colonne GRES (`%G`).
- Preferer le MPI / CUDA du site (via `module load` + Spack `externals`)
  quand le centre les optimise pour la fabric reseau : c'est souvent plus
  rapide et plus stable que de tout reconstruire. Declarer un paquet externe
  dans `packages:` du `spack.yaml` si besoin.   # a verifier sur votre site
- En cas de doute sur une commande Spack specifique au site, consulter la doc
  du centre : la syntaxe des variants et des cibles peut etre contrainte par
  les paquets externes deja declares.

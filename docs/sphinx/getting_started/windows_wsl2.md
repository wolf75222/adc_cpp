# Windows (WSL2)

Ce guide part de zero sur un PC **Windows 11** et conduit jusqu'a un cas `adc_cases` qui
tourne, sans connaissance prealable du projet. Le chemin recommande est **WSL2 Ubuntu** :
tout l'outillage Linux deja documente ([Installation](installation.md)) s'applique tel quel
une fois dans WSL2. Le Windows natif n'est pas un objectif (le coeur a besoin d'une toolchain
Linux ; cf. epic Port Windows).

Les ecarts par rapport aux chemins Linux/macOS/ROMEO sont signales par **(ecart Windows)**.

## 1. Prerequis

### WSL2 + Ubuntu

```powershell
wsl --install            # depuis PowerShell admin ; installe Ubuntu, redemarre
wsl -l -v                # verifier : Ubuntu, VERSION 2
```

Tout le reste se fait **dans le terminal Ubuntu** (WSL2), pas dans PowerShell.

### GPU NVIDIA (optionnel, pour le CUDA)

Aucun driver a installer cote WSL : le driver **Windows** NVIDIA expose le GPU a WSL2. Verifier
depuis Ubuntu :

```bash
nvidia-smi               # doit lister le GPU (ex. RTX 3090, driver 591.86, CUDA 13.x)
```

Si `nvidia-smi` repond, le passthrough GPU fonctionne. Le toolkit CUDA s'installe **cote WSL**
via conda (section GPU), pas besoin de l'installer globalement.

### conda (Miniforge)

```bash
curl -fsSL -o /tmp/mf.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash && exec bash
```

### Cloner les depots dans le FS WSL

**(ecart Windows)** Cloner dans le systeme de fichiers **WSL** (`~/dev/...`), **pas** dans
`/mnt/c/...` : les I/O sur le disque Windows monte sont beaucoup plus lentes et ralentissent
fortement les builds C++.

```bash
mkdir -p ~/dev/Stage_Romain && cd ~/dev/Stage_Romain
git clone <url>/adc_cpp.git
git clone <url>/adc_cases.git
```

## 2. Environnement de build

Identique au Linux documente : depuis `adc_cpp/`,

```bash
bash scripts/setup_env.sh     # cree l'env `adc` + gcc 14 conda (CC/CXX figes)
conda activate adc
```

### (ecart Windows) Kokkos conda = variante CUDA par defaut

conda-forge sert desormais `kokkos` en **build CUDA** par defaut. Son `KokkosConfig.cmake`
exige `CUDAToolkit` : sans toolkit CUDA installe, **tous** les presets echouent au configure
avec `Could NOT find CUDAToolkit (missing CUDA_CUDART)` (meme `serial`, car `find_package(Kokkos)`
prend le kokkos conda via `CMAKE_PREFIX_PATH`).

Pour le **dev CPU**, fixer la variante CPU de Kokkos (elle embarque OpenMP + Serial) :

```bash
conda install -n adc -c conda-forge "kokkos=*=*hbbfbac7*"   # build sans 'cuda' dans le hash
# (verifier : `conda list kokkos` ne doit PAS afficher un build 'cuda12...')
```

Le GPU se traite dans un env dedie (section 5), pour ne pas melanger CPU et CUDA.

## 3. Coeur C++ et tests

```bash
conda activate adc
cmake --preset serial
cmake --build --preset serial -j 6      # (ecart Windows) borner -j !
ctest --preset serial                   # attendu : 100% tests passed
```

**(ecart Windows)** WSL2 plafonne la RAM (souvent ~50% de l'hote, ex. 15 Go). Les grosses
unites `system.cpp` / `amr_system.cpp` a -O3 peuvent faire **OOM `cc1plus`** a parallelisme
plein. Borner a `-j 6` (~2,5 Go/job) evite l'OOM. ccache (fourni par l'env) accelere les
rebuilds ulterieurs.

CPU multi-thread (Kokkos OpenMP) :

```bash
cmake --preset parallel
cmake --build --preset parallel -j 6
OMP_NUM_THREADS=4 ctest --preset parallel
```

MPI (CPU, optionnel) :

```bash
cmake --preset mpi && cmake --build --preset mpi -j 6
OMPI_MCA_btl_smcuda_use_cuda_ipc=0 ctest --preset mpi
```

## 4. Module Python et cas `adc_cases`

```bash
conda activate adc
cd ~/dev/Stage_Romain/adc_cpp
cmake --build --preset python-parallel -j 6        # module _adc (Kokkos)
export PYTHONPATH="$PWD/build-py-kokkos/python:$PWD/python"
python -c "import adc; adc.doctor()"               # doit etre tout vert
```

```bash
cd ~/dev/Stage_Romain/adc_cases
pip install -e '.[figures]'
python check_cases.py                              # garde-fous docs/manifeste
export ADC_KOKKOS_ROOT="$CONDA_PREFIX"             # voir encadre ci-dessous
python tutorial/equivalence.py
python tutorial/run.py
```

### (ecart Windows) DSL + module Kokkos : `ADC_KOKKOS_ROOT`

Quand `_adc` est compile **avec Kokkos**, le backend DSL `production` (qui compile un `.so` du
modele a l'execution) doit l'etre aussi, sinon le garde ABI le rejette (`kokkos=0` != `1`) et
aucun backend DSL ne compile. Exporter `ADC_KOKKOS_ROOT=$CONDA_PREFIX` avant de lancer un cas
qui utilise le DSL. Avec un module **serie** (preset `python`, sans Kokkos), ce n'est pas requis.

## 5. GPU CUDA (WSL2)

Le GPU est valide dans un **env conda dedie** `adc-gpu` (kokkos CUDA + toolkit CUDA), separe de
l'env `adc` (CPU) pour ne pas melanger les deux Kokkos.

```bash
mamba create -y -n adc-gpu -c conda-forge \
  "kokkos=*=*cuda12*" cuda-toolkit "cuda-version=12" cmake ninja cxx-compiler
conda activate adc-gpu
```

**(ecart Windows)** Le `kokkos` CUDA de conda-forge est compile pour `sm_80` ; il tourne sur
une RTX 3090 (`sm_86`) en compatibilite (Kokkos emet un avertissement de perf, le calcul est
correct). Pour un binaire natif `sm_86`, recompiler Kokkos CUDA a la main.

**(gotcha)** Les scripts d'activation conda de `cuda-nvcc` referencent `NVCC_PREPEND_FLAGS`
sans valeur ; sous `set -u`/`nounset` ils tuent le shell. Exporter avant toute activation :
`export NVCC_PREPEND_FLAGS="" NVCC_APPEND_FLAGS=""`.

Le build adc GPU utilise `nvcc_wrapper` comme compilateur C++ :

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/nvcc_wrapper" \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$CONDA_PREFIX" -DADC_BUILD_TESTS=ON
```

### Statut GPU verifie (WSL2, RTX 3090)

- **Harness Kokkos CUDA minimal : OK.** Un `parallel_reduce` Kokkos s'execute sur le GPU
  (`DefaultExecutionSpace = Cuda`, resultat correct).
- **Cas adc GPU dedie : OK.** Un petit programme utilisant le **seam de calcul d'adc**
  (`Fab2D` / `Array4` / `for_each_cell` / `for_each_cell_reduce_sum`) avec une lambda
  **top-level** compile sous `nvcc_wrapper` et tourne sur le GPU ; valide contre un **oracle CPU**
  (chaque cellule == formule analytique, reduction GPU == somme hote). Le coeur d'adc est donc
  device-clean sur le GPU local.

```bash
# exemple : compiler un programme adc GPU dedie
cmake -S <prog> -B build -G Ninja -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/nvcc_wrapper" \
  -DADC_INC=<adc_cpp>/include   # + target_compile_definitions(... ADC_HAS_KOKKOS)
```

- **(limite)** Compiler la **suite `ctest` C++ complete sous nvcc reste KO** : certains fichiers
  de test emploient des lambdas device **imbriquees dans d'autres lambdas**, que nvcc refuse
  (`__wrapper__device_stub_..._ParallelFor<...> does not match any template declaration`). Ce n'est
  PAS le coeur adc (le cas dedie ci-dessus le prouve) ni le Kokkos conda (un harness
  `MDRangePolicy<Rank<2>>` minimal passe) ; c'est le pattern cote test. La suite `ctest` sous nvcc
  n'est verte sur aucune plateforme : **valider le GPU par programmes dedies** (ou le module
  `_adc` / cas Python), comme sur ROMEO.
- **(perf)** Le `kokkos` CUDA conda est compile pour `sm_80` ; il tourne sur une RTX 3090
  (`sm_86`) en compatibilite (avertissement Kokkos). Pour un binaire natif `sm_86`, recompiler
  Kokkos CUDA avec `Kokkos_ARCH_AMPERE86=ON`.

## 6. Recapitulatif des ecarts Windows

| Sujet | Linux/Mac/ROMEO | Windows (WSL2) |
|---|---|---|
| OS de build | natif | WSL2 Ubuntu (pas de natif Windows) |
| Emplacement repo | n'importe ou | FS WSL (`~/dev`), **pas** `/mnt/c` |
| Kokkos conda | suppose Serial-only | **CUDA par defaut** -> pinner variante CPU `hbbfbac7` |
| Parallelisme build | machine-dependant | borner `-j 6` (RAM WSL ~15 Go, OOM sinon) |
| GPU | ROMEO (Spack, GH200) | WSL2 local (env `adc-gpu`, kokkos `sm_80` sur `sm_86`) |
| DSL + module Kokkos | idem | exporter `ADC_KOKKOS_ROOT=$CONDA_PREFIX` |
| MPI | idem | `OMPI_MCA_btl_smcuda_use_cuda_ipc=0` |

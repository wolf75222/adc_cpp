# Verifier son backend

Le meme code source de `adc_cpp` tourne sur le seul backend on-node Kokkos, dont la cible se
choisit a l'installation de Kokkos : Kokkos Serial (sequentiel), Kokkos OpenMP (CPU multi-thread),
Kokkos CUDA/HIP (GPU). MPI s'ajoute en option pour le distribue. Kokkos est obligatoire et fixe a la
compilation. Cette page explique comment savoir quel espace d'execution tourne, car la reponse n'est
pas toujours celle qu'on croit.

## La regle d'or : le backend est fixe a la compilation

Le backend n'est pas un argument runtime. C'est une propriete de la cible `adc`, attachee a
l'interface CMake : tout ce qui lie `adc` (la facade compilee, les tests) en herite, et le seam
`for_each_cell` bascule sur l'espace d'execution choisi. Pour changer de backend, on
reconfigure et on recompile (voir [Installation](installation.md)). Il n'y a donc rien a
activer dans un script Python.

## Le module Python est serie

Souvent surprenant : le module `_adc` distribue/teste en CI n'est construit
qu'en Kokkos Serial (pas de MPI dans le job Python). Tout
script Python qui fait `import adc` tourne donc en sequentiel, quel que soit le materiel. C'est
documente dans la matrice [`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md) (section 2 :
"le binding Python ne route que vers Kokkos Serial").

Le tutoriel l'affiche d'ailleurs au demarrage : il sonde quelques attributs optionnels du module
(`backend`, `parallel_backend`, `build_info`) et, faute d'un tel attribut, retombe explicitement
sur la mention "serial (defaut)". Le module ne ment pas sur son parallelisme ; il n'expose
pas (encore) de drapeau de backend cote Python. Le seul diagnostic d'ABI exporte est
`adc.abi_key()` (compilateur + standard C++ + signature des en-tetes), qui sert au DSL natif,
pas a identifier le backend de dispatch.

Pour du parallelisme multi-thread ou GPU, c'est la facade C++ qu'il faut compiler contre un Kokkos
installe pour l'espace d'execution voulu (OpenMP, CUDA), et y porter le calcul cote C++ (tests,
executables, ou un cas `adc_cases` compile en natif).

## Python pilote, le C++ compile calcule

Le partage des roles ne depend pas du materiel. Python construit le `System`, deroule la boucle
en temps et ecrit les sorties ; le travail par cellule passe par le seam `for_each_cell`, compile
pour l'espace d'execution Kokkos choisi a l'install (Serial, OpenMP, ou CUDA). Le mot "sequentiel"
decrit le module `_adc` tel que la CI le construit, en Kokkos Serial ; ce n'est pas une limite du
Python. Le module lie le meme `adc::adc` que les tests (cf. `python/CMakeLists.txt`) et heriterait
d'un Kokkos OpenMP ou CUDA s'il etait construit contre.

Sur GPU, et donc sur ROMEO, le calcul tourne aujourd'hui via des harnais C++ generes, pas via le
module Python. Le flux est dans [`GPU_ROMEO.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md) :
un script hote (`python/tests/gpu/gen_kokkos_harness.py`, puis `gen_kokkos_sim.py` pour une
simulation complete) emet un `.cpp` ou `.cu` a partir du modele ; on l'envoie sur ROMEO ; `srun` le
compile avec `nvcc_wrapper` et l'execute sur le noeud GH200. Python reste l'auteur et
l'orchestrateur cote hote, le binaire compile fait le calcul sur le device. Le module `_adc`
n'est pas construit en CUDA sur les noeuds de calcul.

## Couverture des backends

| Backend | Comment l'obtenir | Ou il est valide |
|---|---|---|
| Kokkos Serial | build par defaut (`Kokkos_ENABLE_SERIAL=ON`) ; module Python | CI (gate `build-and-test`, C++ + Python) |
| MPI CPU | `-DADC_USE_MPI=ON`, lance via `mpirun -np {1,2,4}` | CI (ci-full) |
| Kokkos OpenMP | Kokkos installe avec `Kokkos_ENABLE_OPENMP=ON` | CI (ci-full, 91/91 ctest) |
| Kokkos CUDA (GH200) | `-DADC_USE_KOKKOS=ON` + `Kokkos_ARCH_HOPPER90` + `nvcc_wrapper` | ROMEO, manuel (jamais en CI) |
| MPI + Kokkos CUDA | meme build + OpenMPI CUDA-aware, `srun --gpus-per-task=1` | ROMEO, manuel |

La CI ne construit jamais avec CUDA active (les runners n'ont pas de GPU) : toutes les
cellules "Kokkos CUDA" de la matrice sont soit ROMEO, soit non exercees. Le decompte a jour et
le statut de chaque test par backend vivent dans [`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md)
(ne pas le dupliquer ici).

## Sur GPU : ROMEO, manuel

Le modele d'execution GPU est MPI + Kokkos, pas du CUDA ecrit a la main : MPI distribue les
sous-domaines (un GPU par rang), Kokkos parallelise le calcul local et abstrait le materiel via
son `ExecutionSpace` (backend `Cuda` sur NVIDIA). Le meme `.cpp` passe en `exec=Serial` /
`OpenMP` sur CPU et en `exec=Cuda` sur GPU selon le backend choisi a la compilation ; `nvcc_wrapper`
n'est que le compilateur exige par le backend Cuda de Kokkos. Detail :
[`GPU_RUNTIME_PORT.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).

La validation GH200 est faite a la main sur ROMEO (noeud `armgpu`, Grace-Hopper, `cuda/12.6`,
Kokkos 4.4.01, `Kokkos_ARCH_HOPPER90`, OpenMPI CUDA-aware), via des harnais SLURM dedies. Etat
(juin 2026) : le solveur mono-grille complet, les ops de champ AMR, l'echange de halos multi-GPU
(np=1/2/4 bit-identiques) et la validation integree AmrSystem + MPI + GPU sont valides. La
recette `srun` type et les preuves (`maxdiff=0` contre le CPU) sont dans
[`GPU_ROMEO.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md) et [`GPU_RUNTIME_PORT.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_RUNTIME_PORT.md).

## Ce qu'il faut retenir

- Le module `_adc` tel que la CI le construit = Kokkos Serial ; Python pilote, le C++ compile calcule.
- Pour CPU multi-thread / GPU : recompiler la facade C++ contre un Kokkos installe pour l'espace voulu (OpenMP, CUDA) ; pour le distribue, ajouter `-DADC_USE_MPI=ON`.
- Le GPU est ROMEO-manuel, via des harnais C++ generes ; la CI est CPU uniquement.
- La source de verite du "quel test, quel backend" est [`BACKEND_COVERAGE.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).

# Tutoriels exécutables

Scripts qui tournent vraiment et produisent une figure ou un GIF. Les tutoriels en
markdown ([../README.md](../README.md)) expliquent la physique ; ceux-ci la font tourner.
Chaque script Python pilote la façade `adc` (bindings), valide un invariant par `assert`,
et écrit sa sortie sous `docs/`.

## Prérequis

```bash
cmake -S . -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py -j
export PYTHONPATH=$PWD/build-py/python
pip install numpy matplotlib
```

## Python (bindings)

| Script | Ce qu'il montre | Sortie | Vérifié |
|---|---|---|---|
| [`diocotron.py`](diocotron.py) | instabilité diocotron (dérive E x B), animation | `docs/tut_diocotron_py.gif` | masse conservée 3e-11 |
| [`diocotron_growth.py`](diocotron_growth.py) | taux de croissance de l'instabilité (phase linéaire) | `docs/tut_diocotron_growth.png` | gamma > 0 ajusté |
| [`two_fluid_ap.py`](two_fluid_ap.py) | schéma AP : stabilisé borné vs non-stabilisé qui explose | `docs/tut_tfap_ap.png` | AP borné, l'autre explose |
| [`euler_poisson.py`](euler_poisson.py) | Euler auto-gravitant, conservation | `docs/tut_euler_poisson.png` | masse + qté de mvt à l'arrondi |

```bash
PYTHONPATH=build-py/python python3 tutorials/run/diocotron.py 128 400
PYTHONPATH=build-py/python python3 tutorials/run/two_fluid_ap.py
PYTHONPATH=build-py/python python3 tutorials/run/euler_poisson.py
PYTHONPATH=build-py/python python3 tutorials/run/diocotron_growth.py
```

Chaque script se termine par un `assert` sur l'invariant physique : s'il s'exécute sans
erreur, le solveur a bien conservé / borné / cru comme attendu.

## C++ (exemples + GIFs)

L'AMR (multi-niveaux, multi-patch) n'est pas exposé en Python (la façade ne porte que les
solveurs mono-grille) : il se pilote par les exemples C++, et les GIFs sont produits par
les scripts `scripts/make_*.py`.

```bash
cmake -S . -B build && cmake --build build -j
./build/bin/diocotron_amr3      out 128 500
python3 scripts/make_diocotron_amr3_gif.py out docs/anim_diocotron_amr3.gif
./build/bin/diocotron_multipatch out 128 480
python3 scripts/make_diocotron_multipatch_gif.py out docs/anim_diocotron_multipatch.gif
```

Banc de mesure (scaling OpenMP, figure du rapport) :

```bash
OMP_NUM_THREADS=4 ./build-omp/bin/bench_amr 512 100
python3 scripts/plot_bench_scaling.py /tmp/scaling.csv docs/fig_openmp_scaling.png
```

## MPI distribué sur ROMEO

Le seam MPI (`comm.hpp`) donne un résultat **bit-identique** quel que soit le nombre de
rangs. Sur ROMEO 2025 (cluster du centre de calcul), un job CPU lance les tests MPI via
`srun` :

```bash
sbatch scripts/romeo_mpi.sbatch       # partition x64cpu, 8 rangs sur 2 noeuds
```

Le script charge la toolchain Spack (gcc 14 + openmpi 5 + cmake), compile dans le scratch
du job avec `-DADC_USE_MPI=ON`, et exécute les tests MPI (dont `test_mpi_array_reduce`,
la brique `all_reduce` du reflux multi-patch distribué, et `test_mpi_diocotron`, le pas
couplé E x B distribué). `srun` lit la configuration du job (8 rangs) pour placer les
processus ; pas de `mpirun -np` manuel.

En local, le même test tourne sous `mpirun` :

```bash
cmake -S . -B build-mpi -DADC_USE_MPI=ON && cmake --build build-mpi -j
ctest --test-dir build-mpi            # 55/55, bit-identique à np=1
```

## GPU GH200 sur ROMEO

La façade et le pas deux-fluides AP compilent pour le GPU via Kokkos (backend Cuda),
bit-identiques au CPU :

```bash
sbatch scripts/romeo_tfap.sbatch          # partition armgpu (Nvidia GH200), Kokkos + CUDA
sbatch scripts/romeo_libadc_gpu.sbatch    # compile la facade libadc pour le GPU
```

`aarch64 + CUDA` ne se compile pas sur le nœud de login (x86_64) : il faut un job
`armgpu`. Les checksums GPU doivent coïncider avec ceux du CPU.

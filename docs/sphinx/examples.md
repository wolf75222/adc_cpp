# Exemples

Les pilotes C++ sous `examples/` font la config, les diagnostics et l'I/O ; toute la
physique est dans `libadc` (façade) ou la pile générique `adc::adc` (moteur AMR/MPI). Les
animations sont régénérées par les scripts `scripts/make_*.py` et `scripts/plot_*.py`.

## Diocotron (façade `adc::solver`)

```bash
./build/bin/diocotron        out 128 500   # instabilité de base
./build/bin/diocotron_column out 128 500   # variante colonne
```

Tracés : `scripts/make_diocotron_gif.py`, `scripts/make_diocotron_column_gif.py`.
Figures : `docs/anim_diocotron.gif`, `docs/fig_diocotron_growth.png`,
`docs/fig_diocotron_modes.png`.

## AMR (moteur `adc::adc`)

```bash
./build/bin/diocotron_amr       out 128 500   # 2 niveaux
./build/bin/diocotron_amr3      out 128 500   # 3 niveaux emboîtés
./build/bin/diocotron_multipatch out 128 480  # multi-patch + regrid Berger-Rigoutsos
```

Tracés : `scripts/make_diocotron_amr3_gif.py`, `scripts/make_diocotron_multipatch_gif.py`.
Figures : `docs/anim_diocotron_amr3.gif`, `docs/anim_diocotron_multipatch.gif`. Détail :
tutoriels [04](https://github.com/wolf75222/adc_cpp/blob/master/tutorials/04_amr_multilevel.md)
et [05](https://github.com/wolf75222/adc_cpp/blob/master/tutorials/05_amr_multipatch.md).

## MPI distribué

```bash
mpirun -np 4 ./build-mpi/bin/diocotron_mpi out 128 600
```

Résultat bit-identique à np=1 (invariance au nombre de rangs).

## Banc de mesure

```bash
OMP_NUM_THREADS=4 ./build-omp/bin/bench_amr 512 100        # deux-fluides AP + coupleur AMR
OMP_NUM_THREADS=4 ./build-omp/bin/bench_amr 768 40 tf      # two-fluide seul (scaling)
python3 scripts/plot_bench_scaling.py /tmp/scaling.csv docs/fig_openmp_scaling.png
```

Mesures et interprétation (FFT vs MG, scaling OpenMP, saturation bande passante) :
[PERFORMANCE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/PERFORMANCE.md).

## GPU (Kokkos / GH200)

Les démos `examples/gpu/` sont compilées quand `-DADC_USE_KOKKOS=ON` ; elles héritent du
backend Kokkos de la cible `adc`, bit-identiques au CPU.

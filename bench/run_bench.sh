#!/usr/bin/env bash
# Pilote de profilage : configure + compile le harnais profile_step pour UN backend, puis le lance
# (serie ou MPI) sur une grille representative et imprime le tableau phase x temps x %. ZERO
# optimisation : ce script ne fait que MESURER. Le harnais est HORS du build par defaut (option
# ADC_BUILD_BENCH=OFF) ; ce script l'active explicitement, donc le CI n'est jamais touche.
#
# Usage :
#   bench/run_bench.sh serie                 # Serie CPU (defaut)
#   bench/run_bench.sh kokkos-omp  <Kroot>   # Kokkos OpenMP (Kokkos installe avec ENABLE_OPENMP)
#   bench/run_bench.sh kokkos-cuda <Kroot>   # Kokkos Cuda  (nvcc_wrapper ; GH200)
#   bench/run_bench.sh mpi          [NP]     # MPI CPU (NP rangs, defaut 2)
#   bench/run_bench.sh mpi-cuda <Kroot> [NP] # MPI + Kokkos Cuda (NP rangs, 1 GPU/rang)
#
# Variables : N (grille, defaut 256), STEPS (50), WARMUP (5), SOLVER (geometric_mg), LIMITER (minmod).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"          # racine du depot adc_cpp
MODE="${1:-serie}"
N="${N:-256}"; STEPS="${STEPS:-50}"; WARMUP="${WARMUP:-5}"
SOLVER="${SOLVER:-geometric_mg}"; LIMITER="${LIMITER:-minmod}"

run_bin() {  # $1 = build dir, $2... = lanceur eventuel (mpirun ...)
  local bdir="$1"; shift
  "$@" "$bdir/bin/profile_step" --n "$N" --steps "$STEPS" --warmup "$WARMUP" \
       --solver "$SOLVER" --limiter "$LIMITER"
}

case "$MODE" in
  serie)
    B="$ROOT/build-bench-serie"
    cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DADC_BUILD_TESTS=OFF -DADC_BUILD_BENCH=ON >/dev/null
    cmake --build "$B" --target profile_step -j 2 >/dev/null
    run_bin "$B"
    ;;
  kokkos-omp)
    KROOT="${2:?Kokkos_ROOT requis}"
    B="$ROOT/build-bench-komp"
    cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DADC_BUILD_TESTS=OFF -DADC_BUILD_BENCH=ON \
      -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" >/dev/null
    cmake --build "$B" --target profile_step -j 2 >/dev/null
    run_bin "$B"
    ;;
  kokkos-cuda)
    KROOT="${2:?Kokkos_ROOT requis}"
    B="$ROOT/build-bench-kcuda"
    cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DADC_BUILD_TESTS=OFF -DADC_BUILD_BENCH=ON \
      -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper" >/dev/null
    cmake --build "$B" --target profile_step -j 4 >/dev/null
    run_bin "$B"
    ;;
  mpi)
    NP="${2:-2}"
    B="$ROOT/build-bench-mpi"
    cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DADC_BUILD_TESTS=OFF -DADC_BUILD_BENCH=ON \
      -DADC_USE_MPI=ON >/dev/null
    cmake --build "$B" --target profile_step -j 2 >/dev/null
    run_bin "$B" mpirun -np "$NP"
    ;;
  mpi-cuda)
    KROOT="${2:?Kokkos_ROOT requis}"; NP="${3:-2}"
    B="$ROOT/build-bench-mpicuda"
    cmake -S "$ROOT" -B "$B" -DCMAKE_BUILD_TYPE=Release -DADC_BUILD_TESTS=OFF -DADC_BUILD_BENCH=ON \
      -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
      -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper" >/dev/null
    cmake --build "$B" --target profile_step -j 4 >/dev/null
    run_bin "$B" mpirun -np "$NP"
    ;;
  *)
    echo "mode inconnu : $MODE (serie|kokkos-omp|kokkos-cuda|mpi|mpi-cuda)" >&2
    exit 2
    ;;
esac

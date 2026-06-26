#!/usr/bin/env bash
# Construit scaling_step pour UN backend, puis joue un BALAYAGE (threads / rangs / tailles) et APPEND
# les lignes JSONL (schema pops_perf_v1) dans $OUT. MEMES recettes cmake que run_bench.sh ; SHA/branche
# injectes a la configuration. Le balayage depend du mode (cf. plan : strong/weak CPU/GPU/MPI).
#
# Usage :
#   bench/run_scaling.sh serie                  # 1 run de controle (serie)
#   bench/run_scaling.sh kokkos-omp  <Kroot>    # strong CPU : balayage OMP_NUM_THREADS = $THREADS
#   bench/run_scaling.sh kokkos-cuda <Kroot>    # GH200 mono-GPU : balayage tailles = $SIZES
#   bench/run_scaling.sh mpi                    # MPI CPU : balayage rangs = $RANKS
#   bench/run_scaling.sh mpi-cuda <Kroot>       # MPI+Cuda : balayage GPUs = $RANKS (1 GPU/rang)
#
# Variables : WORKLOAD (transport|poisson|amr), SCALING (strong|weak), N (4096 transport / 1024
#   poisson), STEPS (20), WARMUP (3), MAX_GRID (256), THREADS ("1 2 4 8"), RANKS ("1 2 4"),
#   SIZES ("512 1024 2048 4096"), MACHINE, OUT (fichier JSONL ; defaut out/scaling.jsonl).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MODE="${1:-serie}"
WORKLOAD="${WORKLOAD:-transport}"; SCALING="${SCALING:-strong}"
N="${N:-4096}"; STEPS="${STEPS:-20}"; WARMUP="${WARMUP:-3}"; MAX_GRID="${MAX_GRID:-256}"
THREADS="${THREADS:-1 2 4 8}"; RANKS="${RANKS:-1 2 4}"; SIZES="${SIZES:-512 1024 2048 4096}"
MACHINE="${MACHINE:-$(hostname -s 2>/dev/null || echo unknown)}"
OUT="${OUT:-$ROOT/out/scaling.jsonl}"
mkdir -p "$(dirname "$OUT")"
SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
cfg_common=( -DCMAKE_BUILD_TYPE=Release -DPOPS_BUILD_TESTS=OFF -DPOPS_BUILD_BENCH=ON
             -DPOPS_BUILD_SHA="$SHA" -DPOPS_BUILD_BRANCH="$BRANCH" )

run_one() {  # $1 = build dir, $2 = grille n, $3... = lanceur (mpirun -np K) eventuel
  local bdir="$1" nn="$2"; shift 2
  local line
  line="$("$@" "$bdir/bin/scaling_step" --workload "$WORKLOAD" --scaling "$SCALING" --n "$nn" \
          --steps "$STEPS" --warmup "$WARMUP" --max-grid "$MAX_GRID" --backend "$MODE" \
          --machine "$MACHINE" | tail -1)"
  echo "$line" | tee -a "$OUT"
}

build() {  # $1 = build dir ; flags supplementaires en $2...
  local bdir="$1"; shift
  cmake -S "$ROOT" -B "$bdir" "${cfg_common[@]}" "$@" >/dev/null
  cmake --build "$bdir" --target scaling_step -j 4 >/dev/null
}

echo "# scaling $MODE workload=$WORKLOAD scaling=$SCALING -> $OUT"
case "$MODE" in
  serie)
    B="$ROOT/build-scaling-serie"; build "$B"
    run_one "$B" "$N" ;;
  kokkos-omp)
    KROOT="${2:?Kokkos_ROOT requis}"; B="$ROOT/build-scaling-komp"
    build "$B" -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT"
    for th in $THREADS; do
      echo "# OMP_NUM_THREADS=$th"
      OMP_NUM_THREADS="$th" run_one "$B" "$N"
    done ;;
  kokkos-cuda)
    KROOT="${2:?Kokkos_ROOT requis}"; B="$ROOT/build-scaling-kcuda"
    build "$B" -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
      -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper"
    for nn in $SIZES; do
      echo "# GH200 mono-GPU n=$nn"
      run_one "$B" "$nn" ;
    done ;;
  mpi)
    B="$ROOT/build-scaling-mpi"; build "$B" -DPOPS_USE_MPI=ON
    for np in $RANKS; do
      echo "# np=$np"
      run_one "$B" "$N" mpirun -np "$np"
    done ;;
  mpi-cuda)
    KROOT="${2:?Kokkos_ROOT requis}"; B="$ROOT/build-scaling-mpicuda"
    build "$B" -DPOPS_USE_MPI=ON -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
      -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper"
    for np in $RANKS; do
      echo "# $np GPU(s) (1 rang/GPU)"
      run_one "$B" "$N" mpirun -np "$np"
    done ;;
  *)
    echo "mode inconnu : $MODE (serie|kokkos-omp|kokkos-cuda|mpi|mpi-cuda)" >&2; exit 2 ;;
esac
echo "# done -> $OUT"

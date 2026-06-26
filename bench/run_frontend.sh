#!/usr/bin/env bash
# Construit + lance le front C++ DIRECT (frontend_cpp) du CAS SUR pour UN backend, et imprime sa
# ligne JSONL (schema pops_perf_v1). MEMES recettes cmake que run_bench.sh ; le SHA/branche sont
# injectes a la configuration (le binaire ne lance pas git). ZERO optimisation : on ne fait que MESURER.
#
# Usage :
#   bench/run_frontend.sh serie                 # Serie CPU (defaut)
#   bench/run_frontend.sh kokkos-omp  <Kroot>   # Kokkos OpenMP (OMP_NUM_THREADS pilote les threads)
#   bench/run_frontend.sh kokkos-cuda <Kroot>   # Kokkos Cuda (nvcc_wrapper ; GH200)
#   bench/run_frontend.sh mpi          [NP]     # MPI CPU (NP rangs)
#   bench/run_frontend.sh mpi-cuda <Kroot> [NP] # MPI + Kokkos Cuda (NP rangs, 1 GPU/rang)
#
# Variables : N (256), STEPS (50), WARMUP (5), POISSON (off|on), MACHINE (hostname), OUT (fichier
# ou APPENDER la ligne JSONL ; defaut : stdout seulement).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MODE="${1:-serie}"
N="${N:-256}"; STEPS="${STEPS:-50}"; WARMUP="${WARMUP:-5}"; POISSON="${POISSON:-off}"
MACHINE="${MACHINE:-$(hostname -s 2>/dev/null || echo unknown)}"
SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

cfg_common=( -DCMAKE_BUILD_TYPE=Release -DADC_BUILD_TESTS=OFF -DADC_BUILD_BENCH=ON
             -DADC_BUILD_SHA="$SHA" -DADC_BUILD_BRANCH="$BRANCH" )

emit() {  # $1 = build dir, $2... = lanceur eventuel (mpirun ...)
  local bdir="$1"; shift
  local line
  line="$("$@" "$bdir/bin/frontend_cpp" --n "$N" --steps "$STEPS" --warmup "$WARMUP" \
          --poisson "$POISSON" --backend "$MODE" --machine "$MACHINE" | tail -1)"
  echo "$line"
  if [[ -n "${OUT:-}" ]]; then echo "$line" >> "$OUT"; fi
}

case "$MODE" in
  serie)
    B="$ROOT/build-frontend-serie"
    cmake -S "$ROOT" -B "$B" "${cfg_common[@]}" >/dev/null
    cmake --build "$B" --target frontend_cpp -j 2 >/dev/null
    emit "$B" ;;
  kokkos-omp)
    KROOT="${2:?Kokkos_ROOT requis}"; B="$ROOT/build-frontend-komp"
    cmake -S "$ROOT" -B "$B" "${cfg_common[@]}" -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" >/dev/null
    cmake --build "$B" --target frontend_cpp -j 2 >/dev/null
    emit "$B" ;;
  kokkos-cuda)
    KROOT="${2:?Kokkos_ROOT requis}"; B="$ROOT/build-frontend-kcuda"
    cmake -S "$ROOT" -B "$B" "${cfg_common[@]}" -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
      -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper" >/dev/null
    cmake --build "$B" --target frontend_cpp -j 4 >/dev/null
    emit "$B" ;;
  mpi)
    NP="${2:-2}"; B="$ROOT/build-frontend-mpi"
    cmake -S "$ROOT" -B "$B" "${cfg_common[@]}" -DADC_USE_MPI=ON >/dev/null
    cmake --build "$B" --target frontend_cpp -j 2 >/dev/null
    emit "$B" mpirun -np "$NP" ;;
  mpi-cuda)
    KROOT="${2:?Kokkos_ROOT requis}"; NP="${3:-2}"; B="$ROOT/build-frontend-mpicuda"
    cmake -S "$ROOT" -B "$B" "${cfg_common[@]}" -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON \
      -DKokkos_ROOT="$KROOT" -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper" >/dev/null
    cmake --build "$B" --target frontend_cpp -j 4 >/dev/null
    emit "$B" mpirun -np "$NP" ;;
  *)
    echo "mode inconnu : $MODE (serie|kokkos-omp|kokkos-cuda|mpi|mpi-cuda)" >&2; exit 2 ;;
esac

#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=adc_mpicuda_halo
#SBATCH --output=/home/rmdraux/adc_perf_master_halo_20260608/logs/mpicuda_halo_%j.out
#SBATCH --error=/home/rmdraux/adc_perf_master_halo_20260608/logs/mpicuda_halo_%j.err

set -u

BASE="${ADC_PERF_BASE:-$HOME/adc_perf_master_halo_20260608}"
SRC="$BASE/adc_cpp"
RESULTS="$BASE/results"
LOGS="$BASE/logs"
KROOT="${ADC_KOKKOS_CUDA_ROOT:-$HOME/adc_gpu_p1/kinstall}"
NW="$KROOT/bin/nvcc_wrapper"
mkdir -p "$RESULTS" "$LOGS"

echo "=== ADC MPI+CUDA HALO job=$SLURM_JOBID node=$(hostname) date=$(date) ==="

module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda 2>/dev/null || true
spack load cmake@3.30.5 2>/dev/null || true
export KOKKOS_DISABLE_WARNINGS=1

which cmake srun "$NW" || true
cmake --version | head -1 || true
nvidia-smi -L || true

if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone https://github.com/wolf75222/adc_cpp.git "$SRC"
fi
git -C "$SRC" fetch origin --prune
git -C "$SRC" checkout --detach origin/master

# Base mesuree = master recent (#254 halo pinned host), avec uniquement les harnais perf de la branche.
git -C "$SRC" checkout origin/feat/perf-campaign-bench -- \
  bench/CMakeLists.txt bench/scaling_step.cpp

ADC_CPP_COMMIT="$(git -C "$SRC" rev-parse --short HEAD)"
ADC_CPP_BRANCH="master+perf-harness"
echo "ADC_CPP_COMMIT=$ADC_CPP_COMMIT"
echo "ADC_CPP_BRANCH=$ADC_CPP_BRANCH"
git -C "$SRC" log -1 --oneline
git -C "$SRC" diff --stat

if [ ! -x "$NW" ]; then
  echo "MISSING_NVCC_WRAPPER $NW"
  exit 2
fi

BUILD="$BASE/build_mpicuda_halo"
rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
  -DCMAKE_CXX_COMPILER="$NW" \
  -DADC_BUILD_BENCH=ON -DADC_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DADC_BUILD_SHA="$ADC_CPP_COMMIT" -DADC_BUILD_BRANCH="$ADC_CPP_BRANCH" \
  > "$LOGS/cfg_${SLURM_JOBID}.log" 2>&1
rc=$?
echo "CONFIG rc=$rc"
[ $rc -eq 0 ] || { tail -120 "$LOGS/cfg_${SLURM_JOBID}.log"; exit $rc; }

cmake --build "$BUILD" --target scaling_step -j 8 > "$LOGS/build_${SLURM_JOBID}.log" 2>&1
rc=$?
echo "BUILD rc=$rc"
[ $rc -eq 0 ] || { tail -160 "$LOGS/build_${SLURM_JOBID}.log"; exit $rc; }

OUT="$RESULTS/mpicuda_halo_${ADC_CPP_COMMIT}_${SLURM_JOBID}.jsonl"
: > "$OUT"

run_case() {
  local label="$1" np="$2" n="$3" scaling="$4"
  local log="$LOGS/${label}_np${np}_n${n}_${SLURM_JOBID}.log"
  echo "RUN $label np=$np n=$n scaling=$scaling"
  timeout 240 srun -n "$np" --gpus-per-task=1 "$BUILD/bin/scaling_step" \
    --workload transport --scaling "$scaling" --n "$n" --steps 12 --warmup 2 \
    --max-grid 512 --backend mpi-cuda --machine romeo-gh200 \
    > "$log" 2>&1
  rc=$?
  echo "RC=$rc log=$log"
  if [ $rc -eq 0 ]; then
    tail -1 "$log" | tee -a "$OUT"
  else
    echo "{\"schema\":\"adc_perf_v1\",\"front\":\"cpp_scaling\",\"adc_cpp_sha\":\"$ADC_CPP_COMMIT\",\"adc_cpp_branch\":\"$ADC_CPP_BRANCH\",\"backend\":\"mpi-cuda\",\"machine\":\"romeo-gh200\",\"ranks\":$np,\"workload\":\"transport\",\"scaling\":\"$scaling\",\"nx\":$n,\"status\":\"failed\",\"rc\":$rc}" | tee -a "$OUT"
    tail -80 "$log"
  fi
}

# Strong scaling : taille globale fixe, un GPU par rang.
run_case strong 1 2048 strong
run_case strong 2 2048 strong
run_case strong 4 2048 strong

# Weak scaling 2D : taille locale constante pour np carres parfaits.
run_case weak 1 1024 weak
run_case weak 4 2048 weak

echo "=== DONE ==="
echo "OUT=$OUT"
cat "$OUT"

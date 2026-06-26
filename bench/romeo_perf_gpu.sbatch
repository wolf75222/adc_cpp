#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=pops_perf_gpu
#SBATCH --output=/home/rmdraux/pops_perf_20260608/logs/gpu_%j.out
#SBATCH --error=/home/rmdraux/pops_perf_20260608/logs/gpu_%j.err

set -u
BASE="$HOME/pops_perf_20260608"
SRC="$BASE/adc_cpp"
CASES="$BASE/adc_cases"
RESULTS="$BASE/results"
LOGS="$BASE/logs"
mkdir -p "$RESULTS" "$LOGS"

echo "=== PoPS GPU PERF job=$SLURM_JOBID node=$(hostname) arch=$(uname -m) date=$(date) ==="

module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda
spack load cmake@3.30.5 2>/dev/null || true
export KOKKOS_DISABLE_WARNINGS=1

echo "--- toolchain ---"
which cmake c++ "$HOME/pops_gpu_p1/kinstall/bin/nvcc_wrapper" srun || true
cmake --version | head -1 || true
nvidia-smi -L || true
ompi_info --version 2>/dev/null | head -2 || true

if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone https://github.com/wolf75222/adc_cpp.git "$SRC"
fi
if [ ! -d "$CASES/.git" ]; then
  rm -rf "$CASES"
  git clone https://github.com/wolf75222/adc_cases.git "$CASES"
fi
git -C "$SRC" fetch origin --prune
git -C "$SRC" checkout --detach origin/master
git -C "$CASES" fetch origin --prune
git -C "$CASES" checkout --detach origin/master
POPS_CPP_COMMIT="$(git -C "$SRC" rev-parse --short HEAD)"
POPS_CASES_COMMIT="$(git -C "$CASES" rev-parse --short HEAD)"
echo "POPS_CPP_COMMIT=$POPS_CPP_COMMIT"
echo "POPS_CASES_COMMIT=$POPS_CASES_COMMIT"

KROOT="$HOME/pops_gpu_p1/kinstall"
NW="$KROOT/bin/nvcc_wrapper"
if [ ! -x "$NW" ]; then
  echo "MISSING_NVCC_WRAPPER $NW"
  exit 2
fi

N_STRONG=256
N_WEAK_LOCAL=128
STEPS=8
WARMUP=2
LIMITER=minmod
SOLVER=geometric_mg

PHASE_CSV="$RESULTS/perf_phases_gpu_${POPS_CPP_COMMIT}_${SLURM_JOBID}.csv"
SCALING_CSV="$RESULTS/perf_scaling_gpu_${POPS_CPP_COMMIT}_${SLURM_JOBID}.csv"
printf "commit,case,backend,np,threads,n,steps,phase,total_s,per_step_ms,pct\n" > "$PHASE_CSV"
printf "commit,repo,case,scaling,backend,machine,np,threads,gpus,n_global,n_local,steps,warmup,wall_s,per_step_ms,speedup,efficiency,notes\n" > "$SCALING_CSV"

parse_profile_log () {
  local log="$1" case="$2" scaling="$3" backend="$4" np="$5" gpus="$6" n="$7" notes="$8"
  python3 - "$log" "$PHASE_CSV" "$SCALING_CSV" "$POPS_CPP_COMMIT" "$case" "$scaling" "$backend" "$np" "$gpus" "$n" "$STEPS" "$WARMUP" "$notes" <<'PY'
import csv, re, sys
log, phase_csv, scaling_csv, commit, case, scaling, backend, np, gpus, n, steps, warmup, notes = sys.argv[1:]
phase_re = re.compile(r"^(transport|poisson|halos|aux_derive|reduction|fence|alloc_tmp|TOTAL)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)%")
wall_re = re.compile(r"wall_total=([-\d.eE+]+) s.*per_step=([-\d.eE+]+) ms")
rows = []
wall = ""
per_step = ""
with open(log, errors="replace") as fh:
    for line in fh:
        m = wall_re.search(line)
        if m:
            wall, per_step = m.group(1), m.group(2)
        m = phase_re.match(line.strip())
        if m:
            rows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
with open(phase_csv, "a", newline="") as fh:
    wr = csv.writer(fh)
    for phase, total_s, step_ms, pct in rows:
        wr.writerow([commit, case, backend, np, "", n, steps, phase, total_s, step_ms, pct])
total = next((r for r in rows if r[0] == "TOTAL"), None)
if total:
    if not per_step:
        per_step = total[2]
    if not wall:
        wall = total[1]
    with open(scaling_csv, "a", newline="") as fh:
        csv.writer(fh).writerow([commit, "adc_cpp", case, scaling, backend, "romeo-armgpu",
                                 np, "", gpus, n, "", steps, warmup, wall, per_step, "", "", notes])
PY
}

configure_bench () {
  local build="$1"; shift
  rm -rf "$build"
  cmake -S "$SRC" -B "$build" "$@" \
    -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" -DCMAKE_CXX_COMPILER="$NW" \
    -DPOPS_BUILD_BENCH=ON -DPOPS_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    > "$LOGS/$(basename "$build")_cfg.log" 2>&1
  local rc=$?
  echo "CONFIG $(basename "$build") rc=$rc"
  [ $rc -eq 0 ] || { tail -80 "$LOGS/$(basename "$build")_cfg.log"; return $rc; }
  cmake --build "$build" --target profile_step -j 8 > "$LOGS/$(basename "$build")_build.log" 2>&1
  rc=$?
  echo "BUILD $(basename "$build") rc=$rc"
  [ $rc -eq 0 ] || tail -80 "$LOGS/$(basename "$build")_build.log"
  return $rc
}

BUILD_CUDA="$BASE/build_gpu_kokkos"
if configure_bench "$BUILD_CUDA"; then
  log="$LOGS/profile_gpu_strong_kokkos-cuda_np1_n${N_STRONG}.log"
  echo "RUN kokkos-cuda strong np=1 n=$N_STRONG"
  timeout 600 srun -n 1 --gpus-per-task=1 "$BUILD_CUDA/bin/profile_step" --n "$N_STRONG" --steps "$STEPS" --warmup "$WARMUP" --solver "$SOLVER" --limiter "$LIMITER" > "$log" 2>&1
  echo "RC=$? log=$log"
  parse_profile_log "$log" "profile-step" "strong" "kokkos-cuda" 1 1 "$N_STRONG" "single-gpu"
fi

BUILD_MPI_CUDA="$BASE/build_gpu_mpi_kokkos"
if configure_bench "$BUILD_MPI_CUDA" -DPOPS_USE_MPI=ON; then
  for np in 1 2 4; do
    log="$LOGS/profile_gpu_strong_mpi-kokkos-cuda_np${np}_n${N_STRONG}.log"
    echo "RUN mpi+kokkos-cuda strong np=$np n=$N_STRONG"
    timeout 600 srun -n "$np" --gpus-per-task=1 "$BUILD_MPI_CUDA/bin/profile_step" --n "$N_STRONG" --steps "$STEPS" --warmup "$WARMUP" --solver "$SOLVER" --limiter "$LIMITER" > "$log" 2>&1
    echo "RC=$? log=$log"
    parse_profile_log "$log" "profile-step" "strong" "mpi+kokkos-cuda" "$np" "$np" "$N_STRONG" "one-node-gh200"
  done
  for np in 1 4; do
    n=$((N_WEAK_LOCAL * np))
    log="$LOGS/profile_gpu_weak_mpi-kokkos-cuda_np${np}_n${n}.log"
    echo "RUN mpi+kokkos-cuda weak np=$np n=$n"
    timeout 600 srun -n "$np" --gpus-per-task=1 "$BUILD_MPI_CUDA/bin/profile_step" --n "$n" --steps "$STEPS" --warmup "$WARMUP" --solver "$SOLVER" --limiter "$LIMITER" > "$log" 2>&1
    echo "RC=$? log=$log"
    parse_profile_log "$log" "profile-step" "weak" "mpi+kokkos-cuda" "$np" "$np" "$n" "weak-local-n${N_WEAK_LOCAL}-per-rank-axis"
  done
fi

AMR_SRC="$BASE/gpu_amr_src"
mkdir -p "$AMR_SRC"
cp "$SRC/python/tests/gpu/amrmpi_integrated.cpp" "$AMR_SRC/amrmpi_integrated.cpp"
cp "$SRC/python/tests/gpu/amrmpi_CMakeLists.txt" "$AMR_SRC/CMakeLists.txt"
AMR_BUILD="$BASE/build_gpu_amrmpi"
rm -rf "$AMR_BUILD"
if cmake -S "$AMR_SRC" -B "$AMR_BUILD" -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ROOT="$KROOT" \
    -DPOPS_INCLUDE="$SRC/include" -DPOPS_SRC="$SRC/python" -DCMAKE_BUILD_TYPE=Release > "$LOGS/amrmpi_cfg.log" 2>&1 && \
   cmake --build "$AMR_BUILD" -j 8 > "$LOGS/amrmpi_build.log" 2>&1; then
  AMR_OUT="$LOGS/amrmpi_gpu_${POPS_CPP_COMMIT}_${SLURM_JOBID}.txt"
  : > "$AMR_OUT"
  for np in 1 2 4; do
    echo "--- amrmpi np=$np ---" | tee -a "$AMR_OUT"
    timeout 900 srun -n "$np" --gpus-per-task=1 "$AMR_BUILD/amrmpi_integrated" 128 2>&1 | tee -a "$AMR_OUT"
  done
  python3 - "$AMR_OUT" "$SCALING_CSV" "$POPS_CPP_COMMIT" <<'PY'
import csv, re, sys
path, out, commit = sys.argv[1:]
sig = {}
tms = {}
for line in open(path, errors="replace"):
    m = re.search(r"AMRMPI\[(\w+)\] np=(\d+).*cmax=([-\d.eE+]+).*cmax_crossrank_spread=([-\d.eE+]+)", line)
    if m:
        sig[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    m = re.search(r"AMRMPI\[(\w+)\].*per_step_ms=([-\d.eE+]+).*n=(\d+), measured=(\d+)", line)
    if m:
        tag = m.group(1)
        nps = [n for (t, n) in sig if t == tag]
        if nps:
            tms[(tag, max(nps))] = (float(m.group(2)), int(m.group(3)), int(m.group(4)))
with open(out, "a", newline="") as fh:
    wr = csv.writer(fh)
    for (tag, np), (ms, n, steps) in sorted(tms.items()):
        wr.writerow([commit, "adc_cpp", "amr-synthetic-%s" % tag, "strong", "mpi+kokkos-cuda",
                     "romeo-armgpu", np, "", np, n, "", steps, 4, "", ms, "", "",
                     "amrmpi_integrated cmax_spread=%s" % (sig.get((tag, np), ("", ""))[1],)])
PY
else
  echo "AMRMPI_BUILD_FAIL"
  tail -120 "$LOGS/amrmpi_cfg.log" "$LOGS/amrmpi_build.log" 2>/dev/null || true
fi

echo "=== GPU PERF DONE ==="
echo "PHASE_CSV=$PHASE_CSV"
echo "SCALING_CSV=$SCALING_CSV"

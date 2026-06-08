#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=adc_perf_mbox_cpu
#SBATCH --output=/home/rmdraux/adc_perf_20260608/logs/transport_mbox_cpu_%j.out
#SBATCH --error=/home/rmdraux/adc_perf_20260608/logs/transport_mbox_cpu_%j.err

set -u

BASE="$HOME/adc_perf_20260608"
SRC="$BASE/adc_cpp"
RESULTS="$BASE/results"
LOGS="$BASE/logs"
SCRIPT_DIR="${ADC_PERF_SCRIPT_DIR:-$BASE/scripts}"
TARGET_COMMIT="${ADC_CPP_TARGET_COMMIT:-origin/master}"
KROOT="${ADC_KOKKOS_CPU_ROOT:-$HOME/adc_cpu_mpiomp/kinstall_omp}"

mkdir -p "$RESULTS" "$LOGS"

echo "=== ADC TRANSPORT MBOX CPU job=$SLURM_JOBID node=$(hostname) date=$(date) ==="

romeo_load_x64cpu_env >/dev/null 2>&1
eval "$(spack load --sh /fmrcj5j 2>/dev/null)"   # gcc 14.2.0
eval "$(spack load --sh /kvj6wp2 2>/dev/null)"   # cmake 3.30.5
eval "$(spack load --sh /lni3xp2 2>/dev/null)"   # openmpi 5.0.5
eval "$(spack load --sh /43yg4qb 2>/dev/null)"   # libmd for cmake on compute nodes
eval "$(spack load --sh /2celb2j 2>/dev/null)"   # python 3.11.9 gcc14
LIBMD_DIR=/apps/2025/spack_install/linux-rhel9-zen4/linux-rhel9-zen4/gcc-14.2.0/libmd-1.0.4-43yg4qbrgl3k5orluem3lljkhxyr7tqa/lib
export LD_LIBRARY_PATH="$LIBMD_DIR:${LD_LIBRARY_PATH:-}"
export OMPI_CXX=g++ OMPI_CC=gcc
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_WAIT_POLICY=passive
export KOKKOS_DISABLE_WARNINGS=1
export OMPI_MCA_mtl=^ofi
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_rmaps_base_oversubscribe=1

which gcc g++ cmake python3 mpicxx srun || true
gcc --version | head -1 || true
g++ --version | head -1 || true
cmake --version | head -1 || true
python3 --version || true
mpicxx --version 2>&1 | head -1 || true

if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone https://github.com/wolf75222/adc_cpp.git "$SRC"
fi
git -C "$SRC" fetch origin --prune
git -C "$SRC" checkout --detach "$TARGET_COMMIT"
ADC_CPP_COMMIT="$(git -C "$SRC" rev-parse --short HEAD)"
echo "ADC_CPP_COMMIT=$ADC_CPP_COMMIT"

if [ -f "$SCRIPT_DIR/profile_transport_mbox.cpp" ]; then
  cp "$SCRIPT_DIR/profile_transport_mbox.cpp" "$SRC/bench/profile_transport_mbox.cpp"
fi
if ! grep -q "profile_transport_mbox" "$SRC/bench/CMakeLists.txt"; then
  {
    echo ""
    echo "add_executable(profile_transport_mbox profile_transport_mbox.cpp)"
    echo "target_link_libraries(profile_transport_mbox PRIVATE adc::adc)"
  } >> "$SRC/bench/CMakeLists.txt"
fi

if [ ! -f "$SRC/bench/profile_transport_mbox.cpp" ]; then
  echo "MISSING profile_transport_mbox.cpp after injection"
  exit 3
fi

if [ ! -f "$KROOT/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KROOT/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
  echo "MISSING_KOKKOS_OPENMP $KROOT"
  exit 2
fi

STEPS="${ADC_TRANSPORT_STEPS:-24}"
WARMUP="${ADC_TRANSPORT_WARMUP:-5}"
LIMITER="${ADC_TRANSPORT_LIMITER:-minmod}"
MAX_GRID="${ADC_TRANSPORT_MAX_GRID:-128}"
N_STRONG="${ADC_TRANSPORT_N_STRONG:-1024}"
N_WEAK_ROOT="${ADC_TRANSPORT_N_WEAK_ROOT:-384}"

PHASE_CSV="$RESULTS/perf_phases_transport_mbox_cpu_${ADC_CPP_COMMIT}_${SLURM_JOBID}.csv"
SCALING_CSV="$RESULTS/perf_scaling_transport_mbox_cpu_${ADC_CPP_COMMIT}_${SLURM_JOBID}.csv"
printf "commit,case,backend,np,threads,n,steps,phase,total_s,per_step_ms,pct\n" > "$PHASE_CSV"
printf "commit,repo,case,scaling,backend,machine,np,threads,gpus,n_global,n_local,steps,warmup,wall_s,per_step_ms,speedup,efficiency,notes\n" > "$SCALING_CSV"

parse_transport_log () {
  local log="$1" case="$2" scaling="$3" backend="$4" np="$5" threads="$6" n="$7" nlocal="$8" notes="$9"
  python3 - "$log" "$PHASE_CSV" "$SCALING_CSV" "$ADC_CPP_COMMIT" "$case" "$scaling" "$backend" "$np" "$threads" "$n" "$nlocal" "$STEPS" "$WARMUP" "$notes" <<'PY'
import csv, re, sys
log, phase_csv, scaling_csv, commit, case, scaling, backend, np, threads, n, nlocal, steps, warmup, notes = sys.argv[1:]
phase_re = re.compile(r"^([A-Za-z_]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)%")
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
if not rows:
    print("PARSE_EMPTY", log)
    sys.exit(7)
with open(phase_csv, "a", newline="") as fh:
    wr = csv.writer(fh)
    for phase, total_s, step_ms, pct in rows:
        wr.writerow([commit, case, backend, np, threads, n, steps, phase, total_s, step_ms, pct])
total = next((r for r in rows if r[0] == "TOTAL"), None)
if total:
    if not per_step:
        per_step = total[2]
    if not wall:
        wall = total[1]
    with open(scaling_csv, "a", newline="") as fh:
        csv.writer(fh).writerow([commit, "adc_cpp", case, scaling, backend, "romeo-x64cpu",
                                 np, threads, 0, n, nlocal, steps, warmup, wall, per_step, "", "", notes])
PY
}

configure_bench () {
  local build="$1"; shift
  rm -rf "$build"
  cmake -S "$SRC" -B "$build" "$@" \
    -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
    -DADC_BUILD_BENCH=ON -DADC_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    > "$LOGS/$(basename "$build")_cfg_${SLURM_JOBID}.log" 2>&1
  local rc=$?
  echo "CONFIG $(basename "$build") rc=$rc"
  [ $rc -eq 0 ] || { tail -120 "$LOGS/$(basename "$build")_cfg_${SLURM_JOBID}.log"; return $rc; }
  cmake --build "$build" --target profile_transport_mbox -j 16 > "$LOGS/$(basename "$build")_build_${SLURM_JOBID}.log" 2>&1
  rc=$?
  echo "BUILD $(basename "$build") rc=$rc"
  [ $rc -eq 0 ] || tail -160 "$LOGS/$(basename "$build")_build_${SLURM_JOBID}.log"
  return $rc
}

BUILD_KOKKOS="$BASE/build_transport_mbox_cpu_kokkos"
if configure_bench "$BUILD_KOKKOS"; then
  for thr in 1 2 4 8 16; do
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/transport_mbox_strong_kokkos-openmp_np1_thr${thr}_n${N_STRONG}_${SLURM_JOBID}.log"
    echo "RUN transport mbox kokkos-openmp strong threads=$thr n=$N_STRONG"
    timeout 900 "$BUILD_KOKKOS/bin/profile_transport_mbox" --n "$N_STRONG" --max-grid "$MAX_GRID" --steps "$STEPS" --warmup "$WARMUP" --limiter "$LIMITER" > "$log" 2>&1
    rc=$?
    echo "RC=$rc log=$log"
    [ $rc -eq 0 ] && parse_transport_log "$log" "transport-euler-periodic-mbox" "strong" "kokkos-openmp" 1 "$thr" "$N_STRONG" "" "single-rank-true-multibox"
  done
fi

BUILD_MPI="$BASE/build_transport_mbox_cpu_mpi_kokkos"
if configure_bench "$BUILD_MPI" -DADC_USE_MPI=ON -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc; then
  for np in 1 2 4 8; do
    thr=4
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/transport_mbox_strong_mpi-kokkos-openmp_np${np}_thr${thr}_n${N_STRONG}_${SLURM_JOBID}.log"
    echo "RUN transport mbox mpi+kokkos-openmp strong np=$np threads=$thr n=$N_STRONG"
    timeout 900 srun -n "$np" --cpus-per-task="$thr" --cpu-bind=none "$BUILD_MPI/bin/profile_transport_mbox" --n "$N_STRONG" --max-grid "$MAX_GRID" --steps "$STEPS" --warmup "$WARMUP" --limiter "$LIMITER" > "$log" 2>&1
    rc=$?
    echo "RC=$rc log=$log"
    [ $rc -eq 0 ] && parse_transport_log "$log" "transport-euler-periodic-mbox" "strong" "mpi+kokkos-openmp" "$np" "$thr" "$N_STRONG" "" "one-node-true-multibox"
  done

  for np in 1 2 4 8; do
    thr=4
    n=$(python3 - "$N_WEAK_ROOT" "$np" <<'PY'
import math, sys
root = int(sys.argv[1])
np = int(sys.argv[2])
print(max(16, int(round(root * math.sqrt(np)))))
PY
)
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/transport_mbox_weak_mpi-kokkos-openmp_np${np}_thr${thr}_n${n}_${SLURM_JOBID}.log"
    echo "RUN transport mbox mpi+kokkos-openmp weak np=$np threads=$thr n=$n local_root=$N_WEAK_ROOT"
    timeout 900 srun -n "$np" --cpus-per-task="$thr" --cpu-bind=none "$BUILD_MPI/bin/profile_transport_mbox" --n "$n" --max-grid "$MAX_GRID" --steps "$STEPS" --warmup "$WARMUP" --limiter "$LIMITER" > "$log" 2>&1
    rc=$?
    echo "RC=$rc log=$log"
    [ $rc -eq 0 ] && parse_transport_log "$log" "transport-euler-periodic-mbox" "weak" "mpi+kokkos-openmp" "$np" "$thr" "$n" "$N_WEAK_ROOT" "weak-2d-local-root-${N_WEAK_ROOT}"
  done
fi

echo "=== TRANSPORT MBOX CPU DONE ==="
echo "PHASE_CSV=$PHASE_CSV"
echo "SCALING_CSV=$SCALING_CSV"

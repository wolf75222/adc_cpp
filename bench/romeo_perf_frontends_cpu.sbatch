#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=pops_perf_frontends
#SBATCH --output=/home/rmdraux/pops_perf_20260608/logs/frontends_cpu_%j.out
#SBATCH --error=/home/rmdraux/pops_perf_20260608/logs/frontends_cpu_%j.err

set -u

BASE="${POPS_PERF_BASE:-$HOME/pops_perf_20260608}"
SRC="$BASE/adc_cpp"
RESULTS="$BASE/results"
LOGS="$BASE/logs"
FRONT="$BASE/frontend_cpu"
SCRIPT_DIR="${POPS_PERF_SCRIPT_DIR:-$BASE/scripts}"
PY_HARNESS="$SCRIPT_DIR/frontend_bench_romeo.py"
CPP_HARNESS="$SCRIPT_DIR/frontend_bench.cpp"
CPP_CMAKE="$SCRIPT_DIR/frontend_CMakeLists.txt"
KOKKOS_SRC="${POPS_KOKKOS_SRC:-$HOME/pops_cpu_mpiomp/kokkos}"
KROOT="${POPS_KOKKOS_CPU_PIC_ROOT:-$BASE/kinstall_omp_pic}"
KBUILD="${POPS_KOKKOS_CPU_PIC_BUILD:-$BASE/kbuild_omp_pic}"
TARGET_COMMIT="${POPS_CPP_TARGET_COMMIT:-1f9fb4a}"
RUN_CPP_FRONTEND="${RUN_CPP_FRONTEND:-1}"

mkdir -p "$RESULTS" "$LOGS" "$FRONT"

echo "=== ADC CPU FRONTENDS job=$SLURM_JOBID node=$(hostname) date=$(date) ==="

romeo_load_x64cpu_env >/dev/null 2>&1
eval "$(spack load --sh /fmrcj5j 2>/dev/null)"   # gcc 14.2.0
eval "$(spack load --sh /kvj6wp2 2>/dev/null)"   # cmake 3.30.5
eval "$(spack load --sh /43yg4qb 2>/dev/null)"   # libmd
eval "$(spack load --sh /2celb2j 2>/dev/null)"   # python
eval "$(spack load --sh /rkvb73h 2>/dev/null)"   # numpy
eval "$(spack load --sh /j4cl5xe 2>/dev/null)"   # pybind11

LIBMD_DIR=/apps/2025/spack_install/linux-rhel9-zen4/linux-rhel9-zen4/gcc-14.2.0/libmd-1.0.4-43yg4qbrgl3k5orluem3lljkhxyr7tqa/lib
export LD_LIBRARY_PATH="$LIBMD_DIR:${LD_LIBRARY_PATH:-}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_WAIT_POLICY=passive
export KOKKOS_DISABLE_WARNINGS=1

which gcc g++ cmake python3 || true
gcc --version | head -1 || true
g++ --version | head -1 || true
cmake --version | head -1 || true
python3 --version || true

if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone https://github.com/wolf75222/adc_cpp.git "$SRC"
fi
git -C "$SRC" fetch origin --prune
git -C "$SRC" checkout --detach "$TARGET_COMMIT"
BASE_COMMIT="$(git -C "$SRC" rev-parse --short HEAD)"
PATCH_TAG=""
if [ -f "$SCRIPT_DIR/dsl.py" ]; then
  cp "$SCRIPT_DIR/dsl.py" "$SRC/python/pops/dsl.py"
  PATCH_TAG="+localdsl"
fi
if [ -f "$SCRIPT_DIR/abi_key.hpp" ]; then
  cp "$SCRIPT_DIR/abi_key.hpp" "$SRC/include/pops/runtime/abi_key.hpp"
  if [ -z "$PATCH_TAG" ]; then
    PATCH_TAG="+localabi"
  else
    PATCH_TAG="${PATCH_TAG}+abi"
  fi
fi
if [ -n "$PATCH_TAG" ]; then
  echo "APPLY_LOCAL_PATCHES tag=$PATCH_TAG"
  git -C "$SRC" diff --stat || true
fi
POPS_CPP_COMMIT="${POPS_CPP_RUN_LABEL:-${BASE_COMMIT}${PATCH_TAG}}"
echo "POPS_CPP_COMMIT=$POPS_CPP_COMMIT"

if [ ! -f "$FRONT/frontend_bench.cpp" ] && [ -f "$CPP_HARNESS" ]; then
  cp "$CPP_HARNESS" "$FRONT/frontend_bench.cpp"
fi
if [ ! -f "$FRONT/CMakeLists.txt" ] && [ -f "$CPP_CMAKE" ]; then
  cp "$CPP_CMAKE" "$FRONT/CMakeLists.txt"
fi
if [ ! -f "$FRONT/frontend_bench.py" ] && [ -f "$PY_HARNESS" ]; then
  cp "$PY_HARNESS" "$FRONT/frontend_bench.py"
fi

if [ ! -f "$FRONT/frontend_bench.cpp" ] || [ ! -f "$FRONT/frontend_bench.py" ]; then
  echo "MISSING_FRONTEND_HARNESS $FRONT"
  exit 3
fi

if [ ! -f "$KROOT/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KROOT/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
  if [ ! -d "$KOKKOS_SRC" ]; then
    echo "MISSING_KOKKOS_SRC $KOKKOS_SRC"
    exit 4
  fi
  rm -rf "$KBUILD" "$KROOT"
  cmake -S "$KOKKOS_SRC" -B "$KBUILD" \
    -DCMAKE_INSTALL_PREFIX="$KROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_TESTS=OFF \
    -DKokkos_ENABLE_BENCHMARKS=OFF \
    > "$LOGS/kokkos_pic_cfg_${SLURM_JOBID}.log" 2>&1
  rc=$?
  echo "CONFIG kokkos_pic rc=$rc"
  [ $rc -eq 0 ] || { tail -120 "$LOGS/kokkos_pic_cfg_${SLURM_JOBID}.log"; exit $rc; }
  cmake --build "$KBUILD" -j 16 > "$LOGS/kokkos_pic_build_${SLURM_JOBID}.log" 2>&1
  rc=$?
  echo "BUILD kokkos_pic rc=$rc"
  [ $rc -eq 0 ] || { tail -120 "$LOGS/kokkos_pic_build_${SLURM_JOBID}.log"; exit $rc; }
  cmake --install "$KBUILD" > "$LOGS/kokkos_pic_install_${SLURM_JOBID}.log" 2>&1
  rc=$?
  echo "INSTALL kokkos_pic rc=$rc"
  [ $rc -eq 0 ] || { tail -120 "$LOGS/kokkos_pic_install_${SLURM_JOBID}.log"; exit $rc; }
else
  echo "REUSE_KOKKOS_PIC $KROOT"
fi

FRONTEND_CSV="$RESULTS/perf_frontends_cpu_pic_${POPS_CPP_COMMIT}_${SLURM_JOBID}.csv"
printf "commit,case,frontend,backend,n,steps,dt,cache_state,setup_ms,compile_ms,advance_ms,step_loop_ms,extract_ms,total_ms,ratio_vs_cpp,notes\n" > "$FRONTEND_CSV"

if [ "$RUN_CPP_FRONTEND" = "0" ]; then
  PREV_COMMIT="${POPS_CPP_PREVIOUS_COMMIT:-$BASE_COMMIT}"
  prev="$(find "$RESULTS" -maxdepth 1 \( -name "perf_frontends_cpu_pic_${PREV_COMMIT}_*.csv" -o -name "perf_frontends_cpu_${PREV_COMMIT}_*.csv" \) -print 2>/dev/null | sort | while read -r f; do
    if grep -q ',cpp-native,' "$f"; then
      printf '%s\n' "$f"
      break
    fi
  done)"
  if [ -n "$prev" ]; then
    echo "REUSE_CPP_FRONTEND_ROWS $prev"
    awk -F, -v OFS=, -v commit="$POPS_CPP_COMMIT" '$3=="cpp-native" {$1=commit; print}' "$prev" >> "$FRONTEND_CSV"
  else
    echo "NO_PREVIOUS_CPP_FRONTEND_ROWS"
  fi
else
  BUILD_FRONT_CPP="$BASE/build_frontend_cpp_pic"
  rm -rf "$BUILD_FRONT_CPP"
  cmake -S "$FRONT" -B "$BUILD_FRONT_CPP" \
    -DKokkos_ROOT="$KROOT" -DADC_SRC="$SRC" -DCMAKE_BUILD_TYPE=Release \
    > "$LOGS/frontend_cpp_pic_cfg_${SLURM_JOBID}.log" 2>&1
  rc=$?
  echo "CONFIG frontend_cpp_pic rc=$rc"
  [ $rc -eq 0 ] || { tail -120 "$LOGS/frontend_cpp_pic_cfg_${SLURM_JOBID}.log"; exit $rc; }
  cmake --build "$BUILD_FRONT_CPP" -j 16 > "$LOGS/frontend_cpp_pic_build_${SLURM_JOBID}.log" 2>&1
  rc=$?
  echo "BUILD frontend_cpp_pic rc=$rc"
  [ $rc -eq 0 ] || { tail -120 "$LOGS/frontend_cpp_pic_build_${SLURM_JOBID}.log"; exit $rc; }

  for thr in 1 4 8; do
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/frontend_cpp_pic_thr${thr}_${SLURM_JOBID}.log"
    "$BUILD_FRONT_CPP/frontend_cpp" 128 40 5 1e-4 > "$log" 2>&1
    python3 - "$log" "$FRONTEND_CSV" "$POPS_CPP_COMMIT" "$thr" <<'PY'
import csv, re, sys
log, out, commit, thr = sys.argv[1:]
text = open(log, errors="replace").read()
m = re.search(r"advance_ms=([-\d.eE+]+) extract_ms=([-\d.eE+]+) total_ms=([-\d.eE+]+)", text)
if not m:
    print("FRONTEND_CPP_PARSE_FAIL", log)
    sys.exit(5)
with open(out, "a", newline="") as fh:
    csv.writer(fh).writerow([commit, "frontend-euler-periodic", "cpp-native", "kokkos-openmp-pic", 128, 40, "1e-4", "n/a", "", 0, m.group(1), "n/a", m.group(2), m.group(3), 1.0, "threads=%s" % thr])
PY
  done
fi

BUILD_PY="$BASE/build_python_kokkos_pic"
rm -rf "$BUILD_PY"
cmake -S "$SRC" -B "$BUILD_PY" \
  -DADC_BUILD_PYTHON=ON -DADC_BUILD_TESTS=OFF \
  -DADC_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
  -DPython_EXECUTABLE="$(which python3)" \
  -DCMAKE_BUILD_TYPE=Release \
  > "$LOGS/python_pic_cfg_${SLURM_JOBID}.log" 2>&1
rc=$?
echo "CONFIG python_pic rc=$rc"
[ $rc -eq 0 ] || { tail -120 "$LOGS/python_pic_cfg_${SLURM_JOBID}.log"; exit $rc; }
cmake --build "$BUILD_PY" --target _pops -j 16 > "$LOGS/python_pic_build_${SLURM_JOBID}.log" 2>&1
rc=$?
echo "BUILD python_pic rc=$rc"
[ $rc -eq 0 ] || { tail -160 "$LOGS/python_pic_build_${SLURM_JOBID}.log"; exit $rc; }

export PYTHONPATH="$BUILD_PY/python:$SRC/python:${PYTHONPATH:-}"
export POPS_INCLUDE="$SRC/include"
export POPS_KOKKOS_ROOT="$KROOT"

for thr in 1 4 8; do
  export OMP_NUM_THREADS="$thr"
  export KOKKOS_NUM_THREADS="$thr"
  log="$LOGS/frontend_python_pic_thr${thr}_${SLURM_JOBID}.log"
  python3 "$FRONT/frontend_bench.py" "$FRONTEND_CSV" "$POPS_CPP_COMMIT" "$thr" > "$log" 2>&1
  rc=$?
  echo "PY_FRONTEND_PIC thr=$thr rc=$rc log=$log"
  [ $rc -eq 0 ] || tail -160 "$log"
done

echo "=== CPU FRONTENDS DONE ==="
echo "FRONTEND_CSV=$FRONTEND_CSV"

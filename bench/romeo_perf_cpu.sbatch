#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=pops_perf_cpu
#SBATCH --output=/home/rmdraux/pops_perf_20260608/logs/cpu_%j.out
#SBATCH --error=/home/rmdraux/pops_perf_20260608/logs/cpu_%j.err

set -u
BASE="$HOME/pops_perf_20260608"
SRC="$BASE/adc_cpp"
CASES="$BASE/adc_cases"
RESULTS="$BASE/results"
LOGS="$BASE/logs"
mkdir -p "$RESULTS" "$LOGS"

echo "=== PoPS CPU PERF job=$SLURM_JOBID node=$(hostname) arch=$(uname -m) date=$(date) ==="

romeo_load_x64cpu_env >/dev/null 2>&1
eval "$(spack load --sh /fmrcj5j 2>/dev/null)"   # gcc 14.2.0
eval "$(spack load --sh /kvj6wp2 2>/dev/null)"   # cmake 3.30.5
eval "$(spack load --sh /lni3xp2 2>/dev/null)"   # openmpi 5.0.5
eval "$(spack load --sh /43yg4qb 2>/dev/null)"   # libmd for cmake on compute nodes
eval "$(spack load --sh /2celb2j 2>/dev/null)"   # python 3.11.9 gcc14
eval "$(spack load --sh /rkvb73h 2>/dev/null)"   # numpy 1.26.4 gcc14
eval "$(spack load --sh /j4cl5xe 2>/dev/null)"   # pybind11 2.13.5
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

echo "--- toolchain ---"
which gcc g++ cmake python3 mpicxx mpirun || true
gcc --version | head -1 || true
g++ --version | head -1 || true
cmake --version | head -1 || true
python3 --version || true
mpicxx --version 2>&1 | head -1 || true
mpirun --version 2>&1 | head -1 || true

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

KROOT="$HOME/pops_cpu_mpiomp/kinstall_omp"
if [ ! -f "$KROOT/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KROOT/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
  echo "MISSING_KOKKOS_OPENMP $KROOT"
  exit 2
fi

N_STRONG=256
N_WEAK_LOCAL=128
STEPS=12
WARMUP=3
LIMITER=minmod
SOLVER=geometric_mg

PHASE_CSV="$RESULTS/perf_phases_cpu_${POPS_CPP_COMMIT}_${SLURM_JOBID}.csv"
SCALING_CSV="$RESULTS/perf_scaling_cpu_${POPS_CPP_COMMIT}_${SLURM_JOBID}.csv"
FRONTEND_CSV="$RESULTS/perf_frontends_cpu_${POPS_CPP_COMMIT}_${SLURM_JOBID}.csv"
printf "commit,case,backend,np,threads,n,steps,phase,total_s,per_step_ms,pct\n" > "$PHASE_CSV"
printf "commit,repo,case,scaling,backend,machine,np,threads,gpus,n_global,n_local,steps,warmup,wall_s,per_step_ms,speedup,efficiency,notes\n" > "$SCALING_CSV"
printf "commit,case,frontend,backend,n,steps,dt,cache_state,setup_ms,compile_ms,advance_ms,step_loop_ms,extract_ms,total_ms,ratio_vs_cpp,notes\n" > "$FRONTEND_CSV"

parse_profile_log () {
  local log="$1" case="$2" scaling="$3" backend="$4" np="$5" threads="$6" n="$7" notes="$8"
  python3 - "$log" "$PHASE_CSV" "$SCALING_CSV" "$POPS_CPP_COMMIT" "$case" "$scaling" "$backend" "$np" "$threads" "$n" "$STEPS" "$WARMUP" "$notes" <<'PY'
import csv, re, sys
log, phase_csv, scaling_csv, commit, case, scaling, backend, np, threads, n, steps, warmup, notes = sys.argv[1:]
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
        wr.writerow([commit, case, backend, np, threads, n, steps, phase, total_s, step_ms, pct])
total = next((r for r in rows if r[0] == "TOTAL"), None)
if total:
    if not per_step:
        per_step = total[2]
    if not wall:
        wall = total[1]
    with open(scaling_csv, "a", newline="") as fh:
        csv.writer(fh).writerow([commit, "adc_cpp", case, scaling, backend, "romeo-x64cpu",
                                 np, threads, 0, n, "", steps, warmup, wall, per_step, "", "", notes])
PY
}

configure_bench () {
  local build="$1"; shift
  rm -rf "$build"
  cmake -S "$SRC" -B "$build" "$@" \
    -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
    -DPOPS_BUILD_BENCH=ON -DPOPS_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    > "$LOGS/$(basename "$build")_cfg.log" 2>&1
  local rc=$?
  echo "CONFIG $(basename "$build") rc=$rc"
  [ $rc -eq 0 ] || { tail -80 "$LOGS/$(basename "$build")_cfg.log"; return $rc; }
  cmake --build "$build" --target profile_step -j 16 > "$LOGS/$(basename "$build")_build.log" 2>&1
  rc=$?
  echo "BUILD $(basename "$build") rc=$rc"
  [ $rc -eq 0 ] || tail -80 "$LOGS/$(basename "$build")_build.log"
  return $rc
}

BUILD_KOKKOS="$BASE/build_cpu_kokkos"
if configure_bench "$BUILD_KOKKOS"; then
  for thr in 1 2 4 8; do
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/profile_cpu_strong_kokkos-openmp_np1_thr${thr}_n${N_STRONG}.log"
    echo "RUN kokkos-openmp strong threads=$thr n=$N_STRONG"
    timeout 600 "$BUILD_KOKKOS/bin/profile_step" --n "$N_STRONG" --steps "$STEPS" --warmup "$WARMUP" --solver "$SOLVER" --limiter "$LIMITER" > "$log" 2>&1
    echo "RC=$? log=$log"
    parse_profile_log "$log" "profile-step" "strong" "kokkos-openmp" 1 "$thr" "$N_STRONG" "single-rank-kokkos-openmp"
  done
fi

BUILD_MPI="$BASE/build_cpu_mpi_kokkos"
if configure_bench "$BUILD_MPI" -DPOPS_USE_MPI=ON -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc; then
  for np in 1 2 4 8; do
    thr=4
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/profile_cpu_strong_mpi-kokkos-openmp_np${np}_thr${thr}_n${N_STRONG}.log"
    echo "RUN mpi+kokkos-openmp strong np=$np threads=$thr n=$N_STRONG"
    timeout 600 srun -n "$np" --cpus-per-task="$thr" --cpu-bind=none "$BUILD_MPI/bin/profile_step" --n "$N_STRONG" --steps "$STEPS" --warmup "$WARMUP" --solver "$SOLVER" --limiter "$LIMITER" > "$log" 2>&1
    echo "RC=$? log=$log"
    parse_profile_log "$log" "profile-step" "strong" "mpi+kokkos-openmp" "$np" "$thr" "$N_STRONG" "one-node"
  done
  for np in 1 4; do
    thr=4
    n=$((N_WEAK_LOCAL * np))
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/profile_cpu_weak_mpi-kokkos-openmp_np${np}_thr${thr}_n${n}.log"
    echo "RUN mpi+kokkos-openmp weak np=$np threads=$thr n=$n"
    timeout 600 srun -n "$np" --cpus-per-task="$thr" --cpu-bind=none "$BUILD_MPI/bin/profile_step" --n "$n" --steps "$STEPS" --warmup "$WARMUP" --solver "$SOLVER" --limiter "$LIMITER" > "$log" 2>&1
    echo "RC=$? log=$log"
    parse_profile_log "$log" "profile-step" "weak" "mpi+kokkos-openmp" "$np" "$thr" "$n" "weak-local-n${N_WEAK_LOCAL}-per-rank-axis"
  done
fi

FRONT="$BASE/frontend_cpu"
mkdir -p "$FRONT"
cat > "$FRONT/frontend_bench.cpp" <<'CPP'
#include <pops/runtime/model_spec.hpp>
#include <pops/runtime/system.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
using namespace pops;
static std::vector<double> initial_state(int n, double gamma) {
  const double pi = 3.14159265358979323846;
  std::vector<double> U(static_cast<std::size_t>(4) * n * n, 0.0);
  for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
    const double x = (i + 0.5) / n;
    const double y = (j + 0.5) / n;
    const double rho = 1.0 + 0.1 * std::sin(2*pi*x) * std::cos(2*pi*y);
    const double u = 0.2 + 0.05 * std::sin(2*pi*y);
    const double v = -0.1 + 0.05 * std::cos(2*pi*x);
    const double p = 1.0;
    const std::size_t q = static_cast<std::size_t>(j) * n + i;
    U[q] = rho;
    U[static_cast<std::size_t>(n)*n + q] = rho * u;
    U[static_cast<std::size_t>(2)*n*n + q] = rho * v;
    U[static_cast<std::size_t>(3)*n*n + q] = p/(gamma-1.0) + 0.5*rho*(u*u+v*v);
  }
  return U;
}
int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#endif
  const int n = argc > 1 ? std::atoi(argv[1]) : 128;
  const int steps = argc > 2 ? std::atoi(argv[2]) : 40;
  const int warmup = argc > 3 ? std::atoi(argv[3]) : 5;
  const double dt = argc > 4 ? std::atof(argv[4]) : 1e-4;
  const double gamma = 1.4;
  double advance_ms = 0.0, extract_ms = 0.0, mass = 0.0;
  {
    SystemConfig cfg;
    cfg.n = n;
    cfg.L = 1.0;
    cfg.periodic = true;
    System sim(cfg);
    ModelSpec spec;
    spec.transport = "compressible";
    spec.source = "none";
    spec.elliptic = "background";
    spec.gamma = gamma;
    spec.alpha = 0.0;
    spec.n0 = 0.0;
    sim.add_block("gas", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
    sim.set_poisson("charge_density", "fft");
    sim.set_state("gas", initial_state(n, gamma));
    for (int s = 0; s < warmup; ++s) sim.step(dt);
#if defined(POPS_HAS_KOKKOS)
    Kokkos::fence();
#endif
    const auto t0 = std::chrono::steady_clock::now();
    sim.advance(dt, steps);
#if defined(POPS_HAS_KOKKOS)
    Kokkos::fence();
#endif
    const auto t1 = std::chrono::steady_clock::now();
    advance_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto e0 = std::chrono::steady_clock::now();
    const auto U = sim.get_state("gas");
    const auto e1 = std::chrono::steady_clock::now();
    extract_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();
    for (std::size_t q = 0; q < static_cast<std::size_t>(n) * n; ++q) mass += U[q];
  }
  std::printf("FRONTEND_CPP n=%d steps=%d warmup=%d dt=%.17g advance_ms=%.6f extract_ms=%.6f total_ms=%.6f mass=%.17e\n",
              n, steps, warmup, dt, advance_ms, extract_ms, advance_ms + extract_ms, mass);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return 0;
}
CPP
cat > "$FRONT/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(pops_frontend_cpp CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Kokkos REQUIRED)
add_executable(frontend_cpp frontend_bench.cpp ${POPS_SRC}/python/system.cpp)
target_include_directories(frontend_cpp PRIVATE ${POPS_SRC}/include)
target_compile_definitions(frontend_cpp PRIVATE POPS_HAS_KOKKOS)
target_link_libraries(frontend_cpp PRIVATE Kokkos::kokkos)
CMAKE

BUILD_FRONT_CPP="$BASE/build_frontend_cpp"
rm -rf "$BUILD_FRONT_CPP"
if cmake -S "$FRONT" -B "$BUILD_FRONT_CPP" -DKokkos_ROOT="$KROOT" -DPOPS_SRC="$SRC" -DCMAKE_BUILD_TYPE=Release > "$LOGS/frontend_cpp_cfg.log" 2>&1 && \
   cmake --build "$BUILD_FRONT_CPP" -j 16 > "$LOGS/frontend_cpp_build.log" 2>&1; then
  for thr in 1 4 8; do
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/frontend_cpp_thr${thr}.log"
    "$BUILD_FRONT_CPP/frontend_cpp" 128 40 5 1e-4 > "$log" 2>&1
    python3 - "$log" "$FRONTEND_CSV" "$POPS_CPP_COMMIT" "$thr" <<'PY'
import csv, re, sys
log, out, commit, thr = sys.argv[1:]
text = open(log, errors="replace").read()
m = re.search(r"advance_ms=([-\d.eE+]+) extract_ms=([-\d.eE+]+) total_ms=([-\d.eE+]+)", text)
if m:
    with open(out, "a", newline="") as fh:
        csv.writer(fh).writerow([commit, "frontend-euler-periodic", "cpp-native", "kokkos-openmp", 128, 40, "1e-4", "n/a", "", 0, m.group(1), "n/a", m.group(2), m.group(3), 1.0, "threads=%s" % thr])
else:
    print("FRONTEND_CPP_PARSE_FAIL", log)
PY
  done
else
  echo "FRONTEND_CPP_BUILD_FAIL"
  tail -80 "$LOGS/frontend_cpp_cfg.log" "$LOGS/frontend_cpp_build.log" 2>/dev/null || true
fi

BUILD_PY="$BASE/build_python_kokkos"
rm -rf "$BUILD_PY"
if cmake -S "$SRC" -B "$BUILD_PY" \
    -DPOPS_BUILD_PYTHON=ON -DPOPS_BUILD_TESTS=OFF -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT="$KROOT" \
    -DPython_EXECUTABLE="$(which python3)" -DCMAKE_BUILD_TYPE=Release > "$LOGS/python_cfg.log" 2>&1 && \
   cmake --build "$BUILD_PY" --target _pops -j 16 > "$LOGS/python_build.log" 2>&1; then
  cat > "$FRONT/frontend_bench.py" <<'PY'
import csv, os, shutil, sys, tempfile, time
import numpy as np
import pops
from pops import dsl

GAMMA = 1.4

def initial_state(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x)
    rho = 1.0 + 0.1 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    u = 0.2 + 0.05 * np.sin(2*np.pi*Y)
    v = -0.1 + 0.05 * np.cos(2*np.pi*X)
    p = np.ones_like(rho)
    U = np.zeros((4, n, n))
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * v
    U[3] = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v)
    return U

def dsl_model():
    m = dsl.Model("frontend_euler_periodic")
    rho, rhou, rhov, E = m.conservative_vars("rho", "rho_u", "rho_v", "E",
                                             roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", GAMMA)
    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
    H = (E + p) / rho
    c = dsl.sqrt(g * p / rho)
    m.flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u],
           y=[rhov, rhov*u, rhov*v + p, rho*H*v])
    m.eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([prho, prho*pu, prho*pv, pp/(g-1.0) + 0.5*prho*(pu*pu + pv*pv)])
    m.elliptic_rhs(0.0 * rho)
    m.check()
    return m

def build_bricks(n, U):
    t0 = time.perf_counter()
    sim = pops.System(n=n, L=1.0, periodic=True)
    spec = pops.Model(state=pops.FluidState("compressible", gamma=GAMMA),
                     transport=pops.CompressibleFlux(),
                     source=pops.NoSource(),
                     elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0))
    sim.add_equation("gas", spec, spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"))
    sim.set_poisson(rhs="charge_density", solver="fft")
    sim.set_state("gas", U.reshape(-1))
    return sim, (time.perf_counter() - t0) * 1000.0

def build_dsl(n, U, cache_dir, cache_state):
    old = os.environ.get("POPS_CACHE_DIR")
    os.environ["POPS_CACHE_DIR"] = cache_dir
    try:
        m = dsl_model()
        c0 = time.perf_counter()
        cm = m.compile(backend="production")
        compile_ms = (time.perf_counter() - c0) * 1000.0
        if cm.backend != "production" or cm.adder != "add_native_block":
            raise RuntimeError("bad DSL backend/adderr: %s %s" % (cm.backend, cm.adder))
        t0 = time.perf_counter()
        sim = pops.System(n=n, L=1.0, periodic=True)
        sim.add_equation("gas", cm, spatial=pops.FiniteVolume(limiter="minmod", riemann="rusanov"))
        sim.set_poisson(rhs="charge_density", solver="fft")
        sim.set_state("gas", U.reshape(-1))
        setup_ms = (time.perf_counter() - t0) * 1000.0
        return sim, setup_ms, compile_ms
    finally:
        if old is None:
            os.environ.pop("POPS_CACHE_DIR", None)
        else:
            os.environ["POPS_CACHE_DIR"] = old

def time_sim(sim, dt, steps, warmup):
    for _ in range(warmup):
        sim.step(dt)
    t0 = time.perf_counter()
    sim.advance(dt, steps)
    advance_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    U = np.asarray(sim.get_state("gas"))
    extract_ms = (time.perf_counter() - t1) * 1000.0
    return advance_ms, extract_ms, float(U[: U.size // 4].sum())

def time_step_loop(build_fn, dt, steps, warmup):
    sim, setup_ms, compile_ms = build_fn()
    for _ in range(warmup):
        sim.step(dt)
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step(dt)
    return (time.perf_counter() - t0) * 1000.0

def write_row(path, commit, frontend, cache_state, setup_ms, compile_ms, advance_ms, step_loop_ms, extract_ms, notes):
    total = setup_ms + compile_ms + advance_ms + extract_ms
    with open(path, "a", newline="") as fh:
        csv.writer(fh).writerow([commit, "frontend-euler-periodic", frontend, "kokkos-openmp", 128, 40, "1e-4", cache_state,
                                 setup_ms, compile_ms, advance_ms, step_loop_ms, extract_ms, total, "", notes])

def main():
    out, commit, thr = sys.argv[1], sys.argv[2], sys.argv[3]
    n, steps, warmup, dt = 128, 40, 5, 1e-4
    U = initial_state(n)
    sim, setup_ms = build_bricks(n, U)
    adv, ext, mass = time_sim(sim, dt, steps, warmup)
    loop_ms = time_step_loop(lambda: (*build_bricks(n, U), 0.0), dt, steps, warmup)
    write_row(out, commit, "python-bricks", "n/a", setup_ms, 0.0, adv, loop_ms, ext, "threads=%s mass=%.17e" % (thr, mass))

    cache = tempfile.mkdtemp(prefix="pops_dsl_cache_")
    try:
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache, exist_ok=True)
        sim, setup_ms, compile_ms = build_dsl(n, U, cache, "cold")
        adv, ext, mass = time_sim(sim, dt, steps, warmup)
        loop_ms = time_step_loop(lambda: build_dsl(n, U, cache, "cold-loop"), dt, steps, warmup)
        write_row(out, commit, "python-dsl-production", "cold", setup_ms, compile_ms, adv, loop_ms, ext, "threads=%s mass=%.17e" % (thr, mass))

        sim, setup_ms, compile_ms = build_dsl(n, U, cache, "warm")
        adv, ext, mass = time_sim(sim, dt, steps, warmup)
        loop_ms = time_step_loop(lambda: build_dsl(n, U, cache, "warm-loop"), dt, steps, warmup)
        write_row(out, commit, "python-dsl-production", "warm", setup_ms, compile_ms, adv, loop_ms, ext, "threads=%s mass=%.17e" % (thr, mass))
    finally:
        shutil.rmtree(cache, ignore_errors=True)

if __name__ == "__main__":
    main()
PY
  export PYTHONPATH="$BUILD_PY/python"
  export POPS_INCLUDE="$SRC/include"
  for thr in 1 4 8; do
    export OMP_NUM_THREADS="$thr"
    export KOKKOS_NUM_THREADS="$thr"
    log="$LOGS/frontend_python_thr${thr}.log"
    python3 "$FRONT/frontend_bench.py" "$FRONTEND_CSV" "$POPS_CPP_COMMIT" "$thr" > "$log" 2>&1
    echo "PY_FRONTEND thr=$thr rc=$? log=$log"
  done
else
  echo "PYTHON_BUILD_FAIL"
  tail -120 "$LOGS/python_cfg.log" "$LOGS/python_build.log" 2>/dev/null || true
fi

echo "=== CPU PERF DONE ==="
echo "PHASE_CSV=$PHASE_CSV"
echo "SCALING_CSV=$SCALING_CSV"
echo "FRONTEND_CSV=$FRONTEND_CSV"

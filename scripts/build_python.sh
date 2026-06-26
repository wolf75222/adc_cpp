#!/usr/bin/env bash
# One command to build + install the Python module `_pops` for END USERS, applying the build knobs that
# scripts/setup_env.sh only *recommends*. Run `bash scripts/setup_env.sh` ONCE first (it creates the
# `adc` env and pins the per-platform toolchain); then this script, on every (re)build:
#
#   - activates the conda env `adc` (override: POPS_ENV_NAME), without tripping `set -u`;
#   - sizes the heavy-TU Ninja pool (POPS_HEAVY_TU_POOL) from cores AND free RAM so the split module TUs
#     compile in PARALLEL without OOM (each -O3 leaf peaks at several GB; the CMake default is 1, the
#     CI out-of-memory guard). Pre-set POPS_HEAVY_TU_POOL to pin it by hand.
#   - exports the Kokkos / CMake discovery vars (Kokkos_ROOT, POPS_KOKKOS_ROOT, CMAKE_PREFIX_PATH) and a
#     STABLE, cross-worktree ccache (CCACHE_DIR + CCACHE_BASEDIR -> a file already compiled in another
#     worktree is reused instead of recompiled);
#   - runs `pip install . --no-build-isolation` so the build reuses the env's pinned
#     scikit-build-core / pybind11 (the SAME stack as the toolchain) instead of a fresh pip build env;
#   - ends on `import adc; pops.doctor()`.
#
#   bash scripts/build_python.sh            # build + install into the active env
#   bash scripts/build_python.sh --clean    # drop the scikit-build wheel cache first
#   bash scripts/build_python.sh --fresh    # --clean AND `ccache -C`: a true COLD compile (measuring)
#   bash scripts/build_python.sh --mpi      # also build the distributed MPI backend (POPS_USE_MPI=ON)
#   POPS_HEAVY_TU_POOL=4 bash scripts/build_python.sh    # pin the pool by hand (skip auto-sizing)
#   bash scripts/build_python.sh -- -e      # pass extra args through to pip (here: editable install)
#
# NOT `set -u`: `conda activate` references unset variables in its own shell hook.
set -eo pipefail

ENV_NAME="${POPS_ENV_NAME:-adc}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# --- arguments --------------------------------------------------------------------------------------
DO_CLEAN=0
DO_FRESH=0
WITH_MPI=0
EXTRA_PIP=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) DO_CLEAN=1 ;;
    --fresh) DO_CLEAN=1; DO_FRESH=1 ;;
    --mpi)   WITH_MPI=1 ;;
    -h|--help)
      sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    --) shift; EXTRA_PIP=("$@"); break ;;
    *) echo "unknown argument: $1 (use --clean | --fresh | --mpi | --help, or -- <pip args>)" >&2
       exit 2 ;;
  esac
  shift
done

# --- conda present + env active ----------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Run 'bash scripts/setup_env.sh' first (it bootstraps the env and toolchain)." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' is missing. Create it first: bash scripts/setup_env.sh" >&2
  exit 1
fi
conda activate "$ENV_NAME"
echo "--- env '$ENV_NAME' active (CONDA_PREFIX=$CONDA_PREFIX) ---"

# --- heavy-TU pool: cores capped by RAM (each -O3 leaf peaks ~3-4 GB) --------------------------------
ncpu="$( (nproc 2>/dev/null) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
if [[ "$(uname)" == "Darwin" ]]; then
  mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
else
  mem_kb="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  mem_bytes=$(( mem_kb * 1024 ))
fi
mem_gb=$(( mem_bytes / 1024 / 1024 / 1024 ))
if [[ -n "${POPS_HEAVY_TU_POOL:-}" ]]; then
  pool="$POPS_HEAVY_TU_POOL"
  echo "heavy-TU pool: $pool (from POPS_HEAVY_TU_POOL)"
else
  ram_cap=$(( mem_gb / 4 )); [[ $ram_cap -lt 1 ]] && ram_cap=1
  pool=$ncpu; [[ $pool -gt $ram_cap ]] && pool=$ram_cap
  echo "heavy-TU pool: $pool (min of ${ncpu} cores and ${ram_cap} = ${mem_gb}GB/4; export POPS_HEAVY_TU_POOL to override)"
fi

# --- discovery vars + stable cross-worktree ccache --------------------------------------------------
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export Kokkos_ROOT="${Kokkos_ROOT:-$CONDA_PREFIX}"
export POPS_KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-$CONDA_PREFIX}"
# A ccache shared by every checkout, with absolute paths rewritten relative to the repo root that owns
# all worktrees (git common dir's parent), so the same TU compiled in another worktree is a cache hit.
main_root="$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
main_root="${main_root%/.git}"
[[ -n "$main_root" && -d "$main_root" ]] || main_root="$HERE"
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/adc-ccache}"
export CCACHE_BASEDIR="${CCACHE_BASEDIR:-$main_root}"
echo "ccache: dir=$CCACHE_DIR basedir=$CCACHE_BASEDIR"

# --- clean / fresh ----------------------------------------------------------------------------------
if [[ $DO_CLEAN -eq 1 ]]; then
  # scikit-build-core caches under build/<wheel_tag>/ (e.g. build/cp312-cp312-macosx_14_0_arm64). Remove
  # ONLY those tag dirs, never the C++ preset build/ root (its CMakeCache.txt sits at build/).
  shopt -s nullglob
  removed=0
  for d in "$HERE"/build/cp3*/; do rm -rf "$d"; removed=1; done
  [[ $removed -eq 1 ]] && echo "--clean: removed scikit-build wheel cache (build/cp3*/)" \
                       || echo "--clean: no scikit-build wheel cache to remove"
fi
if [[ $DO_FRESH -eq 1 ]]; then
  if command -v ccache >/dev/null 2>&1; then
    ccache -C >/dev/null && echo "--fresh: ccache cleared (cold build)" \
                         || echo "--fresh: ccache -C failed; build may not be fully cold" >&2
  fi
fi

# --- build + install --------------------------------------------------------------------------------
[[ $WITH_MPI -eq 1 ]] && { export POPS_USE_MPI=ON; echo "MPI backend: ON"; }
cd "$HERE"
pip_args=(install -v . -C cmake.define.POPS_HEAVY_TU_POOL="$pool")
if python -c "import scikit_build_core, pybind11" >/dev/null 2>&1; then
  pip_args=(install -v . --no-build-isolation -C cmake.define.POPS_HEAVY_TU_POOL="$pool")
else
  echo "note: scikit-build-core/pybind11 not in '$ENV_NAME'; using pip build isolation"
  echo "      (slower, unpinned build deps). Add 'scikit-build-core' to environment.yml + 'conda env update' to fix."
fi
echo "--- python -m pip ${pip_args[*]} ${EXTRA_PIP[*]} ---"
python -m pip "${pip_args[@]}" "${EXTRA_PIP[@]}"

# --- diagnose ---------------------------------------------------------------------------------------
echo ""
echo "--- import adc; pops.doctor() ---"
python -c "import adc; print('adc', pops.__version__); pops.doctor()"

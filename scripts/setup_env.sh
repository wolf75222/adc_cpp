#!/usr/bin/env bash
# Create (or update) the conda env `adc` AND pin the best toolchain for the platform in it.
#
#   bash scripts/setup_env.sh            # CPU install (default)
#   bash scripts/setup_env.sh --cuda     # allow the CUDA Kokkos variant (NVIDIA host)
#   conda activate adc
#   pip install . -v
#
# Why this script rather than `conda env create` alone: environment.yml cannot make a PER-PLATFORM
# choice, yet the right compiler is not the same everywhere --
#   - macOS  : AppleClang (Xcode CLT). Measured on the module's translation units: a vanilla LLVM clang
#     (Homebrew, and very likely the conda one, same family) compiles system.cpp >15x slower
#     (>1h24 vs 5min21). So we pin CC/CXX=AppleClang IN the env (`conda env config vars`): every
#     `conda activate adc` exports them, taking priority over a polluted PATH (real case:
#     /opt/homebrew/opt/llvm/bin at the head of PATH via ~/.zshrc).
#   - Linux  : `cxx-compiler` conda-forge (gcc 14.2 via cxx-compiler 1.11.0, C++23) -- full toolchain, no root,
#     installed here as the pinned Linux default; its activation scripts export CC/CXX automatically.
#     This is the fix for the slow `-j40` Linux build: a wrong/floating host gcc, plus the (now split,
#     ADC-335) heavy TUs, made it crawl; a pinned conda gcc + the split restore a fast parallel build.
# Overrides remain possible: CC/CXX set by hand before a build win, and the DSL runtime follows the
# compiler baked into _adc anyway.
#
# It also makes the Linux/Ubuntu user path reliable end to end (cf.
# docs/sphinx/getting-started/installation.md): it bootstraps conda guidance, configures conda-forge to
# survive HTTP 429, forces a CPU Kokkos by default (the bare `kokkos` resolves to the CUDA variant on a
# host with an NVIDIA driver -> `pip install .` then fails "Could not find nvcc"), persists the DSL
# runtime variables in the env, and ends on `adc.doctor()`.
set -euo pipefail

ENV_NAME="${ADC_ENV_NAME:-adc}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# --- arguments --------------------------------------------------------------------------------------
ADC_WITH_CUDA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)  ADC_WITH_CUDA=0 ;;
    --cuda) ADC_WITH_CUDA=1 ;;
    -h|--help)
      sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown argument: $1 (use --cpu | --cuda | --help)" >&2; exit 2 ;;
  esac
  shift
done

# --- conda present? otherwise guide the bootstrap (no silent install) -------------------------------
if ! command -v conda >/dev/null 2>&1; then
  cat >&2 <<EOF
conda not found. On a fresh machine, bootstrap Miniforge (conda-forge), then re-run this script:

    cd /tmp
    curl -L -o Miniforge3.sh \\
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh"
    bash Miniforge3.sh -b -p "\$HOME/miniforge3"
    source "\$HOME/miniforge3/etc/profile.d/conda.sh"
    conda init "\$(basename "\$SHELL")"     # then open a new shell

If Miniforge is already installed, just load it:
    source "\$HOME/miniforge3/etc/profile.d/conda.sh"

Reference: https://conda-forge.org/download/
EOF
  exit 1
fi

# --- conda-forge robustness (survive HTTP 429) ; prefer mamba for the heavy solves ------------------
# These edit ~/.condarc (global) ; they are the conda-forge recommended defaults and harmless.
conda config --set channel_priority strict          >/dev/null 2>&1 || true
conda config --set solver libmamba                   >/dev/null 2>&1 || true
conda config --set remote_max_retries 10             >/dev/null 2>&1 || true
conda config --set remote_backoff_factor 2           >/dev/null 2>&1 || true
conda config --set remote_read_timeout_secs 120      >/dev/null 2>&1 || true
PKG="conda"
command -v mamba >/dev/null 2>&1 && PKG="mamba"

# --- CPU Kokkos by default ---------------------------------------------------------------------------
# The bare `kokkos` resolves to the CUDA variant whenever conda sees a `__cuda` virtual package (NVIDIA
# driver present). CONDA_OVERRIDE_CUDA="" tells conda there is no CUDA, so it picks the CPU builds --
# robust, with no fragile build-string pin. `--cuda` leaves conda's real CUDA detection in place.
if [[ "$ADC_WITH_CUDA" == "0" ]]; then
  export CONDA_OVERRIDE_CUDA=""
  echo "--- CPU install (CONDA_OVERRIDE_CUDA=\"\" forces CPU Kokkos ; pass --cuda for the CUDA variant) ---"
else
  echo "--- CUDA install requested (--cuda): conda may resolve the CUDA Kokkos variant ---"
fi

# --- create / update the env -------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "--- updating env '$ENV_NAME' (environment.yml) ---"
  "$PKG" env update -n "$ENV_NAME" -f "$HERE/environment.yml" --prune
else
  echo "--- creating env '$ENV_NAME' (environment.yml) ---"
  "$PKG" env create -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

# --- safety net: reject a CUDA Kokkos build in CPU mode ----------------------------------------------
if [[ "$ADC_WITH_CUDA" == "0" ]]; then
  kbuild="$(conda list -n "$ENV_NAME" kokkos 2>/dev/null | awk '$1 == "kokkos" {print $3}')"
  if printf '%s' "$kbuild" | grep -qi cuda; then
    echo "ERROR: a CUDA Kokkos build was selected ($kbuild) in CPU mode." >&2
    echo "       pip install . would then fail 'Could not find nvcc'. Re-create with the CPU override:" >&2
    echo "           CONDA_OVERRIDE_CUDA=\"\" conda env update -n $ENV_NAME -f environment.yml --prune" >&2
    echo "       or pass --cuda if you really want the CUDA variant." >&2
    exit 1
  fi
fi

# --- per-platform toolchain --------------------------------------------------------------------------
case "$(uname)" in
  Darwin)
    if ! xcode-select -p >/dev/null 2>&1; then
      echo "Xcode Command Line Tools are missing (AppleClang required):" >&2
      echo "    xcode-select --install" >&2
      exit 1
    fi
    conda env config vars set -n "$ENV_NAME" \
      CC=/usr/bin/clang CXX=/usr/bin/clang++ >/dev/null
    echo "macOS: CC/CXX -> AppleClang pinned in the env (exported on each activation, priority over PATH)."
    ;;
  Linux)
    echo "--- conda C++23 toolchain (gcc) ---"
    "$PKG" install -y -n "$ENV_NAME" -c conda-forge cxx-compiler
    echo "Linux: cxx-compiler installed; CC/CXX exported automatically on activation."
    ;;
  *)
    echo "Platform $(uname): toolchain left to the system (CC/CXX not pinned)."
    ;;
esac

# --- persist the DSL runtime variables in the env (exported on each `conda activate adc`) ------------
# ADC_INCLUDE   : the adc headers for the DSL production/aot backend (here: the source checkout).
# ADC_KOKKOS_ROOT / Kokkos_ROOT : the Kokkos install the DSL .so compiles against (the env Kokkos;
#                 Serial is enough on CPU). Without it, the tutorial dead-ends on "no DSL backend".
# ADC_CACHE_DIR : a stable cache for the compiled DSL .so.
# CMAKE_PREFIX_PATH : point find_package at the env prefix (env Kokkos/pybind11/MPI) even outside an
#                 activated shell or when a system CMAKE_PREFIX_PATH would otherwise win.
ADC_PREFIX="$(conda run -n "$ENV_NAME" printenv CONDA_PREFIX 2>/dev/null || true)"
[ -n "$ADC_PREFIX" ] || ADC_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')"
conda env config vars set -n "$ENV_NAME" \
  ADC_INCLUDE="$HERE/include" \
  ADC_KOKKOS_ROOT="$ADC_PREFIX" \
  Kokkos_ROOT="$ADC_PREFIX" \
  CMAKE_PREFIX_PATH="$ADC_PREFIX" \
  ADC_CACHE_DIR="$HERE/.adc_cache" >/dev/null
mkdir -p "$HERE/.adc_cache"
echo "env vars pinned: ADC_INCLUDE, ADC_KOKKOS_ROOT, Kokkos_ROOT, CMAKE_PREFIX_PATH, ADC_CACHE_DIR (prefix: $ADC_PREFIX)."

# --- final diagnostic --------------------------------------------------------------------------------
echo ""
echo "Env ready. Next, in one command (sizes the heavy-TU pool, exports the discovery vars + ccache,"
echo "installs, then runs adc.doctor()):"
echo "    bash scripts/build_python.sh"
echo ""
echo "Or by hand:"
echo "    conda activate $ENV_NAME"
echo "    pip install . -v          # builds the Kokkos module (Kokkos is ON and mandatory)"
echo ""
# ADC-338: after the ADC-335 split, the heavy module TUs are small but a size-1 Ninja pool
# (ADC_HEAVY_TU_POOL, the CI 7GB-runner OOM guard) still serializes them. On a high-RAM local box,
# widen it so -j actually compiles the sub-TUs in parallel (this, not -j alone, bounds the heavy TUs).
# scripts/build_python.sh sizes this automatically (cores capped by RAM); the manual knob:
_ncpu="$( (nproc 2>/dev/null) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "Manual heavy-TU pool (build_python.sh does this for you):"
echo "    pip install . -v -C cmake.define.ADC_HEAVY_TU_POOL=$_ncpu      # or a C++ preset: -DADC_HEAVY_TU_POOL=$_ncpu"
echo "    (leave it at the default 1 on memory-constrained machines / CI -- it is the OOM guard.)"
echo ""
if conda run -n "$ENV_NAME" python -c "import adc" >/dev/null 2>&1; then
  echo "--- adc.doctor() ---"
  conda run -n "$ENV_NAME" python -c "import adc; adc.doctor()" || true
else
  echo "adc is not installed in '$ENV_NAME' yet. Install it, then check the environment:"
  echo "    conda activate $ENV_NAME"
  echo "    pip install . -v"
  echo "    python -c 'import adc; adc.doctor()'"
fi

#!/usr/bin/env bash
# Cree (ou met a jour) l'env conda `adc` ET y fige la meilleure toolchain de la plateforme.
#
#   bash scripts/setup_env.sh
#   conda activate adc
#   pip install .
#
# Pourquoi ce script plutot que `conda env create` seul : environment.yml ne sait pas faire de
# choix PAR PLATEFORME, or le bon compilateur n'est pas le meme partout --
#   - macOS  : AppleClang (Xcode CLT). Mesure sur les TU du module : un clang LLVM vanilla
#     (Homebrew, et tres probablement celui de conda, meme famille) compile system.cpp >15x
#     plus lentement (>1h24 vs 5min21). On fige donc CC/CXX=AppleClang DANS l'env
#     (`conda env config vars`) : chaque `conda activate adc` les exporte, prioritaires sur un
#     PATH pollue (cas reel : /opt/homebrew/opt/llvm/bin en tete de PATH via ~/.zshrc).
#   - Linux  : `cxx-compiler` conda-forge (gcc 14, C++23) -- toolchain complete sans droits
#     root ; ses scripts d'activation exportent CC/CXX automatiquement.
# Les overrides restent possibles : CC/CXX poses A LA MAIN avant un build l'emportent, et le
# DSL runtime suit de toute facon le compilateur bake dans _adc.
set -euo pipefail

ENV_NAME="${ADC_ENV_NAME:-adc}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

command -v conda >/dev/null 2>&1 || {
  echo "conda introuvable. Installer miniforge/miniconda d'abord : https://conda-forge.org/download/" >&2
  exit 1
}

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "--- mise a jour de l'env '$ENV_NAME' (environment.yml) ---"
  conda env update -n "$ENV_NAME" -f "$HERE/environment.yml" --prune
else
  echo "--- creation de l'env '$ENV_NAME' (environment.yml) ---"
  conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"
fi

case "$(uname)" in
  Darwin)
    if ! xcode-select -p >/dev/null 2>&1; then
      echo "Les Command Line Tools Xcode manquent (AppleClang requis) :" >&2
      echo "    xcode-select --install" >&2
      exit 1
    fi
    conda env config vars set -n "$ENV_NAME" \
      CC=/usr/bin/clang CXX=/usr/bin/clang++ >/dev/null
    echo "macOS : CC/CXX -> AppleClang figes dans l'env (exportes a chaque activation,"
    echo "        prioritaires sur le PATH)."
    ;;
  Linux)
    echo "--- toolchain C++23 conda (gcc) ---"
    conda install -y -n "$ENV_NAME" -c conda-forge cxx-compiler
    echo "Linux : cxx-compiler installe ; CC/CXX exportes automatiquement a l'activation."
    ;;
  *)
    echo "Plateforme $(uname) : toolchain laissee au systeme (CC/CXX non figes)."
    ;;
esac

echo ""
echo "Env pret. Suite :"
echo "    conda activate $ENV_NAME"
echo "    pip install .          # module serie ; ADC_USE_KOKKOS=ON Kokkos_ROOT=\$CONDA_PREFIX pour Kokkos"
echo "    python -c 'import adc; adc.doctor()'"

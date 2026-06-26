#!/usr/bin/env bash
# Compile et installe Kokkos (backends Serial + OpenMP) DANS l'env conda actif.
#
# POURQUOI : le paquet `kokkos` de conda-forge est typiquement compile avec le SEUL backend
# Serial (pas d'OpenMP ; CUDA = paquet a part). Il suffit pour compiler POPS_USE_KOKKOS=ON mais
# ne scale pas en threads. Ce script comble le trou : UNE commande, et l'env conda possede un
# Kokkos OpenMP -- l'outillage (cmake, ninja, libomp) vient deja d'environment.yml.
#
#   conda activate pops
#   bash scripts/kokkos_openmp_conda.sh
#   cmake --preset python-parallel && cmake --build --preset python-parallel
#
# Variables : KOKKOS_VERSION (defaut 4.7.01), KOKKOS_INSTALL_PREFIX (defaut $CONDA_PREFIX),
# KOKKOS_SRC_DIR (defaut $TMPDIR/kokkos-src-<ver>). Compilateur : celui du systeme (le MEME que
# le build d'adc_cpp -- AppleClang sur macOS -- pour rester ABI-coherent avec le module _pops).
# Sur macOS, libomp est pris dans l'env conda (paquet llvm-openmp) via les hints CMake standard,
# exactement comme le CMakeLists d'adc_cpp le fait pour POPS_USE_OPENMP.
set -euo pipefail

: "${CONDA_PREFIX:?Activez d'abord l'env conda : conda activate pops}"
VER="${KOKKOS_VERSION:-4.7.01}"
PREFIX="${KOKKOS_INSTALL_PREFIX:-$CONDA_PREFIX}"
SRC="${KOKKOS_SRC_DIR:-${TMPDIR:-/tmp}/kokkos-src-$VER}"

# Le paquet conda `kokkos` (Serial) dans le meme prefix serait masque/ecrase par cette install :
# on previent (sans bloquer -- l'install locale est plus capable, Serial+OpenMP).
if [ -f "$PREFIX/include/Kokkos_Core.hpp" ]; then
  echo "NOTE : un Kokkos existe deja dans $PREFIX (probablement le paquet conda Serial-only)."
  echo "       Cette installation (Serial+OpenMP) va l'ecraser dans ce prefix. Pour garder les"
  echo "       deux : KOKKOS_INSTALL_PREFIX=\$HOME/opt/kokkos-omp bash $0"
fi

if [ ! -d "$SRC" ]; then
  echo "--- clone kokkos $VER (shallow) ---"
  git clone --depth 1 --branch "$VER" https://github.com/kokkos/kokkos.git "$SRC"
fi

EXTRA=()
if [ "$(uname)" = "Darwin" ] && [ -f "$CONDA_PREFIX/lib/libomp.dylib" ]; then
  # AppleClang ne trouve pas libomp seul : hints standard vers le libomp de l'env conda.
  EXTRA+=(
    "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I$CONDA_PREFIX/include"
    "-DOpenMP_CXX_LIB_NAMES=omp"
    "-DOpenMP_omp_LIBRARY=$CONDA_PREFIX/lib/libomp.dylib"
  )
fi

echo "--- configure (Serial + OpenMP, C++20 comme adc_cpp sous Kokkos) ---"
cmake -S "$SRC" -B "$SRC/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_CXX_STANDARD=20 \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  "${EXTRA[@]}"
echo "--- build + install ---"
cmake --build "$SRC/build" -j
cmake --install "$SRC/build"

echo ""
echo "OK : Kokkos $VER (Serial + OpenMP) installe dans $PREFIX"
echo "Builds adc_cpp : cmake --preset python-parallel  (Kokkos_ROOT=\$CONDA_PREFIX deja cable)"
[ "$PREFIX" != "$CONDA_PREFIX" ] && echo "  (prefix custom : passer -DKokkos_ROOT=$PREFIX)"
echo "DSL backend production : export POPS_KOKKOS_ROOT=$PREFIX (parite loader/module)"

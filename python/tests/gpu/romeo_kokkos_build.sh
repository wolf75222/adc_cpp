#!/bin/bash -l
# Build Kokkos (CUDA + Serial, Hopper90) puis le harnais utilisant la brique generee, et execute
# sur le GPU. Tout sur le noeud GPU (aarch64) ou nvcc s'execute.
module load cuda/12.6
cd "$HOME/pops_dsl_kk" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
NW="$PWD/kokkos/bin/nvcc_wrapper"

# 1. Kokkos (une fois)
if ! ls kinstall/lib*/cmake/Kokkos/KokkosConfig.cmake >/dev/null 2>&1; then
  echo "== configure Kokkos =="
  cmake -S kokkos -B kbuild \
    -DCMAKE_CXX_COMPILER="$NW" \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON -DKokkos_ENABLE_SERIAL=ON \
    -DCMAKE_INSTALL_PREFIX="$PWD/kinstall" -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_BUILD_TYPE=Release > kokkos_cfg.log 2>&1 || { echo KOKKOS_CFG_FAIL; tail -25 kokkos_cfg.log; exit 1; }
  echo "== build + install Kokkos (peut durer quelques minutes) =="
  cmake --build kbuild -j 16 --target install > kokkos_build.log 2>&1 \
    || { echo KOKKOS_BUILD_FAIL; tail -25 kokkos_build.log; exit 1; }
fi
echo KOKKOS_OK

# 2. harnais (brique generee + Kokkos)
NWI="$(ls -d "$PWD"/kinstall/bin/nvcc_wrapper 2>/dev/null || echo "$NW")"
cmake -S harness -B hbuild \
  -DCMAKE_CXX_COMPILER="$NWI" \
  -DKokkos_ROOT="$PWD/kinstall" -DPOPS_INCLUDE="$PWD/include" \
  -DCMAKE_BUILD_TYPE=Release > harness_cfg.log 2>&1 || { echo HARNESS_CFG_FAIL; tail -30 harness_cfg.log; exit 1; }
cmake --build hbuild -j 8 > harness_build.log 2>&1 || { echo HARNESS_BUILD_FAIL; tail -30 harness_build.log; exit 1; }
echo HARNESS_OK

# 3. run sur GPU
./hbuild/kkbrick
echo "exit=$?"

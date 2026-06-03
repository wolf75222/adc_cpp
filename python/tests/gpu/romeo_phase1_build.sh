#!/bin/bash -l
module load cuda/12.6
cd "$HOME/adc_gpu_p1" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
NW="$PWD/kokkos/bin/nvcc_wrapper"
if ! ls kinstall/lib*/cmake/Kokkos/KokkosConfig.cmake >/dev/null 2>&1; then
  echo "== build Kokkos (CUDA+Serial, Hopper90) =="
  cmake -S kokkos -B kbuild -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_HOPPER90=ON -DKokkos_ENABLE_SERIAL=ON -DCMAKE_INSTALL_PREFIX="$PWD/kinstall" \
    -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release > kcfg.log 2>&1 || { echo KCFG_FAIL; tail -20 kcfg.log; exit 1; }
  cmake --build kbuild -j 16 --target install > kbuild.log 2>&1 || { echo KBUILD_FAIL; tail -25 kbuild.log; exit 1; }
fi
echo KOKKOS_OK
cmake -S sim -B sbuild -DCMAKE_CXX_COMPILER="$PWD/kinstall/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$PWD/kinstall" -DADC_INCLUDE="$PWD/include" -DCMAKE_BUILD_TYPE=Release > scfg.log 2>&1 \
  || { echo SCFG_FAIL; tail -40 scfg.log; exit 1; }
cmake --build sbuild -j 8 > sbuild.log 2>&1 || { echo P1_BUILD_FAIL; tail -60 sbuild.log; exit 1; }
echo P1_BUILD_OK
./sbuild/p1
echo "exit=$?"

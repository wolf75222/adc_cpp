#!/bin/bash -l
module load cuda/12.6
cd "$HOME/pops_gpu_p1" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
cmake -S sim7 -B sbuild7 -DCMAKE_CXX_COMPILER="$PWD/kinstall/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$PWD/kinstall" -DADC_INCLUDE="$PWD/include" -DCMAKE_BUILD_TYPE=Release > s7cfg.log 2>&1 \
  || { echo CFG_FAIL; tail -40 s7cfg.log; exit 1; }
cmake --build sbuild7 -j 16 > s7build.log 2>&1 || { echo BUILD_FAIL; grep -iE "error" s7build.log | head -20; exit 1; }
echo P7_BUILD_OK
./sbuild7/p7
echo "exit=$?"

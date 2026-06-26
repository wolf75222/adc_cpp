#!/bin/bash -l
module load cuda/12.6
cd "$HOME/pops_gpu_p1" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
cmake -S sim5 -B sbuild5 -DCMAKE_CXX_COMPILER="$PWD/kinstall/bin/nvcc_wrapper" \
  -DKokkos_ROOT="$PWD/kinstall" -DPOPS_INCLUDE="$PWD/include" -DCMAKE_BUILD_TYPE=Release > s5cfg.log 2>&1 \
  || { echo CFG_FAIL; tail -40 s5cfg.log; exit 1; }
cmake --build sbuild5 -j 8 > s5build.log 2>&1 || { echo BUILD_FAIL; tail -50 s5build.log; exit 1; }
echo P5_BUILD_OK
for t in test_flux_register test_amr_diffusion; do
  if ./sbuild5/$t >/dev/null 2>&1; then echo "[GPU ok] $t"; else echo "[GPU FAIL] $t (exit $?)"; fi
done

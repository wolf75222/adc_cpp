#!/bin/bash -l
# Compile (nvcc, aarch64 du noeud GPU) + execute la brique generee sur H100.
module load cuda/12.6
cd "$HOME/pops_dsl_gpu" || exit 3
echo "noeud=$(hostname)  arch=$(uname -m)"
nvcc -std=c++20 -arch=sm_90 -I include euler_gpu.cu -o euler_gpu 2> nvcc.log
if [ $? -ne 0 ]; then echo "BUILD_FAIL"; tail -30 nvcc.log; exit 1; fi
echo "BUILD_OK"
./euler_gpu
echo "exit=$?"

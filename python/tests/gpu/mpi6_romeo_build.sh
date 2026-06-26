#!/usr/bin/env bash
# Phase 6 (MPI multi-GPU) sur ROMEO : MPI + Kokkos (backend Cuda) + OpenMPI CUDA-aware.
# Batit mpi6_fillboundary et le lance en np=1/2/4 (un GH200 par rang). gfails=0 attendu partout.
# Place mpi6_fillboundary.cpp + mpi6_CMakeLists.txt (-> CMakeLists.txt) dans $HOME/pops_gpu_p1/sim_mpi6,
# les en-tetes adc dans $HOME/pops_gpu_p1/include, Kokkos (Cuda+Serial, Hopper90) dans kinstall.
# Soumettre : sbatch python/tests/gpu/mpi6_romeo_build.sh
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=mpi6
module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda          # OpenMPI 4.1.7 CUDA-aware (UCX)
cd "$HOME/pops_gpu_p1"
NW="$PWD/kinstall/bin/nvcc_wrapper"
rm -rf mpi6_build
cmake -S sim_mpi6 -B mpi6_build -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ROOT="$PWD/kinstall" \
  -DADC_INCLUDE="$PWD/include" -DCMAKE_BUILD_TYPE=Release || exit 1
cmake --build mpi6_build -j 8 || exit 1
for NP in 1 2 4; do
  echo "--- np=$NP ---"
  srun -n $NP --gpus-per-task=1 ./mpi6_build/mpi6_fillboundary
done

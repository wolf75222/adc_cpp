#!/usr/bin/env bash
# Preuve GPU du pas deux-fluides 2D AP sur ROMEO 2025, partition armgpu (Nvidia GH200).
# Compile examples/gpu/two_fluid_ap_kokkos.cpp (TwoFluidAP2D<GeometricMG> : transport
# Rusanov des deux especes + Lorentz implicite + Poisson reformule par multigrille,
# tout via for_each_cell backend Kokkos -> Cuda) et l'execute en regime raide
# (omega_pe=1e3, dt*omega_pe=5). Les checksums doivent coincider avec ceux du CPU.
#
# aarch64 + CUDA ne se compile PAS sur le login (x86_64) ; il faut un job armgpu.
# Soumettre depuis la racine du depot : sbatch scripts/romeo_tfap.sbatch
#
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=adc_tfap
#SBATCH --output=tfap_gpu.%J.out
#SBATCH --error=tfap_gpu.%J.err

set -e

romeo_load_armgpu_env
spack load cuda@12.6.3
spack load /vziqyqt          # gcc@11.4.1 (armgpu) : pin par hash (specs multiples)
spack load cmake 2>/dev/null || spack load /phhwuh3

cd "$SLURM_SUBMIT_DIR"
K=/scratch_p/rmdraux/kokkos-install
rm -rf build-kokkos
echo "===== CONFIGURE ====="
cmake -S examples/gpu -B build-kokkos \
  -DCMAKE_CXX_COMPILER="$K/bin/nvcc_wrapper" -DKokkos_ROOT="$K" \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -6
echo "===== BUILD ====="
cmake --build build-kokkos --target two_fluid_ap_kokkos -j8 2>&1 | tail -15
echo "===== RUN GPU ====="
./build-kokkos/two_fluid_ap_kokkos

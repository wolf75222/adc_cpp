#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=gpuval2
# Build + run "round 2" : validation device (Kokkos Cuda) des features post-#48 a chemin device.
# Compile via nvcc_wrapper (backend Cuda) sur le noeud armgpu ET un oracle Serial (g++), puis
# compare bit-a-bit (diff_bin). NE recompile PAS Kokkos : reutilise kinstall prebuilt.
set -u
module load cuda/12.6
romeo_load_armgpu_env
cd "$HOME/pops_gpu_p1" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
NW="$PWD/kinstall/bin/nvcc_wrapper"
INC="$PWD/gpuval2_include"          # en-tetes A JOUR (rsync depuis master)
SRC="$PWD/gpuval2_src"              # les .cpp des harness + diff_bin + CMakeLists
RES="$PWD/gpuval2_results"
mkdir -p "$RES"

# --- build device (Kokkos Cuda) ------------------------------------------------------------------
rm -rf gpuval2_build
cmake -S "$SRC" -B gpuval2_build -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ROOT="$PWD/kinstall" \
  -DADC_INCLUDE="$INC" -DCMAKE_BUILD_TYPE=Release \
  > "$RES/cfg.log" 2>&1 || { echo CFG_FAIL; tail -50 "$RES/cfg.log"; exit 1; }
cmake --build gpuval2_build -j 16 > "$RES/build.log" 2>&1 \
  || { echo BUILD_FAIL; grep -iE "error" "$RES/build.log" | head -40; exit 1; }
echo GPUVAL2_BUILD_OK

# --- build oracle Serial (g++, POPS_HAS_KOKKOS off => Serial(host)) -------------------------------
echo "=== build Serial oracle (g++) ==="
g++ -std=c++20 -O2 -I "$INC" "$SRC/gpu_epm_validate.cpp" -o "$RES/epm_serial" \
  > "$RES/serial_epm.log" 2>&1 || { echo SERIAL_EPM_FAIL; tail -30 "$RES/serial_epm.log"; }
g++ -std=c++20 -O2 -I "$INC" "$SRC/gpu_amr_bz_validate.cpp" -o "$RES/amrbz_serial" \
  > "$RES/serial_amrbz.log" 2>&1 || { echo SERIAL_AMRBZ_FAIL; tail -30 "$RES/serial_amrbz.log"; }
g++ -std=c++20 -O2 -I "$INC" "$SRC/gpu_aux_validate.cpp" -o "$RES/aux_serial" \
  > "$RES/serial_aux.log" 2>&1 || { echo SERIAL_AUX_FAIL; tail -30 "$RES/serial_aux.log"; }
g++ -std=c++20 -O2 "$SRC/diff_bin.cpp" -o "$RES/diff_bin" 2>/dev/null

cd "$RES" || exit 3
# Pour chaque feature : MEME logique en exec=Cuda (srun 1 GPU) et oracle exec=Serial (g++), puis
# diff_bin -> dmax sur CHAQUE cellule (vise 0 = bit-identique). for_each_cell est ASYNC sous Cuda :
# chaque harness fait device_fence() avant la lecture hote / le dump.
echo "######## (2)+(3) EPM Helmholtz + anisotrope ########"
srun -n 1 --gpus-per-task=1 "$PWD/../gpuval2_build/gpu_epm_validate" --dump=epm_cuda; echo "rc=$?"
./epm_serial --dump=epm_serial; echo "rc=$?"
./diff_bin epm_cuda_screened64.bin epm_serial_screened64.bin
./diff_bin epm_cuda_aniso64.bin    epm_serial_aniso64.bin

echo "######## (1) T_e via load_aux<5> ########"
srun -n 1 --gpus-per-task=1 "$PWD/../gpuval2_build/gpu_aux_validate" --dump=aux_cuda; echo "rc=$?"
./aux_serial --dump=aux_serial; echo "rc=$?"
./diff_bin aux_cuda_te.bin aux_serial_te.bin

echo "######## (4) B_z par niveau AMR ########"
srun -n 1 --gpus-per-task=1 "$PWD/../gpuval2_build/gpu_amr_bz_validate" --dump=amrbz_cuda; echo "rc=$?"
./amrbz_serial --dump=amrbz_serial; echo "rc=$?"
./diff_bin amrbz_cuda_amrbz.bin amrbz_serial_amrbz.bin

echo GPUVAL2_DONE

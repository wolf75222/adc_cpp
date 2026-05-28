#!/usr/bin/env bash
# Preuve GPU sur ROMEO 2025, partition armgpu (Nvidia GH200 Grace Hopper).
# Compile et execute le kernel CUDA examples/gpu/advect_cuda.cu, qui reutilise
# les vues Array4 de la couche maillage (annotees ADC_HD). Soumettre depuis la
# racine du depot : sbatch scripts/romeo_gpu.sbatch
#
# Important : l'aarch64 + CUDA ne se compile PAS sur le login (x86_64) ; il faut
# un job armgpu (ce script, ou un salloc/srun --constraint=armgpu).
#
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --job-name=adc_gpu
#SBATCH --output=adc_gpu.%J.out
#SBATCH --error=adc_gpu.%J.err

set -e

romeo_load_armgpu_env
spack load cuda@12.6.3
spack load gcc@11.4.1   # compilateur hote compatible nvcc (CUDA 12.6)

cd "$SLURM_SUBMIT_DIR"
nvcc -O2 -std=c++17 -I include examples/gpu/advect_cuda.cu -o /tmp/advect_cuda
/tmp/advect_cuda

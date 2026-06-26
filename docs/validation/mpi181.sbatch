#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=adc181mpi
#SBATCH --output=adc181mpi.%J.out
#SBATCH --error=adc181mpi.%J.err

# ADC-181 (corroboration substrat multi-GPU) : re-execute le harnais multi-box MPI multi-GPU
# (#59 / fix halos #254) sur CE noeud, np=1 (oracle multi-box mono-rang) vs np=2 (boites
# reparties sur 2 GH200). Invariants globaux DOIVENT etre bit-identiques. Modele = B_z device
# (fonction pure de la position) : c'est le SUBSTRAT halos/MPI que le .so hyqmom15 emprunte,
# pas le modele hyqmom15 lui-meme (cf. note GH200_HYQMOM15.md, perimetre).
#
# Reproductibilite : le binaire gpu_amr_bz_mpi_validate est VERSIONNE dans le depot
# (python/tests/gpu/gpu_amr_bz_mpi_validate.cpp) et se construit par le script versionne
# python/tests/gpu/romeo_gpuval2_mpi_build.sh (nvcc_wrapper + OpenMPI CUDA-aware). On le
# reutilise ici ; ce script ne reconstruit que la partie multi-GPU specifique a ADC-181.
set -u
module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda
echo "noeud=$(hostname) arch=$(uname -m)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -2
BIN="$HOME/pops_gpu_p1/gpuval2_mpi_build/gpu_amr_bz_mpi_validate"
test -x "$BIN" || { echo "BIN introuvable: $BIN"; exit 2; }
OUT="/scratch_p/$USER/adc181/mpi_out.txt"
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
for NP in 1 2; do
  echo "--- np=$NP ---" | tee -a "$OUT"
  srun -n $NP --gpus-per-task=1 "$BIN" 2>&1 | tee -a "$OUT"
done
echo "ADC181_MPI_DONE"

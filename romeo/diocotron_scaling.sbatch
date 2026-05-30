#!/usr/bin/env bash
# Etude de scaling GPU du diocotron sur ROMEO armgpu. Lance le MEME probleme a un nombre
# CROISSANT de GPU H100 (scaling FORT : resolution fixe, on regarde l'acceleration) puis,
# en variante, resolution proportionnelle au nombre de GPU (scaling FAIBLE : temps/pas
# constant attendu). Produit un CSV (gpus, res, t_par_pas) a tracer.
#
# Reserver le MAX de GPU une fois, puis sous-allouer par srun -n. Soumettre :
#   sbatch romeo/diocotron_scaling.sbatch

#SBATCH --account=R00000
#SBATCH --job-name=dio_scal
#SBATCH --constraint=armgpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=320G
#SBATCH --time=01:30:00
#SBATCH --error=dio_scal.%J.err
#SBATCH --output=dio_scal.%J.out

set -euo pipefail
romeo_load_armgpu_env
spack load cmake; spack load cuda
spack load kokkos +cuda +wrapper cuda_arch=90
spack load openmpi +cuda

RUN=/scratch_p/$USER/$SLURM_JOBID; mkdir -p "$RUN"; cd "$RUN"
SRC=$SLURM_SUBMIT_DIR; BUILD=$RUN/build-gh200
KROOT=$(spack location -i kokkos)
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper" -DKokkos_ROOT="$KROOT"
cmake --build "$BUILD" -j --target diocotron_mpi

BIN=$BUILD/bin/diocotron_mpi
STEPS=400

echo "mode,gpus,res,wall_s" | tee scaling.csv
# --- scaling FORT : 4096^2 fixe, 1 -> 32 GPU ---
for NP in 1 2 4 8 16 32; do
  t0=$(date +%s.%N)
  srun -n "$NP" --gpu-bind=single:1 "$BIN" "$RUN/s_$NP" 4096 "$STEPS" >/dev/null 2>&1 || true
  t1=$(date +%s.%N)
  echo "fort,$NP,4096,$(echo "$t1 - $t0" | bc)" | tee -a scaling.csv
done
# --- scaling FAIBLE : ~2M cellules/GPU (res = 1448*sqrt(NP)) ---
for NP in 1 4 16 32; do
  R=$(python3 -c "import math;print(int(1448*math.sqrt($NP)//2*2))")
  t0=$(date +%s.%N)
  srun -n "$NP" --gpu-bind=single:1 "$BIN" "$RUN/w_$NP" "$R" "$STEPS" >/dev/null 2>&1 || true
  t1=$(date +%s.%N)
  echo "faible,$NP,$R,$(echo "$t1 - $t0" | bc)" | tee -a scaling.csv
done
echo "ecrit $RUN/scaling.csv (recopier hors de /scratch_p)."

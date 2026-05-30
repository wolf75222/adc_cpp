#!/usr/bin/env bash
# Hero run diocotron sur ROMEO armgpu (Nvidia GH200 Grace-Hopper) : 1 rang MPI par GPU
# H100, chaque rang sature son GPU via le seam for_each_cell -> Kokkos/CUDA. Multi-noeud
# par Infiniband 800 Gb/s + NVLink/NVSWITCH intra-noeud. C'est le "full machine" reel :
# MPI a travers les 232 H100, Kokkos/CUDA dedans. (OpenMP est le mode CPU separe, x64cpu.)
#
# But scientifique : pousser la resolution du diocotron assez haut pour que le bord
# d'anneau reste net et que le taux de croissance numerique atteigne l'analytique 0.911
# (cf. docs/fig_diocotron_reproduction.png : la grille uniforme plafonne a ~60% a n<=256).
#
# Soumettre :  sbatch romeo/diocotron_hero.sbatch
# Adapter ACCOUNT (code projet), NODES, RES (resolution), STEPS avant de lancer.

#SBATCH --account=R00000              # <- METTRE le vrai code projet (demander a Romain)
#SBATCH --job-name=dio_hero
#SBATCH --constraint=armgpu
#SBATCH --nodes=8                     # 8 serveurs x 4 GH200 = 32 GPU H100
#SBATCH --ntasks-per-node=4           # 1 rang MPI par GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72            # 288 coeurs Grace / 4 rangs
#SBATCH --mem=320G                    # par SERVEUR (385 Go HBM3 + 1.9 To LPDDR5 dispo)
#SBATCH --time=02:00:00
#SBATCH --error=dio_hero.%J.err
#SBATCH --output=dio_hero.%J.out

set -euo pipefail

# --- environnement ARM/GPU (jamais dans .bashrc, jamais melange avec x64cpu) ---
romeo_load_armgpu_env
spack load cmake
spack load cuda
spack load kokkos +cuda +wrapper cuda_arch=90   # Hopper = sm_90 ; adapter au spec installe
spack load openmpi +cuda                          # MPI CUDA-aware (GPU-direct sur IB)

# --- repertoire de travail rapide (le home a un quota de 20 Go) ---
RUN=/scratch_p/$USER/$SLURM_JOBID
mkdir -p "$RUN"
cd "$RUN"
SRC=$SLURM_SUBMIT_DIR            # racine du depot adc_cpp (lieu de soumission)
BUILD=$RUN/build-gh200

# --- build hybride MPI + Kokkos/CUDA (aarch64 : OBLIGATOIREMENT dans un job armgpu) ---
KROOT=$(spack location -i kokkos)
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KROOT/bin/nvcc_wrapper" -DKokkos_ROOT="$KROOT"
cmake --build "$BUILD" -j --target diocotron_mpi

# --- parametres du run (resolution = le levier ; le bord net exige le grand n) ---
RES=${RES:-8192}                # 8192^2 = 67 M cellules, reparti sur les rangs
STEPS=${STEPS:-4000}
NP=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "hero diocotron : $NP rangs (= GPU H100), resolution ${RES}^2, ${STEPS} pas"

# 1 rang -> 1 GPU. srun lit la conf du job (remplace mpirun). Binding GPU par tache.
srun --gpu-bind=single:1 \
  "$BUILD/bin/diocotron_mpi" "$RUN/out" "$RES" "$STEPS"

echo "termine. sorties dans $RUN/out (densite gather rang 0)."
# Penser a recopier $RUN ailleurs : /scratch_p n'est PAS sauvegarde.

#!/usr/bin/env bash
# Build + execution des tests MPI sur ROMEO 2025 (partition CPU x64cpu).
# Soumettre depuis la racine du depot :
#   sbatch scripts/romeo_mpi.sbatch
# Adapter --account a son projet (cf. `sacctmgr show Association where user=$USER`).
#
# Toolchain validee (mai 2026) : gcc@14.2.0 + openmpi@5.0.5 + cmake@3.31.8, via
# Spack. mpicxx enveloppe gcc 14 (C++23). Si un `spack load` est ambigu, preciser
# par hash : `spack find -l <paquet>` puis `spack load /<hash>`.
#
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --job-name=adc_mpi
#SBATCH --output=adc_mpi.%J.out
#SBATCH --error=adc_mpi.%J.err

set -e

romeo_load_x64cpu_env
spack load gcc@14.2.0
spack load cmake@3.31.8
spack load openmpi@5.0.5
export OMPI_CXX=$(command -v g++)  # mpicxx -> gcc 14 (C++23)

# build dans le scratch du job (le home a un petit quota)
SRC="$SLURM_SUBMIT_DIR"
BUILD="/scratch_p/$USER/$SLURM_JOBID/build-mpi"
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DADC_USE_MPI=ON -DADC_USE_EIGEN=OFF \
      -DCMAKE_CXX_COMPILER="$(command -v mpicxx)"
cmake --build "$BUILD" -j

# srun lance les rangs MPI en lisant la configuration du job (8 rangs, 2 noeuds).
# test_mpi_array_reduce exerce all_reduce_sum_inplace (brique du reflux multi-patch
# distribue) ; test_mpi_diocotron le pas couple E x B distribue.
for t in test_mpi_smoke test_mpi_fillboundary test_mpi_advect \
         test_mpi_array_reduce test_mpi_diocotron test_mpi_overlap; do
  echo "=== $t ==="
  srun "$BUILD/bin/$t"
done

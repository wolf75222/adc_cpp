#!/usr/bin/env bash
# Produit les FRAMES 2D du diocotron AMR 3 niveaux sur ROMEO (x64cpu, OpenMP) pour en
# faire un GIF. Les hero-runs de croissance n'ecrivent que ring_amp.csv (amplitude(t)),
# pas de champ 2D : ce job lance l'exemple diocotron_amr3 qui dumpe dens_c/1/2 + boxes.csv.
# Build DIRECT g++ (le cmake de Spack est casse sur les noeuds de calcul ; l'exemple est
# header-only, le seam for_each_cell s'active sur _OPENMP). Determinisme OpenMP == serie.
#
#   sbatch romeo/diocotron_amr3_gif.sbatch
# puis recopier out_gif_amr3/ et : python3 scripts/make_diocotron_amr3_gif.py out_gif_amr3 <gif>

#SBATCH --account=r250127
#SBATCH --job-name=adc_gif
#SBATCH --constraint=x64cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --error=adc_gif.%J.err
#SBATCH --output=adc_gif.%J.out

set -uo pipefail
romeo_load_x64cpu_env
spack load gcc@14.2.0 2>/dev/null || spack load gcc

cd "$SLURM_SUBMIT_DIR"
BIN=/scratch_p/$USER/diocotron_amr3_gif
g++ -std=c++23 -O3 -fopenmp -I include examples/diocotron_amr3.cpp -o "$BIN" \
  || { echo "BUILD_KO"; exit 1; }
echo "== build OK =="

export OMP_NUM_THREADS=96
OUT="$SLURM_SUBMIT_DIR/out_gif_amr3"
mkdir -p "$OUT"
"$BIN" "$OUT" 256 600
echo "== frames ecrits =="
ls "$OUT" | head
echo "nb dens_c frames: $(ls "$OUT"/dens_c_*.txt 2>/dev/null | wc -l)"
echo "FIN"

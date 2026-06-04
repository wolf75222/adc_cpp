#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=gpuval2mpi
# B_z par niveau AMR MULTI-BOX distribue sur plusieurs GPU (#59) : build header-only (advance_amr +
# MPI) via nvcc_wrapper (Cuda) + OpenMPI CUDA-aware, run np=1/2/4 (un GH200 par rang). np=1 = oracle
# multi-box mono-rang ; np=2/4 doivent etre BIT-IDENTIQUES (mass/csum/csumsq/cmax invariants au nb de
# rangs : B_z est fonction pure de la position). NE recompile PAS Kokkos.
set -u
module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda
cd "$HOME/adc_gpu_p1" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
NW="$PWD/kinstall/bin/nvcc_wrapper"
INC="$PWD/gpuval2_include"
SRC="$PWD/gpuval2_mpi_src"
rm -rf gpuval2_mpi_build
cmake -S "$SRC" -B gpuval2_mpi_build -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ROOT="$PWD/kinstall" \
  -DADC_INCLUDE="$INC" -DCMAKE_BUILD_TYPE=Release > gpuval2_mpi_cfg.log 2>&1 \
  || { echo CFG_FAIL; tail -50 gpuval2_mpi_cfg.log; exit 1; }
cmake --build gpuval2_mpi_build -j 8 > gpuval2_mpi_build.log 2>&1 \
  || { echo BUILD_FAIL; grep -iE "error" gpuval2_mpi_build.log | head -40; exit 1; }
echo GPUVAL2MPI_BUILD_OK
OUT=gpuval2_mpi_out.txt
: > "$OUT"
for NP in 1 2 4; do
  echo "--- np=$NP ---" | tee -a "$OUT"
  srun -n $NP --gpus-per-task=1 ./gpuval2_mpi_build/gpu_amr_bz_mpi_validate 2>&1 | tee -a "$OUT"
done
echo "=== PARITE np=1/2/4 (invariants globaux bit-identiques attendus) ===" | tee -a "$OUT"
python3 - "$OUT" <<'PY' | tee -a "$OUT"
import re, sys
rows = []
for line in open(sys.argv[1]):
    m = re.search(r'AMRBZMPI np=(\d+).*mass=([-\d.eE+]+) csum=([-\d.eE+]+) csumsq=([-\d.eE+]+) cmax=([-\d.eE+]+) \| bz_bad=(\d+)', line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6))))
if not rows:
    print("PARITE: aucune ligne AMRBZMPI parsee"); sys.exit(0)
ref = rows[0]
print(f"oracle np={ref[0]} : mass={ref[1]:.17e} csum={ref[2]:.17e} bz_bad={ref[5]}")
worst = 0.0
for r in rows:
    d = max(abs(r[1]-ref[1]), abs(r[2]-ref[2]), abs(r[3]-ref[3]), abs(r[4]-ref[4]))
    worst = max(worst, d)
    print(f"np={r[0]:>1} | dmass={abs(r[1]-ref[1]):.3e} dcsum={abs(r[2]-ref[2]):.3e} dcsumsq={abs(r[3]-ref[3]):.3e} dcmax={abs(r[4]-ref[4]):.3e} | bz_bad={r[5]}")
print(f"PARITE dmax (mass/csum/csumsq/cmax, tous np vs oracle np=1) = {worst:.3e}")
print("PARITE OK (bit-identique)" if worst == 0.0 else "PARITE NON BIT-IDENTIQUE")
PY
echo GPUVAL2MPI_DONE

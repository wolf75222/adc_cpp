#!/usr/bin/env bash
# VALIDATION INTEGREE AmrSystem + MPI + GPU sur ROMEO (GH200) : MPI + Kokkos (backend Cuda) +
# OpenMPI CUDA-aware. Batit amrmpi_integrated et le lance en np=1/2/4 (un GH200 par rang). Le run
# np=1 est l'ORACLE mono-GPU ; np=2/4 doivent etre BIT-IDENTIQUES (mass / csum / csumsq / cmax). On
# diff les sorties a la fin (dmax_csum == 0 attendu). Masse conservee a l'arrondi FMA.
# Place amrmpi_integrated.cpp + amrmpi_CMakeLists.txt (-> CMakeLists.txt) dans $HOME/adc_gpu_p1/sim_amrmpi,
# python/amr_system.cpp dans $HOME/adc_gpu_p1/src_amr, les en-tetes adc a jour dans $HOME/adc_gpu_p1/include,
# Kokkos (Cuda+Serial, Hopper90) dans kinstall. Soumettre : sbatch amrmpi_romeo_build.sh
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=amrmpi
module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda          # OpenMPI 4.1.7 CUDA-aware (UCX)
cd "$HOME/adc_gpu_p1" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
NW="$PWD/kinstall/bin/nvcc_wrapper"
rm -rf amrmpi_build
cmake -S sim_amrmpi -B amrmpi_build -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ROOT="$PWD/kinstall" \
  -DADC_INCLUDE="$PWD/include" -DADC_SRC="$PWD/src_amr" -DCMAKE_BUILD_TYPE=Release \
  > amrmpi_cfg.log 2>&1 || { echo CFG_FAIL; tail -40 amrmpi_cfg.log; exit 1; }
cmake --build amrmpi_build -j 8 > amrmpi_build.log 2>&1 || { echo BUILD_FAIL; tail -60 amrmpi_build.log; exit 1; }
echo AMRMPI_BUILD_OK
OUT=amrmpi_out.txt
: > "$OUT"
for NP in 1 2 4; do
  echo "--- np=$NP ---" | tee -a "$OUT"
  srun -n $NP --gpus-per-task=1 ./amrmpi_build/amrmpi_integrated 2>&1 | tee -a "$OUT"
done
echo "=== PARITE np=1/2/4 (csum bit-identique attendu) ===" | tee -a "$OUT"
python3 - "$OUT" <<'PY' | tee -a "$OUT"
import re, sys
rows = []
for line in open(sys.argv[1]):
    m = re.search(r'AMRMPI np=(\d+).*mass=([-\d.eE+]+) \| csum=([-\d.eE+]+) csumsq=([-\d.eE+]+) cmax=([-\d.eE+]+) \| crossrank_spread=([-\d.eE+]+)', line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
if not rows:
    print("PARITE: aucune ligne AMRMPI parsee"); sys.exit(0)
ref = rows[0]
print(f"oracle np={ref[0]} : mass={ref[1]:.17e} csum={ref[2]:.17e}")
worst = 0.0
for r in rows:
    dmass = abs(r[1]-ref[1]); dcsum = abs(r[2]-ref[2]); dq = abs(r[3]-ref[3]); dx = abs(r[4]-ref[4])
    worst = max(worst, dmass, dcsum, dq, dx)
    print(f"np={r[0]:>1} | dmass={dmass:.3e} dcsum={dcsum:.3e} dcsumsq={dq:.3e} dcmax={dx:.3e} | spread={r[5]:.3e}")
print(f"PARITE dmax (sur mass/csum/csumsq/cmax, tous np vs oracle np=1) = {worst:.3e}")
print("PARITE OK (bit-identique)" if worst == 0.0 else "PARITE NON BIT-IDENTIQUE")
PY
echo AMRMPI_DONE

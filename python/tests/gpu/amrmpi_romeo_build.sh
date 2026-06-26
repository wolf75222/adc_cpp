#!/usr/bin/env bash
# VALIDATION INTEGREE AmrSystem + MPI + GPU + STRONG-SCALING sur ROMEO (GH200) : MPI + Kokkos (backend
# Cuda) + OpenMPI CUDA-aware. Batit amrmpi_integrated et le lance en np=1/2/4 (un GH200 par rang).
# CHAQUE run mesure DEUX modes dans le meme binaire : grossier REPLIQUE (defaut, ne scale pas) puis
# REPARTI (distribute_coarse=true, 2x2, le mode strong-scaling). Le script :
#   - verifie cmax bit-identique cross-rang dans les deux modes (max insensible a l'ordre) ;
#   - reporte per_step_ms np=1/2/4 pour replique ET reparti -> montre (ou non) le strong-scaling.
# Place amrmpi_integrated.cpp + amrmpi_CMakeLists.txt (-> CMakeLists.txt) dans $HOME/pops_gpu_p1/sim_amrmpi,
# python/amr_system.cpp dans $HOME/pops_gpu_p1/src_amr, les en-tetes pops a jour dans $HOME/pops_gpu_p1/include,
# Kokkos (Cuda+Serial, Hopper90) dans kinstall. Soumettre : sbatch amrmpi_romeo_build.sh
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:60:00
#SBATCH --job-name=amrmpi
module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda          # OpenMPI 4.1.7 CUDA-aware (UCX)
cd "$HOME/pops_gpu_p1" || exit 3
echo "noeud=$(hostname) arch=$(uname -m)"
NW="$PWD/kinstall/bin/nvcc_wrapper"
rm -rf amrmpi_build
cmake -S sim_amrmpi -B amrmpi_build -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ROOT="$PWD/kinstall" \
  -DPOPS_INCLUDE="$PWD/include" -DPOPS_SRC="$PWD/src_amr" -DCMAKE_BUILD_TYPE=Release \
  > amrmpi_cfg.log 2>&1 || { echo CFG_FAIL; tail -40 amrmpi_cfg.log; exit 1; }
cmake --build amrmpi_build -j 8 > amrmpi_build.log 2>&1 || { echo BUILD_FAIL; tail -60 amrmpi_build.log; exit 1; }
echo AMRMPI_BUILD_OK
OUT=amrmpi_out.txt
: > "$OUT"
for NP in 1 2 4; do
  echo "--- np=$NP ---" | tee -a "$OUT"
  srun -n $NP --gpus-per-task=1 ./amrmpi_build/amrmpi_integrated 2>&1 | tee -a "$OUT"
done
echo "=== PARITE cmax + STRONG-SCALING per_step_ms (replique vs reparti, np=1/2/4) ===" | tee -a "$OUT"
python3 - "$OUT" <<'PY' | tee -a "$OUT"
import re, sys
# AMRMPI[tag] np=N ... cmax=... cmax_crossrank_spread=...
sig = {}   # (tag, np) -> (cmax, spread)
tms = {}   # (tag, np) -> per_step_ms
for line in open(sys.argv[1]):
    m = re.search(r'AMRMPI\[(\w+)\] np=(\d+).*cmax=([-\d.eE+]+) \| cmax_crossrank_spread=([-\d.eE+]+)', line)
    if m:
        sig[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    # ligne per_step_ms : "AMRMPI[tag] exec=... per_step_ms=X (max over ranks, n=N, measured=M)".
    # le np n'y figure pas ; on le retient depuis la derniere ligne signature du meme tag.
    m = re.search(r'AMRMPI\[(\w+)\].*per_step_ms=([-\d.eE+]+)', line)
    if m:
        tag = m.group(1)
        nps_tag = [n for (t, n) in sig if t == tag]
        if nps_tag:
            tms[(tag, max(nps_tag))] = float(m.group(2))
# cmax bit-identique cross-rang (les deux modes) + cmax identique entre np (max insensible a l'ordre)
worst = 0.0
for tag in ("replique", "reparti"):
    nps = sorted(n for (t, n) in sig if t == tag)
    if not nps: continue
    ref = sig[(tag, nps[0])][0]
    for n in nps:
        cmax, spread = sig[(tag, n)]
        worst = max(worst, abs(cmax - ref), spread)
        print(f"[{tag}] np={n} cmax={cmax:.17e} dcmax_vs_np1={abs(cmax-ref):.3e} crossrank_spread={spread:.3e}")
print(f"PARITE cmax dmax (tous tags/np) = {worst:.3e}")
print("PARITE cmax OK (bit-identique)" if worst == 0.0 else "PARITE cmax NON BIT-IDENTIQUE")
print("--- STRONG-SCALING (per_step_ms, max sur les rangs) ---")
def scaling(tag):
    nps = sorted(n for (t, n) in tms if t == tag)
    if not nps: return
    base = tms[(tag, nps[0])]
    for n in nps:
        ms = tms[(tag, n)]
        sp = base / ms if ms > 0 else float('nan')
        eff = sp / n * 100.0
        print(f"[{tag}] np={n} per_step_ms={ms:.4f} speedup={sp:.2f}x efficiency={eff:.1f}%")
scaling("replique")
scaling("reparti")
PY
echo AMRMPI_DONE

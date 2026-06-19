#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=adcdiompi
#SBATCH --output=adcdiompi.%J.out
#SBATCH --error=adcdiompi.%J.err

# Diocotron hyqmom15 multi-GPU + MPI (ferme la branche 3 de GH200_HYQMOM15.md via une
# decomposition MPI de System, mono-boite round-robin -- meme topologie que le run_mpi.py
# valide CPU). Le MEME driver diocotron_gpu.cpp (briques Hyqmom15Hyp/Src/Ell + Poisson
# geometric_mg), lie a OpenMPI CUDA-aware (-DADC_VALIDATION_MPI=ON), tourne sous
# srun -n {1,2,4} --gpus-per-task=1 (un GH200 par rang). np=1 = oracle ; np=2/4 DOIVENT
# coincider au niveau ulp sur la masse globale (la somme du V-cycle Poisson se reassocie
# entre nombres de rangs : PAS bit-exact, cf. section 2/3 de GH200_HYQMOM15.md).
#
# Reproductibilite (mirroir de parity181.sbatch / mpi181.sbatch) :
#   1. git archive master | ssh ROMEO 'cd ~/adc_cpp && tar x' (PAS rsync/clone).
#   2. SRC = docs/validation versionne (diocotron_gpu.cpp, CMakeLists.txt).
#   3. brique DSL hyqmom15_brick.hpp + etat initial ic_128.raw sont GENERES (artefacts
#      derives, bit-reproductibles, NON versionnes) par make_brick_and_ic.py sur une machine
#      disposant du module python adc, puis deposes dans WORK :
#        ADC_CASES=/chemin/adc_cases PYTHONPATH=/chemin/adc_cpp/python \
#          python docs/validation/make_brick_and_ic.py --ns 128 --out WORK
#      (md5 ic_128.raw = da245ba8934546986508976a64156d2e, stable d'un run a l'autre.)
#   4. sbatch docs/validation/diocotron_mpi.sbatch
set -u
module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda  # OpenMPI CUDA-aware (meme que gpu_amr_bz_mpi_validate / mpi181.sbatch)

# Tampons de comm epingles (PR #254 : comm_allocator = SharedHostPinnedSpace) : evitent le
# deadlock CUDA-IPC des buffers UVM/managed entre GPU isoles par cgroup (srun --gpus-per-task=1).
# On garde l'env historique ceinture-et-bretelles (inoffensif si redondant) ; un suivi pourra le
# retirer une fois qu'un run propre prouve que les tampons epingles suffisent seuls.
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0

echo "noeud=$(hostname) arch=$(uname -m)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -4

SRC="${ADCDIO_SRC:-$HOME/adc_cpp/docs/validation}"
WORK="${ADCDIO_WORK:-$HOME/adcdio_work}"
SCR="/scratch_p/$USER/adcdio"
KGPU="$HOME/adc_gpu_p1/kinstall"
NVCC="$KGPU/bin/nvcc_wrapper"
ADCINC="$HOME/adc_cpp/include"
mkdir -p "$SCR" "$WORK"

# --- 0) staging : sources versionnees depuis SRC, brique + IC generes deja presents dans WORK ---
# diocotron_amr_gpu.cpp is referenced by the shared CMakeLists add_executable (ADC-320) even though
# this recipe only builds diocotron_gpu -- stage it too or the configure fails before building.
for f in diocotron_gpu.cpp diocotron_amr_gpu.cpp CMakeLists.txt; do
  cp -f "$SRC/$f" "$WORK/$f" || { echo "SRC introuvable: $SRC/$f"; exit 2; }
done
for f in hyqmom15_brick.hpp ic_128.raw; do
  test -f "$WORK/$f" || { echo "ARTEFACT GENERE MANQUANT: $WORK/$f -- voir entete (make_brick_and_ic.py --ns 128)"; exit 2; }
done
echo "[stage] sources versionnees + brique/IC generes en place dans $WORK"

# --- 1) build device (Cuda) + MPI du MEME driver ---
test -x "$NVCC" || { echo "nvcc_wrapper introuvable: $NVCC"; exit 2; }
rm -rf "$SCR/build"
cmake -S "$WORK" -B "$SCR/build" -DCMAKE_CXX_COMPILER="$NVCC" -DKokkos_ROOT="$KGPU" \
  -DADC_INCLUDE="$ADCINC" -DADC_VALIDATION_MPI=ON -DCMAKE_BUILD_TYPE=Release \
  > "$SCR/cfg.log" 2>&1 || { echo CFG_FAIL; tail -50 "$SCR/cfg.log"; exit 1; }
cmake --build "$SCR/build" -j "${SLURM_CPUS_PER_TASK:-8}" > "$SCR/build.log" 2>&1 \
  || { echo BUILD_FAIL; grep -iE "error" "$SCR/build.log" | head -40; exit 1; }
BIN="$SCR/build/diocotron_gpu"
test -x "$BIN" || { echo NO_BIN; exit 2; }
echo BUILD_OK

# --- 2) parite multi-rang : np=1 (oracle) vs np=2/4 (boites reparties sur N GH200) ---
cd "$WORK"
OUT="$SCR/diompi_out.txt"; : > "$OUT"
for NP in 1 2 4; do
  echo "--- np=$NP ---" | tee -a "$OUT"
  rm -rf "$SCR/out_np$NP"
  srun -n $NP --gpus-per-task=1 "$BIN" --n 128 --ic ic_128.raw --out "$SCR/out_np$NP" \
       --max-steps 200 --snap-every 9999 --diag-every 50 --tend 99 2>&1 | tee -a "$OUT"
done

echo "=== PARITE np=1/2/4 (cons. masse <1e-12 par run ; masse np vs np=1 rel <1e-12, ulp attendu) ===" | tee -a "$OUT"
python3 - "$OUT" <<'PY' | tee -a "$OUT"
import re, sys
rows = []
for ln in open(sys.argv[1]):
    m = re.search(r'HYQMOMPI np=(\d+).*mass=([-\d.eE+]+) massdrift=([-\d.eE+]+)', ln)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
if not rows:
    print('PARITE: aucune ligne HYQMOMPI parsee')
    sys.exit(1)
ref = rows[0]
ok = True
for np_, mass, drift in rows:
    rel = abs(mass - ref[1]) / abs(ref[1]) if ref[1] else float('nan')
    cons = drift < 1e-12
    par = (np_ == 1) or (rel < 1e-12)
    ok = ok and cons and par
    print('np=%d masscons=%.3e(%s) massrel_vs_np1=%.3e(%s)'
          % (np_, drift, 'OK' if cons else 'FAIL', rel, 'OK' if par else 'FAIL'))
print('MULTIGPU_MPI_PASS' if ok else 'MULTIGPU_MPI_FAIL')
sys.exit(0 if ok else 1)
PY
echo ADCDIO_MPI_DONE

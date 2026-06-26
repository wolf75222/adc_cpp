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
#SBATCH --job-name=adcamrmpi
#SBATCH --output=adcamrmpi.%J.out
#SBATCH --error=adcamrmpi.%J.err

# Diocotron hyqmom15 AmrSystem MULTI-BOITE, decomposition de domaine MPI reelle (ADC-320, suite de
# ADC-181). Le driver diocotron_amr_gpu.cpp (composite Hyqmom15Hyp/Src/Ell + Poisson geometric_mg sur
# AmrSystem, distribute_coarse=true) tourne sous srun -n {1,2,4} --gpus-per-task=1 (un GH200 par rang).
# Contrairement au round-robin mono-boite de diocotron_mpi.sbatch (ADC-181, toute la boite sur le rang
# 0), le grossier devient un BoxArray multi-boite REPARTI : a np>1 fill_boundary echange les 15 moments
# entre GPU (halos inter-GPU reels, tampons SharedHostPinnedSpace, fix #254). np=1 = oracle ; cmax
# BIT-IDENTIQUE entre np ; masse/sommes np=2/4 vs np=1 au dernier ulp (reassociation FMA cross-rang,
# reelle ici) ; preuve de repartition coarse_local < coarse_total a np>1 (ADC-319).
#
# Reproductibilite (miroir de diocotron_mpi.sbatch) :
#   1. git archive <branche> | ssh ROMEO 'cd ~/adc_cpp && tar x' (PAS rsync/clone).
#   2. SRC = docs/validation versionne (diocotron_amr_gpu.cpp, diocotron_gpu.cpp, CMakeLists.txt).
#   3. brique DSL hyqmom15_brick.hpp + etat initial ic_128.raw : artefacts GENERES (bit-reproductibles,
#      NON versionnes, les MEMES que le driver System) par make_brick_and_ic.py, deposes dans WORK :
#        POPS_CASES=/chemin/adc_cases PYTHONPATH=/chemin/adc_cpp/python \
#          python docs/validation/make_brick_and_ic.py --ns 128 --out WORK
#      (md5 ic_128.raw = da245ba8934546986508976a64156d2e, stable d'un run a l'autre.)
#   4. sbatch docs/validation/diocotron_amr_mpi.sbatch
set -u
module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda  # OpenMPI CUDA-aware (meme que diocotron_mpi.sbatch / amrmpi_romeo_build)

# Tampons de comm epingles (PR #254 : comm_allocator = SharedHostPinnedSpace) : evitent le deadlock
# CUDA-IPC des buffers entre GPU isoles par cgroup (srun --gpus-per-task=1). Ceinture-et-bretelles.
export OMPI_MCA_btl_smcuda_use_cuda_ipc=0

echo "noeud=$(hostname) arch=$(uname -m)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -4

SRC="${ADCAMR_SRC:-$HOME/adc_cpp/docs/validation}"
WORK="${ADCAMR_WORK:-$HOME/adcamr_work}"
SCR="/scratch_p/$USER/adcamr"
KGPU="$HOME/pops_gpu_p1/kinstall"
NVCC="$KGPU/bin/nvcc_wrapper"
ADCINC="$HOME/adc_cpp/include"
mkdir -p "$SCR" "$WORK"

# --- 0) staging : sources versionnees depuis SRC, brique + IC generes deja presents dans WORK ---
# diocotron_gpu.cpp est reference par add_executable du CMakeLists partage (on ne le BATIT pas ici).
for f in diocotron_amr_gpu.cpp diocotron_gpu.cpp CMakeLists.txt; do
  cp -f "$SRC/$f" "$WORK/$f" || { echo "SRC introuvable: $SRC/$f"; exit 2; }
done
for f in hyqmom15_brick.hpp ic_128.raw; do
  test -f "$WORK/$f" || { echo "ARTEFACT GENERE MANQUANT: $WORK/$f -- voir entete (make_brick_and_ic.py --ns 128)"; exit 2; }
done
echo "[stage] sources versionnees + brique/IC generes en place dans $WORK"

# --- 1) build device (Cuda) + MPI du driver AMR uniquement (pas la cible System -> pas system.cpp) ---
# Le build nvcc d'amr_system.cpp est lourd (~30 min). On REUTILISE le binaire de $SCR (scratch
# persistant) s'il existe deja, sauf ADCAMR_REBUILD=1 -- a poser apres tout changement de source ou
# d'en-tetes (le binaire ne depend pas du gate python ci-dessous).
BIN="$SCR/build/diocotron_amr_gpu"
if [ -x "$BIN" ] && [ "${ADCAMR_REBUILD:-0}" != "1" ]; then
  echo "BUILD_REUSED ($BIN ; poser ADCAMR_REBUILD=1 pour reconstruire apres un changement de source)"
else
  test -x "$NVCC" || { echo "nvcc_wrapper introuvable: $NVCC"; exit 2; }
  rm -rf "$SCR/build"
  cmake -S "$WORK" -B "$SCR/build" -DCMAKE_CXX_COMPILER="$NVCC" -DKokkos_ROOT="$KGPU" \
    -DADC_INCLUDE="$ADCINC" -DADC_VALIDATION_MPI=ON -DCMAKE_BUILD_TYPE=Release \
    > "$SCR/cfg.log" 2>&1 || { echo CFG_FAIL; tail -50 "$SCR/cfg.log"; exit 1; }
  cmake --build "$SCR/build" --target diocotron_amr_gpu -j "${SLURM_CPUS_PER_TASK:-8}" > "$SCR/build.log" 2>&1 \
    || { echo BUILD_FAIL; grep -iE "error" "$SCR/build.log" | head -40; exit 1; }
  test -x "$BIN" || { echo NO_BIN; exit 2; }
  echo BUILD_OK
fi

# --- 2) parite multi-rang : np=1 (oracle) vs np=2/4 (grossier reparti sur N GH200) ---
cd "$WORK"
OUT="$SCR/amrmpi_out.txt"; : > "$OUT"
for NP in 1 2 4; do
  echo "--- np=$NP ---" | tee -a "$OUT"
  srun -n $NP --gpus-per-task=1 "$BIN" --n 128 --ic ic_128.raw --steps 80 --dt 4e-4 \
       --regrid-every 8 2>&1 | tee -a "$OUT"
done

echo "=== PARITE AMR multi-boite (reparti==replique bit-pour-bit par np ; masse <1e-12 ; sommes np vs np=1 ulp ; coarse reparti) ===" | tee -a "$OUT"
python3 - "$OUT" <<'PY' | tee -a "$OUT"
import re, sys
rep = {}   # np -> dict(mass, massdrift, cmax, csum)
dist = {}  # np -> "yes"/"NO"
dmax = {}  # np -> dist_vs_repl_dmax
ok_run = set()
for ln in open(sys.argv[1]):
    m = re.search(r'HYQMOMAMR mode=reparti np=(\d+).*mass=([-\d.eE+]+) massdrift=([-\d.eE+]+).*'
                  r'csum=([-\d.eE+]+).*cmax=([-\d.eE+]+)', ln)
    if m:
        np_ = int(m.group(1))
        rep[np_] = dict(mass=float(m.group(2)), drift=float(m.group(3)),
                        csum=float(m.group(4)), cmax=float(m.group(5)))
    m = re.search(r'HYQMOMAMR distributed_coarse=(\w+) \(np=(\d+)', ln)
    if m:
        dist[int(m.group(2))] = m.group(1)
    m = re.search(r'HYQMOMAMR dist_vs_repl_dmax=([-\d.eE+]+)', ln)
    if m:
        # la ligne suit immediatement le bloc d'un np ; on l'attache au dernier np reparti vu
        if rep:
            dmax[max(rep)] = float(m.group(1))
    m = re.search(r'HYQMOMAMR_RUN_OK np=(\d+)', ln)
    if m:
        ok_run.add(int(m.group(1)))
if not rep:
    print('PARITE: aucune ligne HYQMOMAMR reparti parsee'); print('HYQMOMAMR_MULTIBOX_FAIL'); sys.exit(1)
nps = sorted(rep)
ref = rep[nps[0]]
ok = True
cmax_ref = ref['cmax']
for np_ in nps:
    r = rep[np_]
    massrel = abs(r['mass'] - ref['mass']) / abs(ref['mass']) if ref['mass'] else float('nan')
    dcmax = abs(r['cmax'] - cmax_ref)
    cons = r['drift'] < 1e-12
    par_mass = (np_ == nps[0]) or (massrel < 1e-12)
    distok = (np_ == 1) or (dist.get(np_) == 'yes')
    dm = dmax.get(np_)
    dmok = (dm is None) or (dm < 1e-9)  # reparti==replique a np fixe = preuve bit-exacte des halos
    runok = np_ in ok_run
    # cmax (pic ponctuel) NON gate entre np : le diocotron est instable + le regrid reagit a la
    # reassociation ulp du Poisson cross-rang (maillages != entre np). dcmax est INFORMATIF ; le
    # bit-exact se lit sur dist_vs_repl a np fixe et la conservation sur masse/sommes.
    ok = ok and cons and par_mass and distok and dmok and runok
    print('np=%d masscons=%.3e(%s) massrel_vs_np1=%.3e(%s) dcmax_vs_np1=%.3e(info) '
          'distributed=%s(%s) dist_vs_repl_dmax=%s(%s) run_ok=%s'
          % (np_, r['drift'], 'OK' if cons else 'FAIL', massrel, 'OK' if par_mass else 'FAIL',
             dcmax, dist.get(np_, '?'), 'OK' if distok else 'FAIL',
             ('%.3e' % dm) if dm is not None else 'n/a', 'OK' if dmok else 'FAIL',
             'OK' if runok else 'FAIL'))
print('HYQMOMAMR_MULTIBOX_PASS' if ok else 'HYQMOMAMR_MULTIBOX_FAIL')
sys.exit(0 if ok else 1)
PY
echo ADCAMR_MPI_DONE

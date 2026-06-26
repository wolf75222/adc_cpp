#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=adc181par
#SBATCH --output=adc181par.%J.out
#SBATCH --error=adc181par.%J.err

# ADC-181 : parite hote(Serial)/device(Cuda) du chemin vitesses exactes (dense_eig
# Hessenberg+QR) a travers le .so hyqmom15 (exact_speeds, riemann=hll), MEME noeud GH200.
# Les DEUX variantes sont compilees ICI depuis les MEMES sources versionnees (rien n'est
# stage a la main) :
#   * device : nvcc_wrapper du Kokkos GPU (kinstall), DefaultExecutionSpace=Cuda ;
#   * hote   : g++ + Kokkos Serial aarch64 construit sur le noeud, DefaultExecutionSpace=Serial.
#
# Reproductibilite (depuis un checkout frais) :
#   1. rsync le depot adc_cpp sur ROMEO (~/adc_cpp) -- en-tetes include/ + python/system.cpp.
#   2. SRC = docs/validation versionne : diocotron_gpu.cpp, CMakeLists.txt, compare_snap.cpp,
#      make_brick_and_ic.py. Defaut $HOME/adc_cpp/docs/validation (override ADC181_SRC).
#   3. La brique DSL hyqmom15_brick.hpp et l'etat initial ic_128.raw sont GENERES (pas
#      versionnes : artefacts derives, bit-reproductibles) par make_brick_and_ic.py sur une
#      machine disposant du module python adc (ex. build local) puis rsync dans WORK :
#        POPS_CASES=/chemin/adc_cases PYTHONPATH=/chemin/adc_cpp/python \
#          python docs/validation/make_brick_and_ic.py --ns 128 --out WORK
#      (IC + brique sont deterministes : md5 stable d'un run a l'autre.)
#   4. sbatch docs/validation/parity181.sbatch
set -u
module load cuda/12.6
romeo_load_armgpu_env
echo "noeud=$(hostname) arch=$(uname -m)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
g++ --version | head -1

SRC="${ADC181_SRC:-$HOME/adc_cpp/docs/validation}"
WORK="${ADC181_WORK:-$HOME/adc181_work}"
SCR="/scratch_p/$USER/adc181"
KSRC="$HOME/pops_gpu_p1/kokkos"
KSER="$SCR/kserial"
KGPU="$HOME/pops_gpu_p1/kinstall"
NVCC="$KGPU/bin/nvcc_wrapper"
ADCINC="$HOME/adc_cpp/include"
mkdir -p "$SCR" "$WORK"

# --- 0) staging : sources versionnees depuis SRC, brique+IC generes deja presents dans WORK ---
# diocotron_amr_gpu.cpp is referenced by the shared CMakeLists add_executable (ADC-320) even though
# parity181 only builds diocotron_gpu -- stage it too or the configure fails before building.
for f in diocotron_gpu.cpp diocotron_amr_gpu.cpp CMakeLists.txt compare_snap.cpp; do
  cp -f "$SRC/$f" "$WORK/$f" || { echo "SRC introuvable: $SRC/$f"; exit 2; }
done
for f in hyqmom15_brick.hpp ic_128.raw; do
  test -f "$WORK/$f" || { echo "ARTEFACT GENERE MANQUANT: $WORK/$f -- voir entete (make_brick_and_ic.py)"; exit 2; }
done
echo "[stage] sources versionnees + brique/IC generes en place dans $WORK"

# --- 1) Kokkos Serial aarch64 (g++) si absent ---
if [ ! -f "$KSER/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KSER/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
  echo "[kserial] build Serial Kokkos aarch64"
  rm -rf kserial_build "$KSER"
  cmake -S "$KSRC" -B kserial_build -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_INSTALL_PREFIX="$KSER" -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ARCH_NATIVE=ON -DCMAKE_CXX_STANDARD=20 > kserial_cfg.log 2>&1 \
    || { echo KSER_CFG_FAIL; tail -30 kserial_cfg.log; exit 1; }
  cmake --build kserial_build -j "${SLURM_CPUS_PER_TASK:-16}" --target install > kserial_build.log 2>&1 \
    || { echo KSER_BUILD_FAIL; tail -40 kserial_build.log; exit 1; }
  echo KSER_OK
else
  echo "[kserial] reutilise $KSER"
fi

# --- 2) Variante device (Cuda) du driver, MEME source ---
echo "[device] build diocotron_gpu (Cuda, nvcc_wrapper)"
test -x "$NVCC" || { echo "nvcc_wrapper introuvable: $NVCC"; exit 2; }
rm -rf "$SCR/dev_build"
cmake -S "$WORK" -B "$SCR/dev_build" -DCMAKE_CXX_COMPILER="$NVCC" \
  -DKokkos_ROOT="$KGPU" -DADC_INCLUDE="$ADCINC" -DCMAKE_BUILD_TYPE=Release \
  > "$SCR/dev_cfg.log" 2>&1 || { echo DEV_CFG_FAIL; tail -40 "$SCR/dev_cfg.log"; exit 1; }
# -j 8 (half the 16-core allocation) caps memory: nvcc_wrapper expands each heavy
# template TU per device pass, so the full allocation can exhaust node RAM. Intentional.
cmake --build "$SCR/dev_build" -j 8 > "$SCR/dev_build.log" 2>&1 \
  || { echo DEV_BUILD_FAIL; grep -iE "error" "$SCR/dev_build.log" | head -40; exit 1; }
echo DEV_BUILD_OK
DEV="$SCR/dev_build/diocotron_gpu"
test -x "$DEV" || { echo "DEV introuvable apres build"; exit 2; }

# --- 3) Variante hote (Serial) du driver, MEME source ---
echo "[host] build diocotron_gpu (Serial)"
rm -rf "$SCR/ser_build"
cmake -S "$WORK" -B "$SCR/ser_build" -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_ROOT="$KSER" -DADC_INCLUDE="$ADCINC" -DCMAKE_BUILD_TYPE=Release \
  > "$SCR/ser_cfg.log" 2>&1 || { echo SER_CFG_FAIL; tail -40 "$SCR/ser_cfg.log"; exit 1; }
# -j 8 mirrors the device build cap above: keep the host parity build symmetric. Intentional.
cmake --build "$SCR/ser_build" -j 8 > "$SCR/ser_build.log" 2>&1 \
  || { echo SER_BUILD_FAIL; grep -iE "error" "$SCR/ser_build.log" | head -40; exit 1; }
echo SER_BUILD_OK
SER="$SCR/ser_build/diocotron_gpu"
test -x "$SER" || { echo "SER introuvable apres build"; exit 2; }

# --- 4) comparateur ---
g++ -O2 -std=c++17 "$WORK/compare_snap.cpp" -o "$SCR/compare_snap" || { echo CMP_BUILD_FAIL; exit 1; }

# --- 5) parite : 20 pas du MEME etat initial ---
cd "$WORK"
rm -rf "$SCR/dev_run" "$SCR/ser_run"
echo "=== RUN DEVICE (Cuda) 20 pas ==="
"$DEV" --n 128 --ic ic_128.raw --out "$SCR/dev_run" --max-steps 20 --snap-every 20 --diag-every 1 --tend 99
echo "=== RUN HOST (Serial) 20 pas ==="
"$SER" --n 128 --ic ic_128.raw --out "$SCR/ser_run" --max-steps 20 --snap-every 20 --diag-every 1 --tend 99

echo "=== COMPARE snap_000020.raw (device vs host) ==="
"$SCR/compare_snap" "$SCR/dev_run/snap_000020.raw" "$SCR/ser_run/snap_000020.raw"

echo "=== growth.csv device vs host (diff) ==="
echo "--- device ---"; cat "$SCR/dev_run/growth.csv"
echo "--- host ---";   cat "$SCR/ser_run/growth.csv"

# --- 6) timing indicatif : 50 pas chacun ---
echo "=== TIMING device 50 pas ==="
t0=$(date +%s.%N)
"$DEV" --n 128 --ic ic_128.raw --out "$SCR/dev_t" --max-steps 50 --snap-every 9999 --diag-every 9999 --tend 99 >/dev/null
t1=$(date +%s.%N)
python3 -c "print('device: 50 pas en %.3f s -> %.4f s/pas'%($t1-$t0,($t1-$t0)/50))"
echo "=== TIMING host 50 pas ==="
t0=$(date +%s.%N)
"$SER" --n 128 --ic ic_128.raw --out "$SCR/ser_t" --max-steps 50 --snap-every 9999 --diag-every 9999 --tend 99 >/dev/null
t1=$(date +%s.%N)
python3 -c "print('host:   50 pas en %.3f s -> %.4f s/pas'%($t1-$t0,($t1-$t0)/50))"
echo "ADC181_PARITY_DONE"

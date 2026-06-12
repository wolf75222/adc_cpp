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
# Variante device = binaire reutilise du job 654562 (meme source diocotron_gpu.cpp +
# hyqmom15_brick.hpp, nvcc_wrapper). Variante hote = MEME source recompilee g++ + Kokkos Serial.
set -u
module load cuda/12.6
romeo_load_armgpu_env
echo "noeud=$(hostname) arch=$(uname -m)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
g++ --version | head -1

WORK="$HOME/adc181_work"
SCR="/scratch_p/$USER/adc181"
KSRC="$HOME/adc_gpu_p1/kokkos"
KSER="$SCR/kserial"
ADCINC="$HOME/adc_cpp/include"
mkdir -p "$SCR"
cd "$SCR"

# --- 1) Kokkos Serial aarch64 (g++) si absent ---
if [ ! -f "$KSER/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KSER/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
  echo "[kserial] build Serial Kokkos aarch64"
  rm -rf kserial_build "$KSER"
  cmake -S "$KSRC" -B kserial_build -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_INSTALL_PREFIX="$KSER" -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ARCH_NATIVE=ON -DCMAKE_CXX_STANDARD=20 > kserial_cfg.log 2>&1 \
    || { echo KSER_CFG_FAIL; tail -30 kserial_cfg.log; exit 1; }
  cmake --build kserial_build -j 16 --target install > kserial_build.log 2>&1 \
    || { echo KSER_BUILD_FAIL; tail -40 kserial_build.log; exit 1; }
  echo KSER_OK
else
  echo "[kserial] reutilise $KSER"
fi

# --- 2) Variante hote (Serial) du driver, MEME source ---
echo "[host] build diocotron_gpu (Serial)"
rm -rf "$SCR/ser_build"
cmake -S "$WORK" -B "$SCR/ser_build" -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_ROOT="$KSER" -DADC_INCLUDE="$ADCINC" -DCMAKE_BUILD_TYPE=Release \
  > "$SCR/ser_cfg.log" 2>&1 || { echo SER_CFG_FAIL; tail -40 "$SCR/ser_cfg.log"; exit 1; }
cmake --build "$SCR/ser_build" -j 8 > "$SCR/ser_build.log" 2>&1 \
  || { echo SER_BUILD_FAIL; grep -iE "error" "$SCR/ser_build.log" | head -40; exit 1; }
echo SER_BUILD_OK
SER="$SCR/ser_build/diocotron_gpu"
DEV="$SCR/diocotron_gpu_device"   # binaire device reutilise (job 654562)
test -x "$DEV" || { echo "DEV introuvable"; exit 2; }

# --- 3) comparateur ---
g++ -O2 -std=c++17 "$WORK/compare_snap.cpp" -o "$SCR/compare_snap" || { echo CMP_BUILD_FAIL; exit 1; }

# --- 4) parite : 20 pas du MEME etat initial ---
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

# --- 5) timing indicatif : 50 pas chacun ---
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

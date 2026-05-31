#!/usr/bin/env bash
# Porte de CORRECTION GPU sur UN GH200 d'armgpu : passe les binaires Kokkos/CUDA sous
# compute-sanitizer (NVIDIA, successeur de cuda-memcheck) + rejoue la validation
# bit-identique CPU vs GPU. But : attraper la classe de bugs la plus subtile du code
# (cf. docs/CHOICES.md : un device_fence() oublie = lecture hote de memoire unifiee
# encore en cours d'ecriture par un kernel, bug GPU SILENCIEUX invisible en CI CPU).
#
# Deux filets COMPLEMENTAIRES :
#   1. compute-sanitizer : erreurs memoire GPU larges (acces hors borne / mal alignes,
#      lectures non initialisees via initcheck, mauvais usage des primitives de synchro
#      via synccheck). Ne detecte PAS directement une course hote/device sur memoire
#      managee, mais couvre tout le reste de l'hygiene GPU.
#   2. checksum CPU vs GPU (coupled_kokkos) : c'est LUI qui attrape un fence manquant.
#      Un fence oublie fait DIVERGER le resultat GPU du resultat CPU connu
#      (4394594.404318) -> le checksum casse. C'est le detecteur reel des fences.
#
# Pre-requis : kokkos-src clone sous /scratch_p/$USER (les noeuds n'ont pas internet),
# cf. romeo/README.md. Soumettre depuis la racine du depot sur ROMEO :
#   sbatch romeo/sanitizer.sbatch
#
# NOTE : ce script n'a PAS pu etre execute hors ROMEO (pas de GPU) ; il reprend a
# l'identique la section build validee de diocotron_amr_gpu.sbatch. Verifier la
# premiere execution.

#SBATCH --account=r250127
#SBATCH --job-name=adc_sanitizer
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --error=adc_sanitizer.%J.err
#SBATCH --output=adc_sanitizer.%J.out

set -uo pipefail
module load cuda/12.6
echo "== uname=$(uname -m) ; nvcc=$(which nvcc) ; sanitizer=$(which compute-sanitizer) =="
nvcc --version | tail -2
compute-sanitizer --version 2>/dev/null | head -2 || echo "compute-sanitizer introuvable (verifier le module cuda)"

ROOT=/scratch_p/$USER
KSRC=$ROOT/kokkos-src
KINST=$ROOT/kokkos-install

# 1. Kokkos depuis les sources (cache : on saute si deja installe). IDENTIQUE a
#    diocotron_amr_gpu.sbatch (Serial + CUDA, Hopper sm_90, nvcc_wrapper).
if [ ! -f "$KINST/lib64/cmake/Kokkos/KokkosConfig.cmake" ] && [ ! -f "$KINST/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
  echo "== build Kokkos (Serial+CUDA, Hopper90) =="
  cmake -S "$KSRC" -B "$ROOT/kokkos-build" -DCMAKE_INSTALL_PREFIX="$KINST" \
    -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_HOPPER90=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DCMAKE_CXX_COMPILER="$KSRC/bin/nvcc_wrapper" -DCMAKE_CXX_STANDARD=17 \
    || { echo "KOKKOS_CFG_KO"; exit 1; }
  cmake --build "$ROOT/kokkos-build" -j 24 --target install || { echo "KOKKOS_BUILD_KO"; exit 1; }
fi
echo "== Kokkos installe: $KINST =="

# 2. le projet, backend Kokkos/CUDA. On construit les binaires qui EXERCENT la
#    frontiere hote/device : la reference auto-validee (coupled_kokkos), le pas AMR
#    complet, le test du reducteur (sommes/normes par le seam, cf. commit recent) et
#    les coupleurs (dont mass() route maintenant par for_each_cell_reduce_sum).
SRC=$SLURM_SUBMIT_DIR
BUILD=$SRC/build-gh200
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$KINST/bin/nvcc_wrapper" -DKokkos_ROOT="$KINST" \
  || { echo "ADC_CFG_KO"; exit 1; }
cmake --build "$BUILD" -j 10 --target coupled_kokkos diocotron_amr_kokkos \
  test_reduce test_amr_coupler_mp test_fft_coupler \
  || { echo "ADC_BUILD_KO"; exit 1; }

# --- helper : un binaire sous un outil compute-sanitizer, verdict par ERROR SUMMARY ---
# On NE met PAS --error-exitcode : Kokkos peut declencher des avertissements initcheck
# benins sur ses buffers internes ; on lit "ERROR SUMMARY: N errors" et on rapporte.
run_san () {  # $1=tool  $2=binaire  $3...=args
  local tool="$1"; shift
  local bin="$1"; shift
  echo "----- compute-sanitizer --tool $tool  $(basename "$bin") -----"
  local log; log=$(compute-sanitizer --tool "$tool" --leak-check full "$bin" "$@" 2>&1)
  echo "$log" | tail -25
  if echo "$log" | grep -q "ERROR SUMMARY: 0 errors"; then
    echo "  >>> $tool $(basename "$bin") : OK"
  else
    echo "  >>> $tool $(basename "$bin") : ERREURS (voir ci-dessus)"
  fi
}

echo "===== compute-sanitizer (memcheck / initcheck / synccheck) ====="
# memcheck (defaut) : acces hors borne / mal alignes / fuites. Sur les petits tests
# (nc=32) le ralentissement sanitizer est negligeable.
for t in memcheck initcheck synccheck; do
  run_san "$t" "$BUILD/bin/test_reduce"
  run_san "$t" "$BUILD/bin/test_amr_coupler_mp"
  run_san "$t" "$BUILD/bin/test_fft_coupler"
done
# coupled_kokkos sous memcheck (reference GPU compacte, exerce le pas couple complet).
run_san memcheck "$BUILD/bin/coupled_kokkos"

echo "===== detecteur reel des fences manquants : checksum bit-identique CPU vs GPU ====="
# Un device_fence() oublie fait diverger ce checksum du CPU connu. C'est le filet qui
# attrape la classe de bug visee (compute-sanitizer ne voit pas la course hote/device
# sur memoire managee). coupled_kokkos s'auto-valide CPU vs GPU en interne.
"$BUILD/bin/coupled_kokkos"
echo "== checksum CPU attendu (serie + Kokkos-OpenMP local) = 4394594.404318 =="
echo "== diocotron_amr_kokkos (pas AMR complet, doit rester bit-identique au CPU) =="
"$BUILD/bin/diocotron_amr_kokkos"
echo "== FIN sanitizer =="

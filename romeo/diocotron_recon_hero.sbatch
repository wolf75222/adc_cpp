#!/usr/bin/env bash
# Hero run "vers 0.911" sur ROMEO x64cpu (AMD EPYC, OpenMP) : pousser le taux de croissance
# diocotron mode 4 vers l'analytique 0.911 par la RECONSTRUCTION D'ORDRE 2 (MUSCL VanLeer),
# qui coupe la diffusion numerique du bord d'anneau (le vrai verrou de M1). On balaie la
# resolution effective en uniforme ET en AMR multi-niveau, en NoSlope (ordre 1, reference) et
# VanLeer (ordre 2, recon=1), et on extrait gamma_norm. Le pas AMR du diocotron lance une
# myriade de petits kernels : le CPU OpenMP (192 coeurs EPYC) y est plus efficace que le GPU
# (cf. docs/HERO_RUN_AMR.md), d'ou x64cpu. La stabilite haute resolution vient de
# GeometricMG::solve_robust (multigrille durci au bord embedded).
#
# Soumettre depuis la racine du depot sur ROMEO :  sbatch romeo/diocotron_recon_hero.sbatch
# Reglages (env) : EFFS="512 1024 2048", STEPSCALE (pas = STEPSCALE*eff_coarse), THREADS.

#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --job-name=dio_recon_hero
#SBATCH --output=dio_recon_hero.%J.out
#SBATCH --error=dio_recon_hero.%J.err

set -uo pipefail

romeo_load_x64cpu_env
spack load gcc@14.2.0
spack load cmake@3.31.8

SRC="$SLURM_SUBMIT_DIR"
RUN="/scratch_p/$USER/$SLURM_JOBID"
BUILD="$RUN/build-omp"
OUT="$RUN/out"
mkdir -p "$OUT"

# --- build OpenMP (gcc 14, C++23), determinisme OpenMP == serie garanti par le depot ---
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DADC_USE_OPENMP=ON \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" || { echo "CFG_KO"; exit 1; }
cmake --build "$BUILD" -j --target diocotron_column_amr || { echo "BUILD_KO"; exit 1; }
BIN="$BUILD/bin/diocotron_column_amr"

export OMP_NUM_THREADS=${THREADS:-$SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores OMP_PROC_BIND=close
echo "== diocotron recon hero : OMP_NUM_THREADS=$OMP_NUM_THREADS =="

EFFS=${EFFS:-"512 1024 2048"}
STEPSCALE=${STEPSCALE:-3}   # pas ~ STEPSCALE * (eff grossier) ; couvre la phase lineaire + saturation

run () {  # $1=label $2=nc $3=nsteps $4=refine $5=ml $6=recon
  local t0=$SECONDS
  "$BIN" "$OUT/$1" "$2" "$3" "$4" 4 "$5" "$6" > "$OUT/$1.log" 2>&1
  echo "  [$1] nc=$2 nsteps=$3 refine=$4 ml=$5 recon=$6  ($((SECONDS-t0)) s)  $(tail -1 $OUT/$1.log)"
}

for eff in $EFFS; do
  base=$((eff/2))
  nsu=$((STEPSCALE*eff)); nsa=$((STEPSCALE*base))
  echo "=== eff $eff ==="
  run "uni_ns_$eff"  "$eff"  "$nsu" 0 0 0   # uniforme NoSlope (reference ordre 1)
  run "uni_vl_$eff"  "$eff"  "$nsu" 0 0 1   # uniforme VanLeer (ordre 2)
  run "aml_vl_$eff"  "$base" "$nsa" 1 1 1   # AMR ml VanLeer (ordre 2, ~40% des cellules)
done

# --- extraction gamma_norm SANS numpy (regression lineaire log(a) sur une fenetre physique) ---
echo "===== gamma_norm (omega_D = 0.9/(2 pi) = 0.143239 ; analytique mode 4 = 0.911) ====="
python3 - "$OUT" <<'PY'
import sys, os, math
OUT=sys.argv[1]; omd=0.9/(2*math.pi)
def fit(path,t0,t1):
    ts=[];ls=[]
    for ln in open(path):
        ln=ln.strip()
        if not ln or ln[0]=='#' or ln.startswith('t,'): continue
        x,y=ln.split(','); x=float(x); y=float(y)
        if t0<=x<=t1 and y>0: ts.append(x); ls.append(math.log(y))
    n=len(ts)
    if n<4: return float('nan')
    sx=sum(ts); sy=sum(ls); sxx=sum(v*v for v in ts); sxy=sum(a*b for a,b in zip(ts,ls))
    return (n*sxy-sx*sy)/(n*sxx-sx*sx)
print(f"{'cas':<16}{'win[4,11]':>12}{'win[5,14]':>12}")
for f in sorted(os.listdir(OUT)):
    d=os.path.join(OUT,f,'ring_amp.csv')
    if not os.path.isfile(d): continue
    g1=fit(d,4,11)/omd; g2=fit(d,5,14)/omd
    print(f"{f:<16}{g1:>12.3f}{g2:>12.3f}")
PY

echo "== termine. sorties dans $OUT (recopier hors de /scratch_p, non sauvegarde). =="

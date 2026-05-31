#!/usr/bin/env bash
# Hero run "vers 0.911 a l'erreur du papier" sur ROMEO x64cpu (AMD EPYC, OpenMP) : on mesure le
# taux de croissance diocotron des modes l=3,4,5 avec la voie HAUTE PRECISION (WENO5-Z ordre 5 +
# SSPRK3, Poisson resolu a chaque etage RK, paroi embedded solve_robust) et on compare a
# l'analytique (g3=0.772, g4=0.911, g5=0.683), cible erreur relative < 1 % (idealement < 0.5 %,
# niveau du papier Hoffart arXiv:2510.11808). Le pas haute precision (3 solves Poisson/pas) est
# lourd : le CPU OpenMP (192 coeurs EPYC) convient mieux que le GPU a petits kernels.
#
# Soumettre depuis la racine du depot sur ROMEO :  sbatch romeo/diocotron_highorder_hero.sbatch
# Reglages (env) : MODES="3 4 5", EFFS="512 1024 2048", CFL=0.4, THREADS, TSCALE (pas ~ TSCALE*eff).

#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=long
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --job-name=dio_ho_hero
#SBATCH --output=dio_ho_hero.%J.out
#SBATCH --error=dio_ho_hero.%J.err

set -uo pipefail
romeo_load_x64cpu_env
spack load gcc@14.2.0
command -v g++ >/dev/null || { echo "GXX_KO"; exit 1; }

SRC="$SLURM_SUBMIT_DIR"
RUN="/scratch_p/$USER/$SLURM_JOBID"
OUT="$RUN/out"; mkdir -p "$OUT"
# build direct g++ -fopenmp (adc header-only ; pas de cmake, cf. note libmd dans recon_hero).
BIN="/scratch_p/$USER/adc-ho-bin/diocotron_highorder"
if [ ! -x "$BIN" ]; then
  mkdir -p "$(dirname "$BIN")"
  g++ -std=c++23 -O3 -fopenmp -I "$SRC/include" "$SRC/examples/diocotron_highorder.cpp" \
    -o "$BIN" || { echo "BUILD_KO"; exit 1; }
fi
export OMP_NUM_THREADS=${THREADS:-$SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores OMP_PROC_BIND=close
echo "== diocotron highorder hero : OMP_NUM_THREADS=$OMP_NUM_THREADS , bin=$BIN =="

# eff 2048 en WENO5+SSPRK3 (3 solves Poisson/pas) depasse le mur horaire avant l'extraction ;
# defaut 256/512/1024 (assez pour la PLATITUDE en resolution) ; ajouter 2048 via EFFS si temps long.
MODES=${MODES:-"3 4 5"}; EFFS=${EFFS:-"256 512 1024"}; CFL=${CFL:-0.4}; TSCALE=${TSCALE:-12}
for l in $MODES; do for eff in $EFFS; do
  ns=$(( (TSCALE*eff)/10 ))   # pas ~ TSCALE * (eff/10) -> temps physique ~ TSCALE (dt0 ~ 10/eff)
  t0=$SECONDS
  "$BIN" "$OUT/m${l}_e${eff}" "$eff" "$ns" "$l" 3 "$CFL" > "$OUT/m${l}_e${eff}.log" 2>&1
  echo "  [m=$l eff=$eff] ns=$ns ($((SECONDS-t0)) s)  $(tail -1 $OUT/m${l}_e${eff}.log)"
done; done

# --- gamma par fenetre glissante (numpy-free) : on rend le PIC du taux (regime lineaire) et
# l'erreur relative a l'analytique. width physique ~ fenetre du papier (mode-dependante). ---
echo "===== gamma_norm pic (regime lineaire) vs analytique (g3=0.772 g4=0.911 g5=0.683) ====="
python3 - "$OUT" <<'PY'
import sys, os, math
OUT=sys.argv[1]; omd=0.9/(2*math.pi)
G={3:0.772,4:0.911,5:0.683}
def read(p):
    t=[];a=[]
    for ln in open(p):
        ln=ln.strip()
        if not ln or ln[0]=='#' or ln.startswith('t,'): continue
        x,y=ln.split(','); t.append(float(x)); a.append(float(y))
    return t,a
def fit(t,a,t0,t1):
    xs=[];ys=[]
    for x,y in zip(t,a):
        if t0<=x<=t1 and y>0: xs.append(x); ys.append(math.log(y))
    n=len(xs)
    if n<4: return None
    sx=sum(xs);sy=sum(ys);sxx=sum(v*v for v in xs);sxy=sum(p*q for p,q in zip(xs,ys))
    den=n*sxx-sx*sx
    if abs(den)<1e-30: return None
    g=(n*sxy-sx*sy)/den
    mean=sy/n;b=(sy-g*sx)/n
    ssr=sum((q-(g*p+b))**2 for p,q in zip(xs,ys));sst=sum((q-mean)**2 for q in ys)
    return g/omd,(1-ssr/sst if sst>0 else 0)
# Mesure HONNETE et convergente : le PIC du taux sur les fenetres a exponentielle propre
# (R^2 > 0.9995). En regime lineaire le pic surestime a maillage grossier (comme le papier :
# 0.935 a maillage grossier) et converge vers l'analytique en raffinant. On rend pic + erreur.
print(f"{'cas':<12}{'l':>3}{'gamma_pic':>11}{'relerr%':>9}{'fenetre_pic':>16}")
for f in sorted(os.listdir(OUT)):
    d=os.path.join(OUT,f,'ring_amp.csv')
    if not os.path.isfile(d): continue
    try: l=int(f.split('_')[0][1:])
    except: continue
    t,a=read(d); width=1.0
    peak=None;t0=t[0]
    while t0+width<=t[-1]:
        r=fit(t,a,t0,t0+width)
        if r and r[1]>0.9995 and (peak is None or r[0]>peak[0]): peak=(r[0],t0,t0+width)
        t0+=width/4
    if peak:
        rel=abs(peak[0]-G[l])/G[l]*100
        print(f"{f:<12}{l:>3}{peak[0]:>11.4f}{rel:>9.2f}   [{peak[1]:.2f},{peak[2]:.2f}]")
PY
echo "== termine. sorties dans $OUT (recopier hors de /scratch_p). =="

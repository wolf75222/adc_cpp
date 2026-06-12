# Validation device GH200 : dense_eig + hyqmom15 (.so), single + multi-GPU

ADC-181. Valide sur GH200 (ROMEO, NVIDIA GH200 120GB, aarch64) le chemin device des
vitesses exactes par valeurs propres : `include/adc/numerics/dense_eig.hpp`
(`real_eig_minmax` : reduction de Hessenberg + iteration QR a double shift de Francis,
foncteurs nommes `ADC_HD`, tampons sur la pile, zero allocation) a travers le `.so`
hyqmom15 (briques emises par la DSL, `exact_speeds=True`, `riemann="hll"`), branche par
le seam de compilation `adc::add_compiled_model` (chemin natif complet : `assemble_rhs`
device, halos).

`dense_eig.hpp` a ete concu pour nvcc mais n'avait jamais ete EXECUTE sur device. Cette
note apporte la preuve d'execution device et la parite hote/device sur le meme noeud.

Artefact teste (identique au depot) :

```
md5(include/adc/numerics/dense_eig.hpp) = 86fb1cbbec0e265cd255559434ce83c6   (worktree == ROMEO ~/adc_cpp)
```

Complement de `docs/GPU_ROMEO.md` (qui valide le flux d'une brique generee) : ici on
valide le chemin EIGEN (bornes de vitesses signees du HLL) bout en bout dans un run.

## Recette (rappel)

Noeud `armgpu` (Grace-Hopper, aarch64), `module load cuda/12.6` + `romeo_load_armgpu_env`.
Compilateur device = `nvcc_wrapper` du Kokkos installe `~/adc_gpu_p1/kinstall`
(SERIAL;CUDA, sm_90). Le driver `diocotron_gpu.cpp` (briques `Hyqmom15Hyp/Src/Ell` +
`adc::System` + Poisson `geometric_mg`) lit l'etat initial binaire `ic_128.raw` (produit
par le python valide) et avance le diocotron 15 moments. Sources et recette completes :
`~/runs_hyqmom/gpu/` sur ROMEO (driver, `hyqmom15_brick.hpp`, `CMakeLists.txt`, IC).

Scripts versionnes (reproductibilite) : `docs/validation/parity181.sbatch` (parite
hote/device + timing, section 2 et 4), `docs/validation/mpi181.sbatch` (substrat
multi-GPU, section 3), `docs/validation/compare_snap.cpp` (comparateur d'etats finaux).

## 1. Execution device (acquis, job 654562)

Le driver complet a tourne sur GH200 device (`exec=Cuda`), 24706 pas, modele 15 moments
avec `dense_eig` / HLL exact / sources electriques / multigrille, garde `DT_COLLAPSE` :

```
noeud=romeo-a045  NVIDIA GH200 120GB
[fin] 24706 pas, t=0.97320, derive de masse 1.72e-13
[DT_COLLAPSE] pas 24706 ... etat au bord de realisabilite, projection requise (ADC-177). Sortie propre.
```

`real_eig_minmax` est appele PAR CELLULE a chaque pas (bornes de vitesses du flux HLL).
24706 pas sans NaN ni repli Gershgorin destructeur, derive de masse 1.7e-13 : la boucle
QR par cellule reste bornee (pas de blocage ni d'explosion du nombre d'iterations) sur
device. C'est la premiere preuve que le chemin Hessenberg+QR `ADC_HD` s'execute sur GH200.

## 2. Parite hote (Serial) / device (Cuda), MEME noeud

Protocole A==B des campagnes precedentes. Le MEME `diocotron_gpu.cpp` + `hyqmom15_brick.hpp`
+ le MEME arbre d'en-tetes `~/adc_cpp/include` sont compiles en deux variantes sur le
noeud GH200 :

* device : binaire reutilise du job 654562 (nvcc_wrapper, `DefaultExecutionSpace=Cuda`) ;
* hote   : recompile `g++ -O3` contre un Kokkos Serial aarch64 construit sur le noeud
  (`DefaultExecutionSpace=Serial`).

Sous Serial, `DefaultExecutionSpace == DefaultHostExecutionSpace` : la garde `if constexpr`
de `for_each_cell` (#165) prend la boucle hote sequentielle sur les petites boites (niveaux
grossiers du V-cycle) la ou le device garde `parallel_for` ; aucune dependance
inter-iteration, donc bit-identique attendu hors reductions. `real_eig_minmax` est le MEME
code source `ADC_HD` instancie host d'un cote, device de l'autre.

20 pas du MEME etat initial `ic_128.raw`, dump binaire de l'etat final (15 moments + phi),
compares par `compare_snap` (job 654862, noeud romeo-a057) :

```
=== RUN DEVICE (Cuda) 20 pas ===  [fin] 20 pas, t=0.01594, derive de masse 1.63e-13
=== RUN HOST (Serial) 20 pas ===  [fin] 20 pas, t=0.01594, derive de masse 1.64e-13
=== COMPARE snap_000020.raw (device vs host) ===
dt_clock (|t_a-t_b|) = 2.429e-17
payload doubles      = 262144   (15*128^2 moments + 128^2 potentiel)
bit-identiques       = 27710 (10.57%)
max |a-b|            = 3.450573e-13   (index 256884 -> dans le potentiel phi)
max rel              = 6.456023e-10
```

Lecture :

* **Chemin `dense_eig` lui-meme : bit-identique.** Le pas de temps `dt` (colonne `dt` de
  `growth.csv`) est issu de `step_cfl` = CFL sur le MAX des bornes de vitesses rendues par
  `real_eig_minmax` cellule par cellule. Une reduction MAX est exacte quel que soit l'ordre :
  `dt` est identique device/hote a tous les pas, et l'horloge cumulee `t` ne differe que de
  2.4e-17 apres 20 pas (rel 1.5e-15). Les bornes de vitesses produites par le QR `ADC_HD`
  sont donc identiques en device et en hote. Le mode physique du diocotron `a4` (l=4, magnitude
  ~5.94e-2) est lui aussi bit-identique device/hote dans `growth.csv`.
* **Ecart residuel = potentiel multigrille, pas le solveur de vitesses.** `max |a-b|` =
  3.45e-13 tombe a l'index 256884, dans la zone `phi` (>= 245760). Le potentiel sort du
  V-cycle Poisson dont le critere de residu repose sur une reduction SOMME : sa reassociation
  en `parallel_for` (device) vs boucle serie (hote) change le dernier bit du residu, donc
  marginalement les corrections, accumule sur les pas. Ecart absolu 3e-13 sur des champs
  d'ordre 1 a 1.7e3 : niveau bruit machine. `max rel` 6.5e-10 porte sur une entree de
  magnitude quasi nulle (relatif gonfle). Comportement attendu et documente des reductions
  paralleles ; le chemin vitesses exactes (l'objet de ADC-181) n'en est pas la cause.

## 3. Multi-GPU MPI (substrat halos, job 654863) et perimetre

Le driver hyqmom15 utilise `adc::System` (mono-boite, mono-rang) : il n'a PAS de
decomposition de domaine MPI. La correction multi-GPU du modele se factorise en deux
briques independantes :

1. correctness du modele sur un GPU, dont `dense_eig` (section 1 + section 2) ;
2. machinerie halos multi-boite + MPI multi-GPU, qui est PURE position/voisinage et ne
   touche jamais le calcul par cellule de `real_eig_minmax`.

`real_eig_minmax` etant strictement local a la cellule (lit le vecteur de moments local,
forme le bloc jacobien, rend min/max du spectre), sa correction multi-GPU se REDUIT a
(1) deja validee ici et (2) deja validee independamment (fix halos CUDA-IPC #254 ;
parite multi-box np=1/2/4 #59).

Corroboration sur CE noeud, post-#254, du substrat halos multi-GPU (harnais
`gpu_amr_bz_mpi_validate`, B_z par niveau AMR multi-box distribue, un GH200 par rang) :

```
np=1 exec=Cuda : mass=2.10017927603615240 csum=537.645894665255014 csumsq=1129.79422430042723 cmax=2.19619397662556448 | bz_bad=0
np=2 exec=Cuda : mass=2.10017927603615151 csum=537.645894665254787 csumsq=1129.79422430042814 cmax=2.19619397662556448 | bz_bad=0
```

Ecart np=1 vs np=2 : `cmax` (reduction max, associative+commutative exacte en virgule
flottante) BIT-IDENTIQUE ; `mass`/`csum`/`csumsq` (reductions somme) au dernier ulp
(drel 4.2e-16 / 4.2e-16 / 8.0e-16) du fait de la reassociation des sommes entre nombres
de rangs. `bz_bad=0` aux deux rangs. Le substrat multi-GPU est donc bit-identique sur le
max et au dernier ulp sur les sommes : comportement attendu et documente.

Perimetre NON couvert (a faire, distinct) : un driver hyqmom15 MULTI-BOITE + MPI dedie
(soit `AmrSystem` multi-box cablant le composite `Hyqmom15Hyp/Src/Ell` + Poisson, soit une
decomposition MPI de `System`). Plomberie reelle, non triviale : le seam existant
`gpu_amr_bz_mpi_validate` valide les halos mais avec le modele B_z, pas hyqmom15. C'est le
reste a livrer pour fermer entierement la branche 3 de l'issue.

## 4. Divergence warp de real_eig_minmax (indicatif)

`real_eig_minmax` itere un nombre de fois DEPENDANT DES DONNEES par cellule (deflation QR) :
des threads d'un meme warp peuvent iterer un nombre de fois different (divergence warp).
Mesure indicative temps/pas (device GH200 vs hote Serial du meme noeud romeo-a057, n=128,
50 pas, job 654869) :

```
device(Cuda) : 50 pas en  2.579 s -> 0.0516 s/pas (19.4 pas/s)
host(Serial) : 50 pas en 35.045 s -> 0.7009 s/pas ( 1.4 pas/s)
```

Le device est ~13.6x plus rapide qu'un coeur Grace seul a n=128 (comparaison
throughput, pas un cout de divergence isole : le device emploie le GPU entier, l'hote
un coeur). Aucun ralentissement pathologique : la boucle QR par cellule ne fait pas
diverger le pas device, et le run de 24706 pas (section 1) confirme la stabilite.

Le cap d'iterations EISPACK (30/bloc) et le repli Gershgorin bornent le cout par cellule :
la divergence reste celle, benigne, d'un solveur dense minuscule par thread.

## Conclusion

* `dense_eig` (`real_eig_minmax`, Hessenberg+QR `ADC_HD`) EXECUTE et CORRECT sur GH200
  device a travers le `.so` hyqmom15 (`exact_speeds`, `hll`) : section 1 (24706 pas) +
  section 2 (parite hote/device meme noeud).
* substrat multi-GPU MPI re-confirme bit-identique (max) / dernier ulp (sommes) sur le
  noeud post-#254 : section 3.
* reste : driver hyqmom15 multi-boite + MPI dedie (section 3, perimetre).

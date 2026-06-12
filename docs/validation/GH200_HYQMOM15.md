# Validation device GH200 : dense_eig + hyqmom15 (.so), single + multi-GPU

ADC-181. Valide sur GH200 (ROMEO, NVIDIA GH200 120GB, aarch64) le chemin device des
vitesses exactes par valeurs propres : `include/adc/numerics/dense_eig.hpp`
(`real_eig_minmax` : reduction de Hessenberg + iteration QR a double shift de Francis,
foncteurs nommes `ADC_HD`, tampons sur la pile, zero allocation) a travers le `.so`
hyqmom15 (briques emises par la DSL, `exact_speeds=True`, `riemann="hll"`), branche par
le seam de compilation `adc::add_compiled_model` (chemin natif complet : `assemble_rhs`
device, halos).

`dense_eig.hpp` a ete concu pour nvcc mais n'avait jamais ete EXECUTE sur device. Cette
note apporte la preuve d'execution device et la parite hote/device sur le meme noeud (les
md5 des sources/artefacts testes sont en section Reproductibilite ci-dessous).

Complement de `docs/GPU_ROMEO.md` (qui valide le flux d'une brique generee) : ici on
valide le chemin EIGEN (bornes de vitesses signees du HLL) bout en bout dans un run.

## Recette (rappel)

Noeud `armgpu` (Grace-Hopper, aarch64), `module load cuda/12.6` + `romeo_load_armgpu_env`.
Compilateur device = `nvcc_wrapper` du Kokkos installe `~/adc_gpu_p1/kinstall`
(SERIAL;CUDA, sm_90). Le driver `diocotron_gpu.cpp` (briques `Hyqmom15Hyp/Src/Ell` +
`adc::System` + Poisson `geometric_mg`) lit l'etat initial binaire `ic_128.raw` et avance le
diocotron 15 moments.

## Reproductibilite (sources versionnees)

Tout ce qui produit les chiffres ci-dessous est dans le depot, sous `docs/validation/` :

* `make_brick_and_ic.py` : generateur de la brique DSL `hyqmom15_brick.hpp` (emise par
  `emit_cpp_{brick,source,elliptic}` depuis le modele hyqmom15 valide) ET de l'etat initial
  `ic_<n>.raw` (`diocotron_state` du python valide) ;
* `diocotron_gpu.cpp` : driver (lit l'IC binaire, assemble le composite par
  `add_compiled_model`, avance, dump `snap_*.raw` + `growth.csv`) ;
* `CMakeLists.txt` : build du driver (compile aussi `python/system.cpp`) ;
* `compare_snap.cpp` : comparateur d'etats finaux ;
* `parity181.sbatch` (parite + timing, sections 2 et 4), `mpi181.sbatch` (substrat
  multi-GPU, section 3).

`hyqmom15_brick.hpp` et `ic_<n>.raw` ne sont PAS versionnes : ce sont des artefacts derives,
regeneres a l'identique par `make_brick_and_ic.py` sur une machine disposant du module python
adc (le module C++ n'est pas requis pour l'emission DSL, pure-python) :

```
ADC_CASES=/chemin/adc_cases PYTHONPATH=/chemin/adc_cpp/python \
  python docs/validation/make_brick_and_ic.py --ns 128 256 --out $WORK
```

Determinisme verifie (regeneration locale vs artefacts du run ROMEO, bit-pour-bit) :

```
md5(ic_128.raw)          = da245ba8934546986508976a64156d2e   (regenere == ROMEO)
md5(hyqmom15_brick.hpp)  = d785b13ac0da1dd349ff4775368c8ff2   (--ns 128 256, 1940 lignes, == ROMEO)
md5(include/.../dense_eig.hpp) = 86fb1cbbec0e265cd255559434ce83c6   (worktree == ROMEO ~/adc_cpp)
```

Le binaire device n'est plus stage a la main : `parity181.sbatch` recompile les DEUX
variantes (device nvcc_wrapper / hote g++ Serial) depuis les MEMES sources versionnees
stagees dans `$WORK`, sur le noeud GH200. Seules la brique et l'IC (regeneres) doivent etre
deposes dans `$WORK` au prealable (cf. entete du script).

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
noeud GH200, par `parity181.sbatch`, depuis les MEMES sources versionnees stagees dans
`$WORK` (plus aucun binaire stage a la main) :

* device : `nvcc_wrapper` du `kinstall` GPU, `DefaultExecutionSpace=Cuda` ;
* hote   : `g++` contre un Kokkos Serial aarch64 construit sur le noeud,
  `DefaultExecutionSpace=Serial`.

Sous Serial, `DefaultExecutionSpace == DefaultHostExecutionSpace` : la garde `if constexpr`
de `for_each_cell` (#165) prend la boucle hote sequentielle sur les petites boites (niveaux
grossiers du V-cycle) la ou le device garde `parallel_for` ; aucune dependance
inter-iteration, donc bit-identique attendu hors reductions. `real_eig_minmax` est le MEME
code source `ADC_HD` instancie host d'un cote, device de l'autre.

20 pas du MEME etat initial `ic_128.raw`, dump binaire de l'etat final (15 moments + phi),
compares par `compare_snap`. Run initial job 654862 ; REPRODUIT byte-pour-byte par le
`parity181.sbatch` versionne (job 654998, romeo-a057, les DEUX variantes recompilees depuis
les sources du depot, brique + IC regeneres) :

```
=== RUN DEVICE (Cuda) 20 pas ===  [fin] 20 pas, t=0.01594, derive de masse 1.63e-13
=== RUN HOST (Serial) 20 pas ===  [fin] 20 pas, t=0.01594, derive de masse 1.64e-13
=== COMPARE snap_000020.raw (device vs host) ===
n_a=128 t_a=0.015936559201338053 k_a=20
n_b=128 t_b=0.015936559201338029 k_b=20
dt_clock (|t_a-t_b|) = 2.429e-17
payload doubles      = 262144   (15*128^2 moments + 128^2 potentiel)
bit-identiques       = 27710 (10.5705%)
max |a-b|            = 3.450573e-13   (index 256884 -> maximum dans le potentiel phi)
max rel              = 6.456023e-10
```

Les `t` complets (17 chiffres) montrent l'ecart d'horloge `...338053` vs `...338029` : c'est
l'evidence ~1 ULP citee plus bas, invisible dans le `%.3e` de `dt`.

Lecture :

* **Chemin `dense_eig` : accord a la precision imprimee (~1 ULP), pas une preuve bit-exacte.**
  Le pas de temps `dt` (colonne `dt` de `growth.csv`) est issu de `step_cfl` = CFL sur le MAX
  des bornes de vitesses rendues par `real_eig_minmax` cellule par cellule. La reduction MAX
  est exacte quel que soit l'ordre : a entree identique, le QR `ADC_HD` rend des bornes
  identiques device/hote (c'est la propriete algorithmique du chemin eigen, deterministe et
  de MEME source host/device). Mais ce qu'on OBSERVE n'est qu'un accord a la precision
  IMPRIMEE : `growth.csv` n'ecrit `dt` qu'en `%.3e` (4 chiffres) et les modes `a_l` qu'en
  `%.6e` (7 chiffres), pas les 16 chiffres d'un double. L'horloge cumulee `t` differe de
  2.4e-17 apres 20 pas (rel 1.5e-15, ~1 ULP) : c'est la preuve DIRECTE que les `dt` ne sont
  PAS tous strictement bit-identiques (sinon, sommes dans le meme ordre, `t` serait egal au
  bit). L'origine est la retroaction phi -> source (bullet suivant) qui perturbe au niveau
  bruit l'etat lu par `real_eig_minmax` des le pas 2. Sur le mode physique du diocotron, `a4`
  (l=4, ~5.94e-2) coincide sur les 7 chiffres imprimes device/hote ; les petits modes de
  bruit `a2/a3/a5/a6` (~1e-16) different deja dans le CSV imprime. Formulation juste : accord
  ~1 ULP sur l'observable, pas une egalite bit-a-bit.
* **Ecarts DIFFUS au niveau bruit machine, MAXIMUM dans le potentiel multigrille.** Le
  comparateur ne donne que 27710/262144 = 10.57% de doubles bit-identiques : ~89% du payload
  differe, champs de moments COMPRIS (cf. `a2/a3/a5/a6` ci-dessus). Les ecarts sont donc
  DIFFUS, pas confines a `phi`. Le MAXIMUM `|a-b|` = 3.45e-13 tombe, lui, a l'index 256884
  (zone `phi` >= 245760). Origine commune : le V-cycle Poisson a un critere de residu en
  reduction SOMME, dont la reassociation `parallel_for` (device) vs boucle serie (hote) change
  le dernier bit ; le potentiel corrige se propage ensuite par le terme source electrique a
  TOUT l'etat (couplage phi -> source -> moments), d'ou la diffusion. Ecart absolu 3e-13 sur
  des champs d'ordre 1 a 1.7e3 : bruit machine. `max rel` 6.5e-10 porte sur une entree de
  magnitude quasi nulle (relatif gonfle). Comportement attendu des reductions paralleles : le
  solveur de vitesses (objet de ADC-181) n'est pas la SOURCE de la divergence, il propage
  fidelement l'etat qu'on lui donne.

## 3. Multi-GPU MPI (substrat halos, jobs 654863 puis 654999) et perimetre

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

Corroboration sur CE noeud, post-#254, du substrat halos multi-GPU (harnais versionne
`python/tests/gpu/gpu_amr_bz_mpi_validate.cpp`, B_z par niveau AMR multi-box distribue, un
GH200 par rang). Re-execute par `mpi181.sbatch` (job 654999, romeo-a057), chiffres identiques
au run initial 654863 :

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
50 pas, job 654998, meme run que la parite ci-dessus) :

```
device(Cuda) : 50 pas en  2.664 s -> 0.0533 s/pas (18.8 pas/s)
host(Serial) : 50 pas en 35.212 s -> 0.7042 s/pas ( 1.4 pas/s)
```

Le device est ~13.2x plus rapide qu'un coeur Grace seul a n=128 (comparaison
throughput, pas un cout de divergence isole : le device emploie le GPU entier, l'hote
un coeur). Aucun ralentissement pathologique : la boucle QR par cellule ne fait pas
diverger le pas device, et le run de 24706 pas (section 1) confirme la stabilite.

Le cap d'iterations EISPACK (30/bloc) et le repli Gershgorin bornent le cout par cellule :
la divergence reste celle, benigne, d'un solveur dense minuscule par thread.

## Conclusion

* `dense_eig` (`real_eig_minmax`, Hessenberg+QR `ADC_HD`) EXECUTE et CORRECT sur GH200
  device a travers le `.so` hyqmom15 (`exact_speeds`, `hll`) : section 1 (24706 pas) +
  section 2 (parite hote/device meme noeud, accord ~1 ULP sur l'observable, ecarts diffus au
  niveau bruit machine maximaux dans le potentiel multigrille).
* substrat multi-GPU MPI re-confirme bit-identique (max) / dernier ulp (sommes) sur le
  noeud post-#254 : section 3.
* sources de validation entierement versionnees (`docs/validation/`), brique + IC regeneres
  bit-pour-bit par `make_brick_and_ic.py`, les deux variantes (device/hote) recompilees par
  `parity181.sbatch` depuis ces sources.

ADC-181 RESTE OUVERTE : cette note (et la PR associee) AVANCE l'issue, elle ne la ferme pas.
Reste a livrer pour la fermer : un driver hyqmom15 MULTI-BOITE + MPI dedie (soit `AmrSystem`
multi-box cablant le composite `Hyqmom15Hyp/Src/Ell` + Poisson, soit une decomposition MPI de
`System`). Le multi-GPU specifique hyqmom15 n'est ici couvert que par SUBSTRAT (harnais
`gpu_amr_bz_mpi_validate`, modele B_z, #59/#254) + l'argument de localite cellule de
`real_eig_minmax`, pas par un run hyqmom15 multi-rang direct.

# Tutoriel A->Z

Ce tutoriel mene une simulation diocotron complete, de `git clone` jusqu'aux figures, au GIF
et a la comparaison uniforme/AMR. Tout le code montre ci-dessous provient d'un script unique et
reproductible, [`diocotron_tutorial.py`](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py) : la doc l'inclut
par `literalinclude` (le code n'est jamais recopie a la main). Le script est autonome ; il ne
depend que de `adc`, `numpy` et `matplotlib`, pas de `adc_cases`, et s'execute ainsi :

```bash
python docs/sphinx/tutorials/diocotron_tutorial.py            # --n 96 --steps 60
python docs/sphinx/tutorials/diocotron_tutorial.py --quick    # passage de fumee rapide
```

:::{admonition} Physique : modele reduit
:class: note
Une seule densite `n`, advectee par la derive E x B `v = (-d_y phi / B0, d_x phi / B0)` (a
divergence nulle), ou `phi` resout le Poisson de systeme `-lap phi = alpha (n - n_i0)`. C'est le
benchmark de normalisation du diocotron, pas une reproduction du systeme Euler-Poisson
complet. Voir les [limites honnetes](#limites-honnetes) en fin de page.
:::

## Etape 1 : Cloner le depot

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
```

## Etape 2 : Dependances

- Compilateur C++23 (AppleClang 16+, GCC 13+, Clang 17+), CMake >= 3.20, Ninja.
- Python >= 3.10 avec `numpy` (et `matplotlib` pour les figures du tutoriel).
- Catch2 / pybind11 sont recuperes automatiquement par CMake.

Detail et options : [Installation](installation.md).

## Etape 3 : Build du module Python

Le coeur est header-only ; pour ce tutoriel on ne construit que l'extension Python `_adc`,
sans la suite de tests C++. Le build tient alors en quelques minutes :

```bash
cmake -S . -B build-py -G Ninja \
  -DADC_BUILD_PYTHON=ON \
  -DADC_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$(which python3.12)
cmake --build build-py --target _adc -j$(sysctl -n hw.logicalcpu)
```

`-DADC_BUILD_TESTS=OFF` saute la suite Catch2 et `--target _adc` ne compile que l'extension. Le
build complet (coeur + tests, pour contribuer) est dans [Installation](installation.md).

## Etape 4 : Variables d'environnement

```bash
export PYTHONPATH=$PWD/build-py/python
export ADC_INCLUDE=$PWD/include
export ADC_CACHE_DIR=$PWD/.adc_cache
```

- `PYTHONPATH` : importer le paquet `adc` construit. Le build y depose `__init__.py`, `dsl.py`
  et l'extension `_adc`, donc ce seul chemin suffit.
- `ADC_INCLUDE` : laisser le DSL compiler ses `.so` contre les en-tetes du depot (backend
  `production`).
- `ADC_CACHE_DIR` : garder les `.so` generes par le DSL en cache pour les relances.

L'extension `_adc` porte le suffixe ABI de l'interpreteur qui l'a construite (`cpython-312`) :
importez `adc` avec ce meme `python3.12`, `numpy` installe, sinon `ModuleNotFoundError: adc._adc`
(cf. [le piege interpreteur](installation.md)).

## Etape 5 : Importer et detecter le backend

Le script importe `adc` et le DSL, puis affiche le backend execute (serie pour un
module Python ; cf. [Verifier son backend](backend.md)).

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:lines: 50-53
```

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: detect_backend_runtime
```

## Etape 6 : Les parametres physiques du modele

Deux constantes pilotent le modele reduit ; elles doivent rester coherentes entre les formules
du flux et le second membre du Poisson.

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:lines: 55-60
```

## Etape 7 : Ecrire le modele en formules (DSL) et le compiler

On ecrit le modele symboliquement avec `adc.dsl.Model` : la variable conservative `n`, les champs
auxiliaires `phi` / `grad_x` / `grad_y` fournis par le solveur, le flux d'advection E x B, les
valeurs propres, et le second membre elliptique `alpha (n - n_i0)`. `m.check()` verifie que toute
variable referencee est declaree.

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: diocotron_model
```

Puis on compile le modele en `.so` et on le branche : le script tente d'abord le backend
`production` (chemin natif zero-copie, prefere en MPI/AMR), puis retombe sur `aot` (numeriquement
identique, marshale cote hote), comme dans les cas applicatifs. Le defaut de
`m.compile(...)` est `aot` ; `production` exige que `_adc` et le `.so` aient ete compiles avec les
memes en-tetes adc (garde d'ABI). C'est aussi ici qu'on choisit le schema spatial (volumes finis,
limiteur minmod, flux de Rusanov), le temps (explicite) et le Poisson de systeme.

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: compile_and_build
```

## Etape 8 : Construire le System

`adc.System(n=, L=, periodic=True)` cree le coupleur ; `add_equation` aiguille sur le type du
modele (un `CompiledModel` part sur l'adder du backend). Tout cela est cable dans
`compile_and_build` ci-dessus.

## Etape 9 : La condition initiale

Une bande horizontale de charge, perturbee sinusoidalement le long de `x` (mode azimutal 2) :
c'est ce qui porte l'instabilite. Convention `ne[j, i]` (indexing numpy `'xy'`), tableau contigu.

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: band_density
```

La densite est posee via `sim.set_density("ne", ne0)` (dans `compile_and_build`), apres avoir
fixe le fond ionique neutralisant `n_i0 = ne0.mean()` (solubilite du Poisson periodique).

## Etape 10 : Volumes finis, temps, Poisson

Ces trois choix sont passes a `add_equation` / `set_poisson` (etape 7) :

- spatial : `adc.FiniteVolume(limiter="minmod", riemann="rusanov")`, reconstruction MUSCL
  minmod + flux de Riemann de Rusanov ;
- temps : `adc.Explicit()` ;
- Poisson : `sim.set_poisson(rhs="charge_density", solver="geometric_mg")`, second membre =
  densite de charge, solveur multigrille geometrique.

## Etape 11 : Integrer en temps

On avance `steps` pas a CFL fixe (`sim.step_cfl(cfl)`), en capturant des trames, l'instant et
l'amplitude L2 de la perturbation au fil du temps.

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: run
```

## Etape 12 : Diagnostics

L'amplitude de la perturbation est l'ecart a la moyenne le long de `x` (la bande non perturbee
est uniforme en `x` ; ce qui en devie porte l'instabilite). Le script verifie en fin de run que
l'instabilite a cru et que la masse est conservee (transport advectif periodique).

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: perturbation_amplitude
```

## Etape 13 : La courbe de croissance

`make_figures` trace l'amplitude (echelle log) en fonction du temps, a cote de la densite finale.

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: make_figures
```

![Croissance de l'instabilite diocotron : amplitude L2 (echelle log) vs temps, et densite finale.](../tutorials/_assets/diocotron_growth.png)

## Etape 14 : Le GIF

La meme fonction `make_figures` assemble l'evolution de la densite en GIF (et une image PNG de
couverture pour les exports statiques).

![Evolution temporelle de la densite du diocotron (animation).](../tutorials/_assets/diocotron.gif)

*Image de couverture statique (la densite finale), affichee la ou le GIF ne s'anime pas,
exports PDF/print :*

![Densite finale du diocotron (image de couverture PNG).](../tutorials/_assets/diocotron_cover.png)

## Etape 14bis : La meme physique, deux fronts (briques == DSL)

Le modele a ete ecrit ici en formules (`adc.dsl.Model`, Etape 7). Mais le coeur sait aussi
composer un modele a partir de briques natives : `adc.Model(state, transport, source, elliptic)`.
Les deux fronts d'ecriture sont interchangeables : ce sont deux facons de decrire la meme
physique, et elles produisent un noyau numerique identique. On l'ecrit en briques :

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: native_diocotron_model
```

puis on rejoue la meme grille / le meme schema / le meme nombre de pas, et on compare l'etat final
des deux fronts :

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: native_vs_dsl
```

L'ecart est nul a la precision binaire (`max|briques - DSL| = 0`, `np.array_equal`) : les
formules DSL reproduisent exactement les conventions des briques `ExBVelocity` et `BackgroundDensity`.
Une divergence (meme $10^{-15}$) trahirait une formule fausse (signe de la derive, borne d'onde,
second membre). Le catalogue complet des briques est dans la
[reference des briques](../reference/bricks_reference.md), et celui du DSL dans la
[reference du DSL](../reference/dsl_reference.md) ; le cas applicatif `tutorial/` de `adc_cases`
pousse la demonstration a trois fronts (helper specialise inclus).

![Densite finale : la meme physique en briques natives (gauche) et en formules DSL (droite) ; ecart maximal nul.](../tutorials/_assets/diocotron_native_vs_dsl.png)

## Etape 15 : Uniforme vs AMR

On rejoue la meme physique sur une grille uniforme (`adc.System`) et sur une hierarchie raffinee
(`adc.AmrSystem`), avec exactement le meme modele compose en briques natives. `AmrSystem` raffine
la ou la densite depasse un seuil (`set_refinement(0.05)`) ; la cadence de regrid est portee par
`AmrSystemConfig.regrid_every`. Les deux densites finales sont tracees cote a cote, avec l'ecart
maximal en titre.

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: uniform_vs_amr
```

![Densite finale : grille uniforme (gauche) vs hierarchie AMR raffinee (droite).](../tutorials/_assets/diocotron_uniform_vs_amr.png)

## Etape 16 : Kokkos OpenMP (parallelisme CPU)

Le module Python distribue tourne en serie parce que la CI le construit sans Kokkos, pas parce que
Python serait bride : `import adc` pilote, et le calcul par cellule herite du backend compile dans
`adc` (voir [Verifier son backend](backend.md)). Pour exploiter plusieurs coeurs, c'est donc la
facade C++ qu'on recompile avec le backend Kokkos (device OpenMP), pas un drapeau de script :

```bash
cmake -S . -B build-kokkos -DADC_USE_KOKKOS=ON \
      -DKokkos_ENABLE_OPENMP=ON
cmake --build build-kokkos -j
ctest --test-dir build-kokkos --output-on-failure
```

Le seam `for_each_cell` bascule alors sur l'espace d'execution OpenMP de Kokkos pour tout ce qui
lie `adc`. La CI valide ce backend (job ci-full, `Kokkos_ENABLE_OPENMP=ON`). Le backend OpenMP
autonome (`-DADC_USE_OPENMP=ON`) existe mais est deprecie au profit de Kokkos.

## Etape 17 : MPI (parallelisme distribue)

De meme, le distribue s'obtient a la compilation, et se lance via `mpirun` :

```bash
cmake -S . -B build-mpi -DADC_USE_MPI=ON
cmake --build build-mpi -j
ctest --test-dir build-mpi --output-on-failure       # rejoue np=1/2/4 via mpirun
```

`comm.hpp` passe alors par `MPI_Comm_rank/size` + collectives. MPI et Kokkos se combinent (un GPU
par rang) pour le GPU. Le GPU lui-meme exige ROMEO : `-DADC_USE_KOKKOS=ON` +
`Kokkos_ARCH_HOPPER90` + `nvcc_wrapper`, valide manuellement sur GH200 (jamais en CI). Voir
[Verifier son backend](backend.md) et [`GPU_ROMEO.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/GPU_ROMEO.md).

(limites-honnetes)=

## Etape 18 : Limites honnetes

- **Modele reduit.** Ce tutoriel transporte une densite par la derive E x B couplee a un Poisson
  scalaire (`alpha (n - n_i0)`). Ce n'est pas le systeme Euler-Poisson complet (pas
  d'equation de quantite de mouvement, pas d'energie), et ce n'est pas une reproduction de la
  configuration de Hoffart. C'est le benchmark de normalisation du diocotron. La fidelite au
  systeme complet est discutee dans [`HOFFART_FIDELITY.md`](https://github.com/wolf75222/adc_cpp/blob/master/docs/HOFFART_FIDELITY.md) ; les
  scenarios complets vivent dans `adc_cases` (cf. [Organisation des depots](organisation.md)).
- **Backend serie.** Le run Python est serie (cf. etapes 16-17 et [backend](backend.md)) ; les
  figures sont produites a basse resolution pour rester rapides et reproductibles.
- **Comparaison AMR indicative.** `uniform_vs_amr` illustre l'usage de `AmrSystem` ; l'ecart
  uniforme/AMR rapporte en titre mesure la coherence, pas une etude de convergence.

Chaque asset est accompagne d'un enregistrement de provenance
([`_assets/provenance.json`](../tutorials/_assets/provenance.json) : SHA `adc_cpp`, backend,
resolution, commande) pour la reproductibilite.

## Le script complet

Pour reference, le script integral, dans son orchestration :

```{literalinclude} ../tutorials/diocotron_tutorial.py
:language: python
:pyobject: main
```

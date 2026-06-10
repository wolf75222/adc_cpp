<div align="center">

# ADC CPP

Coeur C++23 d'un solveur AMR / MPI / GPU pour systemes hyperbolique-elliptique couples.

![Tests](https://img.shields.io/badge/tests-C%2B%2B%20serie%20%2B%20Kokkos%20%2B%20MPI%20%2B%20Python-brightgreen)

</div>

<p align="center">
  <img src="docs/anim_romeo_diocotron_amr3.gif" alt="Instabilite diocotron AMR 3 niveaux sur ROMEO" width="640">
  <br><sub>3 niveaux AMR sur ROMEO (moteur C++ multi-niveaux). Version locale reproductible (facade
  Python, 1 niveau fin) : <a href="https://github.com/wolf75222/adc_cases/tree/master/diocotron_amr"><code>adc_cases/diocotron_amr</code></a> (<code>make_hero_gif.py</code>).</sub>
</p>

<div align="center">
<sub>
Instabilite diocotron (derive E x B) sur AMR 3 niveaux emboites, ROMEO (x64cpu, 96 coeurs AMD EPYC).
Patchs fins suivis par regrid Berger-Rigoutsos, sous-cyclage Berger-Oliger et reflux conservatif (derive de masse ~ 1e-15).
</sub>
</div>

---

`adc_cpp` est la bibliotheque : le moteur generique (coeur sans modele), une bibliotheque de briques
physiques (`include/adc/physics/`) et les bindings Python (`adc`, composition `System` / `AmrSystem`).
Le coeur ne nomme aucun scenario ; il fournit des briques generiques composees en `CompositeModel`. Les
scenarios nommes (diocotron, Euler-Poisson, deux-fluides) vivent dans le depot
[`adc_cases`](https://github.com/wolf75222/adc_cases).

Le coeur resout, sur maillage cartesien adaptatif :

```
d U / d t  +  div F(U, aux)  =  S(U, aux)
D phi = f(U)
```

ou la partie hyperbolique `U` et la partie elliptique `phi` sont couplees a chaque pas via le canal
`aux`. Le contrat de base est `(phi, grad_x, grad_y)` ; un modele peut declarer `n_aux` pour lire des
champs supplementaires (`B_z`, `T_e`), avec retro-compat bit-exacte si `n_aux=3`.

## Ce que fournit le coeur

Concepts, seams, operateurs, integrateurs, solveur elliptique, moteur AMR distribue et facades runtime :
voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (section 6 pour la carte des modules, section 13 pour
l'arborescence par fichier). Algorithmes et formules : [docs/ALGORITHMS.md](docs/ALGORITHMS.md). Profil :
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Ecrire un modele : briques ou formules

Un modele decrit une equation (flux, source, valeurs propres, second membre elliptique). Le module `adc`
permet d'ecrire un modele de deux facons, qui produisent le meme objet cote C++ et se branchent de la meme maniere
sur `adc.System` / `adc.AmrSystem` :

- briques : on compose des briques generiques deja compilees (`adc.Model`), sans compilation a la volee.
- formules (DSL) : on ecrit le modele en formules symboliques (`adc.dsl.Model`), traduites en C++ puis
  compilees en `.so`.

Reference complete : [bricks_reference](docs/sphinx/reference/bricks_reference.md),
[dsl_reference](docs/sphinx/reference/dsl_reference.md) ; tutoriel pas a pas :
[docs/sphinx/getting_started/tutorial.md](docs/sphinx/getting_started/tutorial.md).

### Briques (`adc.Model`)

`adc.Model(state, transport, source, elliptic)` compose quatre briques et renvoie une `ModelSpec` (des
tags lus cote C++ par la fabrique de modeles). Catalogue par slot :

| Slot | Briques |
|---|---|
| `state` | `adc.Scalar()` ; `adc.FluidState(kind="compressible", gamma=)` ; `adc.FluidState(kind="isothermal", cs2=)` |
| `transport` | `adc.ExB(B0=)` ; `adc.CompressibleFlux()` ; `adc.IsothermalFlux()` |
| `source` | `adc.NoSource()` ; `adc.PotentialForce(charge=)` ; `adc.GravityForce()` |
| `elliptic` | `adc.BackgroundDensity(alpha=, n0=)` ; `adc.ChargeDensity(charge=)` ; `adc.GravityCoupling(sign=, four_pi_G=, rho0=)` |

`adc.Model(...)` valide l'appariement etat / transport (`Scalar` avec `ExB` ; `FluidState` compressible
avec `CompressibleFlux` ; isotherme avec `IsothermalFlux`) ; un appariement incoherent leve une
`ValueError`. Exemple, le diocotron reduit (densite scalaire advectee par la derive ExB, fond
neutralisant) :

```python
import adc
model = adc.Model(state=adc.Scalar(),
                  transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(),
                  elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)          # ne0 : tableau 2D (densite initiale)
sim.step_cfl(0.4)
```

### Formules (`adc.dsl.Model`)

`adc.dsl.Model` ecrit le meme modele en formules : variables conservatives, primitives, flux, valeurs
propres, source et contribution elliptique. Le DSL emet du C++, le compile en `.so` et renvoie un
`CompiledModel`. Le meme diocotron en formules :

```python
import adc
from adc import dsl

B0, ALPHA = 1.0, 1.0

def diocotron_model(n_i0):
    m = dsl.Model("diocotron")
    (n,) = m.conservative_vars("n")        # densite scalaire transportee
    m.aux("phi")
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")
    vx = (-grad_y) / B0                     # derive E x B
    vy = grad_x / B0
    m.flux(x=[n * vx], y=[n * vy])
    m.eigenvalues(x=[vx], y=[vy])
    m.primitive_vars(n=n)
    m.conservative_from([n])
    m.elliptic_rhs(ALPHA * (n - n_i0))      # = adc.BackgroundDensity(alpha, n0)
    m.check()
    return m

compiled = diocotron_model(n_i0).compile(backend="production")   # mis en cache par hash
sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_equation("ne", model=compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)
sim.step_cfl(0.4)
```

Le backend conseille est `backend="production"` (chemin natif zero-copie ; GPU np=1 GH200 #97 et MPI
solve_fields np=1/2/4 #93/#99 valides ; le transport production GPU+MPI multi-rang n'est pas encore
exerce, voir [docs/VALIDATION.md](docs/VALIDATION.md)). Les quatre chemins de modele (natif compose,
production, aot, prototype) et leurs limites : [docs/DSL_MODEL_DESIGN.md](docs/DSL_MODEL_DESIGN.md).

### Briques et formules coincident

Briques et formules decrivent le meme modele de deux facons. Sur le diocotron, la sortie DSL est
bit-identique a la composition de briques (cas
[`adc_cases/diocotron_dsl`](https://github.com/wolf75222/adc_cases/tree/master/diocotron_dsl)), et le
tutoriel parcourt les deux. Le branchement differe par l'adder : `sim.add_block(...)` prend une
`ModelSpec` native (briques), `sim.add_equation(...)` aiguille un `CompiledModel` (DSL `.so`) vers
l'adder natif, aot ou prototype selon son backend.

## Systemes multi-especes

On couple N especes (ions, electrons, neutres), chacune avec son `PhysicalModel`, son schema spatial et
sa politique temporelle. Les interactions vivent dans le second membre elliptique (`ChargeDensityRhs`) et
dans la source (`CoupledSource`), jamais dans le flux. Le scheduler gere sous-pas, cadence, IMEX partiel,
pas adaptatif multirate (`step_adaptive`) et un integrateur ecrit a la main. Detail :
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (section 5).

## Backends CMake

```bash
cmake --preset serial                # serie (ou : cmake -B build)
cmake --preset mpi                   # distribue (halos + FFT par MPI)
cmake --preset parallel              # Kokkos CPU (env conda actif)
cmake -B build-gpu -DADC_USE_KOKKOS=ON \
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K   # GPU GH200
```

Chaque option s'accepte aussi en variable d'environnement (`ADC_USE_KOKKOS=ON cmake ...` ou
`pip install .`) ; un `-D` explicite garde la priorite. Kokkos est le backend de dispatch
conseille (CPU et GPU, un seul code) ; l'OpenMP autonome (`ADC_USE_OPENMP`) est deprecie.
Detail : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (section 9).

## Utiliser le coeur

```cmake
include(FetchContent)
FetchContent_Declare(adc_cpp GIT_REPOSITORY https://github.com/wolf75222/adc_cpp.git)
FetchContent_MakeAvailable(adc_cpp)  # les tests d'adc_cpp ne sont PAS compiles chez le consommateur
                                    # (ADC_BUILD_TESTS suit PROJECT_IS_TOP_LEVEL)
target_link_libraries(mon_appli PRIVATE adc::adc)
```

On definit un type qui satisfait `PhysicalModel`, on l'instancie dans un `Coupler<Model, Elliptic>` (ou
`AmrCouplerMP` pour l'AMR), et on avance en temps.

## Module Python `adc`

`bash scripts/setup_env.sh` cree l'env conda (CMake, Ninja, NumPy, pybind11, Kokkos) et y
fige la meilleure toolchain de la plateforme (AppleClang sur macOS, gcc conda sur Linux --
prioritaire sur un PATH pollue). Ensuite, installation en une commande -- `pip install .`
pilote le CMakeLists via scikit-build-core, les backends se choisissent par variables
d'environnement :

```bash
pip install .                                                  # serie
ADC_USE_KOKKOS=ON Kokkos_ROOT=$CONDA_PREFIX pip install . -v  # parallele (Kokkos)
```

Puis, sans `PYTHONPATH` ni variable a exporter :

```python
import adc
adc.set_threads()          # tous les coeurs (ou set_threads(8)) ; avant le 1er System
sim = adc.System(n=256)    # adc.doctor() diagnostique l'environnement en cas de doute
```

Pour iterer sur le C++ sans reinstaller : `cmake --preset python` (serie) ou
`--preset python-parallel` (Kokkos), puis `PYTHONPATH=$PWD/build-py/python` (ou
`build-py-kokkos/python`). L'extension est epinglee a l'interpreteur qui l'a construite
(`cpython-312`) : construire et importer avec le meme python -- en cas d'erreur d'import, le
message indique la cause et la commande de reconstruction. Detail :
[installation](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/getting_started/installation.md).

L'ecriture d'un modele est decrite plus haut. Exemple plus complet, des electrons Euler compressibles
avec mur circulaire Dirichlet et source IMEX, en briques :

```python
import adc
sim = adc.System(n=192, periodic=False)
electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0),
                      elliptic=adc.ChargeDensity(charge=-1.0))
sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"), time=adc.IMEX(substeps=10))
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)
sim.step_cfl(0.4)
```

`adc.AmrSystem` compose un ou plusieurs blocs sur une hierarchie raffinee (API proche de `System`, plus
`set_refinement`). Il gere le mono- et le multi-bloc (capstone #195/#199/#205), la cadence de regrid
(`regrid_every`, `set_refinement` / `set_phi_refinement`), le reflux conservatif, la reconstruction
`conservative|primitive`, les flux `rusanov|hllc|roe` et une source IMEX locale (#132). Le Poisson est
resolu au niveau grossier puis injecte vers le fin (pas de solve elliptique composite ni de Schur global
sur AMR). Detail : [docs/DSL_MODEL_DESIGN.md](docs/DSL_MODEL_DESIGN.md).

## Ecosysteme

| Repo | Role | Socle maillage |
|---|---|---|
| `adc_cpp` (ce depot) | coeur hyperbolique-elliptique sur AMR + GPU/MPI/Kokkos | propre |
| [`adc_cases`](https://github.com/wolf75222/adc_cases) | applications : modeles, facades, exemples, Python | consomme `adc::adc` |
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | solveurs Poisson (Thomas, SOR, CG, DST, multigrille) | propre |
| [`pde_core_cpp`](https://github.com/wolf75222/pde_core_cpp) | infra partagee (mesh, fields, AMR) | propre |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection + Burgers + Chorin NS | `pde_core_cpp` |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | Euler 2D + viscous NS + sources plasma | `pde_core_cpp` |

## Build et tests

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

La CI a trois jobs : Release (serie), MPI (`-DADC_USE_MPI=ON`, bit-identiques np=1/2/4) et Kokkos
(Serial). Le module Python ajoute une suite (bindings et DSL). La CI ignore les changements documentaires
(`paths-ignore: docs/**`, `**.md`).

| Option | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | suite CTest du coeur |
| `ADC_USE_KOKKOS` | `OFF` | dispatch Kokkos (CPU OpenMP + GPU), conseille |
| `ADC_USE_OPENMP` | `OFF` | dispatch OpenMP autonome, deprecie (utiliser Kokkos) |
| `ADC_USE_MPI` | `OFF` | backend distribue (comm, halos, FFT) |
| `ADC_USE_HDF5` | `OFF` | DataWriter HDF5 parallele |

## Organisation du depot

```
include/adc/
  core/         types, etat, PhysicalModel, EquationBlock, CoupledSystem
  mesh/         MultiFab, BoxArray, Geometry, for_each_cell, CL physiques
  parallel/     seam comm (MPI), load balance
  physics/      briques generiques -> CompositeModel
  numerics/     reconstruction, flux numeriques, operateur spatial
  numerics/elliptic/  concept EllipticSolver, multigrille, FFT
  numerics/time/      SSP-RK, scheduler, IMEX, splitting, moteur AMR
  coupling/     Coupler, SystemCoupler, AmrSystemCoupler, AmrCouplerMP
  amr/          clustering BR, regrid, hierarchie
  runtime/      facades System / AmrSystem, model_factory, DSL, canal aux extensible
tests/          tests coeur (MPI via mpirun quand -DADC_USE_MPI=ON)
docs/           ARCHITECTURE.md, ALGORITHMS.md, GPU_RUNTIME_PORT.md, PERFORMANCE.md, ...
```

Detail par fichier : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (section 13).

## Validation

Resume de la couverture (CI, AMR conservatif, GPU GH200, Schur et polaire device, FFT sous MPI) :
[docs/VALIDATION.md](docs/VALIDATION.md). Matrice de couverture backend :
[docs/BACKEND_COVERAGE.md](docs/BACKEND_COVERAGE.md). Validations device en detail :
[docs/GPU_RUNTIME_PORT.md](docs/GPU_RUNTIME_PORT.md). Validation applicative (modeles, diocotron, ROMEO) :
[`adc_cases`](https://github.com/wolf75222/adc_cases).

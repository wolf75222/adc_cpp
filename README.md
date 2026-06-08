<div align="center">

# ADC CPP

**Coeur C++23 d'un solveur AMR / MPI / GPU pour systemes hyperbolique-elliptique couples.**

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
Patchs fins suivis par regrid Berger-Rigoutsos, sous-cyclage Berger-Oliger + reflux conservatif (derive de masse ~ 1e-15).
</sub>
</div>

---

`adc_cpp` est la **bibliotheque** : le moteur generique (coeur sans modele) plus une bibliotheque de
briques physiques (`include/adc/physics/`) et les **bindings Python de la lib**, le module `adc`
(composition `System` / `AmrSystem`). Le coeur est **agnostique au modele** : il ne nomme aucun
scenario, il ne fournit que des briques generiques composees en `CompositeModel`. Les scenarios nommes
(diocotron, Euler-Poisson, deux-fluides...) vivent dans le depot separe
**[`adc_cases`](https://github.com/wolf75222/adc_cases)**.

Le coeur resout, sur maillage cartesien adaptatif :

```
d U / d t  +  div F(U, aux)  =  S(U, aux)
D phi = f(U)
```

ou la partie hyperbolique `U` et la partie elliptique `phi` sont couplees a chaque pas via le canal
`aux`. Contrat de base `(phi, grad_x, grad_y)`, extensible : un modele declare `n_aux` pour lire des
champs supplementaires (`B_z`, `T_e`), avec retro-compat bit-exacte si `n_aux=3`.

## Ce que fournit le coeur

Concepts, seams, operateurs, integrateurs, solveur elliptique, moteur AMR distribue et facades
runtime : voir **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (section 6 : carte des modules par
type ; section 13 : arborescence detaillee fichier par fichier).
Algorithmes et formules : [docs/ALGORITHMS.md](docs/ALGORITHMS.md).
Profil : [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Ecrire un modele : deux fronts (briques OU formules)

Un modele decrit UNE equation (flux, source, valeurs propres, second membre elliptique). Le module
`adc` offre deux fronts d'ecriture interchangeables, qui produisent le meme objet calculatoire cote
coeur C++ et se branchent de la meme maniere sur `adc.System` / `adc.AmrSystem` :

- **briques** -- on COMPOSE des briques generiques deja compilees (`adc.Model`) ; aucune compilation
  a la volee, parite production totale.
- **formules (DSL)** -- on ECRIT le modele en formules symboliques (`adc.dsl.Model`), traduit en C++
  puis compile en `.so`.

Les deux fronts sont des compositions des MEMES briques generiques ; le coeur reste agnostique au
scenario (il ne nomme ni diocotron, ni Euler-Poisson, ni deux-fluides). Reference complete :
**[reference/bricks_reference](docs/sphinx/reference/bricks_reference.md)**,
**[reference/dsl_reference](docs/sphinx/reference/dsl_reference.md)** ; tutoriel pas a pas :
**[docs/sphinx/getting_started/tutorial.md](docs/sphinx/getting_started/tutorial.md)**.

### Front 1 -- briques (`adc.Model`)

`adc.Model(state, transport, source, elliptic)` compose quatre briques generiques et renvoie une
`ModelSpec` (des tags lus cote C++ par la fabrique de modeles). Catalogue des briques par slot, tel
qu'expose par `adc.*` (structs C++ dans `include/adc/physics/`) :

| Slot | Briques |
|---|---|
| `state` | `adc.Scalar()` ; `adc.FluidState(kind="compressible", gamma=)` ; `adc.FluidState(kind="isothermal", cs2=)` |
| `transport` | `adc.ExB(B0=)` ; `adc.CompressibleFlux()` ; `adc.IsothermalFlux()` |
| `source` | `adc.NoSource()` ; `adc.PotentialForce(charge=)` ; `adc.GravityForce()` |
| `elliptic` | `adc.BackgroundDensity(alpha=, n0=)` ; `adc.ChargeDensity(charge=)` ; `adc.GravityCoupling(sign=, four_pi_G=, rho0=)` |

`adc.Model(...)` valide l'appariement etat <-> transport (`Scalar` avec `ExB` ; `FluidState`
compressible avec `CompressibleFlux` ; isotherme avec `IsothermalFlux`) ; un appariement incoherent
leve une `ValueError`. Exemple -- le diocotron reduit (densite scalaire advectee par la derive ExB,
fond neutralisant) :

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

### Front 2 -- formules (`adc.dsl.Model`)

`adc.dsl.Model` ecrit le MEME modele en formules : on declare les variables conservatives, les
primitives, le flux, les valeurs propres, la source et la contribution elliptique ; le DSL emet du
C++, compile en `.so` et renvoie un `CompiledModel`. Le meme diocotron en formules :

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

Le backend RECOMMANDE est `backend="production"` (chemin natif zero-copie ; GPU np=1 GH200 #97 + MPI
solve_fields np=1/2/4 #93/#99 valides ; transport production GPU+MPI multi-rang pas encore exerce --
voir Validation). Les quatre chemins de modele (natif compose, production, aot, prototype) et les
limites honnetes : **[docs/DSL_MODEL_DESIGN.md](docs/DSL_MODEL_DESIGN.md)**.

### Les deux fronts coincident

Briques et formules sont deux front-ends interchangeables du meme modele : sur le diocotron, la
sortie DSL est bit-identique a la composition de briques (cas
[`adc_cases/diocotron_dsl`](https://github.com/wolf75222/adc_cases/tree/master/diocotron_dsl)), et le
tutoriel parcourt les deux fronts. Le branchement differe seulement par l'adder :
`sim.add_block(...)` prend une `ModelSpec` native (briques), `sim.add_equation(...)` aiguille un
`CompiledModel` (DSL `.so`) vers l'adder natif / aot / prototype selon son backend.

## Systemes multi-especes

On couple N especes (ions, electrons, neutres...), chacune avec son `PhysicalModel`, son schema
spatial, sa politique temporelle. Interactions dans le second membre elliptique (`ChargeDensityRhs`)
et dans la source (`CoupledSource`), jamais dans le flux. Le scheduler supporte sous-pas, cadence,
IMEX partiel, pas adaptatif multirate (`step_adaptive`), et integrateur fait maison.
Detail : **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (section 5 : couche temps et couplage).

## Backends CMake

```bash
cmake -B build                       # serie
cmake -B build -DADC_USE_MPI=ON      # distribue (halos + FFT par MPI)
cmake -B build -DADC_USE_KOKKOS=ON   # CPU multi-thread (device OpenMP), recommande
cmake -B build -DADC_USE_KOKKOS=ON \
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K   # GPU GH200
```

**Kokkos est le backend de dispatch recommande** (CPU ET GPU, un seul code). Le backend OpenMP
autonome (`ADC_USE_OPENMP`) est **deprecie**. Detail : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (section 9).

## Utiliser le coeur

```cmake
include(FetchContent)
FetchContent_Declare(adc_cpp GIT_REPOSITORY https://github.com/wolf75222/adc_cpp.git)
FetchContent_MakeAvailable(adc_cpp)
target_link_libraries(mon_appli PRIVATE adc::adc)
```

On definit un type qui satisfait `PhysicalModel`, on l'instancie dans un `Coupler<Model, Elliptic>`
(ou `AmrCouplerMP` pour l'AMR), et on avance en temps.

## Module Python `adc`

Construction : `cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc -j`.

> L'extension compilee est **epinglee a l'interpreteur** (`_adc.cpython-312`). `import adc`
> ne fonctionne QUE sous l'interpreteur correspondant (p.ex. un Python 3.12 anaconda/conda qui a
> AUSSI numpy), avec le dossier `python/` du build sur `sys.path` (`build-py/python` ou
> `build-master/python`). Sous le `python3` systeme il echoue avec `ModuleNotFoundError: adc._adc`.

L'ecriture d'un modele (briques `adc.Model` ou formules `adc.dsl.Model`) est decrite plus haut,
section "Ecrire un modele : deux fronts". Un exemple plus complet -- electrons Euler compressibles,
mur circulaire Dirichlet, source IMEX -- assemble en briques :

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

`adc.AmrSystem` compose un ou plusieurs blocs sur une hierarchie raffinee (API proche de `System`
plus `set_refinement`). Etat actuel d'`AmrSystem` : mono- ET multi-bloc (`add_native_block` /
`add_compiled_block` repetes, capstone #195/#199/#205, cadence de regrid via `regrid_every` +
`set_refinement`/`set_phi_refinement`), reflux conservatif, recon
`conservative|primitive`, flux `rusanov|hllc|roe`, IMEX source locale (Gap 2 #132,
backward_euler_source / mf_apply_source_treatment). Reste hors-perimetre : pas de Schur global sur
AMR. `AmrSystem.potential()` binding SHIPPE (python/bindings.cpp:332, `#135`).
Detail des adders et chemins avances : **[docs/DSL_MODEL_DESIGN.md](docs/DSL_MODEL_DESIGN.md)**.

## Ecosysteme

| Repo | Role | Socle maillage |
|---|---|---|
| **`adc_cpp`** (ce depot) | coeur hyperbolique-elliptique sur **AMR** + GPU/MPI/Kokkos | propre (from scratch) |
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
(Serial). Module Python : suite supplementaire (bindings + DSL). CI ignore les changements
documentaires (`paths-ignore: docs/**`, `**.md`).

| Option | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | suite CTest du coeur |
| `ADC_USE_KOKKOS` | `OFF` | dispatch Kokkos (CPU OpenMP + GPU), **recommande** |
| `ADC_USE_OPENMP` | `OFF` | dispatch OpenMP autonome, **deprecie** (utiliser Kokkos) |
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

## Validation (coeur)

- CI : ctests coeur Release + Kokkos (Serial) ; MPI np=1/2/4 bit-identiques ; module Python (bindings + DSL).
- AMR conservatif : reflux multi-patch a l'arrondi machine (~1e-15).
- GPU GH200 (hors CI) : System production np=1 valide (#97) ; multigrille geometrique device-MPI
  np=1/2/4 valide (#93) ; AmrSystem + MPI + GPU valides bit-identiques (phase 10, dmax=0, #105) ;
  Schur/polaire device : **7/7 device-clean Kokkos Cuda single-GPU + MPI+Kokkos Cuda multi-GPU
  rank-invariant (10 tests, #157) + Kokkos OpenMP CI (#155)** -- condensed_schur, polar_transport,
  lorentz, full_tensor, polar_poisson, krylov, schur_condensation (tous device-clean GH200,
  compute-sanitizer 0 err). Les 4 echecs initiaux etaient TEST-SIDE (foncteurs hote / pointeurs hote
  appeles dans des kernels device, ou lecture hote d'une sortie async sans fence), corriges #150/#152/#158 ;
  le LIBRARY elliptique/Schur/polaire est device-correct.
  Detail : docs/BACKEND_COVERAGE.md.
- FFT sous `System` MPI np>1 : REFUSEE proprement (#106, plus de segfault) ; `DistributedFFTSolver`
  existe et teste a part, mais n'est PAS route dans `System`.

Detail des validations device : [docs/GPU_RUNTIME_PORT.md](docs/GPU_RUNTIME_PORT.md).
Matrice de couverture backend : [docs/BACKEND_COVERAGE.md](docs/BACKEND_COVERAGE.md).
Validation applicative (modeles, diocotron, ROMEO) : [`adc_cases`](https://github.com/wolf75222/adc_cases).

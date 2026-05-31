<div align="center">

# ADC CPP

**Solveur C++23 pour systemes hyperbolique-elliptique couples sur AMR block-structured (multi-niveaux + multi-patch), dispatch unique serie / OpenMP / Kokkos (GPU) et MPI distribue. Cas fil rouge : l'instabilite diocotron (derive E x B) du papier Hoffart (arXiv:2510.11808).**

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue?logo=cplusplus)
![Tests](https://img.shields.io/badge/tests-60%20C%2B%2B%20%2B%2013%20MPI%20%2B%20python-brightgreen)
![Build](https://img.shields.io/badge/build-CMake%203.20%2B-064F8C?logo=cmake)
![GPU](https://img.shields.io/badge/GPU-Nvidia%20GH200%20(Kokkos%2FCUDA)-76B900?logo=nvidia)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20ROMEO-lightgrey)

</div>

<p align="center">
  <img src="docs/anim_diocotron_amr3.gif" alt="Instabilite diocotron sur AMR 3 niveaux" width="640">
</p>

<div align="center">
<sub>
Instabilite diocotron (derive E x B) sur AMR 3 niveaux emboites. Une bande de charge
cree un ecoulement cisaille, instable, qui s'enroule en "cat's eyes". Les patchs fins
(cadres cyan) suivent les zones de fort gradient par regrid dynamique. Densite n_e
transportee par v = (E x B)/B^2, phi resolu par multigrille a chaque etage SSPRK2,
sous-cyclage Berger-Oliger + reflux conservatif aux interfaces coarse-fine (derive de
masse ~ 1e-15). Le pas couple est porte par le composant <code>AmrCoupler</code> sur
la pile MultiFab.
Reproduction :
<code>./build/bin/diocotron_amr3 out && python3 scripts/make_diocotron_amr3_gif.py out docs/anim_diocotron_amr3.gif</code>.
</sub>
</div>

---

ADC resout, sur maillage cartesien adaptatif, la forme generale

```
d U / d t  +  div F(U, phi)  =  div H(U, grad U)  +  S(U, phi)
D phi = f(U)
```

ou la partie hyperbolique (U) et la partie elliptique (phi) sont couplees a chaque
pas. Cas de validation fil rouge : l'instabilite **diocotron** (Hoffart et al.,
arXiv:2510.11808), puis le **deux-fluides isotherme** plasma.

## Solveurs

| Module | Role | Detail |
|---|---|---|
| [`model::Diocotron`](include/adc/model/diocotron.hpp) | derive E x B (vorticite reduite, scalaire) | flux advectif, `elliptic_rhs = alpha (n_e - n_i0)` |
| [`model::Euler`](include/adc/model/euler.hpp) | Euler compressible (γ = 1.4, 4 var) | validé free-stream + tourbillon isentropique (ordre 1.86) |
| [`model::EulerPoisson`](include/adc/model/euler_poisson.hpp) | Euler couple Poisson : gravite OU plasma (`InteractionKind`) | source g = -grad phi, un seul signe ; Jeans (0.1%) et Bohm-Gross + Coulomb (0.1%) valides |
| [`model::LangmuirMode`, `TwoFluidLinear`](include/adc/model/two_fluid_isothermal.hpp) | noyaux 0D AP (Ä = K A) | 2 branches plasma (Langmuir + ion-acoustique), Vieta exact |
| [`operator::{RusanovFlux,HLLFlux,HLLCFlux}`](include/adc/operator/numerical_flux.hpp) | flux numeriques (politiques `ADC_HD`) | validé Sod vs Riemann exact |
| [`operator::reconstruction`](include/adc/operator/reconstruction.hpp) | MUSCL ordre 2 (NoSlope / Minmod / VanLeer) | limiteur en parametre de template |
| [`operator::assemble_rhs` / `compute_face_fluxes`](include/adc/operator/spatial_operator.hpp) | `R = -div F + S` ; flux de FACE pour le reflux | `<Limiter, NumericalFlux>`, GPU via `for_each_cell` |
| [`integrator::{ssprk2,ssprk3}`](include/adc/integrator/ssprk.hpp) | Shu-Osher SSP-RK | TVD-stable |
| [`integrator::imex_euler_step`](include/adc/integrator/imex.hpp) | IMEX (raide implicite + non-raide explicite) | **asymptotic-preserving** |
| [`integrator::{lie_step,strang_step}`](include/adc/integrator/splitting.hpp) | splitting d'operateurs | ordre 1 / 2 |
| [`integrator::two_fluid_ap`](include/adc/integrator/two_fluid_ap.hpp) | deux-fluides 2D AP (Poisson reformule) | quasi-neutre a `dt` fixe quand `λ_D -> 0` |
| [`integrator::amr_*_mf`](include/adc/integrator/amr_reflux_mf.hpp) | AMR MultiFab : reflux 2-niv, recursion N-niv, **multi-patch coverage-aware** | bit-identique a la reference Fab2D |
| [`elliptic::GeometricMG`](include/adc/elliptic/geometric_mg.hpp) | multigrille geometrique (V-cycle GS rb) | compatible AMR, on-device |
| [`elliptic::PoissonFFTSolver`](include/adc/elliptic/poisson_fft_solver.hpp) | Poisson FFT spectrale directe | mono-niveau periodique, ~5x, distribue par bandes |
| [`coupling::Coupler`](include/adc/coupling/coupler.hpp) | couplage hyperbolique-elliptique par etage | `Coupler<Model, Elliptic = GeometricMG>` |
| [`coupling::AmrCoupler`](include/adc/coupling/amr_coupler.hpp) | couplage E x B sur hierarchie AMR (MultiFab) | conservation a 5.55e-16 |
| [`coupling::SpectralCoupler`](include/adc/coupling/spectral_coupler.hpp) | couplage E x B distribue (FFT par bandes) | MPI, `MPI_Alltoall` |
| [`amr::{cluster,regrid,tag_box}`](include/adc/amr) | tagging + clustering Berger-Rigoutsos + regrid | genere les patchs multi-box |
| [`solver::{Diocotron,EulerPoisson,TwoFluidAP}Solver`](include/adc/solver) | **facades compilees** (PIMPL, `libadc`) | API stable sans template (apps, Python) |

Concepts (`PhysicalModel`, `NumericalFlux`, `EllipticSolver`, `CouplingPolicy`) et
seams (`for_each_cell`, `comm`, `allocator`) : voir
[**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md). Profil run-time :
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Backends : configures UNE fois, herites partout

OpenMP, MPI, HDF5 et Kokkos sont attaches a la cible d'interface `adc` ; **tout ce
qui lie `adc` en herite** : la facade `libadc` (`src/`), les tests, les exemples.
Aucun drapeau rebadge par cible.

```bash
cmake -B build                       # serie
cmake -B build -DADC_USE_OPENMP=ON   # CPU multi-thread
cmake -B build -DADC_USE_MPI=ON      # distribue (halos + FFT par MPI)
cmake -B build -DADC_USE_KOKKOS=ON \ # GPU GH200 (ou CPU portable) ; libadc compile pour le GPU
   -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
```

Le seam `for_each_cell` bascule serie -> `#pragma omp` -> `Kokkos::parallel_for`
(Cuda) sans toucher les operateurs. Pas couple Euler-Poisson et deux-fluides AP
valides **bit-identiques CPU vs GH200** (cf. `scripts/romeo_*.sbatch`).

## Ecosysteme

ADC est le membre **from scratch** d'une famille de solveurs PDE C++ : la ou
`euler_cpp` / `advection_cpp` reutilisent `pde_core_cpp` (mesh, fields, AMR),
**ADC porte sa propre pile AMR** (`BoxArray` / `MultiFab` / `for_each_cell`) pour viser
le GPU et le MPI sans dependance.

| Repo | Role | Socle maillage |
|---|---|---|
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | solveurs Poisson (Thomas, SOR, CG, DST, AMR + multigrille) | propre |
| [`pde_core_cpp`](https://github.com/wolf75222/pde_core_cpp) | infra partagee (mesh, fields, AMR, clustering) | propre |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection scalaire + Burgers + Chorin NS | `pde_core_cpp` |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | Euler 2D + viscous NS + sources plasma + Euler-Poisson | `pde_core_cpp` |
| **`adc_cpp`** (ce depot) | hyperbolique-elliptique sur **AMR** + GPU/MPI/Kokkos | **propre (from scratch)** |

## Documentation

- Tutoriels (C++ et Python en parallele, du diocotron a l'AMR multi-patch) : [tutorials/](tutorials/README.md)
- Algorithmes (formules, pseudocode, validation par methode) : [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- Architecture (couches, seams, frontiere lib/demo, etat AMR) : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Schema deux-fluides AP (modele, reformulation, transport, enveloppe mesuree) : [docs/two_fluid_ap.md](docs/two_fluid_ap.md)
- Performance (profil, scaling, FFT vs MG) : [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- Choix de conception : [docs/CHOICES.md](docs/CHOICES.md) ; bibliographie : [docs/BIBLIOGRAPHY.md](docs/BIBLIOGRAPHY.md) ; roadmap : [docs/ROADMAP.md](docs/ROADMAP.md)

Generer la doc hebergeable :

```bash
doxygen docs/Doxyfile                                   # reference C++ -> docs/_build/doxygen/html
pip install -r docs/sphinx/requirements.txt
python3 -m sphinx -b html docs/sphinx docs/_build/sphinx # site Python + tutoriels
```

## Quick start

### Prerequis

- C++23 (AppleClang 16+, GCC 13+, Clang 17+)
- CMake >= 3.20
- Eigen >= 3.4 *(cote host uniquement, analyse diocotron ; optionnel `-DADC_USE_EIGEN`)*
- MPI *(optionnel `-DADC_USE_MPI=ON`)*, Kokkos *(optionnel `-DADC_USE_KOKKOS=ON`, GPU)*, HDF5 *(optionnel `-DADC_USE_HDF5=ON`)*
- Python 3.10+ *(optionnel, bindings pybind11)*

### Build

```bash
git clone https://github.com/wolf75222/adc_cpp.git
cd adc_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                 # 60 tests C++
```

Options CMake :

| Option | Defaut | Role |
|---|---|---|
| `ADC_BUILD_TESTS` | `ON` | compile la suite CTest |
| `ADC_BUILD_EXAMPLES` | `ON` | compile les drivers (`diocotron`, `diocotron_amr`, ...) |
| `ADC_USE_OPENMP` | `OFF` | backend de dispatch OpenMP |
| `ADC_USE_KOKKOS` | `OFF` | backend de dispatch Kokkos (CPU/GPU portable) |
| `ADC_USE_MPI` | `OFF` | backend distribue (comm, halos, FFT) |
| `ADC_USE_HDF5` | `OFF` | DataWriter HDF5 parallele |
| `ADC_BUILD_PYTHON` | `OFF` | module pybind11 `adc` (facade `libadc`) |
| `ADC_USE_EIGEN` | `ON` | outils d'analyse host (theorie diocotron) |

### Python

`src/` compile la pile template une fois en `libadc` (API stable sans template),
bindee via pybind11 (`-DADC_BUILD_PYTHON=ON`). On expose les solveurs CONCRETS :

```python
import adc, numpy as np

# diocotron (bande de charge), pas stable choisi par la facade
cfg = adc.DiocotronConfig(); cfg.n = 192; cfg.ic = adc.DiocotronIC.Band
s = adc.DiocotronSolver(cfg)
for _ in range(200):
    s.step_cfl(0.4)
rho = s.density()        # numpy (n, n)
phi = s.potential()      # numpy (n, n)

# deux-fluides isotherme 2D, regime raide (asymptotic-preserving)
tc = adc.TwoFluidAPConfig(); tc.n = 64; tc.omega_pe = 1e3
ts = adc.TwoFluidAPSolver(tc)
ts.advance(5e-3, 200)    # dt*omega_pe = 5 : stable, quasi-neutre
print(ts.max_dev(), ts.max_charge())

# Euler-Poisson : meme code, deux physiques (le signe du couplage)
ec = adc.EulerPoissonConfig(); ec.n = 128; ec.use_fft = True
ec.interaction = adc.InteractionKind.Gravity   # attractif : effondrement de Jeans
# ec.interaction = adc.InteractionKind.Plasma  # repulsif : Langmuir + Coulomb
es = adc.EulerPoissonSolver(ec)
for _ in range(100): es.step(2e-3)
print(es.mass(), es.total_momentum(0))
```

## Organisation du depot

```
include/adc/   coeur generique header-only (concepts, MultiFab, for_each_cell, operateurs,
               elliptique, integrateurs, AMR, modeles). Templates -> visibles a l'instanciation.
src/           facade COMPILEE libadc : solveurs concrets PIMPL (Diocotron, EulerPoisson,
               TwoFluidAP). API stable, backend herite de la cible adc.
examples/      pilotes minces (main). diocotron/diocotron_column lient adc::solver (facade) ;
               diocotron_amr/amr3/multipatch/mpi/theory lient adc::adc (moteur : AMR, MPI, Eigen).
examples/gpu/  demos Kokkos/CUDA (GH200), heritent Kokkos de adc.
tests/         CTest (+ tests MPI via mpirun). python/ : module pybind11 + test.
scripts/       generation des GIF + jobs SLURM ROMEO (MPI, GPU).
docs/          ARCHITECTURE.md, PERFORMANCE.md, animations.
```

## Validation

- **60/60** tests C++ (serie), idem OpenMP ; **+13** MPI (`mpirun -np 4`, bit-identique a np=1/2/4) ; **+1** HDF5 ; bindings Python verts.
- **GPU GH200** (CUDA 12.6) : advection, multigrille, pas couple Euler-Poisson, deux-fluides AP + `libadc` compilee GPU, tous **bit-identiques au CPU**.
- **AMR** : reflux 2-niveaux / N-niveaux / multi-patch coverage-aware, tous prouves **bit-identiques** a la reference, conservation a l'arrondi (5.55e-16) ; clustering Berger-Rigoutsos branche. Le demo couple `diocotron_multipatch` (Poisson grossier + reflux multi-patch) re-cluster ses patchs a la volee (`docs/anim_diocotron_multipatch.gif`) en conservant la masse a `~2e-15` sur tout le run.
- **Reproduction papier diocotron** (arXiv:2510.11808, objectif de stage) : la colonne sur AMR avec Poisson multi-niveau egale l'uniforme a resolution effective egale pour ~41-44 % des cellules (taux mode 4 normalise `0.42`/`0.526`/`0.563`/`0.592` aux eff 192/256/320/448). La cible analytique `0.911` n'est pas atteinte : la montee est monotone mais lente, et une instabilite numerique au-dela de eff 448 (champ en `nan`) bloque la resolution. Pipeline valide sur 1 GPU GH200 (ROMEO), qui reproduit eff-448 (uniforme `0.577`, AMR `0.592`). Detail : [tutorials/10_diocotron_reproduction.md](tutorials/10_diocotron_reproduction.md).
- **Deux-fluides AP** : dispersion isotrope (3.1%), borne + quasi-neutre a `omega_pe = 1e3` (`dt*omega_pe = 5`) la ou l'explicite explose.

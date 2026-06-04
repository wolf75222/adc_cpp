# Architecture cible (north star)

Doc de VISION (pas l'etat actuel), issu de la description du tuteur. A relire avant la seance
tableau. Deux niveaux :
- **(A)** l'architecture OO en couches (modeles composables) : DEJA largement realisee dans adc_cpp ;
- **(B)** le DSL symbolique ou Python ECRIT les formules : interprete CPU + codegen C++
  (flux / brique / source / elliptique) + CSE + JIT dlopen du noyau faits (`adc.dsl`) ; restent le
  dispatch dans le solveur template (type-erased) et Kokkos/CUDA + run GPU.

---

## 0. Principe (a graver)

```
Un modele physique ne sait pas avancer le temps.
Un schema spatial ne sait pas ce qu'est Euler.
Un flux numerique ne sait pas ce qu'est un electron.
Un EPM lit le systeme entier.
Une source couplee lit plusieurs blocs.
Le driver orchestre, mais ne contient pas la physique.
```

PDE :
```
HPM : d_t U + div F(U, aux) = S(U, U_all, aux)
EPM : D(phi, aux) = f(U_all, aux)
Chaine : Variables -> Flux physique -> Flux numerique -> SpaceMethod -> TimeMethod
         CoupledSystem orchestre HPM + EPM + sources couplees.
```

---

## 1. Arbre cible (condense)

```
solver/
  core/        scalar, state_vector, field, direction, variable, concepts
  physics/
    hyperbolic/  hyperbolic_model, variables, primitive_conservative, flux, eigenvalues
    source/      source_model, local_source, coupled_source
    elliptic/    elliptic_model, elliptic_operator, elliptic_rhs, elliptic_field
    model/       physical_model, equation_block, coupled_system
  numerics/
    reconstruction/  reconstruction (cons|prim), minmod, vanleer, weno
    flux/            numerical_flux, rusanov, hll, hllc
    space/           space_method, finite_volume
    time/            time_method, explicit, implicit, imex, ssprk, scheduler
  mesh/    mesh, geometry, boundary_conditions, field_layout
  amr/     hierarchy, refinement_criterion, regrid, prolongation, restriction, reflux
  runtime/ registry, factory, simulation, backend
  python/  bindings.cpp
```

### Etat de l'arborescence REELLE (juin 2026)

L'arbre `include/adc/` a ete reorganise vers cette cible (prefixe `<adc/...>` conserve) :
`core/`, `physics/` (euler + hyperbolic / source / elliptic / composite, cf. scission de `bricks.hpp`),
`numerics/` (flux + spatial_operator + `time/` + `elliptic/`), `mesh/`, `amr/`, `runtime/`,
`coupling/`, `parallel/`. Les categories de briques sont desormais des fichiers separes
(`physics/{hyperbolic,source,elliptic,composite}.hpp`), `physics/bricks.hpp` restant un umbrella de
compat. Restent flat (pas de sous-dossiers fins `numerics/{flux,space,reconstruction}`) faute de
correspondance 1:1 fichier<->concept ; `physical_model.hpp` (concepts) et `variables.hpp` restent dans
`core/` (contrats fondamentaux). Renommages : `Variables`->`VariableSet`, `HyperbolicModel`->
`HyperbolicPhysicalModel` (alias de compat conserves).

---

## 2. Couches OO (et etat actuel)

| Couche cible | Etat dans adc_cpp aujourd'hui |
|---|---|
| `Variables` (cons/prim, conversions) | FAIT : `core/variables.hpp` (kind, names, size) + `using State/Prim` + `to_primitive`/`to_conservative` sur la brique hyperbolique. MANQUE : `VariableRole` semantique (Density/MomentumX/...), `index_of(role)`. |
| `HyperbolicModel` (Vars + flux + lambda) | FAIT : concept `HyperbolicModel` + briques Euler / IsothermalFlux / ExBVelocity. MANQUE : `eigenvalues()` rendant le VECTEUR complet (on a `max_wave_speed` + `wave_speeds` signees) ; `Direction` enum (on a `int dir`). |
| `LocalSource` / `CoupledSource` | FAIT : briques source locale (PotentialForce/GravityForce/NoSource) ; sources couplees ionisation/collision/echange (operator-split). MANQUE : hierarchie `CoupledSource` objet propre + composition `source = a + b`. |
| `EllipticPhysicalModel` | FAIT (1er ordre) : `add_elliptic_model` + briques (div_eps_grad, charge_density, electric_field_from_potential). MANQUE : eps(x) variable, operateurs alternatifs (diffusion, projection), rhs au niveau EPM. |
| `PhysicalModel` = HPM + source + elliptic | FAIT : `CompositeModel<Hyperbolic, Source, Elliptic>`. |
| `EquationBlock` (model + space + time + bc + evolve) | FAIT : blocs du runtime System (model + spatial + time + substeps + evolve). |
| `SpaceMethod` / `FiniteVolume` | FAIT : `assemble_rhs<Limiter, Flux>` (volumes finis, generique sur le modele). |
| `Reconstruction` (cons|prim) | FAIT : recon cons/prim + minmod/vanleer/weno. |
| `NumericalFlux` (Rusanov/HLL/HLLC) | FAIT : generiques sur le modele (`m.flux`, `m.max_wave_speed`). |
| `TimeMethod` (explicit/imex/ssprk/scheduler) | FAIT : ForwardEuler/SSPRK2 + IMEX partiel + multirate (`step_adaptive`). MANQUE : implicite TOTAL, `stride` (every-N) expose proprement. |
| `CoupledSystem` + `Assembler`/`Driver` | FAIT : le runtime System (vector de blocs + Poisson + couplages) ; split Assembler/Driver fait cote coeur. |
| `mesh` / `amr` / `runtime` / `python` | FAIT : MultiFab/Geometry/BC, AMR (hierarchie, regrid, reflux), factory (dispatch), Simulation (System), bindings. |

Bilan : **les couches 1-12 du design OO sont ~80 % en place** (organisation et noms differents, mais l'abstraction y est, et le refactor recent l'a renforcee : HyperbolicModel, Variables contrat, EPM, sources couplees, recon primitive).

---

## 3. Le DSL symbolique (l'endgame, NOUVEAU)

But : Python ECRIT les formules (pas une fonction appelee par cellule), ADC en fait un solveur.

```python
e = adc.dsl.HyperbolicModel("electrons")
rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
u = e.primitive("u", rhou / rho)
p = e.primitive("p", (gamma - 1) * (E - 0.5 * rho * (u*u + v*v)))
c = adc.dsl.sqrt(gamma * p / rho)
e.set_flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u], y=[...])
e.set_eigenvalues(x=[u - c, u, u + c], y=[...])
e.set_source([...]); e.set_elliptic_rhs(-qe * rho / me)
```

`rho`, `u`, `p`... ne sont pas des floats : ce sont des EXPRESSIONS SYMBOLIQUES. Python construit un
GRAPHE de formules ; ADC peut alors : (1) l'interpreter en CPU (proto), (2) generer du C++,
(3) generer du Kokkos/CUDA, (4) JIT, (5) verifier les dependances entre variables. `Euler`,
`diocotron`, `two-fluid` deviennent de simples fichiers Python de formules.

Arbre additionnel :
```
adc/symbolic/  expression (Var/Const/Add/Mul/Sqrt...), vector_expr, formula_graph, simplifier, codegen
```

### Avis honnete (ingenierie)

- C'est un **compilateur de domaine**, pas un raffinement. Effort realiste : **pluri-mois**, pas un
  refactor. Le risque principal : le codegen GPU (un graphe Python -> kernel Kokkos/CUDA correct et
  performant) est exactement la partie difficile.
- **Ca existe** par briques : SymPy (codegen C/Fortran), **Pystencils** (stencils -> C/CUDA),
  **Devito** (DSL differences finies -> C optimise), UFL/Firedrake (elements finis). Pour le
  volume-fini hyperbolique (flux + valeurs propres + reconstruction), moins d'off-the-shelf, mais
  les fondations existent ; ne pas reconstruire un moteur symbolique de zero (reutiliser SymPy).
- **Tradeoff vs l'actuel** : aujourd'hui on a deux chemins, briques COMPILEES (template, GPU/MPI,
  production) + `adc.PythonFlux` (proto CPU numpy). Le double-codage (une formule ecrite en C++ ET
  en numpy) est le cout que le DSL supprimerait : UNE source de formules -> interprete CPU + kernel
  GPU genere. Le gain croit avec le nombre de modeles (plasma multi-especes : beaucoup de variantes).
- **Recommandation** : garder le solveur compile actuel (il marche, GPU-ready) comme cible de
  production ; demarrer le DSL en **prototype separe** (`adc/dsl.py`), d'abord un interprete CPU
  d'un graphe de formules (valider le concept sur Euler), PUIS un codegen C++. Ne PAS reecrire le
  solveur en attendant que le DSL soit mur.

### Etat : interprete CPU PROTOTYPE fait (`adc.dsl`)

Le module `python/adc/dsl.py` realise l'etape (1) (interpreter en CPU) et (5) (verifier les
dependances) :
- arbre d'expressions (`Expr` : `Const`, `Var`, `Add/Sub/Mul/Div/Pow/Neg`, `Sqrt`) construit par
  surcharge d'operateurs ; `eval(env)` l'applique a des tableaux numpy (tout le domaine d'un coup) ;
- `HyperbolicModel` declaratif : `conservative_vars` / `primitive` (formules) / `aux` / `set_flux` /
  `set_eigenvalues` / `set_source` / `set_elliptic_rhs` / `check()` (dependances) ;
- `to_python_flux()` branche l'arbre sur le backend hote `adc.PythonFlux` -> **le modele TOURNE**.

Verifie : `python/tests/test_dsl.py` (flux symbolique d'Euler == flux de reference numpy,
`max_wave_speed` coherent, `check()` detecte une variable non definie, masse conservee a l'execution)
et le cas `adc_cases/dsl_euler/` (Euler ecrit en formules, expansion acoustique, masse conservee).

Etape (2) FAITE pour Euler (codegen hote + emballage en brique) :
- `emit_cpp()` genere la fonction flux `template <class Real> void <nom>_flux(const Real*, Real*, int)`
  depuis l'arbre (chaque noeud `Expr` sait s'ecrire via `to_cpp()`) ;
- `emit_cpp_brick()` genere une BRIQUE complete : un struct (`StateVec` / `Aux` / `ADC_HD`) avec flux,
  max_wave_speed, to_primitive, to_conservative, conservative_vars / primitive_vars, qui SATISFAIT le
  concept `adc::HyperbolicModel` (donc utilisable dans un CompositeModel / le solveur). Les conversions
  non inversibles (to_conservative) sont fournies par l'utilisateur (`set_conservative_from`), le DSL
  ne sachant pas inverser symboliquement.

- `emit_cpp_source()` genere une BRIQUE de SOURCE composable (`apply(U, a)`), avec locals aux lus
  comme `a.<champ>` (convention : noms aux = champs de `adc::Aux`, p.ex. grad_x / grad_y).

Verifie (tous en CI) :
- `test_dsl_codegen.py` : flux genere == interprete numpy ;
- `test_dsl_brick.py` : la brique COMPILE contre les en-tetes adc, `static_assert(adc::HyperbolicModel<...>)`,
  et == `adc::Euler` (4 var, sans aux) ET `adc::ExBVelocity` (1 var, flux dependant des aux), ecart nul ;
- `test_dsl_source.py` : la source generee == `adc::PotentialForce` ;
- `test_dsl_compose.py` : `CompositeModel<EulerGen, NoSource, ChargeDensity>` satisfait `adc::PhysicalModel`
  et egale la version ecrite a la main ;
- `test_dsl_jit.py` : JIT-lite end-to-end (formules Python -> `.hpp` genere -> g++ compile un driver
  volumes-finis -> residu Rusanov identique a `adc::Euler`).
Verifie aussi sur ROMEO (g++ 11, C++20) : compilation et egalite au bit pres.

FAIT depuis : codegen elliptique (`emit_cpp_elliptic`, == `adc::ChargeDensity`) ; CSE (`cse=True` par
defaut : H / c factorises en locales `cseK_`, verifie identique a la version sans CSE et a `adc::Euler`) ;
JIT REEL du noyau (`test_dsl_jitlib` : flux genere -> `.so` compile a la volee -> charge dans le process
Python via ctypes -> == interprete numpy).

RUN GPU FAIT : la brique generee compile avec nvcc (`-arch=sm_90`) et s'execute sur NVIDIA GH200
(ROMEO, noeud aarch64), resultat BIT-IDENTIQUE a `adc::Euler` (cf. docs/GPU_ROMEO.md). Elle est
device-ready par construction (`ADC_HD` -> `__host__ __device__`, ops device-safe, `std::sqrt`).

KOKKOS FAIT (verification de la brique) : la brique generee tourne via `Kokkos::parallel_for` sur
l'espace d'execution `Cuda` (GH200 ; Kokkos 4.4 + CUDA 12.6, `HOPPER90`), == `adc::Euler` a un ULP
(5.6e-17, contraction FMA), cf. docs/GPU_ROMEO.md. C'est le meme primitif de dispatch que
`adc/mesh/for_each.hpp`.

(a) FAIT : interface TYPE-ERASED `adc::IModel<NV>` + `ModelAdapter` (include/adc/runtime/dynamic_model.hpp)
ET cablage dans le runtime. `System::add_dynamic_block(name, so)` charge a l'execution (dlopen) une brique
generee compilee en `.so` et cree un bloc pilote par l'IModel (host Rusanov ordre 1), avance par
eval_rhs / step / step_cfl comme n'importe quel bloc ; `dsl.HyperbolicModel.compile_so` fait le JIT.
Bout-en-bout verifie : `test_dsl_block` (DSL -> .so -> add_dynamic_block ; eval_rhs == adc.PythonFlux a
9e-16 ; 25 pas dans le System, masse conservee) ; `test_dynamic_model` (C++) et `test_dsl_dynamic`
(dlopen depuis un main ignorant le type) verrouillent le mecanisme. Chemin HOTE (vtable, hors GPU ;
pendant COMPILE de PythonFlux). Le hot path GPU reste le chemin TEMPLATE.

(b) FAIT : un CAS Euler 2D COMPLET (80 pas, CFL, Rusanov, periodique) avance sur GH200 a travers le
seam Kokkos d'adc (`for_each_cell` / `for_each_cell_reduce_*`), masse exactement conservee, brique
generee == `adc::Euler` a 9e-16 sur 80 pas (cf. docs/GPU_ROMEO.md). Reste : la pile runtime ENTIERE
(System / AMR / MPI) sur GPU, et `sim.add_dynamic_block` cote Python (item (a)).

Le codegen hote et le dispatch type-erased NE remplacent PAS les briques compilees de production (chemin
template, GPU/MPI).

---

## 4. Chemin pragmatique (incremental, sans bloquer sur le DSL)

Petits pas concrets qui rapprochent du tableau SANS le compilateur :
1. `VariableRole` (Density/MomentumX/.../Pressure/Temperature) + `index_of(role)` sur `Variables` :
   donne du sens aux composantes (utile aux sources couplees : « la vitesse de telle espece »).
2. `eigenvalues()` rendant le vecteur complet des valeurs propres (au-dela de `max_wave_speed`).
3. `Direction` enum (X/Y) au lieu de `int dir` (lisibilite, type-safe).
4. Composition de sources locales : `source = ElectricForce(...) + LorentzForce(...)`.
5. Reorg progressive des repertoires vers l'arbre cible (cosmetique, a faire quand on touche un coin).

Le DSL symbolique (section 3) est le cap a long terme, traite comme un sous-projet a part.
```

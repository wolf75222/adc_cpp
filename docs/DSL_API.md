# DSL_API -- Reference courte du DSL Python (adc.dsl)

Document de reference UTILISATEUR. Pour la conception, le raisonnement et l'historique,
voir [docs/DSL_MODEL_DESIGN.md](DSL_MODEL_DESIGN.md).

---

## 1. Ecrire un modele symbolique

```python
import adc
from adc import dsl

m = dsl.Model("mon_modele")

# Variables conservatives : conservative_vars(...) RENVOIE un tuple de Var a depacker
# (roles physiques optionnels).
rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                  roles=["Density", "MomentumX", "MomentumY"])

# Variables primitives (kwargs : nom=expression symbolique)
m.primitive_vars(rho=rho, ux=mx/rho, uy=my/rho)

# Flux physique (declarateur symbolique ; m.eval_flux(...) = evaluateur numpy)
m.flux(x=[mx, mx*mx/rho, mx*my/rho],
       y=[my, mx*my/rho, my*my/rho])

# Source (optionnel -- force du potentiel)
# m.aux('grad_x') declare et renvoie le champ auxiliaire grad_x (Var).
phi_x, phi_y = m.aux("grad_x"), m.aux("grad_y")
m.source([-rho*phi_x, -rho*phi_y])  # exemple ExB / force

# Second membre elliptique (optionnel -- couplage Poisson)
m.elliptic_rhs(rho)

# Parametre nomme (constante inlinee a la compilation)
g = m.param("gamma", 1.4)
```

---

## 2. Compiler

```python
# Le DEFAUT de m.compile(...) est backend="aot" : il faut donc demander explicitement
# "production" pour le chemin natif zero-copie (recommande en MPI/AMR).
compiled = m.compile(backend="production", target="system")
# Pour AMR :
compiled_amr = m.compile(backend="production", target="amr_system")
```

Backends disponibles (`backend=` ; DEFAUT = `aot`) :

| backend | CPU | MPI | AMR | GPU | Remarque |
|---|---|---|---|---|---|
| `production` | oui | oui (np=1/2/4) | via `AmrSystem` | GH200 (cote C++) | **recommande** en MPI/AMR ; natif zero-copie. `_BACKEND_CAPS["production"]["gpu"]` est rapporte `False` cote Python (le module hote teste n'est pas bati Kokkos/CUDA) |
| `aot` | oui | non | non | non | **DEFAUT** ; `.so` a marshaling ; debug/bench CPU. Porte aussi les params runtime (`set_block_params`) |
| `prototype` | oui (Rusanov o1) | non | non | non | JIT proto ; ne pas utiliser en production |

Le `.so` est mis en cache par `model_hash` : un modele inchange n'est pas recompile.

---

## 3. Brancher sur System / AmrSystem

```python
sim = adc.System(n=256, periodic=True)
sim.add_equation("fluide",
                 model=compiled,
                 spatial=adc.FiniteVolume(limiter="vanleer", riemann="rusanov"),
                 time=adc.Explicit(substeps=1))
# 1er argument positionnel = rhs (valide dans {charge_density, composite}) : passer le solveur
# par MOT-CLE, pas en positionnel.
sim.set_poisson(solver="geometric_mg")
sim.run(t_end=10.0, cfl=0.4)
```

```python
# AMR : AmrSystemConfig n'a PAS de champ max_level. Champs reels : n, L, regrid_every, periodic,
# distribute_coarse, coarse_max_grid (regrid_every=0 -> hierarchie figee).
amr = adc.AmrSystem(n=128, L=1.0, regrid_every=4, periodic=True)
amr.add_equation("fluide",
                 model=compiled_amr,
                 spatial=adc.FiniteVolume(limiter="vanleer", riemann="rusanov"),
                 time=adc.Explicit(substeps=1))
```

Points importants :
- `riemann=` nomme le flux NUMERIQUE (`rusanov`/`hllc`/`roe`) ; `m.flux(...)` est le flux PHYSIQUE.
- `fft` n'est pas supporte sous `System` en MPI `np>1` : employer `geometric_mg`.
- `backend="production"` avec `target="amr_system"` : `AmrSystem` est mono- ET multi-bloc,
  explicite ; HLLC/Roe/`primitive` sont rejetes cote facade Python AMR (le moteur C++ les supporte,
  mais le binding Python ne les expose pas encore sur ce chemin).

---

## 4. Cache et reproductibilite

`m.compile()` retourne un objet `CompiledModel` qui porte :
- `so_path` : chemin du `.so` compile.
- `model_hash` : hash stable (formules + roles + params) -- cle de cache.
- `abi_key` : cle compilateur/std/en-tetes -- refus explicite si incompatible au chargement.
- `params` : dict des parametres nommes declares via `m.param(...)`.

---

## 5. Points de vigilance

- `m.param(name, value)` : par defaut (`kind="const"`) constante INLINEE a la compilation ; changer
  la valeur exige un nouvel appel a `m.compile()`. Le mode `runtime` (`kind="runtime"`) est SUPPORTE
  sur le backend `aot` : la valeur est modifiable SANS recompiler via `System.set_block_params`.

  ```python
  m = dsl.Model("iso")
  rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v")
  cs2 = m.param("cs2", 1.0, kind="runtime")          # param RUNTIME (defaut = 1.0)
  u, v = m.primitive("u", mx/rho), m.primitive("v", my/rho)
  p = m.primitive("p", cs2 * rho)
  m.primitive_vars(rho=rho, u=u, v=v, p=p)
  m.conservative_from([rho, rho*u, rho*v])
  m.flux(x=[mx, mx*u+p, my*u], y=[my, mx*v, my*v+p])
  cs = dsl.sqrt(cs2)
  m.eigenvalues(x=[u-cs, u, u+cs], y=[v-cs, v, v+cs])

  compiled = m.compile(backend="aot")                 # cache key inclut les params
  compiled.runtime_param_names                         # -> ['cs2'] (ordre des indices C++)

  sim = adc.System(n=64, periodic=True)
  sim.add_equation("gas", model=compiled,
                   spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"))
  sim.set_block_params("gas", [4.0])                   # change cs2 au RUNTIME, sans recompiler
  ```
- `adc.PythonFlux` : outil de TEST numpy hote, hors hot path GPU/MPI. Ne jamais utiliser en
  production.
- Roles physiques (`Density`, `MomentumX`, `MomentumY`, ...) : requis pour les couplages
  inter-especes et pour que le `System` retrouve les grandeurs par role. A fournir a
  `conservative_vars(roles=...)` ou a `m.compile(require_metadata=True)`.
- `m.projection([...])` (ADC-177) : PROJECTION PONCTUELLE post-pas `U <- P(U, aux)`, une expression
  par composante conservative, emise comme le trait C++ `HasPointwiseProjection` et compilee comme
  flux/source (CSE comprise) -- remplace le callback Python par cellule. SEMANTIQUE : appliquee par
  le System UNE fois a la FIN de chaque macro-pas ENTIER (apres transport + etage source +
  couplages ; jamais par etage RK, y compris en Strang), sur les cellules VALIDES seulement (les
  ghosts sont refaits par le `fill_ghosts` de tete du pas suivant -- pas de `fill_boundary` dans le
  hook). CONTRAT : P idempotente (une vraie projection) et ponctuelle (aucun voisin) ; les clamps
  s'ecrivent SANS branche, en max/min via `dsl.abs_` / `dsl.sign` (derivables par `dsl.diff`), p.ex.
  positivite `(q + abs_(q))/2`. Backends : `aot` et `production` cote `System` ; `prototype` (JIT)
  et `target="amr_system"` REJETTENT explicitement (le hook n'est pas cable sur le pas AMR --
  l'application par niveau apres le reflux du pas est le perimetre suivant).

---

## 6. Demonstrateurs de reference (adc_cases, ci=true)

| Cas | Fichier |
|---|---|
| ExB mono-espece DSL | `diocotron_dsl/run.py` |
| Deux especes DSL | `two_species_dsl/run.py` |
| Isotherme magnetique DSL | `magnetic_isothermal_dsl/run.py` |

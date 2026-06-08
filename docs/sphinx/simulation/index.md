# Simulation

`adc.System` est la facade de COMPOSITION du solveur : on declare un bloc par modele physique
(une espece), on partage un Poisson de systeme entre tous les blocs, on pose des conditions
initiales en numpy, puis on avance le tout par pas. Le coeur ne nomme aucun scenario (diocotron,
Euler-Poisson... vivent cote `adc_cases`) ; ici on assemble des BRIQUES generiques.

Cette page parcourt la mecanique de simulation cote Python : composer un systeme, coupler
plusieurs especes, choisir le schema spatial et la politique temporelle, gerer le multirate,
initialiser et lire les champs. Le detail theorique des methodes numeriques est dans
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) ; l'architecture en couches (seam de dispatch, frontiere
lib/application) dans [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

## System

`adc.System` construit le coupleur. La configuration passee en mots-cles ne decrit que le
MAILLAGE (`n`, `L`, `periodic`) : le domaine est un carre `[0, L]^2` de `n x n` cellules.

```python
import numpy as np
import adc

sim = adc.System(n=128, L=1.0, periodic=False)
```

On ajoute ensuite UN bloc par modele. Un modele est une composition de briques
(`adc.Model(state, transport, source, elliptic)`) ; le bloc recoit en plus un schema spatial
(`adc.Spatial` / `adc.FiniteVolume`) et une politique temporelle (`adc.Explicit`, `adc.IMEX`...).

```python
model = adc.Model(state=adc.Scalar(), transport=adc.ExB(B0=1.0),
                  source=adc.NoSource(), elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0))
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
```

Le Poisson de systeme est PARTAGE par tous les blocs : son second membre est la somme des
contributions elliptiques de chaque bloc (`f = sum_s elliptic_rhs_s(u_s)`). On le configure une
fois, apres les blocs.

```python
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)
```

`rhs` vaut `"charge_density"` (cas usuel : tous les blocs portent une densite de charge
`f = sum_s q_s n_s`) ou `"composite"` (somme generique des briques elliptiques par bloc) ;
`solver` est `"geometric_mg"` (multigrille, tout cas) ou `"fft"` (periodique, `n = 2^k`) ; `bc`
gere la condition aux limites (`"auto"`, `"dirichlet"`, `"periodic"`), `wall`/`wall_radius`
materialisent une paroi conductrice circulaire (cut-cell). `set_poisson` est le raccourci de
`add_elliptic_model` (le Poisson est une instance de modele elliptique composable).

On pose la condition initiale puis on avance :

```python
sim.set_density("ne", ne0)          # tableau (n, n)
sim.step_cfl(0.4)                   # un pas au CFL 0.4 (renvoie le dt effectif)
sim.advance(0.01, 50)               # 50 pas de dt fixe = 0.01
```

`add_block`, `add_equation`, `set_poisson`, `set_density`, `step`, `step_cfl`, `advance` et les
diagnostics sont transmis a la facade compilee. Cote C++ le coupleur vit dans
`runtime/system.hpp` (`System`, multi-blocs mono-niveau, Poisson partage) et est expose a Python
par `python/bindings.cpp`. Le backend (serie / OpenMP / Kokkos GPU / MPI) est celui avec lequel
`libadc` a ete compilee ; la physique ne voit jamais le backend.

> **Variante AMR.** `adc.AmrSystem(n=, L=, periodic=)` est le pendant raffine de `System` :
> memes signatures `add_block` / `add_equation` / `set_poisson` / `set_density` / `step_cfl`,
> avec en plus `set_refinement(threshold)` (raffine ou la densite depasse un seuil) et
> `set_phi_refinement(grad_threshold)` (raffine sur `|grad phi|`). La cadence de regrid se regle
> via `AmrSystemConfig.regrid_every` (0 = hierarchie figee). Detail :
> [AMR_MULTIBLOCK_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/AMR_MULTIBLOCK_DESIGN.md).

## Multi-blocs et multi-especes

Plusieurs blocs co-existent dans un meme `System`, couples UNIQUEMENT par le second membre du
Poisson partage (`f = sum_s q_s n_s`) et, optionnellement, par des sources inter-especes ; jamais
par le flux. Chaque bloc garde son propre modele, son propre schema spatial et sa propre
politique temporelle. En multi-blocs, le NOM du bloc indexe `set_density(name)` / `density(name)`
/ `mass(name)`.

```python
n = 48
electrons = adc.Model(state=adc.FluidState("compressible", gamma=1.4),
                      transport=adc.CompressibleFlux(),
                      source=adc.PotentialForce(charge=-1.0),
                      elliptic=adc.ChargeDensity(charge=-1.0))
ions = adc.Model(state=adc.FluidState("isothermal", cs2=0.5),
                 transport=adc.IsothermalFlux(),
                 source=adc.PotentialForce(charge=+1.0),
                 elliptic=adc.ChargeDensity(charge=+1.0))

sim = adc.System(n=n, L=1.0, periodic=True)
sim.add_block("electrons", model=electrons,
              spatial=adc.Spatial(vanleer=True, flux="hllc"),
              time=adc.IMEX(substeps=10))           # raide : source implicite, sous-cyclee
sim.add_block("ions", model=ions, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("electrons", ne0)
sim.set_density("ions", np.ones((n, n)))
sim.advance(0.001, 8)
print("blocs :", sim.block_names())
```

**Sources couplees inter-especes.** En plus du couplage par le champ, des SOURCES inter-especes
(operator-split, appliquees apres le transport) transferent matiere, quantite de mouvement ou
energie entre blocs. Trois formes figees s'ajoutent via `sim.add_coupling(...)` (ou les methodes
directes `add_ionization` / `add_collision` / `add_thermal_exchange`) :

- `adc.Ionization(electron, ion, neutral, rate)` : ionisation `n_g -> n_i + n_e` (taux
  `k n_e n_g`), masse transferee du neutre vers l'ion ;
- `adc.Collision(a, b, rate)` : friction inter-especes (force `k (u_a - u_b)`), quantite de
  mouvement conservee (blocs fluides, >= 3 variables) ;
- `adc.ThermalExchange(a, b, rate)` : echange thermique `k (T_a - T_b)`, energie conservee
  (blocs Euler a 4 variables).

```python
sim.add_ionization(electron="ne", ion="ni", neutral="ng", rate=0.5)   # n_g diminue, n_i augmente
sim.add_coupling(adc.Collision("a", "b", rate=1.0))                    # transfert de qte de mvt a -> b
sim.add_thermal_exchange("a", "b", rate=1.0)                           # energie chaud -> froid
```

Pour une source inter-especes GENERIQUE (decrite en formules plutot que figee), le DSL
`adc.dsl.CoupledSource(...).compile(...)` produit un descripteur que `sim.add_coupling(...)`
branche aussi (bytecode interprete cote C++, aucun callback Python par cellule, MPI-safe). Le
detail de la realite multi-especes / plasma est dans
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) (section 18, "composition runtime et systeme
multi-especes") et [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md). Surface de couplage exhaustive :
[COUPLING_SURFACE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLING_SURFACE.md), [COUPLER_HIERARCHY.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/COUPLER_HIERARCHY.md).

## Schemas spatiaux

Le schema spatial = reconstruction (limiteur) + flux numerique de Riemann + variables
reconstruites. Deux facades equivalentes le decrivent.

`adc.Spatial(limiter=, flux=, recon=)` est la facade directe, avec des raccourcis booleens
(`minmod=True`, `vanleer=True`, `weno5=True`, `none=True`, `primitive=True`) :

```python
adc.Spatial(minmod=True)                       # MUSCL minmod, Rusanov, variables conservees
adc.Spatial(vanleer=True, flux="hllc")         # MUSCL Van Leer, HLLC
adc.Spatial(weno5=True, primitive=True)        # WENO5-Z, reconstruction primitive
```

`adc.FiniteVolume(limiter=, riemann=, variables=)` est la meme chose, mais le flux NUMERIQUE
de Riemann s'appelle `riemann` (et non `flux`, reserve au flux PHYSIQUE d'un modele DSL) :

```python
adc.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")
```

Les valeurs possibles :

- **limiteur** : `"none"` (Godunov ordre 1), `"minmod"`, `"vanleer"` (MUSCL ordre 2, 2 ghosts),
  `"weno5"` (WENO5-Z, ordre 5 en zone lisse, stencil 5 points / 3 ghosts, capture sans
  oscillation pres d'un front). `weno5` n'est expose que par le chemin natif `add_block` et les
  backends compiles `aot`/`production` (le chemin JIT `prototype` le rejette) ;
- **flux de Riemann** : `"rusanov"` (le plus robuste, defaut du transport scalaire), `"hllc"`,
  `"roe"`. HLLC et Roe exigent un transport compressible (4 variables + pression) ;
- **reconstruction** : `"conservative"` ou `"primitive"`. Le primitif est plus robuste pour Euler
  (positivite de `rho` et `p`).

Cote C++, les limiteurs sont des politiques dans `numerics/reconstruction.hpp` (`NoSlope`,
`Minmod`, `VanLeer`, `Weno5`), les flux dans `numerics/numerical_flux.hpp` (`RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux`). Detail et formules : [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md)
sections 2 et 3.

## Schemas en temps : explicite / IMEX / Strang / Schur

La politique temporelle est par bloc (l'objet passe en `time=`). Quatre familles.

**Explicite** -- `adc.Explicit(substeps=, stride=, method=)` : transport et source avances
explicitement par un Runge-Kutta SSP (`method="ssprk2"` par defaut, Heun 2 etages ; `"ssprk3"`,
ou raccourci `ssprk3=True`, 3 etages ordre 3, a apparier avec `weno5`).

```python
time=adc.Explicit()                       # SSPRK2, defaut
time=adc.Explicit(ssprk3=True)            # SSPRK3, moins dissipatif
```

**IMEX** -- `adc.IMEX(substeps=, stride=, implicit_vars=, implicit_roles=)` (alias clair
`adc.SourceImplicit`) : transport explicite (SSPRK) + SOURCE raide implicite (backward-Euler,
Newton local a la cellule). Traitement PARTIEL : seule la source est implicite, le transport
reste explicite. Ce n'est PAS un solveur implicite global PDE. Le masque `implicit_vars` /
`implicit_roles` choisit quelles variables conservees sont traitees en implicite (les autres
restent explicites) ; il est porte par la POLITIQUE (le bloc), pas par le modele.

```python
time=adc.IMEX(substeps=10)                                  # source raide, sous-cyclee
time=adc.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"])
```

**Splitting Lie / Strang + etage source condense par Schur** -- `adc.Split` et `adc.Strang`
opt-in dans le chantier Schur : un etage de transport hyperbolique EXPLICITE (`adc.Explicit`,
SSPRK) suivi d'un etage SOURCE separe `adc.CondensedSchur`. `adc.Split` enchaine `H(dt) ; S(dt)`
(Lie / Godunov, 1er ordre) ; `adc.Strang` joue `H(dt/2) ; S(dt) ; H(dt/2)` (symetrique, 2e
ordre). L'etage `adc.CondensedSchur` traite la source raide couplee potentiel / vitesse / Lorentz
en assemblant et resolvant un operateur elliptique tensoriel condense (BiCGStab preconditionne
MG) -- c'est un implicite GLOBAL (il couple tout le domaine). `adc.Split` / `adc.Strang` ne sont
cables que par `add_equation` (qui branche l'etage source), PAS par `add_block`.

```python
sim.add_equation("ions", model=compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Strang(hyperbolic=adc.Explicit(),
                                 source=adc.CondensedSchur(theta=0.5, alpha=3.0)))
```

> **Local vs global.** `adc.SourceImplicit` (IMEX) est LOCAL : il ne couple que les composantes
> d'une meme cellule (relaxation, reactions, friction), sans solve elliptique. `adc.CondensedSchur`
> est GLOBAL : pour le couplage Lorentz / electrostatique raide non local. Une source raide
> purement locale n'a pas besoin de Schur.

`adc.Implicit` est DEPRECIE (alias d'IMEX, emet un `DeprecationWarning`) : son nom suggere a tort
un solveur implicite global. Utiliser `adc.SourceImplicit(...)` ou `adc.IMEX(...)`.

Detail : [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) sections 4 a 6,
[SCHUR_CONDENSATION_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/SCHUR_CONDENSATION_DESIGN.md), sequence de pas Hoffart
[HOFFART_STEP_SEQUENCE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/HOFFART_STEP_SEQUENCE.md). Cote C++ : `numerics/time/*.hpp`.

## Sous-pas, stride et multirate

Deux parametres orthogonaux, portes par toute politique temporelle, gerent le multirate (toutes
les especes ne demandent pas le meme `dt`).

- **`substeps=N`** : le bloc avance `N` fois par macro-pas, chaque sous-pas de longueur `dt/N`
  (espece rapide, p.ex. electrons `substeps=10`). Defaut 1 ;
- **`stride=M`** : cadence du bloc, semantique HOLD-THEN-CATCH-UP (rattrapage en FIN de fenetre).
  Le bloc est TENU (etat inchange) tant que `(macro_step + 1) % M != 0`, puis avance d'un pas
  effectif `M*dt` au macro-pas ou `(macro_step + 1) % M == 0` (espece lente, p.ex. neutres
  `stride=20`). Il reste ainsi temporellement coherent avec les blocs rapides (jamais avance
  "dans le futur"). Defaut 1.

```python
sim.add_block("a", model=m, time=adc.Explicit(stride=1))   # chaque macro-pas
sim.add_block("b", model=m, time=adc.Explicit(stride=3))   # avance une fois sur 3 (fin de fenetre)
```

Entre deux rattrapages, le bloc tenu contribue au second membre du Poisson de systeme avec son
etat PERIME (sa derniere densite avancee, figee jusqu'au prochain rattrapage). `step_cfl` honore
la cadence : le pas stable inclut le facteur stride et substeps,
`dt <= cfl * h * substeps / (stride * w)`.

> **Attention bit-parite.** Avec `substeps=1` (quel que soit le stride), `step_cfl` est
> bit-identique a l'historique. Avec `substeps > 1` il avance un `dt` PLUS GRAND (chaque sous-pas
> reste a la limite de stabilite). Pour reproduire un run calibre avec l'ancienne formule, utiliser
> `step(dt)` avec le `dt` historique explicite.
>
> **Note backend.** Le backend `aot` (`add_equation` sur un `CompiledModel` `backend='aot'`) ne
> transporte PAS la cadence et REJETTE `stride > 1` (route explicite, pas d'ignore silencieux) ;
> `add_block` (natif) et `backend='production'` supportent le stride.

Le multirate s'obtient donc simplement en reglant `stride` (et `substeps`) par bloc. Detail :
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md) section 7.

## Conditions initiales

Deux facons de poser l'etat initial d'un bloc, toutes deux en numpy `(n, n)`. La convention de
disposition est ROW-MAJOR `(ny, nx)` : le premier indice (lignes) est l'axe `y` lent, le second
(colonnes) est l'axe `x` rapide -- un champ s'indexe `ne[j, i]` (`j` = ligne / `y`, `i` =
colonne / `x`).

`set_density(name, arr)` pose la densite (composante 0) et laisse le reste AU REPOS (pour un
fluide : vitesse nulle, energie coherente). C'est le raccourci usuel pour un transport scalaire.

```python
coord = (np.arange(n) + 0.5) / n * L
xx, yy = np.meshgrid(coord, coord, indexing="xy")          # xx, yy de forme (n, n) = (ny, nx)
r = np.hypot(xx - 0.5 * L, yy - 0.5 * L)
ne = np.full((n, n), 1e-3)
ne[(r > 0.15) & (r < 0.20)] = 1.0
sim.set_density("ne", ne)
```

`set_primitive_state(name, **prims)` initialise un bloc fluide DEPUIS ses variables primitives,
nommees (`rho`, `u`, `v`, `p`...). Chaque primitive est un tableau `(n, n)` ; le modele du bloc
les convertit en variables conservatives (compressible : `E = p/(g-1) + 1/2 rho|v|^2`). Les noms
attendus sont ceux du modele ; un nom inconnu ou manquant leve une erreur claire.

```python
sim.set_primitive_state("electrons", rho=rho0, u=u0, v=v0, p=p0)
```

Pour un etat conservatif explicite (diagnostic / cas avance), `set_state(name, u)` prend le
tableau aplati `(ncomp, n, n)`.

## Sorties et diagnostics

Le systeme rend ses champs en numpy `(ny, nx)` (ou `(ncomp, ny, nx)`), meme convention row-major
qu'en entree.

- `sim.density(name)` : densite du bloc, tableau `(n, n)` ;
- `sim.mass(name)` : masse totale du bloc (scalaire) -- l'invariant de conservation a verifier ;
- `sim.potential()` : potentiel electrostatique `phi`, tableau `(n, n)` ;
- `sim.time()` : temps physique courant ;
- `sim.block_names()` : noms des blocs, dans l'ordre d'ajout ;
- `sim.get_state(name)` : etat CONSERVATIF complet, `(ncomp, n, n)` (p.ex. `[rho, rho*u, rho*v, E]`
  pour Euler) ;
- `sim.get_primitive_state(name)` : etat rendu en variables PRIMITIVES, dict
  `{nom: (n, n)}` (inverse de `set_primitive_state`, round-trip a la precision machine).

```python
m0 = sim.mass("ne")
for _ in range(500):
    sim.step_cfl(0.4)
rho = sim.density("ne")             # ndarray (n, n)
phi = sim.potential()
print("derive masse :", abs(sim.mass("ne") - m0))   # ~ arrondi machine

U = sim.get_state("electrons")      # (4, n, n) = [rho, rho*u, rho*v, E]
P = sim.get_primitive_state("electrons")            # {"rho": ..., "u": ..., "v": ..., "p": ...}
```

Deux primitives servent a PILOTER le solveur depuis Python (integrateur en temps custom, oracle
de champ) :

- `sim.solve_fields()` : resout le Poisson de systeme sur l'etat courant et repeuple le canal
  `aux` (`phi`, `grad phi`) sans avancer en temps -- utile pour lire `potential()` a un etat fige ;
- `sim.eval_rhs(name)` : evalue le second membre `R = -div F + S` du bloc (le `dU/dt` spatial),
  `(ncomp, n, n)`, pour un integrateur en temps fourni par l'utilisateur.

Reference API condensee : [api](../reference/api_python.md). Recettes completes (figures, AMR) :
[examples](../getting_started/organisation.md). Tutoriel A->Z : [tutoriels](../getting_started/tutorial.md).

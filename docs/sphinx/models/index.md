# Modeles

Un *modele* dans `adc` decrit une equation : ses formules ponctuelles (flux, source,
vitesses d'onde, second membre elliptique). Il existe trois facons d'ecrire un modele, qui
produisent toutes le meme objet calculatoire cote coeur C++ et se branchent de la meme maniere sur
un `adc.System` :

1. **Modele avec briques (natif)** : on compose des briques generiques deja compilees
   (`adc.Model(state, transport, source, elliptic)`). C'est la voie la plus directe pour assembler
   un modele existant : aucune compilation a la volee, parite production totale (MPI/AMR/GPU).
2. **Modele DSL** : on ecrit le modele en formules symboliques (`adc.dsl.Model`), puis on le
   compile en un `.so`. C'est la voie quand le modele voulu n'existe pas comme brique native.
3. **Modele hybride** : on melange, dans un seul modele, des briques natives et des briques DSL
   partielles (`adc.CompositeModel`). C'est l'entre-deux : reutiliser une brique native pour un
   slot et ecrire l'autre en formules.

Ces trois objets sont des compositions de briques generiques. Le coeur reste agnostique au
scenario : il ne nomme aucun cas physique (diocotron, Euler-Poisson, deux-fluides...) ; ce sont des
compositions cote application. Pour le detail des methodes numeriques (reconstruction MUSCL/WENO,
flux de Riemann, integrateurs SSPRK/IMEX, Poisson multigrille), voir
[ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md). Pour l'architecture en couches (modele / maillage / dispatch /
integrateur), voir [ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md).

## PhysicalModel : le concept

Toutes les briques satisfont le meme contrat C++, le concept `adc::PhysicalModel`
([include/adc/core/physical_model.hpp](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/core/physical_model.hpp)). Un
`PhysicalModel` decrit une equation comme un jeu de fonctions pures d'etats ponctuels, rien de
plus. C'est le seul axe "quoi calculer" de l'architecture, separe de l'axe "ou / comment iterer"
(maillage + dispatch) et de l'axe "dans quel ordre" (integrateur + coupleur, cf.
[ARCHITECTURE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ARCHITECTURE.md)).

Le contrat minimal exige quatre fonctions :

- `flux(U, aux, dir)` : le flux physique dans la direction `dir` (0 = x, 1 = y) ;
- `max_wave_speed(U, aux, dir)` : la plus grande vitesse d'onde (pour le CFL et le solveur de
  Riemann) ;
- `source(U, aux)` : le terme source ponctuel ;
- `elliptic_rhs(U)` : le second membre de l'equation elliptique (densite de charge / de masse selon
  le modele).

Point d'unification : `flux` et `source` recoivent `aux` (le canal `adc::Aux` : potentiel `phi`,
gradient `grad_x`/`grad_y`, et champs etendus optionnels `B_z`, `T_e`). C'est ce qui place sous un
meme operateur spatial le transport a derive (l'`aux` est lu dans le flux) et le fluide compressible
auto-gravitant (l'`aux` est lu dans la source).

Une brique hyperbolique complete satisfait en plus `adc::HyperbolicPhysicalModel` : elle porte
les variables (conservatives et primitives) et les conversions `to_primitive` / `to_conservative`,
parce que variables, conversions et flux sont physiquement lies (un flux est ecrit pour une
disposition de variables donnee). C'est cette brique-la que l'on ecrit, native ou DSL.

## Modele avec briques (composition native)

`adc.Model(state, transport, source, elliptic)` compose un modele a partir de quatre briques
generiques deja compilees et renvoie une `ModelSpec` (des tags lus cote C++ par la fabrique de
modeles). Python compose les objets ; le calcul cellule par cellule reste C++ compile (pas de numpy,
GPU/MPI conserves). Les briques disponibles, telles qu'exposees par `adc.*` (et leurs structs C++ dans
[include/adc/physics/](https://github.com/wolf75222/adc_cpp/blob/master/include/adc/physics/)) :

**Etat** (`state=`)
- `adc.Scalar()` : etat scalaire (1 variable, p.ex. une densite transportee).
- `adc.FluidState(kind="compressible", gamma=1.4)` : Euler compressible (l'indice `gamma`).
- `adc.FluidState(kind="isothermal", cs2=0.5)` : Euler isotherme (la vitesse du son `cs2`).

**Transport** (`transport=`)
- `adc.ExB(B0=1.0)` : advection scalaire par la derive ExB (champ magnetique `B0`),
  `adc::ExBVelocity` dans `physics/hyperbolic.hpp`.
- `adc.CompressibleFlux()` : flux d'Euler compressible (`gamma` vient de l'etat),
  `adc::CompressibleFlux` (alias d'`adc::Euler`).
- `adc.IsothermalFlux()` : flux d'Euler isotherme (`cs2` vient de l'etat), `adc::IsothermalFlux`.

**Source** (`source=`)
- `adc.NoSource()` : pas de source, `adc::NoSource` dans `physics/source.hpp`.
- `adc.PotentialForce(charge=1.0)` : force du potentiel `(q/m) rho E` sur la quantite de mouvement
  (plus travail si 4 variables), `adc::PotentialForce`.
- `adc.GravityForce()` : force gravitationnelle `rho g`, `adc::GravityForce`.

**Second membre elliptique** (`elliptic=`)
- `adc.ChargeDensity(charge=1.0)` : densite de charge `f = q n`, `adc::ChargeDensity` dans
  `physics/elliptic.hpp`.
- `adc.BackgroundDensity(alpha=1.0, n0=0.0)` : fond neutralisant `f = alpha (n - n0)`,
  `adc::BackgroundDensity`.
- `adc.GravityCoupling(sign=1.0, four_pi_G=1.0, rho0=1.0)` : couplage self-consistant
  `f = sign * 4piG (rho - rho0)` (`sign = +1` gravite, `-1` plasma), `adc::GravityCoupling`.

`adc.Model(...)` valide la coherence etat <-> transport (Scalar avec ExB ; FluidState compressible
avec CompressibleFlux ; isotherme avec IsothermalFlux) : un appariement incoherent leve une
`ValueError` immediate.

Exemple, le modele diocotron reduit (densite scalaire advectee par ExB, fond neutralisant), tel
qu'utilise dans le tutoriel pour la comparaison uniforme/AMR :

```python
import adc

model = adc.Model(
    state=adc.Scalar(),
    transport=adc.ExB(B0=1.0),
    source=adc.NoSource(),
    elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0),
)

sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_block("ne", model=model, spatial=adc.Spatial(minmod=True), time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)          # ne0 : tableau 2D (densite initiale)
sim.step_cfl(0.4)
```

La meme `ModelSpec` se branche aussi sur `adc.AmrSystem` (raffinement adaptatif) sans changer le
modele : `sa.add_block("ne", model=model, ...)`.

## Modele DSL (ecrit en formules)

`adc.dsl.Model` permet d'ecrire un modele en formules symboliques : Python compose un arbre
d'expressions (les operateurs `+`, `-`, `*`, `/`, `**`, `adc.dsl.sqrt` construisent l'arbre, pas une
fonction appelee par cellule), que le DSL traduit en C++ compilable. On declare les variables
conservatives, les primitives (par des formules), le flux, les valeurs propres, la source et la
contribution elliptique, puis on compile.

Voici le modele diocotron reduit du tutoriel canonique
([docs/sphinx/tutorials/diocotron_tutorial.py](https://github.com/wolf75222/adc_cpp/blob/master/docs/sphinx/tutorials/diocotron_tutorial.py)), ecrit en
formules ; il reproduit exactement les briques natives `ExBVelocity` (transport) et
`BackgroundDensity` (elliptique) :

```python
import adc
from adc import dsl

B0 = 1.0      # champ magnetique de fond (porte la derive E x B)
ALPHA = 1.0   # facteur du second membre elliptique alpha (n - n_i0)

def diocotron_model(n_i0):
    m = dsl.Model("diocotron_tutorial")

    (n,) = m.conservative_vars("n")     # unique variable conservative : la densite (role Density)
    m.aux("phi")                        # champs auxiliaires fournis par le solveur (canal adc::Aux)
    grad_x = m.aux("grad_x")
    grad_y = m.aux("grad_y")

    vx = (-grad_y) / B0                  # derive E x B : v = (-d_y phi / B0, d_x phi / B0)
    vy = grad_x / B0
    m.flux(x=[n * vx], y=[n * vy])       # flux d'advection f = n v(dir)
    m.eigenvalues(x=[vx], y=[vy])        # spectre : une onde, la vitesse de derive

    m.primitive_vars(n=n)                # scalaire transporte : primitif = conservatif
    m.conservative_from([n])
    m.elliptic_rhs(ALPHA * (n - n_i0))   # couple le bloc au Poisson : rhs = alpha (n - n_i0)

    m.check()                            # toute variable referencee doit etre declaree
    return m

compiled = diocotron_model(n_i0).compile(backend="production")   # -> CompiledModel

sim = adc.System(n=96, L=1.0, periodic=True)
sim.add_equation("ne", model=compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 time=adc.Explicit())
sim.set_poisson(rhs="charge_density", solver="geometric_mg")
sim.set_density("ne", ne0)
sim.step_cfl(0.4)
```

Details et points de vigilance du DSL (parametres nommes `m.param`, roles physiques,
`require_metadata`, cache du `.so`) : voir la reference courte [DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md) et la
conception [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Modele hybride (briques native + DSL dans un seul modele)

`adc.Model(...)` compose des briques 100 % natives ; `adc.dsl.Model(...)` genere un modele 100 %
DSL. `adc.CompositeModel(transport, source, elliptic)` comble l'entre-deux : melanger, dans un seul
modele, des briques natives (`adc.ExB`, `adc.PotentialForce`, `adc.ChargeDensity`...) et des briques
DSL partielles compilees (`adc.dsl.HyperbolicBrick`, `adc.dsl.SourceBrick`, `adc.dsl.EllipticBrick`
suivies de `.compile()`).

Chaque slot accepte soit une brique native, soit une brique DSL partielle compilee. Au moins un
slot doit etre DSL : une composition tout-native s'ecrit avec `adc.Model(...)`, sinon
`CompositeModel` leve une `ValueError`. Le melange est compile en un `.so` composite (prototype :
backend `aot`), sur le meme chemin de production qu'un modele DSL complet ; la numerique native est
reutilisee a l'identique (un struct derive cuit les parametres natifs `qom`, `q`, `cs2`... dans le
type ; aucune re-derivation). Le slot transport fixe le layout (`n_vars`, noms conservatifs,
primitives, gamma) ; une brique DSL de source / elliptique doit declarer le meme `n_vars`.

Exemple, transport DSL isotherme + source native + elliptique native (extrait de
`python/tests/test_dsl_hybrid.py`) :

```python
import adc
from adc import dsl

CS2, QOM, Q = 0.7, -1.0, -1.0

# Brique hyperbolique DSL repliquant adc::IsothermalFlux{cs2} (3 variables).
def build_iso_transport(cs2):
    b = dsl.HyperbolicBrick("iso")
    rho, rho_u, rho_v = b.conservative_vars("rho", "rho_u", "rho_v")
    u = b.primitive("u", rho_u / rho)
    v = b.primitive("v", rho_v / rho)
    c = dsl.sqrt(cs2)
    b.flux(x=[rho_u, rho_u * u + cs2 * rho, rho_v * u],
           y=[rho_v, rho_u * v, rho_v * v + cs2 * rho])
    b.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    b.primitive_vars(rho, u, v)
    b.conservative_from([rho, rho * u, rho * v])
    return b

m = adc.CompositeModel(
    transport=build_iso_transport(CS2).compile(),  # transport DSL
    source=adc.PotentialForce(charge=QOM),         # source native
    elliptic=adc.ChargeDensity(charge=Q),          # elliptique native
)
compiled = m.compile(backend="aot")                # -> CompiledModel (adder add_compiled_block)

sim = adc.System(n=48, L=1.0, periodic=True)
sim.add_equation("gas", compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 names=["rho", "rho_u", "rho_v"])
```

Le melange fonctionne dans les deux sens (transport natif + source/elliptique DSL aussi). Source :
[DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md) section "composition hybride" et
`python/tests/test_dsl_hybrid.py`.

## Variables conservatives / primitives

Un modele DSL distingue deux jeux de variables, avec des roles physiques qui permettent au systeme
de retrouver une grandeur par son sens (et non par un indice litteral), indispensable aux couplages
inter-especes.

- `m.conservative_vars("rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"])` declare les
  variables conservatives (l'etat evolue `U`) et renvoie un tuple de `Var` a depacker. Le `roles=`
  est optionnel ; sans lui, un mapping canonique nom -> role s'applique (`rho`/`n` -> `Density`,
  `rho_u` -> `MomentumX`, `E` -> `Energy`...). Un nom non reconnu reste `Custom`.
- `m.primitive(name, expr)` definit une primitive par sa formule (en fonction des conservatives ou
  des primitives precedentes), p.ex. `u = m.primitive("u", mx / rho)`.
- `m.primitive_vars(rho=rho, ux=mx/rho, ...)` (forme kwargs) definit chaque primitive et fixe le
  layout ordonne de `Prim` (l'ordre des kwargs). La forme positionnelle
  `m.primitive_vars(rho, u, v, p)` fixe juste le layout a partir de noms deja definis.
- `m.conservative_from([rho, rho*u, rho*v])` donne l'inverse `Prim -> U` (le DSL ne sait pas inverser
  symboliquement les primitives ; on fournit l'inverse explicitement). Il genere `to_conservative`.

L'operateur spatial peut alors reconstruire en variables primitives (`rho`, `u`, `p`) plutot que
conservatives, plus stable pour Euler (positivite de `rho` et `p`) ; voir le choix
`variables="primitive"` de `adc.FiniteVolume` et les details dans [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## Flux, sources, valeurs propres, RHS elliptique

Ces quatre declarateurs sont le coeur du modele DSL ; ils correspondent un a un aux fonctions du
concept `adc::PhysicalModel` lues par le coeur.

- `m.flux(x=[...], y=[...])` : le **flux physique** `F(U, aux, dir)`, une expression par composante
  conservative et par direction. L'operateur spatial l'evalue aux interfaces puis le passe au
  solveur de Riemann (Rusanov / HLLC / Roe selon `riemann=`). A ne pas confondre avec
  `m.eval_flux(U, aux, dir)`, qui est l'evaluateur numpy (debug / proto hote), ni avec le flux
  numerique `riemann=` de `adc.FiniteVolume`.
- `m.eigenvalues(x=[...], y=[...])` : les **valeurs propres** (vitesses caracteristiques) par
  direction. Le coeur en tire `max_wave_speed` (borne de Rusanov et pas de temps CFL) ; si une
  primitive `p` (pression) est declaree, la brique generee expose aussi `pressure` / `wave_speeds`,
  ce qui la rend compatible avec les flux HLLC / Roe (qui exigent une pression).
- `m.source([...])` : le **terme source** `S(U, aux)`, une expression par composante (optionnel). Il
  lit l'etat exterieur par le canal `adc::Aux` (p.ex. `grad_x` / `grad_y` pour une force de
  potentiel `-rho grad phi`).
- `m.elliptic_rhs(expr)` : la **contribution au second membre elliptique**, qui couple le bloc au
  Poisson de systeme (densite de charge `q n`, fond `alpha (n - n0)`, gravite...). Le Poisson de
  systeme somme les contributions de tous les blocs.

`m.check()` verifie que toute variable referencee (dans les primitives, le flux, les valeurs
propres, la source, l'elliptique) est bien declaree (conservative / primitive / aux), et leve une
`ValueError` factuelle sinon. Pour la signification physique et la discretisation de chaque operateur
(reconstruction, Riemann, multigrille), voir [ALGORITHMS.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/ALGORITHMS.md).

## Compilation : production / AOT / prototype

`m.compile(backend=..., target=...)` traduit le modele symbolique en un `.so` et renvoie un
`CompiledModel` (qui porte `so_path`, `backend`, l'`adder` a employer, les noms/roles/gamma/n_aux,
la `abi_key` et le `model_hash`). Le `.so` est mis en cache par `model_hash` : un modele inchange
n'est pas recompile. Le defaut est `backend="aot"` ; il faut donc demander explicitement
`"production"` pour le chemin natif zero-copie.

Trois backends, materialises cote code dans `_BACKEND_CAPS` (`python/adc/dsl.py`) :

| backend | CPU | MPI | AMR | GPU | role |
|---|---|---|---|---|---|
| `production` | oui | oui (np=1/2/4) | via `AmrSystem` | rapporte `False` cote Python | recommande en MPI/AMR ; loader natif zero-copie (`add_native_block`) |
| `aot` | oui | non | non | non | defaut ; `.so` a marshaling, mono-rang, debug/bench CPU. Porte les params runtime (`set_block_params`) |
| `prototype` | oui (Rusanov o1) | non | non | non | JIT prototype, dispatch virtuel hote ; ne pas utiliser en production |

`_BACKEND_CAPS["production"]` declare `{cpu, mpi, amr} = True`. Le chemin natif `production` partage
le moteur de `add_block` (halos `fill_boundary`, donc MPI-capable par construction) et a un pendant
AMR (`m.compile(backend="production", target="amr_system")` -> `AmrSystem.add_native_block`). `gpu`
est rapporte `False` par prudence : le chemin natif est device-clean en C++ (valide GH200), mais la
validation end-to-end depuis Python sur un module bati Kokkos/CUDA reste une etape dediee ; le module
hote teste en CI n'est pas bati GPU.

Ces capacites sont des drapeaux de diagnostic, verifies au branchement (`add_equation`) ou a
l'execution, pas figes comme un argument `device=` de compilation (un `.so` peut compiler sans que
le module hote soit device-capable). Les garde-fous levent une `ValueError` au plus tot :

- backend inconnu (hors `prototype`/`aot`/`production`) ;
- `target="amr_system"` avec un backend autre que `production` (pas de chemin `.so` AMR hors natif) ;
- `compile(backend="prototype", require_metadata=True)` (le JIT ne transporte pas les metadonnees
  utiles) ;
- cote branchement : `riemann` HLLC/Roe sans pression `p` declaree, `names=` sur le chemin natif
  `production` (les noms viennent des metadonnees du `.so`).

Pour brancher le `CompiledModel`, `System.add_equation` aiguille selon le type : une `ModelSpec`
(`adc.Model(...)`) -> `add_block` (natif) ; un `CompiledModel` -> l'adder du backend
(`add_dynamic_block` pour `prototype`, `add_compiled_block` pour `aot`, `add_native_block` pour
`production`). Detail complet : [DSL_API.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_API.md) et
[DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md). Couverture des backends sur GPU/MPI/AMR :
[BACKEND_COVERAGE.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).

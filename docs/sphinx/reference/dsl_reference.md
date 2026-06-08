# Reference : le DSL symbolique (adc.dsl)

Le DSL `adc.dsl` permet d'ecrire la physique d'un modele comme un arbre d'expressions
symboliques (les operateurs Python `+ - * / ** -` et `dsl.sqrt` construisent l'arbre, pas une
fonction appelee par cellule), que le DSL traduit en C++ compilable puis compile en un `.so`
branchable sur un `adc.System` / `adc.AmrSystem`. Deux points d'entree : `adc.dsl.Model(name)`,
la facade stable recommandee (pur sucre, composition d'un `HyperbolicModel` prive `_m`), et
`adc.dsl.HyperbolicModel(name)`, l'objet backend de plus bas niveau (nommage `set_*`, toujours
utilisable directement). Les deux exposent `compile()`. Cette page est le registre canonique du
DSL. Source : [python/adc/dsl.py](https://github.com/wolf75222/adc_cpp/blob/master/python/adc/dsl.py) ;
conception : [DSL_MODEL_DESIGN.md](https://github.com/wolf75222/adc_cpp/blob/master/docs/DSL_MODEL_DESIGN.md).

## Declarer un modele

Toutes les methodes ci-dessous sont sur la facade `adc.dsl.Model`. Elles deleguent a
`HyperbolicModel` (colonne de droite implicite). `flux`, `eigenvalues`, `source`, `elliptic_rhs`
correspondent un a un aux fonctions du concept `adc::PhysicalModel` lues par le coeur.

| methode | ce qu'elle declare | exemple |
|---|---|---|
| `conservative_vars(*names, roles=None)` | les variables conservatives (l'etat `U`) ; renvoie un tuple de `Var` | `rho, mx, my, E = m.conservative_vars("rho","rho_u","rho_v","E")` |
| `primitive(name, expr)` | une primitive par sa formule ; renvoie un `Var` | `u = m.primitive("u", mx/rho)` |
| `primitive_vars(*vars, roles=None, **named)` | le layout ordonne de `Prim` (et, en kwargs, definit chaque primitive) | `prho,pu,pv,pp = m.primitive_vars(rho=rho,u=u,v=v,p=p)` |
| `aux(name)` | un champ auxiliaire fixe lu a l'execution ; renvoie un `Var` | `gx = m.aux("grad_x")` |
| `flux(x, y)` | le flux physique `F(U)`, une `Expr` par composante et direction | `m.flux(x=[...], y=[...])` |
| `eigenvalues(x, y)` | les vitesses caracteristiques par direction | `m.eigenvalues(x=[u-c,u,u+c], y=[v-c,v,v+c])` |
| `conservative_from(exprs)` | l'inverse `Prim -> U` (a fournir, le DSL n'inverse pas) | `m.conservative_from([rho, rho*u, rho*v, ...])` |
| `source(s)` | le terme source `S(U, aux)` (optionnel), une `Expr` par composante | `m.source([0.0, -rho*gx, -rho*gy, ...])` |
| `elliptic_rhs(e)` | la contribution au second membre du Poisson de systeme (optionnel) | `m.elliptic_rhs(-1.0*(rho - 1.0))` |
| `gamma(value)` | l'indice adiabatique (EOS), exporte dans le `.so` | `m.gamma(1.4)` |
| `param(name, value, kind="const")` | un parametre nomme utilisable dans les formules ; renvoie un `Param` | `g = m.param("gamma", 1.4)` |
| `check()` | verifie que toute variable referencee est declaree | `m.check()` |

### conservative_vars

Declare le vecteur d'etat conservatif et renvoie un `Var` par nom. Le `roles=` est optionnel et de
meme longueur que `names` ; sans lui, le mapping canonique nom -> role s'applique (`rho`/`n` ->
`Density`, `rho_u` -> `MomentumX`, `E` -> `Energy`...), un nom inconnu reste `Custom`. Une longueur
de `roles` differente leve une `ValueError`. Les roles permettent au `System` de resoudre les
couplages inter-especes par `index_of(role)` plutot que par un indice litteral.

```python
rho, rhou, rhov, E = m.conservative_vars(
    "rho", "rho_u", "rho_v", "E",
    roles=["Density", "MomentumX", "MomentumY", "Energy"])
```

### primitive

Definit une primitive par sa formule (en fonction des conservatives, des primitives precedentes ou
de l'aux). L'ordre d'insertion = l'ordre de dependance.

```python
u = m.primitive("u", rhou / rho)
p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
```

### primitive_vars

Fixe le layout ordonne de `Prim` ; deux formes exclusives (les melanger leve une `ValueError`) :

- forme kwargs (style cible) : `primitive_vars(rho=expr, u=expr, v=expr, p=expr)` definit chaque
  primitive et fixe le layout dans l'ordre d'insertion des kwargs (Python 3.7+). Renvoie un tuple de
  `Var`.
- forme positionnelle : `primitive_vars(rho, u, v, p, roles=...)` ne fait que fixer le layout a
  partir de noms / `Var` deja definis. Renvoie `None`.

Garde-fou d'auto-reference : en kwargs, si la valeur est le `Var` du meme nom (p.ex. `u=u` avec `u`
deja issu de `m.primitive("u", ...)`) ou si le nom est deja une conservative (`rho=rho`), la
primitive n'est pas redefinie ; elle rejoint juste le layout. Sans ce garde-fou, le codegen
emettrait `const Real u = u;` (auto-init -> NaN).

```python
prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)   # definit ET ordonne
```

### aux

Declare un champ auxiliaire lu a l'execution (canal `adc::Aux`) ; renvoie un `Var` lu en C++ comme
`a.<name>`. Le nom doit etre une clef de la table fixe (un nom inconnu leve une `ValueError`) :

| nom | indice | role |
|---|---|---|
| `phi` | 0 | potentiel |
| `grad_x` | 1 | gradient x du potentiel |
| `grad_y` | 2 | gradient y du potentiel |
| `B_z` | 3 | champ magnetique (canal etendu) |
| `T_e` | 4 | temperature electronique (canal etendu) |

`phi`/`grad_x`/`grad_y` sont le contrat de base (3 composantes). Utiliser `B_z` ou `T_e` elargit le
canal : la brique generee declare alors `n_aux` (4 ou 5) pour que le systeme dimensionne le canal
partage.

```python
gx, gy = m.aux("grad_x"), m.aux("grad_y")   # champ E = -grad phi
```

### flux

Declare le flux physique `F(U)`. `x` et `y` sont des listes d'`Expr`, une par composante
conservative. A ne pas confondre avec l'evaluateur numpy `m.eval_flux(U, aux, dir)` (debug / proto
hote, qui renvoie un tableau numpy stacke) ni avec le flux numerique `riemann=` de
`adc.FiniteVolume` (Rusanov / HLLC / Roe).

```python
m.flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u],
       y=[rhov, rhov*u, rhov*v + p, rho*H*v])
```

### eigenvalues

Declare les vitesses caracteristiques par direction (listes d'`Expr`). Le coeur en tire
`max_wave_speed` (borne de Rusanov et pas de temps CFL) ; si une primitive nommee `p` (pression)
existe, la brique generee expose aussi `pressure` / `wave_speeds`, ce qui la rend compatible avec
les flux HLLC / Roe. Sans `p`, le modele reste limite a Rusanov.

```python
m.eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])
```

### conservative_from

Donne l'inverse `Prim -> U`, une `Expr` par conservative dans l'ordre de `conservative_vars`. Le
DSL ne sait pas inverser les primitives symboliquement ; l'utilisateur fournit l'inverse a la main.
Genere `to_conservative`. Obligatoire pour le codegen complet de la brique (`emit_cpp_brick`).

```python
m.conservative_from([rho, rho*u, rho*v, p/(g-1.0) + 0.5*rho*(u*u + v*v)])
```

### source

Terme source `S(U, aux)` optionnel, une `Expr` par composante (les scalaires sont promus en
`Const`). Lit l'etat exterieur par le canal `adc::Aux` (p.ex. `grad_x` / `grad_y` pour une force de
potentiel). Sans `source`, la brique vaut `adc::NoSource`.

```python
m.source([0.0, -rho*gx, -rho*gy, -(rhou*gx + rhov*gy)])
```

### elliptic_rhs

Contribution au second membre elliptique (couplage Poisson du systeme : densite de charge, fond
neutralisant, gravite), une seule `Expr`. Le Poisson de systeme somme les contributions de tous les
blocs. Sans elle, le rhs du bloc est nul.

```python
m.elliptic_rhs(-1.0 * (rho - 1.0))   # gravite self-consistante sign=-1, rho0=1
```

### gamma et param

`gamma(value)` fixe l'indice adiabatique, transporte par le `.so` (symbole optionnel) pour que les
couplages du `System` utilisent le bon gamma au lieu du defaut historique 1.4.

`param(name, value, kind="const")` declare un parametre nomme utilisable comme une `Expr`, range
dans `m.params` (introspection / reproductibilite). Deux modes :

- `kind="const"` (defaut) : la valeur est inlinee en dur au codegen (litteral dans le `.so`), tout
  en gardant son identite pour l'introspection.
- `kind="runtime"` : la valeur emet `params.get(<indice>)` (lecture d'un membre
  `adc::RuntimeParams`), modifiable a l'execution via `System.set_block_params(name, values)` sans
  recompiler. Supporte par le backend `aot` uniquement ; sur `prototype` / `production` un param
  runtime est fige a sa valeur de declaration.

Cas special `name == "gamma"` : `param` appelle aussi `set_gamma(value)` pour que la metadonnee ABI
reste coherente.

```python
g   = m.param("gamma", 1.4)                   # const : inline + set_gamma
cs2 = m.param("cs2", 1.0, kind="runtime")     # runtime : params.get(0), ecrasable (aot)
```

`adc.dsl.RuntimeParam(name, value)` est un sucre equivalant a `Param(name, value, kind="runtime")`.

```{note}
Sur `HyperbolicModel`, ces declarateurs portent le prefixe `set_` (`set_flux`, `set_eigenvalues`,
`set_source`, `set_elliptic_rhs`, `set_gamma`, `set_primitive_state`, `set_conservative_from`) ;
`cons(name)` ajoute une seule conservative. La facade `Model` les renomme en formes declaratives.
`param` n'existe que sur la facade `Model`.
```

## Algebre d'expressions

Toute formule est un arbre d'`Expr`. Les feuilles sont `Var`, `Const`, `Param`, `RuntimeParamRef`,
`_CsField` ; les operateurs construisent des noeuds composes. Operateurs supportes :

| Python | noeud | C++ emis |
|---|---|---|
| `a + b` | `Add` | `(a + b)` |
| `a - b` | `Sub` | `(a - b)` |
| `a * b` | `Mul` | `(a * b)` |
| `a / b` | `Div` | `(a / b)` |
| `a ** b` | `Pow` | `std::pow(a, b)` |
| `-a` | `Neg` | `(-a)` |
| `+a` | identite | renvoie le noeud interne |

Les scalaires Python (`int` / `float`) sont auto-promus en `Const(float(o))` (via `_wrap`). Un
`Param` est promu par son noeud interne (`Const` pour const, `RuntimeParamRef` pour runtime), donc
`dsl.sqrt(param_runtime)` emet correctement `params.get(...)` et non la valeur figee.

`dsl.sqrt(x)` (-> `std::sqrt(...)`) est la seule fonction mathematique nommee de l'algebre. Tout le
reste passe par les operateurs : pour un carre ou une racine on ecrit `x*x` ou `x**0.5`.

Ce qui n'est pas supporte, volontairement :

- pas de `exp`, `log`, `min`, `max`, `abs`, ni conditionnel / ternaire. Les `min` / `max` sur les
  valeurs propres sont generes en interne cote C++ a partir de `eigenvalues`, pas exposes comme
  operateurs.
- pas d'operateur `grad` / `div` ni de differentiation symbolique : les derivees spatiales arrivent
  par les champs `aux("grad_x")` / `aux("grad_y")` fournis par le solveur, pas par l'algebre.
- pas d'indexation sur une `Expr` : les composantes sont adressees par la position dans les listes
  Python de `flux` / `source` / `conservative_from`.

CSE (elimination des sous-expressions communes) : `cse=True` (defaut de tous les emetteurs) factorise
les sous-expressions compound repetees en locales `cseK_`, en ordre de dependance (les plus petites
d'abord), via une clef structurelle par noeud (deux sous-arbres identiques partagent une locale ; un
param runtime se cle par son nom).

## Compiler

`m.compile(...)` traduit le modele symbolique en un `.so` et renvoie un `CompiledModel`. La facade :

```python
Model.compile(so_path=None, include=None, backend="aot", target="system",
              name=None, cxx=None, std=None, require_metadata=False)
```

Le backing `HyperbolicModel.compile(...)` a la meme signature (ordre `backend, name, cxx, std,
require_metadata, target`) et renvoie le `so_path` (chaine).

Semantique des arguments :

- `so_path=None` : cache hors source (`adc_cache_dir()` : `$ADC_CACHE_DIR`, sinon
  `$XDG_CACHE_HOME/adc/dsl`, sinon `~/.cache/adc/dsl`). Le nom de fichier est keye sur `model_hash`
  + `abi_key` (+ backend / target / name). Cache hit (le `.so` existe deja pour cette clef) -> aucune
  recompilation. Passer `so_path=` force ce chemin et recompile toujours.
- `include=None` : auto-detecte par `adc_include()` (`$ADC_INCLUDE`, sinon le paquet `adc` installe,
  sinon le depot voisin). Critere de validite : `adc/mesh/multifab.hpp` existe ; sinon `RuntimeError`.
- `cxx=None` : autodetect `c++` / `g++` / `clang++` (via `shutil.which`).
- `std=None` : defaut par backend. Pour `production` (natif), la norme du loader via
  `loader_cxx_std()` (= `_adc.__cxx_std__` : c++20 sous Kokkos car CUDA 12.x n'a pas `-std=c++23`,
  c++23 sinon). Pour `prototype` / `aot`, `"c++20"`.
- `require_metadata=False` : si `True`, exige des roles physiques utiles et un `gamma` explicite,
  faute de quoi le `.so` retomberait sur les defauts du `System` (roles `custom` / gamma 1.4). Ce
  garde-fou tourne avant le cache. Incompatible avec `prototype` (leve une `ValueError`).
- `name` : override optionnel du nom de base du struct / type genere.

### Les trois backends

| backend | moteur | adder System | numerique | CPU | MPI | AMR | GPU | quand |
|---|---|---|---|---|---|---|---|---|
| `prototype` | JIT (`compile_so`) | `add_dynamic_block` | `IModel` virtuel, residu hote, Rusanov ordre 1 seul | oui | non | non | non | iteration rapide / debug |
| `aot` | AOT (`compile_aot`) | `add_compiled_block` | `.so` ABI plate, chemin de production (HLLC/Roe, ordre 2, WENO5) mais grille locale mono-rang a marshaling (non zero-copie) | oui | non | non | non | defaut ; debug / bench CPU ; seul a porter les params runtime |
| `production` | natif (`compile_native`) | `add_native_block` | loader `.so` qui inline `add_compiled_model<ProdModel>` sur le `grid_context()` -> zero-copie, meme chemin que `add_block`, foncteurs nommes | oui | oui | via `AmrSystem` | rapporte `False` (hote non-Kokkos) | recommande en MPI / AMR |

Le defaut du code est `backend="aot"` : il faut demander explicitement `"production"` pour le chemin
natif zero-copie. Les capacites sont materialisees dans `_BACKEND_CAPS` :
`production` declare `{cpu, mpi, amr} = True`. `gpu` est rapporte `False` par prudence : le chemin
natif est device-clean en C++ (valide GH200, foncteurs nommes), mais la validation end-to-end depuis
Python sur un module bati Kokkos/CUDA reste une etape dediee et le module hote teste en CI n'est pas
bati GPU. Ces capacites sont des drapeaux de diagnostic, verifies au branchement (`add_equation`) ou
a l'execution, et non un argument `device=` fige a la compilation.

### Modeles hybrides (natif + DSL)

On peut melanger des briques natives et des briques DSL partielles dans un seul modele via
`adc.CompositeModel(transport, source, elliptic)`, qui renvoie un `dsl.HybridModel` ; son
`.compile(backend="aot")` rend un `CompiledModel` branchable par `add_equation`. Au moins un slot doit
etre une brique DSL (sinon utiliser `adc.Model(...)`). Catalogue des briques et exemple :
[reference des briques](bricks_reference.md).

### Cle d'ABI (production)

Le loader `production` appelle des methodes hors-ligne du module `_adc` deja charge
(`install_block` / `grid_context` / `ensure_aux_width` ; `set_compiled_block` pour l'AMR), donc il
est compile avec `-undefined dynamic_lookup` sur macOS et bake `-DADC_HEADER_SIG=<signature>` a
l'identique du build du module. Loader et module doivent partager la meme ABI (en-tetes + compilateur
+ norme C++). `add_native_block` compare `adc_native_abi_key()` a `module.abi_key()` et rejette avec
"ABI incompatible" s'ils divergent. Un `std` different change `__cplusplus` donc la clef d'ABI : c'est
pourquoi `std=None` derive la norme du loader au lieu de figer c++23. Les backends `prototype` / `aot`
ne ciblent que `System` et ne se cross-branchent pas.

### target='system' vs 'amr_system'

- `target="system"` (defaut) : facade `adc::System`. Le loader natif emet le symbole
  `adc_install_native(System&, ..., evolve, stride)`. Tous les backends sont permis.
- `target="amr_system"` : facade `adc::AmrSystem`. Valide uniquement avec `backend="production"` (les
  autres levent une `ValueError`, il n'existe pas de chemin `.so` AMR hors natif). Le loader inclut
  `amr_dsl_block.hpp` et emet un symbole distinct `adc_install_native_amr(AmrSystem&, ...)` (sans
  argument `evolve`) -> `add_compiled_model(AmrSystem&)` (reflux conservatif, regrid). On branche via
  `AmrSystem.add_native_block`. Un loader System ne se branche pas sur AmrSystem et inversement.

### CompiledModel

L'objet renvoye par `Model.compile`. Il porte : `so_path`, `backend`, `target`, `adder`,
`cons_names`, `cons_roles`, `prim_names`, `n_vars`, `gamma`, `n_aux`, `params` (dict nom -> `Param`),
`caps` (cpu/mpi/amr/gpu), `abi_key`, `model_hash`, `cxx`, `std`. Proprietes :
`runtime_param_names` (params runtime tries, = l'ordre des indices C++ et l'ordre attendu par
`set_block_params`) et `runtime_param_values()`. Les metadonnees ne sont pas relues du `.so` : Python
les detient deja. On branche via `System.add_equation(name, compiled, ...)`, qui aiguille selon
`compiled.adder`.

## Sources couplees inter-especes (CoupledSource)

`adc.dsl.CoupledSource` decrit un echange inter-especes arbitraire en formules (au-dela des couplages
nommes Ionization / Collision / ThermalExchange). Il compile en bytecode plat (machine a pile, pas de
`.so`, pas de callback Python par cellule) interprete dans un `for_each_cell` device. Applique en
splitting explicite, apres le transport.

- `CoupledSource(name="coupled_source")` cree la source.
- `src.block(name).role(role)` -> un `_CsField` (une `Var` de nom d'environnement `"<bloc>::<role>"`).
  Les roles sont canonises en lowercase (`density`, `momentum_x`, `energy`, `velocity_x`, `pressure`,
  `temperature`, `scalar`...).
- `src.param(name, value)` -> un `Param` const (inline comme registre constant du bytecode).
- `src.add(block, role=, expr=)` ajoute un terme `d_t (block.role) += expr` ; plusieurs `add` sur le
  meme `(block, role)` s'additionnent.
- `src.add_pair(block_a, block_b, role=, expr=)` : echange conservatif par construction. `block_a`
  gagne `+expr`, `block_b` perd `-expr` (le meme sous-arbre, en `Neg`), donc `sum(role)` est conserve
  cellule par cellule. `block_a != block_b` requis. Chainable (renvoie `self`).
- `src.compile(backend="production", verify_conservation=False)` -> `CompiledCoupledSource`. Le
  backend documente l'intention ; la numerique est identique (bytecode interprete cote C++).

Opcodes (miroir de `adc::CsOp`) : `PUSHREG`(0), `ADD`(1), `SUB`(2), `MUL`(3), `DIV`(4), `NEG`(5),
`POW`(6), `SQRT`(7). Seuls `+ - * / ** -unaire sqrt` plus champ et constante sont supportes (tout
autre noeud leve une `TypeError`). Limites de capacite figees, diagnostiquees cote Python avant la
frontiere C++ : 32 registres (entrees + constantes), 16 termes de source, 256 opcodes par terme.

`CompiledCoupledSource` porte l'ABI plate (`in_blocks`, `in_roles` canoniques, `consts`,
`out_blocks`, `out_roles`, `prog_ops`, `prog_args`, `prog_lens`) et un evaluateur numpy de reference
`reference_terms(fields)` (memes `Expr` que le bytecode). On branche via `sim.add_coupling(compiled)`
(-> `System.add_coupled_source`).

```python
from adc import dsl

src = dsl.CoupledSource("ionization")
ne = src.block("electrons").role("density")
ni = src.block("ions").role("density")
ng = src.block("neutrals").role("density")
k  = src.param("Kiz", 0.7)
src.add("electrons", role="density", expr=+k*ne*ng)
src.add("ions",      role="density", expr=+k*ne*ng)
src.add("neutrals",  role="density", expr=-k*ne*ng)
sim.add_coupling(src.compile(backend="production"))
```

`verify_conservation=True` controle symboliquement, role par role, que la somme des termes s'annule
(chaque `+E` compense par un `-E` de meme corps structurel sur un autre bloc) ; il leve une
`ValueError` explicite sinon. C'est ce que `add_pair` garantit par construction ; le flag etend le
controle aux couplages ecrits a la main. Le controle est conservateur : il peut signaler a tort un
couplage ecrit avec des formes algebriquement egales mais structurellement differentes (`k*ne` vs
`ne*k`), jamais l'inverse. Off par defaut : une creation / destruction nette volontaire (ionisation)
reste licite sans le flag.

## Validation (check)

`m.check()` collecte `known = cons_names + prim_defs + aux_names`, puis `used` = toutes les
dependances (`deps()`) apparaissant dans chaque definition de primitive, dans `flux` x/y, dans
`eigenvalues` x/y, dans `source` et dans `elliptic_rhs`. Si `used - known` est non vide, il leve
`ValueError("modele '<name>' : variables non definies [...]")`. Renvoie `True` sinon.

`check()` ne verifie pas : que `flux` / `eigenvalues` / `source` ont ete poses, la completude du
layout, la validite des roles, ni la conservation. Ces erreurs surfacent plus tard :

- `emit_cpp_brick` exige `set_primitive_state`, `set_conservative_from` (n_vars exprs), `set_flux`
  (n_vars/dir) et `set_eigenvalues`, sinon `ValueError` a l'emission.
- `aux(name)` leve sur un nom d'aux inconnu.
- l'attribution des indices runtime leve si le modele depasse `kMaxRuntimeParams=32`.
- au branchement, `add_equation` rejette HLLC/Roe sans primitive `p`, une longueur de `names=`
  incorrecte, et `names=` sur le chemin natif `production` (les noms viennent du `.so`).
- `CoupledSource.compile(verify_conservation=True)` controle la conservation role par role.

## Exemple complet

Modele Euler avec couplage gravite-Poisson, de la declaration au run (copier-coller). La compilation
demande des en-tetes adc et un compilateur C++ ; sans eux, `compile` leve.

```python
import numpy as np
import adc
from adc import dsl

GAMMA = 1.4

def build_euler_poisson():
    m = dsl.Model("euler_poisson")
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.param("gamma", GAMMA)                      # const : inline + set_gamma
    u = m.primitive("u", rhou / rho)
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u*u + v*v)))
    H = (E + p) / rho
    c = dsl.sqrt(g * p / rho)
    m.flux(x=[rhou, rhou*u + p, rhou*v, rho*H*u],
           y=[rhov, rhov*u, rhov*v + p, rho*H*v])
    m.eigenvalues(x=[u-c, u, u+c], y=[v-c, v, v+c])
    gx, gy = m.aux("grad_x"), m.aux("grad_y")        # E = -grad phi
    m.source([0.0, -rho*gx, -rho*gy, -(rhou*gx + rhov*gy)])
    m.elliptic_rhs(-1.0 * (rho - 1.0))               # gravite self-consistante
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([prho, prho*pu, prho*pv,
                         pp/(g-1.0) + 0.5*prho*(pu*pu + pv*pv)])
    m.check()
    return m

m = build_euler_poisson()
compiled = m.compile(backend="production")           # include / so_path auto, cache

n = 32
s = adc.System(n=n, L=1.0, periodic=True)
s.add_equation("gas", compiled,
               spatial=adc.FiniteVolume(limiter="minmod", riemann="hllc",
                                        variables="primitive"))
s.set_poisson(rhs="charge_density", solver="geometric_mg")

xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs)
U = np.zeros((4, n, n))
U[0] = 1.0 + 0.3*np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.02)
U[3] = 1.0/(GAMMA - 1.0)
s.set_state("gas", U.reshape(-1).tolist())

nsteps = s.run(t_end=0.02, cfl=0.4)
final = np.array(s.get_state("gas")).reshape(4, n, n)
```

Note sur `adc.FiniteVolume(limiter=, riemann=, variables=)` : `riemann` est le flux numerique
(`rusanov` / `hll` / `hllc` / `roe`), distinct du flux physique `m.flux` ; `limiter` parmi
`none` / `minmod` / `vanleer` / `weno5` ; `variables` parmi `conservative` / `primitive`. HLLC / Roe
exigent une primitive nommee `p`.

## Pieges

1. **Deux `Model`**. `adc.Model(state, transport, source, elliptic)` (dans `__init__.py`) compose des
   briques natives pre-compilees (`ModelSpec`) ; `adc.dsl.Model(name)` ecrit des formules
   symboliques. Signatures et fichiers differents.
2. **`flux` vs `eval_flux`**. `m.flux(x=, y=)` declare ; `m.eval_flux(U, aux, dir)` evalue (numpy).
   Methodes distinctes. Sur `HyperbolicModel`, la collision est resolue dans l'autre sens :
   `flux(U, aux, dir)` evalue et `set_flux` declare.
3. **`primitive_vars(u=u, ...)` auto-reference**. Passer un `Var` deja defini du meme nom (ou une
   conservative) ne le redefinit pas, sinon le codegen emettrait `const Real u = u;` -> NaN.
4. **`conservative_from` est obligatoire et manuel**. Le DSL n'inverse pas les primitives ; il faut
   fournir `cons = f(prim)` explicitement pour le codegen complet de la brique.
5. **Noms d'aux fixes**. Seuls `phi` / `grad_x` / `grad_y` / `B_z` / `T_e`. Un nom inconnu leve.
   `B_z` / `T_e` elargissent `n_aux` (4 / 5).
6. **Concordance d'ABI (production)**. Loader et module `_adc` doivent partager en-tetes + compilateur
   + norme C++ ; un ecart -> `add_native_block` leve "ABI incompatible". Ne pas forcer `std` pour
   `production` : laisser `std=None` deriver la norme du loader (c++20 sous Kokkos, c++23 sinon).
7. **Params runtime sur `aot` seulement**. `kind="runtime"` est fige a la valeur de declaration sur
   `prototype` / `production`. `set_block_params` sur un bloc 100 % const leve. `runtime_param_names`
   trie = l'ordre des indices C++ = l'ordre attendu par `set_block_params(name, values)`.
8. **`target="amr_system"` exige `backend="production"`**. Pas de chemin `.so` AMR hors natif. En AMR,
   HLLC / Roe / `primitive` sont rejetes a la facade Python (le moteur C++ les supporte : limitation
   de facade).
9. **Cle de cache**. `so_path=None` cache par `model_hash` (formules + roles + n_aux + gamma + params,
   valeurs de declaration runtime comprises) et `abi_key`. Changer une formule / un param / la
   toolchain -> cache miss -> recompilation. Passer `so_path=` force la recompilation. Les params
   runtime changent a l'execution via `set_block_params` sans recompiler (sur `aot`).
10. **Device-clean**. `production` utilise des foncteurs nommes, pas des lambdas etendues
    `__host__ __device__` (qui segfaultaient sous nvcc). C'est le seul chemin valide GPU / MPI.
11. **CoupledSource**. Pas de `.so`, pas de Python par cellule ; bytecode interprete en C++. Seuls
    `+ - * / ** -unaire sqrt` ; limites 32 registres / 16 termes / 256 opcodes. `add_pair` garantit la
    conservation par construction ; `verify_conservation=True` controle les `.add` ecrits a la main
    (controle conservateur : jamais de faux positif inverse).

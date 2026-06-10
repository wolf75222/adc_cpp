# Reference : les briques natives (composition)

Cette page est le registre exhaustif des briques natives composables d'`adc` : chaque brique,
sa signature, ses parametres, ce qu'elle declare ou ajoute, et ses contraintes. C'est le
complement detaille de la [page modeles](../models/index.md) (qui presente les trois facons
d'ecrire un modele) et de l'[API Python](api_python.md) (autodoc curee) ; ici le detail au
niveau du parametre.

Un modele natif est une composition de quatre briques de role via
`adc.Model(state=, transport=, source=, elliptic=)` : la math cellule par cellule reste C++
compile (pas de boucle numpy sur le hot path, GPU/MPI conserves), Python ne fait qu'assembler
des objets. Le coeur reste agnostique au scenario : aucun nom physique (diocotron,
Euler-Poisson, deux-fluides) ne vit dans `adc` ; les compositions nommees vivent dans
[`adc_cases`](https://github.com/wolf75222/adc_cases). `adc.Model(...)` valide la coherence
etat <-> transport et reporte les parametres dans une `ModelSpec` (des tags lus cote C++ par la
fabrique de modeles) ; un appariement incoherent leve une `ValueError` immediate.

## Etat (state)

La brique d'etat fixe le nombre de variables conservatives, leurs noms / primitives, et impose
la brique de transport compatible. C'est l'argument `state=` de `adc.Model(...)`.

| Brique | Signature | Variables declarees | Transport requis |
|---|---|---|---|
| `Scalar` | `adc.Scalar()` | 1 variable conservative `n` (densite transportee) ; primitif = conservatif (`n`). | `ExB` |
| `FluidState` (compressible) | `adc.FluidState(kind="compressible", gamma=1.4)` | 4 variables `[rho, rho_u, rho_v, E]`, primitives `[rho, u, v, p]` ; porte `gamma` (reporte dans `spec.gamma`). | `CompressibleFlux` |
| `FluidState` (isotherme) | `adc.FluidState(kind="isothermal", cs2=0.5)` | 3 variables `[rho, rho_u, rho_v]`, primitives `[rho, u, v]` ; porte `cs2` (reporte dans `spec.cs2`). | `IsothermalFlux` |

`FluidState(kind=...)` n'accepte que `"compressible"` ou `"isothermal"` (toute autre valeur leve
une `ValueError`). Les arguments `gamma` / `cs2` sont stockes meme quand le `kind` ne les utilise
pas ; seul celui du `kind` choisi est reporte dans la spec.

## Transport

La brique de transport ecrit le flux physique hyperbolique. C'est l'argument `transport=` de
`adc.Model(...)`. Les parametres physiques (`gamma`, `cs2`) viennent de l'etat, pas du transport.

| Brique | Signature | Physique | Etat requis |
|---|---|---|---|
| `ExB` | `adc.ExB(B0=1.0)` | Advection scalaire par la derive E x B, `v = (-d_y phi, d_x phi) / B0`. Pose `spec.transport="exb"`, `spec.B0`. Struct C++ `adc::ExBVelocity`. | `Scalar` |
| `CompressibleFlux` | `adc.CompressibleFlux()` | Flux d'Euler compressible (`gamma` vient de `FluidState`). Pose `spec.transport="compressible"`. Struct C++ `adc::CompressibleFlux` (alias `adc::Euler`). | `FluidState(compressible)` |
| `IsothermalFlux` | `adc.IsothermalFlux()` | Flux d'Euler isotherme (`cs2` vient de `FluidState`). Pose `spec.transport="isothermal"`. Struct C++ `adc::IsothermalFlux`. | `FluidState(isothermal)` |

Il n'existe aucune autre brique de transport native. Pour un flux hyperbolique inedit, on passe
par le DSL (`adc.dsl.HyperbolicBrick`, cf. [page modeles](../models/index.md)).

## Source

La brique de source ajoute le terme source ponctuel `S(U, aux)` au RHS du bloc. C'est l'argument
`source=` de `adc.Model(...)`. Elle lit l'etat exterieur par le canal `adc::Aux` (potentiel
`phi`, gradients `grad_x` / `grad_y`).

| Brique | Signature | Ajoute au RHS | Variables min. |
|---|---|---|---|
| `NoSource` | `adc.NoSource()` | rien. Pose `spec.source="none"`. Struct C++ `adc::NoSource`. | 1 |
| `PotentialForce` | `adc.PotentialForce(charge=1.0)` | Force du potentiel `(q/m) rho E` sur la quantite de mouvement (+ terme de travail si 4 variables). Pose `spec.source="potential"`, `spec.qom=charge`. Struct C++ `adc::PotentialForce`. | 3 |
| `GravityForce` | `adc.GravityForce()` | Force gravitationnelle `rho g` (+ travail si 4 variables). Pose `spec.source="gravity"`. Struct C++ `adc::GravityForce`. | 3 |

Note : `PotentialForce(charge=...)` nomme le parametre `charge` cote Python mais le reporte dans
`spec.qom` (rapport charge/masse `q/m`) cote C++.

### Couplages inter-especes (add_coupling)

Les couplages inter-especes ne sont pas des sources `Model(source=)` : ils sont passes a
`System.add_coupling(...)`, appliques en operator-split apres le transport (pas integres dans le
RHS du bloc). Ils relient deux blocs (ou trois pour l'ionisation) par leur nom.

| Brique | Signature | Effet | Cible |
|---|---|---|---|
| `Ionization` | `adc.Ionization(electron, ion, neutral, rate)` | Ionisation `n_g -> n_i + n_e`, taux `k n_e n_g` ; masse transferee du neutre vers l'ion. Route vers `add_ionization`. | 3 blocs (electron, ion, neutre) |
| `Collision` | `adc.Collision(a, b, rate)` | Friction inter-especes : force `k (u_a - u_b)`, quantite de mouvement conservee. Route vers `add_collision`. | blocs fluides (>= 3 variables) |
| `ThermalExchange` | `adc.ThermalExchange(a, b, rate)` | Echange thermique `k (T_a - T_b)`, energie conservee. Route vers `add_thermal_exchange`. | blocs Euler (4 variables) |

`add_coupling` accepte aussi un `dsl.CompiledCoupledSource` (couplage generique decrit en
formules, transporte en bytecode et interprete cote C++) ; cf. [page modeles](../models/index.md).

## Second membre elliptique (elliptic)

La brique elliptique fixe la contribution du bloc au second membre du Poisson de systeme. C'est
l'argument `elliptic=` de `adc.Model(...)`. Le Poisson de systeme somme les contributions de
tous les blocs.

| Brique | Signature | Contribution au RHS elliptique |
|---|---|---|
| `ChargeDensity` | `adc.ChargeDensity(charge=1.0)` | Densite de charge `f = q n`. Pose `spec.elliptic="charge"`, `spec.q=charge`. Struct C++ `adc::ChargeDensity`. |
| `BackgroundDensity` | `adc.BackgroundDensity(alpha=1.0, n0=0.0)` | Fond neutralisant `f = alpha (n - n0)`. Pose `spec.elliptic="background"`, `spec.alpha`, `spec.n0`. Struct C++ `adc::BackgroundDensity`. |
| `GravityCoupling` | `adc.GravityCoupling(sign=1.0, four_pi_G=1.0, rho0=1.0)` | Couplage self-consistant `f = sign 4piG (rho - rho0)` (`sign=+1` gravite, `sign=-1` plasma). Pose `spec.elliptic="gravity"`, `spec.sign`, `spec.four_pi_G`, `spec.rho0`. Struct C++ `adc::GravityCoupling`. |

(briques-epm)=
### Briques EPM : l'operateur de Poisson est lui-meme composable

Le modele elliptique (EPM, EllipticPhysicalModel) n'est pas un cas hard-code : c'est une
composition de briques (inconnue + operateur + second membre + sortie). Le Poisson en est
l'instance courante. Ces briques se composent via `adc.elliptic(...)` puis se branchent via
`System.add_elliptic_model(...)`.

| Brique / fabrique | Signature | Role |
|---|---|---|
| `DivEpsGrad` | `adc.DivEpsGrad(epsilon=1.0)` | Operateur `D = div(eps grad .)`. `eps=1` -> Poisson ; `eps != 1` constant supporte (`eps lap phi = f`). `eps(x)` variable se branche via `set_epsilon_field`. |
| `div_eps_grad` | `adc.div_eps_grad(epsilon=1.0)` | Fabrique : renvoie un `DivEpsGrad`. |
| `CompositeRhs` | `adc.CompositeRhs()` | Second membre generique `f = somme_s elliptic_rhs_s(u_s)` : la somme des briques elliptiques portees par les blocs. Ne suppose aucune forme particuliere. |
| `composite_rhs` | `adc.composite_rhs()` | Fabrique : renvoie un `CompositeRhs`. |
| `ChargeDensitySource` | `adc.ChargeDensitySource()` (sous-classe de `CompositeRhs`) | Cas usuel : tous les blocs portent une densite de charge, donc `f = somme_s q_s n_s`. Alias historique de `CompositeRhs` (meme calcul, la somme des briques). |
| `charge_density` | `adc.charge_density()` | Fabrique : renvoie un `ChargeDensitySource`. |
| `ElectricFieldFromPotential` | `adc.ElectricFieldFromPotential()` | Sortie / post-traitement `E = -grad phi`, reinjecte dans l'`aux` des modeles hyperboliques. |
| `electric_field_from_potential` | `adc.electric_field_from_potential()` | Fabrique : renvoie un `ElectricFieldFromPotential`. |
| `EllipticModel` | `adc.EllipticModel(unknown, operator, rhs, output)` | Porte les 4 slots de l'EPM (inconnue + operateur + second membre + sortie). |
| `elliptic` | `adc.elliptic(unknown="phi", operator=None, rhs=None, output=None)` | Compose un EPM. Defauts : `operator=DivEpsGrad()`, `rhs=CompositeRhs()`, `output=ElectricFieldFromPotential()`. |
| `EllipticSolver` | `adc.EllipticSolver(kind="geometric_mg")` | Choix du solveur : `"geometric_mg"` (tout cas, parois) ou `"fft"` (periodique, `n = 2^k`). |

Le Poisson canonique s'ecrit donc :

```python
poisson = adc.elliptic(
    operator=adc.div_eps_grad(1.0),         # D = lap
    rhs=adc.charge_density(),               # f = somme_s q_s n_s
    output=adc.electric_field_from_potential(),  # E = -grad phi
)
```

`System.add_elliptic_model(name, model, solver=None, bc="auto", wall="none", wall_radius=0.0)`
cable cet EPM : il valide que `operator` est un `DivEpsGrad` (sinon `NotImplementedError` :
seul `div_eps_grad` est supporte, diffusion / projection demanderaient un autre solveur) et que
`rhs` est un `CompositeRhs` (sinon `NotImplementedError`), puis forwarde a `set_poisson(...)`.
Le token de second membre est `"charge_density"` quand `rhs` est exactement un
`ChargeDensitySource`, sinon `"composite"` (memes numeriques C++ : la somme des briques
elliptiques par bloc). `add_elliptic_model(...)` est donc la forme explicite de `set_poisson` :

```python
# ces deux appels sont equivalents (memes numeriques) :
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc="dirichlet",
                wall="circle", wall_radius=0.40)

sim.add_elliptic_model(
    "poisson",
    adc.elliptic(operator=adc.div_eps_grad(1.0), rhs=adc.charge_density(),
                 output=adc.electric_field_from_potential()),
    solver=adc.EllipticSolver("geometric_mg"),
    bc="dirichlet", wall="circle", wall_radius=0.40,
)
```

## Composer un modele

`adc.Model(state, transport, source, elliptic)` renvoie une `ModelSpec` (l'objet modele 100 %
natif, consomme par `add_block` / `add_equation`). La validation des quatre roles :

- `state` doit etre `Scalar` ou `FluidState(...)` (sinon `ValueError`) ;
- la coherence etat <-> transport est imposee : `Scalar` exige `ExB` ; `FluidState(compressible)`
  exige `CompressibleFlux` ; `FluidState(isothermal)` exige `IsothermalFlux` ;
- `source` doit etre `NoSource` / `PotentialForce` / `GravityForce` ;
- `elliptic` doit etre `ChargeDensity` / `BackgroundDensity` / `GravityCoupling`.

```python
model = adc.Model(
    state=adc.Scalar(),
    transport=adc.ExB(B0=1.0),
    source=adc.NoSource(),
    elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0),
)
```

### CompositeModel : modele hybride natif + DSL

`adc.CompositeModel(transport, source, elliptic, name="hybrid")` melange, dans un seul modele,
des briques natives (`adc.ExB`, `adc.PotentialForce`, `adc.ChargeDensity`...) et des briques DSL
partielles compilees (`adc.dsl.HyperbolicBrick(...).compile()`, `SourceBrick`, `EllipticBrick`).
Chaque slot accepte soit une brique native, soit une brique DSL compilee.

- Au moins un slot doit etre une brique DSL : une composition tout-native s'ecrit avec
  `adc.Model(...)` (sinon `CompositeModel` leve une `ValueError`).
- Une brique placee dans le mauvais slot leve une `ValueError` (le slot est verifie).
- `CompositeModel(...)` renvoie un `dsl.HybridModel` ; on appelle `.compile(backend="aot")` pour
  un `CompiledModel` branchable via `System.add_equation`. Prototype : seul `backend="aot"` est
  cable.

```python
m = adc.CompositeModel(
    transport=build_iso_transport(0.7).compile(),  # transport DSL
    source=adc.PotentialForce(charge=-1.0),        # source native
    elliptic=adc.ChargeDensity(charge=-1.0),       # elliptique native
)
compiled = m.compile(backend="aot")                # -> CompiledModel
sim.add_equation("gas", compiled,
                 spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov"),
                 names=["rho", "rho_u", "rho_v"])
```

Le slot transport fixe le layout (`n_vars`, noms conservatifs, primitives, gamma) ; une brique
DSL de source / elliptique doit declarer le meme `n_vars`. Detail : [page modeles](../models/index.md).

### PythonFlux : flux ecrit en Python (prototypage hote)

`adc.PythonFlux(flux, max_wave_speed)` est un backend de prototypage : l'utilisateur fournit le flux
physique `flux(U, dir)` et la vitesse d'onde `max_wave_speed(U)` en numpy, et `PythonFlux` assemble le
residu `-div(F*)` par flux de Rusanov (ordre 1, domaine periodique) sur tout le tableau. C'est un
chemin hote pur (jamais un kernel Kokkos), hors du hot path GPU / MPI ; il sert a iterer sur un flux
inedit sans recompiler (motif du cas `custom_scheme`, avec `adc.System` comme oracle de Poisson). Pour
la production, composer un flux compile (`adc.CompressibleFlux`, `adc.ExB`, ou un modele DSL).

```python
import adc
pf = adc.PythonFlux(flux=mon_flux, max_wave_speed=ma_vitesse)
dUdt = pf.residual(U, dx)              # -div(F*) par Rusanov ordre 1, periodique
dt = pf.cfl_dt(U, h, cfl=0.4)          # dt = cfl * h / max_wave_speed(U)
```

## Schemas spatiaux (Spatial / FiniteVolume)

Le schema spatial est porte par le bloc (argument `spatial=` de `add_block` / `add_equation`),
non par le modele. Il combine reconstruction (limiteur) + flux numerique de Riemann + variables
reconstruites.

`adc.Spatial(limiter="minmod", flux="rusanov", recon="conservative", *, none=False,
minmod=False, vanleer=False, weno5=False, primitive=False)` :

| Argument | Valeurs | Detail |
|---|---|---|
| `limiter` | `"none"`, `"minmod"`, `"vanleer"`, `"weno5"` | Reconstruction MUSCL (none / minmod / vanleer, 2 ghosts) ou WENO5-Z. `weno5` = ordre 5 en zone lisse, stencil 5 points -> 3 ghosts ; seul le chemin natif `add_block` (et les backends `aot` / `production` / AMR) l'exposent ; le backend `prototype` (JIT) le rejette. Raccourcis booleens `none=` / `minmod=` / `vanleer=` / `weno5=`. |
| `flux` | `"rusanov"`, `"hll"`, `"hllc"`, `"roe"` | Flux numerique de Riemann. `rusanov` = generique minimal (seul `max_wave_speed` requis). `hll` = generique a ondes signees : exige `model.wave_speeds` (modele natif isotherme / compressible, ou modele DSL avec primitive `p` declaree) ; c'est le chemin recommande pour un modele NON Euler a ondes signees (`hll` + `minmod`). `hllc` / `roe` = **Euler 2D seulement** (4 variables + pression gaz parfait) ; ils exigent un transport compressible et une primitive `p` declaree (sur un modele compile) ; sans `p`, le branchement leve une `ValueError`. |
| `recon` | `"conservative"`, `"primitive"` | Variables reconstruites. `primitive` est plus stable pour Euler (positivite de `rho` et `p`). Raccourci `primitive=`. |

`adc.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative")` est la fabrique
de surface stable : elle remappe sur `adc.Spatial`. Le flux numerique s'y nomme `riemann` (et non
`flux`, reserve au flux physique du modele DSL `m.flux`, pour ne pas collisionner les deux sens) :

| `FiniteVolume(...)` | -> | `Spatial(...)` |
|---|---|---|
| `limiter` | -> | `Spatial.limiter` |
| `riemann` | -> | `Spatial.flux` |
| `variables` | -> | `Spatial.recon` |

`FiniteVolume(...)` renvoie un `Spatial` (consomme tel quel) ; `adc.Spatial` reste disponible a
l'identique.

```{note}
Les seuls limiteurs existants sont none / minmod / vanleer / weno5 ; aucun autre limiteur n'est
expose.
```

## Traitement temporel (time)

Le traitement temporel est porte par le bloc (argument `time=`), pas par le modele : le meme
modele se reutilise avec des politiques distinctes.

| Brique | Signature | Detail |
|---|---|---|
| `Explicit` | `adc.Explicit(substeps=1, method="ssprk2", stride=1, *, ssprk3=False)` | Integration explicite. `method="ssprk2"` (Shu-Osher 2 etages ordre 2, defaut bit-identique) ou `"ssprk3"` (3 etages ordre 3, moins dissipatif, a apparier a weno5) ; raccourci `ssprk3=True`. Expose `.kind` = `"explicit"` ou `"ssprk3"`. |
| `IMEX` | `adc.IMEX(substeps=1, stride=1, implicit_vars=None, implicit_roles=None)` | Transport explicite (SSPRK) + source raide implicite (backward-Euler, Newton cellule-local). Pas un solveur implicite global PDE. `kind="imex"`. |
| `SourceImplicit` | `adc.SourceImplicit(substeps=1, stride=1, implicit_vars=None, implicit_roles=None)` | Nom clair du schema IMEX source-only ; `kind="imex"` (meme chemin C++ que `IMEX`, bit-identique). La doc contraste local (cette brique) vs global (`CondensedSchur`). |
| `Implicit` | `adc.Implicit(dt_ratio=1, substeps=None, stride=1)` | Obsolete : alias d'`IMEX`. Emet un `DeprecationWarning` (le nom suggere a tort un solveur implicite global) et renvoie un `IMEX(...)`. Utiliser `SourceImplicit` / `IMEX`. |
| `Split` | `adc.Split(hyperbolic=None, source=None)` | Politique de splitting explicite / implicite : etage transport `adc.Explicit` (defaut `Explicit()`) + etage source separe `adc.CondensedSchur` (requis). `scheme="lie"` (Godunov, 1er ordre). Relaie `kind` / `method` / `substeps` / `stride` de l'etage hyperbolique. Cable uniquement par `add_equation` (rejete par `add_block`). |
| `Strang` | `adc.Strang(hyperbolic=None, source=None)` | Sous-classe de `Split` : splitting de Strang (symetrique, 2e ordre) `H(dt/2); S(dt); H(dt/2)`. Pose `scheme="strang"` ; re-resout les champs entre les etages. |
| `CondensedSchur` | voir ci-dessous | Etage source condense par Schur (global). C'est le `source=` d'un `Split` / `Strang`. |
| `Role` | constantes | Roles physiques : `Density`, `MomentumX`, `MomentumY`, `MomentumZ`, `Energy`, `VelocityX/Y/Z`, `Pressure`, `Temperature`, `Scalar` (cles stables snake_case). Sert dans `CondensedSchur` et les masques IMEX. |

### CondensedSchur (etage source global)

`adc.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, alpha=1.0, density=Role.Density,
momentum=(Role.MomentumX, Role.MomentumY), energy=None, magnetic_field="B_z", potential="phi")`
est l'etage source condense de Schur (Hoffart et al., arXiv:2510.11808). Il assemble l'operateur
elliptique condense `A = I + theta^2 dt^2 alpha rho B^{-1}`, le resout (BiCGStab preconditionne
MG) et reconstruit la vitesse. Tout est C++ (aucun callback Python par cellule).

| Parametre | Contrainte |
|---|---|
| `kind` | seul `"electrostatic_lorentz"` (toute autre valeur leve une `ValueError`). |
| `theta` | theta-schema dans `(0, 1]` (`0.5` Crank-Nicolson, `1` Euler retrograde). |
| `alpha` | constante de couplage electrostatique du sous-systeme source. |
| `density` / `momentum` / `energy` / `magnetic_field` / `potential` | Descripteurs de roles / champs. Tous rejetes s'ils s'ecartent du defaut : l'etage source C++ fige en dur les roles `Density` / `MomentumX` / `MomentumY` (`Energy` optionnel) et les champs `B_z` / `phi`. La signature est gardee pour quand le C++ les transportera, mais un descripteur different leve une `ValueError` (rejet plutot qu'ignore silencieux). |

`CondensedSchur` exige du bloc les roles `Density` / `MomentumX` / `MomentumY` et un champ `B_z`
(`set_magnetic_field`) ; un role / `B_z` manquant leve une erreur explicite a `add_equation`. Il
est cable en cartesien et en polaire ; le pendant polaire est mono-rang (`n_ranks > 1` leve).

### Multirate : substeps et stride

`substeps` et `stride` sont orthogonaux (valables sur `Explicit` / `IMEX` / `SourceImplicit`) :

- `substeps=N` : le bloc avance N fois par macro-pas, chaque sous-pas de longueur `dt/N`
  (electrons rapides : `substeps=10`). Defaut 1 = bit-identique a l'historique.
- `stride=M` : cadence hold-then-catch-up (rattrapage en fin de fenetre). Le bloc est tenu tant
  que `(macro_step + 1) % M != 0`, puis avance d'un pas effectif `M dt` quand
  `(macro_step + 1) % M == 0` (bloc lent, p.ex. neutres : `stride=20`). Entre deux rattrapages, son
  etat perime (derniere densite / charge avancee, figee) contribue quand meme au Poisson de
  systeme et aux sources couplees. `step_cfl` honore la cadence : `dt <= cfl h substeps / (stride w)`.

### Masque IMEX (implicit_vars / implicit_roles)

`implicit_vars` (noms de variables conservees) et `implicit_roles` (roles physiques, normalises
en cles stables via `Role`) listent les composantes traitees en implicite dans le pas de source ;
le reste reste explicite. Le masque est porte par la politique temporelle / le bloc, pas par le
modele -> le meme modele se reutilise avec des traitements implicites distincts. Defaut `[]`
(union vide) = defaut du modele, bit-identique. La resolution noms / roles -> indices et la
validation (nom / role absent du bloc) sont C++-side (source unique de verite). Une chaine seule
est toleree (`implicit_vars="rho_u"` -> `["rho_u"]`).

### Garde-fous par backend

| Chemin | Stride > 1 | evolve=False | weno5 | flux non-rusanov | masque IMEX |
|---|---|---|---|---|---|
| `add_block` natif (`ModelSpec`) | supporte | supporte | supporte | supporte | supporte |
| `add_equation` backend `production` | supporte | supporte | supporte | supporte | rejete (`.so` natif) |
| `add_equation` backend `aot` | rejete | rejete | supporte | supporte | rejete |
| `add_equation` backend `prototype` | (substeps seul) | rejete | rejete | rejete (rusanov seul) | rejete |

Les rejets sont explicites (`ValueError`), jamais un ignore silencieux : l'ABI `.so` de ces
backends ne transporte pas l'argument concerne (cadence, `evolve`, masque), donc le bloc
tournerait a la valeur par defaut sans le dire. `Split` / `Strang` sont rejetes par `add_block`
(seul `add_equation` branche l'etage source `set_source_stage`).

## Maillage (mesh)

Le choix de la geometrie vit dans un objet maillage passe en `mesh=` a `adc.System(...)`, pas
dans le schema (`adc.FiniteVolume` reste reconstruction + Riemann + variables, sans argument de
geometrie). Le maillage est applique apres `**cfg_kw`, donc `mesh=` prevaut sur les `n=` / `L=`
passes en mots-cles.

| Brique | Signature | Effet |
|---|---|---|
| `CartesianMesh` | `adc.CartesianMesh(n=64, L=1.0, periodic=True)` | Domaine carre `[0, L]^2`, `n x n` cellules (defaut implicite). `adc.System(mesh=adc.CartesianMesh(n, L, periodic))` est strictement equivalent (bit-identique) a `adc.System(n=n, L=L, periodic=periodic)`. Pose `config.geometry="cartesian"`, `n`, `L`, `periodic`. |
| `PolarMesh` | `adc.PolarMesh(r_min, r_max, nr, ntheta)` | Anneau global `r in [r_min, r_max] x theta in [0, 2pi)`, `nr x ntheta` cellules. theta periodique, r porte une condition aux limites physique (direction 0 = radiale, 1 = azimutale). Pose `config.geometry="polar"`, `nr`, `ntheta`, `r_min`, `r_max`, et `config.n = nr` (taille par defaut des diagnostics). |

Validation de `PolarMesh` : `r_max > r_min >= 0` (sinon `ValueError`), `nr >= 3` (le stencil
radial decentre d'ordre 2 aux parois lirait `phi` hors bornes sinon), `ntheta >= 1`.

Limites du chemin polaire (Phase 2b, branche dans `System.step` : transport polaire + Poisson
polaire + aux en base locale `e_r` / `e_theta`) : transport ExB scalaire seulement (limiter /
riemann fluides leves cote C++), mono-rang (le solveur polaire direct refuse MPI), pas de couplage
cartesien <-> polaire (anneau global).

### Champs de configuration

`SystemConfig` (champs readwrite) : `n`, `L`, `periodic`, `geometry`, `nr`, `ntheta`, `r_min`,
`r_max`.

`AmrSystemConfig` (champs readwrite) : `n`, `L`, `regrid_every`, `periodic`, `distribute_coarse`,
`coarse_max_grid`. `regrid_every == 0` -> hierarchie figee (regrid jamais appele, bit-identique).

## System / AmrSystem : methodes

Le wrapper Python definit quelques methodes (composition, primitives, EPM, disque) ; tout le
reste est delegue a la facade C++ compilee via `__getattr__`. Reference compacte.

### System

| Methode | Signature | Role |
|---|---|---|
| `add_block` | `add_block(name, model, spatial=None, time=None, evolve=True)` | Ajoute un bloc a partir d'une `ModelSpec`. Defauts `Spatial()` / `Explicit()`. Rejette `Split` / `Strang` (utiliser `add_equation`). |
| `add_equation` | `add_equation(name, model, spatial=None, time=None, substeps=None, names=None, evolve=True, stride=None)` | Aiguille sur le type : `ModelSpec` -> `add_block` ; `CompiledModel` -> l'adder du backend (`add_dynamic_block` prototype / `add_compiled_block` aot / `add_native_block` production) ; gere `Split` / `Strang` (etage hyperbolique puis `set_source_stage` + `set_time_scheme`). Applique les garde-fous backend. |
| `run` | `run(t_end, cfl=0.4, max_steps=1_000_000)` | Sucre `while time() < t_end: step_cfl(cfl)` ; renvoie le nombre de pas. |
| `add_background` | `add_background(name, model, density, spatial=None)` | Espece gelee = `add_block(evolve=False)` + `set_density`. |
| `add_elliptic_model` | `add_elliptic_model(name, model, solver=None, bc="auto", wall="none", wall_radius=0.0)` | Cable un EPM (valide `DivEpsGrad` + `CompositeRhs`) ; forwarde a `set_poisson`. |
| `set_disc_domain` | `set_disc_domain(cx, cy, R, mode="none")` | Domaine de transport en disque ; `mode` : `"none"` (masque pose, transport plein cartesien, bit-identique) / `"staircase"` (transport masque conservatif) / `"cutcell"` (cut-cell / embedded-boundary). Cartesien seulement. |
| `set_geometry_mode` | `set_geometry_mode(mode)` | Bascule le mode de transport disque sans redefinir le disque. |
| `disc_mask` | `disc_mask()` | Masque 0/1 cellule-centre `(ny, nx)` (diagnostic). |
| `add_coupling` | `add_coupling(coupling)` | Route `Ionization` / `Collision` / `ThermalExchange` ou un `CompiledCoupledSource`. |
| `block_names` | `block_names()` | Noms des blocs dans l'ordre (inclut les blocs dynamiques / compiles). |
| `set_primitive_state` | `set_primitive_state(name, **prims)` | Initialise un bloc depuis ses primitives nommees (`rho` / `u` / `v` / `p`), assemblees `(ncomp, n, n)` dans l'ordre du modele, converties en conservatif cote C++. |
| `get_primitive_state` | `get_primitive_state(name)` | Inverse : renvoie un dict `{nom_primitive: (n, n)}`. |
| `abi_key` | `System.abi_key()` (statique) | Cle d'ABI du module. |

Methodes de la facade C++ atteintes par `__getattr__` (avec defauts) :

- `set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto", wall="none", wall_radius=0.0, epsilon=1.0)` : `bc` p.ex. `"dirichlet"` ; `wall` p.ex. `"circle"` + `wall_radius`.
- `set_density(name, rho)` : `rho` tableau `n x n`.
- `set_epsilon_field(eps)`, `set_epsilon_anisotropic_field(eps_x, eps_y)`, `set_reaction_field(kappa)`, `set_magnetic_field(bz)`, `set_electron_temperature_from(name)`.
- `set_source_stage(name, kind, theta, alpha)`, `set_time_scheme(scheme)` (`"lie"` / `"strang"`).
- `add_coupled_source(in_blocks, in_roles, consts, out_blocks, out_roles, prog_ops, prog_args, prog_lens)`.
- `add_dynamic_block` / `add_compiled_block` / `add_native_block` / `set_block_params` : adders de backend (utilises en interne par `add_equation`).
- `add_ionization(electron, ion, neutral, rate)` / `add_collision(a, b, rate)` / `add_thermal_exchange(a, b, rate)`.
- `variable_names(name, kind="conservative")` / `variable_roles(name, kind="conservative")` / `block_gamma(name)` / `n_vars(name)`.
- `solve_fields()` ; `step(dt)` ; `advance(dt, nsteps)` ; `step_cfl(cfl)` ; `step_adaptive(cfl)`.
- `eval_rhs(name)` / `get_state(name)` / `set_state(name, u)` : primitives d'integrateur Python custom.
- `nx()` ; `ny()` ; `time()` ; `n_species()` ; `mass(name)` ; `density(name)` -> `(ny, nx)` ; `potential()` -> `(ny, nx)`.

### AmrSystem

`adc.AmrSystem` est le pendant raffine : un ou plusieurs blocs portes sur une hierarchie AMR
partagee, avec un Poisson de systeme a second membre somme `somme_b q_b n_b` et conservation par
bloc. En multi-blocs le nom du bloc indexe `set_density(name)` / `mass(name)` / `density(name)`.

| Methode | Signature | Role |
|---|---|---|
| `add_block` | `add_block(name, model, spatial=None, time=None)` | Rejette `Split` ; threade substeps / stride + masque IMEX vers le C++. |
| `add_equation` | `add_equation(name, model, spatial=None, time=None, substeps=None)` | `ModelSpec` -> `add_block` (forwarde stride + masque) ; un `CompiledModel` doit etre `backend="production"`, `target="amr_system"` -> `add_native_block`. Rejette stride > 1 et masque IMEX sur le `.so` production (ABI plate) ; exige `p` pour hllc / roe. Recon primitive + flux roe / hllc + weno5 sont cables sur AMR (parite avec `add_block`). |
| `set_refinement` | `set_refinement(threshold)` | Tag la ou la densite du bloc (composante 0) depasse `threshold`. |
| `set_phi_refinement` | `set_phi_refinement(grad_threshold)` | Ajoute un tag base sur `|grad phi|` a l'union de regrid (multi-blocs + `regrid_every > 0` ; `<= 0` desactive, defaut). |
| `set_poisson` | `set_poisson(rhs="charge_density", solver="geometric_mg", bc="auto", wall="none", wall_radius=0.0)` | Poisson de systeme (pas d'argument `epsilon` sur ce chemin). |
| `set_density` | `set_density(name, rho)` | Densite initiale d'un bloc. |
| `add_coupled_source` | `add_coupled_source(...)` | Source couplee generique (bytecode). |
| `step` / `advance` / `step_cfl` | `step(dt)` / `advance(dt, nsteps)` / `step_cfl(cfl)` | Avance. |
| `nx` / `time` / `n_blocks` / `n_patches` | (sans argument) | Diagnostics scalaires. `n_blocks()` = nombre de blocs ; `n_patches()` = nombre de patchs fins. |
| `mass` / `density` | `mass()` / `mass(name)` ; `density()` / `density(name)` -> `(nx, nx)` | Nom vide -> 1er bloc ; en multi-blocs le nom indexe le bloc. |
| `potential` | `potential()` -> `(n, n)` | phi du niveau grossier (Poisson de systeme partage). |
| `patch_boxes` | `patch_boxes()` (recent) | Empreintes index-space des patchs fins : liste de `(level, ilo, jlo, ihi, jhi)`, coins inclusifs, dans l'espace d'indices du niveau (`n << level` cellules/direction, ratio 2). Rank-independent (MPI-safe). |
| `patch_rectangles` | `patch_rectangles()` (recent) | Convertit `patch_boxes()` en rectangles physiques `(x0, y0, w, h)` dans `[0, L]^2` (un par patch fin). Pratique pour tracer les patchs (p.ex. `matplotlib.Rectangle`). |

```{note}
`patch_boxes()` / `patch_rectangles()` exposent la geometrie des patchs fins (ajout
recent). Si le module `adc` bati sur votre branche est anterieur a cet ajout, ces methodes
peuvent ne pas exister encore ; elles sont listees ici pour la reference complete de l'API.
```

## Exemple complet : un diocotron depuis les briques

Un diocotron reduit = une densite electronique scalaire advectee par E x B, avec un fond
neutralisant, et un Poisson a mur conducteur circulaire. Construit entierement a partir de
briques natives (aucun helper `models.diocotron`) :

```python
import numpy as np
import adc

# --- maillage + systeme (carre cartesien, non periodique pour le mur conducteur) ---
n = 192
sim = adc.System(mesh=adc.CartesianMesh(n=n, L=1.0, periodic=False))

# --- le modele, compose des quatre briques de role ---
#   state    : Scalar          (une variable : la densite electronique n)
#   transport: ExB(B0=1.0)     (derive E x B)
#   source   : NoSource        (pas de source ponctuelle pour un scalaire)
#   elliptic : BackgroundDensity(alpha=1.0, n0=0.0)   (f = alpha (n - n0))
model = adc.Model(
    state=adc.Scalar(),
    transport=adc.ExB(B0=1.0),
    source=adc.NoSource(),
    elliptic=adc.BackgroundDensity(alpha=1.0, n0=0.0),
)

# --- branche le bloc : schema spatial + integrateur temporel ---
sim.add_block(
    "ne",
    model=model,
    spatial=adc.FiniteVolume(limiter="minmod", riemann="rusanov", variables="conservative"),
    time=adc.Explicit(),                  # SSPRK2, substeps=1, stride=1
)

# --- Poisson de systeme avec mur conducteur circulaire (Dirichlet) ---
sim.set_poisson(bc="dirichlet", wall="circle", wall_radius=0.40)

# --- densite annulaire initiale (anneau d'electrons creux) ---
xs = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xs, xs, indexing="ij")
r = np.hypot(X - 0.5, Y - 0.5)
ne0 = np.where((r > 0.20) & (r < 0.30), 1.0, 0.0)
# germe azimutal pour declencher l'instabilite
theta = np.arctan2(Y - 0.5, X - 0.5)
ne0 = ne0 * (1.0 + 0.01 * np.cos(5.0 * theta))
sim.set_density("ne", ne0)

# --- avance quelques pas limites par le CFL ---
for _ in range(50):
    sim.step_cfl(0.4)

print("t        =", sim.time())
print("mass(ne) =", sim.mass("ne"))      # invariant conserve
phi = sim.potential()                    # (n, n)
ne = sim.density("ne")                    # (n, n)
print("phi range:", float(phi.min()), float(phi.max()))
```

La meme `ModelSpec` se branche sur `adc.AmrSystem` (raffinement adaptatif) sans changer le
modele : `sa.add_block("ne", model=model, ...)`. Pour la forme explicite de `set_poisson` (via un
EPM compose), voir la section [Briques EPM](#briques-epm).

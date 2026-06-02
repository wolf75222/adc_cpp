# Plan : niveau Variables + HPM / EPM (chantier abstraction)

Reference de travail pour le chantier issu des reunions tuteur (01-02/06/2026). A relire
avant la seance tableau avec Sacha. Le coeur `adc_cpp` est deja agnostique (briques) et son
flux numerique est deja generique sur le modele ; l'abstraction manquante est le niveau
**Variables** (conservatif U / primitif P + conversions + choix de reconstruction), et la
seconde correction est de **ne pas hard-coder Poisson** mais d'en faire une instance d'un
**EllipticPhysicalModel (EPM)**.

---

## 0. Etat actuel

### Fait (Phase 1, teste : ctest 46/46 + recon end-to-end Python)
- Concept `PhysicalModel` etendu par l'extension OPTIONNELLE `HasPrimitiveVars`
  (`core/physical_model.hpp`) : `using Prim`, `to_primitive(U)->P`, `to_conservative(P)->U`.
- Conversions implementees dans les briques transport (`model/euler.hpp`, `model/bricks.hpp`) :
  CompressibleFlux P=(rho,u,v,p), IsothermalFlux P=(rho,u,v), ExBVelocity P=cons (identite).
  `CompositeModel` forwarde Prim + conversions.
- `max_wave_speed` / `wave_speeds` calcules VIA le primitif (centralisation : plus de recalcul
  u=rho u/rho, p=(g-1)(E-...) eparpille).
- Reconstruction en variables primitives (`operator/spatial_operator.hpp`) : `reconstruct`
  convertit le stencil U->P, limite sur P, reconvertit P->U ; flux numerique inchange ; update
  toujours conservatif. Choix porte par un flag RUNTIME `recon_prim` (pas d'explosion de
  templates), repli conservatif si le modele n'expose pas les conversions.
- Expose : `add_block(..., recon="conservative"|"primitive", ...)` ; Python
  `adc.Spatial(recon="primitive")` ou `Spatial(primitive=True)`. AMR rejette le primitif
  proprement (les cas AMR utilisent NoSlope, ou prim == cons).
- Tests : `tests/test_primitive_recon.cpp` (round-trip + concept) + test Python (Euler recon
  cons vs prim : masse conservee ~1e-15 dans les deux, positivite, fini).

### Fait (priorites 5-7 + cas, ce chantier)
- #57 cas "deux Euler independants" (adc_cases/two_euler) : meme schema HLLC + recon primitif,
  multirate ; masse/bloc, positivite, electrons plus rapides.
- #54 espece GELEE : flag `evolve` sur le bloc (non avance, vu par Poisson) + `add_background`.
- #52 mecanisme CoupledSource : passe operator-split dans le runtime System (apres transport, lit
  plusieurs blocs au meme point).
- #53 briques de couplage : `add_ionization` (n_g -> n_i + n_e, masse n_i+n_g conservee),
  `add_collision` (friction, qte de mvt conservee), `add_thermal_exchange` (energie conservee).
- #55 (partiel) cas plasma (adc_cases/plasma) : e + i + n, Poisson + ionisation + collision cables
  de bout en bout ; n_i+n_g conserve a ~1e-15, Poisson actif, densites positives. models.py :
  recettes `euler()` et `neutral_isothermal()` ajoutees.
- #58 EPM premier ordre : add_elliptic_model + briques (elliptic / div_eps_grad / charge_density /
  electric_field_from_potential) ; Poisson = instance ; set_poisson raccourci ; eps!=1 et operateurs
  alternatifs rejetes (raffinements). Pur Python au-dessus du solveur existant.
- #55/#56/#59 : recettes systeme models.two_fluid / models.plasma ; objets de couplage
  (adc.Ionization / Collision / ThermalExchange) + sim.add_coupling ; descripteur de variables
  (sim.variable_names cons/prim par bloc, introspection).
- Verifie : test_bindings (briques + frozen + 3 couplages + EPM + introspection + garde-fous),
  cas two_euler et plasma, ctest coeur 46/46 inchange.

### Deja generique : NE PAS refaire
- `RusanovFlux` / `HLLFlux` / `HLLCFlux` sont `template<class Model>` et appellent `m.flux`,
  `m.max_wave_speed`, `m.wave_speeds`, `m.pressure` (`operator/numerical_flux.hpp`). Le schema
  ne connait PAS Euler. (C'est `euler_cpp` qui avait ce probleme, pas `adc_cpp`.)
- `System` / `AmrSystem` / dispatch (transport x source x elliptic), separation deux depots
  (adc_cpp coeur, adc_cases cas), multirate, IMEX partiel, Poisson multi-especes Sum_s q_s n_s.

---

## 1. Architecture cible (tableau du tuteur)

```
PhysicalModel
├── HPM : HyperbolicPhysicalModel        d_t U + div F(U, aux) = S(U_self, U_all, aux)
│     Vars   : U (conservatif), P (primitif) ; cons_to_prim(U), prim_to_cons(P)
│     Flux   : F(U) [F(P) optionnel] ; lambda ; |lambda|max = |u| + c, c = sqrt(gamma p/rho)
│     Source : locale + COUPLEE (peut lire les autres especes)
│
└── EPM : EllipticPhysicalModel          D(eps, phi) = f(U_all, aux)
      unknown  : phi
      operator : D = div(eps grad .)        (Poisson : un cas particulier)
      coeff    : eps(x)
      rhs      : f(U_all) = Sum_s q_s n_s    (densite de charge)
      output   : phi, E = -grad phi          (reinjecte dans les sources des HPM)

SpaceMethod (deja generique)
      Reconstruction (conservative | primitive) + NumericalFlux (Rusanov/HLL/HLLC) + div(F*)

Coupler / Assembler
      tient plusieurs HPM + les EPM ; construit les sources inter-especes ; construit le rhs
      des EPM ; reinjecte phi / E dans les sources des HPM. AMR = assembleur sur grille raffinee.
```

### Cas plasma de reference (3 HPM couples par les sources)
- H1 electrons (Euler + Lorentz, 4 var) : source = ionisation `m_e n_e n_g K_iz`, force
  `-e n_e E`, magnetique `-e n_e (u_e x B)`, collisions `-m_e n_e nu_e u_e`, echange thermique
  `(3 m_e/M) n_e nu_e (T_e - T_g)`.
- H2 ions (isotherme, 3 var) : ionisation `M n_e n_g K_iz`, force `e n_i E`, collisions
  `-M n_i nu_i u_i` (ions non magnetises : pas de u_i x B).
- H3 neutres (isotherme, 3 var) : `-M_n n_e n_g K_iz`, collisions `-M_n n_i nu_i (u_g - u_i)`.
- Le MEME terme d'ionisation apparait avec des SIGNES OPPOSES : on perd un neutre, on gagne un
  ion et un electron (conservation). Couplage NON mediatise par Poisson.
- Poisson : `div(eps grad phi) = e(n_e - n_i)/eps0`, puis `E = -grad phi` revient dans les sources.

---

## 2. Niveau Variables (FAIT) : pipeline de reconstruction

```
Conservatif :  U_i  -> limiteur sur U -> U_L, U_R -> flux numerique -> update conservatif
Primitif    :  U_i  -> P_i = cons_to_prim(U_i) -> limiteur sur P -> P_L, P_R
                    -> U_L = prim_to_cons(P_L), U_R = prim_to_cons(P_R) -> flux numerique
            update TOUJOURS conservatif : U^{n+1} = U^n - dt div(F*) + dt S
```
Pour Euler le primitif est plus robuste (positivite de rho et p).

### Vars comme BRIQUE de premiere classe (precision du tuteur)

Au tableau, `Vars` est une brique du PhysicalModel, AU MEME NIVEAU que `Flux` et `Source` :
```
PhysicalModel
   |-- Vars     (description des variables)
   |-- Flux     (operateur de transport)
   |-- Source   (couplage / reactions / forces)
```
`Vars` n'est donc PAS seulement `using State` / `using Prim` + conversions (ce que la Phase 1
a fait, et qui est le sous-ensemble FONCTIONNEL qui marche). Conceptuellement c'est un objet qui
DECRIT les variables :
```cpp
enum class VariableKind { Conservative, Primitive };
struct Variables {
  VariableKind kind;
  std::vector<std::string> names;  // ex. {"rho","rho_u","rho_v","rho_E"} ou {"rho","u","v","p"}
  int size;
};
```
et le modele porte plusieurs jeux :
```cpp
Variables conservative_vars;   // (rho, rho u, rho v, rho E)
Variables primitive_vars;      // (rho, u, v, p)
State to_primitive(const State& U) const;
State to_conservative(const State& P) const;
```
Exemples (noms) :
```
Euler      cons (rho, rho u, rho v, rho E)   prim (rho, u, v, p)
isotherme  cons (rho, rho u, rho v)          prim (rho, u, v)
diocotron  cons (n)                          prim (n)
```
Le coeur numerique manipule ces objets Variables GENERIQUEMENT, sans savoir si les composantes
sont rho/u/p, n, rho*u... La reconstruction s'ecrit conceptuellement :
```cpp
if (recon == Conservative) reconstruct(model.conservative_vars, U);
if (recon == Primitive)  { P = model.to_primitive(U); reconstruct(model.primitive_vars, P);
                           U = model.to_conservative(P); }
```
Etat : la Phase 1 fournit le coeur FONCTIONNEL (Prim + to_primitive/to_conservative ; la
reconstruction limite composante par composante, deja aveugle au sens des composantes). Les NOMS
des variables (cons/prim par bloc) sont exposes pour l'introspection via `sim.variable_names`
(#59, FAIT, metadonnee hote). Un objet descripteur `Variables` (kind + names + size) porte
directement par la brique (vraie "Vars" au meme niveau que Flux/Source) resterait une forme plus
poussee, non necessaire au calcul.

---

## 3. EPM : la correction (NE PAS hard-coder Poisson)

Poisson ne doit pas etre un cas special au niveau architecture. Il faut un
`EllipticPhysicalModel` ; Poisson n'en est qu'une instance (inconnue phi, operateur
div(eps grad), rhs charge_density, sortie E = -grad phi).

### Cible API (au lieu de `set_poisson(...)` comme abstraction finale)
```python
sim.add_elliptic_model(
    name="phi",
    model=models.elliptic(
        unknown="phi",
        operator=models.div_eps_grad(epsilon=1.0),
        rhs=models.charge_density(species={"ne": -1.0}, background=n_i0),
        output=models.electric_field_from_potential(),   # E = -grad phi
    ),
    solver=adc.EllipticSolver("geometric_mg"),
)
```
`div_eps_grad`, `charge_density`, `electric_field_from_potential` sont des BRIQUES, pas
`Poisson` code en dur. On peut remplacer UNIQUEMENT l'EPM sans toucher System/AmrSystem :
```python
models.elliptic(unknown="T", operator=models.diffusion(coeff="kappa"), rhs=models.heat_source())
models.elliptic(unknown="p", operator=models.pressure_projection(), rhs=models.divergence_constraint())
```
`set_poisson(...)` RESTE comme raccourci pratique (= add_elliptic_model avec les briques de
Poisson), mais pas comme architecture centrale.

Etat (premier ordre, FAIT, #58) : add_elliptic_model + briques exposes en Python ; Poisson = instance ;
set_poisson raccourci. Pour l'instant operateur div(eps grad) a eps=1 + densite de charge (charges sur
les blocs) ; eps(x), charges au niveau EPM, et autres operateurs (diffusion, projection) = raffinements
(rejetes proprement par NotImplementedError).

### Forme math
```
HPM : d_t U + div F(U) = S(U, aux)
EPM : D(phi, aux) = f(U_all, aux)
Poisson : D = div(eps grad) ; f = e(n_e - n_i)/eps0 ; plus generalement f = Sum_s q_s n_s ; E = -grad phi
```

---

## 4. Sources inter-especes (CoupledSource)

Aujourd'hui `source(U, aux)` ne voit qu'un etat + le champ ; le couplage direct (ionisation,
collisions, echange thermique) doit lire les AUTRES especes. Signature cible :
```
source(block_id, local_state U_self, SystemView all, aux) -> S
```
Place : au niveau Coupler/Assembler (comme le rhs elliptique somme deja Sum_s q_s n_s). Briques :
ionisation `m n_e n_g K_iz` (signes opposes), collisions `nu (u_a - u_b)`, echange `(T_a - T_b)`.
Depend des variables primitives (vitesses, temperatures). Test : somme des sources de masse = 0.

---

## 5. Espece gelee (background evolue / non-evolue)

Un bloc peut etre GELE : `evolve=False`, non avance en temps, mais visible des sources couplees
(l'ionisation a besoin de n_g) et du rhs des EPM. Generalise `BackgroundDensity` (n0 constant)
vers une vraie espece de fond n_g(x). Passer evolue <-> gele doit etre "une ligne".

---

## 6. Echelle de tests du tuteur (meme SpaceMethod partout, sans copier-coller)

1. advection scalaire
2. Euler isotherme
3. Euler complet (1 bloc)
4. DEUX Euler INDEPENDANTS (electrons + ions, rapport de masses, vitesses differentes, detente, non couples)
5. + source couplee (ionisation / collisions)
6. + EPM / Poisson (≈ multispecies actuel)
7. modele plasma de Sacha

---

## 7. TODO (ordre de priorite du tuteur)

```
[x] 1. Cons/Prim + conversions dans les modeles            (#47, #48)
[x] 2. max_wave_speed via le primitif                       (#48)
[x] 3. reconstruction en primitif + flag recon expose       (#49, #50)
[x] 4. test Euler recon=cons vs recon=prim                  (#51 + test Python)
[x] 5. cas "deux Euler independants" (adc_cases)            (#57, etape 4 de l'echelle)
[x] 6. CoupledSource (operator-split) + ionisation/collision/echange thermique + espece gelee
       (#52, #53, #54) : sim.add_ionization / add_collision / add_thermal_exchange / add_background
[x] 7. EPM premier ordre (#58) : add_elliptic_model + briques elliptic / div_eps_grad /
       charge_density / electric_field_from_potential ; Poisson = instance ; set_poisson raccourci.
       eps(x) et operateurs alternatifs (diffusion, projection) = raffinements (rejetes proprement).
[x] -- niveau systeme dans adc_cases : models.py recipes (two_fluid, plasma) + couplages (#55)
[x] -- Python : Spatial(recon=), objets de couplage (add_coupling), add_background, add_elliptic_model (#56)
[x] -- descripteur Variables : sim.variable_names (cons/prim par bloc), introspection (#59)
```

Note : ordre des priorites = celui du tuteur. #59 (objet Variables descripteur) est un
ENRICHISSEMENT conceptuel, non bloquant ; le calcul tourne deja avec le sous-ensemble fonctionnel
de la Phase 1.

Chemin critique vers "deux Euler, meme code" demandé par le tuteur : 1-4 (fait) + 5.

---

## 8. Esquisses de code (du tableau)

### HPM
```cpp
struct HPM {
  using ConsVars = ...;   // U
  using PrimVars = ...;   // P
  PrimVars cons_to_prim(const ConsVars& U) const;   // = to_primitive
  ConsVars prim_to_cons(const PrimVars& P) const;   // = to_conservative
  ConsVars flux(const ConsVars& U, const Aux& aux, int dir) const;
  double   max_wave_speed(const ConsVars& U, const Aux& aux, int dir) const;  // |u| + c
  ConsVars source(const ConsVars& U_self, const SystemView& all, const Aux& aux) const;
};
```

### EPM
```cpp
struct EPM {
  using Unknown = ...;   // phi
  auto   coefficient(Position x) const;            // eps(x), operateur D = div(eps grad)
  double rhs(const SystemView& all, Position x) const;   // f(U_all) = Sum_s q_s n_s
  AuxFields postprocess(const Unknown& phi) const;       // E = -grad phi
};
```

### Flux numerique generique (deja en place dans adc_cpp)
```cpp
template <class Model>
struct RusanovFlux {
  typename Model::State operator()(const Model& m, const State& UL, const State& UR,
                                   const Aux& aux, int dir) const {
    auto FL = m.flux(UL, aux, dir), FR = m.flux(UR, aux, dir);
    double a = std::max(m.max_wave_speed(UL, aux, dir), m.max_wave_speed(UR, aux, dir));
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL);
  }
};
```

### Volumes finis + flux numerique a l'interface (convention du tableau)
```
dU_ij/dt + (1/dx)(F*_{i+1/2,j} - F*_{i-1/2,j}) + (1/dy)(F*_{i,j+1/2} - F*_{i,j-1/2}) = S_ij
F*_{i+1/2,j} = NF(U^R_{i,j}, U^L_{i+1,j})        # etats RECONSTRUITS (pas U_i et U_{i+1})
Rusanov directionnel : a = max(|lambda^x_max(U^L)|, |lambda^x_max(U^R)|)   (idem en y)
|lambda|_max = |sqrt(dp/drho) +- u| = |u| + c    # generique ; Euler : c = sqrt(gamma p/rho)
valeurs propres (Euler, dir x) : u - c, u, u + c
```
NF est GENERIQUE : il appelle `model.flux` et `model.max_wave_speed`, jamais Euler en dur (deja
le cas dans adc_cpp). `|lambda|_max = |sqrt(dp/drho) +- u|` est la forme valable pour toute EOS,
pas seulement le gaz parfait.

### Cas diocotron, cible avec EPM explicite
```python
sim = adc.AmrSystem(n=n, L=L, regrid_every=10, periodic=True)
sim.add_block("ne", model=models.diocotron(B0=1.0, alpha=1.0),
              spatial=adc.Spatial(none=True, flux="rusanov"))
sim.add_elliptic_model("phi", model=models.elliptic(
    unknown="phi", operator=models.div_eps_grad(epsilon=1.0),
    rhs=models.charge_density(species={"ne": -1.0}, background=n_i0),
    output=models.electric_field_from_potential()),
    solver=adc.EllipticSolver("geometric_mg"))
```

---

## 9. Decision : interface unique (Vars + Flux + Source), deux backends de Flux

Design converge avec le user. L'INTERFACE du modele reste la meme partout : `Vars + Flux + Source`
(+ EPM pour l'elliptique). Tout le numerique (reconstruction, flux numerique NF, assembleur) ne
consomme QUE cette interface. Derriere l'interface `Flux`, deux implementations interchangeables :

- **CompiledFlux** (defaut, PRODUCTION) : briques C++ compilees `ADC_HD` (CompressibleFlux=Euler,
  IsothermalFlux, ExBVelocity...). Rapides, compatibles Kokkos / GPU / MPI. Vivent dans adc_cpp comme
  operateurs generiques (au meme titre que Rusanov ou les limiteurs). C'est le chemin de production.
- **PythonFlux** (PROTOTYPAGE) : une fonction fournie depuis Python / adc_cases (numpy vectorise :
  flux(U)->F, max_wave_speed(U)->a, cons<->prim). Lent, CPU / HOTE uniquement. Permet de prototyper
  un flux inedit sans recompiler. REGLE D'OR : ne doit JAMAIS etre utilisee dans un kernel Kokkos /
  GPU (garde-fou : si backend Kokkos/GPU actif -> erreur claire).

Cote Python, `adc.Model(...)` peut donc SOIT selectionner une brique compilee (`adc.CompressibleFlux()`),
SOIT fournir une fonction prototype (PythonFlux). Meme interface, meme assembleur ; seul le backend de
flux change. La performance impose CompiledFlux ; PythonFlux est pour iterer vite, hors hot path GPU/MPI.

Frontiere adc_cpp / adc_cases :
- adc_cpp : moteur generique + briques compilees (flux, sources, operateurs elliptiques, numerique,
  reconstruction, time steppers, Poisson) + le contrat d'interface. Aucun SCENARIO nomme.
- adc_cases : SCENARIOS = compositions Python (diocotron, two-fluid, plasma...) + eventuels PythonFlux
  de prototypage. Les NOMS de scenarios vivent ici.

Le pattern `custom_scheme` (tache #40, deja faite) est l'ancetre du PythonFlux ; reste a le faire
entrer proprement DERRIERE l'interface Vars+Flux+Source, avec le garde-fou Kokkos/GPU (tache #62).
Les flux nommes (Euler/isotherme/ExB) NE sont PAS deplaces hors de adc_cpp : ils sont le CompiledFlux.

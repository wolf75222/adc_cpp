# adc — Cap multi-espèces : du `PhysicalModel` au `CoupledSystem`

*Document de conception pour la session tableau (avec Sacha). Met à jour la feuille de
route après les remarques du tuteur et les incréments déjà réalisés.*

---

## 1. Le cadre (ce que le tuteur valide, ce qu'il manque)

Le tuteur **ne remet pas en cause `PhysicalModel`**. Sur la transcription :
« le modèle physique, je n'ai rien à dire, ça a l'air flexible ; la structure avec
templates va dans le bon sens ». `PhysicalModel` est le **bon niveau local** : une
équation d'une espèce (flux, max_wave_speed, source, elliptic_rhs), device-callable.

Ce qui manque, c'est le **niveau au-dessus** : assembler **plusieurs** `PhysicalModel`,
chacun avec **sa** méthode spatiale, **son** intégrateur temporel, **son** pas de temps,
et des couplages **globaux** (Poisson et sources voient toutes les espèces).

```
PhysicalModel   décrit une équation LOCALE          (déjà bon)
EquationBlock   = State + Model + Spatial + Time     (à ajouter)
CoupledSystem   = plusieurs EquationBlock            (à ajouter)
Scheduler       = ordre des steps, sous-pas, IMEX    (à ajouter)
PoissonRHS      = assemblé depuis toutes les espèces (à ajouter)
```

Le cœur garantit que ces choix restent compatibles **AMR / MPI / GPU**.

## 1bis. Posture (ce qu'il attend MAINTENANT)

Quatre principes qui cadrent l'objectif immédiat (ne pas survendre, ne pas figer) :

- **Un squelette, pas le solveur final.** L'objectif est de poser les bonnes briques et de
  les **tester sur des cas simples**, pas de tout coder ni d'optimiser.
- **Abstraction avant structure de données.** Il distingue trois niveaux : (1) abstraction,
  (2) architecture / interaction des classes, (3) structure de données. La (3) « se change
  facilement plus tard ». On montre d'abord *qui dépend de qui, qui assemble quoi*, pas le
  layout mémoire (`MultiFab`, stockage stacké…).
- **Performance après stabilisation.** « Optimiser, je saurai le faire quand le code sera
  propre et figé. » Message : bonnes abstractions d'abord, optimisation ensuite, pas
  l'inverse.
- **Validé par un utilisateur, pas par le compilateur.** Sacha est un utilisateur clé. Le
  vrai test n'est pas « est-ce que ça compile ? » mais : **un utilisateur peut-il décrire
  son système physique sans comprendre l'AMR / MPI / GPU ?** Question centrale : *quelle API
  minimale permet à Sacha de décrire son cas ?*

**Diffusion = un flux, pas une nouvelle couche.** Le parabolique est une contribution de
flux supplémentaire dans l'opérateur spatial, pas une grande abstraction. (Fait sur grille
uniforme via `DiffusiveModel` ; sur AMR le moteur travaille par **flux de face** pour le
reflux, donc la diffusion doit y passer comme **flux de face diffusif** : c'est le
follow-up, pas un nouveau niveau.)

## 2. Traduction des remarques du tuteur en exigences

| Il dit | Ça veut dire |
|---|---|
| « U pour les ions, U pour les électrons » | `State` ne doit plus être unique : plusieurs champs `U_k`. |
| « ions isothermes, électrons Euler » | chaque espèce a son propre `PhysicalModel`. |
| « 10 pas électrons, 1 pas ion » | un `Scheduler`/`TimePolicy` par bloc (sous-cyclage). |
| « électrons implicites, ions explicites » | le `TimeIntegrator` est **par bloc**, pas global. |
| « Rusanov ions, HLL électrons » | la `SpatialDiscretisation` est **par bloc**, pas globale. |
| « ils interagissent dans `f(U)` et dans `S` » | `elliptic_rhs(u)` et `source(u,aux)` locaux sont trop étroits : il faut une source de **couplage** inter-espèces + un RHS elliptique **global**. |

Le `Coupler` cible n'est plus « hyperbolique + elliptique » mais un **assembleur de
système** : il prend `{U_e, U_i, U_n, …}` + méthodes spatiales + méthodes temporelles +
solveurs elliptiques + ordre d'exécution.

## 3. Vision cible — abstractions C++

```cpp
// Un bloc = un état + un modèle + une discrétisation spatiale + une politique temporelle.
template <class Model, class Spatial, class Time>
struct EquationBlock {
  Model    model;
  MultiFab U;          // (ou une vue dans un état stacké, cf. §7)
  BCRec    bc;         // conditions au bord PAR BLOC (cf. ci-dessous)
  // Spatial = SpatialDiscretisation<Limiter, NumericalFlux>  (existe deja)
  // Time    = SSPRK2 / SSPRK3 / ImexImplicit ...             (tags, existent en partie)
  int substeps = 1;    // sous-cyclage temporel relatif
};

// Un systeme = plusieurs blocs + un couplage elliptique + un ordonnanceur.
template <class... Blocks>
struct CoupledSystem {
  std::tuple<Blocks...> blocks;
  PoissonCoupling       poisson;   // rhs = Σ_s q_s n_s, solveur MG/FFT
  Scheduler             scheduler; // qui avance quand, implicite/explicite
};
```

- **`EquationBlock`** : déjà à moitié là. `SpatialDiscretisation<Limiter, Flux>` existe
  et est sélectionnable au coupleur (uniforme **et** AMR). Les tags temporels `SSPRK2`/
  `SSPRK3` existent. Manque : les regrouper *par bloc*, avec `substeps` et le choix
  implicite/explicite.
- **`CoupledSystem`** : à créer. Couche d'assemblage de N blocs.
- **`Scheduler`** : à créer. Encode l'ordre (ex. 10 sous-pas électrons par pas ion),
  l'implicite ciblé (IMEX partiel).
- **`PoissonCoupling`** : à créer. Le RHS elliptique somme les contributions de toutes
  les espèces (`f = Σ_s α_s · model_s.elliptic_rhs(U_s)`), **champ φ unique partagé**.
  Conforme à la demande : couplage dans `f(U)`, pas dans `F`.

Adaptateur de non-régression : `SingleFieldSystem<Model>` enveloppe le `Coupler` actuel
comme un système à un bloc, pour ne rien casser pendant la transition.

### 3bis. Points d'architecture à acter

- **Conditions au bord par bloc.** Aujourd'hui les `BCRec` sont globales dans les coupleurs.
  Le multi-espèces exige des BC par bloc : électrons périodiques, ions mur/extrapolation,
  neutres profil imposé. → `bc` dans `EquationBlock` (ci-dessus), plus de BC globale unique.
- **L'AMR est un orchestrateur, pas une option de maillage.** Il prend une définition
  *locale, cellule par cellule* et la projette sur toute une hiérarchie de grilles :
  `fill_ghosts → subcycling → average_down → reflux → regrid`. Donc `AmrCoupler` /
  `AmrCouplerMP` ne sont pas de simples variantes techniques mais des **orchestrateurs
  d'exécution** : ils jouent déjà le rôle du futur `Scheduler` au niveau d'un bloc.
- **`Coupler` : un nom historique.** Conceptuellement cette couche est un **Assembler /
  Simulation Driver** (elle prend tous les ingrédients et les met sur le maillage). Le nom
  « Coupler » vient de « coupler hyperbolique + elliptique » ; on ne le défend pas à tout
  prix, on assume qu'il désigne l'assemblage.
- **Templates, mais on dessine les concepts.** Le tuteur accepte les templates (« ça va
  dans le bon sens ») tout en notant que c'est moins lisible que le virtuel. Donc on
  présente les **objets conceptuels** (`PhysicalModel`, `SpatialDiscretisation`,
  `TimeIntegrator`, `EquationBlock`, `CoupledSystem`, `Scheduler`), même si techniquement
  ce sont des templates.

## 4. API Python cible (composition, pas calcul)

Python **configure** le système ; les boucles cellules / AMR / MPI / GPU restent en C++.

```python
import adc

mesh = adc.Mesh2D(nx=512, ny=512, xlim=(0,1), ylim=(0,1),
                  amr=adc.AMR(levels=3, ratio=2))
sim  = adc.Simulation(mesh, backend="kokkos")   # cpu / openmp / mpi / kokkos

sim.add_equation(name="electrons",
    model=adc.models.ElectronEuler(charge=-1, mass=1, gamma=5/3),
    spatial=adc.FiniteVolume(reconstruction="vanleer", flux="hllc"),
    time=adc.Implicit(scheme="imex", substeps=10))

sim.add_equation(name="ions",
    model=adc.models.IonEuler(charge=+1, mass=1836, gamma=5/3),
    spatial=adc.FiniteVolume(reconstruction="minmod", flux="rusanov"),
    time=adc.Explicit(scheme="ssprk2", substeps=1))

sim.add_poisson(unknown="phi",
    rhs=adc.ChargeDensity(positive=["ions"], negative=["electrons"]),
    solver=adc.GeometricMG(tol=1e-10, max_iter=200))

sim.run(t_end=1.0, cfl=0.4,
        output=adc.Output(path="runs/two_species", every=20,
                          fields=["electrons.rho", "ions.rho", "phi"]))
```

Les chaînes (`flux="hllc"`, `time="imex"`) **sélectionnent des briques C++ compilées** ;
elles ne sont jamais des callbacks cellule-par-cellule (lent, non GPU/MPI-friendly). Un
utilisateur avancé écrit son `PhysicalModel` en C++ (`StateVec<N>`, `ADC_HD`) et l'expose
ensuite à Python — toujours en composition, jamais en boucle interne Python.

## 5. État actuel vs cible

| | Aujourd'hui | Cible |
|---|---|---|
| États | un seul `U` | `{U_e, U_i, U_n, …}` |
| Modèle | un `PhysicalModel` | un par bloc |
| Spatial | sélectionnable (uniforme + AMR) | **par bloc** |
| Temps | SSPRK2/3 sélectionnable au coupleur | **par bloc** (+ IMEX partiel, sous-cyclage) |
| Poisson | `f = model.elliptic_rhs(U)` (1 état) | `f = Σ_s q_s n_s` (toutes espèces) |
| Coupler | `Coupler<Model, Elliptic>` mono-bloc | `CoupledSystem<Blocks…>` |
| Python | pilote des façades (`DiocotronSolver`, …) | compose un système |

## 6. Ce qui est déjà fait (prépare le terrain, ne pré-empte pas la décision data)

Tous poussés, tous verts (adc_cpp 30/30, adc_cases 44/44 ; MPI 7+7) :

1. **Split cœur/applications** : `adc_cpp` = moteur générique (zéro modèle), `adc_cases`
   = modèles/façades/exemples/Python, via FetchContent (`adc::adc`).
2. **`SpatialDiscretisation<Limiter, NumericalFlux>`** + tags `SSPRK2`/`SSPRK3` +
   `Coupler::step<Disc, TimeInteg, Policy>` : discrétisation spatiale et intégrateur
   temps **sélectionnables** (le futur « par bloc » se branchera dessus).
3. **Diffusion comme un flux de plus** : trait `DiffusiveModel` (`+ν∆U` dans
   `assemble_rhs`), modèle `AdvectionDiffusion`. (Grille uniforme ; AMR = follow-up.)
4. **Solveur elliptique choisi à l'exécution** dans la façade diocotron (MG/FFT, pattern
   `variant`).
5. **AMR applique `model.source`** (le chemin AMR l'ignorait : bug de correction) +
   **`SpatialDiscretisation` câblée en AMR** (`AmrCoupler/MP::step<Disc>`, MUSCL conservatif).

## 7. Modifications à faire (ordonnées)

**Décision-mère (au tableau) — design des données :**
- `tuple<Blocks…>` (chaque bloc garde son `StateVec<n_k>`, composition variadique) **vs**
  `StateVec<N_total>` empilé (un bloc mémoire contigu, offsets par espèce). Conditionne
  tout le reste (perf de localité, vues, GPU). À trancher avec Sacha.

**Puis, dans l'ordre :**
1. **`EllipticRhsAssembler` / `PoissonCoupling`** : sortir `elliptic_rhs` du chemin
   mono-état. `f = Σ_s α_s · model_s.elliptic_rhs(U_s)`, φ partagé.
   Fichiers : `coupling/coupler.hpp` (`detail::coupler_eval_rhs`), `core/physical_model.hpp`.
2. **`EquationBlock<Model, Spatial, Time>`** : type d'agrégation (état + politiques).
   Adaptateur `SingleFieldSystem<Model>` pour non-régression.
3. **`CoupledSystem<Blocks…>` + `Scheduler`** : assemble N blocs ; gère l'ordre des
   sous-pas et l'implicite/explicite par bloc.
4. **Source de couplage inter-espèces** : distinguer `source` locale du modèle et
   couplage (collisions, échange) qui voit les autres états.
5. **IMEX partiel** : intégrateur traitant implicitement un **sous-ensemble** des
   variables (électrons) et explicitement le reste (ions). Trait `which_implicit()`.
6. **Sous-cyclage temporel par espèce** : réutiliser le sous-cyclage AMR (Berger-Oliger)
   pour les blocs (10 pas électrons / 1 pas ion).
7. **Façade compilée `MultiSpeciesSolver(vector<SpeciesConfig>)`** + **API Python de
   composition** (`sim.add_equation(...)`, `sim.add_poisson(...)`, `sim.run()`).
8. **Nettoyages** : `SpectralCoupler` ne doit plus coder la physique diocotron en dur
   (appeler `model.elliptic_rhs` / `model.max_wave_speed`) ; clarifier le contrat `Aux`
   (fixe `phi, grad φ` ou réellement `typename M::Aux`) ; diffusion sur AMR (flux de face
   diffusif pour rester conservatif au reflux).

## 7bis. Cas simples pour tester le squelette

L'architecture se valide sur des cas **utilisateur** simples (pas un solveur de prod) :

- électrons Euler + ions Euler **isothermes** + Poisson (le cas canonique deux-espèces) ;
- **diocotron à ions fixes** (≈ ce qui existe : `n_i0` constant) ;
- **diocotron à ions mobiles** (les ions deviennent un second bloc) ;
- gaz / neutres **résolus** ou **prescrits** (profil imposé, pas résolu chaque pas).

Critère de réussite : un utilisateur (Sacha) peut décrire ces cas **sans toucher** à
l'AMR / MPI / GPU. Si l'API ne le permet pas, l'abstraction est incomplète.

## 7ter. Lexique (pour les slides, à définir avant de jeter les sigles)

| Terme | Sens court |
|---|---|
| `BoxArray` | découpage du domaine en blocs (boîtes) |
| `MultiFab` | les champs `U` stockés sur ces blocs (collection distribuée) |
| `BCRec` | conditions aux limites d'un champ |
| `aux` | variable auxiliaire transportée (ici `phi, grad phi`) |
| seam | couture où vit le parallélisme (`for_each_cell`, `comm`) |

Note Python : ne jamais appeler `flux()` cellule par cellule depuis Python (lent, non
GPU/MPI). Python **configure** le système, ou fournit des champs **vectorisés** (numpy sur
toutes les cellules). Le hot path cellule reste en C++.

## 8. Synthèse (phrase tableau)

> `adc` sait déjà prendre une **loi physique locale** et la faire tourner sur un maillage
> avec Poisson, AMR, MPI et GPU. Ce qui manque pour devenir une **bibliothèque de
> construction de solveurs**, c'est un niveau d'**assemblage multi-blocs** : plusieurs
> états, plusieurs modèles, plusieurs méthodes numériques, plusieurs pas de temps, et des
> couplages globaux dans Poisson et dans les sources.
>
> Le `PhysicalModel` décrit une équation locale. Le `CoupledSystem` décrit un système
> physique. Le `Scheduler` décrit l'ordre d'exécution. Le cœur `adc` garantit que ces
> choix restent compatibles AMR / MPI / GPU.

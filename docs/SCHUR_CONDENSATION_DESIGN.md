> **STATUT : IMPLÉMENTÉ.** Ce document est la spec de conception d'origine. L'étage Schur condensé est livré (`CondensedSchur`, `schur_condensation.hpp`, exposé Python `adc.CondensedSchur` ; tests test_schur_via_system / test_schur_conservation). Lire ce fichier comme historique de conception, pas comme état courant.

# Conception : source implicite condensee par Schur (reproduction arXiv:2510.11808)

DESIGN-ONLY, aucune implementation. Ce document est une SPEC raisonnee, honnete sur ses
limites, d'une nouvelle ABSTRACTION numerique pour `adc_cpp` : la condensation de Schur de la
source implicite couplee (potentiel / vitesse / Lorentz) du schema de Hoffart, Maier, Shadid,
Tomas (arXiv:2510.11808). Il decrit la cible, les contraintes et le sequencement ; il ne
promet PAS la reproduction du papier (voir section 9, la feuille de route en est volontairement
prudente).

Le document s'appuie sur l'architecture deja en place (sources lues) :
- `docs/ARCHITECTURE.md` sections 1 (cinq couches orthogonales), 6 (modele physique generique
  `CompositeModel`), 7 (etage elliptique : `EllipticProblem` / `EllipticOperator` /
  `LinearSolver` / `FieldPostProcess`), 8 (AMR distribue) ;
- `docs/ALGORITHMS.md` (briques FV, operateur elliptique, cut-cell Shortley-Weller) ;
- `docs/PAPER_ROADMAP.md` (etat de reproduction Hoffart, verrou du bord d'anneau cartesien) ;
- `docs/DSL_MODEL_DESIGN.md` (facade `dsl.Model`, roles physiques, `add_native_block`) ;
- `docs/GPU_RUNTIME_PORT.md` (recette device-clean : foncteurs nommes, pas de lambda etendue) ;
- la Phase 1 polaire MERGEE (#116, commit `004efca`) : l'abstraction MAILLAGE
  (`adc.CartesianMesh` / `adc.PolarMesh` -> `System(mesh=)`), avec `adc.FiniteVolume` = recon +
  Riemann + variables UNIQUEMENT (aucun argument de geometrie) ;
- `include/adc/core/variables.hpp` (`VariableRole` : `Density`, `MomentumX`, `MomentumY`,
  `MomentumZ`, `Energy`, ...) ;
- `docs/BIBLIOGRAPHY.md` section 3 (entree Hoffart).

NOTE D'HONNETETE LIMINAIRE. Ce qui suit decrit une CIBLE. Rien n'est livre. Le verrou
scientifique du diocotron (bord d'anneau cartesien, `docs/PAPER_ROADMAP.md` panier 3) est
ORTHOGONAL a ce chantier : la condensation de Schur est un schema en TEMPS pour la source
couplee, elle ne corrige PAS la diffusion spatiale du transport. Les deux peuvent atterrir
independamment.


## 1. Le splitting du papier

Hoffart et al. integrent le systeme deux-fluides de derive en SEPARANT trois operateurs, ce qui
est exactement le style de `TimeIntegrator` de la couche 5 (`docs/ARCHITECTURE.md` section 1 :
splitting / IMEX / AP composant des operateurs sans connaitre leur interieur) :

1. **Transport hyperbolique** : l'advection conservative des champs fluides (densite, quantite
   de mouvement, energie le cas echeant). C'est exactement le chemin FV existant
   (`numerics/spatial_operator.hpp`, reconstruction + flux numerique), explicite, SSPRK2/SSPRK3.
2. **Source implicite couplee** : le sous-systeme potentiel / vitesse / force de Lorentz, traite
   IMPLICITEMENT (theta-schema) parce que la rotation cyclotron et le rappel electrostatique sont
   raides dans la limite de derive.
3. **Condensation de Schur** : on ELIMINE algebriquement la vitesse du sous-systeme implicite, ce
   qui reduit l'etage source a un unique solve elliptique de type Poisson MODIFIE pour le
   potentiel, suivi d'une reconstruction explicite de la vitesse. C'est la cle du cout : un solve
   elliptique par etage source au lieu d'une inversion couplee dense.

Le decoupage (1) explicite + (2)-(3) implicite est une IMEX au sens de la couche temps. La
contribution de ce document est l'etage (2)-(3) : la condensation, et surtout l'operateur
elliptique MODIFIE qu'elle produit.


## 2. Les equations de la source

Sous-systeme source (transport gele), avec `rho` la densite, `v` la vitesse, `phi` le potentiel,
`Omega` le vecteur cyclotron (porte par `B_z` hors-plan), `alpha` une constante de couplage :

```
d_t rho = 0
d_t v   = -grad(phi) + v x Omega
d_t(-Lap phi) = -alpha div(rho v)
```

La densite est gelee dans la source (tout son transport est dans l'etage (1)). La vitesse subit
le rappel electrostatique `-grad(phi)` et la rotation de Lorentz `v x Omega`. La derniere
equation est la contrainte de Poisson differentiee en temps (la charge evolue par la divergence
du courant `rho v`).

### 2.1 Theta-schema et condensation

On discretise en temps par un theta-schema (`theta in [0,1]`, `theta=1/2` = Crank-Nicolson). On
note `B` l'operateur LOCAL (point par point) de rotation implicite :

```
B v = v - theta dt (v x Omega)
```

`B` est une matrice 2x2 par cellule (plan (x,y)), inversible des que `theta dt |Omega|` est fini ;
son inverse `B^{-1}` est CLOS (rotation-dilatation 2x2). La vitesse implicite s'ecrit alors

```
v^{n+theta} = B^{-1} (v^n - theta dt grad phi^{n+theta})
```

En injectant cette expression dans la contrainte de Poisson differentiee, on ELIMINE `v` (c'est
le complement de Schur du bloc vitesse) et on obtient une equation pour le SEUL `phi^{n+theta}` :

```
-Lap phi^{n+theta}
  - theta^2 dt^2 alpha div( rho^n B^{-1} grad phi^{n+theta} )
  = -Lap phi^n
  - theta dt alpha div( rho^n B^{-1} v^n )
```

puis on RECONSTRUIT la vitesse :

```
v^{n+theta} = B^{-1} ( v^n - theta dt grad phi^{n+theta} )
```

et, le cas echeant, on extrapole l'etat complet `U^{n+theta}` (et l'energie) avant de remplir les
ghosts.


## 3. POINT-CLE : pourquoi `source(U, aux)` ne suffit pas

L'erreur de conception a eviter est de traiter cet etage comme une source LOCALE de plus, du type
`source(U, aux)` (la signature des briques source actuelles, `docs/ARCHITECTURE.md` section 6 :
une source ne voit qu'une cellule et ses champs `Aux`). C'est IMPOSSIBLE ici, pour une raison
STRUCTURELLE et non d'ergonomie :

> La condensation de Schur MODIFIE l'operateur elliptique lui-meme.

L'equation condensee n'est pas `-Lap phi = f(rho, v)` (un Poisson standard avec un second membre
recalcule, ce que le couplage actuel sait deja faire via `elliptic_rhs(U)`). C'est

```
L_schur(phi) = -Lap phi - theta^2 dt^2 alpha div( A grad phi )    avec A = rho B^{-1}
```

soit un operateur `-div(A grad phi) + (terme Laplacien)` dont le COEFFICIENT TENSORIEL `A`
2x2 :
- depend de l'etat (`rho^n` et `Omega` par cellule), donc CHANGE a chaque pas de temps ;
- est en general NON SYMETRIQUE : `B^{-1}` est une rotation-dilatation (partie antisymetrique non
  nulle des que `Omega != 0`), donc `A = rho B^{-1}` a une partie antisymetrique. L'operateur
  `-div(A grad .)` n'est donc PAS auto-adjoint en general.

Une source locale ne peut pas exprimer cela : `source(U, aux)` produit une contribution PAR
CELLULE au residu, alors que `div(A grad phi)` est un OPERATEUR GLOBAL (stencil couplant les
voisins, a inverser). Le terme appartient au membre de GAUCHE de l'etage elliptique, pas au
membre de droite. Concretement :
- l'`elliptic_rhs(U)` actuel (`physics/elliptic.hpp`, second membre de Poisson, cote couche 5)
  ne touche que le RHS : il ne peut PAS injecter un terme dans l'OPERATEUR ;
- `poisson_operator.hpp` (le Laplacien 5 points canonique, `docs/ARCHITECTURE.md` section 7) est
  un Laplacien SCALAIRE a coefficient constant : il n'a pas de place pour un coefficient tensoriel
  `A(x)` par cellule.

Donc l'etage source de Schur exige (a) un NOUVEL operateur elliptique tensoriel, et (b) un schema
local-global qui le CONSTRUIT a partir de l'etat, le RESOUT, puis reconstruit la vitesse. Ce
n'est ni une source locale ni un Poisson standard : c'est une abstraction numerique a part
entiere.


## 4. POINT-CLE : ne PAS coder un `DiocotronSchurSolver` dans le coeur

La tentation symetrique est de baker un solveur nomme `DiocotronSchurSolver` dans le coeur. C'est
contraire au principe directeur d'`adc_cpp` (`docs/ARCHITECTURE.md` : "le coeur est AGNOSTIQUE au
modele, il ne nomme aucun scenario"). La separation correcte est :

- **Modele = PHYSIQUE.** Un modele declare ses variables et leurs ROLES (`VariableRole` :
  `Density`, `MomentumX`, `MomentumY`, `Energy`, ...), son flux, sa source LOCALE, son second
  membre elliptique, ses parametres. Il ne sait rien de Schur ni du theta-schema.
- **SchurCondensation = ALGORITHME numerique local-global.** C'est une politique de la couche
  numerique + temps (couches 2 et 5) : a partir d'un etat qui EXPOSE les bons roles, elle
  assemble l'operateur tensoriel, le resout, reconstruit la vitesse. Elle ne nomme aucun
  scenario.

Premiere implementation concrete : **`ElectrostaticLorentzCondensation`**, GENERIQUE sur toute
espece fluide qui expose les roles
- `Density`, `MomentumX`, `MomentumY` (et `Energy` optionnellement),
- plus l'acces a `phi`, `grad phi`, le champ `B_z` / `Omega`, et la constante `alpha`.

Elle marche pour un modele ecrit en BRIQUES C++ (`CompositeModel`, `add_block`) OU en DSL compile
(`dsl.Model` -> `add_native_block`), pourvu que les ROLES soient presents. Le diocotron de derive
n'est qu'UN client ; un modele a deux especes magnetisees en serait un autre, sans une ligne de
code Schur supplementaire. C'est l'inverse exact d'un solveur nomme par scenario.

CONTRAT MINIMAL exige du modele par `ElectrostaticLorentzCondensation` :
1. un role `Density` (lecture de `rho^n`) ;
2. les roles `MomentumX`/`MomentumY` (ou `VelocityX`/`VelocityY`) pour lire/reconstruire `v` ;
3. un champ aux `B_z` (ou `Omega`) deja peuple (canal `Aux`, comme `set_magnetic_field`
   aujourd'hui) ;
4. un potentiel `phi` et un acces `grad phi` (deja fournis par l'etage elliptique couple) ;
5. la constante de couplage `alpha`.
Tout modele qui satisfait ce contrat est eligible, qu'il vienne des briques ou du DSL.


## 5. Architecture C++ cible (descriptive, AUCUNE implementation)

Cinq niveaux, alignes sur les cinq couches existantes. Chaque niveau est device-callable la ou il
tourne dans la boucle chaude, sans allocation, sans `std::function`, sans polymorphisme dynamique
(`docs/ARCHITECTURE.md` section 1 ; recette device-clean de `docs/GPU_RUNTIME_PORT.md`).

### Niveau 1 : `TensorEllipticOperator` (couche numerique)

L'operateur discret `L(phi) = -div(A grad phi) + kappa phi`, avec `A` un coefficient TENSORIEL 2x2
par cellule (potentiellement non symetrique) et `kappa` un terme de masse scalaire optionnel. Il
GENERALISE `poisson_operator.hpp` (Laplacien scalaire 5 points) : meme role d'`OperatorSpec`
partage (`apply`, `residual`, restriction/prolongation pour le MG), mais avec un stencil a flux de
face tensoriel. Les coefficients `A` vivent dans un `MultiFab` (composante par composante du 2x2,
ou stockage compact des 4 entrees), restreints par `average_down` pour les niveaux grossiers du
MG, comme `eps(x)` aujourd'hui (`set_epsilon(eps_fine)`, `docs/ARCHITECTURE.md` section 7). Le cas
`A = I`, `kappa = 0` doit retomber EXACTEMENT (bit-identique) sur le Laplacien canonique : c'est le
garde-fou de non-regression.

### Niveau 2 : le solveur, et la QUESTION de la non-symetrie (NOTE TECHNIQUE A TRANCHER)

Le solveur lineaire depend de la symetrie de `A`, qui depend de la PHYSIQUE :

- **`A` symetrique** quand la partie antisymetrique de `B^{-1}` s'annule, c.-a-d. `Omega = 0`
  (pas de champ magnetique : on retombe sur un Poisson a coefficient `rho`, symetrique defini
  positif si `rho > 0`). Le cas `B_z` constant ne suffit PAS a lui seul : `B^{-1}` reste une
  rotation des que `Omega != 0`. Le cas reellement symetrique est donc `Omega = 0`. Le cas
  `B_z` constant ET `rho` constant donne un operateur a coefficient CONSTANT mais toujours NON
  symetrique (rotation pure) : il est juste plus facile a preconditionner, pas symetrique.
- **`A` non symetrique** des que `Omega != 0` (cas diocotron magnetise), et a fortiori quand
  `rho` varie en espace (coefficient variable ET non symetrique).

Consequence : le `GeometricMG` actuel (V-cycle, lisseur Gauss-Seidel rouge-noir) suppose un
operateur symetrique defini positif. Pour `A` non symetrique, le V-cycle MG seul n'est PAS un
solveur fiable. Direction CIBLE (a CONFIRMER, c'est le point technique a trancher avant
l'implementation du niveau 2) :

> Un solveur de KRYLOV non symetrique (GMRES ou BiCGStab) PRECONDITIONNE par un V-cycle MG
> applique a la PARTIE SYMETRIQUE de l'operateur (le `-div(rho B^{-1}_sym grad .)` symetrise, ou
> simplement le Poisson scalaire `rho`-pondere).

Cette piste est PLAUSIBLE (le terme antisymetrique est en `theta^2 dt^2 alpha`, donc petit a CFL
source raisonnable : le preconditionneur symetrique devrait capturer l'essentiel). Mais elle reste
A VALIDER : on ne sait pas a ce stade (a) le nombre d'iterations Krylov reel, (b) la robustesse du
preconditionnement MG-sur-partie-symetrique quand `theta dt |Omega|` grandit, (c) si un simple
preconditionneur de Jacobi par blocs 2x2 suffirait pour les regimes vises. Aucune brique GMRES /
BiCGStab n'existe aujourd'hui dans `numerics/elliptic/` (le concept `EllipticSolver` n'a que MG et
FFT). C'est donc un AJOUT, derriere le concept `EllipticSolver` existant (`rhs`/`phi`/`solve`/
`residual`/`geom`), pour ne casser aucun appelant. FLAG : trancher (Krylov + preconditionneur) en
PR3 avant tout cablage facade.

### Niveau 3 : `LocalLorentzEliminator` (couche numerique, device-callable)

Le coeur LOCAL : par cellule, construire `B = I - theta dt [Omega]_x` (la matrice 2x2 de rotation
implicite) et son inverse `B^{-1}` en forme close (pas de solve, pas d'allocation). Device-callable
(`ADC_HD`), foncteur NOMME (pas de lambda etendue, cf. `docs/GPU_RUNTIME_PORT.md` : la limite nvcc
des lambdas etendues premiere-instanciees cross-TU est contournee par des foncteurs nommes, recette
deja validee GH200 pour le transport #64 et l'elliptique #97). Sert a deux endroits : (a) assembler
le coefficient `A = rho B^{-1}` du niveau 1, (b) reconstruire `v^{n+theta}` au niveau 4. Aucune
matrice n'est materialisee globalement : `B^{-1}` est recalcule a la volee par cellule.

### Niveau 4 : `CondensedSchurSourceStepper` (couche temps / couplage)

L'orchestrateur de l'etage source, joue UNE fois par etage IMEX (frequence portee par
`CouplingPolicy`, `PerStage` ou `OncePerStep`, `docs/ARCHITECTURE.md` section 7). Sequence :
1. **Construire l'operateur** : remplir le `MultiFab` de coefficients `A = rho^n B^{-1}` (niveau 3
   par cellule) ; configurer `L_schur` (niveau 1) avec `theta`, `dt`, `alpha`.
2. **Construire le RHS** : `-Lap phi^n - theta dt alpha div( rho^n B^{-1} v^n )` (divergence
   discrete d'un flux de face tensoriel ; reutilise l'assemblage de face existant).
3. **Resoudre `phi^{n+theta}`** : appeler le solveur du niveau 2 derriere le concept
   `EllipticSolver`.
4. **Reconstruire `U^{n+theta}`** : `v^{n+theta} = B^{-1}(v^n - theta dt grad phi^{n+theta})`
   (niveau 3), puis recomposer momentum/energie selon les roles du modele.
5. **Extrapoler** : passer de l'etat `n+theta` a `n+1` selon le theta-schema (extrapolation
   lineaire ; `phi^{n+1}`, `v^{n+1}` deduits).
6. **Mise a jour energie** : si le modele porte un role `Energy`, recalculer l'energie coherente
   avec la nouvelle vitesse (travail de la force electrostatique).
7. **Remplir les ghosts** (`fill_boundary`, halos MPI) avant de rendre la main a l'etage transport.

Le stepper ne nomme aucun scenario : il lit les roles, appelle les niveaux 1-3, et delegue le
solve au concept `EllipticSolver`. C'est lui qui materialise le local-global (assemblage local du
coefficient -> solve global -> reconstruction locale).

### Niveau 5 : exposition DSL (couche facade)

Au depart, le DSL n'a RIEN a ajouter : la condensation est selectionnee par la PRESENCE des roles
requis (`Density`/`MomentumX`/`MomentumY` + `phi`/`B_z`/`alpha`) et par le choix de l'integrateur
cote facade (section 6). Un modele DSL existant (`dsl.Model`) qui declare ces roles est
directement eligible via `add_native_block` (chemin natif zero-copie, GPU/MPI, `docs/
DSL_MODEL_DESIGN.md`). PLUS TARD seulement, un declarateur explicite `m.implicit_coupling(...)`
(nommant `phi`, `v`, `Omega`, `alpha`, `theta`) pourrait rendre l'intention lisible dans la
formule et lever des erreurs au plus tot ; ce n'est PAS requis pour le premier client (roles
suffisent) et reste differe.


## 6. API Python cible

Le point d'entree est un splitting EXPLICITE/IMPLICITE explicite, qui compose un integrateur
hyperbolique et un etage source condense :

```python
sim.add_equation(
    "ions",
    model=model,                       # briques natives OU CompiledModel DSL, roles requis
    time=adc.Split(
        hyperbolic=adc.Explicit(ssprk3=True),
        source=adc.CondensedSchur(
            kind="electrostatic_lorentz",
            theta=0.5,
            density="rho",             # role Density
            momentum=("mx", "my"),     # roles MomentumX / MomentumY
            energy="E",                # role Energy (optionnel)
            magnetic_field="B_z",      # champ aux Omega / B_z
            potential="phi",
        ),
    ),
)
```

Principes de cette API :
- `adc.Split(hyperbolic=, source=)` est un nouvel integrateur de la couche temps : il joue
  l'etage hyperbolique explicite puis l'etage source `CondensedSchur` (IMEX au sens couche 5).
- `adc.CondensedSchur(kind=, theta=, ...)` NOMME l'algorithme et MAPPE les champs sur les roles.
  `kind="electrostatic_lorentz"` selectionne `ElectrostaticLorentzCondensation` (niveau 4-3) ;
  d'autres `kind` pourront s'ajouter sans toucher la facade.
- **Le chemin par defaut est INCHANGE.** Rien ne casse : `adc.Explicit`, `adc.IMEX`,
  `adc.Implicit`, `add_block`, `add_equation` continuent de fonctionner a l'identique. Un modele
  qui n'emploie pas `adc.Split(... source=adc.CondensedSchur ...)` ne voit jamais le nouvel etage.
  La selection est OPT-IN, comme la grille polaire (#116, defaut cartesien bit-identique).

ENCART -- `adc.SourceImplicit` (LOCAL) vs `adc.CondensedSchur` (GLOBAL). Deux mecanismes
distincts traitent une source raide implicitement ; ne pas les confondre.
- `adc.SourceImplicit` (= IMEX source-only) est LOCAL : l'implicite ne couple que les composantes
  d'UNE MEME cellule (backward-Euler resolu par Newton a la cellule), AUCUN couplage spatial.
  C'est le bon choix pour les termes raides purement locaux (relaxation, reactions, friction) :
  pas de solve elliptique, donc bien moins cher.
- `adc.CondensedSchur` (via `adc.Split`, cf. sections 2 a 5) est GLOBAL : il assemble l'operateur
  elliptique tensoriel condense et le resout par Krylov (BiCGStab), couplant TOUT le domaine.
  C'est le bon choix UNIQUEMENT pour un couplage raide non local (Lorentz / electrostatique, ex.
  Euler-Poisson magnetise de Hoffart). Une source raide locale n'a pas besoin de Schur.


## 7. Geometrie = abstraction MAILLAGE, pas une option de `FiniteVolume`

CONTRAINTE STRUCTURELLE, deja posee par la Phase 1 polaire MERGEE (#116, commit `004efca`). Le
CHOIX de geometrie vit dans un objet MAILLAGE, jamais dans le schema :
- `adc.CartesianMesh(...)` / `adc.PolarMesh(...)` -> `adc.System(mesh=...)` portent la geometrie
  (`SystemConfig` porte `geometry`, `nr`, `ntheta`, `r_min`, `r_max`).
- `adc.FiniteVolume(limiter=, riemann=, variables=)` reste reconstruction + flux numerique +
  variables UNIQUEMENT. Il N'A AUCUN argument de geometrie, et n'en aura jamais.

Pour la condensation de Schur, cela impose que le `TensorEllipticOperator` (niveau 1) et son
assemblage de flux de face respectent la GEOMETRIE portee par le `Mesh` (metriques cartesiennes ou
polaires). La divergence `div(A grad phi)` et l'inverse de Lorentz `B^{-1}` doivent etre exprimes
dans la base de la geometrie active (en polaire : la divergence ponderee par le rayon de face
`(1/r) d_r(r F_r) + (1/r) d_theta F_theta`, comme `assemble_rhs_polar` de #116 ; et l'inverse de
Lorentz en base locale `(r, theta)`, comme `ExBVelocityPolar`). La condensation est donc
parametree par le `Mesh`, pas par `FiniteVolume`. C'est coherent avec le verrou du
`docs/PAPER_ROADMAP.md` (panier 3) : si l'on veut un jour combiner Schur + grille polaire pour le
diocotron, le seam est deja a la bonne place (le maillage), mais ce sont deux chantiers distincts.


## 8. GPU / MPI des le depart, AMR ensuite

Contrainte non negociable, alignee sur l'etat de production du coeur
(`docs/DSL_MODEL_DESIGN.md` section 5, `docs/GPU_RUNTIME_PORT.md`) :
- **Foncteurs NOMMES**, pas de lambda etendue `__host__ __device__` premiere-instanciee cross-TU
  (limite nvcc contournee, validee GH200 pour le transport #64 et l'elliptique #97). Les niveaux 1
  et 3 (operateur tensoriel, eliminateur de Lorentz) doivent etre des foncteurs nommes
  device-clean des l'ecriture.
- **Coefficients en `MultiFab`** : `A = rho B^{-1}` est un champ par cellule (4 composantes du
  2x2, ou stockage compact), pas un parametre scalaire ; restreint par `average_down` pour le MG,
  comme `eps(x)`.
- **`local_size() == 0` sur, MPI-propre** : tout post-traitement par cellule garde les rangs sans
  box (cf. le correctif `solve_fields` MPI #99 : `fab(0)` sans garde segfaute sur un rang vide).
  L'assemblage du coefficient et la reconstruction de vitesse doivent etre gardes.
- **Aucun Python dans le hot path** : la facade Python ne fait que CONFIGURER l'etage (roles,
  `theta`, `alpha`) ; la boucle est entierement C++ (chemin `add_native_block` zero-copie).
- **AMR seulement APRES le chemin uniforme**. Le `TensorEllipticOperator` sur hierarchie AMR
  (restriction/prolongation du coefficient tensoriel, reflux coherent avec l'etage source) est un
  chantier a part, a ne PAS tenter avant que l'uniforme mono-niveau soit valide CPU + GPU + MPI.
  Reserve connue : `AmrSystem` n'est pas a parite avec `System` (mono-bloc, explicite,
  `docs/ARCHITECTURE.md` section 8) ; brancher la condensation sur AMR demandera d'abord cette
  parite.


## 9. Sequencement des PR (feuille de route, HONNETE sur l'horizon)

PR doc-only d'abord, puis une montee progressive. Cette liste est une CIBLE, pas un engagement de
calendrier ; les PR tardives sont LOIN et certaines dependent de questions non tranchees (le
solveur non symetrique du niveau 2, l'AMR). Ce document NE PROMET PAS la reproduction du papier :
au mieux, il pose l'INFRASTRUCTURE numerique de l'etage source condense, qui est NECESSAIRE mais
PAS SUFFISANTE (le verrou du bord d'anneau cartesien, `docs/PAPER_ROADMAP.md` panier 3, est
orthogonal et reste ouvert).

- **PR0 (ce document)** : la SPEC de conception. Doc-only.
- **PR1** : `TensorEllipticOperator` (niveau 1) en SCALAIRE generalise + non-regression bit-
  identique au Laplacien canonique pour `A = I`, `kappa = 0`. MMS ordre 2 sur un coefficient
  tensoriel constant SYMETRIQUE (cas test analytique, pas de physique). CPU serie.
- **PR2** : `LocalLorentzEliminator` (niveau 3), foncteur nomme device-clean, test unitaire
  `B B^{-1} = I` par cellule. Assemblage du coefficient `A = rho B^{-1}` en `MultiFab`.
- **PR3** : le solveur non symetrique du niveau 2. C'est le POINT TECHNIQUE A TRANCHER (Krylov
  GMRES/BiCGStab + preconditionneur MG sur la partie symetrique, ou Jacobi par blocs). PR de
  recherche : on mesure les iterations et la robustesse en `theta dt |Omega|` avant de figer le
  choix. Risque le plus eleve de la sequence.
- **PR4** : `CondensedSchurSourceStepper` (niveau 4) + `adc.Split` / `adc.CondensedSchur` (API
  Python, section 6), chemin natif `add_native_block`. Defaut inchange. Validation : un cas
  manufacture (MMS sur le sous-systeme source seul, transport gele) ; PAS encore le diocotron.
- **PR5** : portage GPU (foncteurs nommes deja en place ; valider parite Serial vs Cuda sur GH200,
  comme #97).
- **PR6** : portage MPI (parite bit-invariante au nombre de rangs, `local_size()==0` garde, comme
  #99/#93).
- **PR7** : declarateur DSL optionnel `m.implicit_coupling(...)` (niveau 5, sucre + erreurs au plus
  tot). Non bloquant.
- **PR8** : AMR (`TensorEllipticOperator` sur hierarchie, restriction du coefficient, reflux).
  LOIN, conditionne par la parite `AmrSystem` <-> `System` (`docs/ARCHITECTURE.md` section 8).

LIMITES connues a garder en tete (ne PAS les masquer) :
- la non-symetrie du niveau 2 est un VRAI risque numerique non resolu (PR3) ;
- aucune de ces PR ne traite le bord d'anneau cartesien : la reproduction QUANTITATIVE du taux
  diocotron de Hoffart restera bornee par ce verrou (`docs/PAPER_ROADMAP.md`) meme avec un etage
  source condense parfait ;
- le modele deux-fluides MAGNETISE complet (couplage E x B + diamagnetique inhomogene au transport,
  au-dela de la limite de derive) est encore au-dela de cet etage source (`docs/PAPER_ROADMAP.md`
  panier 4) ;
- l'horizon est LONG : PR1-PR4 sont l'ossature credible a moyen terme ; PR5-PR8 dependent de
  validations materielles (GH200) et de chantiers transverses (parite AMR) qui ne sont pas
  garantis sur l'horizon du stage.

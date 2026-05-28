# adc_cpp

Solveur advection-diffusion-couplage : systeme hyperbolique-elliptique couple,
concu des le depart pour l'AMR dynamique, OpenMP, MPI et Kokkos, cible cluster.

Forme generale resolue :

```
d U / d t + div F(U, phi) = div H(U, grad U) + S(U, phi)
D phi = f(U)
```

Cas H = 0 pour l'instant. Cible de validation : l'instabilite diocotron
(transport E x B d'une densite electronique couple a Poisson).

## Niveau d'abstraction

Trois axes orthogonaux qui ne se melangent jamais :

| Axe | Question | Qui le porte |
|---|---|---|
| Quoi calculer | la physique | `PhysicalModel` : fonctions pures sur des etats ponctuels |
| Ou / comment iterer | le parallelisme | maillage + dispatch (`parallel_for`) |
| Dans quel ordre | le temps | integrateur + coupleur |

Invariant central : **AMR, MPI et Kokkos sont des proprietes de la couche
donnees/maillage. Ils n'apparaissent jamais dans la couche physique.** Un
`PhysicalModel` ne voit jamais une box, un rang MPI, un ghost ni une vue
Kokkos. On lui passe des etats ponctuels, il rend des flux/sources ponctuels.
C'est ce qui le rend portable CPU et device : il ne touche a aucun parallelisme.

## Couches

Fige (le socle, ecrit une fois) :
- maillage : abstraction box/patch, hierarchie de niveaux
- tableau distribue (equivalent `MultiFab`) : memoire, layout, rangs MPI, ghosts
- dispatch `parallel_for` (Kokkos) sur les boxes
- echange de halos
- machinerie AMR : regridding (tagging, clustering Berger-Rigoutsos, equilibrage
  de charge), prolongation/restriction, registres de reflux
- squelette d'avancement temporel (sous-cyclage en temps entre niveaux)

Generalise (interchangeable, exprime en concept/template) :
- `PhysicalModel` : formules ponctuelles
- reconstruction (MUSCL/WENO)
- solveur de Riemann
- backend elliptique
- integrateur temporel (SSPRK2/3, IMEX)
- strategie de couplage (split, par etage, monolithique)

## Operateur spatial

L'objet qui assemble le residu spatial complet :

```
R(U, aux) = -div F(U, aux) + S(U, aux)
```

Par box : echange de halos, reconstruction, flux numerique (Riemann aux faces),
divergence, source, et sur AMR enregistrement des flux coarse-fine pour le
reflux. Il consomme `aux` (phi et grad phi produits par la resolution
elliptique), il ne possede ni l'integration temporelle ni l'elliptique.
C'est la fleche "PDE -> systeme d'ODE" de la methode des lignes : l'integrateur
reste agnostique du modele et de la discretisation.

`flux` et `source` du modele prennent tous deux `aux`. C'est le point qui unifie
diocotron (le potentiel entre par le flux) et Euler-Poisson (il entre par la
source) sous un seul operateur, sans branchement par modele.

Boucle par pas :

```
advance(U, dt):
    fill_ghosts(U)                    # maillage (MPI/Kokkos)
    b   = model.elliptic_rhs(U)       # modele (ponctuel)
    phi = elliptic.solve(b)           # backend elliptique (decomposition propre)
    aux = derive(phi)                 # phi, grad phi
    R   = spatial_op.assemble(U, aux) # reconstruction + flux + div + source + reflux
    U   = integrator.update(U, R, dt) # integrateur (agnostique)
```

## Solveur elliptique

Maillage cartesien structure + Poisson a coefficient constant (ou epsilon lisse)
appelle une multigrille geometrique (FAC / MLMG), pas une FFT (incompatible AMR
et difficile a distribuer) ni une multigrille algebrique (utile seulement sur
maillage non structure ou coefficients durs). Interface abstraite `EllipticSolver`
pour brancher PETSc/hypre BoomerAMG comme oracle de verification et repli sur
coefficients durs.

## Decisions

- socle maillage/AMR : mini-AMReX maison (block-structured AMR ecrit a la main,
  inspire d'AMReX / Parthenon / AthenaK)
- dispatch / donnees : seam maison (`Box2D` / `Fab2D` / `for_each_cell`) calque
  sur la semantique Kokkos (handle `Array4` capture par valeur, fonctor sur
  indices). Backend OpenMP maintenant, backend Kokkos branchable au passage
  cluster sans toucher la logique AMR. La physique reste agnostique du backend.
- elliptique : multigrille geometrique maison (FAC sur la hierarchie), interface
  abstraite pour brancher PETSc/hypre en oracle de verification
- couplage : a fixer (splitting pour diocotron non raide, IMEX/well-balanced
  pour le regime quasi-neutre, longueur de Debye -> 0)

## Plan mini-AMReX

1. index space : `Box2D` (fait)
2. donnees mono-grille : `Fab2D` + `Array4` + `for_each_cell` (fait)
3. decomposition : `BoxArray` + `DistributionMapping` (rang unique, interface MPI)
4. conteneur multi-grille : `MultiFab` (collection de `Fab2D` + ghosts)
5. echange de halos : `fill_boundary` (intra-niveau, MPI ensuite)
6. CL physiques : periodique / Dirichlet / Neumann au bord du domaine
7. hierarchie AMR : niveaux, regrid (tagging, Berger-Rigoutsos, proper nesting),
   prolongation / restriction
8. reflux : `FluxRegister` coarse-fine
9. operateur spatial : reconstruction + Riemann + divergence
10. integrateur temporel : SSPRK2/3, sous-cyclage
11. multigrille geometrique maison + coupleur

## Etat

Couche physique : concept `PhysicalModel`, types ponctuels `StateVec` / `Aux`,
premier modele conforme (`Diocotron`).

Couche donnees/maillage : index space `Box2D`, donnees mono-grille `Fab2D`
(layout composante-lente, ghosts) avec handle `Array4` capturable par valeur, et
dispatch `for_each_cell` (backend OpenMP, miroir de Kokkos `parallel_for`).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Pre-requis : un compilateur C++23 (AppleClang/Homebrew clang 17+, GCC 13+),
CMake 3.20+.

# Choix de conception

Les décisions d'architecture d'`adc_cpp`, avec leur contexte et leur coût. Complète
[ARCHITECTURE.md](ARCHITECTURE.md) (qui décrit l'état) en disant *pourquoi*.

---

## D-1. Pile AMR écrite *from scratch*, pas sur `pde_core_cpp`

**Contexte.** `euler_cpp` et `advection_cpp` dépendent de `pde_core_cpp` (primitives AMR
partagées via FetchContent). Une option était de faire pareil pour `adc_cpp`.

**Décision.** Pile MultiFab + BoxArray + DistributionMapping + seam écrite entièrement dans
`include/adc/`, indépendante.

**Pourquoi.** Le couplage hyperbolique-elliptique sur AMR distribué (diocotron, deux-fluides
plasma) demande une couche mesh proche d'AMReX (MultiFab distribué, FluxRegister, FillPatch)
que `pde_core_cpp` n'avait pas. La construire from scratch fige une pile AMR maîtrisée de
bout en bout, but pédagogique du stage.

**Coût.** Duplication conceptuelle avec AMReX ; pas d'optimisations matures (MFIter, EB
courbe). Assumé : on reste lisible et portable plutôt qu'optimal.

---

## D-2. Trois axes orthogonaux + concepts

**Décision.** `PhysicalModel` (concept), `NumericalFlux` (policy), `EllipticSolver`
(concept), couplage. Un solveur = un point dans ce produit.

**Pourquoi.** Ajouter un flux (HLLC), un elliptique (FFT), ou un modèle (deux-fluides) sans
toucher le reste. `compute_face_fluxes<Limiter, NumericalFlux, Model>` est le point de
jonction. Inspiré du design de PLUTO (voir BIBLIOGRAPHY).

---

## D-3. Le seam `for_each_cell` (dispatch unique)

**Décision.** Une seule primitive de boucle `for_each_cell(box, lambda ADC_HD)` dispatche
vers série / `#pragma omp` / `Kokkos::parallel_for`. `Array4` POD device-callable,
`device_fence()`, `comm.hpp` pour MPI.

**Pourquoi.** La physique est écrite une fois et tourne partout. Le backend est une
**propriété de la cible `adc`** (target_compile_definitions INTERFACE), pas un drapeau par
solveur : changer de backend = reconfigurer CMake, rien dans le code.

**Coût.** Discipline de fences GPU (toute fonction kernel-device puis boucle-hôte sur la
même mémoire unifiée doit `device_fence()` entre les deux). Le bug le plus subtil rencontré.

---

## D-4. MultiFab / BoxArray / DistributionMapping façon AMReX

**Décision.** BoxArray **global** (tous les rangs connaissent toutes les boîtes),
DistributionMapping (propriétaire par boîte, équilibrage SFC), MultiFab n'allouant que les
fabs locaux.

**Pourquoi.** C'est le modèle qui rend l'AMR distribué possible et la couverture multi-patch
correcte sous n'importe quelle distribution (la couverture se calcule du BoxArray global).

---

## D-5. `EllipticSolver` concept : multigrille ET FFT

**Décision.** `GeometricMG` (itératif, warm-start, on-device) et `PoissonFFTSolver`
(direct, périodique) modèlent le même concept ; le coupleur est générique dessus.

**Pourquoi.** Le bon solveur dépend de la charge (mesuré : FFT gagne ~4.8x sur Euler-Poisson
Poisson-dominé, MG gagne ~2.4x sur le deux-fluides transport-dominé). Le concept laisse
choisir sans réécrire le coupleur.

---

## D-6. Façade compilée `src/` (PIMPL) + bindings

**Décision.** `src/*_solver.cpp` expose des solveurs CONCRETS sans template
(`DiocotronSolver`, ...) ; `python/bindings.cpp` les bind 1:1.

**Pourquoi.** Une surface stable et bindable, jamais `Coupler<Model, Elliptic>` au-dehors.
Le cœur générique reste header-only ; la façade donne une API Python propre et une frontière
de compilation.

---

## D-7. Pile Fab2D de référence gardée comme oracle de test

**Décision.** L'ancienne pile mono-box `Fab2D` (`amr_reflux.hpp`, `amr_multilevel.hpp`)
n'est plus en production mais reste compilée et testée.

**Pourquoi.** Elle sert d'**oracle** : chaque brique MultiFab est prouvée bit-identique à
elle (`test_amr_*_mf`). Refonte sûre par équivalence plutôt que par confiance.

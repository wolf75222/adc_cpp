# Algorithmes

Catalogue des methodes numeriques GENERIQUES implementees dans le coeur `adc_cpp`, avec leur
formule, leur discretisation et le test du depot qui les exerce. Le coeur est AGNOSTIQUE au
modele : il ne nomme aucun scenario (diocotron, Euler-Poisson, deux-fluides...). Ces scenarios
sont des COMPOSITIONS de briques generiques et vivent cote application
([`adc_cases`](https://github.com/wolf75222/adc_cases)) ; leurs validations bout-en-bout aussi.

Chaque section suit le meme plan : **Intuition** (a quoi ca sert), **Formule** (d'ou ca vient),
**Code** (ou c'est, comment c'est appele), **Validation** (le test ctest qui le verifie).
Tous les chemins de fichiers et noms de tests cites existent dans ce depot.

Architecture (couches, seam de dispatch, frontiere lib/application) :
[ARCHITECTURE.md](ARCHITECTURE.md). Choix de conception : [CHOICES.md](CHOICES.md). Trace des
validations device GH200 : [GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md). References :
[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md).

Le coeur resout, sur maillage cartesien adaptatif, la forme generique

```
d U / d t  +  div F(U, aux)  =  S(U, aux)        (hyperbolique, par bloc)
div(eps grad phi) - kappa phi = f(U)             (elliptique, partage)
```

ou la partie hyperbolique `U` et la partie elliptique `phi` sont couplees a chaque pas par le
canal `aux` (contrat de base `(phi, grad_x, grad_y)`, extensible). Un modele est une
composition `CompositeModel<Transport, Source, Elliptic>` ; le couplage entre par le FLUX (aux
lu dans `F`) ou par la SOURCE (aux lu dans `S`), sous le meme operateur spatial.

## 1. Volumes finis : Godunov ordre 1

**Intuition.** Le profil dans chaque maille est remplace par sa moyenne. A chaque interface,
deux moyennes se rencontrent : un probleme de Riemann local. Le flux numerique le resout
(approximativement) et la mise a jour conservative transporte la matiere d'une maille a l'autre.

**Formule.** Integration de la loi de conservation sur la maille `(i,j)` et le pas `dt` :

$$U_{ij}^{n+1} = U_{ij}^n - \frac{\Delta t}{\Delta x}\big(\hat F_{i+1/2,j} - \hat F_{i-1/2,j}\big)
                             - \frac{\Delta t}{\Delta y}\big(\hat G_{i,j+1/2} - \hat G_{i,j-1/2}\big)$$

La forme conservative (difference de flux de face) garantit la conservation discrete exacte : ce
que la maille `i` perd a sa face droite, la maille `i+1` le gagne a sa face gauche. C'est la
propriete dont depend tout l'AMR (le reflux corrige justement ces flux de face).

**Code.** `numerics/spatial_operator.hpp::assemble_rhs` calcule directement `R = -div F + S`.
`compute_face_fluxes` ecrit les flux de FACE `Fx, Fy` (ce dont le reflux a besoin), avec la MEME
reconstruction et le MEME flux numerique que `assemble_rhs`, donc leur divergence redonne
exactement le residu. La boucle passe par le seam `for_each_cell` (serie / OpenMP / Kokkos).

**Validation.** `test_spatial_discretisation` (le couple reconstruction x flux est un type nomme),
`test_cfl_dt` (`dt = cfl * min(dx,dy) / max|lambda|` multi-especes). **Stabilite.** CFL :
`dt <= C dx / max|lambda|`, `lambda` = vitesse d'onde locale.

## 2. Flux numeriques : Rusanov, HLL, HLLC, Roe

**Intuition.** Quatre niveaux de fidelite pour le Riemann de face. Rusanov pose une seule bosse
de diffusion (le plus robuste, le plus diffusif) ; HLL estime deux vitesses acoustiques ; HLLC
ajoute l'onde de contact (la discontinuite de densite passive) ; Roe linearise le systeme par la
moyenne de Roe et resout exactement le Riemann linearise.

**Formules.** Rusanov (Lax-Friedrichs local) :

$$\hat F_{i+1/2} = \tfrac12\big(F(U_L)+F(U_R)\big) - \tfrac12\,a\,(U_R - U_L),
\qquad a = \max(|\lambda_L|, |\lambda_R|)$$

HLL avec estimations de vitesses `S_L, S_R` :

$$\hat F^{HLL} = \frac{S_R F(U_L) - S_L F(U_R) + S_L S_R (U_R - U_L)}{S_R - S_L}$$

HLLC restaure l'onde de contact `S_*` au milieu (etats etoile `U_L^*, U_R^*`). Roe :
`F = (F_L + F_R)/2 - (1/2) sum_k |lambda_k| alpha_k r_k` sur les vecteurs propres de la matrice
de Roe.

**Code.** Politiques SANS etat dans `numerics/numerical_flux.hpp` : `RusanovFlux`, `HLLFlux`,
`HLLCFlux`, `RoeFlux` (toutes `ADC_HD`, device-callable). `compute_face_fluxes<Limiter,
NumericalFlux, Model>` est temple sur le flux.

**Validation.** `test_roe_flux` : consistance `F*(U,U) = F(U)` et resolution exacte d'un Riemann
linearise + `eigenvalues()` d'Euler. **Pieges.** HLLC sur un etat du vide (densite nulle)
demande un garde-fou ; Rusanov reste le defaut robuste pour le transport scalaire.

## 3. Reconstruction MUSCL (ordre 2) et WENO5-Z (ordre 5)

**Intuition.** Godunov ordre 1 est tres diffusif. MUSCL reconstruit un profil LINEAIRE par
maille a partir de pentes limitees, puis evalue le flux sur les valeurs reconstruites aux faces.
Le limiteur coupe la pente pres des extrema pour eviter les oscillations (TVD). WENO5-Z monte a
l'ordre 5 en zone lisse via une moyenne ponderee de trois stencils, sans limiteur explicite.

**Formule.** Pente limitee par maille, par exemple minmod :

$$\sigma_i = \mathrm{minmod}(U_i - U_{i-1},\ U_{i+1} - U_i)$$

Etats reconstruits aux faces : `U_L = U_i + sigma_i/2`, `U_R = U_{i+1} - sigma_{i+1}/2`. MUSCL
demande 2 ghosts (pente en `i+-1`), WENO5 en demande 3. Les poids WENO-Z (Borges 2008) utilisent
`tau5 = |beta0 - beta2|` pour rester moins dissipatifs que Jiang-Shu en zone lisse.

**Code.** Politiques `Limiter` dans `numerics/reconstruction.hpp` : `NoSlope` (ordre 1), `Minmod`,
`VanLeer`, `Weno5` (WENO5-Z). La reconstruction peut se faire sur les variables CONSERVEES ou
PRIMITIVES (`rho, u, p`) selon le bloc.

**Validation.** `test_weno_convergence` (la reconstruction de face d'une fonction lisse est ordre 5),
`test_primitive_recon` (conversions cons <-> prim et leur usage dans la reconstruction).
**Pieges.** Reconstruire la variable conservee vs primitive change le comportement aux chocs forts.

## 4. Integration en temps : SSPRK, integrateurs objets, integrateur utilisateur

**Intuition.** Strong-Stability-Preserving Runge-Kutta : des combinaisons convexes d'Euler
explicites, donc la propriete TVD de l'operateur spatial est preservee a l'ordre 2 ou 3.

**Formule.** SSPRK2 (Heun) :

$$U^{(1)} = U^n + \Delta t\,L(U^n),\qquad U^{n+1} = \tfrac12 U^n + \tfrac12\big(U^{(1)} + \Delta t\,L(U^{(1)})\big)$$

SSPRK3 (Shu-Osher) ajoute un etage, ordre 3, meme coefficient SSP `C=1`. `L(U)` = `assemble_rhs`.

**Code.** Deux expressions coexistent. Les TAGS `SSPRK2`/`SSPRK3` (`numerics/time/time_integrator.hpp`)
nomment, par bloc, le traitement temporel via une `TimePolicy` (explicite / implicite / IMEX /
prescrit). Les integrateurs OBJETS (`numerics/time/time_steppers.hpp` : `ForwardEuler`,
`SSPRK2Step`, `SSPRK3Step`) exposent `take_step(rhs, U, dt)` ; l'utilisateur peut fournir le sien
(tout objet a `take_step`).

**Validation.** `test_user_time_integrator` (un integrateur en temps fourni par l'utilisateur
donne le meme resultat qu'un SSPRK du coeur). **Pieges.** En couple, l'ordre du solve elliptique
limite l'ordre global : Poisson resolu 1x/pas plafonne le champ a l'ordre 1.

## 5. Sources raides : IMEX asymptotic-preserving et IMEX partiel

**Intuition.** Une source RAIDE (relaxation rapide, frequence plasma) impose au schema explicite
un `dt` minuscule. L'IMEX traite la partie raide IMPLICITEMENT (stable a grand `dt`) et la partie
non raide explicitement. La propriete asymptotic-preserving (AP) garantit que le schema reste
consistant et stable quand la raideur tend vers l'infini.

**Formule.** Un pas IMEX d'Euler sur `dU/dt = T(U) + (1/eps) R(U)` traite `T` explicite et
`(1/eps) R` implicite. Quand `eps -> 0`, le schema relaxe vers la variete d'equilibre `R(U)=0`
sans contrainte `dt < eps`.

**Code.** `numerics/time/imex.hpp::imex_euler_step` (IMEX d'Euler), `numerics/time/implicit_stepper.hpp`
(defaut implicite par Newton local). L'IMEX PARTIEL n'integre implicitement qu'une SOUS-PARTIE des
variables : le modele declare `is_implicit(c)` par composante (moins cher quand seule une variable
est raide). Le transport d'un bloc IMEX reste explicite.

**Validation.** `test_imex_ap` (propriete AP sur une source de relaxation lineaire raide),
`test_ap_limit` (limite AP QUANTIFIEE : balayage de la raideur sur 8 decades a `dt` fixe),
`test_imex_partial` (un modele a 2 variables, une seule implicite), `test_imex_transport` (le
transport d'un bloc IMEX est bien avance explicitement).

## 6. Splitting d'operateurs : Lie et Strang

**Intuition.** Quand le RHS est une somme d'operateurs (transport + source + rotation cyclotron),
on les applique en SEQUENCE plutot que simultanement. Lie (ordre 1) les enchaine, Strang (ordre 2)
symetrise la sequence (`R(dt/2) . pas-central . R(dt/2)`).

**Code.** `numerics/time/splitting.hpp::lie_step` / `strang_step`.

**Validation.** `test_splitting` : ordre du splitting mesure sur un systeme lineaire 2x2 NON
commutant dont le flot exact est connu (Lie ordre 1, Strang ordre 2 verifies par la pente).

## 7. Multirate : sous-cyclage, cadence, pas adaptatif

**Intuition.** Toutes les especes ne demandent pas le meme `dt`. Une espece raide (electrons) fait
plusieurs SOUS-PAS pour un macro-pas ; une espece lente (gaz peu resolu) n'est avancee qu'une fois
sur N macro-pas (CADENCE / stride). Le pas macro peut etre derive du CFL par espece.

**Code.** `numerics/time/scheduler.hpp::advance_subcycled` lit la `TimePolicy` de chaque
`EquationBlock` (`ExplicitTime<Method, substeps, stride>`) et appelle l'operateur adapte. Le
`SystemDriver` expose `step_cfl(cfl)` (pas macro par CFL) et `step_adaptive(cfl)` (stride derive du
CFL par espece). Une espece `PrescribedTime` est sautee par le scheduler.

**Validation.** `test_multirate_stride` (une espece lente avancee une fois sur N),
`test_adaptive_multirate` (`step_adaptive`, pas macro fixe par l'espece la plus contraignante),
`test_cfl_dt` (`step_cfl` multi-especes).

## 8. Terme parabolique : diffusion en flux de face

**Intuition.** Un terme `+nu Lap(U)` (diffusion, viscosite) s'ecrit comme la divergence d'un flux
de face Fickien `-nu grad U`. Le mettre en flux de face le rend AMR-compatible : le reflux le
corrige a l'interface fin-grossier comme un flux hyperbolique.

**Formule.** `F_diff = -nu grad U`, ajoute aux flux de face avant la divergence ; `div(F_diff) =
-nu Lap(U)`, donc `R += +nu Lap(U)`.

**Code.** Un modele qui declare `diffusivity()` recoit le flux Fickien dans `assemble_rhs` ;
`compute_face_fluxes` l'inclut pour le reflux.

**Validation.** `test_diffusion` (le terme parabolique du coeur, `+nu Lap(U)` via la divergence du
flux Fickien), `test_amr_diffusion` (la diffusion en flux de face traverse correctement l'AMR).

## 9. Elliptique : multigrille geometrique

**Intuition.** Le lisseur Gauss-Seidel tue vite les hautes frequences de l'erreur mais rampe sur
les basses. La multigrille restreint l'erreur basse frequence sur des grilles plus grossieres (ou
elle redevient haute frequence), la lisse, et la prolonge. Cout O(N).

**V-cycle.** lisser (`nu1` balayages GS rouge-noir) -> residu -> restreindre -> recursion grossiere
-> prolonger la correction -> lisser (`nu2`). Le rouge-noir rend le balayage independant des
donnees (parallelisable).

**Code.** `numerics/elliptic/geometric_mg.hpp::GeometricMG` modele le concept `EllipticSolver`
(`rhs()`, `solve()`, `residual()`, `phi()`). Entierement on-device (le V-cycle passe par
`for_each_cell`), AMR-compatible, accepte tout `n`. Le Laplacien 5 points et le lisseur sont
dans `numerics/elliptic/poisson_operator.hpp` (operateur canonique partage). `solve_robust`
ajoute un garde-fou anti-divergence ; le warm-start repart du `phi` precedent (1-2 V-cycles en
regime etabli).

**Validation.** `test_geometric_mg` (convergence rapide quasi independante du maillage sur
solutions manufacturees), `test_poisson_convergence` (ordre 2 quantitatif du Laplacien 5 points),
`test_solve_robust` (le garde-fou anti-divergence).

## 10. Elliptique : Poisson spectral (FFT), mono-rang et distribue

**Intuition.** Sur un domaine periodique, le Laplacien est diagonal en Fourier : une transformee
directe + division + transformee inverse resout Poisson EXACTEMENT (au residu machine), sans
iteration.

**Formule.** `phi_hat(k) = -rhs_hat(k) / (k_x^2 + k_y^2)`, mode `k=0` fixe a 0 (jauge).

**Code.** `numerics/elliptic/poisson_fft_solver.hpp` : `PoissonFFTSolver` (mono-rang, boite unique,
assert `n_ranks()==1 && ba.size()==1`) et `DistributedFFTSolver` (FFT distribuee par BANDES via
`MPI_Alltoall`). Les deux modelent le meme concept `EllipticSolver` que la multigrille, donc le
coupleur est generique sur le backend. Un correctif gere `n` non puissance de 2.

**Validation.** `test_poisson_fft` (non-regression, taille `n` non puissance de 2),
`test_elliptic_operator` : le MEME operateur canonique `poisson_residual` applique aux solutions
MG et FFT donne des residus a l'arrondi (`~1e-14`) et des solutions identiques a `~1e-16` -> les
deux inversent prouvablement le meme Laplacien discret. **Pieges.** Le FFT exige periodique ; le
mode `k=0` doit etre fixe (second membre a moyenne nulle), sinon `phi` derive.

## 11. Elliptique etendu : eps(x), Helmholtz/ecrante, anisotrope

**Intuition.** Le meme operateur multigrille couvre trois generalisations du Laplacien, toutes
opt-in et bit-identiques au chemin historique quand on ne les active pas :

- **permittivite variable** `div(eps(x) grad phi) = f` : chaque face est la moyenne HARMONIQUE des
  deux centres adjacents du champ `eps` ;
- **operateur ecrante / Helmholtz** `div(eps grad phi) - kappa phi = f` : un terme de reaction
  `kappa >= 0` (ecrantage de Debye `kappa = 1 / lambda_D^2`), qui rend l'operateur plus
  diagonalement dominant ;
- **permittivite anisotrope** `div(diag(eps_x, eps_y) grad phi) = f` : les faces normales a x
  lisent `eps_x`, les faces normales a y lisent `eps_y` (milieu tensoriel diagonal).

**Code.** `GeometricMG::set_epsilon(eps_fn | eps_fine)`, `set_reaction(kappa_fn)`,
`set_epsilon_anisotropic(eps_x, eps_y)` (`numerics/elliptic/geometric_mg.hpp`). Le terme `kappa`,
les champs `eps`/`eps_y` vivent dans les `for_each_cell` ADC_HD du smoother, du residu et de
l'apply (`poisson_operator.hpp`) -> device. `eps` est echantillonne PAR NIVEAU (permittivite
exacte au grossier, ordre 2 preserve). Les trois sont composables. Donner `eps_x == eps_y` redonne
l'isotrope ; ne pas appeler `set_reaction` redonne Poisson pur.

**Validation.** `test_variable_epsilon` (eps(x), MMS ordre 2), `test_screened_poisson`
(Helmholtz/ecrante, MMS ordre 2), `test_anisotropic_epsilon` (anisotrope `eps_x != eps_y`, MMS
ordre 2). Ces trois chemins sont aussi exerces cote Python (`test_poisson_eps`,
`test_poisson_screened`, `test_poisson_eps_aniso`) et valides bit-identiques sur GH200
(cf. GPU_RUNTIME_PORT.md, round 2).

## 12. Bord embedded : cut-cell Shortley-Weller

**Intuition.** Une paroi non alignee sur la grille (conducteur circulaire) n'est pas un escalier :
la cut-cell Shortley-Weller corrige le stencil au bord pour placer la condition de Dirichlet a la
position REELLE de l'interface, retrouvant l'ordre 2 la ou l'escalier tombe a l'ordre 1.

**Code.** Le masque + les coefficients cut-cell sont calcules au SETUP (hote, one-shot) puis lus
par le V-cycle on-device. Compatible avec l'operateur anisotrope.

**Validation.** `test_cut_cell` (cut-cell vs escalier sur solution manufacturee),
`test_cut_cell_anisotropic` (cut-cell + operateur anisotrope), `test_cut_cell_anisotropic_multibox`
(domaine multi-box mono-rang), `test_mpi_cutcell_multibox` (multi-box distribue np=1/2/4, verrou de
non-regression du bug average_down hors bornes sur hierarchie MG degenere).

## 13. AMR : sous-cyclage Berger-Oliger + reflux conservatif

**Intuition.** Raffiner seulement la ou il le faut. Un niveau fin (pas `dx/2`) recouvre une
sous-region ; pour respecter sa CFL il fait `r` sous-pas de `dt/r` pendant que le grossier fait 1
pas de `dt`. A l'interface fin-grossier, les deux niveaux calculent des flux differents -> la
conservation est cassee. Le REFLUX (FluxRegister) corrige la maille grossiere par la difference
(flux fin integre - flux grossier).

**Formule.** A chaque face fin-grossier, on accumule le flux fin sur les `r` sous-pas et on
remplace la contribution grossiere :

$$U_c \mathrel{-}= \frac{1}{\Delta x_c}\Big(\textstyle\sum_s \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)$$

**Code.** `numerics/time/amr_reflux_mf.hpp::advance_amr` est le moteur de production (multi-patch,
N niveaux, distribue MPI) ; la pile mono-box `amr_*_mf` y vit en `detail::` comme ORACLE de
validation. Transferts : `average_down` (fin -> grossier, conservatif) et `interpolate` (injection
grossier -> fin) dans `mesh/refinement.hpp`. Roles promus en types : `FluxRegister` accumule les
flux de face, `CoverageMask` evite les doubles corrections.

**Validation.** `test_refinement` (average_down conservatif + interpolate), `test_amr_hierarchy`
(construction niveau grossier + fin imbrique + interpolation ghosts), `test_flux_register`
(indexation du registre de flux), `test_coverage_mask` (marquage des cellules couvertes),
`test_amr_diagnostics` (masse / vitesse de derive via le seam reducteur). **Pieges.** Le flux
grossier doit etre echantillonne AVANT d'avancer le grossier ; l'average_down doit preceder la
mesure de masse initiale.

## 14. AMR multi-patch : reflux coverage-aware, distribue MPI

**Intuition.** Un niveau fin n'est pas une seule boite mais un ENSEMBLE de patchs. Deux subtilites :
(a) au joint entre deux patchs voisins (interface fin-fin), il ne faut PAS refluxer ; (b) la
correction doit aller dans la BONNE boite parente quand le grossier est lui-meme multi-box.

**Code.** Le masque de couverture est bati sur le BoxArray GLOBAL (toutes les boites, connues de
tous les rangs), donc correct sous n'importe quelle distribution MPI. Sous MPI le grossier est
soit REPLIQUE (defaut, `replicated_coarse=true`), soit DE-REPLIQUE (multi-box reparti). Le reflux
distant passe par `parallel_copy` (copie inter-niveaux) + gather du registre par
`all_reduce_sum_inplace`.

**Validation.** `test_amr_spatial_parity` (le coeur spatial du chemin AMR == celui de `System` :
meme reconstruction primitive, meme flux HLLC/Roe), `test_mpi_mbox_parity` (le residu du chemin
compile a foncteurs nommes est invariant au decoupage en boites ET au nombre de rangs np=1/2/4,
`dmax=0`), `test_mpi_amr_distributed_coarse` (grossier reparti == replique bit a bit, np=1/2/4).
**Pieges.** Sans masque de couverture, le joint fin-fin serait reflue deux fois -> non-conservation.

## 15. Clustering Berger-Rigoutsos et regrid

**Intuition.** Etant donne les cellules marquees (fort gradient), trouver un petit nombre de boites
rectangulaires qui les couvrent sans trop de gaspillage. L'algorithme coupe recursivement une
region la ou la signature (histogramme des marques projete) presente un trou ou une inflexion (le
laplacien de la signature).

**Code.** `amr/cluster.hpp::berger_rigoutsos` (signature par axe -> trou ou inflexion -> coupe ->
recursion jusqu'a l'efficacite de remplissage cible `ClusterParams`). `amr/regrid.hpp` produit et
dilate les marques, impose le proper nesting, interpole les nouveaux patchs depuis le grossier.
Sous MPI le regrid est rendu correct pour un grossier reparti (OU global des tags par
`all_reduce_or_inplace` avant le clustering, sinon la BoxArray fine differerait par rang).

**Validation.** `test_cluster` (bloc plein -> 1 boite, deux blocs separes -> 2 boites, gros bloc
decoupe par `max_box_size`), `test_regrid` (un niveau fin est cree autour de la region taguee, les
donnees fines interpolees depuis le grossier). **Pieges.** Le proper nesting (chaque patch fin
strictement interieur a la couverture parente) doit etre impose apres le clustering, sinon le
ghost-fill inter-niveaux echoue.

## 16. Maillage distribue : BoxArray global, halos, equilibrage

**Intuition.** Le AMR distribue exige que tous les rangs connaissent toutes les boites (BoxArray
GLOBAL) pour calculer la couverture multi-patch correctement, mais que chaque rang n'alloue que
ses fabs locaux (via la `DistributionMapping`). L'echange de halos comble les ghosts paralleles ;
l'equilibrage repartit les boites sur les rangs.

**Code.** `mesh/box_array.hpp` (BoxArray global), `mesh/distribution_mapping.hpp` (box -> rang),
`mesh/multifab.hpp` (n'alloue que les fabs locaux), `mesh/fill_boundary.hpp` (echange de halos
intra-niveau, begin/end non-bloquant), `parallel/load_balance.hpp` (Z-order SFC + knapsack).
`parallel/comm.hpp` enveloppe les collectives MPI (`all_reduce_sum`, `all_reduce_or_inplace`,
send-recv), degenere en serie.

**Validation.** `test_box_array`, `test_multifab`, `test_load_balance` ; sous MPI :
`test_mpi_fillboundary` (echange de halos), `test_mpi_poisson` (Poisson distribue),
`test_mpi_fft_distributed` (FFT par bandes), `test_mpi_redistribute`, `test_mpi_array_reduce`,
`test_mpi_coupler_inject` (np=4, bit-identiques np=1/2/4).

## 17. Canal aux extensible

**Intuition.** Le couplage hyperbolique-elliptique passe par un canal `aux` de contrat de BASE
`(phi, grad_x, grad_y)` (3 composantes). Certains modeles ont besoin de champs SUPPLEMENTAIRES : un
champ magnetique `B_z` hors-plan, une temperature electronique `T_e` derivee de `p/rho`. Le canal
est elargi a la demande, bit-identique a l'historique quand `n_aux = 3`.

**Code.** Un modele declare un membre statique `n_aux > 3` ; `load_aux<NComp>` / `aux_comps<Model>()`
lisent les composantes supplementaires. `B_z` (comp 3) est une fonction PURE de la position,
echantillonnee par niveau ; `T_e` (comp 4) est derivee d'un bloc fluide. Un `CompositeModel`
propage la largeur aux maximale de ses briques.

**Validation.** `test_aux_extra` (un modele declare `n_aux > 3`), `test_aux_composite` (un modele
compose propage la largeur aux de ses briques), `test_aux_coupler_bz` / `test_aux_system_bz` /
`test_amr_aux_bz` / `test_amr_system_bz_pop` / `test_amr_system_bz_multibox` (B_z lu et peuple le
long des chemins coupleur, systeme, AMR, multi-box), `test_aux_te` (T_e derive p/rho),
`test_aux_single_source` (une source unique genere `load_aux` + marshaling, tous champs couverts).

## 18. Composition runtime et systeme multi-especes

**Intuition.** Python compose QUOI (un bloc par espece : modele compose + schema spatial + politique
temporelle), le C++ calcule par CELLULE. N especes interagissent dans le SECOND MEMBRE elliptique
(`f = sum_s q_s n_s`) et dans la SOURCE inter-especes, jamais dans le flux.

**Code.** `runtime/system.hpp::System` (multi-blocs mono-niveau, Poisson partage),
`runtime/amr_system.hpp::AmrSystem` (un bloc sur hierarchie raffinee). Cote couplage :
`coupling/system_coupler.hpp` (`SystemAssembler` assemble, `SystemDriver` avance),
`coupling/amr_system_coupler.hpp` (le systeme porte sur AMR). `runtime/model_factory.hpp` assemble
un `CompositeModel` depuis une spec de briques (le coeur ne nomme aucun scenario).

**Validation.** `test_system_abstraction`, `test_system_coupler`, `test_two_species_minimal`,
`test_coupled_source` (source inter-especes), `test_system_two_explicit`, `test_assembler_driver`
(l'assembleur assemble, le driver avance), `test_amr_system_coupler`, `test_system_hardening`,
`test_variable_role` (adresser une composante par son ROLE physique plutot que par indice).

## 19. DSL symbolique : codegen, JIT, AOT

**Intuition.** Un mini-DSL cote Python decrit un modele en FORMULES et l'emet en brique C++ : flux,
source, second membre elliptique, avec elimination de sous-expressions communes (CSE). Trois
chemins de mise en oeuvre, du plus dynamique au plus natif.

**Code.**
- **JIT** : `runtime/dynamic_model.hpp` (`IModel` virtuel, charge via `.so` / `dlopen`) ;
  `System.add_dynamic_block` branche un modele a dispatch virtuel (prototypage hote).
- **AOT marshale** : `runtime/compiled_block_abi.hpp` (`add_compiled_block`, ABI `extern "C"`,
  marshaling de tableaux plats, sans AMR ni MPI).
- **AOT natif** : `runtime/dsl_block.hpp` (`add_compiled_model`) branche un `CompositeModel` connu
  a la compilation comme un bloc NATIF, sans marshaling. La machinerie equivalente device-clean est
  `runtime/block_builder.hpp` (fermetures de bloc instanciables hors `System`, foncteurs nommes).

**Validation.** Cote C++ : `test_dynamic_model` (modele type-erased == Euler statique),
`test_block_builder` (fermetures de bloc instanciables hors System), `test_compiled_model_parity`
(AOT natif == bloc natif sur CPU/Serial), `test_amr_compiled_model` (AOT natif sur hierarchie AMR).
Cote Python : `test_dsl*` (codegen flux/source/elliptique, CSE, JIT `.so`, dispatch type-erased,
recon, roles, aux), exerces par la suite `ctest` du module Python. Sur GH200, le chemin a foncteurs
nommes est valide bit-identique (cf. GPU_RUNTIME_PORT.md, phase 9) ; `add_compiled_model` a lambdas
etendues bute encore sur une limite nvcc (cf. GPU_RUNTIME_PORT.md, phase 8).

## 20. Le seam de dispatch (serie / OpenMP / Kokkos / MPI)

Pas un algorithme numerique mais le point de bascule qui les rend tous portables. Detail dans
[ARCHITECTURE.md](ARCHITECTURE.md) section 4 (couche execution). En bref :
`for_each_cell(box, lambda ADC_HD)` dispatche vers une boucle serie, `#pragma omp parallel for`, ou
`Kokkos::parallel_for` (OpenMP / Cuda) selon le backend de la cible `adc`, choisi A LA COMPILATION.
`for_each_cell_reduce_sum` / `_max` portent les reductions (reducteurs `Kokkos::Sum`/`Max`
deterministes). `comm.hpp` enveloppe les collectives MPI. La physique ne voit jamais le backend ; on
n'ecrit AUCUN kernel CUDA a la main. Discipline GPU : `device_fence()` entre un kernel device et une
boucle hote sur la meme memoire unifiee (cf. CHOICES.md).

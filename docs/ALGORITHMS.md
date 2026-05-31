# Algorithmes

Les méthodes numériques implémentées dans `adc_cpp`, avec leurs formules, leur
discrétisation et le test qui les exerce. Chaque section suit le même plan :
**Intuition** (à quoi ça sert), **Dérivation** (d'où vient la formule),
**Pseudocode** (ce que fait le code), **Stabilité** (borne CFL), **Validation**
(quel test), **Pièges** (modes de défaillance connus). Références : Birdsall &
Langdon (PIC, dérive), Toro 2009 (solveurs de Riemann), Berger & Oliger 1984 et
Berger & Colella 1989 (AMR), Berger & Rigoutsos 1991 (clustering), Hoffart
arXiv:2510.11808 (deux-fluides).

Architecture (les couches, le seam de dispatch, le découpage lib/démo) :
[ARCHITECTURE.md](ARCHITECTURE.md). Schéma deux-fluides détaillé :
[two_fluid_ap.md](two_fluid_ap.md). Profil run-time : [PERFORMANCE.md](PERFORMANCE.md).

## 1. Le modèle diocotron (dérive E x B)

**Intuition.** Une couche d'électrons non neutre dans un champ magnétique uniforme
`B = B_z ẑ` hors-plan. Les électrons ne suivent pas le champ électrique `E` qu'ils
créent : ils **dérivent perpendiculairement** à `E` et à `B`, à la vitesse de dérive
`E x B`. Une perturbation de densité crée un champ qui fait tourner la couche, ce qui
amplifie la perturbation : c'est l'instabilité diocotron, l'analogue électrostatique de
Kelvin-Helmholtz.

**Équation.** Densité `n`, potentiel `phi`. Champ `E = -grad phi`, vitesse de dérive

$$\mathbf{v} = \frac{\mathbf{E}\times\mathbf{B}}{B^2} = \frac{1}{B}\,(-\partial_y\phi,\ \partial_x\phi)$$

Transport conservatif de la densité, couplé à Poisson pour le champ :

$$\partial_t n + \nabla\cdot(n\,\mathbf{v}) = 0, \qquad \nabla^2\phi = \alpha\,(n - n_{i0})$$

`n_i0` est le fond ionique neutralisant, `alpha` le couplage. La vitesse dérive d'un
potentiel-flux (`v = ẑ × grad(phi)/B`), donc `div(v) = 0` : le transport est
**incompressible**, la densité est advectée sans compression. C'est ce qui rend
l'instabilité purement cinématique (enroulement), sans choc.

**Couplage.** `aux = (phi, ∂x phi, ∂y phi)` est calculé par le solveur elliptique puis
passé au flux hyperbolique. Le modèle (`include/adc/model/diocotron.hpp`) lit `aux` et
forme le flux `n v` à chaque face.

**Validation.** `examples/diocotron*`, `analysis/diocotron_growth.hpp` (taux de
croissance vs théorie linéaire). **Pièges.** Si Poisson n'est pas résolu à chaque étage
RK, le couplage tombe à l'ordre 1 en temps (voir `poisson_per_stage`).

## 2. Volumes finis : Godunov ordre 1

**Intuition.** On remplace le profil dans chaque maille par sa moyenne. À chaque
interface, deux moyennes se rencontrent : un problème de Riemann local. Le flux numérique
résout (approximativement) ce Riemann et la mise à jour conservative transporte la
matière d'une maille à l'autre.

**Dérivation.** Intégrer la loi de conservation sur la maille `(i,j)` et le pas `dt` :

$$U_{ij}^{n+1} = U_{ij}^n - \frac{\Delta t}{\Delta x}\big(\hat F_{i+1/2,j} - \hat F_{i-1/2,j}\big)
                             - \frac{\Delta t}{\Delta y}\big(\hat G_{i,j+1/2} - \hat G_{i,j-1/2}\big)$$

La forme conservative (différence de flux de face) garantit la conservation discrète
exacte : ce que la maille `i` perd à sa face droite, la maille `i+1` le gagne à sa face
gauche. C'est la propriété centrale dont dépend tout l'AMR (le reflux corrige justement
ces flux de face).

**Pseudocode.** `operator/spatial_operator.hpp::compute_face_fluxes` calcule les flux de
FACE `Fx, Fy` (ce dont le reflux a besoin) ; `mf_advance_faces` applique la différence de
flux via `for_each_cell` (portable série / OpenMP / GPU).

**Stabilité.** CFL : `dt <= C dx / max|lambda|`, `lambda` = vitesse d'onde locale.
**Validation.** `test_face_fluxes` : `div(face_fluxes)` vs `assemble_rhs` à `0` exact.

## 3. Flux numériques : Rusanov, HLL, HLLC

**Intuition.** Trois niveaux de fidélité pour le Riemann de face. Rusanov met une seule
bosse de diffusion (la plus robuste, la plus diffusive) ; HLL estime deux vitesses
acoustiques ; HLLC ajoute l'onde de contact (la discontinuité de densité passive).

**Formules.** Rusanov (Lax-Friedrichs local) :

$$\hat F_{i+1/2} = \tfrac12\big(F(U_L)+F(U_R)\big) - \tfrac12\,a\,(U_R - U_L),
\qquad a = \max(|\lambda_L|, |\lambda_R|)$$

HLL avec estimations de vitesses `S_L, S_R` :

$$\hat F^{HLL} = \frac{S_R F(U_L) - S_L F(U_R) + S_L S_R (U_R - U_L)}{S_R - S_L}$$

HLLC restaure l'onde de contact `S_*` au milieu (états étoile `U_L^*, U_R^*`).

**Pseudocode.** Policies sans état dans `numerics/` ; `compute_face_fluxes<Limiter,
NumericalFlux, Model>` est templé sur le flux. **Validation.** `test_riemann`.
**Pièges.** HLLC sur un état du vide (densité nulle) demande un garde-fou ; Rusanov reste
le défaut robuste pour le diocotron (transport scalaire).

## 4. Reconstruction MUSCL (ordre 2)

**Intuition.** Godunov ordre 1 est très diffusif. MUSCL reconstruit un profil **linéaire**
dans chaque maille à partir de pentes limitées, puis évalue le flux sur les valeurs
reconstruites aux faces. Le **limiteur** coupe la pente près des extrema pour éviter les
oscillations (TVD).

**Dérivation.** Pente limitée par maille, par exemple minmod :

$$\sigma_i = \mathrm{minmod}(U_i - U_{i-1},\ U_{i+1} - U_i),\qquad
\mathrm{minmod}(a,b) = \begin{cases} a & |a|<|b|,\ ab>0\\ b & |b|\le|a|,\ ab>0\\ 0 & ab\le 0\end{cases}$$

États reconstruits aux faces : `U_L = U_i + σ_i/2`, `U_R = U_{i+1} - σ_{i+1}/2`. La
reconstruction demande **2 ghosts** (pente en `i±1`). Limiteurs disponibles : NoSlope
(ordre 1), Minmod, VanLeer.

**Pseudocode.** Policy `Limiter` templée dans `compute_face_fluxes`. **Stabilité.** TVD
sous CFL `<= 0.5` typiquement. **Validation.** `test_two_fluid_ap_amplitude` compare
centré vs reconstruction limitée sur un front raide (la continuité upwind MUSCL ne
sur-diffuse pas le pic lisse, 0.4% de perte). **Pièges.** Reconstruire la variable
conservée vs primitive change le comportement aux chocs forts.

## 5. Intégration en temps : SSPRK2/3

**Intuition.** Strong-Stability-Preserving Runge-Kutta : des combinaisons convexes d'Euler
explicites, donc la propriété TVD de l'opérateur spatial est préservée à l'ordre 2 ou 3.

**Formules.** SSPRK2 (Heun) :

$$U^{(1)} = U^n + \Delta t\,L(U^n),\qquad U^{n+1} = \tfrac12 U^n + \tfrac12\big(U^{(1)} + \Delta t\,L(U^{(1)})\big)$$

SSPRK3 (Shu-Osher) ajoute un étage, ordre 3, même coefficient SSP `C=1`.

**Pseudocode.** `integrator/` ; `L(U)` = `assemble_rhs` (divergence des flux). En couplé,
Poisson est résolu une fois par étage (`poisson_per_stage = true`, ordre 2) ou une fois
par pas (ordre 1, ~2.6x plus rapide, voir PERFORMANCE.md). **Validation.** advection
bout-en-bout (`test` SSPRK2). **Pièges.** L'ordre du couplage limite l'ordre global :
SSPRK3 + Poisson 1x/pas reste ordre 1 sur le champ.

## 6. Elliptique : multigrille géométrique

**Intuition.** Le lisseur Gauss-Seidel tue vite les hautes fréquences de l'erreur mais
rampe sur les basses. La multigrille restreint l'erreur basse fréquence sur des grilles
plus grossières (où elle redevient haute fréquence), la lisse, et la prolonge : chaque
fréquence est traitée à l'échelle où le lisseur est efficace. Coût O(N), pas O(N^1.5).

**V-cycle.** lisser (`nu1` balayages GS red-black) -> résidu -> restreindre -> récursion
sur la grille grossière -> prolonger la correction -> lisser (`nu2`). Le rouge-noir rend
le balayage indépendant des données (parallélisable).

**Pseudocode.** `elliptic/geometric_mg.hpp::GeometricMG` modèle le concept
`EllipticSolver` (`rhs()`, `solve()`, `phi()`). Entièrement on-device (le V-cycle passe
par `for_each_cell`), donc la façade compile pour le GPU. Warm-start : `solve()` repart du
`phi` précédent -> 1-2 V-cycles suffisent en régime établi.

**Validation.** `test_poisson`, `test_mpi_poisson`. **Pièges.** Itératif : le résidu n'est
pas nul à la machine ; pour des CL périodiques exactes, préférer le FFT (section 7).

## 7. Elliptique : Poisson spectral (FFT)

**Intuition.** Sur un domaine périodique, le Laplacien est diagonal en Fourier : `lap`
devient une multiplication par `-(k_x^2 + k_y^2)`. Une transformée directe + division +
transformée inverse résout Poisson **exactement** (au résidu machine), sans itération.

**Formule.** `phî(k) = -rhŝ(k) / (k_x^2 + k_y^2)`, mode `k=0` fixé à 0 (jauge).

**Pseudocode.** `elliptic/poisson_fft_solver.hpp::PoissonFFTSolver` modèle le même concept
`EllipticSolver` que la multigrille -> le coupleur est générique sur le backend. Limites :
CL périodiques, `N` puissance de 2, mono-rang.

**Validation.** `test_fft_coupler` : MG vs FFT `maxdiff = 1.6e-14` (ils inversent le MÊME
Laplacien 5 points). **Pièges.** Le FFT n'est PAS toujours plus rapide : il l'est sur le
couplé Euler-Poisson Poisson-dominé (~4.8x), mais la multigrille warm-startée gagne sur le
deux-fluides transport-dominé (voir PERFORMANCE.md).

## 8. Euler-Poisson : couplage hyperbolique-elliptique (gravité OU plasma)

**Intuition.** Un gaz compressible (Euler) dont chaque maille crée un potentiel `phi` via
Poisson, et subit en retour la force `g = -grad phi`. Selon le SIGNE de la source
elliptique, le même code fait deux physiques opposées : auto-gravité attractive
(astrophysique, effondrement de Jeans) ou électrostatique répulsive mono-espèce (plasma :
oscillation de Langmuir + explosion de Coulomb).

**Équations.**

$$\partial_t U + \nabla\!\cdot F(U) = S(U,\nabla\phi),\qquad \nabla^2\phi = s\,4\pi G\,(\rho-\rho_0),\qquad s=\pm 1$$

avec `g = -grad phi` et `S = (0, rho g_x, rho g_y, rho u . g)`. `rho0` est le fond
neutralisant (la moyenne de `rho`) : en périodique, Poisson exige un second membre à
moyenne nulle pour être soluble.

**La dualité en une ligne.** `s = +1` (attractif) : là où `rho > rho0`, `phi` creuse un
puits, `g` pointe vers la sur-densité, elle s'accentue (instabilité de Jeans). `s = -1`
(répulsif) : `phi` fait une bosse, `g` pointe vers l'extérieur, la sur-densité se disperse
(Coulomb). Retourner `coupling_sign` retourne `phi`, donc `g` : une seule ligne sépare
gravité et plasma (`model/euler_poisson.hpp::elliptic_rhs`, exposée par `InteractionKind`).

**Dispersion (validation quantitative).** Une perturbation acoustique au repos
`delta_rho = eps rho0 cos(kx)` obéit à

$$\omega^2 = c_s^2 k^2 \;\mp\; \omega_p^2,\qquad \omega_p^2 = 4\pi G\,\rho_0$$

signe `-` en gravité (critère de Jeans : `omega^2 < 0` dès que `omega_p > c_s k`, donc
effondrement), signe `+` en plasma (Bohm-Gross : `omega^2 > 0` toujours, donc
inconditionnellement stable).

**Pseudocode.** Le modèle `EulerPoisson` délègue toute l'hydrodynamique à `Euler` (flux,
vitesses d'onde) et n'ajoute que `source` (la force, via `aux = grad phi`) et `elliptic_rhs`
(le second membre signé). C'est le chemin « aux entre par la SOURCE » du concept
`PhysicalModel` (contraste avec le diocotron, « aux entre par le FLUX »). Branché tel quel
sur `Coupler<EulerPoisson>` : `elliptic_rhs -> multigrille/FFT -> aux=grad phi -> assemble_rhs`.

**Validation.** `test_euler_poisson` (Jeans stable : `omega` mesuré à 0.1% de la théorie,
masse et qté de mouvement conservées) ; `test_euler_poisson_plasma` (Bohm-Gross à 0.1%, et
un même grumeau gaussien dont le pic CROÎT en gravité et DÉCROÎT en plasma : signes
opposés). Démo Python : `tutorials/run/plasma.py`. **Pièges.** Le FFT direct exige `N`
puissance de 2 (sinon UB) ; `rho0` doit valoir `<rho>` exactement, sinon le second membre
périodique n'est pas à moyenne nulle et `phi` dérive.

## 9. AMR : sous-cyclage Berger-Oliger + reflux

**Intuition.** Raffiner seulement là où il le faut. Un niveau fin (pas d'espace `dx/2`)
recouvre une sous-région ; pour respecter sa propre CFL il fait `r=2` sous-pas de `dt/2`
pendant que le grossier fait 1 pas de `dt`. Problème : à l'interface fin-grossier, les
deux niveaux calculent des flux différents -> la conservation est cassée. Le **reflux**
(FluxRegister) corrige la maille grossière par la différence (flux fin intégré - flux
grossier).

**Dérivation.** À chaque face fin-grossier, on accumule le flux fin sur les `r` sous-pas
et on remplace la contribution grossière :

$$U_c \mathrel{-}= \frac{1}{\Delta x_c}\Big(\underbrace{\textstyle\sum_s \Delta t_f\,\bar F_f^{(s)}}_{\text{flux fin intégré}} - \Delta t_c\,F_c\Big)$$

**Pseudocode.** `integrator/amr_reflux_mf.hpp` : `amr_step_2level_mf` (2 niveaux),
`amr_step_multilevel_mf` (récursion N niveaux, `subcycle_level_mf`). Transfert :
`mf_average_down` (fin -> grossier), `mf_fill_fine_ghosts_t` (interp espace+temps des
ghosts fins).

**Validation.** `test_amr_reflux_mf`, `test_amr_multilevel_mf` : **bit-identique** à la
pile Fab2D de référence (`amr_multilevel.hpp`) ; conservation à `~1e-12`. **Pièges.** Le
flux grossier doit être échantillonné AVANT d'avancer le grossier (sinon mauvais
centrage temporel) ; la moyenne descendante doit précéder la mesure de masse initiale.

## 10. AMR multi-patch : reflux coverage-aware

**Intuition.** Un niveau fin n'est pas une seule boîte mais un **ensemble de patchs**
(plusieurs zones raffinées). Deux subtilités : (a) au joint entre deux patchs voisins
(interface fin-fin), il ne faut PAS refluxer (ce n'est pas une interface fin-grossier) ;
(b) la correction doit aller dans la BONNE boîte parente quand le grossier est lui-même
multi-box.

**Dérivation.** Masque de couverture : une cellule grossière est *couverte* si un patch
fin la recouvre. Le reflux ne corrige une cellule grossière adjacente à un patch que si
elle n'est PAS couverte par un autre patch. La correction est routée vers la boîte parente
qui contient la cellule (recherche `mf_find_box`).

**Pseudocode.** `amr_step_2level_multipatch` (2 niveaux, grossier mono-box) et
`amr_step_multilevel_multipatch` (`subcycle_level_mp`, multi-box à CHAQUE niveau). La
couverture est bâtie sur le **BoxArray global** (toutes les boîtes, connues de tous les
rangs) : correcte sous n'importe quelle distribution MPI.

**Validation.** `test_amr_multipatch` (« 2 boîtes pavant = 1 grande boîte », `0` exact) ;
`test_amr_multilevel_multipatch` (3 niveaux mono-box = `amr_step_multilevel_mf` ; 2 niveaux
multi-box = `amr_step_2level_multipatch` ; 3 niveaux avec niveau intermédiaire multi-box
conservatif), les trois à `0`. Sous MPI, `test_mpi_amr_multipatch3` (3 niveaux, niveau
intermédiaire multi-box réparti dont le parent d'un patch fin tombe sur un autre rang) est
bit à bit identique np=1/2/4, masse conservée. **Pièges.** Sans le masque de couverture, le
joint fin-fin serait reflué deux fois -> non-conservation ; le grossier doit être disponible,
ce que le distribué assure par `parallel_copy` (copie inter-niveaux parent->enfant-coarsen) +
gather du registre par `all_reduce_sum_inplace` (niveau 0 répliqué, niveaux >0 répartis).

## 11. Clustering Berger-Rigoutsos

**Intuition.** Étant donné les cellules marquées (fort gradient), trouver un petit nombre
de boîtes rectangulaires qui les couvrent sans trop de gaspillage. L'algorithme coupe
récursivement une région là où la « signature » (histogramme des marques projeté) présente
un trou ou un changement de courbure marqué (le laplacien de la signature).

**Pseudocode.** `amr/cluster.hpp::berger_rigoutsos` : signature par axe -> trou (zéro) ou
inflexion (laplacien max) -> coupe -> récursion jusqu'à une efficacité de remplissage
suffisante (`ClusterParams`). `amr/regrid.hpp::tag_cells` + `grow_tags` produisent et
dilatent les marques.

**Validation.** `test_amr_cluster_step` (clustering branché sur le pas multipatch
conservatif, masse `6.4e-12`) ; le démo `diocotron_multipatch` re-clusterise à la volée.
**Pièges.** Le nesting propre (chaque patch fin strictement intérieur à la couverture
parente) doit être imposé après le clustering, sinon le ghost-fill inter-niveaux échoue.

## 12. Deux-fluides isotherme asymptotic-preserving

Résumé ; détail complet dans [two_fluid_ap.md](two_fluid_ap.md).

**Intuition.** Deux espèces (électrons, ions) couplées par Poisson. La fréquence plasma
`omega_pe` rend le système **raide** : un schéma explicite demande `omega_pe dt < O(1)`,
minuscule en régime quasi-neutre. Le schéma AP traite la force de Lorentz implicitement et
reformule Poisson pour rester stable quand `omega_pe -> inf`.

**Formule clé.** `beta0 = dt^2 (omega_pe^2 + omega_pi^2)`, Poisson reformulé
`lap(phi) = (n_e^* - n_i^*)/(1 + beta0)`. Le facteur `1/(1+beta0)` est l'ingrédient AP :
il tend vers 0 quand la raideur explose, forçant la quasi-neutralité.

**Validation.** `test_two_fluid_ap_2d_mf` : dispersion isotrope (3.1%), borne AP et
quasi-neutralité à `omega_pe = 1e3` là où le non-stabilisé explose, conservation par
espèce. **Pièges.** La continuité centrée est dispersive sur les fronts raides (option
upwind MUSCL `upwind_continuity`) ; le sous-dépassement mesuré est surtout physique
(raréfaction acoustique), pas du Gibbs.

## 13. Champ magnétique : rotation cyclotron

**Intuition.** Un champ `B_z` hors-plan ajoute la force de Lorentz magnétique `z (m x B)`
qui ne fait que **faire tourner** `(m_x, m_y)` à la fréquence cyclotron `w_c`, sans changer
`|m|` ni `n`. La rotation exacte d'angle `theta = z w_c dt` est inconditionnellement stable
(aucune limite `w_c dt`).

**Formule.** Rotation analytique des composantes de quantité de mouvement :

$$\begin{pmatrix} m_x' \\ m_y' \end{pmatrix} =
  \begin{pmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{pmatrix}
  \begin{pmatrix} m_x \\ m_y \end{pmatrix}$$

composée au pas électrostatique par splitting de Strang (`R(θ/2) ∘ pas-ES ∘ R(θ/2)`,
ordre 2). Opt-in via `omega_ce`, `omega_ci` (0 = pas de champ, comportement inchangé).

**Validation.** `test_two_fluid_cyclotron` : plasma uniforme (charge nulle, `E=0`,
transport inerte) -> rotation pure à `w_c` mesurée à **0.00%** d'écart, `|m|` conservée à
`8.9e-16`. **Pièges.** En régime fortement magnétisé ET fort champ, le splitting de Strang
de E et B perd en précision face à un push de Boris combiné (E+B au même centrage temporel) :
piste d'amélioration, pas encore implémentée.

## 14. Le seam de dispatch (série / OpenMP / Kokkos / MPI)

Pas un algorithme numérique mais le point de bascule qui les rend tous portables. Détail
dans [ARCHITECTURE.md](ARCHITECTURE.md) section 4 (couche execution). En bref : `for_each_cell(box, lambda)`
dispatche vers une boucle série, `#pragma omp parallel for`, ou `Kokkos::parallel_for`
selon le backend de la cible `adc` ; `comm.hpp` enveloppe les collectives MPI
(`all_reduce_sum`, `all_reduce_sum_inplace`). La physique ne voit jamais le backend.

# Algorithmes

Catalogue des methodes numeriques generiques du coeur `adc_cpp`. Pour chacune : l'intuition, la
formule et sa discretisation, un pseudocode, le fichier C++ qui l'implemente, et les contraintes. Le
coeur est agnostique au modele ; il ne nomme aucun scenario (diocotron, Euler-Poisson, deux-fluides).
Ces scenarios sont des compositions de briques generiques et vivent cote application
([`adc_cases`](https://github.com/wolf75222/adc_cases)) ; leurs validations bout-en-bout aussi.

Chaque section suit le meme plan : intuition (a quoi ca sert), formule et discretisation (d'ou ca
vient), pseudocode (l'algorithme), code (le fichier et les fonctions), contraintes et remarques (la
stabilite, les limites, le test ctest qui le couvre). Tous les chemins de fichiers et noms de tests
cites existent dans ce depot.

Architecture (couches, seam de dispatch, frontiere bibliotheque/application) :
[ARCHITECTURE.md](ARCHITECTURE.md). Choix de conception : [CHOICES.md](CHOICES.md). Trace des
validations device GH200 : [GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md). References :
[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md).

## Sommaire

- [Equations modele](#equations-modele)
- [1. Volumes finis : Godunov ordre 1](#1-volumes-finis--godunov-ordre-1)
- [2. Flux numeriques : Rusanov, HLL, HLLC, Roe](#2-flux-numeriques--rusanov-hll-hllc-roe)
- [3. Reconstruction MUSCL (ordre 2) et WENO5-Z (ordre 5)](#3-reconstruction-muscl-ordre-2-et-weno5-z-ordre-5)
- [4. Integration en temps : SSPRK, integrateurs objets, integrateur utilisateur](#4-integration-en-temps--ssprk-integrateurs-objets-integrateur-utilisateur)
- [5. Sources raides : IMEX asymptotic-preserving et IMEX partiel](#5-sources-raides--imex-asymptotic-preserving-et-imex-partiel)
- [6. Splitting d'operateurs : Lie et Strang](#6-splitting-doperateurs--lie-et-strang)
- [7. Multirate : sous-cyclage, cadence, pas adaptatif](#7-multirate--sous-cyclage-cadence-pas-adaptatif)
- [8. Terme parabolique : diffusion en flux de face](#8-terme-parabolique--diffusion-en-flux-de-face)
- [9. Elliptique : multigrille geometrique](#9-elliptique--multigrille-geometrique)
- [10. Elliptique : Poisson spectral (FFT), mono-rang et distribue](#10-elliptique--poisson-spectral-fft-mono-rang-et-distribue)
- [11. Elliptique etendu : eps(x), Helmholtz/ecrante, anisotrope](#11-elliptique-etendu--epsx-helmholtzecrante-anisotrope)
- [12. Elliptique a tenseur plein : Krylov matrice-libre (BiCGStab)](#12-elliptique-a-tenseur-plein--krylov-matrice-libre-bicgstab)
- [13. Source implicite condensee : condensation de Schur](#13-source-implicite-condensee--condensation-de-schur)
- [14. Bord embedded : cut-cell Shortley-Weller](#14-bord-embedded--cut-cell-shortley-weller)
- [15. Domaine disque : masque, transport masque, transport cut-cell](#15-domaine-disque--masque-transport-masque-transport-cut-cell)
- [16. Geometrie polaire : transport et Poisson sur anneau (r, theta)](#16-geometrie-polaire--transport-et-poisson-sur-anneau-r-theta)
- [17. AMR : sous-cyclage Berger-Oliger + reflux conservatif](#17-amr--sous-cyclage-berger-oliger--reflux-conservatif)
- [18. AMR multi-patch : reflux coverage-aware, distribue MPI](#18-amr-multi-patch--reflux-coverage-aware-distribue-mpi)
- [19. Clustering Berger-Rigoutsos et regrid](#19-clustering-berger-rigoutsos-et-regrid)
- [20. Maillage distribue : BoxArray global, halos, equilibrage](#20-maillage-distribue--boxarray-global-halos-equilibrage)
- [21. Canal aux extensible](#21-canal-aux-extensible)
- [22. Composition runtime et systeme multi-especes](#22-composition-runtime-et-systeme-multi-especes)
- [23. DSL symbolique : codegen, JIT, AOT](#23-dsl-symbolique--codegen-jit-aot)
- [24. Le seam de dispatch (serie / OpenMP / Kokkos / MPI)](#24-le-seam-de-dispatch-serie--openmp--kokkos--mpi)
- [Quel schema ou solveur quand](#quel-schema-ou-solveur-quand)
- [References](#references)

---

## Equations modele

Le coeur resout, sur maillage cartesien adaptatif (et, en option, sur anneau polaire ou sous-domaine
disque immerge), la forme generique

$$\partial_t U + \mathrm{div} F(U, \mathrm{aux}) = S(U, \mathrm{aux}) \qquad \text{(hyperbolique, par bloc)}$$

$$\mathrm{div}(\varepsilon\,\nabla \phi) - \kappa\,\phi = f(U) \qquad \text{(elliptique, partage)}$$

La partie hyperbolique `U` et la partie elliptique `phi` se couplent a chaque pas par le canal `aux`
(contrat de base `(phi, grad_x, grad_y)`, extensible a `B_z`, `T_e`). Un modele est une composition
`CompositeModel<Transport, Source, Elliptic>` ; le couplage entre par le flux (`aux` lu dans `F`) ou
par la source (`aux` lu dans `S`), sous le meme operateur spatial. Les briques de reconstruction, de
flux et de source sont reutilisees entre les geometries (cartesien, polaire, cut-cell) ; seules les
metriques et les divergences changent. La discretisation spatiale est en volumes finis (section 1) ;
l'avancee en temps est une methode des lignes (section 4) ; l'elliptique est resolu a chaque pas
(sections 9 a 13) et relu par `aux`.


---

## 1. Volumes finis : Godunov ordre 1

**Intuition.** Le profil dans chaque maille est remplace par sa moyenne. A chaque interface,
deux moyennes se rencontrent : un probleme de Riemann local. Le flux numerique le resout
(approximativement) et la mise a jour conservative transporte la matiere d'une maille a l'autre.

**Formule / discretisation.** Integration de la loi de conservation $\partial_t U + \mathrm{div}\,F(U,\mathrm{aux}) = S(U,\mathrm{aux})$
sur la maille $(i,j)$ et le pas $\Delta t$ :

$$U_{ij}^{n+1} = U_{ij}^n - \frac{\Delta t}{\Delta x}\big(\hat F_{i+1/2,j} - \hat F_{i-1/2,j}\big)
                             - \frac{\Delta t}{\Delta y}\big(\hat G_{i,j+1/2} - \hat G_{i,j-1/2}\big)
                             + \Delta t\, S_{ij}$$

La forme conservative (difference de flux de face) garantit la conservation discrete exacte : ce
que la maille $i$ perd a sa face droite, la maille $i+1$ le gagne a sa face gauche. C'est la
propriete dont depend tout l'AMR (le reflux corrige justement ces flux de face). Le coeur ne fait
pas la mise a jour lui-meme : il assemble le residu de la methode des lignes

$$R_{ij} = -\,\mathrm{div}\,\hat F + S = S_{ij}
          - \frac{\hat F_{i+1/2,j} - \hat F_{i-1/2,j}}{\Delta x}
          - \frac{\hat G_{i,j+1/2} - \hat G_{i,j-1/2}}{\Delta y},$$

et l'integrateur en temps applique $U^{n+1} = U^n + \Delta t\, R$ (Euler) ou une combinaison SSPRK.
Le flux de face $\hat F_{i+1/2}$ est evalue a partir des etats reconstruits de part et d'autre :
$\hat F_{i+1/2} = \texttt{nflux}(\texttt{recon}_L, \texttt{aux}_i, \texttt{recon}_R, \texttt{aux}_{i+1}, \texttt{dir})$.
L'auxiliaire `aux` (`phi`, `grad phi`) n'est pas reconstruit (champ lisse issu de l'elliptique) : on
prend la valeur de cellule de chaque cote. Un terme parabolique optionnel ($+\nu\,\mathrm{Lap}\,U$,
garde par le concept `DiffusiveModel`) s'ajoute, soit en differences centrees 5 points dans
`assemble_rhs`, soit en flux de face Fickien $-\nu\,\nabla U$ dans `compute_face_fluxes` (cf. section 8).

```
function assemble_rhs(model, U, aux, geom, R, recon_prim):   # R = -div Fhat + S
    dx, dy = geom.dx(), geom.dy()
    for box li in U:                                          # seam for_each_cell : serie / OpenMP / Kokkos
        for each valid cell (i, j):
            Ac  = load_aux(aux, i, j)                         # [phi, grad_x, grad_y, extra...]
            # face x gauche (i-1/2) et droite (i+1/2)
            Lxm = reconstruct(model, U, i-1, j, dir=0, sgn=+1, recon_prim)   # etat L de la face i-1/2
            Rxm = reconstruct(model, U, i,   j, dir=0, sgn=-1, recon_prim)   # etat R de la face i-1/2
            Lxp = reconstruct(model, U, i,   j, dir=0, sgn=+1, recon_prim)
            Rxp = reconstruct(model, U, i+1, j, dir=0, sgn=-1, recon_prim)
            Fxm = nflux(model, Lxm, A(i-1,j), Rxm, Ac,     dir=0)
            Fxp = nflux(model, Lxp, Ac,       Rxp, A(i+1,j), dir=0)
            # faces y : idem dans la direction j
            Fym, Fyp = (analogue avec dir=1)
            S = model.source(load_state(U, i, j), Ac)
            for c in 0 .. n_vars-1:
                R(i,j,c) = S[c] - (Fxp[c]-Fxm[c])/dx - (Fyp[c]-Fym[c])/dy
            if DiffusiveModel(model):                          # if constexpr -> zero codegen sinon
                R(i,j,c) += nu * (Lap_x U + Lap_y U)           # +nu Lap(U), 5 points centres
```

**Code.** [`include/adc/numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp) :
`assemble_rhs<Limiter, NumericalFlux>` calcule directement $R = -\mathrm{div}\,\hat F + S$, en passant
par le foncteur nomme device `detail::AssembleRhsKernel<Limiter, NumericalFlux, Model>` (foncteur
plutot que lambda etendue : emission device fiable sous nvcc depuis une TU externe, chemin AOT
`add_compiled_model`). `compute_face_fluxes<Limiter, NumericalFlux>` ecrit les flux de face `Fx, Fy`
(via `detail::FaceFluxXKernel` / `FaceFluxYKernel`) avant la divergence : c'est ce dont le reflux AMR a
besoin. Memes `reconstruct` et meme flux numerique que `assemble_rhs`, donc
$R = S - (\texttt{Fx}(i{+}1)-\texttt{Fx}(i))/\Delta x - (\texttt{Fy}(j{+}1)-\texttt{Fy}(j))/\Delta y$
redonne exactement le residu. La boucle passe par le seam `for_each_cell`. Une variante opt-in
`assemble_rhs_masked` (foncteur `AssembleRhsMaskedKernel`) restreint le transport a un sous-domaine
actif : flux normal nul sur les faces touchant une cellule masquee (paroi FV conservative), residu nul
sur les cellules inactives. Le pas CFL global se lit par `max_wave_speed_mf` (reduction sur les boites
locales puis `all_reduce_max` MPI, sinon chaque rang choisirait un `dt` different et divergerait).

**Contraintes / remarques.** Condition CFL : $\Delta t \le C\,\dfrac{\min(\Delta x,\Delta y)}{\max|\lambda|}$,
ou $\lambda$ est la vitesse d'onde locale et $C \le 1$ a l'ordre 1 ; `max_wave_speed_mf` fournit
$\max|\lambda|$. Un modele sans transport ($\max|\lambda| = 0$) ne contraint pas le pas
(`max_wave_speed_mf` renvoie 0). L'operateur n'ecrit que `R` (il ne touche ni `U` ni `aux`, pas de
remplissage de ghosts). Validation : `test_spatial_discretisation` (le couple reconstruction x flux est
un type nomme assemble par `assemble_rhs`), `test_cfl_dt` (`dt = cfl * min(dx,dy) / max|lambda|`
multi-especes). L'invariant cartesien/polaire est verifie bit-a-bit (l'operateur polaire ne touche pas
ce chemin). Les validations bout-en-bout (diocotron, Euler-Poisson) vivent cote `adc_cases`.

## 2. Flux numeriques : Rusanov, HLL, HLLC, Roe

**Intuition.** Quatre niveaux de fidelite pour le Riemann de face, ordonnes par nombre d'ondes
restituees. Rusanov pose une seule bosse de diffusion (le plus robuste, le plus diffusif) ; HLL estime
deux vitesses de signal et garde une seule region etoile ; HLLC ajoute l'onde de contact (la
discontinuite de densite passive) ; Roe linearise le systeme par la moyenne de Roe et resout
exactement le Riemann linearise. Chaque flux est une politique sans etat (foncteur `ADC_HD`,
device-callable, aucun virtuel) de contrat
$\texttt{operator()}(m, U_L, A_L, U_R, A_R, \texttt{dir}) \to \texttt{Model::State}$, choisie en
template a cote du limiteur de reconstruction.

**Formule / discretisation.** Rusanov (Lax-Friedrichs local), composante par composante (upwind
scalaire, sans couplage) :

$$\hat F_{i+1/2} = \tfrac12\big(F(U_L)+F(U_R)\big) - \tfrac12\,\alpha\,(U_R - U_L),
\qquad \alpha = \max\big(s_L(U_L), s_R(U_R)\big),$$

ou $s_{L,R}$ (`max_wave_speed`) de chaque etat ; Rusanov ne demande que ce membre, donc
s'applique a tout `PhysicalModel` de base. HLL utilise les estimees de Davis
$s_L = \min(s_L^{gauche}, s_L^{droit})$, $s_R = \max(s_R^{gauche}, s_R^{droit})$ via `hll_speeds`, et
retombe sur le flux amont en regime supersonique :

$$\hat F^{HLL} = \begin{cases}
F(U_L) & s_L \ge 0 \\[2pt]
\dfrac{s_R F(U_L) - s_L F(U_R) + s_L s_R (U_R - U_L)}{s_R - s_L} & s_L < 0 < s_R \\[6pt]
F(U_R) & s_R \le 0
\end{cases}$$

HLLC restaure l'onde de contact $S_*$ au milieu (vitesse de Toro eq. 10.37) et reconstruit les etats
etoile $U_L^*, U_R^*$ :

$$S_* = \frac{p_R - p_L + \rho_L u_{nL}(s_L - u_{nL}) - \rho_R u_{nR}(s_R - u_{nR})}
             {\rho_L(s_L - u_{nL}) - \rho_R(s_R - u_{nR})},
\qquad \hat F^{HLLC} = F_K + s_K\,(U_K^* - U_K),\ K \in \{L, R\},$$

avec $u_n$ la vitesse normale et le facteur $\rho_K (s_K - u_{nK}) / (s_K - S_*)$ pour les etats etoile.
Roe linearise le systeme par la moyenne ponderee en $\sqrt{\rho}$ :

$$\hat F^{Roe} = \tfrac12\big(F_L + F_R\big) - \tfrac12 \sum_k |\tilde\lambda_k|\,\alpha_k\,r_k,$$

ondes $\{u_n - c,\ u_n,\ u_n,\ u_n + c\}$ avec celerite $c$ deduite de l'enthalpie de Roe $H$, et
correction d'entropie de Harten ($\varepsilon = 0.1\,c$) sur les ondes acoustiques 1 et 5 pour eviter
le choc non-entropique (sonic glitch). HLLC et Roe ciblent Euler 2D (`n_vars == 4`) : indices de
quantite de mouvement normale/tangentielle selon `dir` ; $\gamma - 1$ est deduit de l'etat (gaz
parfait), aucun membre `gamma` n'est requis du modele.

```
function HLLC(m, UL, AL, UR, AR, dir):                 # n_vars == 4 (Euler 2D)
    in = (dir==0 ? 1 : 2);  it = (dir==0 ? 2 : 1)      # qte de mvt normale / tangentielle
    rL, rR   = UL[0], UR[0]
    unL, unR = UL[in]/rL, UR[in]/rR
    pL, pR   = m.pressure(UL), m.pressure(UR)
    sL, sR   = hll_speeds(m, UL,AL, UR,AR, dir)         # Davis : min/max des vitesses signees
    FL, FR   = m.flux(UL,AL,dir), m.flux(UR,AR,dir)
    if sL >= 0: return FL                               # supersonique a droite -> flux amont
    if sR <= 0: return FR
    sStar = (pR - pL + rL*unL*(sL-unL) - rR*unR*(sR-unR))    # Toro 10.37
          / (rL*(sL-unL) - rR*(sR-unR))
    if sStar >= 0:                                      # etat etoile gauche
        fac = rL*(sL - unL) / (sL - sStar)
        Us[0]=fac; Us[in]=fac*sStar; Us[it]=fac*(UL[it]/rL)
        Us[3]=fac*(UL[3]/rL + (sStar-unL)*(sStar + pL/(rL*(sL-unL))))
        return FL + sL*(Us - UL)
    else:                                              # etat etoile droit (symetrique)
        fac = rR*(sR - unR) / (sR - sStar)
        ... return FR + sR*(Us - UR)
```

**Code.** Politiques sans etat dans
[`include/adc/numerics/numerical_flux.hpp`](../include/adc/numerics/numerical_flux.hpp) : `RusanovFlux`,
`HLLFlux`, `HLLCFlux`, `RoeFlux` (toutes `ADC_HD`). `RusanovFlux` boucle composante par composante avec
`m.max_wave_speed` ; `HLLFlux`/`HLLCFlux` partagent la fonction libre `hll_speeds` (estimees de Davis,
requiert `m.wave_speeds`) ; `HLLCFlux`/`RoeFlux` requierent en plus `m.pressure`. La fonction de
compatibilite `rusanov_flux` (dans `spatial_operator.hpp`) delegue a `RusanovFlux{}` pour les references
serie. Le flux est passe en template : `compute_face_fluxes<Limiter, NumericalFlux, Model>` et
`assemble_rhs<Limiter, NumericalFlux, Model>` sont temples sur la politique de flux, choisie
independamment du limiteur. L'adaptateur `SourceFreeModel` (demi-pas explicite IMEX) ne forwarde
`pressure` et `wave_speeds` que si le modele enveloppe les expose (clause `requires`), pour qu'un
demi-pas IMEX reste en flux HLLC.

**Contraintes / remarques.** `RusanovFlux` est le seul flux compatible avec le `PhysicalModel` minimal
(il ne lit que `max_wave_speed`) : c'est le defaut robuste pour le transport scalaire, au prix d'une
diffusion accrue ($\alpha$ borne haute). `HLLFlux` lisse encore la discontinuite de contact (une seule
region etoile). `HLLCFlux` et `RoeFlux` supposent `n_vars == 4` (Euler 2D) ; comportement indefini sur
d'autres modeles. HLLC sur un etat du vide (densite nulle) divise par zero dans le facteur etoile et
demande un garde-fou amont. Roe utilise `std::sqrt` pour la moyenne en $\sqrt{\rho}$ (device-clean
sous Kokkos/nvcc) ; sa propriete cle $F_R - F_L = \tilde A\,(U_R - U_L)$ donne le flux amont exact en
regime supersonique, et la correction de Harten evite l'expansion non-physique au point sonique.
Validation : `test_roe_flux` (consistance $\hat F(U,U) = F(U)$, resolution exacte d'un Riemann linearise,
`eigenvalues()` d'Euler). Le couplage `aux` entre par le flux (`F` lit `aux`) ou par la source, sous le
meme operateur spatial ; les memes politiques de flux servent en cartesien, polaire et cut-cell EB.


---

## 3. Reconstruction MUSCL (ordre 2) et WENO5-Z (ordre 5)

**Intuition.** Godunov ordre 1 (section 1) remplace le profil de chaque maille par sa moyenne, ce qui
est tres diffusif. MUSCL reconstruit un profil lineaire par maille a partir d'une pente limitee, puis
evalue le flux numerique sur les valeurs reconstruites aux faces ; le limiteur ecrete la pente pres des
extrema pour rester TVD (pas d'oscillation parasite). WENO5-Z monte a l'ordre 5 en zone lisse via une
moyenne non lineaire de trois reconstructions d'ordre 3, sans limiteur explicite, en ecartant le stencil
qui traverse un front raide (le bord d'anneau).

**Formule / discretisation.** Une politique de reconstruction est ponctuelle : elle prend les deux
differences finies non centrees autour de la cellule $i$,

$$a = U_i - U_{i-1} \quad (\text{difference arriere}),\qquad b = U_{i+1} - U_i \quad (\text{difference avant}),$$

et rend une pente limitee $\sigma_i = \mathrm{lim}(a,b)$. Les trois limiteurs MUSCL sont :

$$\mathrm{minmod}(a,b) = \begin{cases} \mathrm{sgn}(a)\,\min(|a|,|b|) & ab>0\\ 0 & ab\le 0\end{cases},
\qquad
\mathrm{vanleer}(a,b) = \begin{cases} \dfrac{2ab}{a+b} & ab>0\\ 0 & ab\le 0\end{cases},$$

et $\mathrm{NoSlope}(a,b)=0$ (ordre 1, constant par morceaux). van Leer est la moyenne harmonique des
deux differences : moins dissipatif aux extrema lisses que minmod (qui retombe a l'ordre 1 local sur un
pic). Les etats reconstruits aux faces de l'interface $i+1/2$ sont alors

$$U_L = U_i + \tfrac12\,\sigma_i,\qquad U_R = U_{i+1} - \tfrac12\,\sigma_{i+1},$$

passes au flux numerique $\hat F(U_L,U_R)$. MUSCL demande 2 ghosts (pente en $i\pm 1$).

WENO5-Z reconstruit la valeur a la face entre $v_0$ et $v_{+1}$ a partir du stencil 5 points
$(v_{-2},v_{-1},v_0,v_{+1},v_{+2})$. Trois reconstructions d'ordre 3 :

$$q_0 = \tfrac{2v_{-2}-7v_{-1}+11v_0}{6},\quad
  q_1 = \tfrac{-v_{-1}+5v_0+2v_{+1}}{6},\quad
  q_2 = \tfrac{2v_0+5v_{+1}-v_{+2}}{6},$$

les indicateurs de regularite de Jiang-Shu :

$$\beta_0 = \tfrac{13}{12}(v_{-2}-2v_{-1}+v_0)^2 + \tfrac14(v_{-2}-4v_{-1}+3v_0)^2,$$
$$\beta_1 = \tfrac{13}{12}(v_{-1}-2v_0+v_{+1})^2 + \tfrac14(v_{-1}-v_{+1})^2,$$
$$\beta_2 = \tfrac{13}{12}(v_0-2v_{+1}+v_{+2})^2 + \tfrac14(3v_0-4v_{+1}+v_{+2})^2,$$

et les poids WENO-Z (Borges 2008), avec $\tau_5 = |\beta_0-\beta_2|$ et poids lineaires optimaux
$d_0=\tfrac{1}{10}, d_1=\tfrac{6}{10}, d_2=\tfrac{3}{10}$ :

$$\alpha_k = d_k\Big(1 + \big(\tfrac{\tau_5}{\varepsilon+\beta_k}\big)^2\Big),
\qquad
\omega_k = \frac{\alpha_k}{\alpha_0+\alpha_1+\alpha_2},
\qquad
v_{i+1/2} = \sum_{k=0}^{2}\omega_k\, q_k .$$

En zone lisse $\tau_5 \to 0$ et $\omega_k \to d_k$ : on retrouve l'ordre 5. La mesure $\tau_5$ a base de
$|\beta_0-\beta_2|$ rend WENO-Z moins dissipatif que Jiang-Shu classique, donc meilleur pour preserver le
taux de croissance d'un mode lisse. Stencil 5 points -> 3 ghosts. Pour la face $-x$, on appelle la meme
fonction avec le stencil renverse $(v_{+2},v_{+1},v_0,v_{-1},v_{-2})$.

```
function reconstruct_muscl(U, i, lim):        # face i+1/2, pente limitee
    a   <- U[i]   - U[i-1]                     # difference arriere
    b   <- U[i+1] - U[i]                        # difference avant
    sig <- lim(a, b)                            # minmod / vanleer / NoSlope (=0)
    a2  <- U[i+1] - U[i]
    b2  <- U[i+2] - U[i+1]
    sig_r <- lim(a2, b2)
    U_L <- U[i]   + 0.5 * sig                   # etat gauche reconstruit
    U_R <- U[i+1] - 0.5 * sig_r                 # etat droit reconstruit
    return (U_L, U_R)

function minmod(a, b):
    if a*b <= 0: return 0
    fa <- |a| ; fb <- |b|                       # valeur absolue sans std::abs
    return a if fa < fb else b

function vanleer(a, b):
    ab <- a*b
    if ab <= 0: return 0
    return 2*ab / (a + b)                        # moyenne harmonique

function weno5z(vm2, vm1, v0, vp1, vp2):        # face entre v0 et vp1
    eps <- 1e-40
    q0  <- ( 2*vm2 - 7*vm1 + 11*v0) / 6          # 3 recon. d'ordre 3
    q1  <- (  -vm1 + 5*v0  +  2*vp1) / 6
    q2  <- ( 2*v0  + 5*vp1 -    vp2) / 6
    b0  <- 13/12*(vm2-2*vm1+v0)^2 + 1/4*(vm2-4*vm1+3*v0)^2   # indicateurs beta
    b1  <- 13/12*(vm1-2*v0+vp1)^2 + 1/4*(vm1-vp1)^2
    b2  <- 13/12*(v0-2*vp1+vp2)^2 + 1/4*(3*v0-4*vp1+vp2)^2
    tau5 <- |b0 - b2|                            # ternaire device-safe, pas std::abs
    a0  <- (1/10)*(1 + (tau5/(eps+b0))^2)        # poids WENO-Z non normalises
    a1  <- (6/10)*(1 + (tau5/(eps+b1))^2)
    a2  <- (3/10)*(1 + (tau5/(eps+b2))^2)
    inv <- 1 / (a0 + a1 + a2)
    return (a0*q0 + a1*q1 + a2*q2) * inv

# Cote operateur spatial (spatial_operator::reconstruct) :
#   n_ghost == 1 (NoSlope)  -> Godunov ordre 1, pas de lecture a +/-2
#   n_ghost == 2 (MUSCL)    -> reconstruct_muscl avec le limiteur
#   n_ghost >= 3 (Weno5)    -> weno5z(stencil direct) pour face +dir,
#                              weno5z(stencil renverse) pour face -dir
```

**Code.** Politiques `Limiter` ponctuelles dans
[`include/adc/numerics/reconstruction.hpp`](../include/adc/numerics/reconstruction.hpp) : `NoSlope`
(`n_ghost = 1`, `operator()` rend `Real(0)`), `Minmod` et `VanLeer` (`n_ghost = 2`, `operator()(a,b)`
rend la pente limitee, valeur absolue codee a la main pour rester device-safe sans `<cmath>`), `Weno5`
(`n_ghost = 3`, tag dont l'`operator()` est un no-op qui satisfait juste le concept `Limiter`). La
reconstruction d'ordre 5 vit dans la fonction libre `weno5z(vm2, vm1, v0, vp1, vp2)` du meme header :
elle rend la valeur a la face entre `v0` et `vp1`, et pour la face opposee on lui passe le stencil
renverse. Toutes sont `ADC_HD` (device-callable, polymorphisme statique : le limiteur est un parametre de
template de `assemble_rhs` / `compute_face_fluxes`, inline sur device). L'acces au stencil maillage et le
routage par `n_ghost` sont dans `reconstruct` de `numerics/spatial_operator.hpp` ; la politique elle-meme
ne boucle sur aucune grille. La reconstruction peut porter sur les variables conservees ou primitives
(`rho, u, p`) selon le bloc.

**Contraintes / remarques.** La reconstruction ne change pas la condition de stabilite hyperbolique : le
pas reste borne par la CFL de la section 1, `dt <= C dx / max|lambda|`. Limites et pieges :
- `Minmod` est strictement TVD mais retombe a l'ordre 1 local aux extrema (il efface les pics lisses) ;
  pour les modes de croissance Diocotron on prefere `VanLeer`, moins dissipatif aux extrema.
- `weno5z` est lisse (aucun branchement sur le signe : les $\beta_k$ et $\tau_5$ sont des carres donc
  toujours $\ge 0$, et seul $|\beta_0-\beta_2|$ passe par un ternaire), ce qui le rend pleinement
  device-callable ; le plancher `eps = 1e-40` evite la division par zero sur un stencil constant.
- Reconstruire la variable conservee plutot que la primitive change le comportement aux chocs forts
  (les etats reconstruits peuvent sortir du domaine admissible cote conserve).
- Le cout en ghosts pilote la largeur de halo a echanger : 1 (NoSlope), 2 (MUSCL), 3 (WENO5).

**Validation.** `test_weno_convergence` (la reconstruction de face d'une fonction lisse atteint l'ordre 5),
`test_primitive_recon` (conversions conservees <-> primitives et leur usage dans la reconstruction),
`test_spatial_discretisation` (le couple reconstruction x flux numerique est un type nomme, exerce de bout
en bout).


---

## 4. Integration en temps : SSPRK, integrateurs objets, integrateur utilisateur

**Intuition.** Strong-Stability-Preserving Runge-Kutta : chaque etage est une combinaison convexe
d'Euler explicites, donc toute propriete de stabilite (TVD, positivite, bornes) tenue par un pas
d'Euler avant sous CFL est tenue par le schema entier, a l'ordre 2 ou 3. Le schema en temps est un
objet de premier plan (`take_step`) que le coupleur appelle, plutot que du SSPRK inline
duplique dans chaque coupleur ; le meme contrat permet a un cas d'apporter son propre integrateur.

**Formule / discretisation.** Methode des lignes : l'espace donne $\dot U = L(U)$ avec
$L(U) = -\mathrm{div} F(U) + S(U)$, evalue par $\texttt{rhs}(U, R) \Rightarrow R = L(U)$.
Euler avant : $U^{n+1} = U^n + \Delta t\, L(U^n)$.

SSPRK2 (Shu-Osher, 2 etages, ordre 2, equivalent a Heun) :

$$U^{(1)} = U^n + \Delta t\, L(U^n), \qquad U^{n+1} = \tfrac12 U^n + \tfrac12\big(U^{(1)} + \Delta t\, L(U^{(1)})\big).$$

SSPRK3 (Shu-Osher, 3 etages, ordre 3) :

$$U^{(1)} = U^n + \Delta t\, L(U^n),$$
$$U^{(2)} = \tfrac34 U^n + \tfrac14\big(U^{(1)} + \Delta t\, L(U^{(1)})\big),$$
$$U^{n+1} = \tfrac13 U^n + \tfrac23\big(U^{(2)} + \Delta t\, L(U^{(2)})\big).$$

Les deux ont coefficient SSP $C = 1$ : la condition SSP est exactement la condition CFL d'Euler avant.
En operations $\texttt{MultiFab}$ le code n'utilise que $\texttt{saxpy}(Y, a, X): Y \mathrel{+}= a\,X$ et
$\texttt{lincomb}(Y, a, X_1, b, X_2): Y \leftarrow a\,X_1 + b\,X_2$. L'etage convexe de SSPRK2 s'ecrit
alors comme une mise a jour d'Euler sur la copie $U^{(1)}$ suivie de
$\texttt{lincomb}(U, \tfrac12, U, \tfrac12, U^{(1)})$, algebriquement identique a la forme convexe ci-dessus.

```
function take_step_SSPRK2(rhs, U, dt):
    R  = MultiFab(layout_of(U), ncomp(U), nghost=0)   # scratch, aucun etat porte
    rhs(U, R)                                          # R = L(U^n)
    U1 = copy(U)
    saxpy(U1, dt, R)                                   # U1 = U^n + dt L(U^n)  (= U^(1))
    rhs(U1, R)                                         # R = L(U^(1))
    saxpy(U1, dt, R)                                   # U1 = U^(1) + dt L(U^(1))
    lincomb(U, 1/2, U, 1/2, U1)                        # U^{n+1} = 1/2 U^n + 1/2 U1

function take_step_SSPRK3(rhs, U, dt):
    R  = MultiFab(layout_of(U), ncomp(U), nghost=0)
    rhs(U, R);  U1 = copy(U);  saxpy(U1, dt, R)        # U^(1) = U^n + dt L(U^n)
    rhs(U1, R); U2 = copy(U1); saxpy(U2, dt, R)
    lincomb(U2, 3/4, U, 1/4, U2)                       # U^(2) = 3/4 U^n + 1/4 (U^(1)+dt L)
    rhs(U2, R); U3 = copy(U2); saxpy(U3, dt, R)
    lincomb(U, 1/3, U, 2/3, U3)                        # U^{n+1} = 1/3 U^n + 2/3 (U^(2)+dt L)

# tag -> politique d'emploi par bloc d'equation (pas le schema lui-meme)
struct TimePolicy<Method, Treatment in {Explicit,Implicit,IMEX,Prescribed}, Substeps>=1, Stride>=1>
TimePolicyTraits<P>: extrait (Method, treatment, substeps, stride), defaut Explicit/1/1 sur un tag nu
ExplicitTime<M=SSPRK2,...> / ImplicitTime<...> / IMEXTime<...> / PrescribedTime  # alias de TimePolicy

# integrateur utilisateur : tout objet qui satisfait le concept
concept TimeStepper<I> = I.take_step(rhs, U, dt) compile
```

**Code.** Deux expressions coexistent, separant le schema mathematique de sa politique d'emploi.
Les tags [`include/adc/numerics/time/time_integrator.hpp`](../include/adc/numerics/time/time_integrator.hpp)
(`SSPRK2`, `SSPRK3`, `UserTimeIntegrator`) nomment, par bloc, le traitement temporel via une
`TimePolicy<Method, TimeTreatment, Substeps, Stride>` ; `TimePolicyTraits` lit ces champs (et accepte
un tag nu, traite alors comme `Explicit` a un seul pas). Les alias `ExplicitTime` / `ImplicitTime` /
`IMEXTime` / `PrescribedTime` fixent le `TimeTreatment`. Les integrateurs objets
[`include/adc/numerics/time/time_steppers.hpp`](../include/adc/numerics/time/time_steppers.hpp)
(`ForwardEuler`, `SSPRK2Step`, `SSPRK3Step`) portent la methode : chacun expose
`take_step(rhs, U, dt)` et n'alloue son scratch (`R`, etages `U1`/`U2`/`U3`) que depuis le layout de
`U`, sans etat persistant. L'integrateur ne voit que `rhs(U_stage, R)` (la fleche methode-des-lignes)
et les operations `saxpy`/`lincomb` de [`include/adc/mesh/mf_arith.hpp`](../include/adc/mesh/mf_arith.hpp) :
il est agnostique du modele et de la discretisation. Le concept `TimeStepper` formalise le contrat, de
sorte qu'un cas peut fournir son propre objet `take_step` exactement comme il fournit un `PhysicalModel`.

**Contraintes / remarques.** SSP de coefficient 1 : la stabilite forte n'est garantie que tant que
$\Delta t$ respecte la CFL d'Euler avant ($C = 1$, pas de marge gagnee sur le pas par rapport au
schema d'ordre 1). En systeme couple, l'ordre du solve elliptique plafonne l'ordre global : un Poisson
resolu une fois par pas limite le champ a l'ordre 1, quel que soit le SSPRK choisi sur l'hyperbolique
(voir splitting, section 6). Les champs `Substeps` (sous-pas plus frequents, $n$ pas de $\Delta t/n$
pour une espece rapide) et `Stride` (cadence plus lente, un bloc lent n'avance qu'un macro-pas tous les
`Stride` pas) sont orthogonaux et relevent du scheduler, pas de l'integrateur (section 7) ; `Stride = 1`
redonne le comportement historique. Les objets `SSPRK2Step`/`SSPRK3Step` reproduisent bit-pour-bit les
anciennes copies inline `SystemCoupler::advance_explicit_ssprk2/ssprk3` (deduplication). Validation :
`test_user_time_integrator` verifie qu'un integrateur fourni par l'utilisateur donne le meme resultat
qu'un SSPRK du coeur.


---

## 5. Sources raides : IMEX asymptotic-preserving et IMEX partiel

**Intuition.** Une source raide (relaxation rapide, force de Lorentz, ecrantage de Debye `lambda_D -> 0`)
impose au schema explicite un `dt` du meme ordre que la raideur, donc impraticable. L'IMEX traite le
transport explicitement et la source raide implicitement (stable a `dt` fixe). La propriete
asymptotic-preserving (AP) garantit que, quand le petit parametre `eps` (= `lambda_D^2`, `1/omega_c`,
...) tend vers 0, le schema reste consistant et stable a `dt` fixe et capture la dynamique limite
(equilibre, quasi-neutralite) sans resoudre l'echelle raide.

**Formule / discretisation.** Sur `dU/dt = T(U) + S(U)` ou `S` porte la partie raide, un pas IMEX
d'Euler (forward-backward) traite `T` explicite et `S` implicite :

$$U^{n+1} = U^n + \Delta t\,T(U^n) + \Delta t\,S(U^{n+1}).$$

On le decompose en deux operateurs en place. D'abord le transport explicite produit le membre connu
$\tilde U = U^n + \Delta t\,T(U^n)$, puis le pas implicite resout $W = \tilde U + \Delta t\,S(W)$.
Quand `S` est une relaxation lineaire `S(U) = -(1/eps)(U - U_eq)`, le solve est analytique et
inconditionnellement stable ; la limite `eps -> 0` donne `W -> U_eq` (variete d'equilibre `S(U)=0`)
sans contrainte `dt < eps`. Le pas implicite scalaire (par cellule) est resolu par le residu de Newton

$$F(W) = W - \tilde U - \Delta t\,S(W) = 0,\qquad J = I - \Delta t\,\frac{\partial S}{\partial W},$$

exact en une iteration si `S` est lineaire en `U`, quadratique sinon. La jacobienne est formee par
differences finies (pas de jacobienne analytique a fournir cote modele).

**IMEX partiel.** Quand une seule sous-partie des variables est raide, on n'integre implicitement que
ces composantes. Le solve devient un forward-backward Euler par composante : les composantes explicites
avancent en Euler avant a l'etat d'entree, `W_e = U^n_e + dt S_e(U^n)`, puis les composantes implicites
sont resolues par Newton sur le sous-systeme reduit `n x n` (`n` = nombre d'implicites `<= N`), les
explicites figees a leur valeur avancee comme donnee connue. Le partitionnement vient soit du modele
(trait `is_implicit(c)`), soit d'un masque porte par le bloc (prioritaire sur le defaut modele), ce qui
permet de reutiliser le meme modele avec des traitements differents selon le bloc.

```
function imex_euler_step(U, dt, Texpl, Simpl):
    Texpl(U, dt)            # explicite en place : U <- U^n + dt*T(U^n) (membre connu)
    Simpl(U, dt)            # implicite en place : resout U <- W tel que W = U + dt*S(W)

# pas implicite par cellule (Newton local, IMEX partiel), N = Model::n_vars
function newton_source_solve(model, Un, aux, dt, iters, mask):
    impl <- liste des c dans [0,N) tels que is_implicit_component(mask, c)   # m = |impl| <= N
    W <- Un
    # (1) composantes explicites : Euler avant a l'etat d'entree
    if m < N:
        S_in <- model.source(Un, aux)
        for c not in impl: W[c] <- Un[c] + dt * S_in[c]
    # (2) composantes implicites : Newton sur le sous-systeme reduit m x m
    for it in 0..iters-1:
        S0 <- model.source(W, aux)
        for r in 0..m-1:                          # residu F = W - Un - dt*S(W)
            c <- impl[r];  F[r] <- W[c] - Un[c] - dt * S0[c]
        for cc in 0..m-1:                          # jacobienne par differences finies, colonne par colonne
            col <- impl[cc];  h <- 1e-7*|W[col]| + 1e-7
            Wp <- W;  Wp[col] += h;  Sp <- model.source(Wp, aux)
            for rr in 0..m-1:
                row <- impl[rr];  dSdW <- (Sp[row] - S0[row]) / h
                J[rr][cc] <- (row==col ? 1 : 0) - dt * dSdW       # I - dt*(dS/dW)
        solve_dense(J, F, delta, m)                # Gauss + pivot partiel, tableau fixe N, device-callable
        for r in 0..m-1: W[impl[r]] -= delta[r]
    return W

# stepper de bloc : pas implicite sur la source locale du modele, en place sur tout le MultiFab
function backward_euler_source(model, aux, U, dt, iters, mask):
    for chaque fab local de U:
        for_each_cell(box, BackwardEulerSourceKernel:
            Un <- load_state(U, i, j);  a <- load_aux(aux, i, j)
            W  <- newton_source_solve(model, Un, a, dt, iters, mask)
            U(i,j,:) <- W)
```

**Code.** [`include/adc/numerics/time/imex.hpp`](../include/adc/numerics/time/imex.hpp) :
`imex_euler_step(U, dt, Texpl, Simpl)` enchaine le transport explicite en place puis le solve source
implicite en place (deux callables `TransportStep` / `ImplicitSourceSolve`). Le pas implicite vit dans
[`include/adc/numerics/time/implicit_stepper.hpp`](../include/adc/numerics/time/implicit_stepper.hpp) :
`newton_source_solve<Model>` (Newton local par cellule, forward-backward Euler pour l'IMEX partiel),
`detail::solve_dense<N>` (resolution dense `n x n` par elimination de Gauss avec pivot partiel, tableau
fixe constexpr donc device-callable, aucune allocation), et `backward_euler_source<Model>` qui applique
le noyau `detail::BackwardEulerSourceKernel<Model>` via `for_each_cell` (foncteur nomme et non lambda
etendue, pour une emission device robuste depuis une TU externe). Le partitionnement implicite/explicite
passe par le concept `PartiallyImplicitModel` (trait `M::is_implicit(c)`), `model_is_implicit<Model>`
(defaut : tout implicite quand le trait est absent), le carrier POD `ImplicitMask<N>` (`active`, `flag[N]`,
porte par le bloc, passe par valeur sur le device) et `is_implicit_component<Model, N>` (masque actif
prioritaire sur le defaut modele). `ImplicitSourceStepper` (`iters = 2`) modele le concept
`ImplicitBlockStepper` et branche `backward_euler_source` sur le callback d'avancee implicite du
`SystemCoupler`, qui avance lui-meme les blocs explicites SSPRK et delegue les blocs implicites / IMEX.

**Contraintes / remarques.** Le pas implicite est inconditionnellement stable pour une relaxation
lineaire (la ou un simple point-fixe de Picard divergerait des que `dt * raideur > 1`, justement le
regime raide) ; il est exact en une iteration si `S` est lineaire en `U`, convergence quadratique sinon
(defaut `iters = 2`). La jacobienne par differences finies utilise un pas `h = 1e-7 |W_col| + 1e-7`.
Limites : `imex_euler_step` est d'ordre 1 en temps (forward-backward Euler) ; l'AP couvre la limite de
relaxation, pas la condensation des couplages potentiel-vitesse-Lorentz a `omega_c` eleve, qui releve de
la condensation de Schur (section 13). Masque inactif et modele sans trait `is_implicit` : tout est
implicite (backward-Euler plein), strictement bit-identique au comportement historique. Le transport d'un
bloc IMEX reste avance explicitement par le coeur. Validation :
`test_imex_ap` (propriete AP sur une source de relaxation lineaire raide),
`test_ap_limit` (limite AP quantifiee, balayage de la raideur sur 8 decades a `dt` fixe),
`test_imex_partial` (modele a 2 variables, une seule implicite),
`test_imex_transport` (le transport d'un bloc IMEX est bien avance explicitement).


---

## 6. Splitting d'operateurs : Lie et Strang

**Intuition.** Quand le RHS est une somme d'operateurs au comportement different (transport + source
raide + rotation cyclotron), on les applique en sequence plutot que simultanement : chaque
sous-operateur garde son propre integrateur, donc sa propre raideur, sans contaminer l'autre. Lie
(Godunov, ordre 1) enchaine les flots ; Strang (ordre 2) symetrise la sequence autour du flot central
pour annuler le terme d'erreur dominant.

**Formule / discretisation.** On decompose

$$\frac{\mathrm{d}U}{\mathrm{d}t} = T(U) + S(U)$$

en notant $\Phi^T_{\tau}$ et $\Phi^S_{\tau}$ les flots exacts (ou approches a l'ordre voulu) de
$\dot U = T(U)$ et $\dot U = S(U)$ sur un intervalle $\tau$. Le splitting de Lie applique l'un puis
l'autre sur le pas complet :

$$U^{n+1} = \Phi^S_{\Delta t}\big(\Phi^T_{\Delta t}(U^n)\big)$$

Le splitting de Strang encadre le flot de transport par deux demi-pas de source :

$$U^{n+1} = \Phi^S_{\Delta t/2}\Big(\Phi^T_{\Delta t}\big(\Phi^S_{\Delta t/2}(U^n)\big)\Big)$$

L'ordre se lit sur la formule de Baker-Campbell-Hausdorff. Le flot composite de Lie vaut
$\exp(\Delta t\,T)\exp(\Delta t\,S) = \exp\big(\Delta t (T+S) + \tfrac{\Delta t^2}{2}[T,S] + \dots\big)$ :
l'erreur par pas est $O(\Delta t^2)$, portee par le commutateur $[T,S] = TS - ST$, donc ordre 1 global.
La symetrisation de Strang annule le terme en $\Delta t^2$ : l'erreur par pas tombe a $O(\Delta t^3)$,
soit ordre 2 global. Strang est ordre 2 des que chaque sous-integrateur l'est lui-meme ; si $T$ et $S$
commutent ($[T,S]=0$) le splitting est exact a tout ordre. Le surcout de Strang sur Lie est un seul
demi-pas source de plus par macro-pas (deux $S(\Delta t/2)$ au lieu d'un $S(\Delta t)$).

```
function lie_step(U, dt, T, S):
    # T, S : callables (MultiFab&, Real) -> void, avancent leur sous-systeme en place
    T(U, dt)                 # transport sur le pas plein
    S(U, dt)                 # source sur le pas plein
    # U contient maintenant U^{n+1}, ordre 1

function strang_step(U, dt, T, S):
    S(U, 0.5 * dt)           # demi-pas source
    T(U, dt)                 # pas plein transport (flot central)
    S(U, 0.5 * dt)           # demi-pas source symetrique
    # U contient maintenant U^{n+1}, ordre 2 si S et T sont chacun >= ordre 2
```

**Code.** Les deux briques generiques sont dans
[`include/adc/numerics/time/splitting.hpp`](../include/adc/numerics/time/splitting.hpp) :
`lie_step(MultiFab& U, Real dt, TransportStep T, SourceStep S)` et
`strang_step(...)`. Les deux sont templees sur `TransportStep` / `SourceStep` : $T$ et $S$ sont des
callables `(MultiFab&, Real) -> void` qui avancent leur sous-systeme EN place, donc l'integrateur est
agnostique au contenu physique (pendant maison de `StrangSplitting` / `FractionalTime2OSplitting` de
muffin). L'orchestrateur de production les expose par `SplitScheme::Lie` / `Strang`
(`runtime/system_stepper.hpp` : `SystemStepper::step` pour Lie, `step_strang` pour Strang, #217), ou
la phase transport et la phase source (explicite, IMEX, ou etage condense par Schur, sections 5 et 13)
sont symetrisees a $\Delta t/2$ autour du transport central.

**Contraintes / remarques.** Strang ne donne l'ordre 2 que si chaque sous-pas est lui-meme au moins
ordre 2 : un $S$ ou $T$ ordre 1 plafonne le splitting a l'ordre 1, quelle que soit la symetrisation. En
couple hyperbolique-elliptique, la consistance impose de re-resoudre l'elliptique entre les demi-pas
source : sinon le second demi-pas $S(\Delta t/2)$ lit un $\phi$ perime (champ du demi-pas precedent) et
l'ordre 2 tombe (voir le compte d'appels au solve dans la validation). Le pas $\Delta t$ reste soumis a
la CFL du transport $T$ ; le splitting ne relache pas cette contrainte, il decouple seulement les
raideurs pour qu'une source raide soit traitee implicitement (IMEX / Schur) sans imposer son propre
$\Delta t$ minuscule au transport. Validation : `test_splitting` mesure l'ordre des briques
`lie_step` / `strang_step` sur un systeme lineaire 2x2 non commutant dont le flot exact est connu (Lie
ordre 1, Strang ordre 2 lus sur la pente). `test_strang_splitting` refait la meme mesure d'ordre sur le
vrai orchestrateur (`SystemStepper::step` vs `step_strang`), plus un compte d'appels au solve
elliptique qui verrouille la consistance $\phi$. Cote Python : `test_strang_split`.


---

## 7. Multirate : sous-cyclage, cadence, pas adaptatif

**Intuition.** Toutes les especes d'un systeme couple ne demandent pas le meme pas de temps.
Une espece raide (electrons) decoupe un macro-pas en plusieurs sous-pas ($\text{substeps}$) ; une
espece lente (gaz peu resolu) n'est avancee qu'une fois sur $M$ macro-pas (cadence, $\text{stride}$),
et rattrape alors $M$ pas en une seule avance. Les deux mecanismes sont orthogonaux et lus dans la
politique temporelle de chaque bloc ; le scheduler ne connait aucune physique.

**Formule / discretisation.** Chaque bloc porte une `TimePolicy<Method, Treatment, substeps, stride>`
dont le scheduler n'extrait que trois entiers (et le traitement, pour sauter les blocs prescrits).
Soit $\text{dt}$ le macro-pas, $n = \text{substeps}_b$, $m = \text{stride}_b$ pour le bloc $b$.

Cadence : le bloc $b$ est tenu (hold) tant que le macro-pas $k$ verifie $(k+1) \bmod m \neq 0$, puis
il rattrape en fin de fenetre avec un pas effectif

$$\Delta t^{\text{eff}}_b = m \, \text{dt}.$$

Sur $M$ macro-pas, le bloc avance $M/m$ fois d'un pas $m\,\text{dt}$ : son temps total reste $M\,\text{dt}$,
mais il n'est resolu que $M/m$ fois (le couplage est lache pour ce bloc). Le sous-cyclage decoupe ce
pas effectif en $n$ sous-pas egaux

$$h = \frac{\Delta t^{\text{eff}}_b}{n} = \frac{m \, \text{dt}}{n}.$$

Avec $m = 1$ et $n = 1$ on retrouve a l'identique l'avance d'un pas $\text{dt}$ a chaque macro-pas.

Le macro-pas peut etre choisi par le CFL via `step_cfl`. La condition de stabilite porte sur le
sous-pas reel $m\,\text{dt}/n \le \text{cfl}\, h_{\text{cell}} / w_b$, ce qui donne par bloc

$$\text{dt}_b = \frac{\text{cfl} \; h_{\text{cell}} \; \text{substeps}_b}{\text{stride}_b \; w_b},
\qquad \text{dt} = \min_{b \,\text{evolutif}} \text{dt}_b,$$

ou $h_{\text{cell}} = \min(dx, dy)$ en cartesien, $\min(dr, r_{\min}\, d\theta)$ en polaire (le pas
azimutal physique est minimal au rayon interieur), et $w_b$ est la vitesse d'onde max du bloc.

La variante `step_adaptive` fixe le macro-pas sur le bloc le plus lent,
$\Delta t = \text{cfl}\, h_{\text{cell}} / w_{\min}$, et sous-cycle chaque bloc plus rapide

$$n_b = \left\lceil \text{stride}_b \; \frac{w_b}{w_{\min}} \right\rceil$$

fois sur son pas effectif $\Delta t^{\text{eff}}_b = m\,\Delta t$ ; aux est fige sur le macro-pas
(couplage once-per-step).

```
function advance_subcycled(system, dt, macro_step, advance_block):
    for each block in system:                    # for_each_block, ordre stable
        if time_treatment(block) == Prescribed:
            continue                             # pilote par l'utilisateur, hors scheduler
        m = stride(block)
        if macro_step mod m != 0:
            continue                             # bloc lent : tenu ce macro-pas
        n = substeps(block)
        h = dt * m / n                           # pas effectif (catch-up) decoupe en n sous-pas
        for s in 0 .. n-1:
            advance_block(block, h, s, n)        # callable utilisateur, 1 sous-pas

# surcharge historique : macro_step = 0 -> stride toujours satisfait, tous les blocs avancent
function advance_subcycled(system, dt, advance_block):
    advance_subcycled(system, dt, 0, advance_block)

function step_cfl(cfl):                           # SystemDriver, choix du macro-pas par CFL
    solve_fields()                                # aux (phi, grad) a l'instant courant
    h_cell = polar ? min(dr, r_min*dtheta) : min(dx, dy)
    dt = +inf
    for each block b, evolutif:
        w   = max(max_wave_speed(b.U), 1e-30)     # all_reduce_max sous MPI
        dt  = min(dt, cfl * h_cell * substeps_b / (stride_b * w))
    if dt not finite: dt = cfl * h_cell / 1e-30   # tous geles : pas degenere
    for each block b, evolutif:
        if (macro_step+1) mod stride_b != 0: continue   # hold ; sinon rattrapage
        eff_dt = dt * stride_b
        advance_transport(b, eff_dt)              # substeps_b sous-pas internes
        run_source_stage(b, eff_dt)               # etage source Schur opt-in (no-op sinon)
    apply_couplings(dt); t += dt; macro_step += 1
    return dt

function step_adaptive(cfl):                       # macro-pas = pas du bloc le plus lent
    solve_fields()
    for each block b: w_b = b.evolve ? max_wave_speed(b.U) : 0
    w_min   = min over evolutifs of w_b           # 1e-30 si tous geles
    h_cell  = polar ? min(dr, r_min*dtheta) : min(dx, dy)
    macro_dt = cfl * h_cell / w_min
    for each block b, evolutif:
        if (macro_step+1) mod stride_b != 0: continue
        n      = max(1, ceil(stride_b * w_b / w_min))   # sous-cycles pour rester stable
        eff_dt = macro_dt * stride_b
        advance_transport_n(b, eff_dt, n)
        run_source_stage(b, eff_dt)
    apply_couplings(macro_dt); t += macro_dt; macro_step += 1
    return macro_dt
```

**Code.** Le squelette est [`numerics/time/scheduler.hpp`](../include/adc/numerics/time/scheduler.hpp),
fonction `advance_subcycled` (deux surcharges : avec et sans `macro_step`). Elle lit
`block_substeps_v`, `block_stride_v` et `block_time_treatment_v`, alias de `TimePolicyTraits`
defini dans [`numerics/time/time_integrator.hpp`](../include/adc/numerics/time/time_integrator.hpp)
(`TimePolicy<Method, Treatment, substeps, stride>`, alias `ExplicitTime` / `ImplicitTime` /
`IMEXTime` / `PrescribedTime`). Un bloc `TimeTreatment::Prescribed` est saute (la garde
`!= Prescribed`). Le choix du pas vit dans
[`runtime/system_stepper.hpp`](../include/adc/runtime/system_stepper.hpp) : `step_cfl`,
`step_adaptive`, et l'helper `stride_due(macro_step, stride)` qui materialise la fin de fenetre
$(k+1)\bmod m = 0$. La vitesse $w_b$ vient de `max_wave_speed_mf`
([`numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp)), collective
`all_reduce_max` sous MPI pour que tous les rangs choisissent le meme $\text{dt}$.

**Contraintes / remarques.** `step_cfl` est substeps-aware depuis #121 : la formule
$\text{dt} = \text{cfl}\,h\,\text{substeps}/(\text{stride}\,w)$ rend, pour $\text{substeps}_b > 1$, un
pas $\text{substeps}_b$ fois plus grand que l'ancienne formule $\text{dt} = \text{cfl}\,h/(\text{stride}\,w)$.
La parite bit-identique avec l'historique ne tient donc que pour $\text{substeps} = 1$ (a tout
stride) ; pour rejouer un run calibre sur l'ancienne formule, passer le $\text{dt}$ historique
explicite a `step(dt)`, pas `step_cfl`. Sous MPI, l'absence d'`all_reduce_max` desynchroniserait les
rangs (chacun verrait le max de ses seules boites) et ferait diverger la simulation. La semantique
stride est hold-then-catch-up : le bloc lent est couple de facon lache, ce qui est un choix assume
(le gaz n'est pas resolu a chaque pas). Tests : `test_multirate_stride` (espece lente avancee une
fois sur $N$), `test_adaptive_multirate` (`step_adaptive`, macro-pas fixe par l'espece la plus
contraignante), `test_cfl_dt` (`step_cfl` multi-especes).

## 8. Terme parabolique : diffusion en flux de face

**Intuition.** Un terme parabolique $+\nu\,\Delta U$ (diffusion, viscosite scalaire isotrope) est la
divergence d'un flux Fickien $F_{\text{diff}} = -\nu\,\nabla U$. L'ecrire comme un flux de face plutot
qu'un Laplacien direct le rend compatible AMR : le reflux le voit et le corrige a l'interface
fin-grossier exactement comme un flux hyperbolique, donc la diffusion reste conservative aux jonctions
de niveaux.

**Formule / discretisation.** Le flux Fickien continu $F_{\text{diff}} = -\nu\,\nabla U$ s'ajoute au
flux numerique hyperbolique avant la divergence. En flux de face (gradient centre au face, valeurs de
cellule, pas du niveau $h$) :

$$F^{x}_{i+1/2,j} = -\nu\,\frac{U_{i+1,j} - U_{i,j}}{dx}, \qquad
  F^{y}_{i,j+1/2} = -\nu\,\frac{U_{i,j+1} - U_{i,j}}{dy}.$$

La divergence $-\big(F^{x}_{i+1/2} - F^{x}_{i-1/2}\big)/dx - \big(F^{y}_{j+1/2} - F^{y}_{j-1/2}\big)/dy$
redonne exactement le Laplacien a 5 points :

$$+\nu\,\Delta_h U_{i,j} = \nu\left(
  \frac{U_{i+1,j} - 2U_{i,j} + U_{i-1,j}}{dx^2}
+ \frac{U_{i,j+1} - 2U_{i,j} + U_{i,j-1}}{dy^2}\right),$$

ajoute composante par composante au residu $R = -\mathrm{div}\hat F + S$. Le coeur `assemble_rhs`
ecrit directement ce stencil a 5 points (chemin sans AMR) ; `compute_face_fluxes` produit la forme en
flux de face (chemin reflux AMR). Les deux donnent un residu bit-identique a la machine.

```
# coeur : assemble_rhs, terme additif au residu (5 points), garde par DiffusiveModel
function diffusive_residual_term(model, u, i, j, dx, dy):
    if not DiffusiveModel(model):                 # if constexpr : zero codegen sinon
        return                                     # chemin hyperbolique strictement intouche
    nu   = model.diffusivity()
    idx2 = 1/(dx*dx);  idy2 = 1/(dy*dy)
    for c in 0 .. n_vars-1:
        lap = (u(i+1,j,c) - 2*u(i,j,c) + u(i-1,j,c)) * idx2
            + (u(i,j+1,c) - 2*u(i,j,c) + u(i,j-1,c)) * idy2
        r(i,j,c) += nu * lap                       # +nu Lap(U)

# AMR : compute_face_fluxes, flux de face Fickien ajoute au flux hyperbolique
function face_flux_x(model, u, aux, i, j, dx):     # face entre (i-1,j) et (i,j)
    L = reconstruct(model, u, i-1, j, dir=0, +1)   # etats reconstruits
    R = reconstruct(model, u, i,   j, dir=0, -1)
    F = numerical_flux(model, L, aux(i-1,j), R, aux(i,j), dir=0)   # hyperbolique
    if DiffusiveModel(model):
        nu = model.diffusivity()
        for c in 0 .. n_vars-1:
            F[c] += -nu * (u(i,j,c) - u(i-1,j,c)) / dx     # flux Fickien centre au face
    fx(i,j,:) = F                                  # le reflux AMR voit ce flux -> conservatif
```

**Code.** Le contrat est le concept `DiffusiveModel` dans
[`numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp) : un modele le
satisfait si et seulement si `m.diffusivity()` retourne un `Real` ($\nu \ge 0$). Le terme a 5 points
est ajoute dans `detail::AssembleRhsKernel::operator()` (appele par `assemble_rhs`) sous
`if constexpr (DiffusiveModel<Model>)`. La forme en flux de face vit dans
`detail::FaceFluxXKernel` / `detail::FaceFluxYKernel` (appeles par `compute_face_fluxes`), meme garde.
Les deux noyaux sont des foncteurs nommes device-clean (`ADC_HD`).

**Contraintes / remarques.** Invariant central : un modele qui n'expose pas `diffusivity()` ne change
pas d'un bit, le `if constexpr` etant faux il n'y a aucun codegen supplementaire (chemin hyperbolique
strictement inchange). Les arguments `dx`, `dy` de `compute_face_fluxes` valent 0 par defaut et ne sont
lus que par la branche diffusive, donc un modele non diffusif n'est jamais affecte. Le pas explicite
sur un terme parabolique impose la contrainte de stabilite diffusive
$\nu\,\Delta t \le \tfrac{1}{2}\,(dx^{-2} + dy^{-2})^{-1}$ (plus restrictive en $h^2$ que le CFL
hyperbolique a $h$), non geree par `step_cfl` qui ne pese que la vitesse d'onde : a diffusion
dominante, fixer $\text{dt}$ explicitement. Limite connue : `SourceFreeModel` (demi-pas explicite
IMEX) n'expose pas `diffusivity()`, donc un bloc IMEX diffusif perdrait son flux Fickien dans le
demi-pas explicite (raffinement separe) ; et le chemin masque `assemble_rhs_masked` ne masque pas le
Laplacien. Tests : `test_diffusion` (le $+\nu\,\Delta U$ du coeur via la divergence du flux Fickien),
`test_amr_diffusion` (la diffusion en flux de face traverse correctement le reflux AMR).


---

## 9. Elliptique : multigrille geometrique

**Intuition.** Le lisseur Gauss-Seidel tue vite les hautes frequences de l'erreur mais rampe sur
les basses. La multigrille restreint l'erreur basse frequence sur des grilles plus grossieres (ou
elle redevient haute frequence), la lisse, et la prolonge. Cout $O(N)$ par V-cycle, nombre de
cycles quasi independant du maillage.

**Formule / discretisation.** Operateur 5 points sur $\mathrm{lap}(\phi) = f$ (cas isotrope
$\epsilon = 1$, $\kappa = 0$) :

$$(\mathrm{lap}\,\phi)_{ij} = \frac{\phi_{i+1,j} - 2\phi_{ij} + \phi_{i-1,j}}{\Delta x^2}
                            + \frac{\phi_{i,j+1} - 2\phi_{ij} + \phi_{i,j-1}}{\Delta y^2}$$

Lisseur Gauss-Seidel rouge-noir : une couleur $c \in \{0,1\}$ par balayage, la cellule
$(i,j)$ avec $(i+j) \bmod 2 = c$ est mise a jour depuis ses voisins (de l'autre couleur, donc deja
fige sur ce balayage) :

$$\phi_{ij} \leftarrow \frac{\mathrm{off}_{ij} - f_{ij}}{\mathrm{diag}},
\quad \mathrm{off}_{ij} = \frac{\phi_{i\pm1,j}}{\Delta x^2} + \frac{\phi_{i,j\pm1}}{\Delta y^2},
\quad \mathrm{diag} = \frac{2}{\Delta x^2} + \frac{2}{\Delta y^2}$$

V-cycle : $\nu_1$ balayages de pre-lissage, residu $r = f - \mathrm{lap}\,\phi$, restriction de
$r$ par moyenne $2\times2$ (`average_down`) sur la grille deux fois plus grossiere, resolution
recursive de l'equation de correction $\mathrm{lap}(e) = r$ a conditions homogenes, prolongation
de $e$ (`interpolate`) ajoutee a $\phi$, $\nu_2$ balayages de post-lissage. Au niveau le plus
grossier, `nbottom` balayages tiennent lieu de resolution exacte (bottom solve). Defauts :
$\nu_1 = \nu_2 = 2$, `nbottom = 50`, `min_coarse = 2`.

```
function vcycle_rec(level l, bc):
    L = lev_[l]
    gs_smooth(L.phi, L.rhs, nu1, bc)              # pre-lissage : nu1 balayages rouge-noir
    if l est le plus grossier:
        gs_smooth(L.phi, L.rhs, nbottom, bc)      # bottom solve (longue serie de balayages)
        if masque: zero_conductor(L.phi)          # refige phi=0 dans le conducteur
        return
    poisson_residual(L.phi, L.rhs, -> L.res, bc)  # r = f - lap(phi) (porte aussi termes croises)
    average_down(L.res, C.rhs, ratio=2)           # restriction du residu (moyenne 2x2)
    C.phi = 0                                       # correction a CL homogenes
    vcycle_rec(l+1, homogeneous(bc))              # recursion grossiere
    corr = interpolate(C.phi, ratio=2)            # prolongation de la correction
    L.phi += corr                                  # saxpy
    if masque: zero_conductor(L.phi)
    gs_smooth(L.phi, L.rhs, nu2, bc)              # post-lissage

function solve(rel_tol, max_cycles):
    r0 = current_residual()                        # norm_inf(f - lap(phi)), all_reduce_max
    if r0 <= 0: return 0
    for c in 1..max_cycles:
        vcycle()                                   # warm-start : phi conserve entre appels
        if current_residual() <= rel_tol * r0: return c
    return max_cycles
```

La hierarchie est batie en grossissant le domaine par 2 jusqu'a `min_coarse`, mais on stoppe si
une boite ne se coarsen pas proprement : le test `b.coarsen(2).refine(2) == b` caracterise les
boites alignees et de taille paire. Sur un domaine multi-box (`max_grid_size < n`), les boites
retrecissent par 2 a chaque niveau et finiraient a $1\times1$ ; `coarsen(ba,2)` ferait alors
retomber plusieurs boites fines distinctes sur la meme cellule grossiere (BoxArray degenere), et
`average_down` lirait hors des bornes d'un fab de 1 cellule. En serie le tas est stable, sous MPI
il est remue et la lecture devient erratique (ecart ponctuel jusqu'au blow-up). Le break garde le
niveau courant comme grille la plus grossiere ; mono-box et multi-box non degenere ne franchissent
jamais ce test, hierarchie et resultat strictement inchanges.

Le balayage rouge-noir rend chaque couleur independante des donnees (parallelisable). Entre couleurs
et avant le residu, `device_fence()` + `fill_ghosts` synchronisent le device et remplissent les
halos ; `current_residual` reduit la norme infinie par `all_reduce_max` (obligatoire pour un
grossier multi-box reparti, sinon le critere d'arret se declenche a des iterations differentes par
rang et desynchronise les flux MPI). Le mode `replicated` replique chaque niveau sur tous les
rangs (V-cycle par-fab sans communication), ce qu'attend le coupleur AMR (niveau 0 replique).

**Code.** [`numerics/elliptic/geometric_mg.hpp`](../include/adc/numerics/elliptic/geometric_mg.hpp) :
`GeometricMG` modele le concept `EllipticSolver` (`rhs()`, `phi()`, `solve()`, `residual()`) ;
`vcycle_rec` est la recursion, `solve(rel_tol, max_cycles)` itere les cycles avec warm-start (`phi`
conserve entre appels, 1-2 V-cycles en regime etabli). Le Laplacien 5 points et le lisseur sont
les briques partagees de [`numerics/elliptic/poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp)
(`poisson_residual`, `gs_smooth` -> `gs_rb_sweep` -> `detail::gs_color`, foncteurs nommes ADC_HD
device-clean). Restriction / prolongation reutilisent les operateurs de transfert AMR `average_down`
/ `interpolate` de [`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp). `solve_robust`
ajoute un garde-fou anti-divergence (cf. ci-dessous).

**Contraintes / remarques.** Entierement on-device (le V-cycle passe par `for_each_cell`),
AMR-compatible, accepte tout `n`. Pas de contrainte CFL (solveur stationnaire), mais le V-cycle
GS-5-points suppose un operateur a diagonale dominante : il reste contractant pour un operateur
symetrique defini positif (Poisson, $\epsilon > 0$, $\kappa \ge 0$), et peut diverger sur un
operateur fortement non symetrique (termes croises, cf. sections 11 et 12). Au bord embedded a
haute resolution, le coarsening non-Galerkin et le masque re-evalue par niveau rendent parfois le
cycle non contractant (rayon spectral $> 1$) : `solve_robust` detecte la vraie divergence (residu
final $>$ residu initial), durcit le lissage de facon locale au solve ($\nu$ double jusqu'a 64) et
repart a froid ($\phi = 0$), strictement bit-identique quand le solveur converge ou stagne deja.
**Validation.** `test_geometric_mg` (convergence rapide quasi independante du maillage sur
solutions manufacturees), `test_poisson_convergence` (ordre 2 quantitatif du Laplacien 5 points),
`test_solve_robust` (le garde-fou anti-divergence).

## 10. Elliptique : Poisson spectral (FFT), mono-rang et distribue

**Intuition.** Sur un domaine periodique a coefficient constant, le Laplacien discret est diagonal
en Fourier : une transformee directe, une division mode par mode, une transformee inverse resolvent
Poisson exactement (au residu machine), sans iteration. Bien moins cher que la multigrille quand
l'elliptique domine le run.

**Formule / discretisation.** Le solveur inverse le meme Laplacien 5 points que `GeometricMG`,
dont la valeur propre du mode $(k_x, k_y)$ est

$$\lambda(k_x, k_y) = \frac{2\cos(2\pi k_x / N_x) - 2}{\Delta x^2}
                     + \frac{2\cos(2\pi k_y / N_y) - 2}{\Delta y^2}$$

et non $-(k_x^2 + k_y^2)$ (le symbole exact du stencil discret, pas du Laplacien continu). La
resolution est $\hat\phi(k) = \hat f(k) / \lambda(k)$, avec le mode $k = 0$ fige a 0 (jauge : $\phi$
de moyenne nulle ; le second membre doit donc etre a moyenne nulle, sinon $\phi$ derive).

```
function solve():                                  # PoissonFFTSolver, boite unique
    rho = aplatir rhs en tableau N_x * N_y (row-major)
    fft_.solve(rho -> phil)                         # FFT directe, /lambda(k), k=0 -> 0, FFT inverse
    phi = re-empaqueter phil dans le fab

function solve():                                  # DistributedFFTSolver, FFT par bandes
    rho = aplatir la bande locale [0..Nx-1] x [y0..y0+nyl-1]
    fft_.solve(rho -> phil)                         # transposee parallele (MPI_Alltoall) interne
    phi = re-empaqueter la bande locale
```

**Code.** [`numerics/elliptic/poisson_fft_solver.hpp`](../include/adc/numerics/elliptic/poisson_fft_solver.hpp) :
`PoissonFFTSolver` (mono-rang, boite unique) et `DistributedFFTSolver` (FFT distribuee par bandes /
slabs, 1 box par rang, transposee `MPI_Alltoall` interne a `PoissonFFT`). Les deux modelent le meme
concept `EllipticSolver` (`static_assert`) que la multigrille, donc le coupleur est generique sur le
backend (`Coupler<Model, PoissonFFTSolver>` interchangeable avec `GeometricMG`). Le residu reutilise
l'operateur canonique `poisson_residual` de
[`poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp) ; la variante
distribuee fait un `fill_boundary` (halos inter-bandes) avant la mesure et reduit par
`all_reduce_max`. Le coeur FFT vit dans `poisson_fft.hpp` (un correctif gere $n$ non puissance de 2).

**Contraintes / remarques.** Le FFT exige des CL periodiques et un coefficient constant : ni
$\epsilon(x)$, ni masque embedded, ni termes croises. Le mode $k = 0$ doit etre fixe (second membre
a moyenne nulle), sinon $\phi$ derive. `PoissonFFTSolver` leve un garde-fou dur (actif en Release,
pas un simple `assert`) si `n_ranks() != 1` ou `ba.size() != 1` : sous une `DistributionMapping`
multi-rang certains rangs n'auraient aucune box locale et `solve()` dereferencerait `fab(0)`
inexistant (SIGSEGV) ; le message renvoie vers `DistributedFFTSolver` ou `geometric_mg`. La variante
distribuee impose $N_y$ divisible par `n_ranks()` et $N_x, N_y$ puissances de 2.
**Validation.** `test_poisson_fft` (non-regression, taille $n$ non puissance de 2) ; sous MPI
`test_mpi_fft_distributed` (FFT par bandes). `test_elliptic_operator` applique le meme operateur
canonique `poisson_residual` aux solutions MG et FFT : residus a l'arrondi (`~1e-14`) et solutions
identiques a `~1e-16`, donc les deux inversent prouvablement le meme Laplacien discret.

## 11. Elliptique etendu : eps(x), Helmholtz/ecrante, anisotrope

**Intuition.** Le meme operateur multigrille couvre trois generalisations du Laplacien, toutes
opt-in et bit-identiques au chemin historique quand on ne les active pas (le pointeur de coefficient
correspondant reste `nullptr`) :

- **permittivite variable** $\mathrm{div}(\epsilon(x)\,\mathrm{grad}\,\phi) = f$ : chaque face porte
  la moyenne harmonique des deux centres adjacents du champ $\epsilon$ ;
- **operateur ecrante / Helmholtz** $\mathrm{div}(\epsilon\,\mathrm{grad}\,\phi) - \kappa\phi = f$ :
  un terme de reaction $\kappa \ge 0$ (ecrantage de Debye $\kappa = 1/\lambda_D^2$), diagonal, qui
  rend l'operateur plus diagonalement dominant (la multigrille converge au moins aussi bien) ;
- **permittivite anisotrope** $\mathrm{div}(\mathrm{diag}(\epsilon_x, \epsilon_y)\,\mathrm{grad}\,\phi) = f$ :
  les faces normales a $x$ lisent $\epsilon_x$, les faces normales a $y$ lisent $\epsilon_y$ (milieu
  tensoriel diagonal).

**Formule / discretisation.** Permittivite de face par moyenne harmonique (continuite du flux
normal a une interface, resistances en serie, correct meme pour un $\epsilon$ discontinu) :

$$\epsilon_{i+1/2,j} = \frac{2\,\epsilon_{ij}\,\epsilon_{i+1,j}}{\epsilon_{ij} + \epsilon_{i+1,j}}$$

L'operateur discret 5 points a coefficient de face variable, avec reaction, sur la cellule $(i,j)$ :

$$L\phi_{ij} = w^x_+\phi_{i+1,j} + w^x_-\phi_{i-1,j} + w^y_+\phi_{i,j+1} + w^y_-\phi_{i,j-1}
            - (w^x_+ + w^x_- + w^y_+ + w^y_-)\phi_{ij} - \kappa_{ij}\phi_{ij}$$

avec $w^x_\pm = \epsilon^x_{i\pm1/2,j} / \Delta x^2$ (champ $\epsilon_x$) et
$w^y_\pm = \epsilon^y_{i,j\pm1/2} / \Delta y^2$ (champ $\epsilon_y$ ; en isotrope $\epsilon_y$ pointe
sur le meme champ que $\epsilon_x$). Le lisseur GS gagne $+\kappa_{ij}$ a sa diagonale
($\kappa \ge 0$ => plus dominant). Combinaison cut-cell + $\epsilon$ : chaque poids de face
Shortley-Weller $w_{\bullet}$ est multiplie par sa permittivite de face, la diagonale reste la
somme des poids de face.

```
function ApplyLaplacianKernel(i, j):               # L = div(eps grad phi) - kappa phi, foncteur ADC_HD
    if he (eps actif):
        ec  = eps_x(i,j); ecy = eps_y(i,j)         # eps_y == eps_x en isotrope
        exm = harmonic(ec,  eps_x(i-1,j)); exp = harmonic(ec,  eps_x(i+1,j))
        eym = harmonic(ecy, eps_y(i,j-1)); eyp = harmonic(ecy, eps_y(i,j+1))
        if hc (cut-cell):  wxm,wxp,wym,wyp = coef[0..3] * (exm,exp,eym,eyp)   # poids SW * eps_face
        else:              wxm,wxp = exm,exp / dx^2 ;  wym,wyp = eym,eyp / dy^2
        L(i,j) = wxp*p(i+1,j)+wxm*p(i-1,j)+wyp*p(i,j+1)+wym*p(i,j-1) - (wxm+wxp+wym+wyp)*p(i,j)
    else if hc:  L(i,j) = coef[1]*p(i+1,j)+coef[0]*p(i-1,j)+coef[3]*p(i,j+1)+coef[2]*p(i,j-1)-coef[4]*p(i,j)
    else:        L(i,j) = (p(i+1,j)-2p(i,j)+p(i-1,j))/dx^2 + (p(i,j+1)-2p(i,j)+p(i,j-1))/dy^2
    if hxy or hyx:  L(i,j) += cross_div(...)        # tenseur plein (section 12) : flux croises additifs
    if hk (kappa actif):  L(i,j) -= kappa(i,j) * p(i,j)
```

**Code.** [`numerics/elliptic/geometric_mg.hpp`](../include/adc/numerics/elliptic/geometric_mg.hpp) :
`GeometricMG::set_epsilon(eps_fn | eps_fine)`, `set_reaction(kappa_fn | kappa_fine)`,
`set_epsilon_anisotropic(eps_x, eps_y)`. Chaque champ existe en deux surcharges : analytique
(`std::function`, evaluee par niveau sur toute la hierarchie -> permittivite exacte au grossier,
ordre 2 preserve) et deja-discretise (`MultiFab` du niveau fin, copie composante 0 par
`detail::CopyComp0Kernel` puis restreint par `average_down`, point d'entree pour le cablage depuis
`System`). Le terme $\kappa$ (0 ghost, diagonal), les champs $\epsilon$ / $\epsilon_y$ (1 ghost,
ghosts remplis par `eps_bc` : periodique conserve, bord physique en extrapolation gradient-nul)
vivent dans les `for_each_cell` ADC_HD du smoother, du residu et de l'apply
([`poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp) :
`ApplyLaplacianKernel`, `PoissonResidualKernel`, `GsColorKernel`, `eps_harmonic`) -> device. Les
pointeurs de coefficient du niveau fin sont aussi exposes (`op_eps`, `op_kappa`, `op_eps_y`, ...)
pour que le solveur de Krylov reutilise un operateur coherent avec le residu MG.

**Contraintes / remarques.** Les trois extensions sont composables : $\epsilon(x)$ et $\kappa(x)$
ensemble, ou $\mathrm{diag}(\epsilon_x, \epsilon_y)$ avec $\kappa$. Donner $\epsilon_x \equiv \epsilon_y$ redonne l'isotrope ; ne pas appeler `set_reaction` redonne Poisson pur ; aucun appel =>
chemin historique strictement bit-identique. Le choix harmonique (et non arithmetique) pour la face
preserve la continuite du flux normal a un saut de milieu et reste d'ordre 2 pour un $\epsilon$
lisse. L'echantillonnage par niveau (au lieu de restreindre depuis le fin) donne le coefficient
exact a chaque resolution grossiere, ce qui conserve l'ordre 2 du V-cycle.
**Validation.** `test_variable_epsilon` ($\epsilon(x)$, MMS ordre 2), `test_screened_poisson`
(Helmholtz / ecrante, MMS ordre 2), `test_anisotropic_epsilon` (anisotrope $\epsilon_x \neq \epsilon_y$, MMS ordre 2). Les trois chemins sont aussi exerces cote Python (`test_poisson_eps`,
`test_poisson_screened`, `test_poisson_eps_aniso`) et valides bit-identiques sur GH200
(cf. GPU_RUNTIME_PORT.md, round 2).


---

## 12. Elliptique a tenseur plein : Krylov matrice-libre (BiCGStab)

**Intuition.** Quand l'operateur elliptique porte des termes croises $A_{xy} \neq A_{yx}$ (operateur non auto-adjoint, par exemple la rotation $B^{-1}$ issue de la condensation de Schur), la multigrille geometrique seule, dont le lisseur Gauss-Seidel suppose un operateur auto-adjoint, stagne ou diverge. Il faut un solveur de Krylov non symetrique, preconditionne par le V-cycle MG applique a la partie symetrique de l'operateur.

**Formule / discretisation.** On resout $A\,\phi = f$ avec, dans la convention de [`poisson_operator.hpp`](../include/adc/numerics/elliptic/poisson_operator.hpp) et de `GeometricMG`,

$$L_{\mathrm{int}}(\phi) = \mathrm{div}(A\,\nabla\phi) - \kappa\,\phi, \qquad A = \begin{pmatrix} A_{xx} & A_{xy} \\ A_{yx} & A_{yy}\end{pmatrix},$$

la matvec etant `apply_laplacian` (calcul exact de $L_{\mathrm{int}}$) et le residu $r = f - L_{\mathrm{int}}(\phi)$, bit-coherent avec `poisson_residual`. BiCGStab est matrice-libre : aucune matrice n'est assemblee, seul le produit $A\,d$ est requis, applique par `for_each_cell`. Le preconditionneur est $M^{-1} =$ ($N$ V-cycles de `GeometricMG`) sur le bloc diagonal symetrique (termes croises $A_{xy}/A_{yx}$ largues). La partie antisymetrique etant en $O(\theta^2 dt^2 \alpha)$, petite a CFL source raisonnable, le preconditionneur symetrique capture l'essentiel du spectre.

Choix de BiCGStab et non gmres : il gere le non symetrique sans parametre de redemarrage ni base de Krylov croissante a stocker. L'empreinte memoire est fixe (les MultiFab $r, \hat r, p, v, s, t$ a zero fantome, plus les preconditionnes $\hat p, \hat s$ a un fantome).

Un point delicat est le traitement des conditions de Dirichlet inhomogenes. Le ghost de bord vaut $2v - \phi_{\mathrm{int}}$, donc le stencil des cellules de bord recoit un terme constant $c_{bc} = \mathrm{apply\_operator}(0)$. L'operateur brut est donc affine : $L_{\mathrm{aff}}(\phi) = L_{\mathrm{lin}}(\phi) + c_{bc}$. Pour le residu vrai $r_0$ on garde l'operateur affine (la donnee de Dirichlet s'y replie exactement, ce qu'on veut). Mais les matvec en boucle agissent sur des directions de correction $\hat p = M^{-1}p$, $\hat s = M^{-1}s$ : elles doivent etre lineaires, sinon le terme constant injecte a chaque produit fait osciller ou diverger le residu. On retranche donc $c_{bc}$ (matvec) et $d_{bc} = \mathrm{precond\_raw}(0)$ (preconditionneur), calcules une fois par solve dans `prepare_solve`. Quand la CL Dirichlet est nulle, $c_{bc} = d_{bc} = 0$ et le chemin redevient bit-identique a l'historique.

```
function TensorKrylovSolver.solve(rel_tol, max_iters):
    prepare_solve()                          # c_bc = apply_op(0), d_bc = precond_raw(0) si CL inhomogene
    v   <- apply_operator(phi)               # operateur affine pour le residu vrai
    r   <- rhs - v                           # r0, warm start respecte (phi entrant = depart)
    norm0 <- ||rhs||_2  (sinon 1 si rhs nul) # base relative, reduction MPI collective
    if ||r|| <= rel_tol * norm0: return converged
    rhat <- r                                # vecteur fantome fige de BiCGStab
    p, v <- 0 ;  rho_prev, alpha, omega <- 1
    for k = 1 .. max_iters:
        rho <- dot(rhat, r)                  # collectif (all_reduce, tous rangs)
        if |rho| ~ 0 or |omega| ~ 0: return best_effort     # garde-fou rupture
        beta <- (rho / rho_prev) * (alpha / omega)
        p   <- r + beta * (p - omega * v)
        phat <- M^{-1} p                     # N V-cycles MG sur la partie symetrique, CL homogenes
        v   <- apply_operator_lin(phat)      # matvec lineaire (phat = direction)
        alpha <- rho / dot(rhat, v)          # dot collectif ; garde-fou si ~ 0
        s   <- r - alpha * v
        phi <- phi + alpha * phat            # correction partielle (tampon avant test sur ||s||)
        if ||s|| <= rel_tol * norm0: return converged        # convergence a mi-iteration
        shat <- M^{-1} s
        t   <- apply_operator_lin(shat)
        omega <- dot(t, s) / dot(t, t)       # 0 si dot(t,t) ~ 0
        phi <- phi + omega * shat
        r   <- s - omega * t
        if ||r|| <= rel_tol * norm0: return converged
        rho_prev <- rho
    return best_effort                       # max_iters atteint, converged = false

function apply_operator(in):                 # matvec matrice-libre affine
    device_fence() ; fill_ghosts(in, bc_entiere)
    out <- apply_laplacian(in, eps_x, eps_y, a_xy, a_yx, kappa)   # = L_int(in)
    if mask present: mask_zero(out)          # L_int = 0 sur cellules conductrices (Dirichlet phi=0)
    return out

function apply_operator_lin(in):             # matvec lineaire (directions de correction)
    out <- apply_operator(in)
    if has_op_offset: out <- out - c_bc      # retranche la part inhomogene de bord
    return out

function precond_raw(in):                     # V-cycle brut (affine si CL Dirichlet != 0)
    precond.rhs <- in ; precond.phi <- 0
    repeat N: precond.vcycle()
    return precond.phi

function apply_precond(in):                   # M^{-1} a CL homogenes
    out <- precond_raw(in)
    if has_bc_offset: out <- out - d_bc      # M^{-1} in = precond_raw(in) - precond_raw(0)
    return out
```

**Code.** [`numerics/elliptic/krylov_solver.hpp`](../include/adc/numerics/elliptic/krylov_solver.hpp) : classe `TensorKrylovSolver`, methodes `solve(rel_tol, max_iters)` (renvoie un `KrylovResult` : `iters`, `rel_residual`, `converged`), `apply_operator` / `apply_operator_lin` (matvec affine et lineaire), `precond_raw` / `apply_precond` (V-cycle brut et a CL homogenes), `prepare_solve` (calcul unique des offsets $c_{bc}$, $d_{bc}$), `residual` (residu L2 global courant). Le constructeur prend deux `GeometricMG` distincts : `op` porte l'operateur plein (matvec + stockage de $\phi$/$rhs$), `precond` porte la partie symetrique (memes eps mais `set_cross_terms` non appele). Ils doivent etre des objets separes, impose par `assert(&op != &precond)` : `apply_precond` ecrase `precond.rhs()`/`precond.phi()` a chaque iteration, et les confondre ecraserait l'iterate et le second membre du solve.

**Contraintes / remarques.** Methode iterative, pas de CFL propre ; le cout depend du conditionnement du complement de Schur, d'ou le preconditionnement par MG symetrique (1 a 2 V-cycles, parametre `n_precond_vcycles`). Garde-fous de rupture BiCGStab : si $|\rho|$, $|\omega|$ ou $\mathrm{dot}(\hat r, v)$ tombent sous `kTiny` $= 10^{-300}$, le solve rend le meilleur effort courant sans diviser par zero. Device/MPI : foncteurs nommes uniquement (`mf_arith` : `saxpy`/`lincomb`/`dot`, `apply_laplacian`, V-cycle MG), tous device-clean. Les produits scalaires `dot` sont collectifs (`all_reduce_sum`) et appeles sur tous les rangs, y compris un rang sans box (`local_size() == 0`) : aucun court-circuit, donc pas d'interblocage MPI ni de desynchronisation du critere d'arret. Limitation connue : le preconditionneur symetrique perd en efficacite quand la part antisymetrique grossit (CFL source elevee, $\omega_c$ grand) ; le nombre d'iterations augmente alors.

Validation : `test_krylov_solver`. Cas (A) sur le Laplacien canonique ($A_{xy} = A_{yx} = 0$, $A = I$, $\kappa = 0$), BiCGStab converge vers la meme solution que `GeometricMG` a la tolerance. Cas (B) sur un operateur a termes croises non triviaux, BiCGStab converge la ou le V-cycle MG seul stagne ($c = 0.1$ a $0.4$) ou diverge ($c = 0.7$). Sous MPI, le nombre d'iterations et la convergence sont invariants au nombre de rangs (critere d'arret reduit par `all_reduce`).

## 13. Source implicite condensee : condensation de Schur

**Intuition.** Une source raide qui couple potentiel, vitesse et force de Lorentz (diocotron a $\omega_c$ eleve) ne se traite pas composante par composante : la rotation cyclotron couple les deux composantes de vitesse, et le potentiel reagit au deplacement de charge. On theta-discretise la source implicite, on elimine algebriquement la vitesse via l'inverse ferme $B^{-1}$ de la rotation 2x2, ce qui ne laisse qu'une elliptique sur le seul potentiel $\phi^{n+\theta}$ (complement de Schur), puis on reconstruit la vitesse.

**Formule / discretisation.** L'eliminateur de Lorentz code la rotation-dilatation

$$B = \begin{pmatrix} 1 & -w \\ w & 1 \end{pmatrix}, \qquad B^{-1} = \frac{1}{\det B}\begin{pmatrix} 1 & w \\ -w & 1\end{pmatrix}, \qquad w = \theta\, dt\, B_z, \quad \det B = 1 + w^2 > 0,$$

ferme et toujours inversible (aucun appel a `std::`, quatre additions/multiplications, device-safe). Avec $c = \theta^2 dt^2 \alpha$ (Hoffart et al., arXiv:2510.11808), l'operateur condense s'ecrit

$$L_{\mathrm{schur}}(\phi) = -\Delta\phi - c\mathrm{div}(\rho\, B^{-1}\nabla\phi) = -\mathrm{div}\!\big((I + c\,\rho\, B^{-1})\,\nabla\phi\big),$$

ce qui identifie le tenseur plein $A = I + c\,\rho\, B^{-1}$, soit, par cellule,

$$\varepsilon_x = 1 + c\rho\,B^{-1}_{11},\quad \varepsilon_y = 1 + c\rho\,B^{-1}_{22},\quad a_{xy} = c\rho\,B^{-1}_{12},\quad a_{yx} = c\rho\,B^{-1}_{21}.$$

Le terme de masse $\kappa$ reste nul (la condensation ne produit pas de Helmholtz). En $B_z = 0$ : $w = 0$, $B^{-1} = I$, donc $a_{xy} = a_{yx} = 0$ et $\varepsilon_x = \varepsilon_y = 1 + c\rho$ ; si de plus $c = 0$, $A = I$ et $L_{\mathrm{schur}}$ degenere exactement en le Laplacien canonique. Le second membre condense est

$$\mathrm{rhs} = -\Delta\phi^n - \theta\, dt\, \alpha \mathrm{div}(\rho\, B^{-1} v^n), \qquad v^n = (m_x, m_y)/\rho,$$

ou $-\Delta\phi^n$ est le Laplacien 5 points canonique negue et la divergence du flux explicite $F = \rho B^{-1} v^n = B^{-1}(m_x, m_y)$ (applique a la quantite de mouvement, ce qui evite la division par $\rho$) est centree d'ordre 2 :

$$\mathrm{div} F(i,j) = \frac{F_x(i{+}1,j) - F_x(i{-}1,j)}{2\,dx} + \frac{F_y(i,j{+}1) - F_y(i,j{-}1)}{2\,dy}.$$

L'operateur condense est en general a tenseur plein (d'ou le solveur de Krylov, section 12). Convention de signe du solve : `TensorKrylovSolver` resout $L_{\mathrm{int}} = +\mathrm{div}(A\nabla\phi)$, donc $L_{\mathrm{schur}} = -L_{\mathrm{int}}$ et l'etage passe $\mathrm{rhs}_{\mathrm{kry}} = -\mathrm{rhs}_{\mathrm{schur}}$ au solveur. Apres resolution, la vitesse est reconstruite par $v^{n+\theta} = B^{-1}(v^n - \theta\, dt\,\nabla\phi^{n+\theta})$ (gradient centre, coherent avec la divergence du RHS), puis extrapolee du theta-stage au pas plein par $U^{n+1} = U^n + \tfrac{1}{\theta}(U^{n+\theta} - U^n)$. L'energie, si le role Energy est present, n'est mise a jour que par l'increment d'energie cinetique $E^{n+1} = E^n + \tfrac{1}{2}\rho^n(|v^{n+1}|^2 - |v^n|^2)$, la rotation de Lorentz ne travaillant pas et $\rho$ etant gelee.

```
function CondensedSchurSourceStepper.step(state, phi, bz_field, c_bz, theta, dt):
    # -1) figer phi^n pour l'extrapolation finale
    phi_n <- copy(phi)
    # 0) extraire v^n = (mx, my)/rho ; copier B_z dans le tampon interne (1 ghost)
    for each cell: vx_n, vy_n <- ExtractVelocity(state) ; bz <- CopyBz(bz_field, c_bz)
    fill_ghosts(bz, foextrap)
    # 1) assembler (builder #124) :  A_op = I + c rho B^{-1},  rhs_schur = -Lap phi^n - theta dt alpha div(rho B^{-1} v^n)
    builder <- ElectrostaticLorentzCondensation(vars, alpha, theta, dt)   # c = theta^2 dt^2 alpha
    builder.assemble_operator(state, bz -> eps_x, eps_y, a_xy, a_yx)
    builder.assemble_rhs(phi, state, bz -> rhs_schur)
    # 2) resoudre par BiCGStab :  L_int(phi) = -rhs_schur  (convention de signe)
    op.set_epsilon_anisotropic(eps_x, eps_y) ; op.set_cross_terms(a_xy, a_yx)     # operateur plein
    precond.set_epsilon_anisotropic(eps_x, eps_y)                                  # partie symetrique
    op.phi <- phi (warm start)  ;  op.rhs <- -rhs_schur
    last_result <- TensorKrylovSolver(op, precond, N).solve(1e-10, 400)
    phi <- op.phi                                                                  # phi^{n+theta}
    # 3) reconstruire la vitesse
    fill_ghosts(phi, bcPhi)
    for each cell:
        g <- grad_centre(phi)                                  # (d_x phi, d_y phi)
        rhs_v <- v_n - theta*dt * g
        v_theta <- B^{-1}(rhs_v)  with  w = theta*dt*bz        # LorentzEliminator.apply_Binv
        state.mom <- rho^n * v_theta                           # rho gelee
    # 5) extrapoler theta-stage -> pas plein :  f^{n+1} = f^n + (1/theta)(f^{n+theta} - f^n)
    phi <- phi_n + (1/theta)(phi - phi_n)
    v   <- v_n   + (1/theta)(v_theta - v_n) ;  state.mom <- rho^n * v
    # 4) energie (si role Energy present)
    if has_energy: state.E <- state.E + 0.5 * rho^n * (|v^{n+1}|^2 - |v^n|^2)
    # 6) publier : ghosts de l'etat et du potentiel (halos MPI / CL physiques)
    device_fence() ; fill_ghosts(state, foextrap) ; fill_ghosts(phi, bcPhi)
```

**Code.** [`numerics/lorentz_eliminator.hpp`](../include/adc/numerics/lorentz_eliminator.hpp) : struct POD `LorentzEliminator(theta, dt, B_z)`, methodes `apply_B`/`apply_Binv`, accesseurs `binv_11..binv_22` ; trivially copyable (static_assert), capturable par valeur dans un kernel. [`coupling/schur_condensation.hpp`](../include/adc/coupling/schur_condensation.hpp) batit l'operateur et le RHS sans resoudre ni reconstruire : classe `ElectrostaticLorentzCondensation`, methodes `assemble_operator` (foncteur `SchurOperatorCoeffKernel`), `assemble_rhs` (foncteurs `SchurExplicitFluxKernel`, `SchurRhsAssembleKernel`, `NegateKernel`), `assemble` (en un `SchurCondensationOperator`), accesseur `c_coeff()` ; le contrat de roles Density/MomentumX/MomentumY est valide a l'hote (exception sinon). [`coupling/condensed_schur_source_stepper.hpp`](../include/adc/coupling/condensed_schur_source_stepper.hpp) : classe `CondensedSchurSourceStepper`, methode `step` qui compose les trois briques (assembleur #124, `TensorKrylovSolver` #122, `LorentzEliminator` #118), foncteurs `SchurReconstructKernel`, `SchurExtrapolateScalarKernel`, `SchurExtrapolateVelocityKernel`, `SchurEnergyKernel`, `ExtractVelocityKernel`, `CopyBzKernel`, diagnostic `last_solve()`. C'est l'etage source de production (#126), opt-in via `adc.Split(source=CondensedSchur)`.

**Contraintes / remarques.** Stabilite : le theta-schema est inconditionnellement stable pour $\theta \geq 1/2$ ($\theta = 1$ implicite pur, l'extrapolation est l'identite ; $\theta = 1/2$ Crank-Nicolson, facteur d'extrapolation 2). La discretisation centree d'ordre 2 (Laplacien 5 points, divergence et gradient centres) fixe l'ordre spatial. C'est l'etage source seul (transport gele) : $\rho$ est constante dans l'etage, $\rho^{n+1} = \rho^n$, et toute la dynamique de transport reste dans l'etage hyperbolique du splitting. Garde-fou : $c = 0$ et $B_z = 0$ donnent $A = I$, le solve devient $\Delta\phi^{n+\theta} = \Delta\phi^n$ donc $\phi^{n+\theta} = \phi^n$ (a une constante pres), et la reconstruction degenere en la poussee electrostatique explicite $v^{n+\theta} = v^n - \theta\, dt\,\nabla\phi^n$. La tolerance du solve interne ($10^{-10}$, 400 iterations max) borne la precision de la relation implicite $B v = v^n - \theta\, dt\,\nabla\phi$ (verifiee terme a terme). Device/MPI : tous les kernels sont des foncteurs nommes device-clean (pas de lambda etendue cross-TU, limite nvcc #64/#97) ; les tampons MultiFab sont alloues une fois a la construction et reutilises a chaque `step` ; les boucles iterent sur `local_size()` (rang sans box -> aucun kernel) et le solve de Krylov est collectif, donc MPI-propre.

Validation : `test_schur_condensation` (operateur et RHS condenses corrects, dont le garde-fou C2 du cas degenere) et `test_condensed_schur_source_stepper` (l'etage source avance correctement). Cote Python : `test_schur_split`, `test_schur_via_system`, `test_schur_conservation`.


---

## 14. Bord embedded : cut-cell Shortley-Weller

**Intuition.** Une paroi non alignee sur la grille (conducteur circulaire, bord d'anneau de
diocotron) n'est pas un escalier : la cut-cell Shortley-Weller corrige le stencil 5 points la ou le
level set du disque coupe une face, de sorte que la condition de Dirichlet soit imposee a la position
reelle de l'interface et non a la face de cellule la plus proche. On retrouve l'ordre 2 la ou
l'escalier 0/1 tombe a l'ordre 1.

**Formule / discretisation.** Le bord est porte par le level set canonique du disque
$ls(x,y) = \mathrm{hypot}(x - c_x, y - c_y) - R$, negatif a l'interieur. Pour une cellule active
$ls(x_c, y_c) < 0$, chaque face cardinale est coupee a une fraction lineaire : si le voisin est
interieur ($l_n < 0$) la face est pleine et la demi-distance vaut $h$ ; si le level set change de
signe ($l_n \ge 0$) le croisement lineaire entre $l_c < 0$ et $l_n \ge 0$ donne

$$\theta = \frac{l_c}{l_c - l_n}, \qquad a = \theta\, h, \qquad \theta \in [10^{-3}, 1]$$

(le clamp inferieur $10^{-3}$ est la garde anti-division qui empeche $w \to \infty$ quand la face
rase le bord). Avec les quatre demi-distances $a_{xm}, a_{xp}, a_{ym}, a_{yp}$, et en posant
$s_x = a_{xm}+a_{xp}$, $s_y = a_{ym}+a_{yp}$, le Laplacien $-\Delta\phi$ devient un stencil 5 points a
pas inegaux (Shortley-Weller) de poids

$$w_{xm} = \frac{2}{a_{xm}\, s_x},\quad w_{xp} = \frac{2}{a_{xp}\, s_x},\quad
w_{ym} = \frac{2}{a_{ym}\, s_y},\quad w_{yp} = \frac{2}{a_{yp}\, s_y},$$

$$w_{\mathrm{diag}} = \frac{2}{a_{xm}\, a_{xp}} + \frac{2}{a_{ym}\, a_{yp}},$$

et le residu en cellule active est
$L\phi = w_{xm}\phi_{i-1} + w_{xp}\phi_{i+1} + w_{ym}\phi_{i,j-1} + w_{yp}\phi_{i,j+1} - w_{\mathrm{diag}}\phi_{i,j}$
(la valeur de Dirichlet du bord est injectee via le ghost place a $a$, pas $h$). Loin du bord toutes
les demi-distances valent $h$ : les poids redonnent exactement le stencil 5 points uniforme
$1/h^2,\dots,-4/h^2$. Pour une cellule conducteuse ($ls \ge 0$, masque a 0) la cellule est sautee
(coefficient inutilise).

```
function shortley_weller_coefs(level L, geometry g, level_set ls):
    # one-shot au setup, par niveau MG, sur l'hote (puis lu par le V-cycle on-device)
    for each active cell (i, j) of L:            # m(i,j) != 0 ; conducteur saute
        lc  = ls(g.x_cell(i), g.y_cell(j))       # < 0 par construction (cellule active)
        axm = cut_distance(lc, ls(x - dx, y), dx)  # voisin interieur -> dx ; sinon theta*dx
        axp = cut_distance(lc, ls(x + dx, y), dx)
        aym = cut_distance(lc, ls(x, y - dy), dy)
        ayp = cut_distance(lc, ls(x, y + dy), dy)
        sx = axm + axp ; sy = aym + ayp
        c(i,j,0) = 2 / (axm * sx)                # w_xm  sur p(i-1, j)
        c(i,j,1) = 2 / (axp * sx)                # w_xp  sur p(i+1, j)
        c(i,j,2) = 2 / (aym * sy)                # w_ym  sur p(i, j-1)
        c(i,j,3) = 2 / (ayp * sy)                # w_yp  sur p(i, j+1)
        c(i,j,4) = 2/(axm*axp) + 2/(aym*ayp)     # w_diag (coefficient central)

function cut_distance(lc, ln, h):
    if ln < 0:        return h                   # face pleine (voisin interieur)
    th = lc / (lc - ln)                          # crossing lineaire
    return clamp(th, 1e-3, 1) * h                # garde anti-division par 0
```

**Code.** La geometrie de coupe est centralisee dans
[`include/adc/numerics/elliptic/cut_fraction.hpp`](../include/adc/numerics/elliptic/cut_fraction.hpp) :
`detail::cut_distance` (croisement lineaire d'une face), `detail::cut_fraction` (les 4 demi-distances
+ apertures + fraction de volume `kappa`), et `detail::shortley_weller` qui rend les 5 poids
`ShortleyWellerWeights{w_xm, w_xp, w_ym, w_yp, w_diag}`. Le V-cycle
[`include/adc/numerics/elliptic/geometric_mg.hpp`](../include/adc/numerics/elliptic/geometric_mg.hpp)
les ecrit une fois par niveau dans son champ `coef` (5 composantes) au setup (hote) puis les lit
on-device ; il saute les cellules conductrices (`m(i,j) == 0`). C'est la meme `cut_fraction` que le
transport EB consomme (section 15) : geometrie d'ouverture bit-coherente entre Poisson et transport.

**Contraintes / remarques.** Le clamp $\theta \ge 10^{-3}$ borne $w_{\mathrm{diag}}$ (sans lui une
face rasante ferait diverger le poids et casserait la diagonale-dominance du lisseur). Compatible avec
l'operateur anisotrope (les poids cut-cell se composent avec les coefficients $\varepsilon_x, \varepsilon_y$). Validation : `test_cut_cell` (cut-cell vs escalier sur solution manufacturee, gain
d'ordre), `test_cut_cell_anisotropic` (cut-cell + operateur anisotrope), `test_cut_cell_anisotropic_multibox`
(multi-box mono-rang), `test_mpi_cutcell_multibox` (multi-box distribue np=1/2/4 ; verrou de
non-regression du bug `average_down` hors bornes sur hierarchie MG degeneree). Pour l'elliptique sur
disque immerge, `test_poisson_disc` exerce le solveur (convergence + amelioration a la resolution).

## 15. Domaine disque : masque, transport masque, transport cut-cell

**Intuition.** Un sous-domaine de transport en disque (anneau de diocotron) impose une frontiere
circulaire non alignee sur la grille cartesienne. Trois modes, du plus simple au plus precis :
`none` (masque materialise mais ignore, bit-identique au plein cartesien), `staircase` (frontiere
crenelee, porte de face 0/1, ordre 1 au bord), `cutcell` / embedded-boundary (apertures continues
`alpha_f` + fraction de volume `kappa`, frontiere lisse, ordre 2 a l'interieur du disque). La cut-cell
generalise la porte 0/1 a une ouverture $\alpha_f \in [0,1]$ et divise le residu par le vrai volume
immerge $\kappa$, ce qui de-crenele le bord et restaure l'ordre 2 que le masque escalier ne donne pas.

**Formule / discretisation.** Forme conservative embedded-boundary pour la cellule $(i,j)$ de volume
$\kappa\, dx\, dy$ (advection $\partial_t U = -\mathrm{div}\,F + S$) :

$$\kappa\, dx\, dy\; \partial_t U = -\big[\alpha_{xp} F^x_{i+1} - \alpha_{xm} F^x_i\big] dy
 - \big[\alpha_{yp} F^y_{j+1} - \alpha_{ym} F^y_j\big] dx - \alpha_w |w| F_w + \kappa\, dx\, dy\, S,$$

soit, apres division par $\kappa\, dx\, dy$ (avec $\kappa$ clampe, cf. plus bas) :

$$R = S - \frac{1}{\kappa}\left[\frac{\alpha_{xp} F^x_{i+1} - \alpha_{xm} F^x_i}{dx}
 + \frac{\alpha_{yp} F^y_{j+1} - \alpha_{ym} F^y_j}{dy}\right]
 - \frac{1}{\kappa}\,\frac{\alpha_w |w|}{dx\, dy}\, F_w.$$

Le flux de paroi immergee est un no-penetration $F_w = 0$ (pendant FV du mur conducteur : le terme est
identiquement nul, ecrit explicitement comme point d'accroche pour un futur flux non nul). Les
apertures et $\kappa$ viennent de la meme `cut_fraction` que la cut-cell elliptique :
$\alpha_f = a_f / h$, et $\kappa = \tfrac{1}{2}(\alpha_{xm}+\alpha_{xp})\cdot\tfrac{1}{2}(\alpha_{ym}+\alpha_{yp})$.
Une face entre deux cellules actives a la meme ouverture des deux cotes
($\alpha_{xp}(i) = \alpha_{xm}(i{+}1)$, fonction du seul level set), donc les flux telescopent et la
masse $\sum_{ij} n_{ij}\, \kappa_{ij}\, dx\, dy$ est conservee A LA machine ; une face touchant une
cellule inactive est fermee ($\alpha_f = 0$) et $F_w = 0$, donc aucune masse ne franchit le bord.

```
function assemble_rhs_eb(model, U, aux, ls, geom, R, kappa_min):
    # passe 1 : flux de face ponderes par l'ouverture alpha_f (MultiFab Fx, Fy temporaires)
    for each x-face i between cells (i-1,j) and (i,j):
        lL = ls(x_cell(i-1), y_cell(j)) ; lR = ls(x_cell(i), y_cell(j))
        alpha = face_aperture(lL, lR)            # voisin inactif -> 0 ; sinon cut_distance/dx
        if alpha < 1e-6:  Fx(i,j,:) = 0          # face fermee = paroi immergee, flux normal nul
        else:
            L  = reconstruct(U, i-1, j, dir=0, +1)   # reconstruction reutilisee du cartesien
            Rr = reconstruct(U, i,   j, dir=0, -1)
            Fx(i,j,:) = alpha * numerical_flux(L, Rr, dir=0)   # on stocke alpha * F
    ... idem pour les y-faces -> Fy ...

    # passe 2 : divergence EB / kappa_eff + source
    for each cell (i, j):
        if ls(x_cell(i), y_cell(j)) >= 0:        # hors disque : residu nul, cellule non avancee
            R(i,j,:) = 0 ; continue
        kappa     = cut_fraction(ls, x_cell(i), y_cell(j), dx, dy).kappa
        kappa_eff = max(kappa, kappa_min)        # clamp petite cellule (defaut 1e-2)
        S = model.source(U(i,j), aux(i,j))
        for each component c:
            div_x = (Fx(i+1,j,c) - Fx(i,j,c)) / dx     # Fx contient deja alpha*F
            div_y = (Fy(i,j+1,c) - Fy(i,j,c)) / dy
            # accumulation terme A terme : avec kappa_eff=1 et alpha=1, bit-identique au cartesien
            R(i,j,c) = S[c] - div_x/kappa_eff - div_y/kappa_eff - 0   # F_wall = 0

function face_aperture(lc, ln):
    if ln >= 0:  return 0                         # voisin inactif : face fermee (no-penetration)
    return cut_distance(lc, ln, h) / h            # voisin actif : ouverture lineaire (== mur elliptique)
```

**Code.** `System::set_disc_domain(cx, cy, R, mode)` (#216,
[`include/adc/runtime/system.hpp`](../include/adc/runtime/system.hpp), defini dans
[`python/system.cpp`](../python/system.cpp)) pose un `DiscDomain`
([`include/adc/runtime/wall_predicate.hpp`](../include/adc/runtime/wall_predicate.hpp), `level_set`) et
le mode de transport ; `set_geometry_mode(mode)` bascule le mode seul ; `disc_mask()` materialise le
masque (tout-actif si aucun disque). Le stepper aiguille chaque bloc : `assemble_rhs` (plein),
`assemble_rhs_masked`
([`include/adc/numerics/spatial_operator.hpp`](../include/adc/numerics/spatial_operator.hpp), porte
0/1) ou `assemble_rhs_eb`
([`include/adc/numerics/spatial_operator_eb.hpp`](../include/adc/numerics/spatial_operator_eb.hpp),
EB). Les kernels device sont des foncteurs nommes (`detail::EbFaceFluxXKernel`,
`EbFaceFluxYKernel`, `EbAssembleRhsKernel`, et l'adaptateur `detail::DiscLevelSet` qui forwarde
`DiscDomain::level_set`) pour l'emission cross-TU sous nvcc ; `eb_face_aperture` ferme la face vers un
voisin inactif. Les apertures et `kappa` viennent de
[`include/adc/numerics/elliptic/cut_fraction.hpp`](../include/adc/numerics/elliptic/cut_fraction.hpp)
(meme geometrie que la cut-cell elliptique de la section 14) ; la reconstruction (`reconstruct<>`) et
le flux numerique (`RusanovFlux`) sont reutilises verbatim depuis l'operateur cartesien.

**Contraintes / remarques.** Probleme de la petite cellule : le facteur $1/\kappa$ amplifie le residu
quand $\kappa \to 0$ sur la couche de coupe, ce qui ferait exploser un pas explicite fixe. Deux gardes
empilees : (i) le plancher $\theta \ge 10^{-3}$ de `cut_distance` herite du mur elliptique (borne
$\kappa \gtrsim 2.5\times 10^{-7}$, insuffisant seul) ; (ii) le clamp DE volume retenu
$\kappa_{\mathrm{eff}} = \max(\kappa, \kappa_{\min})$, $\kappa_{\min} = 10^{-2}$ par defaut, qui borne
l'amplification a $1/\kappa_{\min} = 100$ (volume merging implicite, calibre pour un pas fixe stable
quel que soit le degre de coupe). Le clamp n'agit que sur le denominateur (volume), pas sur les flux :
la masse globale reste exacte, au prix d'une legere non-conservation locale sur les cellules les plus
coupees. Alternative documentee (hors scope) : la redistribution de flux d'AMReX-EB (conservation
locale exacte, stencil non local). Le chemin est purement additif et opt-in : un run sans disque EB est
bit-identique au cartesien. Validation : `test_disc_domain_mask` (bornes du masque, tout-actif sans
disque), `test_eb_transport` (transport cut-cell EB : conservation et frontiere lisse). Cote Python :
`test_disc_domain_mask`.

## 16. Geometrie polaire : transport et Poisson sur anneau (r, theta)

**Intuition.** Pour un anneau de diocotron, la grille cartesienne paie un sur-taux structurel aux bords
de l'anneau (verrou "bords d'anneau cartesiens"). Une grille polaire annulaire
$r \in [r_{\min}, r_{\max}] \times \theta \in [0, 2\pi)$ aligne la geometrie sur le probleme : theta est
periodique, r est physique (parois), et l'anneau exclut $r = 0$ ($r_{\min} > 0$) donc aucune
singularite de coordonnee. Transport et Poisson polaires reutilisent les memes briques de
reconstruction, de flux et de source que le cartesien ; seules les metriques changent.

**Formule / discretisation (transport, FV conservatif).** La divergence polaire
$\mathrm{div}\,F = \tfrac{1}{r}\partial_r(r F_r) + \tfrac{1}{r}\partial_\theta(F_\theta)$ se discretise
en stockant le flux radial pondere par le rayon de face, $r_{i\pm1/2} F_r$, et le flux azimutal direct,
puis en differenciant :

$$R = S + S_g - \frac{1}{r_i}\frac{r_{i+1/2} F_r^{i+1} - r_{i-1/2} F_r^{i}}{dr}
 - \frac{1}{r_i}\frac{F_\theta^{j+1} - F_\theta^{j}}{d\theta}.$$

$S_g$ est la source geometrique de courbure ($-\rho v_\theta^2/r$ etc.), non capturee par la divergence
conservative en base locale tournante ; elle est portee en cellule (nulle pour une brique scalaire ExB
-> bit-identique au transport ExB polaire historique). Le poids $r_{i+1/2}$ d'une face interieure est
partage par les deux cellules voisines, donc le terme radial telescope ; le terme azimutal telescope
exactement (periodique). Avec `wall_radial`, le flux radial est force a zero aux deux faces physiques
de bord -> masse $\sum n_{ij}\, r_i\, dr\, d\theta$ conservee A LA machine quel que soit $v_r$.

**Formule / discretisation (Poisson, FFT-en-theta + tridiag-en-r).** On resout
$\tfrac{1}{r}\partial_r(r\,\partial_r\phi) + \tfrac{1}{r^2}\partial_\theta^2\phi = f$ par voie directe
(pas de multigrille, qui stagne sur l'operateur $1/r^2$). Theta etant periodique a coefficient
constant, une FFT en theta diagonalise exactement $\partial_\theta^2$ : le mode DFT $m$ a le nombre
d'onde signe $k(m) = m$ si $m \le n_\theta/2$, sinon $m - n_\theta$, et la valeur propre spectrale
$-k(m)^2$ (et non le stencil 2 points $(2\cos - 2)/d\theta^2$, qui n'est qu'une approximation
$O(d\theta^2)$). Le terme azimutal en cellule devient $(-k(m)^2/r_i^2)\,\hat\phi(i,m)$, diagonal en
$m$. Le terme radial est FV ordre 2,

$$\frac{1}{r_i}\left[\frac{r_{i+1/2}(\phi_{i+1}-\phi_i)}{dr} - \frac{r_{i-1/2}(\phi_i-\phi_{i-1})}{dr}\right]\frac{1}{dr},$$

donc, par mode $m$, un systeme tridiagonal en $r$ resolu par Thomas avec

$$a_i = \frac{r_{i-1/2}}{r_i\, dr^2},\qquad c_i = \frac{r_{i+1/2}}{r_i\, dr^2},\qquad
b_i = -(a_i + c_i) - \frac{k(m)^2}{r_i^2}.$$

```
function polar_poisson_solve(geom, bc, rhs f, out phi):
    nr, nth = geom.nr, geom.ntheta
    # 1) FFT en theta, ligne radiale par ligne radiale : f(i, .) -> fhat(i, m)
    for i in 0..nr-1:  fhat[i] = fft( f(i, .) )
    # 2) coefficients radiaux independants du mode (geometrie pure)
    for i in 0..nr-1:
        a[i] = r_face(i)   / (r_cell(i) * dr^2)        # sous-diag
        c[i] = r_face(i+1) / (r_cell(i) * dr^2)        # sur-diag
        d_rad[i] = -(a[i] + c[i]) ;  inv_r2[i] = 1 / r_cell(i)^2
    # 3) une tridiagonale (Thomas) par mode azimutal m
    for m in 0..nth-1:
        k = (m <= nth/2) ? m : m - nth                 # nombre d'onde signe (repliement DFT)
        for i: b[i] = d_rad[i] - k*k * inv_r2[i]        # diag = radiale + azimutale spectrale
        rhs_m = fhat[., m]
        apply_radial_bc(b, rhs_m, m)                   # Dirichlet: b-=a/c, rhs-=2 a v (m=0) ; Neumann: b+=a/c
        pin0 = (deux bords Neumann) and (m == 0)        # operateur radial singulier -> jauge phi_hat(0,0)=0
        phat[., m] = thomas(a, b, c, rhs_m, pin0)
    # 4) FFT inverse en theta : phat(i, m) -> phi(i, theta) (partie reelle)
    for i in 0..nr-1:  phi(i, .) = real( ifft( phat[i] ) )
```

Conditions aux limites en $r$ (par `BCRec.xlo/.xhi`) : Dirichlet (valeur $v$ a la face, ghost de
reflexion $\phi_{-1} = 2v - \phi_0$ -> $b_0 \mathrel{-}= a_0$, et $2 a_0 v$ au second membre du mode
$m=0$ seul) ou Neumann homogene (Foextrap, $\phi_{-1} = \phi_0$ -> $b_0 \mathrel{+}= a_0$). Mode $m=0$
+ deux bords Neumann : l'operateur radial a la constante pour noyau (tridiagonale singuliere) ; on fixe
la jauge en epinglant $\hat\phi(0,0) = 0$ (ligne 0 remplacee par l'identite dans Thomas).

**Code.** [`include/adc/mesh/geometry.hpp`](../include/adc/mesh/geometry.hpp)`::PolarGeometry` (anneau,
opt-in via `adc.PolarMesh` ; `cfg.geometry == "polar"` cote
[`python/system.cpp`](../python/system.cpp)). Transport :
[`include/adc/numerics/spatial_operator_polar.hpp`](../include/adc/numerics/spatial_operator_polar.hpp)`::assemble_rhs_polar<Limiter, NumericalFlux>`
(`recon_prim`, `wall_radial`), via les foncteurs nommes `detail::PolarFaceFluxRKernel` (flux radial
pondere par `r_face`, paroi optionnelle aux faces de bord), `PolarFaceFluxThetaKernel`,
`PolarAssembleRhsKernel` ; la source physique et la source geometrique sont routees par les concepts
`PolarHasSource` / `PolarHasGeomSource` (`if constexpr` : zero codegen pour une brique scalaire,
chemin ExB bit-identique). Instancie via `runtime/block_builder_polar.hpp`, branche dans
`System::step` pour `geometry == "polar"`. Poisson :
[`include/adc/numerics/elliptic/polar_poisson_solver.hpp`](../include/adc/numerics/elliptic/polar_poisson_solver.hpp)`::PolarPoissonSolver`
(FFT-en-theta `fft1d` reutilisee de `poisson_fft.hpp` + `thomas_solve` complexe en r ; modele le
concept `PolarEllipticSolver` `rhs()/phi()/solve()/residual()/geom()`). L'aux est derive en base locale
$(e_r, e_\theta)$ : `aux[1] = d phi/dr`, `aux[2] = (1/r) d phi/d theta`
(`block_builder_polar.hpp`, `System::solve_fields_polar`).

**Operateur polaire tensoriel + Schur polaire.** Quand la source implicite couplee passe en polaire
(diocotron a $\omega_c$ eleve), le Schur condense un operateur tensoriel plein
$A = I + c\,\rho\, B^{-1}$ a termes croises $a_{rt}, a_{tr}$ et a coefficient dependant de theta : la
FFT-en-theta du `PolarPoissonSolver` ne s'applique plus (elle exige un coefficient constant en theta
sans couplage croise).
[`include/adc/numerics/elliptic/polar_tensor_operator.hpp`](../include/adc/numerics/elliptic/polar_tensor_operator.hpp)`::PolarTensorKrylovSolver`
resout alors par BiCGStab matrice-libre (gere le non-symetrique du terme croise), preconditionne
`Jacobi` ou `RadialLine` (Thomas radial par ligne theta, defaut). Pas de V-cycle MG (stagnation sur
$1/r^2$). Operateur singulier (pure Neumann radial + theta periodique) : jauge fixee par projection sur
le sous-espace de moyenne FV nulle (`project_mean`, pendant iteratif du pinning de mode 0). Le stencil
9 points lit les coins diagonaux remplis par `fill_ghosts` (sans quoi le terme croise serait faux au
bord de box). `coupling/polar_condensed_schur_source_stepper.hpp::PolarCondensedSchurSourceStepper` est
le pendant polaire du `CondensedSchurSourceStepper` (#212). Multi-rang MPI / multi-box par decoupage
azimutal seul sous `RadialLine` (le sweep Thomas en r doit rester local a une box, garde-fou
`check_radial_columns`), pavage 2D libre sous `Jacobi` ; Schur polaire leve en MPI (#227).

**Contraintes / remarques.** PolarPoissonSolver : portee mono-rang, boite unique couvrant l'anneau
(la FFT-en-theta + tridiag-en-r exige la ligne theta ET la colonne radiale completes sur un rang ; le
distribue imposerait une transposee parallele, hors scope Phase 2a) -> garde-fou dur (actif en Release)
si `n_ranks() > 1` ou `ba.size() != 1`, leve sur tous les rangs (pas d'interblocage) ; `solve()` /
`residual()` sont `local_size()==0`-safe. Theta spectral : exact pour une donnee a bande limitee
(diocotron = peu de modes azimutaux), `dtheta` n'intervient pas dans la valeur propre. Le tridiag est
diagonale-dominant (terme azimutal $\le 0$, BC repliees) -> Thomas stable sans pivotage. La residence
hote du RHS est synchronisee (`sync_host`) avant toute lecture hote (kernel device eventuellement en
vol ; no-op en serie/OpenMP, `device_fence` cible sous Kokkos Cuda). PolarTensorKrylovSolver :
RadialLine $\sim$ nombre d'iterations a croissance moderee (isotrope $\times 2$ par doublement de grille,
tenseur $\times 2.4$) ; Jacobi croit en $1/h^2$ (sanity check / repli). Le terme croise et le couplage
azimutal ne sont pas dans le preconditionneur (limite honnete, raffinement ulterieur possible).
Validation : `test_polar_transport_mms` / `test_polar_mms_vr` (MMS transport polaire ordre 2),
`test_polar_ring_advection`, `test_polar_fluid_transport`, `test_polar_lorentz_source`,
`test_polar_conservation_radial_flux` (paroi radiale, masse conservee), `test_polar_poisson_mms`
(PolarPoissonSolver, ordre 2 radial), `test_polar_tensor_elliptic_mms` (operateur tensoriel polaire),
`test_polar_condensed_schur_source_stepper`, `test_mpi_polar_schur` (Schur polaire multi-rang).
`test_polar_system_step` valide le chemin `System::step` polaire complet (field-solve + aux en base
locale + SSPRK3 transport + paroi). Cote Python : `test_polar_system`, `test_polar_diocotron`,
`test_polar_rejections`, `test_polar_schur_via_system`, `test_polar_conservation_radial_flux`,
`test_polar_teardown_stability`.


---

## 17. AMR : sous-cyclage Berger-Oliger + reflux conservatif

**Intuition.** Raffiner seulement la ou il le faut. Un niveau fin (pas $\Delta x_c / r$) recouvre une
sous-region ; pour respecter sa CFL il fait $r$ sous-pas de $\Delta t / r$ pendant que le grossier fait
un seul pas de $\Delta t$. A l'interface fin-grossier, les deux niveaux calculent des flux differents,
ce qui casse la conservation discrete. Le reflux corrige la maille grossiere bordante par la difference
(flux fin integre en temps moins flux grossier).

**Formule / discretisation.** Soit une face fin-grossier en $x$ entre la cellule grossiere $(I, J)$ et le
patch fin. Pendant le pas grossier on a deja avance le grossier avec son propre flux de face $F_c$ (sur
$\Delta t$). On accumule en parallele le flux fin de la meme face sur les $r$ sous-pas. La correction
remplace la contribution grossiere par la contribution fine :

$$U_c(I,J) \mathrel{-}= \frac{1}{\Delta x_c}\Big(\textstyle\sum_{s=1}^{r} \Delta t_f\,\bar F_f^{(s)} - \Delta t_c\,F_c\Big)$$

avec $\Delta t_f = \Delta t / r$ et $\bar F_f^{(s)}$ le flux fin moyenne sur les faces fines couvrant la
face grossiere. Dans le code la quantite fine integree est deja $\sum_s \Delta t_f \bar F_f^{(s)}$
(stockee `fL/fR/fB/fT` du registre par patch) et la quantite grossiere est $F_c$ (stockee `cL/cR/cB/cT`),
multipliee par $\Delta t$ au moment du versement. Le ratio est fige a $r = 2$ (`SubcyclingSchedule`),
d'ou l'empreinte grossiere $I_0 = \mathrm{lo}/2$, $I_1 = (\mathrm{hi}-1)/2$ d'un patch fin aligne
(`PatchRange`). L'interpolation des ghosts fins depuis le grossier est lineaire en temps,
$U^\star = (1-\alpha)\,U_c^{\mathrm{old}} + \alpha\,U_c^{\mathrm{new}}$ avec $\alpha = s/r$
(`SubcyclingSchedule::frac`), constante en espace (injection par morceaux).

```
function subcycle_level(coarse_level, fine_level, dt, r=2):
    # 1. avancer le grossier d'un pas dt, en memorisant son flux de face
    #    le long de l'interface fin-grossier avant de le mettre a jour
    Fc = sample_coarse_face_flux(coarse_level)        # cL,cR,cB,cT par patch
    advance_one_step(coarse_level, dt)

    # 2. r sous-pas fins de dt/r
    reg.f{L,R,B,T} = 0                                 # accumulateur fin time-integre
    for s in 1..r:
        alpha = s / r                                  # position temporelle (frac)
        fill_fine_ghosts(fine_level, Uc_old, Uc_new, alpha)   # interp espace + temps
        fill_boundary(fine_level)                      # fin-fin ecrase les ghosts couverts
        Ff = compute_face_fluxes(fine_level)
        advance_one_step(fine_level, dt/r)
        reg.f{L,R,B,T} += (dt/r) * mean_over_fine_faces(Ff)   # sum_s dt_f * Ff^(s)

    # 3. average_down : la zone couverte du grossier = moyenne du fin (conservatif)
    average_down(fine_level, coarse_level, r)

    # 4. reflux : corriger les cellules grossieres bordantes (non couvertes)
    cfi = CoarseFineInterface(coarse_region, fine_boxarray_global)
    for patch g in fine_level:
        cfi.route_reflux(reg[g], dx, dy, dt, flux_register, nc)
    flux_register.gather()                             # all_reduce_sum (identite en serie)
    for cell (I,J) bordante:
        Uc(I,J) -= flux_register.at(I,J,k)
```

**Code.** [`numerics/time/amr_reflux_mf.hpp`](../include/adc/numerics/time/amr_reflux_mf.hpp) est le
parapluie qui agrege les sous-entetes. L'entree de production unifiee est `advance_amr` dans
[`numerics/time/amr_advance.hpp`](../include/adc/numerics/time/amr_advance.hpp) (facade fidele du moteur
N-niveaux multi-patch `detail::amr_step_multilevel_multipatch`). Les roles sont promus en types nommes
dans [`numerics/time/amr_patch_range.hpp`](../include/adc/numerics/time/amr_patch_range.hpp) :
`SubcyclingSchedule` (cadence $r$, $\Delta t/r$, $\mathrm{frac}(s)=s/r$), `PatchRange` (empreinte grossiere
$[I_0..I_1]\times[J_0..J_1]$ d'un patch fin), `FluxRegister` (buffer a index global, accumulation
`add`/`set` puis `gather`), `CoverageMask` (cellules ombragees), `CoarseFineInterface::route_reflux`
(versement bordant). Les transferts inter-niveaux `average_down` (moyenne conservative sur blocs $r\times r$),
`interpolate` (injection constante par morceaux) et `parallel_copy` sont dans
[`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp). Le ghost fin par cellule passe par
`fill_cf_ghost_cell` (interpolation espace + temps), partage par les trois variantes `mf_fill_fine_ghosts_*`.

**Contraintes / remarques.** Le ratio temporel est fige a $r = 2$ : `PatchRange` utilise l'arithmetique
historique $(hi-1)/2$ pour la borne haute, qui n'est pas `Box2D::coarsen` (floor des deux bornes), et qui
suppose des patchs alignes (lo pair, hi impair). L'ordre des operations est critique : le flux grossier
doit etre echantillonne avant d'avancer le grossier ; l'average_down doit preceder toute mesure de masse,
sinon la zone couverte est comptee deux fois. Validation : `test_refinement` (average_down conservatif +
interpolate), `test_amr_hierarchy` (grossier + fin imbrique + interpolation des ghosts), `test_flux_register`
(indexation du registre), `test_coverage_mask` (marquage des cellules couvertes), `test_advance_amr`
(moteur unifie 2 ET 3 niveaux, maxdiff = 0), `test_amr_diagnostics` (masse et vitesse de derive via le
reducteur de seam).

## 18. AMR multi-patch : reflux coverage-aware, distribue MPI

**Intuition.** Un niveau fin n'est pas une seule boite mais un ensemble de patchs. Deux subtilites en
decoulent : (a) au joint entre deux patchs voisins (interface fin-fin) il ne faut pas refluxer, car les
deux cotes sont fins et le bilan est deja conservatif ; (b) la correction doit aller dans la bonne boite
parente quand le grossier est lui-meme multi-box ou reparti sur plusieurs rangs MPI.

**Formule / discretisation.** Le reflux multi-patch est le meme operateur que la section 17, mais filtre
par un masque de couverture. Pour une cellule grossiere bordante $(I,J)$ adjacente a la face d'un patch $g$,
la correction n'est versee que si $(I,J)$ n'est ombragee par aucun patch fin :

$$U_c(I,J) \mathrel{-}= \mathbb{1}\big[\lnot\,\mathrm{covered}(I,J)\big]\cdot\frac{\bar F_f - F_c\,\Delta t}{\Delta x}$$

ou $\mathrm{covered}(I,J)$ teste l'appartenance a l'empreinte grossiere `PatchRange` d'un quelconque patch
fin. Le masque est bati sur le BoxArray global (tous les patchs, connu de tous les rangs), donc independant
de la distribution MPI. Le registre de flux a une indexation globale : chaque rang remplit ses contributions
locales (zero ailleurs), puis $\mathrm{buf} \leftarrow \sum_{\text{rangs}} \mathrm{buf}$ par
`all_reduce_sum_inplace` ; en serie le all_reduce est l'identite, donc bit a bit identique au mono-rang.

```
function reflux_multipatch(coarse_level, fine_boxarray_global, registers, distribution):
    # masque sur le box_array global -> correct sous n'importe quelle distribution MPI
    cfi = CoarseFineInterface(coarse_region, fine_boxarray_global)
        for g in fine_boxarray_global:
            cfi.cmask.mark(PatchRange(g).box())        # empreinte grossiere de chaque patch

    flux_register = FluxRegister(coarse_region, nc)    # index global, buf=0
    for patch g OWNED-LOCALLY by this rank:
        # route gauche/droite (x) puis bas/haut (y), uniquement sur cellules non couvertes
        for J in g.J0..g.J1, k in 0..nc:
            if not cfi.covered(g.I0-1, J): ref.add(g.I0-1, J, -(fL - cL*dt)/dx)
            if not cfi.covered(g.I1+1, J): ref.add(g.I1+1, J, +(fR - cR*dt)/dx)
        for I in g.I0..g.I1, k in 0..nc:
            if not cfi.covered(I, g.J0-1): ref.add(I, g.J0-1, -(fB - cB*dt)/dy)
            if not cfi.covered(I, g.J1+1): ref.add(I, g.J1+1, +(fT - cT*dt)/dy)

    flux_register.gather()                             # all_reduce_sum sur tous les rangs

    if coarse REPLICATED (default):
        apply correction locally (chaque rang a la copie complete)
    else: # coarse DE-replique (multi-box reparti)
        parallel_copy correction into the owning coarse box
        average_down zone couverte via mf_average_down_mb / parallel_copy
```

**Code.** Les types coverage-aware vivent dans
[`numerics/time/amr_patch_range.hpp`](../include/adc/numerics/time/amr_patch_range.hpp) :
`CoverageMask` (construit sur la region grossiere, `mark` marque l'empreinte intersectee, `covered` est
borne hors region), `CoarseFineInterface` (assemble le masque sur `fine_ba.size()` patchs globaux et
expose `route_reflux`, fonction nommee templatee sur le type de registre `Reg`/`RegMP` donc sure sous nvcc),
`FluxRegister::gather` (somme inter-rangs par `all_reduce_sum_inplace`). Le routage MPI du grossier
reparti passe par `parallel_copy` dans
[`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp) (redistribution generale entre deux MultiFab
sur le meme domaine a decompositions differentes : copies locales via `BoxHash::query`, puis jobs
`MPI_Isend`/`MPI_Irecv` enumeres deterministiquement, tag 1). Le grossier replique remplit ses ghosts
periodiques par `fill_periodic_local` (auto-repli purement local, sans plan MPI). Le drapeau
`coarse_replicated` de `LevelHierarchy` (defaut `true`) est transmis au moteur par `advance_amr` dans
[`numerics/time/amr_advance.hpp`](../include/adc/numerics/time/amr_advance.hpp) ; sans ce passage, un
grossier de-replique repasserait en mode replique (`mf_find_box` au lieu de `parallel_copy`).

**Contraintes / remarques.** Sans masque de couverture, le joint fin-fin serait reflue deux fois, donc
non-conservation : le masque est l'invariant central de correction. Le registre doit etre rempli localement
(zero ailleurs) avant `gather`, sinon le all_reduce double-compte. La reproductibilite bit a bit exige un
ordre d'enumeration deterministe des jobs `parallel_copy` (hash spatial sur la source, candidats tries).
Validation : `test_amr_spatial_parity` (le coeur spatial du chemin AMR est identique a celui de `System` :
meme reconstruction primitive, meme flux HLLC/Roe), `test_mpi_mbox_parity` (residu invariant au decoupage
en boites ET au nombre de rangs np = 1/2/4, dmax = 0), `test_mpi_amr_distributed_coarse` (grossier reparti
identique au grossier replique bit a bit, np = 1/2/4).

## 19. Clustering Berger-Rigoutsos et regrid

**Intuition.** Etant donne les cellules marquees (fort gradient, ou tout predicat physique), trouver un
petit nombre de boites rectangulaires qui les couvrent sans trop de gaspillage. L'algorithme coupe
recursivement une region la ou la signature (histogramme des marques projete sur un axe) presente un trou
(colonne vide), sinon une inflexion (extremum du changement de Laplacien de la signature), sinon au milieu.

**Formule / discretisation.** Pour une region $R$, on definit l'efficacite $\eta = N_{\mathrm{tag}}(R) / |R|$
(fraction de cellules taguees). On accepte $R$ comme boite si $\eta \ge \eta_{\min}$
(`min_efficiency`, defaut $0.7$) ou si $R$ n'est pas splittable. Sinon on coupe. La signature sur l'axe
$a$ est la projection $s_a[k] = \sum_{\text{ligne/col } k} \mathrm{tag}$. Un trou est un indice interieur
$k \in [\mathrm{mb}, \mathrm{len}-\mathrm{mb}]$ avec $s_a[k] = 0$, le plus proche du centre. A defaut, on
prend l'inflexion : Laplacien discret $D[k] = s[k+1] - 2 s[k] + s[k-1]$, et on coupe a l'indice qui
maximise $|D[k] - D[k-1]|$. A defaut encore, on coupe au milieu de la plus grande dimension splittable.
Le critere de splittabilite est $n_a \ge 2\,\mathrm{mb}$ avec $\mathrm{mb} = \max(1, b)$ avec $b$ = `min_box_size`.
Apres acceptation, chaque boite brute est chopee en sous-boites de cote $\le$ `max_box_size`. Au regrid,
les boites grossieres sont raffinees par `refine(ref_ratio)` dans l'espace d'indices du niveau fin.

```
function berger_rigoutsos(tags, params):
    raw = []
    cluster_rec(tags, tags.box, params, raw)
    result = []
    for b in raw:
        result += chop(b, params.max_box_size)         # BoxArray::from_domain
    return result

function cluster_rec(tags, region, p, out):
    region = bounding_box_of_tags(region)               # trim
    if region empty: return
    eff = count_tags(region) / num_cells(region)
    mb  = max(1, p.min_box_size)
    sx  = region.nx >= 2*mb ;  sy = region.ny >= 2*mb
    if eff >= p.min_efficiency or (not sx and not sy):
        out.push(region) ; return                       # accepte
    Sx = signature(region, axis=0) ; Sy = signature(region, axis=1)
    hx = sx ? best_hole(Sx, mb) : -1                    # trou interieur le plus central
    hy = sy ? best_hole(Sy, mb) : -1
    choose (axis, kcut):
        both holes  -> couper l'axe le plus long
        one hole    -> cet axe
        no hole     -> best_inflection (max |Laplacien'|), score le plus fort
        no infl.    -> milieu de la plus grande dim splittable
    (left, right) = split(region, axis, kcut)
    cluster_rec(tags, left, p, out)
    cluster_rec(tags, right, p, out)

function regrid_level(hierarchy, coarse_lev, crit, params):
    tags  = tag_cells(data(coarse_lev), domain, crit)   # predicat (Array4, i, j) -> bool
    grown = grow_tags(tags, n_buffer, domain)           # dilatation (nesting + buffer)
    if grown.count() == 0:
        clear_above(coarse_lev) ; return                # plus rien a raffiner
    # MPI : OU global des tags avant clustering, sinon BoxArray fin diverge par rang
    all_reduce_or(grown)                                 # (grossier reparti)
    cboxes = berger_rigoutsos(grown, params.cluster)
    fboxes = [ b.refine(ref_ratio) for b in cboxes ]
    newfine = MultiFab(fboxes, DistributionMapping, ncomp, n_grow)
    interpolate(data(coarse_lev), newfine, ref_ratio)   # injection grossier -> fin
    if niveau fin existant: parallel_copy(newfine, data(coarse_lev+1))  # preserver l'ancien fin
    install_level(coarse_lev+1, fba, newfine)
```

**Code.** Le clustering est `berger_rigoutsos` dans
[`amr/cluster.hpp`](../include/adc/amr/cluster.hpp), avec les helpers `detail::tag_bbox` (trim),
`detail::signature`, `detail::best_hole`, `detail::best_inflection` (max $|D[k]-D[k-1]|$),
`detail::cluster_rec` (recursion), et le chop final par `BoxArray::from_domain(b, max_box_size)`. Les
parametres sont `ClusterParams` (`min_efficiency`, `min_box_size`, `max_box_size`). Le regrid est
`regrid_level` dans [`amr/regrid.hpp`](../include/adc/amr/regrid.hpp) : `tag_cells` (predicat generique
sur `ConstArray4`), `grow_tags` (dilatation carree bornee au domaine), `berger_rigoutsos`, puis
`Box2D::refine(ref_ratio)`, `interpolate` et `parallel_copy` (pour preserver les valeurs de l'ancien fin)
de [`mesh/refinement.hpp`](../include/adc/mesh/refinement.hpp), enfin `AmrHierarchy::install_level`. Sans
tag, `clear_above` supprime le niveau fin et les plus fins. Sous MPI, le OU global des tags
(`all_reduce_or_inplace`) doit precand le clustering, sinon la BoxArray fine differerait par rang.

**Contraintes / remarques.** Le clustering est pur, sequentiel, sans physique ni MPI : il consomme une
`TagBox` deja rassemblee. Le proper nesting (chaque patch fin strictement interieur a la couverture
parente) repose sur la dilatation `grow_tags` (rayon `n_buffer`) et doit etre garanti apres le clustering,
sinon le ghost-fill inter-niveaux lit hors de la couverture parente. Le predicat de tagging est agnostique
de la physique ; pour un critere a gradient l'appelant remplit les ghosts avant. La signature pousse les
coupes vers les vrais trous geometriques : un bloc plein donne 1 boite, deux blocs separes par une bande
vide donnent 2 boites. Validation : `test_cluster` (bloc plein -> 1 boite, deux blocs separes -> 2 boites,
gros bloc decoupe par `max_box_size`), `test_regrid` (un niveau fin est cree autour de la region taguee,
donnees fines interpolees depuis le grossier).


---

## 20. Maillage distribue : BoxArray global, halos, equilibrage

**Intuition.** Le AMR distribue exige que tous les rangs connaissent toutes les boites (BoxArray
global) pour calculer la couverture multi-patch et enumerer les echanges de halos de facon
deterministe, mais que chaque rang n'alloue que ses fabs locaux (via la `DistributionMapping`).
L'echange de halos comble les ghosts paralleles ; l'equilibrage repartit les boites sur les rangs.

**Formule / discretisation.** Le pavage `from_domain(domain, m)` decoupe chaque axe `[lo, hi]`
(longueur `len = hi - lo + 1`) en `n = ceil(len / m)` segments repartis le plus egalement possible :
les `len mod n` premiers segments font `floor(len/n) + 1` cellules, les autres `floor(len/n)` (pas
de queue gloutonne). L'indice global de box est sa position dans l'ordre `y` externe / `x` interne,
identique sur tous les rangs.

L'equilibrage minimise le desequilibre `max_r charge(r) / moyenne(charge)`, la charge d'une box
etant son nombre de cellules (proxy du cout). Deux strategies : Z-order (courbe de Morton, segments
contigus de charge cible `total / nranks` -> localite spatiale) et knapsack/lpt (box la plus lourde
au rang le moins charge -> desequilibre maximal minimal, sans localite). La cle de Morton entrelace
les bits de `(x, y)` ramenes a l'origine du bounding box :

$$\mathrm{morton}(x, y) = \sum_{b\ge 0} \big(x_b\,4^b + y_b\,2\cdot 4^b\big),\qquad
  \mathrm{cible}_r = \frac{\text{total cellules}}{\text{nranks}}$$

L'echange de halos remplit le ghost `D(i,j)` d'un fab depuis la cellule valide decalee
`S(i - s_x, j - s_y)` du voisin, ou `(s_x, s_y)` parcourt `{0, +-L_x} x {0, +-L_y}` (les decalages
periodiques ne sont actifs que si la direction est periodique).

```
function from_domain(domain, m):                      # BoxArray::from_domain
    sx = split_range(domain.lo[0], domain.hi[0], m)   # segments en x
    sy = split_range(domain.lo[1], domain.hi[1], m)   # segments en y
    boxes = []
    for (ylo, yhi) in sy:        # y externe, x interne -> ordre deterministe (= indice global)
        for (xlo, xhi) in sx:
            boxes.push( Box2D{{xlo, ylo}, {xhi, yhi}} )
    return BoxArray(boxes)

function split_range(lo, hi, m):
    len = hi - lo + 1 ; n = ceil(len / m)
    base = len div n ; rem = len mod n ; cur = lo
    for k in 0..n-1:
        l = base + (1 if k < rem else 0)              # rem premiers segments +1
        emit (cur, cur + l - 1) ; cur += l

function make_sfc_distribution(ba, nranks):           # load_balance.hpp (Z-order)
    order = argsort boxes by morton_key(lo - bounding_box.lo)
    target = ba.num_cells() / nranks ; acc = 0 ; r = 0 ; rank[*] = 0
    for k, b in enumerate(order):
        rank[b] = r ; acc += ba[b].num_cells()
        if r < nranks-1 and acc >= target*(r+1) and boxes_left >= ranks_left:
            r += 1                                    # garantit >=1 box/rang
    return DistributionMapping(rank)

function fill_boundary_begin(mf, domain, per):        # halos, 2 phases
    shifts = product({0,+-Lx if per.x}, {0,+-Ly if per.y})
    hash = BoxHash(mf.box_array())                    # accelere la recherche de voisins
    for li in local fabs of mf:                       # --- copies locales ---
        gbox = grow(fab(li).box, ng)
        for (sx, sy) in shifts:
            for gB in hash.query( gbox shifted by (-sx,-sy) ):
                if gB local: copy_shifted(fab(li), fab(gB), region, sx, sy)
    if n_ranks() <= 1: return
    for gF in 0..ba.size()-1:                          # --- enum globale deterministe ---
        for (sx, sy) in shifts:
            for gB in hash.query(...):
                if owner(gF)==me xor owner(gB)==me:    # une extremite locale
                    classer en send[owner(gF)] ou recv[owner(gB)]
    pack(send buffers via for_each PackKernel) ; device_fence()
    post MPI_Isend / MPI_Irecv (tag 0, pointeurs unifies -> GPUDirect si CUDA-aware)

function fill_boundary_end(mf, h):
    MPI_Waitall(h.reqs)
    unpack(recv buffers via for_each UnpackKernel) -> ghosts
```

**Code.** [`mesh/box_array.hpp`](../include/adc/mesh/box_array.hpp) (`BoxArray::from_domain`,
`split_range`, l'ordre du vecteur est l'identite de box) ;
[`mesh/distribution_mapping.hpp`](../include/adc/mesh/distribution_mapping.hpp)
(`DistributionMapping`, round-robin `i % nranks` par defaut, metadonnee repliquee) ;
[`mesh/multifab.hpp`](../include/adc/mesh/multifab.hpp) (`MultiFab` n'alloue que les fabs ou
`dm_[i] == my_rank()`, itere sur `local_size()`, `global_index` / `local_index_of` font le pont) ;
[`mesh/fill_boundary.hpp`](../include/adc/mesh/fill_boundary.hpp) (`fill_boundary_begin` /
`fill_boundary_end` non-bloquants + `fill_boundary` bloquant, `HaloExchange` possede les tampons et
`MPI_Request`, kernels `CopyShiftedKernel` / `PackKernel` / `UnpackKernel` device-clean) ;
[`parallel/load_balance.hpp`](../include/adc/parallel/load_balance.hpp) (`morton_key`,
`make_sfc_distribution`, `make_knapsack_distribution`, `load_imbalance`) ;
[`parallel/comm.hpp`](../include/adc/parallel/comm.hpp) enveloppe les collectives
(`all_reduce_sum`, `all_reduce_sum_inplace` pour le reflux, `all_reduce_or_inplace` pour l'union des
tags de regrid) et degenere proprement en serie.

**Contraintes / remarques.** Les metadonnees (BoxArray + DistributionMapping) sont repliquees sur
tous les rangs : c'est ce qui rend l'enumeration des jobs de halo deterministe, donc les tampons
`sbuf[A->B]` et `rbuf[B<-A]` s'alignent sans negocier les tailles. Si MPI n'est pas initialise,
`my_rank()` rend 0 et `n_ranks()` rend 1 (un test serie linke contre la lib MPI ne casse pas), d'ou
`comm_init()` obligatoire au debut de `main()` pour un vrai run distribue. Les ghosts hors domaine
sans periodicite ne sont pas touches par `fill_boundary` (ce sont les CL physiques,
`physical_bc.hpp`). Les tampons vivent en memoire unifiee et sont passes tels quels a MPI (rebond
hote evite si la pile MPI est CUDA-aware) ; un `device_fence()` separe le pack des `Isend`.
**Validation.** `test_box_array`, `test_multifab`, `test_load_balance` ; sous MPI :
`test_mpi_fillboundary` (echange de halos), `test_mpi_poisson` (Poisson distribue),
`test_mpi_fft_distributed` (FFT par bandes), `test_mpi_redistribute`, `test_mpi_array_reduce`,
`test_mpi_coupler_inject` (np=4, resultats bit-identiques a np=1/2/4).

## 21. Canal aux extensible

**Intuition.** Le couplage hyperbolique-elliptique passe par un canal `aux` de contrat de base
`(phi, grad_x, grad_y)`, soit `kAuxBaseComps = 3` composantes. Certains modeles ont besoin de champs
supplementaires : un champ magnetique `B_z` hors-plan (force de Lorentz), une temperature
electronique `T_e` derivee de `p/rho`. Le canal est elargi a la demande, et reste bit-identique a
l'historique quand `n_aux = 3` (les composantes extra valent 0, jamais lues).

**Formule / discretisation.** Un modele declare un membre statique `n_aux > 3` ; la largeur
effective lue par l'operateur spatial et allouee par le runtime est

$$\mathrm{naux}(M) =
  \begin{cases} M\text{::n}_\text{aux} & \text{si } M \text{ le declare}\\ 3 & \text{sinon}\end{cases}$$

evaluee a la compilation par `aux_comps<M>()`. Une brique magnetisee pose `n_aux = 4` : `B_z` occupe
la composante 3, fonction pure de la position echantillonnee par niveau ; `T_e` (composante 4) est
derivee d'un bloc fluide. Un `CompositeModel<Hyp, Src, Ell>` propage la largeur aux maximale de ses
trois briques au systeme, qui alloue et marshale le bon nombre de composantes.

```
function aux_comps<M>():                       # physical_model.hpp, constexpr
    if requires { M::n_aux }: return M::n_aux   # ex. 4 si une brique lit B_z
    else: return kAuxBaseComps                  # 3 : phi, grad_x, grad_y

# cote bloc compile / runtime : on elargit le canal avant de capturer son adresse
function add_compiled_model(sys, name, model, ...):  # dsl_block.hpp
    sys.ensure_aux_width( aux_comps<Model>() )       # MultiFab aux >= naux comp (no-op si 3)
    ctx = sys.grid_context()                         # ctx.aux pointe l'aux reel, large
    install_block(...)                               # assemble_rhs lit load_aux<naux>

# cote ABI plate (marshaling) : le tableau aux_in porte exactement naux composantes
function make_grid(n, dx, dy, periodic, aux_in, naux):  # compiled_block_abi.hpp
    aux = MultiFab(ba, dm, naux, 1) ; aux.set_val(0)
    for c in 0..naux-1, j, i: aux(i,j,c) = aux_in[c*n*n + j*n + i]
    fill ghosts (memes CL que le System) -> load_aux<naux> lit B_z / T_e
```

**Code.** [`core/physical_model.hpp`](../include/adc/core/physical_model.hpp) :
`aux_comps<M>()` (detecte `M::n_aux` via `requires`, retombe sur `kAuxBaseComps = 3`), vit dans le
header contrat pour que `CompositeModel` propage `n_aux` sans tirer la numerique ; le concept
`PhysicalModel` impose `M::Aux == adc::Aux`. Cote dispatch virtuel,
[`runtime/dynamic_model.hpp`](../include/adc/runtime/dynamic_model.hpp) :
`IModel<NV>::n_aux()` (defaut `kAuxBaseComps`), `ModelAdapter<M>::n_aux()` renvoie `aux_comps<M>()`.
L'elargissement est ancre dans `System::ensure_aux_width` (appele par
[`runtime/dsl_block.hpp`](../include/adc/runtime/dsl_block.hpp) avant `grid_context()`), et le
marshaling plat dans [`runtime/compiled_block_abi.hpp`](../include/adc/runtime/compiled_block_abi.hpp)
(`make_grid(..., naux)`, symbole `adc_compiled_naux()` = `aux_comps<MODEL>()`).

**Contraintes / remarques.** L'elargissement doit precede la capture de l'adresse de l'aux (sinon la
fermeture lirait un aux trop court). `B_z` etant une fonction de la position, il est echantillonne
par niveau (exact au grossier comme `eps`, ordre 2 preserve). Un modele de base (`n_aux` non declare)
retombe sur 3 -> allocation et resultats bit-identiques a l'historique. **Validation.**
`test_aux_extra` (un modele declare `n_aux > 3`), `test_aux_composite` (un `CompositeModel` propage
la largeur aux de ses briques), `test_aux_coupler_bz` / `test_aux_system_bz` / `test_amr_aux_bz` /
`test_amr_system_bz_pop` / `test_amr_system_bz_multibox` (B_z lu et peuple le long des chemins
coupleur, systeme, AMR, multi-box), `test_aux_te` (T_e derive de `p/rho`), `test_aux_single_source`
(une source unique genere `load_aux` + marshaling, tous champs couverts). Valide bit-identique sur
GH200 (B_z device single + multi-box, cf. GPU_RUNTIME_PORT.md).

## 22. Composition runtime et systeme multi-especes

**Intuition.** Python compose quoi (un bloc par espece : modele compose + schema spatial + politique
temporelle), le C++ calcule par cellule. N especes interagissent dans le second membre elliptique
(`f = sum_s q_s n_s`) et dans la source inter-especes, jamais dans le flux : le flux d'un bloc ne
voit que son propre etat et l'aux partage.

**Formule / discretisation.** A chaque macro-pas, le systeme resout le Poisson partage dont le
second membre est la somme co-localisee des briques elliptiques de tous les blocs, peuple l'aux
`(phi, grad phi)`, puis avance chaque bloc selon sa politique (explicite / IMEX, substeps, stride) :

$$f_{ij} = \sum_{b} \mathrm{elliptic\_rhs}_b(U^b_{ij}),\qquad
  \frac{dU^b}{dt} = -\,\mathrm{div}\,F_b(U^b, \mathrm{aux}) + S_b(U^b, \mathrm{aux})$$

Un modele est assemble par `dispatch_model(spec, visitor)` : il construit la brique de transport
(`exb` / `compressible` / `isothermal`), la brique de source (`none` / `potential` / `gravity` /
`magnetic` / `potential_magnetic`, les sources fluides exigeant `NV >= 3`), la brique elliptique
(`charge` / `background` / `gravity`), les combine en `CompositeModel<TR, Src, Ell>` et appelle
`visitor(model)`. Le coeur ne nomme aucun scenario ; un scenario est cette composition, nommee cote
`adc_cases`.

```
function dispatch_model(spec, visitor):              # model_factory.hpp
    dispatch_transport(spec, TR ->                    # exb | compressible | isothermal
      dispatch_source<TR::n_vars>(spec, Src ->        # none | potential | gravity | (potential_)magnetic
        dispatch_elliptic(spec, Ell ->                # charge | background | gravity
          visitor( CompositeModel<TR, Src, Ell>{TR, Src, Ell} ))))
    # combinaison invalide (source fluide sur transport scalaire) -> throw

function System.step(dt):                            # runtime/system.hpp
    solve_fields()                                    # Poisson partage : f = sum_b elliptic_rhs_b
    fill aux (phi, grad phi[, B_z, T_e])
    for b in blocks:
        if not b.evolve: continue                     # espece gelee : vue par Poisson, non avancee
        if b held by stride this macro-step: continue # multirate hold-then-catch-up
        advance b (explicit SSPRK / IMEX, b.substeps sous-pas)

function System.step_cfl(cfl):
    dt = cfl * h * min_b( substeps_b / (stride_b * w_b) )   # honore substeps + stride
    step(dt) ; return dt
```

**Code.** [`runtime/system.hpp`](../include/adc/runtime/system.hpp) : `System` (multi-blocs
mono-niveau, Poisson partage, `add_block(name, ModelSpec, limiter, riemann, recon, time, substeps,
evolve, stride, implicit_vars, implicit_roles)`, `step` / `step_cfl`, `evolve=false` pour une espece
gelee vue par le Poisson) ; [`runtime/amr_system.hpp`](../include/adc/runtime/amr_system.hpp) :
`AmrSystem` (1 bloc -> `AmrCouplerMP<Model>` mono-modele historique ; `>= 2 add_block` -> moteur
`AmrRuntime` multi-blocs sur hierarchie partagee, meme BoxArray + DistributionMapping + dx/dy par
niveau via `same_layout_or_throw`, Poisson grossier somme co-localisee, conservation par bloc,
`add_coupled_source` pour les sources inter-especes, `n_blocks()`). Cote couplage :
`coupling/system_coupler.hpp` (`SystemAssembler` assemble, `SystemDriver` avance),
`coupling/amr_system_coupler.hpp` (le systeme porte sur AMR).
[`runtime/model_factory.hpp`](../include/adc/runtime/model_factory.hpp) :
`dispatch_model` / `dispatch_transport` / `dispatch_source` / `dispatch_elliptic` assemblent un
`CompositeModel` depuis une `ModelSpec` (le coeur ne nomme aucun scenario).

**Contraintes / remarques.** Les blocs partagent un aux et un Poisson ; le couplage entre especes
passe par le second membre elliptique (somme) et les sources couplees, pas par le flux. `substeps`
et `stride` sont orthogonaux (un bloc lent sur `stride=M` est tenu M-1 pas puis rattrape d'un pas
effectif `M*dt`) ; entre deux rattrapages le bloc tenu entre dans la somme du Poisson avec son etat
perime. En AMR multi-blocs, `regrid_every > 0` est refuse (le regrid d'union des tags est une PR
ulterieure) et `set_conservative_state` est mono-bloc seulement. Sans masque IMEX explicite
(`implicit_vars` / `implicit_roles` vides) le defaut du modele s'applique -> bit-identique.
**Validation.** `test_system_abstraction`, `test_system_coupler`, `test_two_species_minimal`,
`test_coupled_source` (source inter-especes), `test_system_two_explicit`, `test_assembler_driver`
(l'assembleur assemble, le driver avance), `test_amr_system_coupler`, `test_system_hardening`,
`test_variable_role` (adresser une composante par son role physique plutot que par indice).

## 23. DSL symbolique : codegen, JIT, AOT

**Intuition.** Un mini-DSL cote Python decrit un modele en formules et l'emet en brique C++ : flux,
source, second membre elliptique, avec elimination de sous-expressions communes (CSE). Trois chemins
de mise en oeuvre, du plus dynamique au plus natif, avec un compromis dispatch / performance / GPU
croissant.

**Formule / discretisation.** Le DSL emet un `CompositeModel<Hyp, Src, Ell>` (le meme type que la
composition native). Les trois branchements different par OU vit la frontiere et CE qui transite :

- JIT type-erased : le `.so` expose `IModel<NV>*` (dispatch virtuel par cellule). Chemin hote,
  Rusanov ordre 1, pour prototyper sans recompiler le coeur. `ModelAdapter<M>` enrobe le modele
  statique et forwarde `source` / `elliptic_rhs` / conversions quand `M` les expose (sinon defaut).
- AOT marshale : ABI `extern "C"` (`compiled_block_abi.hpp`). Le `.so` execute le chemin de
  production (`assemble_rhs<Limiter, Flux>`, SSPRK2/IMEX) sur le `CompositeModel` connu a SA
  compilation ; seuls des tableaux plats composante-majeur `c*n*n + j*n + i` traversent le `dlopen`
  (aucun symbole C++ partage). Pas d'AMR ni MPI. Les params runtime arrivent par les symboles
  suffixes `_p` (`pvals`, `npar`), seedes par les defauts de declaration.
- AOT natif : `add_compiled_model` branche le `CompositeModel` connu a la compilation comme bloc
  natif du `System`, sur le contexte DE grille reel (`grid_context`), sans marshaling : le residu
  fait `fill_boundary` (halos MPI) + `assemble_rhs` (Kokkos) sur les vrais MultiFab. Pour rester
  device-clean, le transport et le maillage passent par des foncteurs nommes (pas de lambdas
  etendues, que nvcc n'emet pas fiablement cross-TU).

```
function add_compiled_model(sys, name, model, lim, riem, recon, time, gamma, substeps, ...):  # dsl_block.hpp
    imex = (time == "imex") ; recon_prim = (recon == "primitive")
    method = "ssprk3" if time=="ssprk3" else "ssprk2"     # ignore en imex
    sys.ensure_aux_width( aux_comps<Model>() )            # canal aux assez large (B_z...)
    ctx = sys.grid_context()                              # vrais dom/CL/aux (zero-copie)
    clo = make_block(model, lim, riem, ctx, imex, recon_prim, method)   # foncteurs nommes
    ms  = make_max_speed(model, ctx)
    pr  = make_poisson_rhs(model)
    sys.install_block(name, Model::n_vars, conservative_vars, primitive_vars, gamma, clo, ms, pr,
                      substeps, evolve, stride)
    sys.set_block_conversion(name, to_primitive, to_conservative)   # cons<->prim du modele
    sys.set_block_ghosts(name, block_n_ghost(lim))         # weno5 -> 3 ghosts (sinon lecture hors bornes)

# AOT marshale : un .so genere se reduit a
#   using Model = adc::CompositeModel<...>;  ADC_DEFINE_COMPILED_BLOCK(Model)
# qui emet adc_compiled_residual/_advance/_max_speed/_poisson_rhs (+ variantes _p params runtime)
function residual<Model>(U, R, aux_in, n, dx, dy, periodic, lim, riem, recon_prim):  # compiled_block_abi.hpp
    lg = make_grid(n, dx, dy, periodic, aux_in, aux_comps<Model>())
    Umf = MultiFab(ncomp=n_vars, ghosts=block_n_ghost(lim)) ; fill_interior(Umf, U)
    clo = make_block(model, lim, riem, ctx, imex=false, recon_prim)
    clo.rhs_into(Umf, Rmf) ; extract(Rmf, R)               # device_fence() avant la lecture hote
```

**Code.**
- JIT : [`runtime/dynamic_model.hpp`](../include/adc/runtime/dynamic_model.hpp) (`IModel<NV>`
  virtuel, `ModelAdapter<M>`, `make_dynamic`) ; `System.add_dynamic_block` branche un modele a
  dispatch virtuel (chemin hote, Rusanov, prototypage).
- AOT marshale : [`runtime/compiled_block_abi.hpp`](../include/adc/runtime/compiled_block_abi.hpp)
  (`make_grid`, `fill_interior` / `extract`, `residual` / `advance` / `max_speed` / `poisson_rhs`,
  macro `ADC_DEFINE_COMPILED_BLOCK`, params runtime via `make_model_with_params` et les symboles
  `_p`) ; `System.add_compiled_block` (ABI `extern "C"`, sans AMR ni MPI).
- AOT natif : [`runtime/dsl_block.hpp`](../include/adc/runtime/dsl_block.hpp) (`add_compiled_model`,
  branche un `CompositeModel` connu a la compilation comme bloc natif, `ensure_aux_width` +
  `grid_context` + `make_block` + `install_block` + `set_block_ghosts`). La machinerie
  device-clean est `runtime/block_builder.hpp` (foncteurs nommes `BlockRhsEval`, `AdvanceExplicit`,
  `AdvanceImex`, `MaxSpeed` ; voir le seam section 24).

**Contraintes / remarques.** Le JIT type-erased coute un saut indirect par cellule (hors hot path
haute performance) ; l'AOT marshale recopie les tableaux a chaque appel mais reste mono-rang ; l'AOT
natif est le seul chemin zero-copie / GPU / MPI / AMR. La parite est verrouillee a chaque niveau.
**Validation.** Cote C++ : `test_dynamic_model` (modele type-erased == Euler statique),
`test_block_builder` (fermetures de bloc instanciables hors System), `test_compiled_model_parity`
(AOT natif == bloc natif sur CPU/Serial), `test_amr_compiled_model` (AOT natif sur hierarchie AMR).
Cote Python : la suite `test_dsl*` (codegen flux/source/elliptique, CSE, JIT `.so`, dispatch
type-erased, recon, roles, aux). Sur GH200, le chemin a foncteurs nommes est valide bit-identique
(GPU_RUNTIME_PORT.md, phase 9) ; `add_compiled_model` a lambdas etendues bute encore sur une limite
nvcc (phase 8).

## 24. Le seam de dispatch (serie / OpenMP / Kokkos / MPI)

**Intuition.** Pas un algorithme numerique mais le point de bascule qui les rend tous portables.
`for_each_cell(box, f)` dispatche la boucle sur les cellules d'une `Box2D` vers serie, OpenMP, ou
Kokkos selon le backend choisi A LA compilation ; les operateurs (assemble_rhs, V-cycle, coupleurs)
ne voient jamais le backend et on n'ecrit aucun kernel CUDA a la main. Detail dans
[ARCHITECTURE.md](ARCHITECTURE.md) section 4 (couche execution).

**Formule / discretisation.** Le foncteur `f(i, j)` est pris par valeur et ne capture que des
handles `Array4` (POD), jamais le `Fab` ni rien de virtuel : exactement la contrainte d'un kernel
device. Sous Kokkos il devient `parallel_for(MDRangePolicy<Rank<2>, IndexType<int>>)` (indices
signes pour les boites de ghosts a bornes negatives) ; sous OpenMP `#pragma omp parallel for
collapse(2)` ; sinon une double boucle sequentielle. Bit-identite : `for_each_cell` n'a aucune
dependance inter-iteration (chaque `f(i,j)` ecrit la seule cellule `(i,j)` et lit des cellules qu'il
n'ecrit pas dans le meme appel : smoother GS rouge-noir, residu/restriction/prolongation ecrivent
une destination distincte), donc le resultat est independant de l'ordre. Les reductions portent un
choix FP : la somme Kokkos reassocie l'addition (non associative en IEEE754), donc `sum` n'est pas
bit-identique a la boucle hote sous Kokkos ; serie et OpenMP gardent la boucle sequentielle (pas de
`reduction(+:)`) donc restent exacts. Le max est exact partout (associatif/commutatif, sans
arrondi). Un seuil `ADC_FOREACH_SERIAL_THRESHOLD` (defaut 4096 cellules) bascule en serie les
petites boites (niveaux grossiers du V-cycle ~2x2..32x32) ou le fork/join ecraserait le calcul, mais
uniquement si l'espace d'execution Kokkos par defaut est l'espace hote (`if constexpr` : sur device,
parallel_for quelle que soit la taille, sinon course de donnees).

```
function for_each_cell(box b, f):                    # for_each.hpp
    if Kokkos and DefaultExecSpace == DefaultHostExecSpace:   # if constexpr
        if (b.nx * b.ny) < foreach_serial_threshold():
            for j in b: for i in b: f(i, j)          # boucle hote, bit-identique
            return
    if Kokkos:  parallel_for( MDRangePolicy<Rank<2>, IndexType<int>>(lo, hi+1), f )
    elif OpenMP: #pragma omp parallel for collapse(2) if(nx*ny >= 4096)
                 for j: for i: f(i, j)
    else:        for j: for i: f(i, j)

function for_each_cell_reduce_sum(b, f):             # Kokkos::Sum deterministe par tuile
    Kokkos:  parallel_reduce(..., acc += f(i,j), Sum<Real>)   # non bit-id a la serie
    else:    acc = 0 ; for j: for i: acc += f(i, j)           # serie/OpenMP exacts

function sync_host():  device_fence()                # avant un acces hote (memoire unifiee)
function sync_device(): pass                          # no-op sous SharedSpace (scaffolding)
```

**Code.** [`mesh/for_each.hpp`](../include/adc/mesh/for_each.hpp) : `for_each_cell` (dispatch
Kokkos / OpenMP / serie, garde `if constexpr` device, seuil `foreach_serial_threshold`),
`for_each_cell_reduce_sum` / `_max` (reducteurs `Kokkos::Sum` / `Max` deterministes), les variantes
a foncteur reducteur `reduce_sum_cell` / `reduce_max_cell` (passees directement a `parallel_reduce`
sans lambda d'enveloppe, chemin device-clean cross-TU pour un noyau Model-template), et le seam de
coherence `sync_host()` (= `device_fence()` cible) / `sync_device()` (no-op sous memoire unifiee).
Les fabs et la reduction `sum(MultiFab)` (all-reduce sur tous les rangs) vivent dans
[`mesh/multifab.hpp`](../include/adc/mesh/multifab.hpp). Les collectives MPI sont enveloppees dans
[`parallel/comm.hpp`](../include/adc/parallel/comm.hpp) (`all_reduce_sum`, `all_reduce_max`,
`all_reduce_sum_inplace`, `all_reduce_or_inplace`, `barrier`, `comm_init` / `comm_finalize`), qui
degenerent en identite serie.

**Contraintes / remarques.** Le passage CPU -> GPU ne change aucun site d'appel : on remplace le
backend ici, la physique reste inchangee. Le foncteur doit etre device-callable sous Kokkos (annote
`ADC_HD`, capture des POD par valeur) ; capturer un objet a vtable ou un `Fab` casse le device. La
bascule serie du seuil n'est sure que sous execution hote (le `if constexpr` s'evapore sur device,
zero surcout, chemin GPU strictement inchange). Discipline GPU : `device_fence()` (via `sync_host`)
entre un kernel device et une boucle hote sur la meme memoire unifiee, sinon course
ecriture-hote / kernel (cf. CHOICES.md). **Validation.** Le seam est exerce transversalement par
toute la suite ; specifiquement les tests MPI de la section 20 (`test_mpi_fillboundary`,
`test_mpi_poisson`, `test_mpi_array_reduce`, np=1/2/4 bit-identiques) et les validations device
GH200 (GPU_RUNTIME_PORT.md) qui confirment que serie, OpenMP et Kokkos donnent les memes resultats
(au choix FP de la somme Kokkos pres, documente).


---

## Quel schema ou solveur quand

| Probleme | Choix | Pourquoi |
|---|---|---|
| transport hyperbolique general | volumes finis Godunov + flux Rusanov | robuste, marche pour toute equation (section 1, 2) |
| Euler compressible avec chocs | reconstruction primitive + HLLC ou Roe | resout le contact, moins diffusif que Rusanov (section 2, 3) |
| zones lisses, haute precision | WENO5-Z + SSPRK3 | ordre 5, peu dissipatif (section 3, 4) |
| source raide (Lorentz, relaxation) | IMEX local, ou condensation de Schur globale | implicite, pas de pas de temps explosif (section 5, 13) |
| Poisson periodique, $n = 2^k$ | `poisson_fft_solver` | direct, $O(N \log N)$ (section 10) |
| Poisson avec paroi, Dirichlet, ou $\varepsilon(x)$ | `geometric_mg` | multigrille, geometrie quelconque (section 9, 11) |
| operateur a tenseur plein (anisotrope, polaire) | `krylov_solver` (BiCGStab matrice-libre) | aucun assemblage de matrice (section 12, 16) |
| feature localisee (front, anneau) | `AmrSystem` + `set_refinement` | raffinement adaptatif, reflux conservatif (section 17 a 19) |
| sources inter-especes | `CoupledSource` (bytecode) | conservatif par construction (section 22) |
| domaine non rectangulaire | cut-cell EB (disque) ou anneau polaire | bord courbe sans escalier (section 14 a 16) |

## References

- Volumes finis et flux de Riemann : LeVeque, *Finite Volume Methods for Hyperbolic Problems*,
  Cambridge, 2002. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer, 2009
  (HLLC, Roe).
- Reconstruction WENO : Jiang & Shu, *Efficient Implementation of Weighted ENO Schemes*, JCP 126
  (1996). WENO-Z : Borges et al., JCP 227 (2008).
- Integration SSP : Gottlieb, Shu, Tadmor, *Strong Stability-Preserving Time Discretization Methods*,
  SIAM Review 43 (2001).
- IMEX : Ascher, Ruuth, Spiteri, *Implicit-explicit Runge-Kutta methods*, Appl. Numer. Math. 25 (1997).
- Multigrille : Briggs, Henson, McCormick, *A Multigrid Tutorial*, SIAM, 2000.
- Krylov : Saad, *Iterative Methods for Sparse Linear Systems*, SIAM, 2003 (BiCGStab, section 7).
- AMR : Berger & Oliger, *Adaptive mesh refinement for hyperbolic partial differential equations*, JCP
  53 (1984). Berger & Colella, *Local adaptive mesh refinement for shock hydrodynamics*, JCP 82 (1989).
  Berger & Rigoutsos, *An algorithm for point clustering and grid generation*, IEEE Trans. SMC 21 (1991).
- Cut-cell : Shortley & Weller, *The numerical solution of Laplace's equation*, J. Appl. Phys. 9 (1938).
- Schur condense (Euler-Poisson magnetise) : voir [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md) et
  [HOFFART_FIDELITY.md](HOFFART_FIDELITY.md).

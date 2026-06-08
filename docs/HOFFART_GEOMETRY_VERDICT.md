# Verdict experience geometrie (juin 2026) -- le cut-cell ne recupere PAS le taux

Experience discriminante sur ROMEO GH200 (job 647507), modele COMPLET system-schur,
n=256, t_end=2.0, fenetres papier, l=3,4,5, trois geometries de transport :
square (boite carree, defaut), staircase (masque disque 0/1 a R=16), cutcell
(embedded-boundary aperture+kappa a R=16).

## Resultat brut (taux gamma_numeric, brut, fenetres papier)

| mode | square | staircase | cutcell | papier | erreur |
|------|--------|-----------|---------|--------|--------|
| 3 | 0.037182 | 0.037182 | 0.037182 | 0.772 | -95.2 % |
| 4 | 0.048897 | 0.048897 | 0.048897 | 0.911 | -94.6 % |
| 5 | 0.121080 | 0.121080 | 0.121080 | 0.683 | -82.3 % |

Les trois geometries donnent le MEME taux (differences ~1e-11 = arrondi machine ;
le masque/EB est bien ACTIF mais physiquement sans effet). CONCLUSION DIRECTE :
le masque/cut-cell de domaine au bord externe R=16 NE CHANGE PAS le taux diocotron.

## Pourquoi le cut-cell-a-R etait le mauvais outil

L'instabilite diocotron vit sur l'anneau r0=6 / r1=8, PROFONDEMENT a l'interieur du
domaine R=16. Le masque de disque agit au bord externe R : il ne touche que les
coins (rayon > 16), qui portent rho_min et sont dynamiquement inertes. Le mur de
Poisson (Dirichlet sur le cercle R=16) impose deja le disque pour phi. Donc
confiner le transport a ||x|| < 16 ne change rien a la dynamique de l'anneau.

## Le point qui REORIENTE le diagnostic : deficit RESOLUTION-INDEPENDANT

Le deficit du modele complet est ~ -95 % a n=256 ET a n=384 (quasi identique). Une
DIFFUSION de bord d'anneau (lissage de l'anneau net par la grille cartesienne)
DIMINUERAIT avec la resolution (cellule plus petite => anneau moins lisse). Le
deficit ne bouge PAS avec n. Donc :

- ce n'est PAS (seulement) une diffusion de bord d'anneau resolvable par n plus
  grand (la voie "cartesien haute-res n=768/1024" ne suffira probablement pas) ;
- ce n'est PAS la geometrie du bord externe (cut-cell sans effet) ;
- c'est donc tres probablement STRUCTUREL : normalisation / echelle de temps /
  couplage du chemin system-schur complet. Le taux brut ~0.037 est un PLATEAU.

## Contraste avec le modele REDUIT (qui, lui, recupere)

Le modele REDUIT (derive ExB scalaire) :
- sur grille POLAIRE : l=4 EXACT (0.913 vs 0.911), l=3/5 proches (diag_polar_omega) ;
- sur grille CARTESIENNE : -5 a -27 % a n=192 minmod, et AMELIORE avec ordre/resolution
  (sweep WENO5) -> comportement de diffusion classique, resolution-DEPENDANT.

Le modele complet (resolution-independant a -95 %) se comporte DIFFEREMMENT du reduit.
Le facteur brut entre reduit-polaire (0.155) et complet-cartesien (0.037) est ~4x,
proche du facteur de diffusion d'anneau invoque en M1, MAIS l'independance en
resolution du complet contredit une explication purement diffusive.

## Verdict honnete

- Le cut-cell (#218/#222/#224) est une vraie capacite testee (MMS ordre ~2, masse
  conservee) mais NE corrige PAS le taux diocotron : mauvais outil pour ce verrou.
- Le bug ABI natif (#225) est corrige (runs natifs GH200 possibles) ; le cas est
  executable en natif (DISC #14). Ces gains d'ingenierie restent valides.
- Le deficit du modele complet est RESOLUTION-INDEPENDANT et GEOMETRIE(bord)-
  INDEPENDANT => suspect = STRUCTUREL (normalisation / echelle / couplage du
  system-schur complet), PAS la diffusion de maille ni le bord externe.
- AUCUNE reproduction du modele complet revendiquee.

## Prochain pas recommande (diagnostic, pas gros GPU)

Isoler le facteur structurel : comparer, sur un MEME setup minimal, le taux brut du
chemin system-schur COMPLET vs le chemin reduit ExB, pour localiser d'ou vient le
plateau ~0.037 (normalisation 2pi/echelle de temps du complet ? force de couplage
Schur ? vitesse de derive initiale ?). C'est une etude de normalisation/structure,
pas une montee en resolution ni une nouvelle geometrie.

## MAJ : tentative VOIE 1 (modele complet sur grille polaire) -- mur de WELL-BALANCING

Le chemin polaire (anneau r0/r1 resolu par un axe de grille) a ete assemble (PR adc_cases
#18 : fluide isotherme polaire #209 + Lorentz + Schur polaire #215 + Poisson polaire ;
observable phi sur r=r0). Il S'ASSEMBLE et DEMARRE mais diverge avant la fenetre de fit.
Caracterisation a 3 niveaux :
1. NaN a t~0.02 ; dt plus petit ne fait que RETARDER (t=0.02 -> 0.101 a dt=1e-4) -> PAS le CFL.
2. IC d'equilibre rotatif derivee (bilan radial : centrifuge rho v_theta^2/r - d_r p
   - rho d_r phi + rho B_z v_theta = 0 ; racine ExB-continuee ; signes verifies vs le moteur,
   PR adc_cases #20). Correcte dans le CONTINU.
3. MAIS l'equilibre continu n'est PAS discretement stationnaire : un run delta=0 (sans
   perturbation, nr=256) fait croitre TOUS les modes azimutaux de 0 a ~1e9 en 200 pas.
   Les operateurs discrets (source centrifuge polar_geom_source vs divergence de flux ;
   et/ou l'etage Schur) ne preservent pas le bilan continu.

VERDICT : le fluide polaire complet aux parametres raides exige un SCHEMA WELL-BALANCED
(qui preserve discretement l'equilibre rotatif source-equilibre) -- un vrai chantier CFD,
pas un knob ni une IC continue. Le modele REDUIT ExB scalaire l'evite (pas d'equation de
moment) et c'est pourquoi LUI recupere l=4 exact en polaire : la preuve que la resolution
d'anneau est la clef existe, mais le fluide COMPLET ne tourne pas stable sans well-balancing.
Chantier en cours : workflow polar-wellbalanced (diagnostic du residu discret + fix
well-balanced + test de stationnarite delta=0). AUCUNE reproduction du modele complet
revendiquee (ni cartesien -82/-95%, ni polaire bloque).

## Acquis d'ingenierie de la campagne (independants du verdict scientifique)
- Schur polaire MULTI-RANG MPI (#227 merge, plus mono-rang ; parite np=1/2/4 ~1e-13) ;
  extension MULTI-BOX (#229, fix Kokkos en cours).
- cut-cell EB (#218/#222/#224), Strang generique (#217), fix ABI natif GH200 (#225,
  + test CI de non-regression), cas hoffart executable en natif (DISC #14).

## MAJ : VOIE 1 (modele complet polaire) -- option (c) frozen-equilibrium, 4 campagnes GH200

Suite au verrou de well-balancing identifie plus haut, l'option (c) frozen-equilibrium a ete
livree et validee sur ROMEO GH200 (mono-rang, Kokkos). Principe : on precalcule le residu
d'equilibre GELE R_eq = step(U_eq) - U_eq UNE FOIS sur l'anneau axisymetrique (perturbation
nulle), puis on avance la carte CORRIGEE U <- step(U) - R_eq. Par construction
(step - R_eq)(U_eq) = U_eq : l'equilibre axisymetrique devient un point fixe discret exact.

### Ce qui est ETABLI (robuste)

1. WELL-BALANCING AXISYMETRIQUE RESOLU. Campagne 1 (delta=0, frozen, nr=ntheta=256, dt=1e-3,
   >=200 pas) : max||U^n - U_eq||_inf = 4.150e-20, tres en dessous du floor
   C*eps*||U_eq||_inf = 2.320e-13 (||U_eq||_inf = 1.045). U_eq est donc un point fixe discret
   exact a la precision machine. Corollaire : step() sur GPU est DETERMINISTE (sinon R_eq ne
   s'annulerait pas a 4e-20). NB : ce check est en partie tautologique (R_eq := step(U_eq) - U_eq)
   et ne prouve PAS la stabilite du linearise : il etablit seulement que le bilan axisymetrique
   O(1) est annule (||R_eq||_inf = 83.6 a n=256) et que step() est reproductible.

2. LE CHEMIN PERTURBE DIVERGE QUAND MEME, avant la fenetre de mesure. Le blow-up (NaN) survient
   a t ~ 0.01, soit ~100x plus vite que la fenetre O(1) du mode diocotron (gamma_papier ~
   0.772/0.911/0.683 pour l=3/4/5 en unites omega_d=1). Le frozen-eq corrige exactement le bilan
   axisymetrique mais n'empeche PAS la divergence de la perturbation non-axisymetrique.

3. LA DIVERGENCE EST INDEPENDANTE DE L'AMPLITUDE INITIALE delta. Campagne 2 (l=3, frozen,
   dt=1e-3, t_end=2, nr=256) : temps de mort ~ 0.01-0.02 pour delta de 1e-1 a 1e-5 (5 ordres de
   grandeur). Cela EXCLUT une loi puissance pilotee par l'amplitude (un forcage residuel O(delta)
   sans operateur instable donnerait t_mort ~ 1/delta). ATTENUATION : la mesure (mort a ~10-20 pas
   sur une grille d'echantillonnage t_end=2) est trop grossiere pour distinguer une stricte
   independance d'un shift logarithmique t_mort ~ -(1/sigma) ln(delta). Enonce correct :
   COMPATIBLE avec une instabilite lineaire (independante OU log-delta), EXCLUT une loi puissance.
   Ce pilier ne discrimine PAS a lui seul "numerique" de "physique" (la phase lineaire d'un mode
   physique est aussi delta-independante) : il etablit la linearite, rien de plus.

4. RAFFINER dt NE CORRIGE PAS (a operateur fixe). Campagne 3 (l=3, delta=0.1, frozen, nr=256,
   t_end=0.05) : le NOMBRE de pas avant mort croit ~lineairement en 1/dt (~9 pas a dt=1e-3 ;
   ~15462 pas a dt=1e-6, soit ~10x par decade), ce qui EXCLUT une instabilite temporelle a
   nombre-de-pas fixe. ATTENUATION IMPORTANTE : le temps de mort PHYSIQUE n'est PAS plat ; il
   croit encore de +29% sur la derniere decade (0.01308 a dt=1e-5 -> 0.015462 a dt=1e-6). Une
   extrapolation geometrique des increments suggere une limite finie t_inf ~ 0.018, mais elle
   repose sur UN SEUL ratio d'increments (3 points -> 2 gaps) et le point dt=1e-3 est meme
   non-monotone. "dt-converge" est donc un abus de langage : ce qui est demontre est
   "raffiner dt a operateur identique RETARDE la mort mais ne l'evite pas". Point qui sauve la
   portee pratique : meme la limite la plus genereuse t_inf ~ 0.018 ne represente que ~0.01-0.02
   e-folding du mode diocotron (1/gamma ~ 1.1-1.5), soit 50-70x trop court pour mesurer le taux.
   Aucun raffinement en dt n'ouvre de fenetre exploitable.

5. LA DIVERGENCE EMPIRE AVEC LA RESOLUTION. Campagne 4 (l=3, delta=0.1, frozen, dt=1e-4,
   t_end=0.05) : n=128 -> mort t=0.035, ||R_eq||=40.6 ; n=256 -> 0.0083, ||R_eq||=83.6 ;
   n=512 -> 0.0017, ||R_eq||=165.1. Pente log-log d(log t_mort)/d(log N) ~ -2.2 (mort ~ 1/N^2),
   ||R_eq|| croit ~lineairement en N. La perturbation est injectee a un nombre d'onde PHYSIQUE
   FIXE (l=3, l/r0 ~ 0.5, N-independant) : un mode PHYSIQUE de l=3 a gamma fini convergerait
   avec h (taux qui se stabilise quand N croit). Ici le taux effectif CROIT sans saturation :
   c'est la signature OPPOSEE d'un mode physique convergent, et c'est le pilier le plus probant.
   ||R_eq|| qui croit avec N corrobore (erreur de troncature non bornee) sans etre une preuve
   autonome. NB : ||R_eq|| a n=256 vaut 83.6 a la fois a dt=1e-3 (campagne 1) et a dt=1e-4
   (campagne 4) : le terme dominant du residu est O(1) (source de Lorentz omega_c=1e12 condensee
   par Schur), PAS O(dt) -- ce qui renforce le point 4 et oriente vers la source raide.

### Diagnostic (le verrou : instabilite spatiale semi-discrete, mais cause racine non close)

La divergence du chemin perturbe polaire complet est imputee a une INSTABILITE DE L'OPERATEUR
SPATIAL SEMI-DISCRET (WENO5-Z + Rusanov + source geometrique polaire 1/r + couplage Schur) a la
raideur papier. Les piliers 3 et 5 (linearite + aggravation ~1/N^2 a nombre d'onde physique fixe)
sont les plus solides ; les piliers 1, 2 et 4 sont corroborants mais non discriminants pris
isolement. Le frozen-eq corrige exactement le bilan axisymetrique O(1) mais ne peut rien sur
l'operateur agissant sur la perturbation non-axisymetrique : c'est coherent avec la persistance
de la divergence.

ATTENUATION sur "mal pose" : nous EVITONS le terme fort de mal-pose au sens de Hadamard. Aucune
mesure directe (spectre du jacobien, donnee lisse) ne l'a etabli, et un mecanisme concurrent
credible reproduit TOUTES les observations sans invoquer une mal-pose intrinseque : un schema
CONSISTANT mais NON PRESERVATEUR DE POSITIVITE echouant sur la donnee a quasi-vide. La densite
initiale a un contraste 1e6 (anneau rho_max=1 vs halo rho_min=1e-6, model.py:38-39) ;
RusanovFlux est composante-par-composante sans floor (numerical_flux.hpp:52-67) ; la source
geometrique et to_primitive divisent par rho (hyperbolic.hpp:228-238, 143-148). WENO5
reconstruisant un saut 1e6 sur ~1 cellule peut produire un rho (ou p=cs2 rho) reconstruit
NEGATIF -> 1/rho et pression a signe inverse -> anti-diffusion locale -> singularite en temps
fini. Ce mecanisme explique aussi le pilier 5 (gradient plus raide a haut N -> overshoot plus
grand -> t_mort ~ 1/N^2) et le fait que le modele REDUIT survive (rho y est scalaire passif :
ni 1/rho, ni pression, ni moment, donc aucun overshoot vers rho/p < 0 possible). Enonce retenu :
operateur spatial semi-discret INSTABLE a la raideur papier sur le chemin non-axisymetrique,
probablement via une RECONSTRUCTION NON POSITIVE au bord d'anneau raide (et/ou la source de
Lorentz raide condensee). Cela reste un defaut du SCHEMA SPATIAL (esprit de la conclusion intact),
mais nous ne revendiquons pas la mal-pose generique.

ATTENUATION sur "PAS IMEX" : le balayage en dt garde l'operateur ET la condensation de Schur
identiques (stepper theta=0.5, Crank-Nicolson, NON L-stable). Il exclut "dt plus petit a
operateur identique", PAS un traitement temporel DIFFERENT de la source raide. Un schema L-stable
(theta=1 / BDF / IMEX raide-stable) ou une integration EXACTE/exponentielle de la rotation de
Lorentz ne corrigerait pas un schema spatial defaillant mais POURRAIT MASQUER le blow-up en
amortissant le mode-maille (fausse stabilisation). Nous n'affirmons donc PAS "IMEX ne change
rien" ; nous affirmons que la cure correcte est SPATIALE.

### Correctif actionnable

Le verrou est dans le SCHEMA SPATIAL semi-discret, pas dans un knob, pas dans dt, pas (au sens
d'une vraie correction) dans un IMEX. Redesign cible, par ordre de probabilite suggere par le
mecanisme concurrent :
1. RECONSTRUCTION / RIEMANN PRESERVANT LA POSITIVITE au bord d'anneau (limiteur de positivite
   sur rho et p reconstruits, etat de Riemann vide-compatible, floor de densite). C'est la piste
   la plus directe vu le contraste 1e6 et l'absence de garde-fou actuelle.
2. Traitement WELL-BALANCED + dissipation/upwinding STABLE de la source geometrique 1/r et de la
   source de Lorentz raide.
Un schema temporel L-stable/exponentiel pour la source raide est utile MAIS doit etre verifie
comme une vraie cure (et non un masquage du mode-maille) en mesurant si le taux diocotron lent
O(1) emerge effectivement.

### Pourquoi le modele REDUIT ExB y echappe

Le modele reduit (derive ExB scalaire, l=4 EXACT 0.913 vs 0.911 papier en polaire) n'a NI
equation de moment, NI source de Lorentz raide, NI 1/rho, NI pression : rho y est un scalaire
passif transporte. Aucun overshoot de reconstruction vers rho/p negatif n'est possible, et il
n'y a pas de source algebrique raide a condenser. C'est exactement ce qui rend l'operateur
spatial du chemin complet fragile et celui du reduit robuste.

### Incertitudes residuelles (honnetes)

- Test discriminant DECISIF non realise : spectre du jacobien semi-discret D[step](U_eq) pour
  N=128/256/512. Si max Re(lambda) ~ +C/h^2 avec vecteur propre dent-de-scie a k_Nyquist ->
  instabilite numerique confirmee ; si Re(lambda) sature avec vecteur propre lisse a k fixe ->
  mode physique. La pente -2.2 predit le premier cas mais ne le mesure pas.
- Sonde directe non faite : min(rho), min(p reconstruit) et LOCALISATION du premier NaN (cellule
  (i,j)) juste avant la mort. Au bord r=r1=8 -> mecanisme vide/positivite ; reparti -> mal-pose
  plus generique.
- Tests de regularisation non faits : (a) hyperdiffusion eps*h^p ou viscosite artificielle ;
  (b) limiteur de positivite + floor ; (c) contraste reduit (rho_min=0.1) ou anneau lisse (tanh)
  au lieu de top-hat ; (d) donnee lisse bornee sans quasi-vide. Si l'un fait emerger le taux lent
  O(1), le correctif annonce est valide.
- dt-sweep non etendu (1e-7, 1e-8) ni delta-sweep refait a dt fin : le plateau t_inf ~ 0.018
  n'est pas etabli (1 seul ratio d'increments). Cela ne change PAS la portee pratique (fenetre
  toujours 50-70x trop courte).
- Ablation composante par composante (WENO5 seul -> +source 1/r -> +Schur -> +Lorentz raide) non
  faite : elle identifierait QUEL terme rend l'operateur instable et validerait la prescription.

### Statut

AUCUNE reproduction du modele complet revendiquee (ni cartesien -82/-95%, ni polaire qui diverge).
Le well-balancing AXISYMETRIQUE est RESOLU (point fixe discret exact, 4e-20). Le verrou restant
est l'INSTABILITE de l'operateur spatial semi-discret sur le chemin perturbe non-axisymetrique a
la raideur papier, dont la cause racine la plus probable est une reconstruction non positive au
bord d'anneau raide (et/ou la source de Lorentz raide). Le modele reduit ExB recupere le taux en
polaire et reste la voie de reproduction credible ; le fluide complet exige un redesign spatial
avant tout fit de taux.

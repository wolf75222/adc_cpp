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

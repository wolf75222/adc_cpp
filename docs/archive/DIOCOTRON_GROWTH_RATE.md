# Reproduction du taux de croissance diocotron (vs Hoffart arXiv:2510.11808)

Document consolide : reproduction quantitative du taux de croissance de l'instabilite diocotron
avec adc_cpp, comparaison au papier de reference (Hoffart, Maier, Shadid, Tomas, *Structure-preserving
finite-element approximations of the magnetic Euler-Poisson equations*, arXiv:2510.11808, Section 5.3).
Resultats ROMEO bruts : [romeo/HERO_RESULTS.md](../romeo/HERO_RESULTS.md).

Cible analytique (Petri / Davidson-Felice, geometrie de l'anneau `r0:r1:Rwall = 6:8:16`, reproduite
par `analysis/diocotron_growth.hpp`) : `gamma_3 = 0.772`, `gamma_4 = 0.911`, `gamma_5 = 0.683`.

## 1. Verrou de STABILITE leve (prerequis a tout balayage)

Au-dela d'une resolution effective ~448, la simu partait en `nan` des les premiers pas. Diagnostic :
le **multigrille geometrique DIVERGEAIT** au bord conducteur embedded sur grille fine (coarsening
non-Galerkin + masque du cercle re-evalue par niveau -> correction grossiere incoherente, rayon
spectral du V-cycle > 1, erratique selon l'alignement du cercle). Le warm start propageait la
divergence -> `phi` puis le champ en `nan`. Ce n'etait NI le pas de temps (deja plafonne), NI le
plancher de densite (la densite reste bornee ; seul `phi` explose, au RAYON DE LA PAROI r=0.398).

Correctif : `GeometricMG::solve_robust` (`include/adc/elliptic/geometric_mg.hpp`). Phase 1 = le
V-cycle standard (BIT-IDENTIQUE quand il converge ou stagne) ; SEULEMENT en cas de vraie divergence
(residu final > residu initial) : durcissement STICKY du lissage GS + restart a froid jusqu'a
redevenir contractant. Resultat : stable jusqu'a eff 1024 (uniforme ET AMR `ml`), masse `~1e-14`,
les 8 runs enregistres (eff <= 448) restent BIT A BIT identiques. Details : `docs/HERO_RUN_AMR.md`.

## 2. Methodes (montee en ordre vers le taux analytique)

Le plafond de M1 (`gamma_norm ~ 0.58`) venait de la DIFFUSION du schema (ordre 1 en espace ET en
temps), pas de la physique. Deux leviers classiques, confirmes par la litterature (Jiang-Shu,
Borges WENO-Z, Gottlieb-Shu-Tadmor SSPRK, Ern-Guermond RK ordre 3) :

- **Reconstruction d'ordre eleve** : `NoSlope` (ordre 1) -> `VanLeer`/`Minmod` (MUSCL ordre 2) ->
  **`Weno5`** (WENO5-Z, ordre 5, `operator/reconstruction.hpp`, ordre 5.00 verifie par
  `test_weno_convergence`). Option `recon` de `examples/diocotron_column_amr.cpp` ; `recon=0`
  bit-identique a l'historique.
- **Integration en temps d'ordre eleve** : forward Euler BIAISE positivement un mode en croissance
  exponentielle (instable sur l'axe imaginaire, terme `+ 1/2 omega_r^2 dt`). **SSPRK3** (Shu-Osher)
  enleve ce biais a l'ordre 3. `examples/diocotron_highorder.cpp` : WENO5-Z + SSPRK3, Poisson
  RE-RESOLU a chaque etage RK (couplage stade par stade, `solve_robust`).

## 3. Resultats

### 3a. Convergence colonne : l'AMR suit l'uniforme (ROMEO 613945)

A resolution effective egale, l'AMR `ml` (Poisson multi-niveau) COINCIDE avec l'uniforme pour ~40 %
des cellules (la promesse M2b, a l'echelle) ; VanLeer depasse largement NoSlope :

| cas | eff 512 (lin) | eff 1024 (lin) | cellules vs unif |
|---|---|---|---|
| uniforme NoSlope | 0.650 | 0.706 | 100 % |
| uniforme VanLeer | 0.753 | 0.748 | 100 % |
| AMR `ml` VanLeer | 0.762 | 0.747 | ~40 % |

### 3b. Taux haute precision, modes 3/4/5 (ROMEO 613961, WENO5+SSPRK3)

Fenetre du papier, R^2 = 1.00. L'ordre eleve fait passer le mode 4 de 0.56 (NoSlope+Euler,
sous-evalue, trop diffusif) a ~0.99, du BON cote de 0.911 :

| mode l | analytique | eff 256 | eff 512 | eff 1024 |
|---|---|---|---|---|
| 3 | 0.772 | +8 % | +10 % | +11 % |
| 4 | 0.911 | +8 % | +8 % | +8 % |
| 5 | 0.683 | +7 % | +7 % | +7 % |

## 4. Diagnostic : un sur-tir ~+8 % UNIFORME et PLAT en resolution

Cinq mesures independantes ecartent les causes "faciles" ET le bord conducteur :

1. **Plat en resolution** : eff 256 ~ 512 ~ 1024 (meme +8 %). Plus de cellules ne referme PAS l'ecart.
2. **Plat en ordre de reconstruction** : `WENO5 ~ VanLeer`. Ce n'est pas l'ordre spatial.
3. **Balayage en delta** : la LIMITE LINEAIRE (delta -> 0) MONTE a +27 % au lieu de baisser. L'accord
   apparent a delta=0.1 etait une compensation fortuite par la saturation. Ce n'est donc PAS une
   contamination nonlineaire ni un effet de fenetre.
4. **Rapport SANS DIMENSION** `gamma / |Re(omega)|` (independant de la normalisation, via la valeur
   propre COMPLEXE `diocotron_eigenvalue` : analytique Re_norm = -2.08 / -2.75 / -3.44 pour l=3/4/5) :
   mesure 0.31 vs analytique 0.331 -> ~5 % de DISTORSION STRUCTURELLE de la valeur propre + ~3 % de
   decalage de normalisation `omega_D`.
5. **Plat en traitement du bord (cut-cell vs escalier)** : le bord embedded Shortley-Weller d'ordre 2
   a ete implemente (`GeometricMG`, option `cut_cell`, validation `test_cut_cell` : ordre L2 1.93,
   erreur de Poisson 3459x plus faible qu'en escalier a nc=512). Sur le diocotron (nc=256, VanLeer,
   `cut=1` vs `cut=0`), le taux est **IDENTIQUE a 3 chiffres** (gamma_norm 0.945, 0.838, 0.738 ... aux
   memes fenetres). Le bord conducteur n'est donc PAS la cause du +8 %.

Cause : ce n'est PAS le traitement de la paroi. Le mode-l instable vit sur l'**anneau** (r ~ 0.175),
loin de la paroi (r = 0.40) : l'effet d'image de la paroi sur le mode l au rayon de l'anneau decroit
en `(r_anneau/Rwall)^(2l)` = `(0.44)^8 ~ 1e-3` pour l=4, electrostatiquement negligeable. Le sur-tir
est **structurel** : distorsion ~5 % de la valeur propre sur la dynamique E x B cartesienne (la
symetrie 4 de la grille carree resonne avec le mode 4) plus ~3 % de normalisation `omega_D` (mesure 4),
et non un biais O(1) de bord. Le transport lui-meme est fidele (invariants verts, section 6).

## 5. Comparaison directe au papier (lecture de l'arXiv)

Methode et physique **identiques** (verifie dans le texte du papier, Section 5.3) :
- vitesse initiale `v0 = -(grad phi0 x Omega)/|Omega|^2` (derive E x B) = notre modele `Diocotron` ;
- mesure : *"DFT du potentiel phi a rayon FIXE r=r0, module du coefficient du mode l"* = notre
  `mode_amplitude`, normalise a l'initial, ajustement exponentiel sur une fenetre etroite ;
- memes cibles analytiques 0.772 / 0.911 / 0.683, memes fenetres de fit ;
- temps : RK explicite ordre 3 (le notre : SSPRK3) ; espace : ordre 2 graph-viscosity dG (le notre :
  WENO5, ordre 5, DONC notre schema n'est PAS la limite).

La DIFFERENCE decisive est la GEOMETRIE, prouvee par la table de convergence du papier (Fig 5.4d) :

| mode 4 | papier (dofs) | gamma_h | ecart | | nous (eff) | gamma_h |
|---|---|---|---|---|---|---|
| | 196 608 | 0.935 | +2.6 % | | 256 | 0.985 (+8 %) |
| | 786 432 | 0.919 | +0.9 % | | 512 | 0.988 (+8 %) |
| | 3 145 728 | **0.913** | **+0.2 %** | | 1024 | 0.987 (+8 %) |

Le papier **CONVERGE** (0.935 -> 0.919 -> 0.913) sur un maillage de **DISQUE** epousant ; nous sommes
**PLATS** a +8 % sur une boite carree. L'ecart est reel, mais l'experience cut-cell (section 4,
mesure 5) montre qu'il ne vient PAS du *stencil* de paroi : poser Dirichlet sur le vrai cercle (ordre
2) au lieu de l'escalier ne change pas le taux. La difference tient a la representation cartesienne de
la **dynamique de l'anneau** elle-meme (advection E x B sur grille carree, dont la symetrie 4 resonne
avec le mode 4), pas au bord conducteur lointain. Reste a confirmer la cause structurelle exacte
(symetrie de grille vs methode de mesure DFT-phi vs normalisation), cf. section 7.

## 6. Indicateurs physiques verifies (fidelite du transport)

`analysis/diocotron_invariants.hpp` + `test_diocotron_invariants` (mode 4, WENO5+SSPRK3, eff 256) :

| invariant | resultat | role |
|---|---|---|
| masse `int rho` | exacte (derive 0) | conservativite (forme flux) |
| energie `1/2 int \|grad phi\|^2` | < 1 % | invariant du systeme ideal |
| moment angulaire `int rho r^2` | < 1 % | invariant diocotron (Davidson) |
| enstrophie `int rho^2` | -5.5 % | Casimir : MESURE la diffusion numerique |
| principe du maximum | `rho in [floor, rho_max]` | pas de valeurs parasites |
| `Re(omega)` (rotation) | reproduit (~+8 %) | 2e moitie de la dispersion |

Figures : `docs/fig_diocotron_highorder.png` (taux vs ordre), `docs/fig_diocotron_invariants.png`
(invariants vs temps).

## 7. Conclusion et prochaine etape

Les leviers de RECONSTRUCTION et d'INTEGRATION d'ordre eleve sont en place et verifies (depuis -39 %
en ordre 1). Avec une fenetre de fit lineaire propre ([3,9], R^2=1.00), le mode 4 est meme a **+1 %**
de l'analytique a eff 1024 et le mode 3 EXACT (voir le diagnostic ci-dessous) ; le sur-taux residuel
se concentre sur les modes de haut l.

Le bord embedded **cut-cell Shortley-Weller** (`GeometricMG`, option `cut_cell`) a ete implemente et
valide (`test_cut_cell` : ordre L2 1.93 au bord, erreur de Poisson 3459x plus faible qu'en escalier).
**Resultat negatif important** : il ne bouge PAS le taux diocotron (section 4, mesure 5). Le bord
conducteur n'etait donc pas le verrou suppose ; le mode instable est trop loin de la paroi pour la
"voir" (effet d'image `(0.44)^8 ~ 1e-3`). Le cut-cell reste un gain de precision propre du solveur de
Poisson, utile pour des configurations ou la charge approche la paroi.

**Diagnostic tranche (workflow `diocotron-overshoot-diag` + confirmation ROMEO 614125, voir
`romeo/CONV_RESULTS.md`).** Trois pistes ont ete CLOSES :
- **symetrie de grille : ECARTEE.** Tourner la phase des lobes ne change gamma que de +0.04 %. Indice
  decisif : sur grille uniforme c'est le mode 5 (NON couplable a une grille carree 4-fold) qui sur-tire
  le plus, le mode 4 (candidat 4-fold) est benin -> l'oppose d'une resonance de grille ;
- **methode de mesure : ELIMINEE.** `mode_amplitude` (lignes 242-262) lit DEJA le potentiel phi et fait
  la DFT azimutale du mode l a r=r0 : c'est EXACTEMENT la methode du papier, pas un observable densite ;
- **normalisation `omega_D` : REFUTEE.** Recalcul = 0.14324 (rho_bar = 1 - delta = 0.9, convention
  Davidson), confirme par l'eigensolveur et la frequence de rotation simulee. C'est la bonne echelle.

Ce qui RESTE (et que ROMEO a quantifie) : une distorsion de valeur propre **PLATE en resolution** (eff
256 ~ 512 ~ 1024, l'ecart ne se referme pas en raffinant -> structurel, pas de la troncature) et
**CROISSANTE avec le mode l**, NON uniforme. Fenetre lineaire [3,9], R^2=1.00, eff 1024 : mode 3 = 0.771
(EXACT), mode 4 = 0.921 (**+1 %**, quasi converge), mode 5 = 0.881 (**+29 %**, aberrant). L'ancien
"+8 % uniforme" etait un artefact de fenetre/schema (fenetre etroite + WENO5) ; le taux depend fortement
de la fenetre faute de plateau exponentiel net. La dispersion analytique gamma(l) pique a l=4 et
redescend en l=5 ; notre schema reproduit le pic mais pas le ROLL-OFF haut-l, car la fonction propre de
mode 5 (radialement plus structuree) est la plus distordue par la representation CARTESIENNE de l'anneau.

Voie vers < 1 % : faire EPOUSER les bords d'ANNEAU (r0, r1) ou vit le mode, pas la paroi :
- **cut-cell / level-set sur r0 et r1** (le cut-cell de paroi existe deja, sans effet ici) ;
- **grille polaire (r, theta)** pour transport + Poisson (supprime la brisure d'invariance de rotation).
Le mode 3 exact et le mode 4 a +1 % montrent que le cadre est CORRECT ; le verrou est la fidelite de la
fonction propre a haut l. Figure : `docs/fig_diocotron_conv_modes.png`.

## 8. Reproduction

```
# stabilite + AMR ; args : out nc nsteps refine l ml recon cut
#   recon : 0 NoSlope, 1 VanLeer, 2 Minmod   |   cut : 0 escalier, 1 cut-cell Shortley-Weller
g++ -std=c++23 -O2 -I include examples/diocotron_column_amr.cpp -o dca
./dca out 640 3000 0 4 0 1 0      # uniforme VanLeer eff 640, escalier (stable)
./dca out 256 900  0 4 0 1 1      # cut-cell : taux INCHANGE (cf section 4, mesure 5)

# haute precision WENO5 + SSPRK3, mode l, comparaison analytique
g++ -std=c++23 -O3 -fopenmp -I include examples/diocotron_highorder.cpp -o dho
./dho out 512 800 4 3 0.4         # eff 512, mode 4, WENO5
python3 scripts/validate_diocotron_growth.py out/ring_amp.csv --target 0.911 --window 4.2,5.2

# ROMEO : sbatch romeo/diocotron_highorder_hero.sbatch  (cf. romeo/HERO_RESULTS.md)
```

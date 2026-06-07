# Proprietes de conservation -- schema FV magnetise (chemin Schur cartesien)

Etat : juin 2026. Source primaire : `python/tests/test_schur_conservation.py` (PR #207,
branche `test/schur-conservation`). Les chiffres cites sont LES VALEURS MESUREES par ce
test ; les bornes assertees sont juste au-dessus du drift reel observe (pas des egalites
machine la ou la physique deplace legitimement la quantite).

Referentiel papier : Hoffart et al., arXiv:2510.11808 (Euler-Poisson magnetise isotherme).
Note methodologique en section 2.


---

## 1. Tableau des proprietes mesurees

| Propriete | Test (fonction) | Setup | Borne assertee | Drift mesure | Regime |
|-----------|----------------|-------|----------------|--------------|--------|
| **Masse -- domaine ferme** | `test_masse` | 64x64, periodique, anneau axisym, 40 pas | drift relatif < 1e-12 | ~1.9e-16 (precision machine) | Quasi-exact (FV conservatif + rho gelee par l'etage source) |
| **Masse -- Dirichlet** | (temoin dans `test_masse`, note du docstring) | Meme setup avec `bc=dirichlet`, Foextrap aux bords | non asserte | ~1e-2 | ARTEFACT CL (condition aux limites Dirichlet / Foextrap, pas du schema) |
| **Momentum -- profil axisymetrique** | `test_momentum` (2a) | 64x64, periodique ET Dirichlet, anneau centre, 20 pas | max|delta_mom|/m0 < 1e-12 | ~5e-18 (precision machine) | Quasi-exact (symetrie discrete : force nette = 0 par symetrie, telescopage FV) |
| **Momentum -- profil asymetrique (impulsion physique)** | `test_momentum` (2b) | 64x64, Dirichlet, bosse decentree, 20 pas | > 1e-6 (source active) et < 1e-1 (pas d'explosion) | ~1.9e-3 (impulsion physique de la force) | O(dt) -- PHYSIQUE, pas une erreur ; converge a T fixe (ratio ~1.04 sous dt->dt/2) |
| **Momentum -- convergence a T fixe** | `test_momentum` (2b, raffinement dt) | T=0.06 fixe, N1/N2 = dt/dt*2, Dirichlet | rapport impulse(dt)/impulse(dt/2) dans [0.5, 2] | ratio ~1.04 | Convergent O(dt) -- quantite physique continue en dt |
| **Energie E > 0** | `test_energie_positivite` (3a, 3d) | 48x48, Dirichlet, compressible (gamma=1.4), 30 pas | E_min > 0 ; croissance bornee 0 < delta_E/E0 < 0.5 | ~12% de croissance totale (travail PHYSIQUE du champ self-consistant) | Bilan sain -- croissance physique bornee, pas d'instabilite |
| **Energie sans source -- invariance machine** | `test_energie_positivite` (3c) | Meme setup sans etage Schur | rel delta_E < 1e-12 | < 1e-12 | Quasi-exact (FV pur a l'equilibre, E invariante) |
| **SchurEnergyKernel actif** | `test_energie_positivite` (3b) | Diff E_schur vs E_nosrc | max|delta_E| > 1e-6 | ~2e-1 | Temoin que le kernel est bien actif |
| **Positivite rho (minmod)** | `test_positivite_densite` (4) | 64x64, anneau raide (rho_fond=0.2, drho=2), recon_prim, 40 pas | rho_min > 0 | rho_min > 0 (rho_floor > 0 maintenu) | Quasi-exact sous reconstruction primitive + limiteur |
| **Positivite rho (vanleer)** | `test_positivite_densite` (4) | Meme, limiteur vanleer | rho_min > 0 | rho_min > 0 | Quasi-exact |
| **Positivite rho (weno5)** | `test_positivite_densite` (4) | Meme, limiteur weno5 | rho_min > 0 | rho_min > 0 | Quasi-exact |
| **Positivite pression p = cs2*rho** | `test_positivite_densite` (4) | Isotherme p = cs2*rho, memes 3 limiteurs | p_min > 0 | p_min > 0 | Derive de la positivite de rho |

Legende colonne "Regime" :
- **Quasi-exact** : conserve a la precision machine (erreur d'arrondi uniquement).
- **O(dt) / physique** : la quantite varie d'une quantite physique reelle, convergente quand dt->0 a T fixe.
- **Artefact CL** : la derive est due a la condition aux limites (Foextrap sur frontiere Dirichlet), pas au schema interne.


---

## 2. Note honnete : FV vs FE structure-preserving

### Ce que garantit adc (FV)

Le schema spatial d'adc est en **VOLUMES FINIS** (FV), pas en elements finis (FE) comme le
papier de reference Hoffart et al. (arXiv:2510.11808). Les consequences sont les suivantes :

**Conservation de la masse -- exacte par construction (FV).**
La discretisation FV des flux de continuite est telescopique : les flux sortant d'une cellule
entrent dans la cellule voisine, et la masse totale est invariante a la somme (precision machine)
dans un domaine ferme (periodique ou bords reflechissants). De plus, l'etage source condense
(`CondensedSchurSourceStepper`) gele rho pendant l'etape source et ne modifie que la vitesse.
Resultat mesure : drift relatif ~1.9e-16 en periodique (cf. #207).

**Conservation du momentum -- exacte SEULEMENT quand la force nette est nulle.**
Le transport FV du moment est egalement telescopique : en l'absence de force volumique, le
moment total serait conserve a la machine. Cependant, la force electrostatique/Lorentz
(-rho*grad(phi) + rho*v x Omega) constitue une source explicite de moment, appliquee par
l'etage Schur. Cette force est une quantite physique reelle :

- Sur un profil **axisymetrique** centre, la force nette s'integre a zero par symetrie, et le
  schema preserve cette symetrie : moment total conserve a la machine (~5e-18, cf. #207 (2a)).
- Sur un profil **asymetrique**, la force net est non nulle et le moment total varie de
  l'impulsion physique de cette force -- c'est de la physique, pas une erreur (mesure ~1.9e-3
  sur 20 pas, convergente a T fixe sous raffinement dt, cf. #207 (2b)).

**En FV, la conservation du momentum N'EST PAS exacte par construction** lorsque des forces
volumiques sont presentes. Elle l'est seulement quand le bilan de forces est discret-exact nul.

**Conservation de l'energie -- bilan sain, croissance physique.**
L'energie totale croit (~12% sur 30 pas) parce que le champ self-consistant fait un travail NET
sur le fluide initialement au repos -- ceci est physiquement attendu (instabilite diocotron =
transfert d'energie du champ vers le fluide). Le `SchurEnergyKernel` comptabilise l'increment
d'energie cinetique du travail electrostatique a chaque pas. En FV, le bilan d'energie n'est
pas ferme a la machine (la source fait un travail explicite) ; on borne la croissance et on
verifie l'absence d'explosion, pas une egalite.

### Ce que garantit le papier Hoffart (FE, structure-preserving au sens fort)

Le papier utilise des **elements finis structure-preserving** (complexe de de Rham discret,
espaces compatibles H(curl)/H(div)/L2). Dans ce cadre :

- La conservation du momentum decoule de la **forme faible discrete** : le bilan de force
  discret exact s'ecrit dans l'espace des formes differentielles discretes, garantissant la
  conservation integrale EXACTE du moment a la force volumique DISCRETE pres (sans erreur
  de telescopage ou de quadrature).
- L'energie peut etre conservee ou dissipee selon le schema temporel (miroir discret de
  l'identite continue).
- La positivite de rho peut etre garantie par construction via des limiteurs FE specifiques.

Ces proprietes sont **strictement plus fortes** que ce qu'offre FV : elles sont vraies
INDEPENDAMMENT de la symetrie de la condition initiale et sans requete de fermeture de
domaine particuliere.

### Validite de la comparaison adc vs papier Hoffart

La comparaison du **taux de croissance diocotron** entre adc et le papier est valide et
significative : le taux est une observable de la dynamique lineaire (pente de log|a_l(t)|),
qui depend de la dispersion de l'operateur d'advection-Poisson, pas de la structure-preservation
exacte du schema. La mesure sur adc reproduit l=3 a -0.38% du papier a n=512 (GH200, cf.
`docs/FULL_MODEL_VALIDATION_ROADMAP.md`).

En revanche, **preter a adc la structure-preservation au sens FE serait inexact** : adc conserve
la masse exactement (FV ferme), preserve le momentum quand le bilan est discret-exact nul (symetrie),
mais n'offre pas de conservation du momentum exacte par construction au sens de la forme faible
discrete. C'est une distinction honnete, documentee ici et dans le test #207.

**Question ouverte :** une preuve formelle de la conservation du momentum au niveau FE (bilan de
forme faible discret) est-elle requise pour le contexte de validation presente dans le rapport/papier ?
Voir `docs/FULL_MODEL_VALIDATION_ROADMAP.md` (PR3/PR4) pour la suite de la roadmap.


---

## 3. Pointeurs

- **Test source** : `python/tests/test_schur_conservation.py` (PR #207,
  branche `test/schur-conservation`). Valeurs mesurees citees en premiere source.
- **Roadmap globale** : `docs/FULL_MODEL_VALIDATION_ROADMAP.md` -- liste les PR de validation
  restantes (PR3 re-fit fenetres l=4/l=5, PR4 polaire, etc.) et les questions ouvertes.
- **Papier de reference** : Hoffart et al., arXiv:2510.11808.
- **Question ouverte** : preuve formelle FE requise ? (cf. roadmap PR2/PR3 et discussion
  section 2 ci-dessus).

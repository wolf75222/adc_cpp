# Roadmap : reproduction du modele COMPLET de Hoffart (Euler-Poisson magnetise)

Issu d'une investigation multi-agents (juin 2026, workflow scope-full-euler-poisson). Etat factuel,
verifie dans le code.

## Fait majeur : le modele COMPLET tourne DEJA (chemin cartesien)

`adc_cases/hoffart_euler_poisson_dsl/run.py` construit le systeme COMPLET du papier :
- modele 3 variables (rho, rho_u, rho_v), pression isotherme p = theta*rho (model.py:90-122),
- force de Lorentz m x Omega = (omega*my, -omega*mx),
- Gauss -alpha*rho (elliptic_rhs),
- resolu par `adc.Split(hyperbolic=Explicit(ssprk3), source=adc.CondensedSchur(theta=0.5, alpha))`,
  c'est-a-dire la pile Schur #118-128 (CondensedSchurSourceStepper, LorentzEliminator,
  TensorEllipticOperator/GeometricMG, BiCGStab) branchee par system_stepper.hpp:86-90.

Mesure : taux diocotron l=3 = -0.38% vs papier a n=512 (GH200). L'observable de taux est
`sample_circle(phi, ring_inner)` + FFT-theta (mode azimutal du POTENTIEL sur un cercle) : deja propre
en metrique polaire MEME sur grille cartesienne. La "diffusion du bord d'anneau" n'affecte que le rendu
de DENSITE brute (schlieren), PAS la mesure de taux.

## Decision de chemin : VOIE B (cartesien-fluide) -- recommandee

La Voie A (fluide POLAIRE + Schur polaire) reconstruirait a grand cout une capacite qui existe deja, et
elle est de niveau RECHERCHE :
- `dispatch_transport_polar` REJETTE le fluide (block_builder_polar.hpp:59-65) ;
- `PolarPoissonSolver` est un solveur DIRECT scalaire (FFT-theta + tridiag-r) structurellement
  INCOMPATIBLE avec l'operateur Schur A = I + c*rho*B^{-1} anisotrope croise (la FFT ne diagonalise plus
  des que les termes croises a_xy/a_yx existent ; risque de stagnation MG sur l'anisotropie 1/r^2) ;
- toute la pile Schur est cablee sur Geometry/dx/dy/GeometricMG cartesiens.

=> Voie A = amelioration ESTHETIQUE ulterieure (figure de densite 2D sans diffusion de bord) SI un
besoin visuel se confirme, PAS un chemin vers la fidelite au papier. A court terme : durcir/valider B.

## Le gap restant = VALIDATION + CONVERGENCE (pas de capacite manquante)

1. Taux l=4/l=5 non-monotones (l=4 -4.9->-8.4%, l=5 +11->+16% de n=384 a n=512) = artefact de FENETRE
   DE FIT (PAPER_FIT_WINDOWS statiques model.py:58, calibrees ~n=128, mordent dans la saturation des
   modes rapides a haute resolution). l=3 converge proprement (-0.38%).
2. Conservation discrete : briques structure-preserving manquantes (masse/momentum/energie/positivite).

## Roadmap

- [x] **PR1 conservation discrete (masse/momentum/energie/positivite)** -- FAIT #207. Tolerances MESUREES :
  masse conservee a la machine (1.9e-16, domaine ferme) ; symetrie momentum a la machine ; impulsion
  momentum = physique O(dt) convergente ; E>0, p>0 sous minmod/vanleer/weno5. Note honnete FV-vs-FE.
  Decouverte : sur Dirichlet la masse fuit ~1e-2 par Foextrap (artefact de CL, pas du schema).
- [ ] **PR2 doc CONSERVATION_SUMMARY** : tableau [propriete, test, assertion, borne] + note FV (momentum
  non exact par construction, contrairement au FE du papier). (petit)
- [ ] **PR3 re-fit fenetre precoce l=4/l=5** : detecter le debut de saturation (d2/dt2 log|a|), balayer
  des fenetres, choisir le regime plat. BLOQUEUR : seul le gamma final (sweep_results.csv) est sauve sur
  ROMEO, PAS l'amplitude(t) -> il faut MODIFIER sweep.py pour sauver amplitude(t) puis REJOUER n=384/512
  (heures GH200). (moyen, ROMEO)
- [ ] **PR4 table de validation haute resolution finale** : n=384 + n=512 O5 avec les fenetres re-fittees ;
  table taux l=3/4/5 vs papier + figure de convergence. (moyen, ROMEO ; depend de PR3 + decision fidelite)
- [ ] **PR5 (optionnel) splitting Strang ordre 2** : demi-pas source / pas transport / demi-pas source
  (actuellement Lie ordre 1). Opt-in, Lie par defaut bit-identique. (moyen ; depend decision)
- [ ] **PR6 (recherche, conditionnel) Voie A polaire-fluide** : SEULEMENT si une figure 2D de densite
  nette (type Fig 5.1) est jugee indispensable. Flux fluide polaire (courbure 1/r) + elliptique polaire
  ITERATIF pour le tenseur croise + stencils Schur polaires. 2-3 PR recherche. (recherche)

## DECISIONS PROPRIETAIRE (juin 2026)

- Q1 (figure 2D nette) : **OUI requise** => Voie A (fluide polaire) POURSUIVIE (plus optionnelle). Etape 1
  = #209 (transport fluide polaire, en CI). Etape 2 = Schur polaire (a faire).
- Q2 (cible fidelite) : **l=4/5 a +-2%** => re-fit fenetre precoce (ROMEO job 647356) + maj PAPER_FIT_WINDOWS
  + table de validation finale (rejouer si besoin).
- Q3 (structure-preservation FE formelle vs tests empiriques O(dt^2)) : OUVERTE.
- Q4 (Lie vs Strang) : OUVERTE (Strang = PR optionnelle PR5).

Etat des PR : PR1 conservation = #207 (merge). doc CONSERVATION_SUMMARY = #208 (CI). Voie A etape 1 = #209 (CI).

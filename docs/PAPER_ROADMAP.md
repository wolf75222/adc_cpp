# Feuille de route reproduction Hoffart (arXiv:2510.11808)

AUDIT (documentaire, aucune implementation). Recense ce qui MANQUE pour reproduire le
benchmark diocotron de Hoffart, Maier, Shadid, Tomas (arXiv:2510.11808, Section 5.3 : taux de
croissance de l'instabilite diocotron d'une colonne creuse, dans la limite de derive
`omega_d << omega_p << omega_c`), et classe chaque manque dans l'un de 4 paniers.

Sources lues pour cet audit :
- `docs/ALGORITHMS.md` (briques numeriques : sections 1-3 FV, 9-12 elliptique + cut-cell,
  13-16 AMR, 19 DSL JIT/AOT) ;
- `docs/ARCHITECTURE.md` (sections 2-4 couches, 7 elliptique, 8 AMR distribue, 10 frontiere
  lib/application) ;
- `docs/GPU_RUNTIME_PORT.md` (phases 8-11 : limite nvcc des lambdas etendues, strong-scaling
  AMR negatif) ;
- `docs/BIBLIOGRAPHY.md` section 3 (entree Hoffart) et `docs/archive/two_fluid_ap.md`
  (note de methode du schema AP deux-fluides) ;
- `todo.md` section 6 (M1/M2/M2b Hoffart) + sections 1-2 (aux/EPM) ;
- cas `adc_cases/diocotron/{run.py,README.md,band_instability.py}`,
  `adc_cases/diocotron_amr/run.py`, `adc_cases/two_fluid_ap/`, `adc_cases/cases_manifest.toml` ;
- bindings : `python/system.cpp`, `python/amr_system.cpp`, `python/bindings.cpp`,
  `python/adc/__init__.py`, `python/adc/dsl.py`, `include/adc/numerics/elliptic/geometric_mg.hpp`.

## Etat de reproduction (factuel)

Deux niveaux dans Hoffart (Section 5.3) :

1. **Cible analytique** (probleme aux valeurs propres radial de Petri/Davidson-Felice).
   REPRODUITE a 3 chiffres en numpy cote `adc_cases/diocotron/run.py` :
   `gamma_3 = 0.772`, `gamma_4 = 0.912`, `gamma_5 = 0.687` (cf. README du cas). Hors perimetre
   du coeur : c'est de la verification analytique pure numpy.
2. **Taux numerique mesure par `adc`** : pipeline complet (composition `ExB` + `BackgroundDensity`,
   Poisson de systeme a paroi conductrice circulaire `wall="circle"`, mesure du mode `l` de
   `phi` par FFT azimutale, ajustement `exp(gamma t)`). Tourne, capture l'instabilite (croissance
   exponentielle, classement des modes correct, `l=4` dominant), mais SOUS-ESTIME le taux :
   `l=3 -22 %`, `l=4 -27 %`, `l=5 -5 %` (n=192, Minmod ordre 2). `todo.md` section 6 : M1
   "limite par la diffusion numerique du bord d'anneau", M2/M2b "AMR sur le bord d'anneau triple
   le taux a base egale". Le balayage ordre x resolution etend desormais cet axe jusqu'a O5
   (WENO5-Z + SSPRK3) et jusqu'a n=512 ; voir la lecture par mode dans la section "verrou" ci-dessous.

A ce stade, l'ecart ressemble davantage a une limite numerique/structurelle du schema FV
cartesien qu'a un bug isole, mais le niveau exact du plancher reste a confirmer. Le candidat
identifie est le **bord d'anneau cartesien**.

## Le verrou structurel : bord d'anneau cartesien

La capacite cut-cell Shortley-Weller (`docs/ALGORITHMS.md` section 12) vit UNIQUEMENT dans
`include/adc/numerics/elliptic/geometric_mg.hpp` : elle place la paroi conductrice circulaire
Dirichlet a sa position REELLE pour le solveur de POISSON. Mais le transport hyperbolique
(`numerics/spatial_operator.hpp`, `numerics/numerical_flux.hpp`, `numerics/reconstruction.hpp`)
n'a AUCUNE notion de bord embedded : l'anneau de charge est advecte sur la grille cartesienne
pleine. Le predicat de paroi (`runtime/wall_predicate.hpp`, `python/system.cpp::wall_active`)
n'alimente que l'operateur elliptique, jamais le flux. Le gradient radial net de l'anneau est
donc diffuse par le schema FV cartesien, ce qui amortit le taux de croissance de facon
l-dependante (les modes a plus courte longueur d'onde, l=4, paient le plus). Monter en
resolution referme partiellement l'ecart mais ne change pas la nature du verrou.

MESURE (`diocotron/SWEEP_RESULTS.md` cote adc_cases). Le balayage ordre x resolution chiffre la
part diffusion vs structurel par mode. Il couvre maintenant l'axe haut ordre O5 = WENO5-Z + SSPRK3
(atteignable depuis Python depuis adc_cpp #88, cf. Panier 1) et la haute resolution n=384/512 (job
ROMEO x64cpu). L'axe ordre reel est donc `{O1 none, O2 minmod, O2 vanleer, O5 weno5}`. But de l'axe
O5 : eclairer la question laissee ouverte a O2 - le residu l-dependant est-il de la diffusion
(refermable par l'ordre) ou un plancher structurel du bord d'anneau cartesien ? Lecture par mode
(les %err detailles, les fenetres de fit et les reserves sont dans `SWEEP_RESULTS.md`, source de
verite) :

- **l = 3 (le signal le plus PROPRE, fenetre de fit homogene a tout n)** : l'`|%err|` se referme
  d'abord nettement avec l'ordre et la resolution, puis a O5 il APLATIT autour de -9 % a haute
  resolution (-10.3 % a n=256, -8.6 % a n=384, -8.8 % a n=512 : un cran plat dans le bruit de
  mesure). Ce n'est pas le comportement d'une diffusion qui s'epuise, c'est le candidat le plus
  CREDIBLE a un residu structurel.
- **l = 4 (le mode-cle)** : a basse resolution O5 tombe a ~ -4 % (n=128, n=256), ce que la lecture
  O2 prenait pour "diffusion presque epuisee" ; mais a n=384/512 il REMONTE vers ~ -9/-10 %. Reserve
  majeure : ces deux points haute resolution ont une fenetre de fit qui s'ouvre tot (t0 = 6.3 et
  5.4, comme le point n=192 deja ecarte), donc ils sous-lisent probablement la pente. On NE peut
  donc PAS conclure fortement sur l=4 : on peut seulement dire que le -4 % ne se reproduit a aucune
  des deux resolutions superieures.
- **l = 5** : deja proche de la cible des O2 a n=192 ; petit residu de signe variable (quelques %),
  ni l'ordre ni la haute resolution n'y font apparaitre de plancher.

CONCLUSION PRUDENTE (a confirmer, ne constitue PAS une preuve definitive). L'axe O5 + haute
resolution AFFAIBLIT l'hypothese "tout l'ecart etait de la diffusion d'ordre 2" : a l'ordre 5, sur
le mode le mieux mesure (l=3), le residu ne continue pas de se refermer mais semble plafonner. Les
DONNEES SUGGERENT un plancher residuel l-dependant de l'ordre de ~9-10 % a l'ordre 5 (contre ~12 %
vus a O2), probablement lie au bord d'anneau cartesien / paroi de transport, RESTE A CONFIRMER. Deux
limites empechent d'en faire un chiffre ferme : (1) le plateau l=3 ne tient pour l'instant que sur
un seul cran plat n=384 -> n=512 (un n=768/1024 ou deux horizons `t_end` excluraient une convergence
tres lente) ; (2) les points l=4 haute resolution sont biaises par leur fenetre de fit precoce
(diagnostic de fenetre robuste a prevoir avant de chiffrer un plancher l=4). Ce candidat structurel
reste l'argument quantitatif pour la PR-A "transport-wall", desormais avec une taille plausible
revisee a ~9-10 % a l'ordre 5.

## API publique recommandee (orientation)

Deux points d'entree utilisateur recommandes, tous deux sur des briques natives (chemin GPU/MPI) :

- **Composer des briques natives** : `adc.Model(state, transport, source, elliptic)` assemble un
  modele a partir de briques d'etat / transport / source / elliptique, consomme par
  `System.add_block(...)` (ou `AmrSystem`). C'est la voie des cas diocotron du sweep.
- **Ecrire un modele en formules** : `adc.dsl.Model(...)` decrit les equations symboliquement, puis
  `m.compile(...)` produit un `.so`. Pour la production, `backend="production"` est le defaut
  recommande (loader natif zero-copie -> `add_native_block`, chemin GPU/MPI).

Chemins AVANCES / LEGACY / TEST, PAS le chemin utilisateur principal :

- `backend="prototype"` (JIT, IModel, dispatch virtuel, Rusanov hote ordre 1) et
  `backend="aot"` (AOT host-marshale) : iteration / verification, pas la production.
- `add_dynamic_block` (prototype JIT) et `add_compiled_block` (AOT) : adders bas niveau correspondants.
- `adc.PythonFlux` : chemin numpy HOTE (HORS hot path GPU/MPI), pour TESTER un flux ecrit en
  formules, pas pour la production.

## Classification des manques (4 paniers)

### Panier 1 : deja possible avec l'API ACTUELLE (a lancer / regler)

Capacites cablees et exposees, suffisantes pour pousser plus loin sans nouveau code.

- **Montee en RESOLUTION et en ORDRE** : reglage pur (le cas diocotron tourne deja a n variable).
  C'est la voie M3 de `todo.md` section 6, et le balayage resolution x ordre est fait (cf.
  `SWEEP_RESULTS.md`). La montee en ORDRE WENO5-Z + SSPRK3 est desormais atteignable depuis Python
  (adc_cpp #88) : `adc.Spatial(limiter="weno5")` (raccourci `weno5=True`) selectionne la
  reconstruction WENO5-Z dans `make_block`, et `adc.Explicit(method="ssprk3")` (raccourci
  `ssprk3=True`) l'integrateur SSPRK3, par le chemin natif `add_block`. Le defaut reste inchange
  (Minmod / SSPRK2, bit-identique au pre-#88). WENO5 est desormais cable AUSSI sur les chemins `.so`
  AOT et production (`add_compiled_block` / `add_native_block`, adc_cpp #102 : grille `.so` a 3
  ghosts), et sur le chemin natif AMR (`AmrSystem` production, adc_cpp #105). Seul le chemin JIT
  prototype (`add_dynamic_block`) reste a 2 ghosts et rejette `"weno5"` (cf. Panier 2). Le balayage
  couvre donc `{O1, O2-minmod, O2-vanleer, O5 weno5}`, jusqu'a n=512.
- **Paroi conductrice circulaire sur Poisson** : `wall="circle"` + `wall_radius` est cable sur
  `System` (`python/bindings.cpp:97`) ET sur `AmrSystem` (`python/bindings.cpp:193`,
  `python/amr_system.cpp:78`). Le cut-cell elliptique est valide (MMS ordre 2, multi-box, MPI ;
  `docs/ALGORITHMS.md` section 12). Rien a ecrire pour la geometrie d'anneau de Petri.
- **AMR sur le bord d'anneau** : `adc.AmrSystem` + `set_refinement(threshold)` tourne et
  conserve la masse (cas `adc_cases/diocotron_amr/run.py`). M2/M2b de `todo.md` notent que
  l'AMR triple le taux a base egale. Pousser le raffinement / le nombre de niveaux est un
  reglage de config. `AmrSystem.potential()` (lecture de `phi` depuis Python pour la FFT
  azimutale) : binding SHIPPE (python/bindings.cpp:272, `#135`).
- **Diagnostic de taux** : la chaine mesure (FFT azimutale du mode `l` de `phi`, ajustement de
  la phase lineaire) est entierement en place cote `adc_cases`.

### Etat des chemins d'execution GPU / MPI (production)

Statut factuel des chemins production (briques natives, pas DSL), independant de la PR-A :

- **`System` production CPU** : valide (ctest serie ; pipeline diocotron tourne).
- **`AmrSystem` production CPU** : valide.
- **`System` GPU production np=1** : valide sur GH200 (adc_cpp #97). #97 corrige le segfault device
  des kernels elliptique/maillage (lambdas etendues premiere-instanciees depuis une TU externe ->
  foncteurs nommes, codegen device robuste sous nvcc) ; parite Cuda vs Serial `dmax_abs` ~ 1e-13
  sur `solve_fields`, `compute-sanitizer` propre.
- **`System::solve_fields` MPI CPU np=1/2/4** : valide (adc_cpp #99). #99 corrige le segfault hote
  du post-traitement par cellule (`fab(0)` sans garde `local_size()` sur les rangs sans box) ;
  resultat bit-invariant au nombre de rangs (`test_mpi_system_solve_fields_np{1,2,4}`, joue en CI MPI).
- **device-MPI production (GPU multi-rang)** : valide sur GH200 (adc_cpp #93). Le chemin production
  DSL (`add_compiled_model`) avec `geometric_mg` est valide device + MPI np=1/2/4 (harness dedie).
- **fft sous `System` en MPI np>1** : REFUSE proprement (adc_cpp #106 : garde-fou dur, plus de
  segfault). En MPI `System` repartit UNE box en round-robin, layout incompatible avec la FFT.
  `DistributedFFTSolver` (layout en bandes, `MPI_Alltoall`) existe et est teste a part, mais N'est
  PAS route dans `System` (layout bandes vs box unique) ; le periodique distribue passe par lui ou
  par `geometric_mg`.

Ces chemins ne sont PAS sur le chemin critique de la cible analytique ni du sweep diocotron (CPU),
mais ils conditionnent la montee en resolution multi-GPU evoquee au Panier 4.

### Panier 2 : facade DSL de production `m.compile(backend=...)`

Le DSL symbolique existe et compile (prototype JIT IModel, AOT, et loader natif de production) ;
la facade de production est consolidee. Reproduire Hoffart NE depend PAS du DSL (les briques
natives suffisent) ; ce panier n'est requis que si l'on veut piloter le modele magnetise complet
en formules depuis Python plutot qu'en composant des briques.

- **Facade `compile` consolidee** : `python/adc/dsl.py` expose `m.compile(backend=...)` ergonomique
  (auto-detection de l'include du coeur, cache `so_path` par cle d'ABI, adc_cpp #103) au-dessus de
  `compile_so` (prototype JIT), `compile_aot` (AOT) et `compile_native` (production, loader natif
  zero-copie -> `add_native_block`, target `"system"` ou `"amr_system"`). Le backend de production
  natif est SHIPPE (#85, #92) : `target="amr_system"` route vers `AmrSystem.add_native_block` (DSL
  Phase D). Les demonstrateurs DSL sont complets et tous `ci = true` en CI adc_cases :
  `diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`. Aucun cas diocotron du sweep ne
  passe par le DSL (les compositions vont par `models.diocotron(...)`, briques natives), mais le
  diocotron EST desormais ecrit en DSL dans `diocotron_dsl`.
- **WENO5 sur `.so`** : SHIPPE (#102). `add_compiled_block` (AOT) et `add_native_block`
  (production) allouent une grille `.so` a 3 ghosts et acceptent `"weno5"` ; le chemin natif AMR
  l'accepte aussi (#105, parite `add_native_block` == `add_compiled_model` == `add_block`, dmax=0).
  Seul le chemin JIT prototype (`add_dynamic_block`) reste a 2 ghosts et rejette `"weno5"`.
- **Pilotage device consolide** : la recette device-clean (lambda etendue -> foncteur nomme, codegen
  device robuste sous nvcc) couvre le transport (`block_builder.hpp`, adc_cpp #64), les kernels
  elliptique/maillage de `solve_fields` (#97, GPU `System` production np=1 valide GH200) ET le chemin
  production DSL device + MPI np=1/2/4 (#93, `geometric_mg` valide GH200). La validation GPU n'est
  donc plus en suspens cote production.

### Panier 3 : domaine-disque FV / capacite de paroi (vrai domaine circulaire, pas bord cartesien)

C'est le panier qui leve le VERROU structurel. Aucune de ces capacites n'existe aujourd'hui.

- **Bord embedded cote TRANSPORT** : porter la notion de cut-cell / paroi reflechissante du
  solveur elliptique (`geometric_mg.hpp`) vers l'operateur hyperbolique (`spatial_operator.hpp`)
  pour que l'anneau de charge ne soit plus advecte sur une grille cartesienne pleine. C'est le
  manque qui explique le sous-taux l-dependant. `docs/ARCHITECTURE.md` section 12 (comparaison
  AMReX) note d'ailleurs un Laplacien EB "en escalier" cote operateur, le cut-cell n'etant que
  pour le bord COURBE elliptique.
- **Maillage circulaire / coordonnees adaptees** : alternative au cartesien embedded, un domaine
  reellement disque (coordonnees polaires ou maillage cut-cell complet) ; non present (le coeur
  est cartesien adaptatif, `docs/ARCHITECTURE.md` section 1).

Tant que ce panier n'est pas traite, la reproduction QUANTITATIVE fine du taux numerique reste
bornee par la diffusion du bord cartesien (constat M1, `todo.md` section 6).

### Panier 4 : AMR multi-bloc avance ou EPM avance

Capacites partiellement presentes mais incompletes pour un usage Hoffart pousse.

- **AMR multi-bloc / multi-niveau a parite `System`** : `AmrSystem` reste MONO-bloc (pas
  multi-espece) ; IMEX source locale OK (Gap 2 #132, backward_euler_source /
  mf_apply_source_treatment) mais Schur global sur AMR et AMR multi-blocs restent a faire. Le
  multi-box natif n'est pas cable cote facade,
  et la facade Python AMR REJETTE HLLC/Roe et la reconstruction primitive (cf. `python/adc/__init__.py`,
  garde-fou `add_equation`) ; ce rejet est PUREMENT facade : le moteur C++ les supporte deja
  (l'API `add_block` accepte la recon primitive cote C++). WENO5 + Rusanov + conservatif EST cable
  sur le chemin natif AMR (#105). Un diocotron AMR a haute resolution + ordre eleve a parite
  `System` (recon primitive, Roe, multi-box) demande d'ouvrir ces options cote facade.
- **Strong-scaling AMR full-device** : le grossier reparti (`replicated_coarse=false`) est cable
  mais NEGATIF a l'echelle testee (`docs/GPU_RUNTIME_PORT.md` phase 11). Requis seulement pour de
  tres grandes resolutions multi-GPU, pas pour la cible Section 5.3.
- **EPM avance** : l'operateur elliptique etendu (eps(x), Helmholtz/ecrante, anisotrope) est fait
  et valide device (`docs/ALGORITHMS.md` section 11, `todo.md` section 2), mais le branchement
  `EllipticProblem` -> stencil par la fabrique additive reste DESCRIPTIF, et le decoupage Schur
  EPM est differe (`docs/ARCHITECTURE.md` section 7, `todo.md` section 7). Non requis pour le
  diocotron de derive (Poisson pur + paroi), pertinent si l'on vise le modele deux-fluides
  magnetise complet.
- **Modele magnetise complet** : le schema AP deux-fluides (`adc_cases/two_fluid_ap/`,
  `docs/archive/two_fluid_ap.md`) porte la rotation cyclotron exacte (section 6 de la note) mais
  PAS encore le couplage `E x B` + diamagnetique inhomogene au transport. C'est l'extension
  Hoffart au-dela de la limite de derive.

## Plan ordonne

### FAIT (a date)

- **WENO5-Z / SSPRK3 atteignables depuis Python** (adc_cpp #88) : `adc.Spatial(limiter="weno5")` +
  `adc.Explicit(method="ssprk3")` via le chemin natif `add_block`, defaut inchange.
- **Balayage ordre x resolution etendu a O5 et a n=384/512** (cf. `SWEEP_RESULTS.md`) : ordre
  `{O1, O2 minmod, O2 vanleer, O5 weno5}`, jusqu'a n=512 (haute resolution sur ROMEO x64cpu).
  Lecture : l=3 plafonne ~ -9 % a O5 haute resolution (candidat structurel le plus propre) ; l=4 ne
  reproduit pas son -4 % basse resolution mais ses points haute resolution sont biaises par leur
  fenetre de fit ; l=5 deja a la cible (cf. conclusion prudente section verrou).
- **GPU `System` production np=1** valide sur GH200 (adc_cpp #97).
- **`solve_fields` MPI CPU np=1/2/4** valide (adc_cpp #99).
- **device-MPI production** (`add_compiled_model` + `geometric_mg`) valide sur GH200 np=1/2/4
  (adc_cpp #93).
- **WENO5 sur les chemins `.so`** (AOT + production) valide (adc_cpp #102), et sur le chemin natif
  AMR (adc_cpp #105, parite `add_native_block` == `add_compiled_model` == `add_block`, dmax=0).
- **`AmrSystem` production natif** (`add_native_block`, `target="amr_system"`) shippe (adc_cpp #92,
  DSL Phase D).
- **fft sous `System` MPI np>1 refuse proprement** (adc_cpp #106 : garde-fou dur, plus de segfault).
- **`compile()` ergonomique + cache `so_path`** (adc_cpp #103).
- **Demonstrateurs DSL complets** (`diocotron_dsl`, `two_species_dsl`, `magnetic_isothermal_dsl`),
  tous `ci = true` en CI adc_cases.

### Prochain VERROU scientifique (le seul qui leve le sous-taux structurel)

- **Panier 3 - bord de transport / domaine-disque / bord embedded** : porter un bord embedded /
  paroi cote transport (ou un domaine reellement disque) pour que l'anneau de charge ne soit plus
  advecte sur une grille cartesienne pleine. C'est la seule voie qui adresse le candidat plancher
  structurel ~9-10 % mis en evidence par le sweep O5. Chantier le plus lourd et le plus payant pour
  le taux numerique. Le sweep n'en est PAS une preuve : il SUGGERE le candidat et reste a confirmer
  (n=768/1024 ou deux horizons `t_end` pour l=3 ; diagnostic de fenetre robuste pour l=4).
  La tentative "paroi-transport Phase 1" (adc_cpp #109) est EXPERIMENTALE et a ete FERMEE SANS
  merge : elle posait un bord de transport sur le CONDUCTEUR externe (mauvais bord), ce qui masque
  le vrai verrou. Le verrou scientifique reste le BORD D'ANNEAU (le gradient radial net de l'anneau
  de charge), pas la paroi conductrice. La PR-A "transport-wall" doit viser ce bord d'anneau.

### Prochains verrous d'infrastructure (peuvent atterrir en parallele)

- **`AmrSystem.potential()` cote Python** : binding SHIPPE (python/bindings.cpp:272, `#135`) ;
  `phi` lisible depuis l'AMR pour la FFT azimutale du diagnostic de taux.
- **fft distribuee routee dans `System`** : `DistributedFFTSolver` (layout en bandes) existe et
  est teste a part mais N'est PAS route dans `System` (layout bandes vs box unique) ; le cabler
  permettrait le periodique distribue MPI sans repli sur `geometric_mg`. Non requis pour la cible.
- **Parite facade AMR <-> `System`** : ouvrir cote facade Python le multi-box natif, la recon
  primitive et les flux HLLC/Roe (deja supportes par le moteur C++, rejet purement facade) pour
  pousser l'AMR a haute resolution + ordre eleve a parite `System`.
- **Panier 4 selon l'ambition** : strong-scaling AMR full-device (grossier reparti negatif a
  l'echelle testee), puis modele magnetise complet (`two_fluid_ap` couple au transport) si l'on
  sort de la limite de derive.

## Resume du verrou

Reproduire la CIBLE analytique de Hoffart est fait (numpy, 3 chiffres). Reproduire le taux
NUMERIQUE a parite demande de lever le bord d'anneau cartesien (panier 3) : aujourd'hui le
cut-cell ne sert que Poisson, le transport reste cartesien, d'ou un sous-taux l-dependant que la
resolution attenue sans supprimer. Le balayage etendu a O5 (WENO5-Z + SSPRK3, atteignable depuis
Python depuis adc_cpp #88) et a la haute resolution n=384/512 AFFAIBLIT l'hypothese "tout l'ecart
etait de la diffusion d'ordre 2" : sur le mode le mieux mesure (l=3), le residu O5 ne se referme
plus mais plafonne autour de -9 %. Les donnees SUGGERENT donc un plancher structurel candidat de
l'ordre de ~9-10 % a l'ordre 5, RESTE A CONFIRMER (un seul cran plat sur l=3 ; fenetre de fit
precoce biaisant l=4) - PAS une preuve definitive, mais l'argument quantitatif pour la PR-A
"transport-wall". Le pilotage WENO5-Z/SSPRK3 depuis Python n'est donc plus un verrou (fait, #88) ;
le verrou restant est bien le bord de transport (panier 3).

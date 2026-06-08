> **ARCHIVE -- document non-normatif.** Plan/spec interne conservee pour l'historique. La source de verite courante est le code + les docs canoniques (ARCHITECTURE.md, ALGORITHMS.md, BACKEND_COVERAGE.md). Ne pas s'y fier comme etat courant.

# Plan de nettoyage documentaire

PLAN UNIQUEMENT. Ce document ne modifie aucun fichier existant. Il recense les chevauchements
entre les docs actuels et propose un decoupage net des sources de verite, puis un plan de
raccourcissement du README.

---

## 1. Sources de verite cibles

Apres nettoyage, chaque document a un perimetre exclusif :

| Document | Source de verite exclusive |
|---|---|
| `README.md` | Vue d'ensemble (quoi / pourquoi / GIF), liens vers les autres docs, installation, CI status |
| `ARCHITECTURE.md` | Cinq couches, seams, concepts C++, frontiere lib/application, etat reel des composants |
| `ALGORITHMS.md` | Methodes numeriques generiques : formules, code, validation, tests nommement |
| `DSL_MODEL_DESIGN.md` (ou futur Sphinx) | API utilisateur Python : facade `dsl.Model`, `CompositeModel`, `add_equation`, statut par phase |
| `GPU_RUNTIME_PORT.md` | Journal de validation GH200 : phases, resultats chiffres, bugs corriges, caveats de perf |
| `PAPER_ROADMAP.md` | Science : cible Hoffart, balayage O5, verrou bord d'anneau, paniers 1-4 |
| `COUPLER_HIERARCHY.md` | Reference exhaustive des coupleurs de `include/adc/coupling/` |
| `SCHUR_CONDENSATION_DESIGN.md` | Spec de l'etage source Schur (design seulement, PR0-PR8) |
| `AMR_MULTIBLOCK_DESIGN.md` | Spec de l'AMR multi-blocs Phase 1 (design seulement, PR i-viii) |
| `BACKEND_COVERAGE.md` (a creer) | Matrice backend : quel chemin (add_block/aot/production/amr), quel GPU/MPI/AMR, quelle limite |
| `PERFORMANCE.md` | Profil ROMEO, banc bench_amr, chiffres mesures, leviers (OncePerStep, FFT vs MG) |
| `CHOICES.md` | Justification des decisions d'architecture (pourquoi, pas quoi) |
| `BIBLIOGRAPHY.md` | References bibliographiques uniquement |

---

## 2. Etat actuel des chevauchements, doc par doc

### 2.1 README.md

Le README fait 373 lignes et contient :

- (a) Vue d'ensemble + GIF (ligne 1-48) : GARDE ici.
- (b) Tableau "Ce que fournit le coeur" (lignes 56-82) : DUPLIQUE ARCHITECTURE.md section 6
  (carte des modules) et section 13 (arborescence detaillee). A DEPLACER vers ARCHITECTURE.md,
  remplace dans le README par un lien.
- (c) Section "DSL symbolique" avec le gros exemple Python (lignes 87-148) : contenu de
  documentation API (DSL_MODEL_DESIGN.md perimetre). Le README peut garder 5-6 lignes de
  presentation + un exemple minimal, le reste se deplace ou pointe vers DSL_MODEL_DESIGN.md.
- (d) Tableau "Quatre chemins de modele" (lignes 129-141) : DUPLIQUE la matrice de capacites de
  DSL_MODEL_DESIGN.md section 5. A deplacer vers DSL_MODEL_DESIGN.md ou BACKEND_COVERAGE.md,
  garder une reference dans le README.
- (e) Section "Limites actuelles" (lignes 152-173) : contenu honnete, mais DUPLIQUE
  DSL_MODEL_DESIGN.md section 0bis (GAP/SHIPPE) et GPU_RUNTIME_PORT.md (caveats). A deplacer,
  remplace par un lien vers les deux.
- (f) Section "Systemes multi-especes" (lignes 174-196) : DUPLIQUE partiellement ARCHITECTURE.md
  section 5 (couche temps/couplage, SystemCoupler). A raccourcir a 4-5 lignes + lien vers
  ARCHITECTURE.md.
- (g) Section "Backends" (lignes 198-213) : contenu ARCHITECTURE.md section 9. A reduire a
  2-3 lignes + lien.
- (h) Section "Utiliser le coeur" (lignes 215-228) : snippet CMake + contrat PhysicalModel. A
  garder en bref (c'est l'accroche), mais la description detaillee (tableau de modules) se
  trouve dans ARCHITECTURE.md.
- (i) Section "Module Python adc" (lignes 230-302) : tres longue. L'essentiel (exemple
  add_block, add_equation, AmrSystem) reste dans le README ; le detail de chaque adder et les
  chemins avances/legacy vont dans DSL_MODEL_DESIGN.md.
- (j) Section "Ecosysteme" (lignes 304-312) : GARDE ici (c'est de la navigation).
- (k) "Build et tests" (lignes 314-337) : GARDE ici (point d'entree obligatoire).
- (l) "Organisation du depot" (lignes 339-354) : GARDE, mais peut etre raccourcie a la liste
  de dossiers sans les descriptions (qui sont dans ARCHITECTURE.md section 13).
- (m) "Validation (coeur)" (lignes 356-373) : DUPLIQUE GPU_RUNTIME_PORT.md. A raccourcir a
  3-4 lignes + lien.

### 2.2 ARCHITECTURE.md

Contient deja les sections justes (couches 1-5, seams, elliptique, AMR distribue, backends, carte
des modules, arborescence detaillee). Chevauchements a corriger :

- Section 1 et section 6 reprennent le tableau de modules que le README presente aussi (points
  (b) et (i) ci-dessus). Apres raccourcissement du README, ces sections RESTENT dans ARCHITECTURE
  et deviennent la source de verite unique.
- Section 8 (AMR distribue) mentionne le regrid d'un niveau intermediaire comme cible ; cela
  CHEVAUCHE AMR_MULTIBLOCK_DESIGN.md section 5 (algorithme de regrid). AMR_MULTIBLOCK_DESIGN est
  la spec de conception ; ARCHITECTURE.md en donne l'etat courant. Garder les deux mais ajouter
  un lien croise.
- Section 11 (Validation) liste les tests ; DUPLIQUE partiellement GPU_RUNTIME_PORT.md (les
  validations device). A reduire : garder la liste des ctests coeur et les chiffres CI ; renvoyer
  vers GPU_RUNTIME_PORT.md pour les validations GH200.
- Section 13 (arborescence detaillee) est exhaustive et NE DUPLIQUE PAS vraiment le README
  (qui est moins detaille). OK en l'etat apres raccourcissement du README.

### 2.3 ALGORITHMS.md

Contient les methodes numeriques avec formules, code, validation. Chevauchements :

- Introduction (lignes 1-29) reprend le principe du coeur (equation + canal aux) deja dans
  ARCHITECTURE.md section 2. A raccourcir a 3-4 lignes avec lien vers ARCHITECTURE.
- Section 18 (composition runtime et systeme multi-especes) CHEVAUCHE DSL_MODEL_DESIGN.md
  section 0bis et ARCHITECTURE.md section 10 (frontiere lib/application). A deplacer les
  aspects "comment composer en Python" vers DSL_MODEL_DESIGN.md ; garder ici uniquement
  le point de vue algorithmique (CoupledSystem, SystemCoupler, secondmembre).
- Section 19 (DSL JIT/AOT) DUPLIQUE DSL_MODEL_DESIGN.md sections 0 et 7 (sequencement). Les
  aspectscodegen + test (test_dynamic_model, test_block_builder, test_compiled_model_parity)
  RESTENT dans ALGORITHMS ; les explications sur l'API Python et les backends vont vers
  DSL_MODEL_DESIGN.md.
- Section 20 (seam dispatch) : contenu ARCHITECTURE.md section 4. A reduire a un lien.

### 2.4 DSL_MODEL_DESIGN.md

Document de spec/API (API Python, statut par phase). Chevauchements :

- La section 0 (etat actuel) et 0bis (statut d'implementation) DUPLIQUENT le tableau "Quatre
  chemins" du README (point (d) ci-dessus). Apres suppression du tableau du README, cette
  section devient source unique.
- La matrice de capacites (section 5) DUPLIQUE le tableau "Quatre chemins" du README. Source
  unique : DSL_MODEL_DESIGN.md.
- Les references de fichiers (python/system.cpp lignes) sont rapidement perimeees. A marquer
  "indicatives" et a ne pas synchroniser a chaque PR.

### 2.5 GPU_RUNTIME_PORT.md

Journal de validation GH200 par phases. Chevauchements :

- L'en-tete (lignes 1-13) reprend l'etat global deja dans README section "Validation (coeur)".
  Apres raccourcissement du README, c'est GPU_RUNTIME_PORT.md qui reste source de verite.
- Les phases 1-11 sont des resultats uniques (chiffres, bugs, harness). PAS de doublon.
- La section "Validation device des features post-#48 (round 2)" et la section "Strategie
  suggeree" n'ont pas de doublon ailleurs.

### 2.6 PAPER_ROADMAP.md

Document science. Chevauchements :

- Reprend l'etat des chemins GPU/MPI en production (section "Etat des chemins d'execution") :
  DUPLIQUE GPU_RUNTIME_PORT.md (phases 7, 9, 10) et DSL_MODEL_DESIGN.md (section 0bis, SHIPPE).
  A raccourcir a 4-5 lignes + lien.
- La section "API publique recommandee" DUPLIQUE DSL_MODEL_DESIGN.md (section 0 + decisions
  tranchees). A remplacer par un lien.
- Les paniers 1-4 et le plan ordonne : contenu UNIQUE (science, verrou bord d'anneau). GARDE ici.

### 2.7 COUPLER_HIERARCHY.md

Reference exhaustive des coupleurs. Chevauchements :

- Section 1 (arbre des coupleurs) CHEVAUCHE ARCHITECTURE.md section 5 (couche temps et
  couplage). ARCHITECTURE.md en donne le principe, COUPLER_HIERARCHY.md les details. OK
  en l'etat : les deux peuvent coexister avec une reference croisee.
- Sections 2-11 (par coupleur) sont uniques (responsabilites, signatures, quand utiliser).
  Pas de doublon.

### 2.8 SCHUR_CONDENSATION_DESIGN.md

Spec de conception (design-only). Chevauchements :

- Introduction mentionne les sources lues (ARCHITECTURE, ALGORITHMS, PAPER_ROADMAP, etc.) :
  OK, c'est un ancrage intentionnel, pas un doublon.
- Section 9 (sequencement PR0-PR8) CHEVAUCHE le "Prochain verrou" de PAPER_ROADMAP.md, mais
  chaque document a son angle (SCHUR = design algorithmique ; PAPER_ROADMAP = angle science).
  Ajouter un lien croise suffit.

### 2.9 AMR_MULTIBLOCK_DESIGN.md

Spec de conception Phase 1. Chevauchements :

- Section 0 (note d'honnetete) et section 2.2 (roles exacts d'AmrSystemCoupler) reprennent
  ARCHITECTURE.md section 8 (AMR distribue) et COUPLER_HIERARCHY.md section 6
  (AmrSystemCoupler). C'est intentionnel (spec s'appuie sur le code existant). OK avec
  references croisees.
- Section 4 (decoupage en PR) est unique a ce document.

### 2.10 PERFORMANCE.md

Mesures de perf (methodologie CS:APP). Chevauchements :

- Chapeau reprend que les mesures viennent d'adc_cases : OK (honnetete, pas un doublon).
- Chiffres (OncePerStep, FFT vs MG, bench_amr) sont UNIQUES ici. Pas de doublon.
- Note sur le scaling GH200 negatif (phrase finale) CHEVAUCHE GPU_RUNTIME_PORT.md phase 11.
  A renvoyer vers GPU_RUNTIME_PORT.md pour les chiffres GPU.

### 2.11 CHOICES.md

Justifications des decisions (D-1 a D-7). PAS de doublon reel : ARCHITECTURE.md decrit l'etat,
CHOICES.md explique le pourquoi. Garder les deux.

### 2.12 BIBLIOGRAPHY.md

References uniquement. Pas de doublon (les citations apparaissent dans les autres docs comme
ancres, la definition complete est ici). OK en l'etat.

---

## 3. Matrice "ou vit chaque affirmation"

| Affirmation / contenu | Actuellement dans | Source de verite cible |
|---|---|---|
| Equation generique dU/dt + div F = S, D phi = f(U) | README, ARCHITECTURE, ALGORITHMS | ARCHITECTURE section 2 (principe) ; README 2-3 lignes |
| Tableau des modules du coeur (PhysicalModel, assemble_rhs, GeometricMG...) | README + ARCHITECTURE | ARCHITECTURE section 6 uniquement |
| Quatre chemins de modele (a)/(b)/(c)/(d) | README + DSL_MODEL_DESIGN | DSL_MODEL_DESIGN section 5 (matrice capacites) uniquement |
| Limites actuelles (AmrSystem mono-bloc, fft refusee MPI, AmrSystem.potential() EN COURS) | README + DSL_MODEL_DESIGN 0bis | DSL_MODEL_DESIGN 0bis uniquement |
| Exemple Python add_block multi-especes | README + DSL_MODEL_DESIGN | README (bref) + DSL_MODEL_DESIGN (detail) |
| Etat de validation GH200 (phases 1-11, chiffres) | README "Validation" + GPU_RUNTIME_PORT | GPU_RUNTIME_PORT uniquement |
| Backends CMake (ADC_USE_KOKKOS, ADC_USE_MPI...) | README + ARCHITECTURE | README (snippet CMake, 6 lignes) ; ARCHITECTURE section 9 (detail) |
| Seam for_each_cell + device_fence | README + ARCHITECTURE + ALGORITHMS | ARCHITECTURE section 4 + ALGORITHMS section 20 (1 lien) |
| Matrice de capacites backend (GPU/MPI/AMR par chemin) | DSL_MODEL_DESIGN section 5 | DSL_MODEL_DESIGN section 5 ET BACKEND_COVERAGE.md (a creer) |
| Etat des chemins GPU/MPI production | README + PAPER_ROADMAP + DSL_MODEL_DESIGN + GPU_RUNTIME_PORT | GPU_RUNTIME_PORT uniquement, avec liens dans les autres |
| API publique recommandee (add_block vs dsl.Model) | README + PAPER_ROADMAP + DSL_MODEL_DESIGN | DSL_MODEL_DESIGN "decisions tranchees" uniquement |
| Verrou bord d'anneau cartesien, sweep O5 | PAPER_ROADMAP | PAPER_ROADMAP uniquement |
| Formules SSPRK2/3, MUSCL, WENO5, multigrille | ALGORITHMS | ALGORITHMS uniquement |
| Justification pile from scratch vs pde_core_cpp | CHOICES | CHOICES uniquement |
| Perf : 86% elliptique, OncePerStep x2.6, FFT vs MG | PERFORMANCE | PERFORMANCE uniquement |

---

## 4. Plan de raccourcissement du README

### Principe

Le README cible est un PORTAIL de navigation, pas une documentation technique. Cible : 120-150
lignes (contre 373 actuelles), soit une reduction de 60-65%.

### Ce qui RESTE dans le README (vue d'ensemble + portail)

1. Header (1-23) : titre, badge CI, GIF, legende. GARDE.
2. Paragraph d'introduction (lignes 25-51) : quoi est adc_cpp, equation generique, canal aux,
   agnostique au modele. RACCOURCI a 10-12 lignes (garder le coeur, supprimer les repetitions
   du canal extensible).
3. Tableau "Ce que fournit le coeur" : REMPLACE par 4-5 lignes + lien vers ARCHITECTURE.md
   section 6.
4. Section DSL : GARDE 8-10 lignes (concept + exemple minimal de 6 lignes) + lien vers
   DSL_MODEL_DESIGN.md pour le detail et les chemins avances.
5. Section multi-especes : RACCOURCI a 5-6 lignes + lien vers ARCHITECTURE.md section 5.
6. Section Backends : RACCOURCI a CMake snippet (4 lignes) + lien vers ARCHITECTURE.md section 9.
7. "Utiliser le coeur" : snippet FetchContent (6 lignes). GARDE.
8. "Module Python adc" : RACCOURCI a l'exemple minimal add_block/set_poisson/step_cfl (12 lignes)
   + breve description de AmrSystem + lien vers DSL_MODEL_DESIGN.md et ARCHITECTURE.md.
9. "Ecosysteme" : GARDE (navigation, tableau 6 lignes).
10. "Build et tests" : GARDE (snippet + tableau options CMake, ~15 lignes).
11. "Organisation du depot" : RACCOURCI a la liste des dossiers (4-5 lignes), sans les
    descriptions (qui sont dans ARCHITECTURE.md section 13).
12. "Validation" : RACCOURCI a 4-5 lignes + lien vers GPU_RUNTIME_PORT.md.

### Ce qui QUITTE le README

- Tableau detaille de 20 lignes "Ce que fournit le coeur" -> ARCHITECTURE.md section 6.
- Tableau "Quatre chemins de modele" (10 lignes) -> DSL_MODEL_DESIGN.md section 5.
- Section "Limites actuelles" (22 lignes) -> DSL_MODEL_DESIGN.md section 0bis.
- Tableau multi-especes (12 lignes) -> ARCHITECTURE.md section 5 + lien.
- Sous-section detail des adders (add_dynamic_block, add_compiled_block, etc.) -> DSL_MODEL_DESIGN.
- Validation device detaillee (18 lignes) -> GPU_RUNTIME_PORT.md.

### Structure cible du README (ordre des sections, longueurs indicatives)

```
1. Header + GIF + legende         (15 lignes)
2. Presentation du coeur          (12 lignes, lien ARCHITECTURE)
3. Ce que fournit le coeur        (5 lignes + lien ARCHITECTURE section 6)
4. DSL symbolique (exemple bref)  (15 lignes, lien DSL_MODEL_DESIGN)
5. Systemes multi-especes         (6 lignes + lien ARCHITECTURE)
6. Backends CMake                 (6 lignes + lien ARCHITECTURE)
7. Utiliser le coeur (FetchContent) (8 lignes)
8. Module Python adc (exemple)    (15 lignes, lien DSL_MODEL_DESIGN)
9. Ecosysteme                     (10 lignes)
10. Build et tests                (15 lignes)
11. Organisation du depot         (8 lignes, lien ARCHITECTURE section 13)
12. Validation                    (5 lignes, lien GPU_RUNTIME_PORT)
---
Total : ~120 lignes
```

---

## 5. Document a creer : BACKEND_COVERAGE.md

Ce document n'existe pas encore. Il materialise la matrice de capacites de facon canonique,
aujourd'hui dispersee entre README (tableau "Quatre chemins") et DSL_MODEL_DESIGN.md
(section 5, matrice `_BACKEND_CAPS`).

Contenu propose :

```
| Chemin       | Adder Python         | CPU serie | MPI | AMR | GPU | zero-copie | Limites |
|---|---|---|---|---|---|---|---|
| (a) natif    | add_block            | oui       | oui | oui | oui | oui        | aucune  |
| (d) production | add_native_block   | oui       | oui | oui | oui | oui        | AmrSystem: mono-bloc, explicite |
| (c) aot      | add_compiled_block   | oui       | non | non | non | non        | mono-rang, SSPRK2 uniquement |
| (b) prototype | add_dynamic_block   | oui       | non | non | non | non        | Rusanov ordre 1, hote |
```

Chaque cellule ancre vers la source (DSL_MODEL_DESIGN, GPU_RUNTIME_PORT) pour le detail.

---

## 6. Ordre de priorite des interventions

Classement par valeur / effort :

1. (HAUTE VALEUR, FAIBLE EFFORT) Raccourcir README : supprimer les 4 sections identifiees
   ci-dessus, ajouter des liens. Objectif 120-150 lignes. Aucun contenu perdu.
2. (HAUTE VALEUR, FAIBLE EFFORT) Creer BACKEND_COVERAGE.md : materialize la matrice
   (contenu deja present dans DSL_MODEL_DESIGN section 5).
3. (VALEUR MOYENNE, EFFORT FAIBLE) Nettoyer les introductions ALGORITHMS.md et GPU_RUNTIME_PORT.md
   (supprimer les duplications d'entree avec ARCHITECTURE et README).
4. (VALEUR MOYENNE, EFFORT FAIBLE) Ajouter des liens croises entre SCHUR_CONDENSATION_DESIGN,
   AMR_MULTIBLOCK_DESIGN et PAPER_ROADMAP (sections qui se referent mutuellement).
5. (VALEUR FAIBLE, EFFORT ELEVE) Migration vers Sphinx/ReadTheDocs pour DSL_MODEL_DESIGN.md :
   utile seulement si le public utilisateur de l'API grossit (hors stage).

---

## 7. Regles de maintenance futures

Pour eviter que les doublons se reconstituent :

- Toute validation GH200 nouvelle -> GPU_RUNTIME_PORT.md uniquement. README pointe.
- Toute evolution d'API Python -> DSL_MODEL_DESIGN.md section 0bis (SHIPPE/GAP). README ne
  liste pas les limites.
- Toute methode numerique nouvelle -> ALGORITHMS.md. Pas de formule dans le README.
- Tout changement de schema de coupleur -> COUPLER_HIERARCHY.md. ARCHITECTURE.md decrit
  le principe, pas les signatures.
- Le tableau "Quatre chemins" (matrice capacites) vit dans BACKEND_COVERAGE.md. Les deux
  autres docs qui le referenciaient pointent vers lui.

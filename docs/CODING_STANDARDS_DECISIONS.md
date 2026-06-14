# Note de decisions : conventions de codage de `adc_cpp`

Date : 2026-06-12.
Base relue : `origin/master` / `ffb9022`.
Perimetre : conventions de codage C++ du coeur (`include/adc/**/*.hpp`, 110 fichiers, 25 089 lignes)
et de la couche de liaison Python (`python/{bindings,system,amr_system}.cpp`). Hors perimetre : le
code Python pur (`python/adc/*.py`) et `adc_cases`, qui ont leurs propres conventions.

Methode : relecture par sous-systeme (nommage, en-tetes, commentaires, gestion d'erreurs, mise en
forme, idiomes), confrontee a trois references publiques (Google C++ Style Guide, C++ Core Guidelines
ci-apres CG, LLVM Coding Standards). Pour chaque axe ou les trois guides divergent, ou ou la pratique
du depot diverge des trois, une regle unique est actee. Chaque constat non cosmetique est verifie sur
piece (`fichier:ligne` cites). Lecture seule, aucun fichier source modifie.

Docs lies : [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md) (audit de maintenabilite, 6 juin 2026),
[`QUALITY_TOOLING.md`](QUALITY_TOOLING.md) (analyse statique, milestone *Qualite de code & CI
durcie*, epic ADC-105). Le `CODEBASE_AUDIT.md` renvoie a
`CODE_DOCUMENTATION_CONVENTION.md`, **present dans l'arbre de travail mais non commite**
(donc hors suivi de version, a committer ; voir D9).

Principe directeur : la coherence avec l'existant prime. La base est saine ; on n'impose pas un
renommage massif. Pour chaque axe on retient la regle la plus proche de la pratique dominante, sauf
si cette pratique pose un probleme reel (securite, lisibilite, outillage). Les choix imposes par la
contrainte device (`nvcc`, code `__device__`, header-only Kokkos) sont signales : la latitude y est
nulle, pas seulement faible.

Statut : chaque decision est marquee **propose**. C'est une proposition soumise au mainteneur, pas
une regle deja appliquee.

## D1 Nommage

Divergence : c'est l'axe le plus eclate. Sur la casse des fonctions les trois guides sont
mutuellement exclusifs (Google `UpperCamelCase`, LLVM `camelBack`, CG `snake_case` NL.10) ; sur les
variables LLVM est seul a imposer `UpperCamelCase` ; sur les constantes Google impose le prefixe `k`,
LLVM `UpperCamelCase`, CG rien. Les macros sont le seul point de convergence (NL.9, `ALL_CAPS`).

Pratique du depot et regle proposee :

| Element | Pratique (chiffres, `fichier:ligne`) | Regle proposee | Guide proche |
|---|---|---|---|
| Types, classes, enums | `PascalCase` ; 219 `struct` / 41 `class` (`mesh/box2d.hpp:35`) | `PascalCase` | Google, LLVM |
| Alias-types | 71 `PascalCase` / 22 STL-like | `PascalCase`, sauf imitation STL (`value_type`) en `snake_case` | Google |
| Fonctions, methodes | `snake_case` (`physics/euler.hpp:49`, `:54`) | `snake_case` | CG (NL.10) |
| Variables locales | `snake_case` court, souvent `const` | `snake_case` | Google, CG |
| Membres non publics | suffixe `_` ; 506 occ. (`mesh/multifab.hpp:49-53`) | suffixe `_` obligatoire | Google |
| Membres publics de POD | sans suffixe (`Box2D::lo/hi`, `Euler::gamma` `physics/euler.hpp:40`) | sans suffixe | les trois |
| Constantes litterales | `kCamelCase` (`kTwoPi` `mesh/geometry.hpp:72`, `kMaxRuntimeParams` `runtime/runtime_params.hpp:34`) | `kCamelCase` | Google |
| Constantes-traits de concept | `snake_case` (`n_vars` `physics/euler.hpp:38`) | `snake_case`, nom impose par `requires` | contrainte |
| Macros | `ADC_` + `SCREAMING_SNAKE` ; `ADC_HD` 338x, `ADC_EXPORT` 24x (hors sa definition) | `ADC_` + `SCREAMING_SNAKE` | les trois (NL.9) |
| Fichiers | `.hpp` `snake_case` ; 110/110, zero `.h`/`.cc` | `.hpp` `snake_case` ; `.cpp` pour les TU | Google (adapte) |
| Namespaces | minuscules ; `adc`, `adc::detail` 45 occ. (`amr/cluster.hpp:49`) | `adc` public, `adc::detail` interne | les trois |
| Parametres template | `PascalCase` descriptif (`Model` 127x) ; lettre pour l'arithmetique (`M`, `N`) | idem | aucun |

Justification : la pratique est deja homogene sur tous ces axes et n'admet, pour chaque ligne, qu'une
seule des trois positions sans renommage massif. Trois points meritent une note. (1) La casse des
fonctions retient `snake_case` (CG NL.10) parce que basculer vers Google ou LLVM renommerait toute
l'API publique pour zero gain. (2) Les constantes ont deux registres assumes : litteral libre en
`kCamelCase`, nom impose par un concept en `snake_case` ; ce n'est pas un flottement mais une
distinction structurelle, a documenter pour ne pas passer pour une incoherence. (3) Le suffixe `_`
sur les membres non publics leve l'ambiguite nom-de-membre / nom-local dans les listes d'init.

Statut : propose.

## D2 Include guard

Divergence : Google et LLVM imposent `#ifndef ..._H_` et proscrivent `#pragma once` ; CG (SF.8)
demande un guard sans imposer la forme et tolere `#pragma once`.

Pratique du depot : `#pragma once` dans 110/110 en-tetes, zero `#ifndef` en garde d'inclusion (les
seuls `#ifndef` servent la compilation conditionnelle : `ADC_HAS_KOKKOS`, `ADC_HEADER_SIG`,
`NOMINMAX`/`WIN32_LEAN_AND_MEAN`). Unanime.

Decision proposee : `#pragma once` en premiere ligne de chaque en-tete.

Justification : en header-only profond, `#pragma once` supprime la maintenance des macros de garde et
les collisions de nom. La position Google/LLVM repond a des contraintes de portage que les
compilateurs cibles (gcc, clang, nvcc, MSVC) n'imposent plus. CG l'autorise.

Statut : propose.

## D3 Ordre des includes

Divergence : Google place le C++ std tot (avant les libs tierces) ; LLVM place le system header en
dernier ; ordres opposes. CG ne tranche pas.

Pratique du depot : bloc `<adc/...>` d'abord, ligne vide, puis bloc STL `<...>` ; chevrons partout ;
tri manuel (`SortIncludes: false`).

Decision proposee : deux groupes, `<adc/...>` puis STL, separes par une ligne vide ; chevrons pour
tout ; ordre maintenu a la main, `SortIncludes` reste `false`.

Justification : ni Google ni LLVM ne correspond a l'usage ; un tri automatique melangerait les deux
blocs et casserait les dependances d'ordre voulues. L'ordre maison (module propre d'abord) est stable
et lisible.

Statut : propose.

## D4 Exceptions et gestion d'erreurs

Divergence : Google et LLVM interdisent les exceptions (LLVM compile `-fno-exceptions`) ; CG les
recommande pour signaler l'echec (E.2, E.3).

Pratique du depot : exceptions host dominantes (`throw std::runtime_error` : 134 dans `include/adc`,
305 bindings Python inclus ; 3 `std::invalid_argument`, zero `std::logic_error`) ; message prefixe
par un contexte puis ` : `
(`runtime/native_loader.hpp:210`, `runtime/wall_predicate.hpp:33`). Zero `std::expected`,
`std::optional` marginal (3).

Decision proposee : exceptions autorisees et idiomatiques sur le chemin host (validation de config,
chargement de `.so`, erreurs d'API) ; `std::runtime_error` par defaut, `std::invalid_argument` pour
un argument invalide ; message toujours prefixe par un contexte puis ` : `. Le chemin device (code
sous `ADC_HD`) ne `throw` jamais.

Justification : la contrainte device impose deja le ban Google/LLVM sur la partie chaude, mais le host
n'a aucune raison de s'en priver et l'usage y est massif et homogene. On scinde selon le chemin
d'execution plutot que d'imposer un ban global qui ne reflete pas le code.

Statut : propose.

## D5 Asserts et contrats

Divergence : LLVM "assert liberally" plus `llvm_unreachable` ; Google `assert()` plus macros maison
`CHECK`/`DCHECK` ; CG `Expects()`/`Ensures()` (I.6, I.8) via GSL.

Pratique du depot : 45 `static_assert` (contraintes compile-time), 8 `assert(` bruts seulement,
aucune macro `ADC_ASSERT` (n'existe pas).

Decision proposee : preference forte pour `static_assert`. `assert` runtime sur le chemin host
uniquement, parcimonieux. Pas de GSL `Expects`/`Ensures`. Pas de macro `CHECK`/`ADC_ASSERT` tant
qu'un besoin recurrent n'est pas etabli.

Justification : `Expects`/`Ensures` ne sont pas appelables sur device et `assert` sur device est
couteux ou desactive ; le compile-time (concepts, `static_assert`) couvre deja l'essentiel des
contrats de cette base.

Statut : propose.

## D6 RTTI

Divergence : Google et LLVM interdisent RTTI (LLVM a `isa<>`/`cast<>`/`dyn_cast<>` maison) ; CG
(C.146) autorise `dynamic_cast` pour une navigation de hierarchie inevitable.

Pratique du depot : design statique par templates/concepts, invariant "device-clean" (pas de vtable
dans les kernels) ; aucun usage significatif de `dynamic_cast`/`typeid`.

Decision proposee : pas de RTTI sur le chemin numerique ; le polymorphisme passe par templates et
policies.

Justification : `typeid`/`dynamic_cast` ne sont pas utilisables sur device ; la contrainte tranche
pour Google/LLVM et l'architecture n'en a pas besoin.

Statut : propose.

## D7 `auto`

Divergence : CG (ES.11) permissif (eviter la repetition de noms de type) ; LLVM et Google restrictifs
(seulement si la lisibilite augmente).

Pratique du depot : 290 usages, dont 168 trailing-return `) -> ` et 10 structured bindings. Usage
pragmatique.

Decision proposee : `auto` quand il clarifie ou evite une repetition lourde (retours template,
iterateurs, structured bindings) ; type explicite quand il porte une information utile (unites,
semantique numerique). Trailing-return-type accepte comme idiome du depot.

Justification : la pratique se situe entre CG et Google, saine et lisible dans une base
template-lourde. On retient une regle pragmatique plutot qu'un ban formel intenable ici.

Statut : propose.

## D8 `struct` vs `class`

Divergence : convergence d'esprit. Google `struct` pour donnees passives ; CG (C.2, C.8) `class` si
invariant ou membre non public ; LLVM idem informellement.

Pratique du depot : 219 `struct` (POD, foncteurs, policies sans invariant) contre 41 `class` portant
les membres prives suffixes `_`.

Decision proposee : `struct` pour un agregat sans invariant (POD, foncteur device, policy) ; `class`
des qu'il existe un invariant a maintenir ou un membre non public, ce qui implique le suffixe `_`.

Justification : les trois guides convergent et la pratique suit deja cette ligne.

Statut : propose.

## D9 Documentation (Doxygen `///`)

Divergence : LLVM impose Doxygen `///` ; Google privilegie `//` sans systeme impose ; CG (NL.1-NL.4)
veut des commentaires d'intention sans systeme impose.

Pratique du depot : `///` pour l'API (5 057 lignes), `//` pour l'interne (5 079), `///<` trailing
pour les membres (312). `/// @file` + `/// @brief` sur 84/110 fichiers (76 %) ; `@param` 137x,
`@return` 17x (sous-employe). Blocs `/** */` isoles a 4 fichiers de `physics/`. Langue : francais sans
accents, quasi exclusif. Subsistent 26 fichiers sans en-tete Doxygen et un double en-tete redondant
(bloc prose `//` paraphrasant le `/// @brief`).

Decision proposee : Doxygen `///` (balises `@`) pour l'API, `//` pour l'interne, `///<` trailing pour
les membres ; `/// @file` + `/// @brief` sur chaque fichier ; proscrire `/** */` ; `@param` pour les
parametres et `@return` des qu'une fonction renvoie une valeur signifiante ; francais sans accents
(ASCII). Supprimer le double en-tete prose au fil des touches. **Committer
`CODE_DOCUMENTATION_CONVENTION.md`** (cible du lien depuis `CODEBASE_AUDIT.md`, presente dans l'arbre
de travail mais non commitee, donc hors suivi de version) pour reparer le lien, ou rediriger ce lien
vers la presente note.

Justification : le depot a deja choisi Doxygen `///` (position LLVM) et le francais ASCII ; il reste a
combler les trous (24 % des en-tetes, `@return`, ilot `/** */`) et a reparer les liens en commitant
le fichier de convention deja present mais non suivi. Pas de changement de systeme, seulement une
mise en conformite.

Statut : propose.

## D10 Format des TODO

Divergence : Google impose `// TODO: <contexte>` (historiquement `// TODO(user):`) ; LLVM et CG ne
prescrivent rien.

Pratique du depot : TODO presents mais sans format unique, souvent numerotes par chantier ("TODO 4"
a `numerics/spatial_operator.hpp:560`, "TODO 2.2").

Decision proposee : `// TODO(<contexte>) : <description>` ou `<contexte>` identifie le chantier ou
l'issue (ex. `// TODO(ADC-124) : ...`). Conserver les numeros de chantier existants comme contexte.

Justification : seul Google tranche ; un format unique rend les TODO greppables et tracables vers
Linear, pour un cout faible.

Statut : propose.

## D11 `[[nodiscard]]`

Divergence : aucun guide n'a de regle numerotee ; LLVM avait `LLVM_NODISCARD` ; CG l'encourage par
l'esprit pour les fonctions renvoyant un statut.

Pratique du depot : 0 `[[nodiscard]]`, volontaire (`-modernize-use-nodiscard` desactive dans
`.clang-tidy`).

Decision proposee : pas d'introduction systematique. Le reserver, au cas par cas et avec
justification, aux rares fonctions dont ignorer le retour est un bug certain (handle a liberer).

Justification : la gestion d'erreur passe par exception (D4), pas par code de retour ; l'interet de
`[[nodiscard]]` est marginal ici. Decision a rouvrir si des routines a code d'erreur apparaissent.

Statut : propose.

## D12 `explicit`

Divergence : Google impose `explicit` sur les constructeurs a un argument et les conversions ; CG
(C.46) "par defaut, declarer explicit les constructeurs mono-argument" ; LLVM silencieux.

Pratique du depot : 39 `explicit` sur des constructeurs mono-argument (`mesh/box_array.hpp:30`,
`runtime/system.hpp:66`).

Decision proposee : `explicit` obligatoire sur tout constructeur appelable avec un seul argument et
sur les operateurs de conversion, sauf intention explicite de conversion implicite (rare, a justifier).

Justification : convergence Google/CG, pratique deja installee, previent les conversions implicites
silencieuses. Cout nul.

Statut : propose.

## D13 Passage de parametres

Divergence : CG detaille (F.16 valeur si copie peu couteuse sinon `const&` ; F.17 in-out `&` ; F.18
`X&&` + `std::move` ; F.20 preferer le retour ; F.21 struct pour plusieurs sorties) ; Google "prefer
return values over output parameters", sorties historiquement par pointeur ; LLVM silencieux.

Pratique du depot : entrees `const&` ou par valeur pour les petits POD numeriques (`Real`, `State`) ;
retour de valeur prefere ; `std::move` dans les listes d'init.

Decision proposee : entree par valeur si la copie est peu couteuse (scalaires, petits POD passes aux
kernels), sinon `const&` ; in-out par reference non-const ; preferer le retour aux parametres de
sortie ; struct dediee pour plusieurs sorties liees ; `X&&` + `std::move` pour les ressources
transferees. Pas de parametre de sortie par pointeur (CG, pas le legacy Google).

Justification : aligne CG/Google sur l'usage dominant ; le passage par valeur des petits POD est en
outre requis par le chemin device.

Statut : propose.

## D14 Early exits et imbrication

Divergence : LLVM prescrit fortement les sorties anticipees (`return`/`continue`, pas de `else` apres
un `return`) ; Google et CG n'ont pas de regle dediee equivalente.

Pratique du depot : pas de regle ecrite ; style globalement plat.

Decision proposee : encourager les sorties anticipees pour reduire l'imbrication, sans en faire une
regle bloquante.

Justification : seul LLVM tranche ; la recommandation ameliore la lisibilite sans imposer de
reecriture. Guide, pas contrainte verifiee. Statut : propose.

## D15 Formatage automatique

Divergence : chaque guide a son `.clang-format` de reference ; le depot a deja le sien.

Pratique du depot : `.clang-format` present (base Google, C++20, `IndentWidth 2`, `ColumnLimit 100`,
`PointerAlignment`/`ReferenceAlignment Left`, `SortIncludes: false`, `ReflowComments: false`,
`FixNamespaceComments: true`), plus `.editorconfig` (LF, 2 espaces C++, 4 espaces Python) et
`.clang-tidy` informatif (`WarningsAsErrors` vide). `ReflowComments: false` laisse ~9 % de lignes
au-dela de 100 colonnes (longs commentaires Doxygen FR). Les jobs `format`/`tidy` de `quality.yml` ne
font que signaler.

Decision proposee : le `.clang-format` existant est la reference de mise en forme ; les decisions
D1-D14 le completent sur ce qu'il ne couvre pas. La bascule des jobs `format`/`tidy` de l'informatif
vers le bloquant est **deferree au milestone *Qualite de code & CI durcie* (epic ADC-105)**, qui
decidera quels checks deviennent un gate une fois la base assainie.

Justification : la mise en forme mecanique est deja outillee et stable ; rien a trancher ici, seulement
a renvoyer le durcissement au milestone qui le porte. Statut : propose.

## Suites

Une fois validees, ces decisions alimentent `CODE_DOCUMENTATION_CONVENTION.md` (deja present, a committer, D9) et le
durcissement progressif de `quality.yml` (D15, milestone ADC-105). Les incoherences relevees qui ne
demandent pas d'arbitrage de style (`CODE_DOCUMENTATION_CONVENTION.md` present mais non commite,
en-tetes Doxygen manquantes sur 26 fichiers, double
en-tete redondant, ilot `/** */` de `physics/`) sont des mises en conformite a la presente note, pas
des choix ouverts.

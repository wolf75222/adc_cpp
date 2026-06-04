# Conception de l'API modele DSL Python (`dsl.Model`)

SPEC documentaire. AUCUNE implementation : ce document decrit l'API cible et la
mappe ligne a ligne sur le code EXISTANT (sur `origin/master`, ref 950d5b2). Chaque
affirmation est ancree dans un fichier lu (cite `chemin:symbole`). L'implementation
est differee (d'autres agents editent les memes fichiers Python).

Sources lues pour cette conception :
- `python/adc/dsl.py` : `HyperbolicModel` (toutes les methodes), le codegen
  (`emit_cpp_brick`/`emit_cpp_source`/`emit_cpp_elliptic`/`emit_cpp_so_source`/
  `emit_cpp_aot_source`), la facade `compile(backend=)` + `_BACKENDS` + `adder_for`.
- `python/adc/__init__.py` : sucre runtime `Model`/`Spatial`/`Explicit`/`IMEX`/
  `Implicit`/`DivEpsGrad`/`System`/`AmrSystem`/`PythonFlux`.
- `python/system.cpp` + `python/bindings.cpp` : adders `add_block`,
  `add_dynamic_block`, `add_compiled_block`, et metadonnees (`read_block_meta`,
  `variable_names`/`variable_roles`/`block_gamma`), `set_poisson`, `step_cfl`/
  `step_adaptive`.
- `include/adc/core/variables.hpp` : `VariableRole`/`VariableSet`/`role_from_name`/
  `ADC_EXPORT_BLOCK_METADATA`/`ADC_EXPORT_BLOCK_GAMMA`.
- `include/adc/core/physical_model.hpp` : contrat `PhysicalModel`/
  `HyperbolicPhysicalModel`/`aux_comps`.
- `include/adc/runtime/compiled_block_abi.hpp` : ABI AOT `ADC_DEFINE_COMPILED_BLOCK`.
- `include/adc/runtime/dynamic_model.hpp` : `IModel`/`ModelAdapter` (JIT).
- `include/adc/runtime/dsl_block.hpp` : `add_compiled_model` (natif, template).
- `include/adc/runtime/amr_dsl_block.hpp` : `add_compiled_model` cote `AmrSystem`.
- `docs/PAPER_ROADMAP.md` (panier 2), `docs/ARCHITECTURE.md` (section runtime/DSL).

NOTE D'ENVIRONNEMENT. Un agent frere ajoute en parallele un vrai chemin natif
`production` (loader natif, adder `System`, garde-fou de cle ABI). Cette spec est
ecrite AUTOUR de cette direction (production = natif zero-copie `add_compiled_model`,
PAS le `add_compiled_block` hote a marshaling) sans presumer de ses noms de symboles
exacts : on decrit le CONTRAT, pas l'implementation du frere.


## Decisions tranchees (revue)

Points d'API fixes apres revue, pour eviter l'ambiguite a l'implementation :
1. `m.flux(x=, y=)` = DECLARATEUR symbolique ; l'evaluateur numpy est `m.eval_flux(...)` (noms distincts). (section 1)
2. `m.primitive_vars(rho=expr, ...)` accepte les KWARGS (ordre = layout `Prim`) ; forme positionnelle aussi. (section 1)
3. Le flux NUMERIQUE est `adc.FiniteVolume(limiter=, riemann=, variables=)` -- `riemann`, PAS `flux` (qui est le flux physique). (section 6)
4. `m.param(name, value)` retourne un objet `Param` NOMME (`name`/`value`/`kind`), pas un `Const` anonyme. (section 2)
5. `CompiledModel` porte `abi_key` + `model_hash` + flags de build, pas seulement `so_path`/backend/noms. (section 3)
6. Statut GPU natif CLARIFIE : le chemin device-clean a foncteurs nommes est VALIDE GH200 ; il manque seulement son binding Python. Production GPU n'est PAS casse. (section 5)
7. `m.compile(backend, target)` SANS `device=` : capacites GPU/MPI/AMR verifiees au branchement/execution, pas figees a la compilation. (sections 1, 5, 7)


## 0. Etat actuel (factuel, point de depart)

`HyperbolicModel` (`dsl.py:266`) est le SEUL objet modele. Il porte les formules
(arbre `Expr`), trois groupes de generateurs (`emit_cpp_brick`, `emit_cpp_source`,
`emit_cpp_elliptic`), trois enrobages (`emit_cpp_so_source` JIT, `emit_cpp_aot_source`
AOT, et la fabrique `_emit_bricks`/`_emit_metadata` partagee), et la facade
`compile(backend=)`. Trois backends sont declares (`dsl.py:792 _BACKENDS`) :
`prototype` -> (`jit`, `add_dynamic_block`), `aot` -> (`compile`, `add_compiled_block`),
`production` -> ALIAS de `aot` (`dsl.py:794`, meme couple `compile`/`add_compiled_block`).

Trois chemins d'execution existent cote C++ :
1. JIT : `.so` expose `adc_make_model`/`adc_model_nvars`/`adc_destroy_model`
   (`dsl.py:emit_cpp_so_source`), charge par `System::add_dynamic_block`
   (`system.cpp:706`) en `IModel<NV>` a dispatch VIRTUEL, residu HOTE Rusanov ordre 1
   (`system.cpp:host_residual`). Hors hot path GPU/MPI (`dynamic_model.hpp:18`).
2. AOT : `.so` expose l'ABI `ADC_DEFINE_COMPILED_BLOCK` (`compiled_block_abi.hpp:156`),
   charge par `System::add_compiled_block` (`system.cpp:738`). Tourne le chemin de
   production (`make_block`<Limiter,Flux>, SSPRK2/IMEX) MAIS sur une grille locale
   reconstruite dans le `.so` avec marshaling de tableaux plats (`copy_state`/
   `write_state`), donc PAS zero-copie, mono-rang (`compiled_block_abi.hpp:56
   DistributionMapping dm(ba.size(), 1)`, commentaire `:24-26` : "sans AMR/MPI").
3. NATIF : `add_compiled_model(System&, ...)` (`dsl_block.hpp:36`), template C++
   qui fabrique les fermetures sur le `grid_context()` REEL du `System` via
   `make_block` (parite bit-identique a `add_block`, validee CPU et GH200,
   `dsl_block.hpp:18-29`). Pendant AMR : `add_compiled_model(AmrSystem&, ...)`
   (`amr_dsl_block.hpp`). CE chemin n'a AUCUN binding Python : c'est un template,
   il exige `Model` connu a la compilation ; il n'est instancie que depuis une TU C++
   (p.ex. `python/tests/gpu/amrmpi_integrated.cpp:69`). C'est la cible du vrai backend
   `production`.

`dsl.Model` n'existe PAS aujourd'hui. Le nom `Model` (`__init__.py:115`) est une
fonction qui compose un `ModelSpec` de briques NATIVES (`adc.ExB`, `adc.CompressibleFlux`...),
chemin distinct du DSL symbolique. La spec ci-dessous reserve `dsl.Model` (dans le
module `adc.dsl`, pas `adc`) a la facade modele symbolique.


## 1. Facade stable `dsl.Model`

CIBLE : `dsl.Model` est la SURFACE stable ; `HyperbolicModel` reste le BACKEND
interne inchange. `dsl.Model` delegue chaque appel a une methode existante de
`HyperbolicModel` (composition, pas heritage : `dsl.Model` detient un
`HyperbolicModel` prive `_m`). Aucune numerique n'est touchee.

Construction : `m = dsl.Model("euler")` cree le `HyperbolicModel("euler")` interne.

Mapping methode cible -> backing `HyperbolicModel` :

| `dsl.Model` (cible) | backe par `HyperbolicModel` (`dsl.py`) | note |
|---|---|---|
| `m.conservative_vars(*names, roles=)` | `conservative_vars(*names, roles=)` `:291` | identique (transfere `roles=`) |
| `m.primitive_vars(rho=expr, u=expr, ...)` (kwargs) ou `(*vars, roles=)` | `set_primitive_state(*vars_or_names, roles=)` `:323` + `primitive(name, expr)` `:301` | STYLE CIBLE = KWARGS `name=expr` : chaque kwarg definit une primitive (`_m.primitive(name, expr)`) ET fixe le layout ORDONNE de `Prim` dans l'ordre des kwargs (Python 3.7+ : ordre d'insertion garanti). La forme positionnelle `(*vars, roles=)` reste acceptee. Les `roles=` (kwarg ou liste) restent supportes pour le mapping role->indice |
| `m.primitive(name, expr)` | `primitive(name, expr)` `:301` | definition d'une primitive par formule |
| `m.aux(name)` | `aux(name)` `:306` | champ auxiliaire (doit etre clef de `AUX_CANONICAL` `:35`) |
| `m.conservative_from(exprs)` | `set_conservative_from(exprs)` `:334` | inverse prim->cons (le DSL ne sait pas inverser) |
| `m.flux(x=, y=)` | `set_flux(x, y)` `:311` | DECLARATEUR symbolique du flux physique (decision tranchee, voir ci-dessous) |
| `m.eval_flux(U, aux, dir)` | `flux(U, aux, dir)` `:354` | EVALUATEUR numpy (debug / proto hote) ; nom DISTINCT du declarateur `m.flux` |
| `m.source(s)` | `set_source(s)` `:313` | optionnel |
| `m.eigenvalues(x=, y=)` | `set_eigenvalues(x, y)` `:312` | |
| `m.elliptic_rhs(e)` | `set_elliptic_rhs(e)` `:314` | optionnel (couplage Poisson) |
| `m.gamma(value)` ou `m.set_gamma(value)` | `set_gamma(gamma)` `:316` | EOS, porte par `ADC_EXPORT_BLOCK_GAMMA` |
| `m.param(name, value)` | (aucun backing) | GAP, voir section 2 |
| `m.check()` | `check()` `:382` | verifie dependances |
| `m.compile(backend, target)` | `compile(...)` `:798` (+ section 3) | renvoie un `CompiledModel`. PAS d'argument `device` : les capacites (GPU/MPI/AMR) sont verifiees AU BRANCHEMENT/EXECUTION (`add_equation`/`run`), pas figees comme un drapeau de compilation (eviterait une fausse garantie si le module n'est pas bati avec Kokkos/CUDA). Voir sections 5 et 7 |

COLLISION DE NOMS : TRANCHEE. Dans `HyperbolicModel`, `flux(U, aux, dir)` `:354` est
l'EVALUATEUR numpy (interprete CPU) et `set_flux` `:311` est le DECLARATEUR. Le plan
cible nomme le declarateur `m.flux`. DECISION : sur `dsl.Model`, `m.flux(x=, y=)` est le
DECLARATEUR symbolique (delegue a `set_flux`) ; l'evaluateur numpy est expose sous le
nom DISTINCT `m.eval_flux(U, aux, dir)` (delegue a `_m.flux`). La surface declarative
prime ; aucun nom ne porte les deux sens. (Pas d'alias `m.set_flux` sur la facade : un
seul nom par intention.)

Methodes cible SANS backing actuel (GAP LIST consolidee section 7) :
- `m.param(name, value)` : aucun mecanisme de parametre nomme dans `dsl.py` (voir 2).
- `m.compile(target=)` : `compile()` `:798` n'a PAS d'argument `target` ; il a `backend`,
  `name`, `cxx`, `std`, `require_metadata`. `target` (`"system"`/`"amr_system"`) est a
  ajouter (section 3). PAS de `device=` : decision (point 7) de ne PAS porter la capacite
  device comme un argument de `compile` -> les capacites sont verifiees au branchement
  (`add_equation`) et a l'execution (`run`), ou l'on sait si le module est bati Kokkos/CUDA
  et le contexte MPI/AMR reel (section 5). Un `device=True` a la compilation donnerait une
  fausse garantie (un `.so` peut compiler sans que le module hote soit device-capable).
- `m.compile()` ne renvoie PAS un objet `CompiledModel` aujourd'hui : il renvoie
  `so_path` (un `str`). Le `CompiledModel` (section 3) est nouveau.


## 2. `m.param(name, value)` : deux modes

### Mode (a) : constante figee a la compilation (FAISABLE aujourd'hui)

Le codegen INLINE deja toute constante. Un scalaire Python passe dans une formule est
promu en `Const(float(o))` (`dsl.py:_wrap :110`), et `Const.to_cpp()` renvoie
`repr(self.value)` (`dsl.py:117`) : la valeur est ecrite EN DUR dans le `.so`. Exemple
reel : `dsl_euler/run.py` ecrit `GAMMA = 1.4` puis `(GAMMA - 1.0) * (...)` ; le `1.4`
est inline. Donc `m.param("gamma", 1.4)` en mode constante n'a besoin d'AUCUN moteur
nouveau : c'est du sucre Python qui retourne un `Const` (ou un scalaire) reutilisable
dans plusieurs formules, recompile a chaque changement de valeur.

Forme proposee : `g = m.param("gamma", 1.4)` retourne un objet `Param` NOMME (pas un
`Const` anonyme), porteur de son IDENTITE : `name`, `value`, `kind` (`"const"` au depart,
`"runtime"` reserve, voir mode b). `Param` se comporte comme un `Expr` dans les formules
(il s'INLINE en `Const(value)` au codegen, donc zero-risque cote brique generee), mais
GARDE name/value/kind pour : l'introspection (`m.params`), les logs/diagnostics, la
reproductibilite (un run trace ses parametres), et la transition future vers les params
runtime (mode b) SANS changer la surface utilisateur. Changer la valeur exige
`m.compile(...)` a nouveau (tout est recompile aujourd'hui).

CAS PARTICULIER deja cable : `gamma` a un canal dedie hors formule via `set_gamma`
`:316` -> `ADC_EXPORT_BLOCK_GAMMA` (`variables.hpp:153`), lu par `read_block_meta`
(`system.cpp:179`) pour les couplages inter-especes. `m.param("gamma", ...)` devrait
donc, en plus de l'inliner dans les formules, appeler `set_gamma` pour que la
metadonnee ABI soit coherente (sinon le `System` retombe sur 1.4, `system.cpp:629`/`:844`).

### Mode (b) : parametre runtime (modifiable SANS recompiler) -> PHASE ULTERIEURE

INFAISABLE avec le codegen actuel sans changement de moteur. Justification ancree :
- Le codegen n'a AUCUN concept d'uniforme/membre. Les briques generees lisent
  uniquement les variables conservatives (`U[i]`, `cons_locals` `:468`), les
  primitives derivees (`prim_locals` `:471`), et les champs `Aux` (`a.<nom>`,
  `aux_locals` `:474`). Toute autre valeur est une `Const` inline.
- L'ABI AOT (`compiled_block_abi.hpp:156`) ne transporte aucun parametre : les
  signatures `adc_compiled_residual`/`advance`/`max_speed` (`compiled_block_abi.hpp:159-176`)
  prennent `U`, `aux`, geometrie, schema, pas de bloc de parametres. `Model model{}`
  est default-construit (`compiled_block_abi.hpp:103`,`:118`,`:130`,`:143`), sans etat.
- Cote JIT, `IModel`/`ModelAdapter` (`dynamic_model.hpp`) est aussi sans etat
  parametrique : `M model{}` default-construit (`dynamic_model.hpp:51`).

Deux voies possibles pour un VRAI parametre runtime, toutes deux PHASE 2 :
1. Passer les parametres par le canal `Aux` : declarer un champ aux constant par
   cellule et le peupler depuis Python (comme `set_magnetic_field` peuple `B_z`,
   `system.cpp:911`). Faisable SANS changer le codegen (le param devient un `m.aux`),
   mais limite aux 5 composantes canoniques (`AUX_CANONICAL` `:35`) et au cout d'un
   champ n*n pour un scalaire (abus du canal aux).
2. Etendre l'ABI : ajouter un parametre/bloc de doubles aux signatures
   `adc_compiled_*`, generer un `Model` construit avec ces parametres (membres), et
   propager via `set_density`-like. C'est un CHANGEMENT D'ABI (header
   `compiled_block_abi.hpp` + lecture `system.cpp:add_compiled_block`) ET de codegen
   (struct a membres + constructeur). A marquer explicitement PHASE 2.

VERDICT. `m.param(name, value)` est livrable en mode (a) (constante figee) immediatement
et sans risque. Le mode (b) (runtime, sans recompil) demande un changement de moteur
(ABI + codegen) et reste une phase ulterieure ; `m.param` doit donc soit ne supporter
que (a) au depart, soit accepter un `kind="const"|"runtime"` ou `kind="runtime"` LEVE
explicitement `NotImplementedError` (voir taxonomie section 4).


## 3. Objet Python `CompiledModel`

Aujourd'hui `compile()` `:798` renvoie un `str` (`so_path`) et expose separement
`adder_for(backend)` `:851` pour savoir quel adder `System` employer. La cible
remplace ce couple `(str, classmethod)` par un objet `CompiledModel` qui PORTE le
chemin et tout ce qu'il faut pour le brancher correctement.

### Champs

| champ | source (ancre) | role |
|---|---|---|
| `so_path` | retour de `compile_so`/`compile_aot` (`dsl.py:705`/`:743`) | chemin du `.so` |
| `backend` | `backend=` passe a `compile` | `prototype`/`aot`/`production` |
| `adder` | `_BACKENDS[backend][1]` (`dsl.py:792`) | nom de methode `System` a employer |
| `cons_names` | `HyperbolicModel.cons_names` | noms conservatifs (override `names=`) |
| `cons_roles` | `roles_for(cons_names, cons_roles)` (`dsl.py:77`) | roles physiques |
| `prim_names` | `HyperbolicModel.prim_state` | layout primitif |
| `n_vars` | `HyperbolicModel.n_vars` (`dsl.py:340`) | nb composantes |
| `gamma` | `HyperbolicModel.gamma` (`dsl.py:316`) | EOS (None = defaut 1.4) |
| `n_aux` | `aux_n_aux(aux_names)` (`dsl.py:39`) | largeur canal aux requise |
| `params` | dict `{name: Param}` (objets `Param` nommes, section 2) | introspection / reproductibilite |
| `caps` | derive du backend (section 5) | drapeaux CPU/MPI/AMR/GPU |
| `abi_key` | cle ABI baked (compilateur + std + signature d'en-tetes, cf. `abi_key.hpp` / `adc_cases/common/native.py`) | refuser un `.so` incompatible AU CHARGEMENT plutot qu'un UB silencieux |
| `model_hash` | hash stable du modele (formules + roles + n_aux + params) | identifier/reutiliser un `.so` deja compile ; tracer le run |
| `cxx`, `std`, `cxx_flags` | compilateur, standard, flags passes a la compilation | reproductibilite + diagnostic d'incompatibilite ABI |

`CompiledModel` est PRODUIT par `m.compile(...)` : la facade compile le `.so` (via
`compile_so`/`compile_aot` inchangees), puis empaquette le chemin avec les
metadonnees deja connues du `HyperbolicModel` (pas de relecture du `.so` : Python
detient deja noms/roles/gamma/n_aux). Les memes metadonnees sont DEJA emises dans le
`.so` par `_emit_metadata` (`dsl.py:675`) et relues cote C++ par `read_block_meta`
(`system.cpp:179`) ; `CompiledModel` les expose juste cote Python pour le dispatch et
les diagnostics, sans nouvelle source de verite.

Un `CompiledModel` n'est donc PAS un simple `str so_path` : il porte aussi la CLE ABI
et les flags de build (`abi_key`, `model_hash`, `cxx`/`std`/`cxx_flags`). C'est ce qui
permet de refuser au CHARGEMENT un `.so` compile avec un etat incompatible (compilateur,
standard, en-tetes divergents) au lieu d'un comportement indefini silencieux : le chemin
natif `production` compare deja cette cle cote C++ (`abi_key.hpp`, garde-fou du loader),
et `CompiledModel.abi_key` rend la verification + le diagnostic disponibles cote Python.

### Consommation : `System.add_equation(model=compiled, ...)`

CIBLE : `sim.add_equation(name, model=compiled, spatial=, time=, substeps=, names=)`
dispatche sur le bon adder selon `compiled.backend`/`compiled.adder` :

- `backend="prototype"` (`adder="add_dynamic_block"`) ->
  `self._s.add_dynamic_block(name, compiled.so_path, substeps, names, recon)`
  (`bindings.cpp:78`, `system.cpp:706`). NB : `add_dynamic_block` ne prend PAS
  `limiter`/`riemann`/`time` (chemin hote Rusanov ordre 1) ; il prend `recon`
  (`none`/`minmod`/`vanleer`) et `substeps`. La facade doit donc IGNORER (ou refuser,
  section 4) un `spatial.riemann != "rusanov"` pour ce backend (`riemann` = flux
  NUMERIQUE de `FiniteVolume`, cf. section 6 ; `flux` reste le flux PHYSIQUE du modele).
- `backend in {"aot","production"}` aujourd'hui (`adder="add_compiled_block"`) ->
  `self._s.add_compiled_block(name, compiled.so_path, spatial.limiter, spatial.riemann,
  spatial.variables, time.kind, substeps, names)` (`bindings.cpp:81`, `system.cpp:738` ;
  l'arg C++ du flux numerique s'appelle deja `riemann`, `variables` mappe `recon`).
- `backend="production"` CIBLE (apres le travail du frere) -> adder natif zero-copie
  (le pendant Python de `add_compiled_model`, `dsl_block.hpp:36`). Cet adder n'existe
  PAS encore cote binding (`add_compiled_model` est un template C++ non bindable tel
  quel : voir section 0). Le `CompiledModel.adder` doit alors pointer sur ce futur
  binding ; jusque-la `production` reste l'alias d'`aot` (`dsl.py:794`,`:810`).

`add_equation` est un NOM CIBLE. Aujourd'hui `System.add_block` (`__init__.py:337`)
prend un `ModelSpec` de briques natives, PAS un `.so`. `add_equation` est une nouvelle
methode de la facade Python `System` (`__init__.py:322`) qui aiguille selon le type
de `model` : un `ModelSpec` -> `add_block` ; un `CompiledModel` -> l'adder de bloc
compile/dynamique. Cela centralise le couplage backend<->adder que `dsl.py` documente
deja comme une regle de surete (`dsl.py:778-796` : "ne pas brancher un `.so` AOT sur
`add_dynamic_block`").


## 4. Taxonomie d'erreurs

Toutes les erreurs sont des `ValueError` (ou `NotImplementedError` pour le differe),
levees AU PLUS TOT (a la declaration ou a `compile`/`add_equation`, pas a l'execution),
message FACTUEL nommant la cause et l'action corrective. Modele des messages existants
de `dsl.py` (p.ex. `:296`, `:822`, `:845`).

| erreur | quand | forme du message (gabarit) | ancre du garde existant |
|---|---|---|---|
| role manquant requis | `compile(require_metadata=True)` sans role ni nom canonique | "le modele '<nom>' ne fournit pas roles physiques (...) ; le .so retomberait sur le fallback (roles 'custom')" | DEJA implemente `dsl.py:837-847` |
| gamma manquant requis | idem, `gamma is None` | "(...) ne fournit pas gamma (set_gamma(...)) (...)" | DEJA implemente `dsl.py:842` |
| param non defini | formule reference un `param` jamais pose | "param '<name>' reference mais non defini (m.param('<name>', valeur))" | etend `check()` `:382` (qui leve deja sur variable non declaree, `:397`) |
| param mode runtime | `m.param(name, value, kind="runtime")` | "param runtime non supporte (changement d'ABI/codegen requis, phase ulterieure) ; utiliser un param constant ou un champ aux" | NOUVEAU, `NotImplementedError` (section 2 mode b) |
| backend inconnu | `compile(backend=x)` hors `_BACKENDS` | "compile : backend <x> inconnu (attendus ['aot','production','prototype'])" | DEJA `dsl.py:821-823` |
| flux incompatible variables | `spatial.riemann in {hllc,roe}` sans primitive `p` declaree | "riemann 'hllc'/'roe' exige une pression : declarer m.primitive('p', ...)" | ANCRE physique : la brique n'emet `pressure`/`wave_speeds` que si `'p' in prim_defs` (`dsl.py:558`) ; `make_block` exige `m.pressure`/`m.wave_speeds` pour HLLC/Roe (cf. `amr_dsl_block.hpp:135`) |
| GPU/MPI/AMR incompatible backend | `add_equation` avec `device="gpu"`/MPI/AMR sur backend non capable | "backend '<b>' n'est pas device-clean / multi-rang : utiliser backend='production' (chemin natif)" | section 5 ; `compiled_block_abi.hpp:24-26` ("sans AMR/MPI"), `dynamic_model.hpp:18-21` (hors hot path GPU) |
| production demande mais seul aot dispo | `compile(backend="production")` tant que le natif n'est pas branche | soit un WARNING explicite ("production -> aot : chemin natif zero-copie non encore branche"), soit erreur si `device="gpu"` exige | `dsl.py:786-791` documente deja l'alias ; aujourd'hui SILENCIEUX (a rendre explicite) |
| require_metadata sur prototype | `compile(backend="prototype", require_metadata=True)` | "backend 'prototype' (...) incompatible avec require_metadata=True ; utiliser 'aot' ou 'production'" | DEJA `dsl.py:830-834` |
| names= mauvaise longueur | `add_equation(names=)` de longueur != n_vars | "names= a K noms mais le bloc '<n>' a NV variables" | DEJA cote C++ `system.cpp:618`/`:832` |

PRIORITE actuelle de resolution des noms/roles (a documenter pour l'utilisateur, NON a
changer) : `names=` explicite > metadonnees ABI du `.so` (`read_block_meta`) > fallback
`u0..` (`system.cpp:826-844` et `:614-628`). Les ROLES et le PRIMITIF ne viennent QUE
de l'ABI (l'API `names=` ne les fournit pas, `system.cpp:828`).


## 5. Matrice de capacites des backends

Etat HONNETE (aujourd'hui) vs cible. Lignes = backend `compile()`, colonnes = capacite.

| backend | CPU serie | MPI | AMR | GPU/Kokkos | zero-copie device | callbacks Python en hot path |
|---|---|---|---|---|---|---|
| `prototype` (JIT, `add_dynamic_block`) | oui (residu hote Rusanov o1) | non | non | non | non | non (C++ dispatch virtuel, sans GIL) |
| `aot` (`add_compiled_block`) | oui (production o2, HLLC/Roe) | non | non | non | non | non |
| `production` AUJOURD'HUI (= alias `aot`) | oui | non | non | non | non | non |
| `production` CIBLE (natif `add_compiled_model`) | oui | oui | oui | oui | oui | non |

Ancres :
- `prototype`/JIT : `dynamic_model.hpp:18-21` ("CHEMIN HOTE / PROTOTYPAGE uniquement ;
  les appels virtuels ne passent PAS dans un kernel GPU (...) ne pas utiliser dans la
  boucle chaude"). Residu hote ordre 1 Rusanov : `system.cpp:host_residual`, recon
  MUSCL hote 0/minmod/vanleer (`system.cpp:41`). Pas de flux HLLC/Roe (Rusanov fige).
- `aot` : tourne le chemin de production `make_block`<Limiter,Flux> / SSPRK2 / IMEX
  (`compiled_block_abi.hpp:21-26`), donc HLLC/Roe et ordre 2 dispos. MAIS sur grille
  LOCALE mono-rang reconstruite dans le `.so` (`compiled_block_abi.hpp:56`
  `DistributionMapping dm(ba.size(), 1)`) avec marshaling `copy_state`/`write_state`
  (`system.cpp:794-824`) : pas zero-copie, pas de halos MPI, pas d'AMR
  (`ARCHITECTURE.md:499` "sans AMR/MPI").
- `production` CIBLE : `add_compiled_model` (`dsl_block.hpp:18-29`) fabrique les
  fermetures sur le `grid_context()` REEL (`system.cpp:688`), residu via `make_block`
  avec `fill_boundary` (halos MPI) + `assemble_rhs` Kokkos sur les vrais `MultiFab`,
  SANS recopie ; parite bit-identique `add_block` validee CPU/Serial ET GH200
  (foncteurs nommes, `dsl_block.hpp:22-29`). AMR : `add_compiled_model(AmrSystem&)`
  (`amr_dsl_block.hpp`). STATUT GPU (a ne PAS lire comme "production GPU casse") :
  l'ANCIEN chemin a LAMBDAS ETENDUES `__host__ __device__` segfautait sur Cuda (limite
  nvcc) ; il a ete remplace par les FONCTEURS NOMMES device-clean de `block_builder.hpp`,
  et c'est CE chemin (le seul que `add_compiled_model` emprunte aujourd'hui) qui est
  VALIDE bit-identique sur GH200 (parite A==B `dres=0`, multi-box + MPI ; #64/#48,
  `dsl_block.hpp:22-29`). La SEULE limite restante est cote PYTHON : `add_compiled_model`
  est un template C++ sans binding, donc le backend `production` n'est pas encore
  re-route dessus depuis Python (c'est le travail du frere ; cf. section 0 et 7). Autrement
  dit : le chemin natif device-clean EXISTE et est valide en C++ ; il manque juste son
  exposition Python.
- Aucun backend n'execute de callback Python par cellule : meme `prototype` est un
  modele C++ (issu du codegen), pas un `adc.PythonFlux`. `adc.PythonFlux`
  (`__init__.py:420`) est un chemin numpy hote SEPARE, hors DSL compile.


## 6. Sucre runtime du plan -> existant

Mapping des objets/methodes runtime cibles du plan sur ce qui EXISTE
(`__init__.py`, `bindings.cpp`, `system.cpp`).

| cible (plan) | existant | statut |
|---|---|---|
| `adc.FiniteVolume(limiter=, riemann=, variables=)` | `adc.Spatial(limiter=, flux=, recon=)` (`__init__.py:271`) | RENOMMAGE + REMAP des args. DECISION (point 3) : le flux NUMERIQUE (Rusanov/HLL/HLLC/Roe) s'appelle `riemann=`, PAS `flux=`, pour ne pas collisionner avec le flux PHYSIQUE `m.flux` du modele. `limiter=` -> `Spatial.limiter` (none/minmod/vanleer, et a terme weno5), `riemann=` -> `Spatial.flux`, `variables=` (`"conservative"`/`"primitive"`) -> `Spatial.recon`. GAP de nommage, pas de moteur. |
| `adc.IMEX(substeps=)` | `adc.IMEX(substeps=)` (`__init__.py:304`) | EXISTE a l'identique. `Explicit(substeps=)` `:295` et `Implicit(dt_ratio=, substeps=)` `:313` (alias d'IMEX) aussi. |
| `adc.DivEpsGrad` | `adc.DivEpsGrad(epsilon=)` (`__init__.py:174`) | EXISTE. Operateur elliptique `div(eps grad)`. eps(x) variable via `set_epsilon_field` (`system.cpp:867`). |
| `adc.DirichletWall.circle(radius=)` | `set_poisson(wall="circle", wall_radius=)` (`bindings.cpp:95`, `system.cpp:854`) | GAP de sucre : pas d'objet `DirichletWall` ; aujourd'hui c'est un argument chaine de `set_poisson`/`add_elliptic_model` (`__init__.py:350`). `DirichletWall.circle(r)` serait un constructeur retournant `(wall="circle", wall_radius=r)`. |
| `sim.add_equation(model=, ...)` | `System.add_block(model=ModelSpec, ...)` (`__init__.py:337`) ; pour un `.so` : `add_dynamic_block`/`add_compiled_block` (`bindings.cpp:78`/`:81`) | GAP : `add_equation` n'existe pas. C'est le dispatcheur de la section 3 (aiguille `ModelSpec` vs `CompiledModel`). |
| `sim.run(...)` | `step(dt)` / `advance(dt, nsteps)` / `step_cfl(cfl)` / `step_adaptive(cfl)` (`bindings.cpp:132-135`, `system.cpp:1050-1097`) | GAP de sucre : pas de `run` ; aujourd'hui boucle Python explicite sur `step_cfl` (cf. `__init__.py:17` docstring, `dsl_euler/run.py`). `run(t_end=, cfl=)` serait une boucle `while time()<t_end: step_cfl(cfl)`. |
| `adc.System(n=, periodic=)` | EXISTE (`__init__.py:330`) | identique. |

Tous ces items sont du SUCRE Python pur-facade : aucun ne touche la numerique C++.
`add_equation` et `run` sont les seuls a porter une vraie logique (dispatch + boucle),
le reste est renommage/remap d'arguments.


## 7. Sequencement d'implementation

Regroupe par dependance. Ecrit pour paralleliser : chaque etape note son WRITE-SET de
fichiers (pour eviter les conflits avec les agents editant deja `python/**`).

### Phase A : facade pure-Python, AUCUN changement de moteur

Livrable des que les fichiers Python sont libres. N'edite que du Python ; ne touche ni
`include/**` ni `python/system.cpp`/`bindings.cpp`.

1. `dsl.Model` (section 1) : delegation vers `HyperbolicModel`. WRITE-SET :
   `python/adc/dsl.py` (ajout d'une classe ; ne pas modifier `HyperbolicModel`).
2. `m.param` mode (a) constante (section 2a) + cas `gamma` -> `set_gamma`. WRITE-SET :
   `python/adc/dsl.py`. S'appuie sur `_wrap`/`Const` inchanges.
3. `CompiledModel` (section 3) produit par `m.compile`, empaquetant `so_path` +
   metadonnees deja connues. WRITE-SET : `python/adc/dsl.py`.
4. Sucre runtime non-dispatch (section 6) : `adc.FiniteVolume` (remap de `Spatial`),
   `adc.DirichletWall.circle`, `sim.run`. WRITE-SET : `python/adc/__init__.py`.
5. `sim.add_equation` (section 3) : dispatch `ModelSpec` -> `add_block` ;
   `CompiledModel(prototype)` -> `add_dynamic_block` ; `CompiledModel(aot/production)`
   -> `add_compiled_block`. WRITE-SET : `python/adc/__init__.py` (methode de la facade
   `System`). N'exige AUCUN nouveau binding (les adders existent, `bindings.cpp:78`/`:81`).
6. Taxonomie d'erreurs (section 4) cote Python : etendre `check()` pour les params
   non poses, garde-fou flux/`p`, garde-fou backend/device. WRITE-SET : `python/adc/dsl.py`
   (+ messages dans `add_equation`, `__init__.py`).

Toute la phase A est testable contre les backends `prototype` et `aot` ACTUELS (CPU
serie), sans GPU/MPI.

### Phase B : depend du backend natif `production` (travail du frere)

A activer une fois le loader natif + son binding `System` disponibles (ne pas presumer
les noms). Ne touche pas la numerique ; c'est du cablage de dispatch.

7. `CompiledModel.adder` pour `production` pointe sur le NOUVEAU binding natif (au lieu
   d'`add_compiled_block`). `dsl.py:_BACKENDS["production"]` (`:794`) passe de
   `("compile","add_compiled_block")` a `("native","<binding natif>")` ; `compile`
   route alors sur le generateur natif (le frere ajoute un `emit_cpp_native_loader`
   equivalent ; la spec ne fixe pas le nom). WRITE-SET : `python/adc/dsl.py`,
   `python/adc/__init__.py` (branche `production` d'`add_equation`).
8. Rendre EXPLICITE l'ecart `production`->`aot` tant que (7) n'est pas en place :
   warning ou erreur si `device="gpu"`/MPI demande (section 4). WRITE-SET :
   `python/adc/dsl.py`.

### Phase C : MPI / GPU (validation, pas API)

9. Validation device/MPI du chemin `production` natif : parite A==B (`dres=0`) sur
   device, bit-identite multi-rang. Deja faite pour `add_compiled_model` C++
   (`dsl_block.hpp:22-29`, `PAPER_ROADMAP.md:91-92`) ; reste a valider de bout en bout
   via le binding Python de la phase B. AUCUN write-set Python nouveau (tests).

### Phase D : `AmrSystem` (phase 2)

10. `add_equation` cote `adc.AmrSystem` (`__init__.py:400`) router vers le pendant AMR
    natif `add_compiled_model(AmrSystem&)` (`amr_dsl_block.hpp`). LIMITE : `AmrSystem`
    n'est PAS a parite avec `System` (mono-bloc, explicite, sans recon primitive ni
    Roe ; `ARCHITECTURE.md:494`, `PAPER_ROADMAP.md:118-121`) : la facade doit refuser
    (section 4) un `spatial.riemann="roe"`/`variables="primitive"` sur AMR. WRITE-SET :
    `python/adc/__init__.py` (facade `AmrSystem`).

### Phase E : `m.param` runtime (phase 2, changement de moteur)

11. Mode (b) (section 2b) : ABI a parametres + codegen a membres, OU canal aux dedie.
    WRITE-SET LOURD : `include/adc/runtime/compiled_block_abi.hpp`,
    `python/system.cpp` (`add_compiled_block`), `python/adc/dsl.py` (codegen). Hors
    chemin critique de reproduction Hoffart (`PAPER_ROADMAP.md:147-150`, panier 2
    transverse, optionnel).

### Resume de dependance

- A (1-6) : independant, pur Python, livrable immediatement. Le gros de la valeur
  (`dsl.Model` stable, `CompiledModel`, `add_equation`, sucre runtime, erreurs).
- B (7-8), C (9) : gates sur le binding natif `production` du frere.
- D (10) : phase 2 AMR (et bornee par la non-parite `AmrSystem`).
- E (11) : phase 2, seul item exigeant un changement d'ABI/codegen (param runtime).

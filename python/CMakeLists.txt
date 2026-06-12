# Module Python pybind11 de la lib adc_cpp. L'extension compilee s'appelle `_adc`
# et est deposee dans un PAQUET `adc/` (avec __init__.py : le sucre lisible
# Spatial/Explicit/IMEX/System + adc.integrate). Avec PYTHONPATH = <build>/python,
# `import adc` charge le paquet, qui importe l'extension _adc.
#
#   cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc
#   PYTHONPATH=build-py/python python3 -c "import adc; print(adc.System)"

# LTO du module _adc. OFF par defaut : le LTO (re)optimise au LINK et rallongerait fortement le build
# -- le backend -O3 est deja le goulot (cf. BUILD_PROFILING.md) -- pour un gain runtime marginal sur ce
# code de dispatch type-erased (la frontiere de bloc passe par std::function, opaque au LTO). Il est par
# ailleurs numeriquement neutre (pas -ffast-math). OFF => NO_EXTRAS : pybind11 N'ESSAIE PAS le LTO, donc
# plus de test "Performing Test HAS_FLTO - Failed" parasite, et le build est rapide/previsible. ON =>
# THIN_LTO (clang) pour un binaire "release perf" PONCTUEL ; exige une toolchain dont le linker accepte
# -flto=thin (sur macOS, plutot la chaine clang conda complete que le melange AppleClang + linker conda).
option(ADC_PY_LTO "LTO (ThinLTO) sur le module _adc (OFF = build rapide, recommande)" OFF)
if(ADC_PY_LTO)
  pybind11_add_module(_adc THIN_LTO bindings.cpp system.cpp amr_system.cpp)
else()
  # NO_EXTRAS empeche pybind11 d'ACTIVER le LTO sur _adc. C'est NECESSAIRE, pas cosmetique : des que
  # le test HAS_FLTO passe (cas du pybind11 de l'env, p.ex. conda), pybind11_add_module activerait le
  # ThinLTO par defaut -> link/codegen au LINK -> build nettement plus long (le backend -O3 est deja le
  # goulot). On re-applique la visibilite cachee (seule "extra" qui compte ici) pour rester identique au
  # binaire produit jusqu'ici ; le chemin DSL "production" exporte quand meme ses symboles via ADC_EXPORT.
  pybind11_add_module(_adc NO_EXTRAS bindings.cpp system.cpp amr_system.cpp)
  set_target_properties(_adc PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()
target_link_libraries(_adc PRIVATE adc::adc adc_dev_options ${CMAKE_DL_LIBS})  # dl : add_dynamic_block (dlopen) ; adc_dev_options : warnings/sanitizers opt-in (vide par defaut, cf. cmake/AdcDevTooling.cmake)
target_compile_features(_adc PRIVATE cxx_std_${ADC_CXX_STD})

# RAM CI : sous Kokkos, cc1plus a -O3 sur les TU geantes du module (system.cpp / amr_system.cpp =
# facades completes, briques + AMR instanciees) pese plusieurs GB CHACUNE ; des que DEUX compilent
# en parallele (preset ci-kokkos-python, jobs: 2), le runner GitHub est tue par famine memoire
# ("runner received a shutdown signal", exit 143). Cas reel post vague ADC-63..71 (job "Kokkos
# (OpenMP)", step build du module) : bindings.cpp sort du ccache en ~20 s, puis system.cpp et
# amr_system.cpp partent ENSEMBLE (jobs: 2, miss ccache) et le runner meurt ~9 min plus tard --
# la veille, la meme paire concurrente passait encore (runs verts a 15-23 min) : la vague a fait
# deborder le pic cumule. On serialise donc les compiles DU MODULE par un pool Ninja taille 1
# (declare dans le CMakeLists racine, bloc Kokkos), pattern adc_heavy_tu de tests/CMakeLists.txt.
# -O3 est CONSERVE : c'est le module de production, seule la CONCURRENCE change (binaire
# identique). Scope ADC_HAS_KOKKOS : hors Kokkos les TU sont bien plus legeres, on garde le build
# local parallele. Sans effet hors generateur Ninja (JOB_POOL_COMPILE y est ignore).
if(ADC_HAS_KOKKOS)
  set_target_properties(_adc PROPERTIES JOB_POOL_COMPILE adc_heavy_module_tu)
endif()

# Norme C++ du LOADER baked dans le module : le DSL backend="production" doit compiler le modele
# natif avec EXACTEMENT cette norme, sinon __cplusplus diverge -> cle d'ABI differente -> rejet par
# add_native_block. ADC_CXX_STD vaut 20 sous Kokkos (CUDA 12.x sans -std=c++23) et 23 sinon (cf.
# CMakeLists.txt racine). On expose la valeur (entier 20/23) au binding, qui la publie en
# adc._adc.__cxx_std__ ; dsl.compile/compile_native en derivent le std natif au lieu de figer c++23.
target_compile_definitions(_adc PRIVATE ADC_CXX_STD=${ADC_CXX_STD})

# Version du projet (source unique : project(VERSION) racine), exposee en adc.__version__.
target_compile_definitions(_adc PRIVATE ADC_VERSION="${PROJECT_VERSION}")

# Chemin du COMPILATEUR du build, bake dans le module (expose en adc._adc.__cxx_compiler__). Le DSL
# backend="production" compile a l'execution un loader .so dont la cle d'ABI encode __VERSION__ du
# compilateur : le SEUL compilateur garanti compatible est CELUI-CI. Sans ce baking, dsl.py retombait
# sur `which c++` du PATH -- dans un env conda active, c'est souvent un autre compilateur (vieux gcc/
# clang) qui rejette -std=c++23 (bug reel : "error: invalid value 'c++23'") ou, pire, compile avec un
# __VERSION__ different -> rejet ABI. dsl.py prefere ce chemin s'il existe encore sur la machine.
target_compile_definitions(_adc PRIVATE ADC_CXX_COMPILER="${CMAKE_CXX_COMPILER}")

# Signature de l'arbre d'en-tetes du coeur, BAKED dans la cle d'ABI du module (adc/runtime/abi_key.hpp
# via -DADC_HEADER_SIG). Le backend DSL "production" (dsl.compile_native) recalcule la MEME signature
# (adc.dsl.adc_header_signature) et la bake dans le loader .so : les cles concordent quand les en-tetes
# concordent, divergent sinon -> add_native_block leve une erreur explicite (pas d'UB silencieux).
# CALCUL (miroir bit-pour-bit de adc_header_signature) : sha256 du concatene TRIE, pour chaque .hpp/.h
# sous include/, de "<chemin relatif>\n<sha256 du contenu>\n".
# CONFIGURE_DEPENDS : CMake re-genere (donc recalcule la signature) si un en-tete est AJOUTE/SUPPRIME.
# Une EDITION de contenu d'en-tete n'invalide pas le glob ; un simple `cmake --build` peut alors garder
# une signature perimee -> compile_native rejetterait les loaders (erreur EXPLICITE, jamais d'UB). Le
# flux standard `cmake -S . -B build-py ... && cmake --build` reconfigure et recalcule : c'est le mode
# nominal. Apres une edition d'en-tete sans reconfigure, relancer la configuration (re-`cmake -S`).
set(_adc_include "${CMAKE_CURRENT_SOURCE_DIR}/../include")
file(GLOB_RECURSE _adc_headers CONFIGURE_DEPENDS RELATIVE "${_adc_include}"
     "${_adc_include}/*.hpp" "${_adc_include}/*.h")
list(SORT _adc_headers)
set(_adc_hdr_blob "")
foreach(_h ${_adc_headers})
  file(SHA256 "${_adc_include}/${_h}" _h_digest)
  string(APPEND _adc_hdr_blob "${_h}\n${_h_digest}\n")
endforeach()
string(SHA256 _adc_header_sig "${_adc_hdr_blob}")
target_compile_definitions(_adc PRIVATE ADC_HEADER_SIG="${_adc_header_sig}")
message(STATUS "adc Python module: signature en-tetes ADC_HEADER_SIG=${_adc_header_sig}")

# L'extension va dans le paquet adc/ ; on y copie aussi les sources Python du paquet.
set(ADC_PY_PKG ${CMAKE_CURRENT_BINARY_DIR}/adc)
set_target_properties(_adc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ADC_PY_PKG})

# Build PIP (scikit-build-core definit SKBUILD) : la wheel embarque le paquet pur python/adc/
# (wheel.packages du pyproject) ; ICI on n'installe QUE l'extension compilee a cote, dans
# site-packages/adc/. Resultat : `pip install .` -> `import adc` SANS PYTHONPATH (pattern
# pyAMReX/WarpX). Le flux dev presets+PYTHONPATH reste inchange (SKBUILD absent).
if(SKBUILD)
  install(TARGETS _adc LIBRARY DESTINATION adc)
endif()
foreach(pyf adc/__init__.py adc/integrate.py adc/dsl.py adc/moments.py)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${pyf} ${CMAKE_CURRENT_BINARY_DIR}/${pyf} COPYONLY)
endforeach()

# Test des bindings : importe le paquet et verifie les invariants (composition, masse,
# Poisson, integrateur temporel Python). Enregistre des que le module est construit (pas
# garde par ADC_BUILD_TESTS), pour que `ctest --test-dir build-py` le voie meme avec
# ADC_BUILD_TESTS=OFF. Lance avec le MEME interpreteur que celui ayant compile _adc.
# Le test vit dans python/tests/ (pas python/) : sinon sys.path[0] (dossier du script)
# contiendrait le paquet source adc/ et masquerait le paquet construit.
#
# Environnement commun a TOUS les tests python ci-dessous. Outre PYTHONPATH (le paquet adc
# construit), on transmet ADC_KOKKOS_ROOT aux tests qui compilent un .so DSL a la volee
# (AOT / "production") : depuis le passage Kokkos-only (#263), compile_aot/compile_native incluent
# les en-tetes adc qui EXIGENT Kokkos et LEVENT une RuntimeError si aucune racine Kokkos n'est
# visible (ADC_KOKKOS_ROOT / Kokkos_ROOT / KOKKOS_ROOT ; cf. adc.dsl._native_kokkos_root). Sans
# elle, un `ctest` nu echoue sur test_dsl_{aot,jit}_*, test_dsl_production*, test_dsl_hybrid*,
# test_dsl_coupled_source* -- ce que ci.yml masquait en exportant la variable a la main. On la derive
# de l'install Kokkos resolu par find_package (cible Kokkos::kokkos) pour que `ctest` soit AUTONOME.
# Reste vide pour un Kokkos FetchContent (pas d'install <root>/include) : on retombe alors sur
# PYTHONPATH seul et l'env du dev/CI tranche, comportement historique inchange.
set(_adc_py_test_env "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
if(ADC_HAS_KOKKOS AND TARGET Kokkos::kokkos)
  get_target_property(_adc_kokkos_incs Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  foreach(_inc IN LISTS _adc_kokkos_incs)
    if(EXISTS "${_inc}/Kokkos_Core.hpp")
      get_filename_component(_adc_kokkos_root "${_inc}/.." ABSOLUTE)
      list(APPEND _adc_py_test_env "ADC_KOKKOS_ROOT=${_adc_kokkos_root}")
      break()
    endif()
  endforeach()
endif()

add_test(NAME test_bindings
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py)
set_tests_properties(test_bindings PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Methode temporelle explicite "euler" (ForwardEuler, ADC-174) : identite de Shu-Osher
# bit-exacte (ssprk2 == 0.5 U0 + 0.5 euler(euler(.))), no-default-change, rejet AMR.
# Modele natif pur transport : aucun compilateur requis.
add_test(NAME test_time_euler
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_time_euler.py)
set_tests_properties(test_time_euler PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Diagnostic dt_hotspot (ADC-182) : cellule dominante de la CFL (cellule chaude plantee,
# coherence w == cfl*h/dt, no-default-change bit-identique, determinisme, garde). Natif.
add_test(NAME test_dt_hotspot
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dt_hotspot.py)
set_tests_properties(test_dt_hotspot PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Poisson FFT a symbole spectral continu (ADC-175) : oracle sinusoide discriminant
# (spectral ~1e-12, fft discret / MG = fenetre O(h^2) -> no-default-change mesure), rejets.
# Modele natif : aucun compilateur requis.
add_test(NAME test_poisson_fft_spectral
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_poisson_fft_spectral.py)
set_tests_properties(test_poisson_fft_spectral PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Test du mini-DSL symbolique (adc.dsl) : pur Python (interprete CPU), meme PYTHONPATH.
add_test(NAME test_dsl
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl.py)
set_tests_properties(test_dsl PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Generateur generique de modeles de moments 2D (adc.moments) : l'algebre M->C->S->C'->M'
# est derivee en boucles sur l'AST, seule la fermeture est fournie (ADC-164). Oracle
# d'Isserlis independant + autodiff/eig + robust + lorentz_sources ; la partie compilee
# (AOT + System hll) s'auto-saute sans compilateur ou sans Kokkos.
add_test(NAME test_dsl_moments
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_moments.py)
set_tests_properties(test_dsl_moments PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Codegen C++ du DSL : compile la fonction flux generee et la compare a l'interprete numpy.
add_test(NAME test_dsl_codegen
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_codegen.py)
set_tests_properties(test_dsl_codegen PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Emballage en brique compilee : compile la brique generee contre les en-tetes adc, verifie le
# concept HyperbolicModel et l'egalite avec adc::Euler (4 var, sans aux) et adc::ExBVelocity
# (1 var, flux dependant des champs auxiliaires). -I include passe via le source du test.
add_test(NAME test_dsl_brick
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_brick.py)
set_tests_properties(test_dsl_brick PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Codegen de la SOURCE : la brique de source generee == adc::PotentialForce ecrite a la main.
add_test(NAME test_dsl_source
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_source.py)
set_tests_properties(test_dsl_source PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Composition : CompositeModel<EulerGen, NoSource, ChargeDensity> satisfait PhysicalModel et egale
# la version ecrite a la main.
add_test(NAME test_dsl_compose
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_compose.py)
set_tests_properties(test_dsl_compose PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# JIT-lite end-to-end : formules Python -> .hpp genere -> g++ compile un driver volumes-finis ->
# residu Rusanov identique a adc::Euler.
add_test(NAME test_dsl_jit
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_jit.py)
set_tests_properties(test_dsl_jit PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Codegen elliptique : la brique de second membre generee == adc::ChargeDensity.
add_test(NAME test_dsl_elliptic
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_elliptic.py)
set_tests_properties(test_dsl_elliptic PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# CSE : cse=True et cse=False donnent le meme resultat ; la version CSE factorise (moins de sqrt).
add_test(NAME test_dsl_cse
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_cse.py)
set_tests_properties(test_dsl_cse PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# JIT reel : flux genere compile en .so, charge dans le process (ctypes), == interprete numpy.
add_test(NAME test_dsl_jitlib
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_jitlib.py)
set_tests_properties(test_dsl_jitlib PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Dispatch type-erased a l'execution : brique generee -> .so -> dlopen -> adc::IModel<4> == adc::Euler.
add_test(NAME test_dsl_dynamic
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_dynamic.py)
set_tests_properties(test_dsl_dynamic PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Bloc dynamique de bout en bout : DSL -> compile_so -> System.add_dynamic_block -> eval_rhs/step.
add_test(NAME test_dsl_block
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_block.py)
set_tests_properties(test_dsl_block PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Bloc dynamique COUPLE : euler_poisson en formules -> .so (CompositeModel) -> add_dynamic_block ;
# elliptic_rhs cable au Poisson (== compile) + eval_rhs == flux + source de gravite.
add_test(NAME test_dsl_coupled
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_coupled.py)
set_tests_properties(test_dsl_coupled PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Backend AOT : euler_poisson compile_or_jit(mode="compile") -> add_compiled_block == bloc natif.
add_test(NAME test_dsl_aot
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_aot.py)
set_tests_properties(test_dsl_aot PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Backend NATIF "production" : euler_poisson compile(backend="production") -> loader .so qui inline
# add_compiled_model<ProdModel> -> add_native_block. Parite BIT-IDENTIQUE au bloc natif add_block
# (eval_rhs + avance SSPRK2) + rejet explicite d'une cle d'ABI falsifiee. S'auto-saute sans compilateur.
add_test(NAME test_dsl_production
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_production.py)
set_tests_properties(test_dsl_production PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Backend NATIF "production" cote AMR (DSL Phase D) : euler compile(backend="production",
# target="amr_system") -> loader .so inline add_compiled_model(AmrSystem&) -> AmrSystem.add_native_block.
# Parite BIT-IDENTIQUE au bloc natif add_block (transport pur dmax=0, euler_poisson couple < 1e-12) +
# limites AMR enforcees (primitive/roe/hllc/weno5 rejetes) + rejet d'une cle d'ABI falsifiee.
# S'auto-saute sans compilateur.
add_test(NAME test_dsl_production_amr
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_production_amr.py)
set_tests_properties(test_dsl_production_amr PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# AmrSystem::set_conservative_state (etat conservatif initial COMPLET sur AMR, mono-bloc) : demarrer
# l'AMR depuis l'etat de derive (rho, rho*u, rho*v) au lieu de m=0. Gardes (ndim/vide/taille/multi-bloc),
# NO-DEFAULT-CHANGE bit-identique au seed densite, conservation de masse, et la qty de mvt advecte la
# densite. La partie parite s'auto-saute sans compilateur ; les gardes tournent toujours.
add_test(NAME test_amr_conservative_state
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_amr_conservative_state.py)
set_tests_properties(test_amr_conservative_state PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Choix de reconstruction (ordre) du bloc dynamique : recon none/minmod/vanleer == reference MUSCL numpy.
add_test(NAME test_dsl_recon
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_recon.py)
set_tests_properties(test_dsl_recon PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Operateur elliptique a permittivite constante : phi(eps=2) == phi(eps=1)/2.
add_test(NAME test_poisson_eps
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_poisson_eps.py)
set_tests_properties(test_poisson_eps PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# WENO5-Z + SSPRK3 exposes depuis l'API : adc.Spatial(limiter="weno5") + adc.Explicit(method="ssprk3")
# end-to-end, defaut SSPRK2 inchange (bit-identique), garde-fous sur les chemins compiles (.so).
add_test(NAME test_weno5_ssprk3
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weno5_ssprk3.py)
set_tests_properties(test_weno5_ssprk3 PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# SSPRK3 sur le chemin de PRODUCTION (loader natif .so), pas seulement add_block : add_native_block
# accepte time="ssprk3" et add_compiled_model threade le schema RK jusqu'au make_block du .so. Garde
# amont (ssprk3 accepte / rk4 rejete) sans compilateur + parite BIT-IDENTIQUE production==add_block et
# production+ssprk3 != production+ssprk2. S'auto-saute sans compilateur pour la partie parite.
add_test(NAME test_ssprk3_production
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ssprk3_production.py)
set_tests_properties(test_ssprk3_production PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# SSPRK3 (ordre 3) sur AMR avec REFLUX PAR ETAGE (ADC-64) : adc.Explicit(ssprk3=True) cable mono-bloc
# (AmrCouplerMP) ET multi-blocs (AmrRuntime), reflux du FLUX EFFECTIF Feff = 1/6 F0 + 1/6 F1 + 2/3 F2.
# (a) mono + multi tournent finis + masse conservee avec patchs fins ; (b) defaut euler bit-identique ;
# (c) erreur L1 temporelle ssprk3 << euler au meme dt (reference Richardson) ; (d) imex+ssprk3 rejete ;
# (e) loader .so + ssprk3 rejete (ABI plate). Test PUR Python (aucune compilation .so), toujours execute.
add_test(NAME test_amr_ssprk3
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_amr_ssprk3.py)
set_tests_properties(test_amr_ssprk3 PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Politique de la loi de Gauss (R0, repro Hoffart) : System.set_gauss_policy "restart" (defaut,
# re-resout Poisson chaque pas, bit-identique) vs "evolve" (apres phi^0, plus de re-solve : l'etage
# Schur fait evoluer phi sans restart). API + NO-DEFAULT-CHANGE (defaut == restart) + evolve actif et
# fini sur un fluide magnetise + CondensedSchur. La partie comportement s'auto-saute sans compilateur.
add_test(NAME test_gauss_policy
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gauss_policy.py)
set_tests_properties(test_gauss_policy PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# Flux HLL expose pour les modeles a vitesses d'onde signees SANS exiger une pression (3-var) :
# make_block accepte riemann="hll" des que model.wave_speeds existe (requires-gate, pas n_vars==4 ni
# pressure). HLL natif 4-var (fini, masse conservee, != Rusanov) + accepte sur DSL 3-var isotherme
# qui declare 'p' (wave_speeds emis) + rejete clair sans 'p'. Partie DSL s'auto-saute sans compilateur.
add_test(NAME test_hll_isothermal
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hll_isothermal.py)
set_tests_properties(test_hll_isothermal PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# WENO5 sur les chemins .so / CompiledModel : add_compiled_block (AOT) ET add_native_block (production)
# allouent block_n_ghost(limiter)=3 ghosts -> weno5 accepte ; parite eval_rhs/potentiel au bloc natif
# add_block weno5 (AOT < 1e-9 marshale, production bit-identique) + no-default-change none/minmod.
# S'auto-saute sans compilateur.
add_test(NAME test_weno5_compiledmodel
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weno5_compiledmodel.py)
set_tests_properties(test_weno5_compiledmodel PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# DSL Phase A : API utilisateur stable (dsl.Model + Param + CompiledModel + add_equation +
# FiniteVolume + run). Erreurs explicites + aot/production end-to-end (production -> add_native_block).
add_test(NAME test_dsl_phase_a
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_phase_a.py)
set_tests_properties(test_dsl_phase_a PROPERTIES
  ENVIRONMENT "${_adc_py_test_env}")

# --- Garde-fou : tests des chantiers aux extensible / EPM / roles, jusqu'ici lances en CI
# (ci.yml) mais PAS enregistres en ctest -> on les rend visibles a `ctest` (et donc enforces
# localement avant tout refactor). Les chemins JIT/AOT (test_dsl_{jit,aot}_{bz,te}) compilent
# un .so a la volee et s'AUTO-SAUTENT (exit 0) si aucun compilateur C++ n'est trouve. ---
foreach(t
    test_dsl_roles            # la brique generee declare ses VariableRole
    test_dsl_aux_naux         # le DSL emet n_aux quand une formule lit un champ aux extra
    test_magnetic_field       # binding set_magnetic_field (B_z)
    test_poisson_composite    # second membre de Poisson de systeme generique
    test_poisson_eps_aniso    # permittivite anisotrope eps_x/eps_y cote Python
    test_poisson_screened     # operateur ecrante / Helmholtz
    test_dsl_jit_bz           # B_z end-to-end via add_dynamic_block (.so JIT)
    test_dsl_jit_te           # T_e (derive p/rho) via add_dynamic_block (.so JIT)
    test_dsl_aot_bz           # B_z via add_compiled_block (.so AOT)
    test_dsl_aot_te           # T_e via add_compiled_block (.so AOT)
    test_aux_named            # ADC-70 phase 1 : champs aux NOMMES par modele (m.aux_field) sur System cartesien
    test_block_names_dynamic  # block_names() voit le bloc dynamique (.so JIT), pas seulement add_block
    test_dsl_abi_metadata     # noms/roles/gamma transportes par l'ABI du .so (AOT+JIT) + retro-compat
    test_dsl_compile_facade   # facade m.compile(backend=...) -> moteurs JIT/AOT + metadonnees preservees
    test_dsl_compile_cache    # ergonomie m.compile() : auto-detect include + cache so_path (HIT/MISS)
    test_stride               # exposition du parametre stride (P1 : cadence par bloc, runtime)
    test_amr_production_stride_reject  # rejet stride/masque IMEX sur le chemin production AMR (.so loader, revue #195)
    test_primitive_state      # init/diagnostic en primitif (P4) : set/get_primitive_state, natif + DSL
    test_schur_via_system     # System -> run_source_stage -> etage Schur condense, briques natives (no AOT)
    test_amr_schur_via_system # AmrSystem -> etage source condense GLOBAL (amr-schur) mono-niveau, briques natives (no AOT)
    test_polar_schur_via_system  # System POLAIRE -> run_source_stage -> etage Schur condense POLAIRE (Voie A etape 2c -> PR6), briques natives
    test_polar_theta_boxes    # System POLAIRE multi-box (decoupage theta, ADC-67) : transport bit-identique theta_boxes=1/2/4, rejets (divisibilite + Poisson direct multi-box), round-trip get/set state multi-box
    test_schur_conservation   # conservation discrete masse/momentum/energie + positivite du chemin cartesien-fluide-Schur (briques natives, no AOT)
    test_disc_domain_mask     # chantier T2 : contrat masque de domaine disque (inerte par defaut) -- disc_mask() tout-actif, inertie de step(), masque == level set, garde-fous
    test_fv_hll_minmod        # chemin generique HLL+minmod VISIBLE : FiniteVolume(riemann="hll") sur isotherme 3-var (non Euler), System + AMR, rejets explicites (hll sans wave_speeds, hllc hors Euler)
    test_newton_options       # options/diagnostics Newton IMEX (audit ch.2) : no-default-change bit-identique, rapport (residu/iters/echecs), tolerance, rejets explicites
    test_imexrk               # famille IMEX-RK schema ARS(2,2,2) (ADC-69) : ordre 2 (gyration cyclotron, ratio ~4 vs IMEX ~2), stabilite raide omega*dt=50, defaut IMEX distinct/inchange, rejets AMR/polaire/Strang/masque partiel
    test_dt_bounds            # politique de pas generique (audit ch.1) : add_dt_bound/last_dt_bound, fallback transport historique, DSL stability_speed/stability_dt (auto-skip sans compilateur)
    test_check_model          # check_model generique (audit ch.6) : formules DSL (finitude/round-trip/positivite) + bloc installe (residu fini, roles, etat restaure)
    test_magnetic_bricks      # briques magnetiques exposees (audit ch.5) : MagneticLorentzForce / PotentialMagneticForce via adc.Model, residu quantitatif q B m, rejet scalaire
    test_role_aware_bricks    # briques natives role-aware (§5) : no-default-change quantitatif (residu magnetique q B m, energie intouchee), API PotentialForce/ChargeDensity, determinisme
    test_amr_weno5_hllc_roe   # registry des tags (§8) : alignement System/AMR weno5 x {hllc,roe} (mono dispatch_amr_compiled + multi dispatch_amr_block), garde de capability intacte
    test_schur_roles          # roles/champs Schur transportes dans l'ABI (vague 2) : descripteurs par nom == canonique bit-identique, rejets explicites (inconnu/polaire/amr)
    test_io_checkpoint        # IO v1 (vague 2) : checkpoint/restart bit-identique (stride=2 via macro_step), write npz/vtk, rejets composition/grille
    test_amr_checkpoint       # checkpoint/restart AMR bit-identique mono-bloc mono-rang (ADC-65) : etat complet par niveau + phi warm-start + hierarchie imposee, rejets multi-blocs/n/regrid_every
    test_v3_features          # vague 3 : CoupledSource.frequency, options Newton AMR (multi natif / rejets mono+diagnostics), set_conservative_state multi-blocs, enable_hllc + source_jacobian DSL (auto-skip)
    test_amr_newton_full      # vague 3 solde : options Newton AMR mono-bloc (coupleur) + newton_diagnostics multi-blocs (newton_report), no-default-change, rejets .so
    test_dsl_roe              # solde audit pt 11 : m.enable_roe() emis depuis les ROLES -- parite 4-var avec le Roe canonique, 3-var isotherme accepte (cisaillement exact), rejets explicites (auto-skip)
    test_dsl_autodiff_roe     # ADC-68 voie FOURNIE : dsl.diff (autodiff Expr) + m.flux_jacobian + m.roe_dissipation(x=,y=) (left/right) reproduisant le Roe isotherme a la main == enable_roe (~1e-12) ; (a)(b)(d) symboliques, (c) compile (auto-skip)
    test_coupled_freq_expr    # frequence PAR CELLULE des sources couplees (GENERICITY 7) : constante (regression), Expr mu(U)=k*rho (dt==cfl/(k*max rho)), suivi d'etat, rejet (bloc,role) non declare, AMR grossier
    test_backend_auto         # ADC-63 : backend par defaut 'auto' = production si parite toolchain etablie, aot sinon (raison posee, jamais muet) ; explicite inchange (auto-skip)
    test_native_abi_std       # regression ABI : le std du modele natif (backend="production") suit le loader (c++20 sous Kokkos, c++23 sinon) -> add_native_block charge sans "ABI incompatible"
    # --- ADC-104 : tests DSL livres mais jamais enregistres en ctest. Ils etaient orphelins du grep
    # litteral (masques par l'interpolation tests/${t}.py de cette boucle) et NE tournaient PAS en
    # ctest ; en CI ils etaient bien lances par auto-decouverte (find|xargs) mais l'echec etait
    # AVALE (le `bash -c '...; python3 {}; echo'` se termine sur un echo -> exit 0). Aucun garde-fou,
    # donc -- c'est ce trou qui a laisse passer la regression Kokkos-only #295.
    # NB : test_dsl_hybrid et test_dsl_hybrid_bz (hybride AOT) sont VOLONTAIREMENT differes ici : ils
    # echouent sur master tant que le correctif #295 (ADC-103 : flags Kokkos sur le chemin aot de
    # HybridModel.compile) n'est pas merge. Ils seront enregistres dans le suivi gate sur #295 (un
    # garde-fou vert ne peut pas preceder le correctif qu'il protege). Les hybrides du chemin
    # "production" (hybrid_amr, hybrid_coupling) passent deja et sont enregistres ci-dessous. ---
    test_dsl_hybrid_amr       # hybride cote AMR : CompositeModel.compile(backend="production", target="amr_system") (.so), rejets aot/system, auto-skip
    test_dsl_hybrid_coupling  # hybride couple ions/elec : transport DSL + elliptic natif ChargeDensity compile production (.so), auto-skip
    test_dsl_coupled_source   # source COUPLEE inter-especes par le DSL (CoupledSource.compile(backend="production"), .so) == reference numpy forward-Euler
    test_dsl_coupled_source_conservation  # echange CONSERVATIF par paire (add_pair) compile production (.so) : masse totale invariante ~1e-12, rejets
    test_dsl_coupled_role_error  # verrou rejet role inconnu : ValueError cote DSL (pur Python) + RuntimeError cote add_coupled_source (compile production, .so)
    test_dsl_runtime_params   # parametres RUNTIME du DSL : const inline vs params.get(0) seede, cache keye model_hash+abi, compile AOT (.so), auto-skip
    test_dsl_wave_speeds_no_pressure)  # m.wave_speeds(x=,y=) sans primitive 'p' -> riemann='hll' ; modele jouet acoustique, compile AOT (.so), auto-skip via dsl._default_cxx
  add_test(NAME ${t}
           COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.py)
  set_tests_properties(${t} PROPERTIES
    ENVIRONMENT "${_adc_py_test_env}")
endforeach()

# Critere d'arret mixte rel/abs du GeometricMG expose par set_poisson(abs_tol=...) : early-exit hors
# pas (etat inchange) + no-default-change. Exige le binding abs_tol (reconstruction _adc) -> CI seule.
add_test(NAME test_mg_cold_tolerance
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mg_cold_tolerance.py)
set_tests_properties(test_mg_cold_tolerance PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

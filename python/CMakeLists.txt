# Module Python pybind11 de la lib adc_cpp. L'extension compilee s'appelle `_adc`
# et est deposee dans un PAQUET `adc/` (avec __init__.py : le sucre lisible
# Spatial/Explicit/IMEX/System + adc.integrate). Avec PYTHONPATH = <build>/python,
# `import adc` charge le paquet, qui importe l'extension _adc.
#
#   cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc
#   PYTHONPATH=build-py/python python3 -c "import adc; print(adc.System)"

pybind11_add_module(_adc bindings.cpp system.cpp amr_system.cpp)
target_link_libraries(_adc PRIVATE adc::adc ${CMAKE_DL_LIBS})  # dl : add_dynamic_block (dlopen)
target_compile_features(_adc PRIVATE cxx_std_${ADC_CXX_STD})

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
foreach(pyf adc/__init__.py adc/integrate.py adc/dsl.py)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${pyf} ${CMAKE_CURRENT_BINARY_DIR}/${pyf} COPYONLY)
endforeach()

# Test des bindings : importe le paquet et verifie les invariants (composition, masse,
# Poisson, integrateur temporel Python). Enregistre des que le module est construit (pas
# garde par ADC_BUILD_TESTS), pour que `ctest --test-dir build-py` le voie meme avec
# ADC_BUILD_TESTS=OFF. Lance avec le MEME interpreteur que celui ayant compile _adc.
# Le test vit dans python/tests/ (pas python/) : sinon sys.path[0] (dossier du script)
# contiendrait le paquet source adc/ et masquerait le paquet construit.
add_test(NAME test_bindings
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py)
set_tests_properties(test_bindings PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Test du mini-DSL symbolique (adc.dsl) : pur Python (interprete CPU), meme PYTHONPATH.
add_test(NAME test_dsl
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl.py)
set_tests_properties(test_dsl PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Codegen C++ du DSL : compile la fonction flux generee et la compare a l'interprete numpy.
add_test(NAME test_dsl_codegen
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_codegen.py)
set_tests_properties(test_dsl_codegen PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Emballage en brique compilee : compile la brique generee contre les en-tetes adc, verifie le
# concept HyperbolicModel et l'egalite avec adc::Euler (4 var, sans aux) et adc::ExBVelocity
# (1 var, flux dependant des champs auxiliaires). -I include passe via le source du test.
add_test(NAME test_dsl_brick
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_brick.py)
set_tests_properties(test_dsl_brick PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Codegen de la SOURCE : la brique de source generee == adc::PotentialForce ecrite a la main.
add_test(NAME test_dsl_source
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_source.py)
set_tests_properties(test_dsl_source PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Composition : CompositeModel<EulerGen, NoSource, ChargeDensity> satisfait PhysicalModel et egale
# la version ecrite a la main.
add_test(NAME test_dsl_compose
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_compose.py)
set_tests_properties(test_dsl_compose PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# JIT-lite end-to-end : formules Python -> .hpp genere -> g++ compile un driver volumes-finis ->
# residu Rusanov identique a adc::Euler.
add_test(NAME test_dsl_jit
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_jit.py)
set_tests_properties(test_dsl_jit PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Codegen elliptique : la brique de second membre generee == adc::ChargeDensity.
add_test(NAME test_dsl_elliptic
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_elliptic.py)
set_tests_properties(test_dsl_elliptic PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# CSE : cse=True et cse=False donnent le meme resultat ; la version CSE factorise (moins de sqrt).
add_test(NAME test_dsl_cse
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_cse.py)
set_tests_properties(test_dsl_cse PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# JIT reel : flux genere compile en .so, charge dans le process (ctypes), == interprete numpy.
add_test(NAME test_dsl_jitlib
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_jitlib.py)
set_tests_properties(test_dsl_jitlib PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Dispatch type-erased a l'execution : brique generee -> .so -> dlopen -> adc::IModel<4> == adc::Euler.
add_test(NAME test_dsl_dynamic
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_dynamic.py)
set_tests_properties(test_dsl_dynamic PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Bloc dynamique de bout en bout : DSL -> compile_so -> System.add_dynamic_block -> eval_rhs/step.
add_test(NAME test_dsl_block
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_block.py)
set_tests_properties(test_dsl_block PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Bloc dynamique COUPLE : euler_poisson en formules -> .so (CompositeModel) -> add_dynamic_block ;
# elliptic_rhs cable au Poisson (== compile) + eval_rhs == flux + source de gravite.
add_test(NAME test_dsl_coupled
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_coupled.py)
set_tests_properties(test_dsl_coupled PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Backend AOT : euler_poisson compile_or_jit(mode="compile") -> add_compiled_block == bloc natif.
add_test(NAME test_dsl_aot
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_aot.py)
set_tests_properties(test_dsl_aot PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Backend NATIF "production" : euler_poisson compile(backend="production") -> loader .so qui inline
# add_compiled_model<ProdModel> -> add_native_block. Parite BIT-IDENTIQUE au bloc natif add_block
# (eval_rhs + avance SSPRK2) + rejet explicite d'une cle d'ABI falsifiee. S'auto-saute sans compilateur.
add_test(NAME test_dsl_production
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_production.py)
set_tests_properties(test_dsl_production PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Backend NATIF "production" cote AMR (DSL Phase D) : euler compile(backend="production",
# target="amr_system") -> loader .so inline add_compiled_model(AmrSystem&) -> AmrSystem.add_native_block.
# Parite BIT-IDENTIQUE au bloc natif add_block (transport pur dmax=0, euler_poisson couple < 1e-12) +
# limites AMR enforcees (primitive/roe/hllc/weno5 rejetes) + rejet d'une cle d'ABI falsifiee.
# S'auto-saute sans compilateur.
add_test(NAME test_dsl_production_amr
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_production_amr.py)
set_tests_properties(test_dsl_production_amr PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Choix de reconstruction (ordre) du bloc dynamique : recon none/minmod/vanleer == reference MUSCL numpy.
add_test(NAME test_dsl_recon
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_recon.py)
set_tests_properties(test_dsl_recon PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# Operateur elliptique a permittivite constante : phi(eps=2) == phi(eps=1)/2.
add_test(NAME test_poisson_eps
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_poisson_eps.py)
set_tests_properties(test_poisson_eps PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# WENO5-Z + SSPRK3 exposes depuis l'API : adc.Spatial(limiter="weno5") + adc.Explicit(method="ssprk3")
# end-to-end, defaut SSPRK2 inchange (bit-identique), garde-fous sur les chemins compiles (.so).
add_test(NAME test_weno5_ssprk3
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weno5_ssprk3.py)
set_tests_properties(test_weno5_ssprk3 PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# WENO5 sur les chemins .so / CompiledModel : add_compiled_block (AOT) ET add_native_block (production)
# allouent block_n_ghost(limiter)=3 ghosts -> weno5 accepte ; parite eval_rhs/potentiel au bloc natif
# add_block weno5 (AOT < 1e-9 marshale, production bit-identique) + no-default-change none/minmod.
# S'auto-saute sans compilateur.
add_test(NAME test_weno5_compiledmodel
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_weno5_compiledmodel.py)
set_tests_properties(test_weno5_compiledmodel PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

# DSL Phase A : API utilisateur stable (dsl.Model + Param + CompiledModel + add_equation +
# FiniteVolume + run). Erreurs explicites + aot/production end-to-end (production -> add_native_block).
add_test(NAME test_dsl_phase_a
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_phase_a.py)
set_tests_properties(test_dsl_phase_a PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

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
    test_block_names_dynamic  # block_names() voit le bloc dynamique (.so JIT), pas seulement add_block
    test_dsl_abi_metadata     # noms/roles/gamma transportes par l'ABI du .so (AOT+JIT) + retro-compat
    test_dsl_compile_facade   # facade m.compile(backend=...) -> moteurs JIT/AOT + metadonnees preservees
    test_dsl_compile_cache    # ergonomie m.compile() : auto-detect include + cache so_path (HIT/MISS)
    test_stride               # exposition du parametre stride (P1 : cadence par bloc, runtime)
    test_primitive_state      # init/diagnostic en primitif (P4) : set/get_primitive_state, natif + DSL
    test_schur_via_system)    # System -> run_source_stage -> etage Schur condense, briques natives (no AOT)
  add_test(NAME ${t}
           COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.py)
  set_tests_properties(${t} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

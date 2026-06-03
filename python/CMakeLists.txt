# Module Python pybind11 de la lib adc_cpp. L'extension compilee s'appelle `_adc`
# et est deposee dans un PAQUET `adc/` (avec __init__.py : le sucre lisible
# Spatial/Explicit/IMEX/System + adc.integrate). Avec PYTHONPATH = <build>/python,
# `import adc` charge le paquet, qui importe l'extension _adc.
#
#   cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc
#   PYTHONPATH=build-py/python python3 -c "import adc; print(adc.System)"

pybind11_add_module(_adc bindings.cpp system.cpp amr_system.cpp two_fluid_ap_solver.cpp)
target_link_libraries(_adc PRIVATE adc::adc ${CMAKE_DL_LIBS})  # dl : add_dynamic_block (dlopen)
target_compile_features(_adc PRIVATE cxx_std_${ADC_CXX_STD})

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

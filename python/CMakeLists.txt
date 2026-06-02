# Module Python pybind11 de la lib adc_cpp. L'extension compilee s'appelle `_adc`
# et est deposee dans un PAQUET `adc/` (avec __init__.py : le sucre lisible
# Spatial/Explicit/IMEX/System + adc.integrate). Avec PYTHONPATH = <build>/python,
# `import adc` charge le paquet, qui importe l'extension _adc.
#
#   cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc
#   PYTHONPATH=build-py/python python3 -c "import adc; print(adc.System)"

pybind11_add_module(_adc bindings.cpp system.cpp amr_system.cpp two_fluid_ap_solver.cpp)
target_link_libraries(_adc PRIVATE adc::adc)
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
# concept HyperbolicModel et l'egalite avec adc::Euler. -I include passe via le source du test.
add_test(NAME test_dsl_brick
         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dsl_brick.py)
set_tests_properties(test_dsl_brick PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

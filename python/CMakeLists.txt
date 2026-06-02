# Module Python pybind11 de la lib adc_cpp. L'extension compilee s'appelle `_adc`
# et est deposee dans un PAQUET `adc/` (avec __init__.py : le sucre lisible
# Spatial/Explicit/IMEX/System + adc.integrate). Avec PYTHONPATH = <build>/python,
# `import adc` charge le paquet, qui importe l'extension _adc.
#
#   cmake -B build-py -DADC_BUILD_PYTHON=ON && cmake --build build-py --target _adc
#   PYTHONPATH=build-py/python python3 -c "import adc; print(adc.System)"

pybind11_add_module(_adc bindings.cpp system.cpp
                    two_fluid_ap_solver.cpp diocotron_amr_solver.cpp)
target_link_libraries(_adc PRIVATE adc::adc)
target_compile_features(_adc PRIVATE cxx_std_${ADC_CXX_STD})

# L'extension va dans le paquet adc/ ; on y copie aussi les sources Python du paquet.
set(ADC_PY_PKG ${CMAKE_CURRENT_BINARY_DIR}/adc)
set_target_properties(_adc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ADC_PY_PKG})
foreach(pyf adc/__init__.py adc/integrate.py)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${pyf} ${CMAKE_CURRENT_BINARY_DIR}/${pyf} COPYONLY)
endforeach()

# Test : importe le paquet et verifie les invariants (composition, masse, Poisson,
# integrateur temporel Python). PYTHONPATH = dossier contenant le paquet adc/.
if(ADC_BUILD_TESTS)
  # Le test vit dans python/tests/ (PAS dans python/) pour que sys.path[0] (= dossier
  # du script) ne contienne pas le paquet source adc/ qui masquerait le paquet construit.
  add_test(NAME test_bindings
           COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bindings.py)
  set_tests_properties(test_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endif()

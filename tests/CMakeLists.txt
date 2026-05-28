# Tests sans dependance externe pour ce premier socle (executable + CTest).
# Catch2 sera cable quand la surface de test le justifiera.

function(adc_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE adc::adc)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

adc_add_test(test_diocotron_model)
adc_add_test(test_box2d)
adc_add_test(test_fab2d)
adc_add_test(test_box_array)
adc_add_test(test_multifab)
adc_add_test(test_fill_boundary)
adc_add_test(test_physical_bc)
adc_add_test(test_refinement)
adc_add_test(test_amr_hierarchy)
adc_add_test(test_cluster)

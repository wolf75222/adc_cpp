add_executable(diocotron diocotron.cpp)
target_link_libraries(diocotron PRIVATE adc::adc)

add_executable(diocotron_column diocotron_column.cpp)
target_link_libraries(diocotron_column PRIVATE adc::adc)

add_executable(diocotron_amr diocotron_amr.cpp)
target_link_libraries(diocotron_amr PRIVATE adc::adc)

add_executable(diocotron_amr3 diocotron_amr3.cpp)
target_link_libraries(diocotron_amr3 PRIVATE adc::adc)

add_executable(diocotron_mpi diocotron_mpi.cpp)
target_link_libraries(diocotron_mpi PRIVATE adc::adc)

if(ADC_HAS_EIGEN)
  add_executable(diocotron_theory diocotron_theory.cpp)
  target_link_libraries(diocotron_theory PRIVATE adc::adc adc_eigen)
endif()

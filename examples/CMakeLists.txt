# Pilotes MINCES sur la facade compilee (adc::solver) : ils ne font que CI (via
# config), diagnostics (lecture des champs) et I/O. Toute la physique est dans libadc.
add_executable(diocotron diocotron.cpp)
target_link_libraries(diocotron PRIVATE adc::solver)

add_executable(diocotron_column diocotron_column.cpp)
target_link_libraries(diocotron_column PRIVATE adc::solver)

# Demos du MOTEUR bas-niveau (AMR, MPI, theorie) : capacites hors perimetre de la
# facade -> ils pilotent directement la pile generique (adc::adc).
add_executable(diocotron_amr diocotron_amr.cpp)
target_link_libraries(diocotron_amr PRIVATE adc::adc)

add_executable(diocotron_amr3 diocotron_amr3.cpp)
target_link_libraries(diocotron_amr3 PRIVATE adc::adc)

add_executable(diocotron_multipatch diocotron_multipatch.cpp)
target_link_libraries(diocotron_multipatch PRIVATE adc::adc)

add_executable(diocotron_mpi diocotron_mpi.cpp)
target_link_libraries(diocotron_mpi PRIVATE adc::adc)

if(ADC_HAS_EIGEN)
  add_executable(diocotron_theory diocotron_theory.cpp)
  target_link_libraries(diocotron_theory PRIVATE adc::adc adc_eigen)
endif()

# Demos GPU (Kokkos) : integres au build principal quand le backend Kokkos est
# configure (-DADC_USE_KOKKOS=ON). Ils lient `adc` et heritent du backend ; aucun
# drapeau rajoute. Hors Kokkos, on les ignore (ils n'ont de sens que sur GPU).
if(ADC_USE_KOKKOS)
  add_subdirectory(gpu)
endif()

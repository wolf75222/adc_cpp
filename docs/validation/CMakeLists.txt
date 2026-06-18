# Driver diocotron HyQMOM GPU/CPU (briques DSL emises + System header-only).
# Local (shakeout, Kokkos OpenMP) :
#   cmake -S . -B build -DADC_INCLUDE=$ADC/include -DKokkos_ROOT=/opt/homebrew/anaconda3 \
#         -DCMAKE_BUILD_TYPE=Debug && cmake --build build
# ROMEO armgpu (CUDA H100) -- cf. diocotron_gpu.sbatch :
#   cmake -S . -B build -DCMAKE_CXX_COMPILER=$HOME/adc_gpu_p1/kinstall/bin/nvcc_wrapper \
#         -DKokkos_ROOT=$HOME/adc_gpu_p1/kinstall -DADC_INCLUDE=... -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.20)
project(diocotron_gpu CXX)
find_package(Kokkos REQUIRED)
if(NOT ADC_INCLUDE)
  message(FATAL_ERROR "Passer -DADC_INCLUDE=/chemin/vers/adc_cpp/include")
endif()
# adc::System n'est pas header-only : ses definitions vivent dans python/system.cpp
# (TU pur C++, sans pybind) -- on le compile DANS le driver (grosse TU, RAM en consequence).
add_executable(diocotron_gpu diocotron_gpu.cpp ${ADC_INCLUDE}/../python/system.cpp)
target_include_directories(diocotron_gpu PRIVATE ${ADC_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(diocotron_gpu PRIVATE Kokkos::kokkos)
target_compile_definitions(diocotron_gpu PRIVATE ADC_HAS_KOKKOS)  # coeur Kokkos-only (#263)
target_compile_features(diocotron_gpu PRIVATE cxx_std_20)

# Driver AMR multi-boite (ADC-320, diocotron_amr_mpi.sbatch) : adc::AmrSystem cable le composite
# hyqmom15 + Poisson sur un grossier MULTI-BOITE reparti (distribute_coarse=true). Ses definitions
# vivent dans python/amr_system.cpp (TU pur C++, le PENDANT AMR de system.cpp -- on ne compile PAS
# system.cpp ici, le bloc unique add_compiled_model(AmrSystem&) passe par AmrCouplerMP<Model>).
add_executable(diocotron_amr_gpu diocotron_amr_gpu.cpp ${ADC_INCLUDE}/../python/amr_system.cpp)
target_include_directories(diocotron_amr_gpu PRIVATE ${ADC_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(diocotron_amr_gpu PRIVATE Kokkos::kokkos)
target_compile_definitions(diocotron_amr_gpu PRIVATE ADC_HAS_KOKKOS)  # coeur Kokkos-only (#263)
target_compile_features(diocotron_amr_gpu PRIVATE cxx_std_20)

# Validation multi-GPU + MPI (cf. diocotron_mpi.sbatch / diocotron_amr_mpi.sbatch) :
# -DADC_VALIDATION_MPI=ON lie OpenMPI CUDA-aware aux drivers et definit ADC_HAS_MPI, ce qui bascule
# comm.hpp + les collectifs (state_global / step_cfl pour System ; coarse fill_boundary +
# density/mass/potential all-reduce pour AmrSystem, deja compiles DANS chaque driver) sur le vrai
# chemin MPI. Defaut OFF : les binaires serie restent identiques au bit pres.
option(ADC_VALIDATION_MPI "Lier MPI aux drivers diocotron (validation multi-GPU + MPI)" OFF)
if(ADC_VALIDATION_MPI)
  enable_language(C)  # find_package(MPI) exige la cible MPI::MPI_C (cf. racine CMakeLists L9)
  find_package(MPI REQUIRED)
  target_link_libraries(diocotron_gpu PRIVATE MPI::MPI_CXX)
  target_compile_definitions(diocotron_gpu PRIVATE ADC_HAS_MPI)
  target_link_libraries(diocotron_amr_gpu PRIVATE MPI::MPI_CXX)
  target_compile_definitions(diocotron_amr_gpu PRIVATE ADC_HAS_MPI)
endif()

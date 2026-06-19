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
# adc::System n'est pas header-only : ses definitions vivent dans python/system.cpp ET, depuis le
# split ADC-335/342, dans les sous-TU par transport/flux -- le dispatch add_block delegue aux seams
# build_block_<transport> / build_block_compressible_<flux> definis la. Il faut donc compiler TOUTES
# ces sous-TU dans le driver, sinon le link echoue (undefined reference build_block_*). Meme liste que
# python/CMakeLists.txt (_adc) et tests/CMakeLists.txt (adc_runtime_system).
set(ADC_SYSTEM_SEAM_TUS
  ${ADC_INCLUDE}/../python/system.cpp
  ${ADC_INCLUDE}/../python/system_exb.cpp
  ${ADC_INCLUDE}/../python/system_isothermal_rusanov.cpp
  ${ADC_INCLUDE}/../python/system_isothermal_hll.cpp
  ${ADC_INCLUDE}/../python/system_compressible_rusanov.cpp
  ${ADC_INCLUDE}/../python/system_compressible_hll.cpp
  ${ADC_INCLUDE}/../python/system_compressible_hllc.cpp
  ${ADC_INCLUDE}/../python/system_compressible_roe.cpp
  ${ADC_INCLUDE}/../python/system_polar.cpp)
add_executable(diocotron_gpu diocotron_gpu.cpp ${ADC_SYSTEM_SEAM_TUS})
target_include_directories(diocotron_gpu PRIVATE ${ADC_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(diocotron_gpu PRIVATE Kokkos::kokkos)
target_compile_definitions(diocotron_gpu PRIVATE ADC_HAS_KOKKOS)  # coeur Kokkos-only (#263)
target_compile_features(diocotron_gpu PRIVATE cxx_std_20)

# Driver AMR multi-boite (ADC-320, diocotron_amr_mpi.sbatch) : adc::AmrSystem cable le composite
# hyqmom15 + Poisson sur un grossier MULTI-BOITE reparti (distribute_coarse=true). Ses definitions
# vivent dans python/amr_system.cpp (TU pur C++, le PENDANT AMR de system.cpp -- on ne compile PAS
# system.cpp ici, le bloc unique add_compiled_model(AmrSystem&) passe par AmrCouplerMP<Model>) ET,
# depuis ADC-335, dans les sous-TU AMR par transport (amr_system delegue aux seams
# build_amr_block_<t> / build_amr_compiled_<t>) -- a compiler ici aussi, sinon undefined reference.
set(ADC_AMR_SEAM_TUS
  ${ADC_INCLUDE}/../python/amr_system.cpp
  ${ADC_INCLUDE}/../python/amr_block_exb.cpp
  ${ADC_INCLUDE}/../python/amr_block_isothermal.cpp
  ${ADC_INCLUDE}/../python/amr_block_compressible.cpp
  ${ADC_INCLUDE}/../python/amr_compiled_exb.cpp
  ${ADC_INCLUDE}/../python/amr_compiled_isothermal.cpp
  ${ADC_INCLUDE}/../python/amr_compiled_compressible.cpp)
add_executable(diocotron_amr_gpu diocotron_amr_gpu.cpp ${ADC_AMR_SEAM_TUS})
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

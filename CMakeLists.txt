cmake_minimum_required(VERSION 3.20)

project(adc_cpp
  VERSION 0.0.1
  DESCRIPTION "Solveur advection-diffusion-couplage hyperbolique-elliptique sur AMR, cible cluster"
  LANGUAGES CXX C  # C : pour que find_package(MPI) cree MPI::MPI_C (exige par le
                   # hdf5-config parallele de spack)
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ADC_BUILD_TESTS    "Build tests"    ON)
option(ADC_BUILD_EXAMPLES "Build examples" ON)
option(ADC_USE_OPENMP     "Enable the OpenMP dispatch backend" OFF)
option(ADC_USE_MPI        "Enable the MPI backend (distributed comm seam)" OFF)
option(ADC_USE_HDF5       "Enable the HDF5 DataWriter (parallel output)" OFF)
option(ADC_USE_EIGEN      "Build host-side Eigen analysis tools (diocotron theory)" ON)

# Eigen : cote host uniquement (analyse, hors hot path). Header-only : on
# localise juste le dossier d'include et on l'expose via une cible d'interface
# adc_eigen. Si introuvable, les cibles d'analyse sont sautees.
set(ADC_HAS_EIGEN OFF)
if(ADC_USE_EIGEN)
  execute_process(COMMAND brew --prefix eigen
                  OUTPUT_VARIABLE EIGEN_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  find_path(EIGEN_INCLUDE_DIR Eigen/Dense
    HINTS "${EIGEN_PREFIX}/include/eigen3" /opt/homebrew/include/eigen3
          /usr/local/include/eigen3 /usr/include/eigen3)
  if(EIGEN_INCLUDE_DIR)
    set(ADC_HAS_EIGEN ON)
    add_library(adc_eigen INTERFACE)
    target_include_directories(adc_eigen INTERFACE ${EIGEN_INCLUDE_DIR})
    message(STATUS "Eigen found at ${EIGEN_INCLUDE_DIR} : building host analysis tools")
  else()
    message(STATUS "Eigen not found : skipping host analysis tools")
  endif()
endif()

# Couche physique + maillage : bibliotheque d'interface (header-only) pour
# l'instant. L'operateur spatial, l'elliptique et l'integrateur viendront en
# couches separees ; le backend parallele (AMR/MPI/Kokkos) reste confine au
# seam de dispatch et ne touche jamais a la couche physique.
add_library(adc INTERFACE)
add_library(adc::adc ALIAS adc)
target_include_directories(adc INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(adc INTERFACE cxx_std_23)

if(ADC_USE_OPENMP)
  # AppleClang / Homebrew clang ne trouvent pas libomp tout seuls : on injecte
  # les hints standards si Homebrew expose libomp.
  if(APPLE AND NOT OpenMP_CXX_FOUND)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
          CACHE STRING "" FORCE)
      set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
      set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib"
          CACHE FILEPATH "" FORCE)
    endif()
  endif()
  find_package(OpenMP REQUIRED)
  target_link_libraries(adc INTERFACE OpenMP::OpenMP_CXX)
endif()

# MPI : backend distribue, confine au seam comm (parallel/comm.hpp). Optionnel :
# par defaut OFF, le build et les tests restent serie (comm.hpp = rang unique).
# Quand ON, comm.hpp passe par MPI_Comm_rank/size + collectives, et le test
# test_mpi_smoke tourne via mpirun. ADC_HAS_MPI est propage a tout ce qui lie adc.
set(ADC_HAS_MPI OFF)
if(ADC_USE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(adc INTERFACE MPI::MPI_CXX)
  target_compile_definitions(adc INTERFACE ADC_HAS_MPI)
  set(ADC_HAS_MPI ON)
  message(STATUS "MPI found (${MPIEXEC_EXECUTABLE}) : enabling distributed comm seam")
endif()

# Sortie HDF5 (DataWriter parallele). En serie : ecriture directe. En parallele
# (HDF5 MPI-IO) : chaque rang ecrit ses boites par hyperslab, sans gather.
set(ADC_HAS_HDF5 OFF)
if(ADC_USE_HDF5)
  # MODULE (wrappers h5cc/h5pcc + variables) plutot que CONFIG : le hdf5-config.cmake
  # de spack exige une cible importee MPI::MPI_C que ce projet CXX-only ne cree pas.
  find_package(HDF5 MODULE REQUIRED COMPONENTS C)
  target_link_libraries(adc INTERFACE ${HDF5_C_LIBRARIES})
  target_include_directories(adc INTERFACE ${HDF5_C_INCLUDE_DIRS})
  target_compile_definitions(adc INTERFACE ADC_HAS_HDF5)
  set(ADC_HAS_HDF5 ON)
  message(STATUS "HDF5 found (${HDF5_VERSION}, parallel=${HDF5_IS_PARALLEL}) : enabling DataWriter")
endif()

# Facade COMPILEE : solveurs concrets (PIMPL) instanciant la pile template
# header-only une seule fois -> une vraie librairie libadc, API stable sans
# template (ideale pour pybind11 et les applications). Le coeur generique reste
# dans include/ (les templates doivent etre visibles a l'instanciation).
add_library(adc_solver STATIC
  src/diocotron_solver.cpp
  src/euler_poisson_solver.cpp)
target_link_libraries(adc_solver PUBLIC adc)
add_library(adc::solver ALIAS adc_solver)

if(ADC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ADC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

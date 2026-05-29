cmake_minimum_required(VERSION 3.20)

project(adc_cpp
  VERSION 0.0.1
  DESCRIPTION "Solveur advection-diffusion-couplage hyperbolique-elliptique sur AMR, cible cluster"
  LANGUAGES CXX C  # C : pour que find_package(MPI) cree MPI::MPI_C (exige par le
                   # hdf5-config parallele de spack)
)

option(ADC_BUILD_TESTS    "Build tests"    ON)
option(ADC_BUILD_EXAMPLES "Build examples" ON)
option(ADC_USE_OPENMP     "Enable the OpenMP dispatch backend" OFF)
option(ADC_USE_KOKKOS     "Enable the Kokkos dispatch backend (CPU/GPU portable)" OFF)
option(ADC_USE_MPI        "Enable the MPI backend (distributed comm seam)" OFF)
option(ADC_USE_HDF5       "Enable the HDF5 DataWriter (parallel output)" OFF)
option(ADC_BUILD_PYTHON   "Build the pybind11 Python bindings (libadc facade)" OFF)
option(ADC_USE_EIGEN      "Build host-side Eigen analysis tools (diocotron theory)" ON)

# Norme C++ : 23 par defaut. Sous Kokkos+nvcc on retombe a 20 (CUDA 12.x ne propose
# pas -std=c++23) ; tout le coeur compile en C++20 (le seul usage >20 est cote hote).
if(ADC_USE_KOKKOS)
  set(ADC_CXX_STD 20)
else()
  set(ADC_CXX_STD 23)
endif()
set(CMAKE_CXX_STANDARD ${ADC_CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

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
    target_compile_definitions(adc_eigen INTERFACE ADC_HAS_EIGEN)  # garde l'en-tete d'analyse
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
target_compile_features(adc INTERFACE cxx_std_${ADC_CXX_STD})

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
  # COMPONENTS CXX : le projet active aussi le langage C (pour MPI::MPI_C), mais on
  # n'utilise OpenMP que cote C++. Sans cette restriction, find_package exigerait
  # OpenMP_C (que les hints Homebrew ne configurent pas) et echouerait sur macOS.
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(adc INTERFACE OpenMP::OpenMP_CXX)
endif()

# Kokkos : backend de dispatch PORTABLE (CPU multi-thread / GPU). Comme MPI, c'est
# une propriete de la BIBLIOTHEQUE, pas un drapeau qu'on rebadge par cible : on
# l'attache a l'interface `adc`, donc TOUT ce qui lie adc (la facade compilee src/,
# les tests, les exemples) herite de ADC_HAS_KOKKOS + Kokkos::kokkos et bascule le
# seam for_each_cell sur l'espace d'execution par defaut (p.ex. Cuda sur GH200).
# La facade libadc se compile alors DIRECTEMENT pour le GPU. Configurer ainsi :
#   cmake -DADC_USE_KOKKOS=ON -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K
set(ADC_HAS_KOKKOS OFF)
if(ADC_USE_KOKKOS)
  find_package(Kokkos REQUIRED)
  target_link_libraries(adc INTERFACE Kokkos::kokkos)
  target_compile_definitions(adc INTERFACE ADC_HAS_KOKKOS)
  set(ADC_HAS_KOKKOS ON)
  message(STATUS "Kokkos found : for_each_cell dispatch backend = Kokkos (${Kokkos_DEVICES})")
  if(ADC_USE_OPENMP)
    message(STATUS "  note: OpenMP aussi actif ; le seam for_each_cell donne la priorite a Kokkos")
  endif()
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
  src/euler_poisson_solver.cpp
  src/two_fluid_ap_solver.cpp)
target_link_libraries(adc_solver PUBLIC adc)
# PIC : la lib statique sera liee dans le module partage pybind11.
set_target_properties(adc_solver PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(adc::solver ALIAS adc_solver)

# Bindings Python (pybind11) de la facade compilee. find_package, sinon FetchContent.
if(ADC_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 2.11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    message(STATUS "pybind11 introuvable ; recuperation via FetchContent")
    include(FetchContent)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6)
    FetchContent_MakeAvailable(pybind11)
  endif()
  add_subdirectory(python)
endif()

if(ADC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ADC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

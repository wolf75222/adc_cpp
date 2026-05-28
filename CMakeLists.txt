cmake_minimum_required(VERSION 3.20)

project(adc_cpp
  VERSION 0.0.1
  DESCRIPTION "Solveur advection-diffusion-couplage hyperbolique-elliptique sur AMR, cible cluster"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ADC_BUILD_TESTS "Build tests" ON)
option(ADC_USE_OPENMP  "Enable the OpenMP dispatch backend" OFF)

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

if(ADC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

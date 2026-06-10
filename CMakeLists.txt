# 3.21 : aligne sur CMakePresets.json (schema v3 = CMake >= 3.21) et debloque
# PROJECT_IS_TOP_LEVEL (garde consommateur ci-dessous). L'env conda sert >= 3.27.
cmake_minimum_required(VERSION 3.21)

project(adc_cpp
  VERSION 0.1.0  # source UNIQUE de la version : propagee au module Python (__version__) ; Doxygen
                 # et Sphinx affichaient deja 0.1.0 (l'ancien 0.0.1 d'ici etait l'incoherence).
  DESCRIPTION "Solveur advection-diffusion-couplage hyperbolique-elliptique sur AMR, cible cluster"
  LANGUAGES CXX C  # C : pour que find_package(MPI) cree MPI::MPI_C (exige par le
                   # hdf5-config parallele de spack)
)

# Backends par VARIABLE D'ENVIRONNEMENT (pattern WarpX/pyAMReX) : permet `ADC_USE_KOKKOS=ON
# Kokkos_ROOT=$CONDA_PREFIX pip install .` sans toucher aux flags CMake (pip ne propage pas de
# -D, mais propage l'environnement ; <Pkg>_ROOT est lu nativement par find_package). La graine
# d'env ne joue QUE si l'option n'est pas posee en cache/-D (CMP0077 NEW : la variable normale
# prime sur option()) -> -DADC_USE_X=... reste prioritaire, rien d'hardcode.
foreach(_adc_opt ADC_BUILD_TESTS ADC_BUILD_PYTHON ADC_USE_KOKKOS ADC_USE_OPENMP ADC_USE_MPI ADC_USE_HDF5)
  if(DEFINED ENV{${_adc_opt}} AND NOT DEFINED ${_adc_opt} AND NOT DEFINED CACHE{${_adc_opt}})
    set(${_adc_opt} "$ENV{${_adc_opt}}")
    message(STATUS "adc: option ${_adc_opt}=$ENV{${_adc_opt}} (heritee de l'environnement)")
  endif()
endforeach()

# Defaut TOP-LEVEL SEULEMENT : un consommateur qui tire adc_cpp via FetchContent/add_subdirectory
# (cf. README "Utiliser le coeur") ne doit PAS compiler les ~140 executables de tests/ ni voir
# enable_testing() pollue. Surchargenable (-DADC_BUILD_TESTS=ON) dans les deux sens.
option(ADC_BUILD_TESTS    "Build tests"    ${PROJECT_IS_TOP_LEVEL})
# Module Python `adc` (pybind11) : c'est LES BINDINGS DE LA LIB, donc ils vivent ici.
# Le coeur reste header-only ; le module compile la facade runtime System (python/system.cpp)
# + les bindings. adc_cases ne contient QUE du code Python qui importe ce module.
option(ADC_BUILD_PYTHON   "Build the pybind11 Python module `adc`" OFF)
# Harnais de PROFILAGE (mesure seule, ZERO optimisation). HORS du build par defaut : le CI ne
# passe JAMAIS ce drapeau, donc bench/ n'est ni configure ni compile en CI. Active a la main par
# -DADC_BUILD_BENCH=ON pour mesurer le cout par phase d'un pas (cf. bench/, docs/PROFILE_RESULTS.md).
option(ADC_BUILD_BENCH    "Build the standalone profiling harness (bench/, OFF by default)" OFF)
# Backends de dispatch on-node : KOKKOS est le chemin recommande (il couvre OpenMP CPU ET
# GPU CUDA/HIP via un seul code portable). ADC_USE_OPENMP (backend OpenMP autonome) est
# DEPRECIE : prefere ADC_USE_KOKKOS (Kokkos avec device OPENMP donne le meme parallelisme CPU
# et ouvre le GPU). On le garde pour compat / comparaison, avec un avertissement.
option(ADC_USE_KOKKOS     "Enable the Kokkos dispatch backend (CPU OpenMP + GPU CUDA/HIP, recommande)" OFF)
option(ADC_USE_OPENMP     "[DEPRECIE -> ADC_USE_KOKKOS] backend OpenMP autonome (CPU seul)" OFF)
option(ADC_USE_MPI        "Enable the MPI backend (distributed comm seam)" OFF)
option(ADC_USE_HDF5       "Enable the HDF5 DataWriter (parallel output)" OFF)

if(ADC_USE_OPENMP AND NOT ADC_USE_KOKKOS)
  message(WARNING "ADC_USE_OPENMP est DEPRECIE : Kokkos (ADC_USE_KOKKOS=ON, device OPENMP) "
                  "couvre le meme parallelisme CPU et ouvre le GPU. Conserve pour compat.")
endif()
if(ADC_USE_OPENMP AND ADC_USE_KOKKOS)
  message(FATAL_ERROR "Choisir UN backend de dispatch : ADC_USE_KOKKOS OU ADC_USE_OPENMP, pas les deux.")
endif()

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

# macOS : si le compilateur retenu est un clang LLVM (Homebrew) et non AppleClang, on INFORME --
# sans forcer quoi que ce soit (le choix reste a l'utilisateur via CC/CXX). Cas reel mesure : un
# `.zshrc` qui met /opt/homebrew/opt/llvm/bin en tete de PATH fait compiler les grosses TU du
# module Python 4-5x plus lentement qu'AppleClang (system.cpp : >1h au lieu de ~8 min), sans
# aucun message expliquant pourquoi. AppleClang s'identifie "AppleClang" ; le LLVM brut "Clang".
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "adc: compilateur = clang LLVM (${CMAKE_CXX_COMPILER}). Sur macOS, AppleClang "
                 "compile les grosses TU du module nettement plus vite ; pour le forcer : "
                 "CXX=/usr/bin/clang++ cmake ... (ou pip install .). Cause frequente : "
                 "/opt/homebrew/opt/llvm/bin en tete de PATH.")
endif()

# ccache : cache de compilation. ZERO impact sur le code genere (meme objets) ; accelere
# fortement les RE-builds (ex. quand seul un .cpp change). Auto-detecte s'il est installe
# (l'env conda le fournit, cf. environment.yml). Inactif si absent, surchargeable
# (-DADC_USE_CCACHE=OFF ou -DCMAKE_CXX_COMPILER_LAUNCHER=...). On l'EVITE sous nvcc_wrapper
# (chemin GPU Kokkos/CUDA) ou ccache peut mal interagir avec le wrapper.
option(ADC_USE_CCACHE "Utiliser ccache comme launcher de compilation s'il est present" ON)
if(ADC_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER MATCHES "nvcc")
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache actif : ${CCACHE_PROGRAM} (cache de compilation ; -DADC_USE_CCACHE=OFF pour desactiver)")
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
  $<INSTALL_INTERFACE:include>  # consommation via find_package(adc) installe (GNUInstallDirs vient plus tard)
)
target_compile_features(adc INTERFACE cxx_std_${ADC_CXX_STD})

# Hints libomp pour AppleClang (qui ne trouve pas libomp seul) : env conda ACTIF (paquet
# llvm-openmp, cf. environment.yml) d'abord, sinon Homebrew. Servent a DEUX consommateurs :
# (1) ADC_USE_OPENMP (find_package(OpenMP) direct) et (2) ADC_USE_KOKKOS quand le Kokkos vise
# a ete compile avec le backend OpenMP -- son KokkosConfig.cmake fait find_dependency(OpenMP
# REQUIRED) et echouerait sur macOS sans ces hints (cas reel : Kokkos OpenMP installe dans
# l'env conda par scripts/kokkos_openmp_conda.sh). Macro idempotente, no-op hors APPLE.
macro(adc_apple_libomp_hints)
  if(APPLE AND NOT OpenMP_CXX_FOUND)
    set(LIBOMP_PREFIX "")
    if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/libomp.dylib")
      set(LIBOMP_PREFIX "$ENV{CONDA_PREFIX}")
    else()
      execute_process(COMMAND brew --prefix libomp
                      OUTPUT_VARIABLE LIBOMP_PREFIX
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
          CACHE STRING "" FORCE)
      set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
      set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib"
          CACHE FILEPATH "" FORCE)
    endif()
  endif()
endmacro()

if(ADC_USE_OPENMP)
  adc_apple_libomp_hints()
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
  # Un Kokkos compile avec le backend OpenMP fait find_dependency(OpenMP REQUIRED) dans son
  # KokkosConfig.cmake : sur macOS/AppleClang il faut les hints libomp AVANT le find_package.
  adc_apple_libomp_hints()
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

# adc_cpp = COEUR de la bibliotheque (generique, header-only). Les modeles
# physiques, les facades compilees (libadc), les bindings Python et les exemples
# vivent dans le depot applicatif separe adc_cases, qui consomme adc::adc via
# FetchContent. Le coeur ne connait aucun modele : aucune dependance coeur->cas.

# enable_testing() des que des tests existent : le coeur (tests/) OU le module Python.
# Sans ca, `ctest --test-dir build-py` avec ADC_BUILD_TESTS=OFF ne voit aucun test.
if(ADC_BUILD_TESTS OR ADC_BUILD_PYTHON)
  enable_testing()
endif()
if(ADC_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Harnais de profilage (mesure seule) : sous-repertoire HORS du build par defaut (OFF). N'ajoute
# AUCUNE cible au build standard ni au CI ; lie adc::adc et herite donc du backend actif (Kokkos /
# MPI) sans find_package redondant.
if(ADC_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# Module Python `adc` (bindings de la lib). Compile la facade runtime System + pybind11.
if(ADC_BUILD_PYTHON)
  include(FetchContent)
  # macOS expose souvent plusieurs Python (framework Python.org, conda, brew). Sans garde,
  # FindPython prend le framework le plus recent comme Interpreter mais un autre pour
  # Development.Module : l'interpreteur ne peut alors pas importer l'extension compilee
  # (suffixe ABI cpython-3XY different). On deprioritise les frameworks pour que les deux
  # concordent. Surchargeable via -DPython_EXECUTABLE=/chemin/python pour pinner.
  if(APPLE)
    set(Python_FIND_FRAMEWORK LAST)
  endif()
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  message(STATUS "adc Python module: interpreteur ${Python_EXECUTABLE} (${Python_VERSION})")
  # pybind11 : on PREFERE celui de l'environnement (conda/pip) plutot que de le retelecharger a
  # chaque configure. L'interpreteur retenu expose son dossier de config CMake via
  # `python -m pybind11 --cmakedir` ; on l'ajoute en tete du chemin de recherche pour que le
  # find_package CONFIG ci-dessous le trouve. Benefice double : pas de FetchContent reseau, et
  # plus de "CMake Deprecation Warning" du CMakeLists amont de pybind11 (son
  # cmake_minimum_required(VERSION 3.5) est deprecie par CMake >= 3.31).
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _adc_pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _adc_pybind11_probe
    ERROR_QUIET)
  if(_adc_pybind11_probe EQUAL 0 AND IS_DIRECTORY "${_adc_pybind11_cmakedir}")
    list(PREPEND CMAKE_PREFIX_PATH "${_adc_pybind11_cmakedir}")
  endif()
  find_package(pybind11 2.11 CONFIG QUIET)
  if(pybind11_FOUND)
    message(STATUS "adc Python module: pybind11 ${pybind11_VERSION} trouve dans l'environnement (pas de FetchContent)")
  else()
    # Repli reseau : aucun pybind11 dans l'environnement. Le CMakeLists de pybind11 v2.13.6 emet un
    # warning de depreciation (cmake_minimum_required 3.5) qui n'est PAS de notre ressort. On le
    # neutralise UNIQUEMENT autour de FetchContent_MakeAvailable, puis on restaure l'etat precedent :
    # les depreciations de notre propre code restent signalees.
    message(STATUS "adc Python module: pybind11 absent de l'environnement -> repli FetchContent (v2.13.6)")
    set(_adc_warn_dep_saved "${CMAKE_WARN_DEPRECATED}")
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6)
    FetchContent_MakeAvailable(pybind11)
    set(CMAKE_WARN_DEPRECATED "${_adc_warn_dep_saved}" CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(python)
endif()

# --- Installation / export du COEUR (find_package(adc)) -------------------------------------
# Rend la lib header-only consommable HORS FetchContent : `cmake --install` depose les en-tetes
# + adcConfig.cmake, et un consommateur fait find_package(adc CONFIG) + adc::adc. Prerequis d'un
# eventuel paquet (conda-forge feedstock, Spack). Defaut : ON seulement en top-level ET hors
# build pip (SKBUILD : la wheel ne doit contenir QUE le paquet Python, pas include/ ni cmake/).
if(PROJECT_IS_TOP_LEVEL AND NOT SKBUILD)
  set(_adc_install_default ON)
else()
  set(_adc_install_default OFF)
endif()
option(ADC_INSTALL "Regles d'installation du coeur header-only (find_package(adc))" ${_adc_install_default})
if(ADC_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(TARGETS adc EXPORT adcTargets)
  install(DIRECTORY include/adc DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT adcTargets NAMESPACE adc:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adc)
  configure_package_config_file(cmake/adcConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/adcConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adc)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/adcConfigVersion.cmake
    COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)  # header-only : version arch-independante
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/adcConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/adcConfigVersion.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/adc)
endif()

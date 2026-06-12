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
foreach(_adc_opt ADC_BUILD_TESTS ADC_BUILD_PYTHON ADC_USE_KOKKOS ADC_USE_MPI ADC_USE_HDF5)
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
# Backend de dispatch on-node : KOKKOS EST LE SEUL backend on-node, et il est OBLIGATOIRE.
# Un meme code source Kokkos couvre les trois cibles on-node via l'espace d'execution choisi
# A L'INSTALLATION DE KOKKOS (pas par un drapeau adc) :
#   - sequentiel       : Kokkos Serial  (Kokkos_ENABLE_SERIAL=ON)
#   - CPU multi-thread : Kokkos OpenMP  (Kokkos_ENABLE_OPENMP=ON)
#   - GPU              : Kokkos Cuda/HIP (Kokkos_ENABLE_CUDA=ON / Kokkos_ENABLE_HIP=ON)
# Il n'y a PLUS de backend OpenMP autonome (ADC_USE_OPENMP), plus de `#pragma omp`, plus de
# boucle hote manuelle comme chemin de production : tout le parallelisme on-node passe par le
# seam for_each_cell -> Kokkos (cf. include/adc/mesh/for_each.hpp).
#
# Kokkos n'a PAS besoin d'etre pre-installe : on utilise une install existante si elle est trouvee
# (-DKokkos_ROOT=..., p.ex. cache CI / module ROMEO), SINON on le RECUPERE + construit a la volee
# (FetchContent). Un simple `cmake -B build` marche donc sur une machine sans Kokkos.
option(ADC_USE_KOKKOS     "Enable Kokkos backend (the only on-node backend, mandatory)" ON)
option(ADC_USE_MPI        "Enable the MPI backend (distributed comm seam)" OFF)
option(ADC_USE_HDF5       "Enable the HDF5 DataWriter (parallel output)" OFF)
# Version de Kokkos recuperee par FetchContent quand aucune install n'est trouvee (release tarball +
# verification SHA256). Surchargeables ensemble (changer la version impose de changer le SHA256, sinon
# la verification d'integrite echoue). Le SHA256 est publie sur la page des releases Kokkos.
set(ADC_KOKKOS_FETCH_VERSION "4.4.01" CACHE STRING "Kokkos release version fetched when no install is found")
set(ADC_KOKKOS_FETCH_SHA256
    "3413f0cb39912128d91424ebd92e8832009e7eeaf6fa8da58e99b0d37860d972"
    CACHE STRING "SHA256 of kokkos-<ADC_KOKKOS_FETCH_VERSION>.tar.gz (empty = skip integrity check)")

# Kokkos-only : configurer sans Kokkos est une ERREUR FATALE (le seam for_each_cell ne compile
# pas sans ADC_HAS_KOKKOS). Le serie passe par Kokkos Serial, pas par une boucle C++ manuelle.
# (Pas besoin d'installer Kokkos a la main : il est recupere via FetchContent si introuvable.)
if(NOT ADC_USE_KOKKOS)
  message(FATAL_ERROR
    "adc_cpp is Kokkos-only: configure with -DADC_USE_KOKKOS=ON. Kokkos est trouve s'il est installe "
    "(-DKokkos_ROOT=...) ou recupere + construit automatiquement (FetchContent). Le backend OpenMP "
    "autonome a ete retire.")
endif()

# Norme C++ : 20. Kokkos etant obligatoire et compile sous nvcc pour la cible Cuda (CUDA 12.x
# ne propose pas -std=c++23), tout le coeur compile en C++20 (le seul usage >20 etait cote hote
# et ne sert plus depuis le retrait du chemin non-Kokkos).
set(ADC_CXX_STD 20)
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
# llvm-openmp) d'abord, sinon Homebrew. Sous Kokkos-only il reste UN consommateur : une install
# Kokkos batie avec Kokkos_ENABLE_OPENMP propage find_dependency(OpenMP REQUIRED) via son
# KokkosConfig.cmake -- sans ces hints, find_package(Kokkos) echouerait sur macOS. Ce N'EST PAS
# le backend OpenMP autonome (retire) : uniquement la plomberie de DECOUVERTE de libomp.
# Macro idempotente, no-op hors APPLE ; invoquee AVANT la resolution de Kokkos ci-dessous.
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

# Resolution de Kokkos (cf. wiki Kokkos "Integrating Kokkos into your CMake project", methode 4 :
# supporter a la fois un Kokkos externe ET embarque). find_package D'ABORD (CONFIG, version mini 4.2,
# QUIET) pour reutiliser une install existante (cache CI, module ROMEO, Homebrew, -DKokkos_ROOT=...).
# A DEFAUT, FetchContent recupere le tarball de release (verifie par SHA256) et le construit dans
# l'arbre de build -> aucune install manuelle requise. La cible on-node se choisit par les options
# Kokkos (Kokkos_ENABLE_SERIAL / _OPENMP / _CUDA / _HIP) -- defaut Serial pour un CPU mono-thread
# universel. Quel que soit le chemin, la cible importee `Kokkos::kokkos` est creee et attachee a adc.
# (On garde la forme explicite if/else plutot que FetchContent FIND_PACKAGE_ARGS, qui exige CMake>=3.24
# alors que ce projet cible 3.20. CMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON force le chemin FetchContent.)
# La version mini 4.2 accepte les versions plus recentes, y compris 5.x (Homebrew 5.1.1, p.ex.).
adc_apple_libomp_hints()
find_package(Kokkos 4.2 QUIET CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Kokkos trouve (install ${Kokkos_VERSION}) : for_each_cell backend = Kokkos (${Kokkos_DEVICES})")
else()
  message(STATUS "Kokkos introuvable -> FetchContent (release ${ADC_KOKKOS_FETCH_VERSION}). "
                 "Cible on-node via -DKokkos_ENABLE_SERIAL/_OPENMP/_CUDA (defaut : Serial).")
  include(FetchContent)
  # Defaut hors-config : au moins Serial, pour un CPU mono-thread fonctionnel sur toute machine.
  # (Si l'utilisateur a deja passe une option Kokkos_ENABLE_*, on la respecte.)
  if(NOT DEFINED Kokkos_ENABLE_SERIAL AND NOT DEFINED Kokkos_ENABLE_OPENMP
     AND NOT DEFINED Kokkos_ENABLE_CUDA AND NOT DEFINED Kokkos_ENABLE_HIP)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Kokkos Serial backend (defaut adc FetchContent)" FORCE)
  endif()
  set(_adc_kokkos_url
      "https://github.com/kokkos/kokkos/releases/download/${ADC_KOKKOS_FETCH_VERSION}/kokkos-${ADC_KOKKOS_FETCH_VERSION}.tar.gz")
  if(ADC_KOKKOS_FETCH_SHA256)
    FetchContent_Declare(Kokkos URL "${_adc_kokkos_url}" URL_HASH SHA256=${ADC_KOKKOS_FETCH_SHA256})
  else()
    FetchContent_Declare(Kokkos URL "${_adc_kokkos_url}")  # pas de SHA256 : verification d'integrite desactivee
  endif()
  FetchContent_MakeAvailable(Kokkos)
  message(STATUS "Kokkos recupere (FetchContent ${ADC_KOKKOS_FETCH_VERSION}) : devices = ${Kokkos_DEVICES}")
endif()
target_link_libraries(adc INTERFACE Kokkos::kokkos)
target_compile_definitions(adc INTERFACE ADC_HAS_KOKKOS)
set(ADC_HAS_KOKKOS ON)
# Pool Ninja taille 1 partage par les cibles a TU geante du module (python/_adc,
# bench/scaling_amr) : serialisation anti-OOM de leurs compiles sous Kokkos, cf.
# python/CMakeLists.txt. Declare ICI et UNE SEULE fois : un nom de pool duplique dans la
# propriete GLOBALE JOB_POOLS rend build.ninja invalide ("ninja: error: duplicate pool").
# Les consommateurs ne posent que JOB_POOL_COMPILE. (tests/ garde son pool distinct
# adc_heavy_tu, declare dans tests/CMakeLists.txt.)
set_property(GLOBAL APPEND PROPERTY JOB_POOLS adc_heavy_module_tu=1)

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

# Outillage QUALITE opt-in (warnings stricts, sanitizers) porte par la cible INTERFACE
# `adc_dev_options`, lie en PRIVATE par les SEULES cibles internes (tests/, module _adc). Inclus ICI :
# apres la resolution de Kokkos, AVANT add_subdirectory(tests)/add_subdirectory(python) qui le lient.
# Tout est OFF par defaut (cible vide) -> aucun impact sur ci.yml, les builds locaux ni `adc_cases`.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AdcDevTooling.cmake)

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

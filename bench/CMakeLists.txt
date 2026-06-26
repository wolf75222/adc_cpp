# Harnais de PROFILAGE (mesure seule). HORS du build par defaut : active par -DPOPS_BUILD_BENCH=ON.
# Le CI ne passe JAMAIS ce drapeau, donc cette cible n'est ni configuree ni compilee en CI (ci-fast,
# MPI, Kokkos restent inchanges). La cible lie pops::pops (interface header-only) : elle herite donc
# AUTOMATIQUEMENT du backend choisi pour la lib (POPS_HAS_KOKKOS + Kokkos::kokkos, POPS_HAS_MPI +
# MPI::MPI_CXX) configure dans le CMakeLists racine -- aucun find_package redondant ici.
#
# Construire (Kokkos-only : -DPOPS_USE_KOKKOS=ON obligatoire ; la cible se choisit a l'install Kokkos) :
#   Kokkos Serial CPU   : cmake -B build-bench -DPOPS_BUILD_BENCH=ON -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT=$K
#   Kokkos OpenMP       : cmake -B build-bench -DPOPS_BUILD_BENCH=ON -DPOPS_USE_KOKKOS=ON \
#                              -DKokkos_ROOT=$K  (Kokkos installe ENABLE_OPENMP)
#   Kokkos Cuda (GH200) : idem avec un Kokkos installe Cuda + -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper
#   MPI + Kokkos Serial : cmake -B build-bench -DPOPS_BUILD_BENCH=ON -DPOPS_USE_MPI=ON -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT=$K
#   MPI + Kokkos Cuda   : -DPOPS_BUILD_BENCH=ON -DPOPS_USE_MPI=ON -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT=$K ...
# Lancer : ./build-bench/bin/profile_step [--n 256 --steps 50 --solver geometric_mg --limiter minmod]
#   sous MPI : mpirun -np 2 ./build-bench/bin/profile_step ...

add_executable(profile_step profile_step.cpp)
target_link_libraries(profile_step PRIVATE pops::pops)

# Front C++ DIRECT (campagne de perf : C++ vs Python briques vs DSL) et driver de SCALING. Memes
# conventions que profile_step (lib header-only, backend herite). SHA/branche injectes a la
# configuration (-DPOPS_BUILD_SHA / -DPOPS_BUILD_BRANCH) : le binaire ne lance pas git (noeud de calcul).
add_executable(frontend_cpp frontend_cpp.cpp)
target_link_libraries(frontend_cpp PRIVATE pops::pops)

add_executable(scaling_step scaling_step.cpp)
target_link_libraries(scaling_step PRIVATE pops::pops)

# Scaling AMR synthetique (4 bulles, AmrSystem + add_compiled_model). AmrSystem est une classe
# COMPILEE (PIMPL) : on lie son implementation python/bindings/amr/amr_system.cpp (comme le test integre
# amrmpi_integrated). Kokkos etant le seul backend, POPS_HAS_KOKKOS est toujours defini et Kokkos est
# toujours initialise (le #ifdef POPS_HAS_KOKKOS reste pour la lisibilite du cycle de vie).
# ADC-335/336 ont SORTI les fabriques de blocs AMR de amr_system.cpp vers les SIX TU du seam :
# amr_block_{exb,isothermal,compressible}.cpp (build_amr_block_*, chemin natif multi-bloc) +
# amr_compiled_{exb,isothermal,compressible}.cpp (build_amr_compiled_*, chemin compile mono-bloc),
# regroupees cote tests dans la lib objet pops_runtime_amr. amr_system.cpp les APPELLE : scaling_amr
# (qui compile amr_system.cpp) DOIT donc lier les six, sinon edition de liens en echec (undefined
# reference to build_amr_block_* / build_amr_compiled_*).
add_executable(scaling_amr scaling_amr.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/amr_system.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/block/base/amr_block_exb.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/block/base/amr_block_isothermal.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/block/compressible/amr_block_compressible.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/block/compressible/amr_block_compressible_rusanov.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/block/compressible/amr_block_compressible_hll.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/block/compressible/amr_block_compressible_hllc.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/block/compressible/amr_block_compressible_roe.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/compiled/base/amr_compiled_exb.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/compiled/base/amr_compiled_isothermal.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/compiled/compressible/amr_compiled_compressible.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/compiled/compressible/amr_compiled_compressible_rusanov.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/compiled/compressible/amr_compiled_compressible_hll.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/compiled/compressible/amr_compiled_compressible_hllc.cpp
               ${CMAKE_SOURCE_DIR}/python/bindings/amr/compiled/compressible/amr_compiled_compressible_roe.cpp)
target_link_libraries(scaling_amr PRIVATE pops::pops)
target_include_directories(scaling_amr PRIVATE ${CMAKE_SOURCE_DIR}/python)
# RAM CI : amr_system.cpp est une des TU geantes du module (cf. python/CMakeLists.txt) ; sous
# Kokkos a -O3 sa compile pese plusieurs GB. Meme pool serie taille 1 que le module (declare
# dans le CMakeLists racine, bloc Kokkos) pour qu'un build bench+python ne puisse pas cumuler
# deux geants en parallele sur un runner CI.
if(POPS_HAS_KOKKOS)
  set_target_properties(scaling_amr PROPERTIES JOB_POOL_COMPILE pops_heavy_module_tu)
endif()

foreach(perf_tgt frontend_cpp scaling_step scaling_amr)
  if(POPS_BUILD_SHA)
    target_compile_definitions(${perf_tgt} PRIVATE POPS_BUILD_SHA="${POPS_BUILD_SHA}")
  endif()
  if(POPS_BUILD_BRANCH)
    target_compile_definitions(${perf_tgt} PRIVATE POPS_BUILD_BRANCH="${POPS_BUILD_BRANCH}")
  endif()
endforeach()

add_executable(profile_transport_mbox profile_transport_mbox.cpp)
target_link_libraries(profile_transport_mbox PRIVATE pops::pops)

# Pas d'add_test : les harnais ne sont PAS des tests (mesure, sortie chronometree non deterministe).
# Le CI ne les verra pas (cibles hors du build par defaut). On les lance a la main / via run_*.sh.

# Harnais de PROFILAGE (mesure seule). HORS du build par defaut : active par -DADC_BUILD_BENCH=ON.
# Le CI ne passe JAMAIS ce drapeau, donc cette cible n'est ni configuree ni compilee en CI (ci-fast,
# MPI, Kokkos restent inchanges). La cible lie adc::adc (interface header-only) : elle herite donc
# AUTOMATIQUEMENT du backend choisi pour la lib (ADC_HAS_KOKKOS + Kokkos::kokkos, ADC_HAS_MPI +
# MPI::MPI_CXX) configure dans le CMakeLists racine -- aucun find_package redondant ici.
#
# Construire :
#   Serie CPU          : cmake -B build-bench -DADC_BUILD_BENCH=ON
#   Kokkos OpenMP       : cmake -B build-bench -DADC_BUILD_BENCH=ON -DADC_USE_KOKKOS=ON \
#                              -DKokkos_ROOT=$K -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper (ou g++)
#   Kokkos Cuda (GH200) : idem avec un Kokkos installe Cuda (nvcc_wrapper)
#   MPI CPU             : cmake -B build-bench -DADC_BUILD_BENCH=ON -DADC_USE_MPI=ON
#   MPI + Kokkos Cuda   : -DADC_BUILD_BENCH=ON -DADC_USE_MPI=ON -DADC_USE_KOKKOS=ON ...
# Lancer : ./build-bench/bin/profile_step [--n 256 --steps 50 --solver geometric_mg --limiter minmod]
#   sous MPI : mpirun -np 2 ./build-bench/bin/profile_step ...

add_executable(profile_step profile_step.cpp)
target_link_libraries(profile_step PRIVATE adc::adc)

# Pas d'add_test : le harnais n'est PAS un test (mesure, sortie chronometree non deterministe). Le
# CI ne le verra pas (cible hors du build par defaut). On le lance a la main / via bench/run_bench.sh.

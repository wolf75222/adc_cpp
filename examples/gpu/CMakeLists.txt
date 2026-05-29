# Build autonome du demo Kokkos (hors du build CPU principal). Necessite un
# Kokkos installe et le compilateur nvcc_wrapper. Sur ROMEO armgpu :
#
#   cmake -S examples/gpu -B build-kokkos \
#     -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper -DKokkos_ROOT=$K-install
#   cmake --build build-kokkos
#   ./build-kokkos/advect_kokkos
#
# (cf. scripts/romeo_gpu.sbatch). Le kernel reutilise les vues Array4 de
# ../../include via Kokkos::parallel_for.
cmake_minimum_required(VERSION 3.20)
project(adc_gpu_kokkos CXX)

find_package(Kokkos REQUIRED)

add_executable(advect_kokkos advect_kokkos.cpp)
target_include_directories(advect_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
# active le backend Kokkos du seam for_each_cell pour cette unite de traduction
target_compile_definitions(advect_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(advect_kokkos PRIVATE Kokkos::kokkos)

# etape 2 : Fab2D en memoire unifiee + for_each_cell GPU (zero copie)
add_executable(advect_fab_kokkos advect_fab_kokkos.cpp)
target_include_directories(advect_fab_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(advect_fab_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(advect_fab_kokkos PRIVATE Kokkos::kokkos)

# etape 3 : le VRAI operateur advance_fab_1c (modele + Rusanov) sur GPU
add_executable(advance_fab_kokkos advance_fab_kokkos.cpp)
target_include_directories(advance_fab_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(advance_fab_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(advance_fab_kokkos PRIVATE Kokkos::kokkos)

# etape A (solveur complet) : assemble_rhs (MUSCL Minmod + SSPRK rhs) sur GPU
add_executable(rhs_kokkos rhs_kokkos.cpp)
target_include_directories(rhs_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(rhs_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(rhs_kokkos PRIVATE Kokkos::kokkos)

# etape B (solveur complet) : briques multigrille (residu + lisseur GS rb) sur GPU
add_executable(poisson_kokkos poisson_kokkos.cpp)
target_include_directories(poisson_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(poisson_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(poisson_kokkos PRIVATE Kokkos::kokkos)

# etape B (suite) : V-cycle multigrille ENTIER (transferts + arith) sur GPU
add_executable(mg_kokkos mg_kokkos.cpp)
target_include_directories(mg_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(mg_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(mg_kokkos PRIVATE Kokkos::kokkos)

# etape D (capstone) : un PAS COUPLE Euler-Poisson entier sur GPU
add_executable(coupled_kokkos coupled_kokkos.cpp)
target_include_directories(coupled_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(coupled_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(coupled_kokkos PRIVATE Kokkos::kokkos)

# etape C : Arena (pool memoire unifiee) -> 0 cudaMallocManaged apres rodage
add_executable(arena_kokkos arena_kokkos.cpp)
target_include_directories(arena_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(arena_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(arena_kokkos PRIVATE Kokkos::kokkos)

# Euler-Poisson (4 variables) : pas couple entier sur GPU, bit-identique au CPU
add_executable(euler_poisson_kokkos euler_poisson_kokkos.cpp)
target_include_directories(euler_poisson_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(euler_poisson_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(euler_poisson_kokkos PRIVATE Kokkos::kokkos)

# Deux-fluides isotherme 2D AP : pas entier sur GPU (transport + lorentz implicite +
# poisson reformule par multigrille), bit-identique au CPU
add_executable(two_fluid_ap_kokkos two_fluid_ap_kokkos.cpp)
target_include_directories(two_fluid_ap_kokkos
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_compile_definitions(two_fluid_ap_kokkos PRIVATE ADC_HAS_KOKKOS)
target_link_libraries(two_fluid_ap_kokkos PRIVATE Kokkos::kokkos)

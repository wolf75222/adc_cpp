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

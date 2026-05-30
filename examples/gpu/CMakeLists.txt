# Demos GPU (backend Kokkos du seam for_each_cell). Le backend n'est PLUS rebadge
# par cible : chaque demo lie simplement `adc` et herite de ADC_HAS_KOKKOS +
# Kokkos::kokkos. Deux modes d'usage :
#
#  - INTEGRE : depuis la racine, `cmake -DADC_USE_KOKKOS=ON ...`. La cible `adc`
#    existe deja et porte Kokkos ; ce fichier est inclus via add_subdirectory et ne
#    fait QUE declarer les executables. Le meme build produit aussi libadc (src/) en
#    version GPU.
#  - AUTONOME : `cmake -S examples/gpu -B build-kokkos -DCMAKE_CXX_COMPILER=$K/bin/nvcc_wrapper
#    -DKokkos_ROOT=$K-install`. Pas de cible `adc` ; on en cree une interface minimale
#    qui trouve Kokkos et expose include/. Pratique pour un demo rapide hors build CPU.

if(NOT TARGET adc)
  cmake_minimum_required(VERSION 3.20)
  project(adc_gpu_kokkos CXX)
  find_package(Kokkos REQUIRED)
  add_library(adc INTERFACE)
  add_library(adc::adc ALIAS adc)
  target_include_directories(adc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  target_compile_features(adc INTERFACE cxx_std_20)
  target_link_libraries(adc INTERFACE Kokkos::kokkos)
  target_compile_definitions(adc INTERFACE ADC_HAS_KOKKOS)  # seul endroit, mode autonome
endif()

# Recette unique : le backend vient de `adc`, jamais d'un drapeau par cible.
function(adc_gpu_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE adc::adc)
endfunction()

adc_gpu_example(advect_kokkos)        # seam for_each_cell sur GPU
adc_gpu_example(advect_fab_kokkos)    # Fab2D memoire unifiee
adc_gpu_example(advance_fab_kokkos)   # operateur advance_fab_1c
adc_gpu_example(rhs_kokkos)           # assemble_rhs MUSCL + SSPRK
adc_gpu_example(poisson_kokkos)       # briques multigrille
adc_gpu_example(mg_kokkos)            # V-cycle entier
adc_gpu_example(coupled_kokkos)       # pas couple Euler-Poisson
adc_gpu_example(arena_kokkos)         # pool memoire unifiee
adc_gpu_example(euler_poisson_kokkos) # Euler-Poisson 4 variables
adc_gpu_example(two_fluid_ap_kokkos)  # deux-fluides 2D AP entier
adc_gpu_example(diocotron_amr_kokkos) # AMR (coupleur + reflux + regrid) sous le seam

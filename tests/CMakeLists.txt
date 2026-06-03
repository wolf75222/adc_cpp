# Tests du COEUR : maillage, AMR, elliptique, integrateurs generiques. Aucun ne
# depend d'un modele physique (ceux-la vivent dans adc_cases). Lient adc::adc seul.

function(adc_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE adc::adc)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- maillage / conteneurs ---
adc_add_test(test_box2d)
adc_add_test(test_fab2d)
adc_add_test(test_box_array)
adc_add_test(test_multifab)
adc_add_test(test_reduce)
adc_add_test(test_fill_boundary)
adc_add_test(test_physical_bc)
adc_add_test(test_geometry)

# --- AMR (primitives, sans modele) ---
adc_add_test(test_refinement)
adc_add_test(test_amr_hierarchy)
adc_add_test(test_cluster)
adc_add_test(test_regrid)
adc_add_test(test_flux_register)
adc_add_test(test_coverage_mask)
adc_add_test(test_amr_diagnostics)
adc_add_test(test_load_balance)

# --- elliptique ---
adc_add_test(test_poisson_smoother)
adc_add_test(test_geometric_mg)
adc_add_test(test_poisson_convergence)
adc_add_test(test_poisson_fft)
adc_add_test(test_elliptic_operator)
adc_add_test(test_elliptic_problem)
adc_add_test(test_elliptic_composite_rhs)
adc_add_test(test_poisson_disc)
adc_add_test(test_solve_robust)
adc_add_test(test_cut_cell)
adc_add_test(test_variable_epsilon)

# --- reconstruction / discretisation / integrateurs generiques ---
adc_add_test(test_spatial_discretisation)
adc_add_test(test_primitive_recon)
adc_add_test(test_system_abstraction)
adc_add_test(test_system_coupler)
adc_add_test(test_two_species_minimal)
adc_add_test(test_coupled_source)
adc_add_test(test_system_two_explicit)
adc_add_test(test_assembler_driver)
adc_add_test(test_amr_system_coupler)
adc_add_test(test_system_hardening)
adc_add_test(test_imex_partial)
adc_add_test(test_imex_transport)
adc_add_test(test_multirate_stride)
adc_add_test(test_cfl_dt)
adc_add_test(test_adaptive_multirate)
adc_add_test(test_user_time_integrator)
adc_add_test(test_diffusion)
adc_add_test(test_amr_diffusion)
adc_add_test(test_amr_spatial_parity)

# --- canal aux extensible (n_aux + load_aux<NComp>) : champ extra B_z lu et peuple ---
adc_add_test(test_aux_extra)
adc_add_test(test_aux_coupler_bz)
adc_add_test(test_aux_system_bz)  # B_z partage multi-blocs (SystemAssembler, aux = max sur les blocs)
adc_add_test(test_aux_composite)  # CompositeModel propage n_aux depuis ses briques (B_z)

adc_add_test(test_weno_convergence)
adc_add_test(test_splitting)
adc_add_test(test_imex_ap)
adc_add_test(test_ap_limit)

# --- dispatch a l'execution (modele type-erased pour brique generee / JIT) ---
adc_add_test(test_dynamic_model)

# --- constructeur de fermetures de bloc instanciable hors System (fondation backend AOT) ---
adc_add_test(test_block_builder)

# --- parite bloc compile AOT (add_compiled_model) == bloc natif (add_block) ---
# Compile le runtime System (python/system.cpp) avec le test : add_compiled_model branche un
# CompositeModel connu a la compilation sur la grille reelle, comme un bloc natif (sans marshaling).
add_executable(test_compiled_model_parity test_compiled_model_parity.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
target_link_libraries(test_compiled_model_parity PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_compiled_model_parity COMMAND test_compiled_model_parity)

# --- runtime System : B_z partage cable de bout en bout (ensure_aux_width + set_magnetic_field) ---
add_executable(test_aux_runtime_bz test_aux_runtime_bz.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
target_link_libraries(test_aux_runtime_bz PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_aux_runtime_bz COMMAND test_aux_runtime_bz)

# --- runtime System : T_e DERIVE d'un bloc fluide (set_electron_temperature_from, 2e champ aux) ---
add_executable(test_aux_te test_aux_te.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
target_link_libraries(test_aux_te PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_aux_te COMMAND test_aux_te)

# --- descripteur de variables : roles physiques (index_of(role)) ---
adc_add_test(test_variable_role)

# --- flux de Roe (Euler) + eigenvalues() ---
adc_add_test(test_roe_flux)

# --- MPI (coeur) : seam comm, reductions, Poisson distribue ---
if(ADC_HAS_MPI)
  function(adc_add_mpi_test name nproc)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE adc::adc)
    add_test(NAME ${name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
  endfunction()
  adc_add_mpi_test(test_mpi_smoke 4)
  adc_add_mpi_test(test_mpi_array_reduce 4)
  adc_add_mpi_test(test_mpi_fillboundary 4)
  adc_add_mpi_test(test_mpi_poisson 4)
  adc_add_mpi_test(test_mpi_redistribute 4)
  adc_add_mpi_test(test_mpi_coupler_inject 4)
  adc_add_mpi_test(test_mpi_fft_distributed 4)

  # --- parite multi-box du chemin compile add_compiled_model (make_block, foncteurs nommes) ---
  # Le residu des fermetures de make_block doit etre invariant au decoupage en boites ET au nombre de
  # rangs (halos intra-rang multi-box + cross-rang MPI). Header-only (adc::adc tire MPI::MPI_CXX), donc
  # CI-landable sans GPU ; la propriete est independante du backend. Un binaire, trois entrees ctest :
  # np=1 couvre le chemin multi-box mono-rang, np=2/4 le chemin MPI cross-rang. Resultat bit-identique.
  add_executable(test_mpi_mbox_parity test_mpi_mbox_parity.cpp)
  target_link_libraries(test_mpi_mbox_parity PRIVATE adc::adc)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_mbox_parity_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_mbox_parity> ${MPIEXEC_POSTFLAGS})
  endforeach()
endif()

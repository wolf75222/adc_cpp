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
adc_add_test(test_elliptic_operator)
adc_add_test(test_elliptic_problem)
adc_add_test(test_poisson_disc)
adc_add_test(test_solve_robust)
adc_add_test(test_cut_cell)

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
adc_add_test(test_weno_convergence)
adc_add_test(test_splitting)
adc_add_test(test_imex_ap)
adc_add_test(test_ap_limit)

# --- dispatch a l'execution (modele type-erased pour brique generee / JIT) ---
adc_add_test(test_dynamic_model)

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
endif()

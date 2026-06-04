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
adc_add_test(test_sync_residence)  # API memoire explicite : sync_host/sync_device (idempotence, no-op sous SharedSpace)
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
adc_add_test(test_patch_range)
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
adc_add_test(test_cut_cell_anisotropic)  # cut-cell + operateur anisotrope (MMS)
adc_add_test(test_variable_epsilon)
adc_add_test(test_screened_poisson)  # operateur Helmholtz/ecrante : div(eps grad phi) - kappa phi
adc_add_test(test_anisotropic_epsilon)  # operateur anisotrope : div(diag(eps_x, eps_y) grad phi)
adc_add_test(test_cut_cell_anisotropic_multibox)  # cut-cell + aniso sur domaine MULTI-BOX (mono-rang)

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
adc_add_test(test_amr_aux_bz)     # lecture + allocation aux width-aware sur les chemins AMR (B_z lu)
adc_add_test(test_amr_system_bz_pop)  # B_z peuple PAR NIVEAU dans AmrSystemCoupler (echantillonne au niveau)
adc_add_test(test_amr_system_bz_multibox)  # B_z PAR NIVEAU multi-box (grossier 2x2, fin multi-patch) mono-rang
adc_add_test(test_aux_single_source)  # source unique ADC_AUX_FIELDS : load_aux + marshaling generes, tous champs couverts (anti-#51)

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

# --- parite bloc compile AOT (add_compiled_model) == bloc natif (add_block) COTE AMR ---
# Compile le runtime AmrSystem (python/amr_system.cpp) avec le test : add_compiled_model branche un
# CompositeModel connu a la compilation sur la hierarchie AMR reelle (AmrCouplerMP + reflux + regrid),
# comme add_block (sans dispatch ModelSpec). Densite grossiere et masse bit-identiques apres N pas.
add_executable(test_amr_compiled_model test_amr_compiled_model.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
target_link_libraries(test_amr_compiled_model PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_amr_compiled_model COMMAND test_amr_compiled_model)

# --- contrat mono-bloc/explicite de la facade AmrSystem : refus explicite des params non cables ---
# Compile le runtime AmrSystem (python/amr_system.cpp) avec le test : verrouille que set_poisson
# refuse solver/rhs hors du domaine cable (GeometricMG + somme des briques elliptiques) et qu'add_block
# refuse time != explicit / recon hors domaine / substeps < 1, au lieu de les ignorer en silence.
add_executable(test_amr_system_contract test_amr_system_contract.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
target_link_libraries(test_amr_system_contract PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_amr_system_contract COMMAND test_amr_system_contract)

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

  # --- cut-cell multi-box DISTRIBUE : verrou de non-regression du bug "cut-cell multi-box MPI" ---
  # Enchaine des solves cut-cell sur un domaine multi-box (16 boites) dont la hierarchie degenerait
  # au plus grossier (average_down lisait hors bornes d'un fab fin 1x1 -> divergence intermittente
  # sous MPI, meme a np=1). Exige finitude + determinisme intra-rang + invariance vs mono-box. np=1
  # couvre le chemin qui flakait ; np=2/4 le multi-box reparti (halos cut-cell cross-rang).
  add_executable(test_mpi_cutcell_multibox test_mpi_cutcell_multibox.cpp)
  target_link_libraries(test_mpi_cutcell_multibox PRIVATE adc::adc)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_cutcell_multibox_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_cutcell_multibox> ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- VALIDATION INTEGREE AmrSystem + MPI + (GPU) : hierarchie AMR (regrid + reflux + Poisson) +
  # modele COMPILE (add_compiled_model) DISTRIBUE sur n_ranks(). Le grossier replique impose un
  # resultat invariant au nombre de rangs : on verifie la consistance cross-rang (spread == 0) et le
  # script ROMEO (python/tests/gpu/amrmpi_romeo_build.sh) diff les sorties np=1/2/4 (np=1 = oracle
  # mono-GPU). Compile le runtime AmrSystem comme test_amr_compiled_model. Independant du backend :
  # vert sous Kokkos Serial (CI, CPU) et Kokkos Cuda (GH200). Un binaire, trois entrees ctest.
  # --- B_z AMR PAR NIVEAU multi-box DISTRIBUE : le meme binaire que le test serie ci-dessus
  # (test_amr_system_bz_multibox, deja construit par adc_add_test) est rejoue sous MPI a np=2/4. Le
  # grossier multi-box (2x2) et le niveau fin multi-patch sont REPARTIS sur les rangs ; B_z etant une
  # fonction PURE de la position, sa valeur par boite est invariante au nombre de rangs. Chaque rang
  # verifie ses boites locales (valides + ghosts) contre l'analytique et all_reduce_sum les echecs :
  # np=2/4 valident le chemin multi-box CROSS-RANG (fill_boundary distant, grossier de-replique). np=1
  # est deja couvert par l'entree serie. Aucun nouveau binaire (reutilise la cible existante).
  foreach(np 2 4)
    add_test(NAME test_amr_system_bz_multibox_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_system_bz_multibox>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  add_executable(test_mpi_amr_compiled_parity test_mpi_amr_compiled_parity.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
  target_link_libraries(test_mpi_amr_compiled_parity PRIVATE adc::adc ${CMAKE_DL_LIBS})
  foreach(np 1 2 4)
    add_test(NAME test_mpi_amr_compiled_parity_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_amr_compiled_parity>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- STRONG-SCALING AMR : grossier REPARTI (distribute_coarse=true) cable dans AmrSystem. Compare
  # le grossier multi-box reparti a l'oracle replique (meme binaire) : meme physique a l'arrondi,
  # cmax bit-identique cross-rang, masse conservee, MG convergent. Compile le runtime AmrSystem comme
  # test_mpi_amr_compiled_parity. Le script ROMEO diff cmax entre np=1/2/4 (bit-identique attendu).
  add_executable(test_mpi_amr_distributed_coarse test_mpi_amr_distributed_coarse.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
  target_link_libraries(test_mpi_amr_distributed_coarse PRIVATE adc::adc ${CMAKE_DL_LIBS})
  foreach(np 1 2 4)
    add_test(NAME test_mpi_amr_distributed_coarse_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_amr_distributed_coarse>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()
endif()

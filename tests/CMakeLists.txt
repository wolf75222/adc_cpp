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
adc_add_test(test_cf_interface)  # SubcyclingSchedule + CoarseFineInterface (couverture + routage reflux)
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
adc_add_test(test_full_tensor_operator)  # operateur tenseur PLEIN : -div(A grad phi)+kappa phi, A=[[Axx,Axy],[Ayx,Ayy]] (termes croises)

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

# --- Proto Phase-0 : mesure decisive pour le chantier "grille polaire diocotron" ---
# Quantifie la diffusion du gradient radial d'un anneau en rotation azimutale :
#   bras 1 (Cartesien) = WENO5+Rusanov+SSPRK3 sur grille (x,y) reguliere,
#   bras 2 (Polaire)   = WENO5+Rusanov 1D en theta sur grille (r,theta).
# Verdict : SUCCESS si perte polaire <= 2% ET rapport C/P >= 5 (ordre de grandeur).
# ABORT si le rapport est marginal (< 5) : le verrou n'est alors pas la geometrie.
adc_add_test(test_polar_ring_advection)

# --- Phase 1 grille polaire : operateur de transport polaire additif (assemble_rhs_polar) ---
# MMS de la divergence polaire (1/r) d_r(r F_r) + (1/r) d_theta(F_theta) sur un anneau global
# (r, theta), champ aux PRESCRIT (transport seul, AUCUN Poisson) : convergence a l'ordre du schema
# (WENO5-Z >= 3.5, minmod >= 1.5) + CONSERVATION FV de la masse a ~machine (champ azimutal pur).
# Verifie que la metrique polaire (facteur r aux faces radiales, 1/r en cellule) est portee
# correctement par l'operateur additif, sans toucher au chemin cartesien (bit-identique par defaut).
adc_add_test(test_polar_transport_mms)

# --- dispatch a l'execution (modele type-erased pour brique generee / JIT) ---
adc_add_test(test_dynamic_model)

# --- constructeur de fermetures de bloc instanciable hors System (fondation backend AOT) ---
adc_add_test(test_block_builder)

# --- WENO5-Z (ordre 5, 3 ghosts) + SSPRK3 cables via make_block/build_block : parite schema, routage
#     du tag temporel, no-default-change (SSPRK2 bit-identique), convergence > O2 ---
adc_add_test(test_weno5_ssprk3)

# --- parite bloc compile AOT (add_compiled_model) == bloc natif (add_block) ---
# Compile le runtime System (python/system.cpp) avec le test : add_compiled_model branche un
# CompositeModel connu a la compilation sur la grille reelle, comme un bloc natif (sans marshaling).
add_executable(test_compiled_model_parity test_compiled_model_parity.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
target_link_libraries(test_compiled_model_parity PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_compiled_model_parity COMMAND test_compiled_model_parity)

# --- WENO5 sur le chemin compile natif (add_compiled_model) == bloc natif (add_block) ---
# Verrou de l'unification ghost : add_compiled_model alloue le bloc a block_n_ghost(limiter) (3 pour
# weno5) APRES install_block, comme add_block. Parite STRICTE weno5 + no-default-change none/minmod
# (allocation <= 2 ghosts inchangee). Compile le runtime System comme test_compiled_model_parity.
add_executable(test_weno5_compiled_model test_weno5_compiled_model.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
target_link_libraries(test_weno5_compiled_model PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_weno5_compiled_model COMMAND test_weno5_compiled_model)

# --- parite bloc compile AOT (add_compiled_model) == bloc natif (add_block) COTE AMR ---
# Compile le runtime AmrSystem (python/amr_system.cpp) avec le test : add_compiled_model branche un
# CompositeModel connu a la compilation sur la hierarchie AMR reelle (AmrCouplerMP + reflux + regrid),
# comme add_block (sans dispatch ModelSpec). Densite grossiere et masse bit-identiques apres N pas.
add_executable(test_amr_compiled_model test_amr_compiled_model.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
target_link_libraries(test_amr_compiled_model PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_amr_compiled_model COMMAND test_amr_compiled_model)

# --- AmrSystem::potential() : phi du niveau grossier expose, pendant de System::potential() ---
# Compile les runtimes AmrSystem ET System (python/amr_system.cpp + python/system.cpp) avec le test :
# sans raffinement, AmrSystem resout le MEME Poisson discret que System(geometric_mg) sur le grossier
# mono-box -> potential() : forme n*n, valeurs finies, champ non trivial, moyenne ~0 (source neutre),
# et PARITE avec System::potential() a la tolerance MG pres. C'est l'observable phi qu'echantillonne le
# benchmark diocotron (FFT azimutale sur un cercle median) cote AMR.
add_executable(test_amr_potential test_amr_potential.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
target_link_libraries(test_amr_potential PRIVATE adc::adc ${CMAKE_DL_LIBS})
add_test(NAME test_amr_potential COMMAND test_amr_potential)

# --- chemin "production" NATIF cote AMR : AmrSystem::add_native_block (DSL Phase D) ---
# Compile A L'EXECUTION un loader .so (la source qu'emet dsl.emit_cpp_native_loader(target="amr_system"),
# ecrite ici pour un test C++ autonome) qui inline add_compiled_model(AmrSystem&), le branche via
# AmrSystem::add_native_block, et compare a add_compiled_model(AmrSystem&) EN DIRECT (memes types/schema)
# -> densite grossiere bit-identique (dmax==0) + rejet d'une cle d'ABI falsifiee. Compile amr_system.cpp
# (AmrSystem::add_native_block) avec le test. Injecte compilateur / include / norme / tmpdir : le loader
# est rebati avec la MEME norme C++ que ce binaire (cle d'ABI concordante, garde-fou explicite sinon).
# Le test s'auto-saute (exit 0) si le compilateur du build ne sait pas compiler le loader.
add_executable(test_amr_native_loader test_amr_native_loader.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
target_link_libraries(test_amr_native_loader PRIVATE adc::adc ${CMAKE_DL_LIBS})
# ELF (Linux) : un executable n'exporte AUCUN symbole dans sa table dynamique sauf -rdynamic. Le loader
# .so est dlopen-e en RTLD_GLOBAL et resout ses indefinis (adc::AmrSystem::set_compiled_block, exportee
# ADC_EXPORT = visibilite defaut) contre l'image deja chargee : ICI le binaire de test lui-meme (auto-
# promu RTLD_NOLOAD|RTLD_GLOBAL via dladdr). Sans ENABLE_EXPORTS (-rdynamic), set_compiled_block n'est
# pas dans la .dynsym de l'exe -> dlopen echoue (undefined symbol). MEME role que le module _adc .so
# cote System (un .so a SES symboles defaut dans la .dynsym ; un exe non). Sans effet sur macOS (le
# loader y resout par -undefined dynamic_lookup), inoffensif.
set_target_properties(test_amr_native_loader PROPERTIES ENABLE_EXPORTS ON)
target_compile_definitions(test_amr_native_loader PRIVATE
  ADC_TEST_CXX="${CMAKE_CXX_COMPILER}"
  ADC_TEST_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}/../include"
  ADC_TEST_CXX_STD="c++${CMAKE_CXX_STANDARD}"
  ADC_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME test_amr_native_loader COMMAND test_amr_native_loader)

# --- WENO5 sur le chemin "production" NATIF cote AMR (Plan Ideal etape 5 / DSL Phase D) ---
# Pendant raffine de test_weno5_compiled_model.cpp : prouve que WENO5-Z (3 ghosts) tourne la MEME
# hierarchie AMR, BIT-IDENTIQUE, sur add_compiled_model == add_block (direct, tous backends) ET
# add_native_block(loader .so) == add_compiled_model (chemin .so, auto-saute sous Kokkos). Verrou
# ghost : les niveaux du coupleur sont alloues a Limiter::n_ghost (3 pour weno5) et le regrid herite
# n_grow() -- MEME mecanisme que System (PR #22/#88), pas de logique dupliquee. + no-default-change
# none/minmod. Compile amr_system.cpp (add_native_block) avec le test ; memes injections que le loader.
add_executable(test_amr_weno5_native test_amr_weno5_native.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
target_link_libraries(test_amr_weno5_native PRIVATE adc::adc ${CMAKE_DL_LIBS})
# ENABLE_EXPORTS (-rdynamic) : le loader .so resout set_compiled_block (ADC_EXPORT) contre l'exe de
# test auto-promu RTLD_GLOBAL (cf. test_amr_native_loader). Inoffensif sur macOS.
set_target_properties(test_amr_weno5_native PROPERTIES ENABLE_EXPORTS ON)
target_compile_definitions(test_amr_weno5_native PRIVATE
  ADC_TEST_CXX="${CMAKE_CXX_COMPILER}"
  ADC_TEST_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}/../include"
  ADC_TEST_CXX_STD="c++${CMAKE_CXX_STANDARD}"
  ADC_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME test_amr_weno5_native COMMAND test_amr_weno5_native)

# --- HLLC/Roe + recon primitive sur le chemin "production" NATIF cote AMR (Gap 1 parite) ---
# Pendant de test_amr_weno5_native : prouve que riemann=hllc/roe et recon=primitive sont cables
# sur la hierarchie AMR (AmrCouplerMP + reflux + regrid) et produisent add_compiled_model ==
# add_block BIT-IDENTIQUE (dmax==0) pour les 4 combos (hllc/roe) x (conservative/primitive).
# Verifie aussi que hllc/roe different de rusanov (flux actif, non silencieux). Compile amr_system.cpp
# comme test_amr_weno5_native ; memes injections compilateur / include / norme / tmpdir.
add_executable(test_amr_riemann_native test_amr_riemann_native.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp)
target_link_libraries(test_amr_riemann_native PRIVATE adc::adc ${CMAKE_DL_LIBS})
set_target_properties(test_amr_riemann_native PROPERTIES ENABLE_EXPORTS ON)
target_compile_definitions(test_amr_riemann_native PRIVATE
  ADC_TEST_CXX="${CMAKE_CXX_COMPILER}"
  ADC_TEST_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}/../include"
  ADC_TEST_CXX_STD="c++${CMAKE_CXX_STANDARD}"
  ADC_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME test_amr_riemann_native COMMAND test_amr_riemann_native)

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

  # --- parite multi-box / MPI d'un composite HYBRIDE (brique source style DSL + transport/elliptique
  # natifs) sur le chemin add_compiled_model (make_block). Pendant C++/MPI de test_dsl_hybrid_amr/_coupling :
  # seul le TYPE de brique change vs test_mpi_mbox_parity. aux periodique analytique -> la source hybride
  # contribue au residu ET reste decomposition-invariante. np=1 multi-box mono-rang, np=2/4 cross-rang MPI.
  add_executable(test_mpi_hybrid_mbox_parity test_mpi_hybrid_mbox_parity.cpp)
  target_link_libraries(test_mpi_hybrid_mbox_parity PRIVATE adc::adc)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_hybrid_mbox_parity_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_hybrid_mbox_parity> ${MPIEXEC_POSTFLAGS})
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

  # --- runtime System + Poisson + solve_fields() DISTRIBUE : verrou de non-regression du bug HOTE
  # fab(0) (#98). System repartit UNE box en round-robin, donc a np>1 un seul rang la possede ; les
  # autres ont local_size()==0. Avant le fix, le post-traitement par cellule de solve_fields()
  # (derivation phi/grad, apply_te, apply_epsilon*, apply_reaction) appelait fab(0) sans garde et
  # crashait cote hote sur les rangs sans box locale. Le solve elliptique reste COLLECTIF (tous les
  # rangs). np=1 couvre le chemin proprietaire (bit-identique au serie) ; np=2/4 le chemin distribue
  # ou des rangs sont vides. Compile le runtime System comme test_aux_te.
  add_executable(test_mpi_system_solve_fields test_mpi_system_solve_fields.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
  target_link_libraries(test_mpi_system_solve_fields PRIVATE adc::adc ${CMAKE_DL_LIBS})
  foreach(np 1 2 4)
    add_test(NAME test_mpi_system_solve_fields_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_system_solve_fields>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- solveur Poisson FFT sous MPI : verrou de non-regression du SIGSEGV "fft + n_ranks>1" (#93).
  # System repartit UNE box en round-robin, donc a np>1 des rangs ont local_size()==0 ;
  # PoissonFFTSolver::solve() y dereferencait fab(0) inexistant (l'assert mono-rang disparaissait en
  # Release). Le fix REFUSE "fft" sous MPI avec une erreur claire levee sur TOUS les rangs (collectif,
  # pas d'interblocage). np=1 : "fft" marche et colle a geometric_mg (FP). np=2/4 : solve_fields() leve
  # sur chaque rang, capture, all_reduce(threw)==np. Compile le runtime System comme
  # test_mpi_system_solve_fields. Un binaire, trois entrees ctest.
  add_executable(test_mpi_system_fft test_mpi_system_fft.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp)
  target_link_libraries(test_mpi_system_fft PRIVATE adc::adc ${CMAKE_DL_LIBS})
  foreach(np 1 2 4)
    add_test(NAME test_mpi_system_fft_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_system_fft>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()
endif()

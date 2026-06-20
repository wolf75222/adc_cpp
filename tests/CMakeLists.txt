# Tests du COEUR : maillage, AMR, elliptique, integrateurs generiques. Aucun ne
# depend d'un modele physique (ceux-la vivent dans adc_cases). Lient adc::adc seul.

# BUILD CI RAPIDE (-O0 sur les TESTS). Sous le backend Kokkos, l'instanciation + l'inlining a -O3 des
# centaines de foncteurs device (assemble_rhs / make_block / for_each_cell) dans les ~242 cibles de test
# rend le build du gate tres long (plusieurs dizaines de minutes) et gourmand en RAM. Les tests sont des
# CONTROLES DE CORRECTION sur des grilles minuscules : leur niveau d'optimisation n'a aucune importance.
# Avec ADC_TESTS_FAST_O0=ON (pose par la CI), on compile TOUTES les cibles de ce dossier a -O0 -> build
# bien plus rapide et borne en memoire, codegen identique entre cibles (parite preservee). Le code de
# PRODUCTION (lib adc::adc dans le module _adc, bench/) reste a -O3 : ce drapeau ne touche QUE tests/.
# GCC/Clang et configs optimisees uniquement (Debug est deja -O0).
option(ADC_TESTS_FAST_O0 "Compile les executables de test a -O0 (build CI rapide ; tests = correction)" OFF)
if(ADC_TESTS_FAST_O0)
  add_compile_options(
    $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
endif()

function(adc_add_test name)
  add_executable(${name} ${name}.cpp)
  # adc_dev_options : cible INTERFACE vide par defaut (cf. cmake/AdcDevTooling.cmake) ; ne porte des
  # flags QUE si ADC_ENABLE_WARNINGS / ADC_ENABLE_SANITIZERS sont ON (workflow quality.yml). PRIVATE :
  # n'affecte que la compilation/edition de liens de ce test, jamais l'API publique adc::adc.
  target_link_libraries(${name} PRIVATE adc::adc adc_dev_options)
  add_test(NAME ${name} COMMAND ${name})
  # LABELS : selection par sous-ensemble (`ctest -L core`). TIMEOUT : un test qui hang (classe
  # deadlock deja rencontree cote MPI) echoue de maniere ciblee au lieu de bloquer le runner.
  set_tests_properties(${name} PROPERTIES LABELS "core" TIMEOUT 600)
endfunction()

# Coupe l'optimisation (-O0) sur une cible de test qui compile python/system.cpp (le runtime System avec le
# dispatch ModelSpec). make_block y instancie, POUR CHAQUE CompositeModel resolu (~33 : transport x source x
# elliptique), TOUTES les familles de flux qui compilent -- rusanov/hll/hllc/roe x 4 limiteurs, chacune un
# build_block d'une dizaine de foncteurs device. Sous le backend Kokkos a -O3, l'inlining de ces centaines
# d'instanciations fait exploser la RSS de cc1plus au-dela des ~7 Go du runner GitHub (build a --parallel 2 :
# DEUX TU lourdes concurrentes dont la somme franchit le plafond) ; la TU stalle sans diagnostic puis
# l'OOM-reclaim tue le job (exit 143). L'ajout de HLL (#239) a fait franchir le seuil aux cibles qui compilent
# system.cpp. -O0 coupe l'optimisation (et son inlining) -- le poste memoire dominant : il effondre la RSS de
# cc1plus ET supprime le thrashing memoire qui ralentissait deja le build. -O1 a ete essaye d'abord mais s'est
# revele MARGINAL (PR #259 : le job Kokkos OpenMP passait, mais Serial -- runner plus serre -- restait
# OOM-reclaime au meme endroit) ; -O0 donne la marge decisive. SANS changer les resultats : ces tests
# comparent deux chemins (AOT add_compiled_model vs natif add_block) PORTES PAR LA MEME CIBLE, donc les deux
# basculent a -O0 ensemble -> codegen identique -> parite bit-a-bit preservee. Le code de PRODUCTION (lib
# adc::adc, module Python _adc) reste a -O3. GCC/Clang uniquement et configs optimisees seulement (Debug deja
# a -O0). Ces tests sont des controles de CORRECTION sur des grilles minuscules (n=48) : le runtime a -O0 est
# negligeable.
# Pool de compilation Ninja de taille 1 : les TU lourdes capees ne se compilent JAMAIS a deux en meme temps.
# C'est la garantie STRUCTURELLE contre l'OOM (la cause racine = somme de DEUX TU lourdes concurrentes sous
# --parallel 2). Avec -O0 + ce pool, le pic memoire est borne par UNE seule TU lourde -O0 (~qq Go) + au plus
# une TU normale non capee de l'autre slot ninja : largement sous les ~7 Go du runner, quel que soit
# l'ordonnancement. Honore par le generateur Ninja (la CI) ; ignore sans effet par Make. APPEND sur la
# propriete GLOBALE JOB_POOLS (le CMakeLists racine y declare aussi adc_heavy_module_tu, nom DISTINCT :
# chaque pool ne doit etre declare qu'UNE fois, sinon "ninja: error: duplicate pool").
set_property(GLOBAL APPEND PROPERTY JOB_POOLS adc_heavy_tu=${ADC_HEAVY_TU_POOL})  # profondeur partagee (ADC-335, defaut 1 ; racine CMakeLists.txt)

function(adc_cap_opt_for_kokkos_ram target)
  target_compile_options(${target} PRIVATE
    $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
  set_target_properties(${target} PROPERTIES JOB_POOL_COMPILE adc_heavy_tu)
endfunction()

# --- ADC-336 (P0-A) : les deux TU lourdes du runtime compilees UNE SEULE FOIS par config -----------
# Avant, ~23 executables de test re-listaient python/system.cpp et/ou python/amr_system.cpp en source
# (chacune une compilation cc1plus de plusieurs Go / centaines de secondes -- la meme TU recompilee
# jusqu'a 20x dans un build de tests). On les compile UNE fois dans deux bibliotheques OBJECT, splicees
# dans chaque executable consommateur EXACTEMENT comme si la .cpp y etait encore listee : memes objets,
# donc tous les symboles (y compris ADC_EXPORT) presents, et -rdynamic les exporte toujours.
# Parite bit-a-bit : la cible de reference est ce que ces tests compilaient deja -- adc::adc seul,
# AUCUN adc_dev_options, AUCUN define ADC_* (les ADC_TEST_* des loaders natifs sont PRIVATE a leur .cpp
# et inutilises par la TU lourde) -- donc on n'ajoute rien.
# -O0 PUBLIC : cape la TU lourde ET se propage a la .cpp de chaque test consommateur, pour que le chemin
# add_compiled_model (instancie dans la .cpp du test) reste au MEME -O que le chemin add_block (dans cet
# objet partage). L'ancien cap par cible couvrait les deux a -O0 ; le preserver garde l'ensemble
# (TU, flags) identique a aujourd'hui et evite toute derive de bits par contraction FMA a -O3 (GCC).
# Le pool ninja adc_heavy_tu (taille 1, declare ci-dessus) reste PRIVATE : seule la TU lourde se
# serialise (anti-OOM), pas les petites .cpp de test.
# _adc (python/CMakeLists.txt) n'est PAS touche : il compile deja chaque TU une fois avec ses propres
# flags de production (ADC_HEADER_SIG/visibilite/...), et AUCUN preset ne co-construit tests+python
# (ADC_BUILD_TESTS et ADC_BUILD_PYTHON sont mutuellement exclusifs dans CMakePresets.json) -- cette
# bibliotheque est donc strictement tests-only et ne peut pas etre partagee avec _adc.
add_library(adc_runtime_system OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../python/system.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_exb.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_isothermal_rusanov.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_isothermal_hll.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_compressible_rusanov.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_compressible_hll.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_compressible_hllc.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_compressible_roe.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../python/system_polar.cpp)
# ADC-335: the AMR transport-seam TUs (multi-block build_amr_block_<t> + single-block
# build_amr_compiled_<t>). amr_system.cpp CALLS them, so they must be compiled/linked wherever
# amr_system.cpp is -- the adc_runtime_amr object lib AND the ENABLE_EXPORTS native-loader tests that
# re-list amr_system.cpp directly (below). One list, reused, so none is forgotten.
set(ADC_AMR_SEAM
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_block_exb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_block_isothermal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_block_compressible.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_block_compressible_rusanov.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_block_compressible_hll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_block_compressible_hllc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_block_compressible_roe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_compiled_exb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_compiled_isothermal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_compiled_compressible.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_compiled_compressible_rusanov.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_compiled_compressible_hll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_compiled_compressible_hllc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_compiled_compressible_roe.cpp)
add_library(adc_runtime_amr    OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp ${ADC_AMR_SEAM})
foreach(_rt adc_runtime_system adc_runtime_amr)
  target_link_libraries(${_rt} PUBLIC adc::adc ${CMAKE_DL_LIBS})
  target_compile_options(${_rt} PUBLIC
    $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
  set_target_properties(${_rt} PROPERTIES JOB_POOL_COMPILE adc_heavy_tu)
endforeach()

# --- maillage / conteneurs ---
adc_add_test(test_box2d)
adc_add_test(test_fab2d)
adc_add_test(test_box_array)
adc_add_test(test_multifab)
adc_add_test(test_sync_residence)  # API memoire explicite : sync_host/sync_device (idempotence, no-op sous SharedSpace)
adc_add_test(test_reduce)
adc_add_test(test_dense_eig)  # numerics/dense_eig.hpp : extremes du spectre d'un petit bloc dense (HLL exact)
adc_add_test(test_fill_boundary)
adc_add_test(test_fill_boundary_cache)
adc_add_test(test_physical_bc)
adc_add_test(test_geometry)

# --- AMR (primitives, sans modele) ---
adc_add_test(test_refinement)
adc_add_test(test_ref_ratio)
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
adc_add_test(test_poisson_smoother_cache)  # ADC-262 : RB-GS reuses the cached halo schedule (no per-sweep re-enumeration)
adc_add_test(test_geometric_mg)
adc_add_test(test_poisson_convergence)
adc_add_test(test_poisson_fft)
adc_add_test(test_elliptic_operator)
adc_add_test(test_elliptic_problem)
adc_add_test(test_elliptic_composite_rhs)
# Solveur Poisson COMPOSITE FAC 2 niveaux (coarse + 1 patch fin, ghosts C-F bilineaires + correction de
# flux C-F two-way) : test MMS montrant que le patch fin REDUIT l'erreur elliptique vs coarse-only (le
# verrou de fidelite AMR -- les patchs raffinent le couplage elliptique, pas seulement le transport).
adc_add_test(test_composite_fac_poisson)
# Composite FAC a COEFFICIENT VARIABLE div(eps grad phi)=f (operateur condense Schur B_z=0, eps=1+c rho) :
# MMS montrant que le patch fin reduit l'erreur (phi + grad phi) avec un coefficient variable.
adc_add_test(test_composite_fac_variable_eps)
# Composite FAC a TENSEUR PLEIN div(A grad phi)=f, A=diag(eps,eps)+[[0,a_xy],[a_yx,0]], a_yx=-a_xy
# (operateur condense Schur B_z!=0, non auto-adjoint) : MMS, le patch fin reduit l'erreur (phi + grad phi).
adc_add_test(test_composite_fac_tensor)
# Poisson AMR COMPOSITE branche dans AmrCouplerMP (set_composite_poisson) : le patch fin pose dans l'aux
# un grad phi resolu a la finesse (vs grad grossier injecte d'Option A) -> verrou de fidelite AMR.
adc_add_test(test_amr_composite_poisson)
adc_add_test(test_poisson_disc)
adc_add_test(test_solve_robust)
adc_add_test(test_cut_cell)
adc_add_test(test_cut_fraction)  # primitive PARTAGEE de fraction de coupe : unite analytique + bit-identite vs formule inline
adc_add_test(test_cut_fraction_mg_identity)  # refactor cut_fraction : coef GeometricMG byte-identique a l'ancienne formule inline
adc_add_test(test_cut_cell_anisotropic)  # cut-cell + operateur anisotrope (MMS)
adc_add_test(test_variable_epsilon)
adc_add_test(test_screened_poisson)  # operateur Helmholtz/ecrante : div(eps grad phi) - kappa phi
adc_add_test(test_anisotropic_epsilon)  # operateur anisotrope : div(diag(eps_x, eps_y) grad phi)
adc_add_test(test_cut_cell_anisotropic_multibox)  # cut-cell + aniso sur domaine MULTI-BOX (mono-rang)
adc_add_test(test_full_tensor_operator)  # operateur tenseur PLEIN : -div(A grad phi)+kappa phi, A=[[Axx,Axy],[Ayx,Ayy]] (termes croises)
# Phase 2a grille polaire : solveur de Poisson POLAIRE direct (PolarPoissonSolver) sur un anneau
# (r, theta), r_min > 0 (AUCUNE singularite r=0). Methode FFT-en-theta (decouple les modes azimutaux,
# valeur propre spectrale -k^2) + une tridiagonale (Thomas) en r par mode -> directe, robuste (pas de
# convergence iterative MG en polaire, le verrou signale par le scoping). MMS : convergence radiale
# O(2) (Dirichlet ET Neumann homogene), spectral en theta (mode pur exact des nth modeste), residu
# discret ~arrondi, jauge mode 0 (double Neumann). C++ SEUL, PAS branche dans System.step (Phase 2b).
# Additif : aucun solveur elliptique cartesien (geometric_mg, PoissonFFTSolver) touche.
adc_add_test(test_polar_poisson_mms)
# Voie A etape 2a : operateur elliptique POLAIRE TENSORIEL iteratif (PolarTensorKrylovSolver) sur un
# anneau (r, theta), r_min > 0. Supporte un coefficient TENSORIEL anisotrope A = [[a_rr,a_rt],[a_tr,a_tt]]
# eventuellement NON symetrique (termes croises), en FV conservatif avec metrique polaire. Solveur
# BiCGStab matrice-libre preconditionne JACOBI (diagonal) -- PAS de V-cycle MG (stagnation 1/r^2 signalee
# par le scoping). MMS : convergence O(2) isotrope (A=I) ET tenseur non symetrique (a_rt != a_tr),
# convergence BiCGStab (pas de stagnation), consistance isotrope vs PolarPoissonSolver direct. C++ SEUL,
# PAS branche dans System.step (etape 2b = SchurReconstructKernel polaire + wiring). Additif : le
# PolarPoissonSolver scalaire direct et le Schur cartesien restent INTOUCHES.
adc_add_test(test_polar_tensor_elliptic_mms)
# Extension MULTI-BOX (plusieurs boites PAR RANG) du Schur polaire tensoriel. B2 a rendu le solveur
# multi-rang (decoupage theta seul, UNE box couvrant l'anneau par rang) mais ne testait que A=I. Ce test
# cible le MULTI-BOX intra-rang avec TENSEUR (termes croises a_rt/a_tr) : (A) parite theta-split 8 boites
# vs mono-box [RadialLine] -- exerce le COIN du 9-points cross-box (ghosts diagonaux des termes croises) ;
# (B) parite pavage 2D coupant r ET theta vs mono-box [Jacobi] -- chemin multi-box COMPLET (Jacobi sans
# contrainte de layout) ; (C) garde-fou : layout coupant r sous RadialLine doit lever, OK sous Jacobi ;
# (D) parite Neumann (jauge project_mean multi-box). Solves/normes COLLECTIFS -> correct aussi sous MPI.
adc_add_test(test_polar_schur_multibox)
# Voie A etape 2b : ETAGE SOURCE condense par Schur en geometrie POLAIRE
# (PolarCondensedSchurSourceStepper). Pendant POLAIRE du CondensedSchurSourceStepper cartesien (#126) :
# assemble le tenseur condense A = I + theta^2 dt^2 alpha rho B^{-1} (re-etiquetage polaire r/theta du
# meme B^{-1} 2x2 de Lorentz), resout L_int(phi)=div(A grad phi)=-rhs_schur via PolarTensorKrylovSolver
# (#210), reconstruit v=B^{-1}(v^n - theta dt grad_polar phi) en metrique polaire, extrapole, energie,
# ghosts. Test SOURCE-ONLY : fluide magnetise RAIDE (B0=100) ou l'Euler explicite explose. Verifie (A)
# la relation implicite reconstruite, (B) la stabilite Schur vs explosion explicite, (C) l'accord avec
# une reference RK4 fine-dt (ordre 1). Mono-rang (PolarTensorKrylovSolver/PolarPoissonSolver). Additif :
# le Schur CARTESIEN reste BIT-IDENTIQUE (chemin separe).
adc_add_test(test_polar_condensed_schur_source_stepper)
# Schur PR3 : solveur de Krylov MATRICE-LIBRE (BiCGStab) preconditionne par le V-cycle MG sur la
# partie SYMETRIQUE, pour l'operateur a tenseur PLEIN non symetrique. Resout les cas MMS ou le MG
# SEUL echoue (#120 : stagnation c=0.1-0.4, divergence c=0.7), + consistance A=I vs GeometricMG.
# (entree serie ; les entrees MPI np=1/2/4 sont ajoutees dans le bloc ADC_HAS_MPI plus bas)
adc_add_test(test_krylov_solver)
# Schur PR3 (batisseur) : ASSEMBLE l'operateur tensoriel A_op = I + theta^2 dt^2 alpha rho B^{-1}
# (eps_x/eps_y diag, a_xy/a_yx croises) + le RHS condense -Lap phi^n - theta dt alpha div(rho B^{-1} v^n)
# depuis les roles Density/MomentumX/MomentumY d'un bloc fluide. PAS de solve (PR4). Tests analytiques :
# coeff vs A=I+c rho B^{-1} ferme, RHS vs analytique interieur, B_z=0 -> Poisson bit-identique.
# (entree serie ; entrees MPI np=1/2/4 ajoutees dans le bloc ADC_HAS_MPI plus bas)
adc_add_test(test_schur_condensation)

# Schur PR4 (etage source) : CondensedSchurSourceStepper COMPOSE le batisseur #124 + le solveur de
# Krylov BiCGStab (#122, preconditionne MG sur la partie symetrique) + LorentzEliminator (#118) pour
# JOUER l'etage source condense (assemble A_op + RHS, resout L_schur(phi)=rhs, reconstruit
# v=B^{-1}(v^n - theta dt grad phi), extrapole, energie, ghosts). Test SOURCE-ONLY : un fluide
# magnetise sous la source implicite, a GRAND dt ou l'Euler explicite explose. Verifie (A) la relation
# implicite reconstruite, (B) la stabilite Schur vs explosion explicite, (C) l'accord avec une
# reference RK4 fine-dt (ordre 1). (entree serie ; entrees MPI np=1/2/4 dans le bloc ADC_HAS_MPI plus bas)
adc_add_test(test_condensed_schur_source_stepper)
# Etage source condense par Schur sur HIERARCHIE AMR (AmrCondensedSchurSourceStepper) : PARITE
# mono-niveau avec l'etage uniforme #126 (un seul niveau couvrant tout le domaine -> bit-pour-bit),
# relation implicite reconstruite, source non triviale, garde multi-niveau (suivi Etape 4).
adc_add_test(test_amr_condensed_schur_source_stepper)
# Etage source condense par Schur COMPOSITE multi-niveau (Phase 3c) : sur une hierarchie 2 niveaux
# (grossier + 1 patch fin), le patch fin RAFFINE l'elliptique (operateur tensoriel Schur resolu par FAC)
# + reconstruction des vitesses par niveau + cascade average_down. Verifie finitude, source non triviale,
# rho gelee, coherence average_down (couvert = moyenne fine), stabilite a grand dt.
adc_add_test(test_amr_condensed_schur_composite)

# --- reconstruction / discretisation / integrateurs generiques ---
adc_add_test(test_dispatch_tags)  # registry UNIQUE des tags (validate_limiter/riemann, limiter_n_ghost) : matrice + messages
adc_add_test(test_model_registry)  # registry UNIQUE des briques modele (transport/source/elliptic) : matrice + messages + perimetre (ADC-331)
adc_add_test(test_spatial_discretisation)
adc_add_test(test_primitive_recon)
adc_add_test(test_system_abstraction)
adc_add_test(test_system_coupler)
adc_add_test(test_two_species_minimal)
adc_add_test(test_coupled_source)
adc_add_test(test_system_two_explicit)
adc_add_test(test_assembler_driver)
adc_add_test(test_amr_system_coupler)
adc_add_test(test_amr_source_covered_cells)  # A3 : cellules grossieres couvertes coherentes apres source par niveau (average_down trailing)
adc_add_test(test_amr_composite_source_conservation)  # A3 : masse totale (somme feuille-consciente) conservee par la source couplee conservative
adc_add_test(test_amr_layout_guard)  # garde-fou de layout AMR partage : AmrHierarchyLayout + same_layout_or_throw (comparaison EXACTE)
adc_add_test(test_amr_stride_cadence)  # cadence HOLD-THEN-CATCH-UP dans AmrSystemCoupler
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
adc_add_test(test_amr_positivity_floor)  # Zhang-Shu floor on the AMR transport + C/F ghost clamp (ADC-259)
# Chantier T2 : masque de domaine DISQUE conservatif (CONTRAT, inerte par defaut). (a) bit-identite
# du residu masque tout-actif vs assemble_rhs historique ; (b) conservation de masse sur les cellules
# actives (flux normal nul aux faces active/inactive) + residu nul sur les cellules inactives.
adc_add_test(test_disc_domain_mask)

# Chantier T5-PR2 : transport FV EMBEDDED-BOUNDARY conservatif sur disque (assemble_rhs_eb). Generalise
# le masque T2 (portes de face 0/1) a un schema CUT-CELL : ouvertures de face alpha_f in [0,1] et
# fraction de volume kappa (detail::cut_fraction, T5-PR1), flux de paroi no-penetration, residu /kappa
# (clampe petite cellule). (a) MMS sur disque : ordre 2 a l'interieur (rapport d'erreur ~4) ; (b)
# conservation de masse a la machine (Sum n kappa_eff dx dy, aucun flux ne franchit le mur immerge) +
# residu FINI (clamp small-cell, pas de NaN) ; (c) bit-identite : sans coupe, EB == operateur cartesien.
adc_add_test(test_eb_transport)

# ADC-327 : contrat GENERIQUE embedded-boundary / level-set (numerics/embedded_boundary.hpp). Prouve que
# le contrat admet un domaine NON-disque (detail::HalfPlaneDomain) de bout en bout : concept LevelSetDomain
# (static_assert), partition + cellules coupees, bit-identite sans coupe (EB == cartesien), residu fini
# (clamp generique), masque staircase materialise (residu nul sur inactives, tout-actif == cartesien),
# conservation a la machine sur une bande non-disque. Le disque devient une instance du contrat.
adc_add_test(test_embedded_boundary_generic)

# Chantier T5-PR3 : CABLAGE du transport disque (staircase + cut-cell EB) dans System::step. Resout le
# footgun T2 (set_disc_domain materialisait un masque que step() ignorait) : le transport est aiguille
# vers assemble_rhs_masked (staircase) / assemble_rhs_eb (cutcell) selon un mode explicite. (a) mode none
# byte-identique au chemin sans disque ; (b) staircase != carre + masse active conservee a la machine ;
# (c) cutcell fini + != carre, disque englobant == carre (sans coupe) ; (d) mode honore sous Lie ET Strang.
# Compile le runtime System (python/system.cpp) avec le test, comme test_compiled_model_parity.
add_executable(test_facade_routing test_facade_routing.cpp)
target_link_libraries(test_facade_routing PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
add_test(NAME test_facade_routing COMMAND test_facade_routing)

# ADC-299 + ADC-290 : validation amont de la config (System / AmrSystem, AVANT Impl) et contrat de
# completude du ModelSpec (transport / elliptic explicites, pas de retombee compressible / charge).
# Compile les deux runtimes (System + AmrSystem) -> link adc_runtime_system ET adc_runtime_amr. La
# boucle de cap memoire en fin de fichier le passe automatiquement a -O0 + pool adc_heavy_tu.
add_executable(test_config_model_validation test_config_model_validation.cpp)
target_link_libraries(test_config_model_validation
                      PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system adc_runtime_amr)
add_test(NAME test_config_model_validation COMMAND test_config_model_validation)

# ADC-291 : la marshaling host (native_loader::host_residual) transporte les aux NOMMES (extra[k]),
# pas seulement les extras canoniques. native_loader.hpp inclut system.hpp -> link adc_runtime_system.
add_executable(test_native_aux_named test_native_aux_named.cpp)
target_link_libraries(test_native_aux_named
                      PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
add_test(NAME test_native_aux_named COMMAND test_native_aux_named)

# --- canal aux extensible (n_aux + load_aux<NComp>) : champ extra B_z lu et peuple ---
adc_add_test(test_aux_extra)
adc_add_test(test_aux_names)  # ADC-291 : table de noms aux canonique C++ <-> Python (aux_names.hpp)
adc_add_test(test_aux_halo)   # ADC-369 : halo/BC par champ aux nomme (fill_physical_bc scope par composant)
adc_add_test(test_aux_coupler_bz)
adc_add_test(test_aux_system_bz)  # B_z partage multi-blocs (SystemAssembler, aux = max sur les blocs)
adc_add_test(test_aux_composite)  # CompositeModel propage n_aux depuis ses briques (B_z)
adc_add_test(test_amr_aux_bz)     # lecture + allocation aux width-aware sur les chemins AMR (B_z lu)
adc_add_test(test_amr_system_bz_pop)  # B_z peuple PAR NIVEAU dans AmrSystemCoupler (echantillonne au niveau)
adc_add_test(test_amr_system_bz_multibox)  # B_z PAR NIVEAU multi-box (grossier 2x2, fin multi-patch) mono-rang
adc_add_test(test_aux_single_source)  # source unique ADC_AUX_FIELDS : load_aux + marshaling generes, tous champs couverts (anti-#51)

adc_add_test(test_weno_convergence)
adc_add_test(test_splitting)
# Strang GENERIQUE orchestre par le STEPPER DE SYSTEME (SystemStepper::step_strang) : ordre TEMPOREL
# observe ~2 (Strang) vs ~1 (Lie) sur deux operateurs jouets NON commutants, non-regression du chemin
# Lie (bit-identique), consistance phi (3 solve_fields/macro-pas en Strang, RE-RESOLU entre les etages,
# cf. docs/HOFFART_STEP_SEQUENCE.md), degenerescences source nulle == transport pur / transport nul ==
# source pure. Instancie SystemStepper sur un Impl JOUET (mono-box, serie) ; aucun Schur/AMReX lourd.
adc_add_test(test_strang_splitting)
adc_add_test(test_imex_ap)
adc_add_test(test_ap_limit)

# --- Schur PR2 : brique Lorentz-elimination (operateur B et son inverse analytique) ---
# Teste les proprietes analytiques de LorentzEliminator : round-trip Binv(B(v))==v,
# coherence des entrees de B^{-1}, relation norme, et cas degenere B_z=0.
adc_add_test(test_lorentz_eliminator)

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

# --- Phase 2b grille polaire : pas COUPLE polaire (transport -> Poisson -> aux -> avance) au niveau
# C++, pendant du chemin cable dans System::step pour geometry == "polar". Exerce PolarPoissonSolver +
# derivation aux en base locale (e_r, e_theta) + transport polaire avec paroi radiale solide. Verifie
# (A) un pas couple modifie reellement la densite (champ non gele) et (B) la masse Sum n r dr dtheta est
# conservee a ~machine sur K pas (paroi radiale no-penetration -> wall_radial). Host / Serial-safe (UNE
# box ; PolarPoissonSolver mono-rang, non enregistre MPI comme test_polar_poisson_mms).
adc_add_test(test_polar_system_step)

# --- Lot A4 grille polaire : MMS DEDIE avec vitesse RADIALE v_r != 0. Avance en temps (assemble_rhs_polar
# + SSPRK3) jusqu'a un t court sous un champ a v_r != 0 constant + un TERME SOURCE manufacture, et asserte
# l'ORDRE DE CONVERGENCE de l'erreur L2 de la solution vs l'exact (>= 1.8, calibre sur l'ordre 2 mesure ;
# divergence FV polaire formellement d'ordre 2, cf. test_polar_transport_mms). Comble le trou laisse par
# #168/#174 : aucun MMS polaire dedie a v_r != 0 ne verifiait quantitativement le transport RADIAL (et pas
# seulement azimutal). Brique de transport + source MANUFACTURE locales au test (aucun code de prod). Host
# / Serial-safe (UNE box, non enregistre MPI comme les autres tests polaires mono-box).
adc_add_test(test_polar_mms_vr)

# --- Voie A etape 1 grille polaire : TRANSPORT FLUIDE ISOTHERME (3 var : rho, rho v_r, rho v_theta)
# via la brique IsothermalFluxPolar (flux physique reutilise d'IsothermalFlux + terme GEOMETRIQUE de
# courbure polar_geom_source = (0, (rho v_theta^2 + p)/r, -(rho v_r v_theta)/r)). Verifie : (A)
# preservation d'un EQUILIBRE ROTATIF exact (rotation rigide + densite cyclostrophique) -> le residu
# radial converge vers 0, PREUVE que le terme centrifuge est correct ; (A') controle negatif : la
# meme config SANS terme geometrique (IsothermalFlux) laisse un residu O(1) non convergent -> le
# terme geometrique est INDISPENSABLE ; (B) convergence MMS d'ordre 2 du systeme 3 var complet (v_r
# != 0) ; (C) conservation de la masse a ~machine (paroi radiale). Host / Serial-safe (UNE box).
adc_add_test(test_polar_fluid_transport)

# Exemption FAST_O0 des tests d'INTEGRATION LONGUE polaires : ils avancent une advection sur de
# nombreux pas (94 s et ~35 s a -O3) ; a -O0 ils passent a 600+ s et 215 s -- le premier depasse
# le TIMEOUT anti-hang de 600 s (cas reel : job MPI + Kokkos Serial, ctest #92 Timeout). On les
# remonte a -O2 : place APRES le add_compile_options(-O0) de tete de fichier, le dernier -O gagne.
# Toujours des controles de correction (rien d'autre ne change) ; no-op quand FAST_O0 est OFF.
if(ADC_TESTS_FAST_O0)
  foreach(_t test_polar_ring_advection test_polar_fluid_transport)
    target_compile_options(${_t} PRIVATE
      $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O2>)
  endforeach()
endif()

# --- FORCE DE LORENTZ MAGNETIQUE v x B_z NATIVE pour le fluide isotherme polaire (3 var) : brique de
# SOURCE MagneticLorentzForce (physics/source.hpp) + composition CompositeSource<PotentialForce,
# MagneticLorentzForce> cablee dans dispatch_source (none/potential/gravity/magnetic/potential_magnetic).
# Comble le contournement du demonstrateur diocotron (la geometrie polaire ne cablait pas v x B_z).
# Verifie : (A) algebre ponctuelle (0, +qom B_z m_theta, -qom B_z m_r), travail nul, composition,
# propagation du canal aux (n_aux=4 -> B_z) ; (B) giration cyclotron sous la SEULE force de Lorentz
# (|m| constant, rotation pure) ; (C) diocotron polaire NATIF (source potential_magnetic) : pas de NaN,
# masse conservee a ~machine, mode azimutal croissant porte par la VRAIE force magnetique (vs controle
# B_z=0). Host / Serial-safe (UNE box). La force est ALGEBRIQUE et invariante par orientation du repere
# -> la MEME brique sert cartesien et polaire (regime explicite ; le raide passe par le Schur #212).
adc_add_test(test_polar_lorentz_source)

# --- dispatch a l'execution (modele type-erased pour brique generee / JIT) ---
adc_add_test(test_dynamic_model)

# --- constructeur de fermetures de bloc instanciable hors System (fondation backend AOT) ---
adc_add_test(test_block_builder)

# --- WENO5-Z (ordre 5, 3 ghosts) + SSPRK3 cables via make_block/build_block : parite schema, routage
#     du tag temporel, no-default-change (SSPRK2 bit-identique), convergence > O2 ---
adc_add_test(test_weno5_ssprk3)

# --- limiteur de POSITIVITE Zhang-Shu (pos_floor, ADC-76) : mecanique du scaling, no-default-change
#     (floor=0 bit-identique), garantie rho_face >= floor sur le top-hat 1e6, rejet sans role Density ---
adc_add_test(test_positivity_floor)

# --- ADC-329 : les briques de VALIDATION/REFERENCE (AdvectionDiffusion, LangmuirMode, TwoFluidLinear)
#     vivent sous adc/validation/physics/ (namespace adc::validation), hors de la surface production.
#     Ce test prouve que les NOUVEAUX chemins canoniques ET les forwarders de compat (adc/physics/...)
#     compilent, et que les alias adc::<Type> deprecies nomment les memes types. ---
adc_add_test(test_physics_validation_compat)

# --- cache des vitesses d'onde HLL (opt-in) : PREUVE D'ENGAGEMENT par comptage de model.wave_speeds.
#     make_block(none, hll) OFF vs ON sur N pas SSPRK2 : bit-exact (0 ulp) + calls_on < calls_off
#     (le cache evalue par cellule, le chemin par face rappelle par face). Attrape un retour en no-op
#     silencieux (calls_on == calls_off) que la bit-identite ON==OFF du test Python ne verrait pas. ---
adc_add_test(test_wave_speed_cache_engagement)

# --- parite bloc compile AOT (add_compiled_model) == bloc natif (add_block) ---
# Compile le runtime System (python/system.cpp) avec le test : add_compiled_model branche un
# CompositeModel connu a la compilation sur la grille reelle, comme un bloc natif (sans marshaling).
add_executable(test_compiled_model_parity test_compiled_model_parity.cpp)
target_link_libraries(test_compiled_model_parity PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
add_test(NAME test_compiled_model_parity COMMAND test_compiled_model_parity)

# --- WENO5 sur le chemin compile natif (add_compiled_model) == bloc natif (add_block) ---
# Verrou de l'unification ghost : add_compiled_model alloue le bloc a block_n_ghost(limiter) (3 pour
# weno5) APRES install_block, comme add_block. Parite STRICTE weno5 + no-default-change none/minmod
# (allocation <= 2 ghosts inchangee). Compile le runtime System comme test_compiled_model_parity.
add_executable(test_weno5_compiled_model test_weno5_compiled_model.cpp)
target_link_libraries(test_weno5_compiled_model PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
add_test(NAME test_weno5_compiled_model COMMAND test_weno5_compiled_model)

# --- parite bloc compile AOT (add_compiled_model) == bloc natif (add_block) COTE AMR ---
# Compile le runtime AmrSystem (python/amr_system.cpp) avec le test : add_compiled_model branche un
# CompositeModel connu a la compilation sur la hierarchie AMR reelle (AmrCouplerMP + reflux + regrid),
# comme add_block (sans dispatch ModelSpec). Densite grossiere et masse bit-identiques apres N pas.
add_executable(test_amr_compiled_model test_amr_compiled_model.cpp)
target_link_libraries(test_amr_compiled_model PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_compiled_model COMMAND test_amr_compiled_model)

# --- ADC-324 : le patch fin SEED n'existe que si le raffinement est configure (set_refinement) ---
# Meme machinerie que test_amr_compiled_model (compile python/amr_system.cpp avec le test) : sans
# set_refinement (refine_threshold == 1e30) la hierarchie compilee est MONO-NIVEAU (n_patches()==0),
# avec set_refinement(seuil fini) le seed est alloue (n_patches()>=1). Verrouille la suppression du
# patch fin rang-0 qui bridait le strong-scaling MPI quand le raffinement etait inactif.
# ADC-335: amr_system.cpp now delegates its dispatch to the per-transport seam TUs (ADC_AMR_SEAM),
# so this ADC-324 test -- added on master after this branch forked and still re-listing amr_system.cpp
# directly -- must pick the seam up too. Link the adc_runtime_amr object lib (amr_system.cpp + the seam,
# compiled once at -O0 under the heavy-TU pool) like test_amr_compiled_model, instead of recompiling.
add_executable(test_amr_seed_no_refine test_amr_seed_no_refine.cpp)
target_link_libraries(test_amr_seed_no_refine PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_seed_no_refine COMMAND test_amr_seed_no_refine)

# --- ADC-296 : critere de regrid configurable par nom/role (au-dela de la composante 0) ---
# Exerce le resolveur detail::resolve_selected_component ET la facade multi-blocs (build_multi resout le
# selecteur par bloc via AmrRuntime::block_cons_vars). Linke adc_runtime_amr (amr_system.cpp + seam),
# comme test_amr_seed_no_refine / test_amr_system_twoblock.
add_executable(test_amr_regrid_variable test_amr_regrid_variable.cpp)
target_link_libraries(test_amr_regrid_variable PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_regrid_variable COMMAND test_amr_regrid_variable)

# --- AmrSystem::potential() : phi du niveau grossier expose, pendant de System::potential() ---
# Compile les runtimes AmrSystem ET System (python/amr_system.cpp + python/system.cpp) avec le test :
# sans raffinement, AmrSystem resout le MEME Poisson discret que System(geometric_mg) sur le grossier
# mono-box -> potential() : forme n*n, valeurs finies, champ non trivial, moyenne ~0 (source neutre),
# et PARITE avec System::potential() a la tolerance MG pres. C'est l'observable phi qu'echantillonne le
# benchmark diocotron (FFT azimutale sur un cercle median) cote AMR.
add_executable(test_amr_potential test_amr_potential.cpp)
target_link_libraries(test_amr_potential PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr adc_runtime_system)
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
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp ${ADC_AMR_SEAM})
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
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp ${ADC_AMR_SEAM})
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
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp ${ADC_AMR_SEAM})
target_link_libraries(test_amr_riemann_native PRIVATE adc::adc ${CMAKE_DL_LIBS})
set_target_properties(test_amr_riemann_native PROPERTIES ENABLE_EXPORTS ON)
target_compile_definitions(test_amr_riemann_native PRIVATE
  ADC_TEST_CXX="${CMAKE_CXX_COMPILER}"
  ADC_TEST_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}/../include"
  ADC_TEST_CXX_STD="c++${CMAKE_CXX_STANDARD}"
  ADC_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME test_amr_riemann_native COMMAND test_amr_riemann_native)

# --- IMEX (source raide implicite) sur le chemin "production" NATIF cote AMR (Gap 2 parite) ---
# Pendant de test_amr_riemann_native : prouve que time="imex" tourne la MEME hierarchie AMR
# (AmrCouplerMP + reflux + regrid), add_compiled_model == add_block == add_native_block BIT-IDENTIQUE
# (dmax==0), que l'IMEX est ACTIF (!= explicite) et STABLE la ou l'explicite EXPLOSE sur une source
# raide (backward_euler_source cellule-local), et que la masse grossiere reste conservee a ~machine
# (la source hors registres de reflux). Compile amr_system.cpp comme test_amr_riemann_native ; memes
# injections compilateur / include / norme / tmpdir.
add_executable(test_amr_imex_native test_amr_imex_native.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../python/amr_system.cpp ${ADC_AMR_SEAM})
target_link_libraries(test_amr_imex_native PRIVATE adc::adc ${CMAKE_DL_LIBS})
set_target_properties(test_amr_imex_native PROPERTIES ENABLE_EXPORTS ON)
target_compile_definitions(test_amr_imex_native PRIVATE
  ADC_TEST_CXX="${CMAKE_CXX_COMPILER}"
  ADC_TEST_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}/../include"
  ADC_TEST_CXX_STD="c++${CMAKE_CXX_STANDARD}"
  ADC_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME test_amr_imex_native COMMAND test_amr_imex_native)

# --- contrat mono-bloc/explicite de la facade AmrSystem : refus explicite des params non cables ---
# Compile le runtime AmrSystem (python/amr_system.cpp) avec le test : verrouille que set_poisson
# refuse solver/rhs hors du domaine cable (GeometricMG + somme des briques elliptiques) et qu'add_block
# refuse time != explicit / recon hors domaine / substeps < 1, au lieu de les ignorer en silence.
add_executable(test_amr_system_contract test_amr_system_contract.cpp)
target_link_libraries(test_amr_system_contract PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_system_contract COMMAND test_amr_system_contract)

# --- AMR MULTI-BLOCS (capstone PR1) : facade AmrSystem -> moteur runtime AmrRuntime ---------------
# Deux blocs EXPLICITES a schemas DIFFERENTS sur UNE hierarchie AMR PARTAGEE : Poisson somme
# co-localise (q0 n0 + q1 n1), masse conservee PAR BLOC, mono-bloc bit-identique (dmax=0), refus
# multi-blocs + regrid. Compile python/amr_system.cpp (la facade) avec le test.
add_executable(test_amr_system_twoblock test_amr_system_twoblock.cpp)
target_link_libraries(test_amr_system_twoblock PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_system_twoblock COMMAND test_amr_system_twoblock)

# --- AMR MULTI-BLOCS COMPILES (capstone "v", DSL production multi-bloc) -----------------------------
# PLUSIEURS blocs COMPILES (add_compiled_model, types connus a la compilation) co-localises sur UNE
# hierarchie AMR PARTAGEE via le moteur runtime AmrRuntime (file de specs construite paresseusement a
# ensure_built). Le 2e bloc compile NE LEVE PLUS (avant : "un seul bloc compile est supporte").
# Verrouille : (A) deux blocs compiles a schemas differents evoluent + masse par bloc, (B) couplage
# inter-blocs compiles conservatif, (C) melange compile + natif, (D) mono-bloc compile bit-identique
# (route par AmrCouplerMP), (E) non-regression du refus. Types CONCRETS / foncteurs nommes (pas de
# lambda generique) -> compilable sous nvcc. Compile python/amr_system.cpp (la facade) avec le test.
add_executable(test_amr_multiblock_compiled test_amr_multiblock_compiled.cpp)
target_link_libraries(test_amr_multiblock_compiled PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_multiblock_compiled COMMAND test_amr_multiblock_compiled)

# --- AMR MULTI-BLOCS MULTIRATE (capstone iv) : substeps + stride PAR BLOC honores par AmrRuntime ---
# substeps reellement exerces (A=4 differe de A=1, etat fini, masse conservee), stride hold-then-catch-up
# (bloc lent tenu au mac0, rattrape au mac1, Poisson somme actif), step_cfl substeps/stride-aware
# (dt = cfl*h*min_b(substeps_b/(stride_b*w_b))), mono-bloc bit-identique. Compile python/amr_system.cpp.
add_executable(test_amr_multiblock_substeps test_amr_multiblock_substeps.cpp)
target_link_libraries(test_amr_multiblock_substeps PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_multiblock_substeps COMMAND test_amr_multiblock_substeps)

# --- AMR MULTI-BLOCS SOURCES COUPLEES (capstone vi) : add_coupled_source sur la facade runtime --------
# echange conservatif PAR CELLULE (add_pair +k*ni*ng / -k*ni*ng, meme cellule), source active (l'etat
# change), conservation GLOBALE composite sous K macro-pas, couverture #169 (grossier couvert = moyenne
# 2x2 du fin apres la source), multirate (stride=2 reste conservatif), opt-in bit-identique. Compile
# python/amr_system.cpp (la facade).
add_executable(test_amr_multiblock_coupled_source test_amr_multiblock_coupled_source.cpp)
target_link_libraries(test_amr_multiblock_coupled_source PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_multiblock_coupled_source COMMAND test_amr_multiblock_coupled_source)

# --- AMR SOURCES COUPLEES : RESOLUTION DE ROLE STRICTE (P0, mirroir de System #181) -------------------
# add_coupled_source sur le moteur runtime AMR : un role CANONIQUE mais NON EXPOSE par le bloc (momentum_x
# sur un bloc scalaire) doit LEVER (avec bloc + role dans le message) au lieu de retomber silencieusement
# sur la composante 0. Trois cas : role valide (reussit), role absent (leve), role inconnu (leve).
add_executable(test_amr_coupled_source_role_strict test_amr_coupled_source_role_strict.cpp)
target_link_libraries(test_amr_coupled_source_role_strict PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_coupled_source_role_strict COMMAND test_amr_coupled_source_role_strict)

# --- AMR MULTI-BLOCS IMEX (capstone vii) : time="imex" PAR BLOC honore par la facade runtime ----------
# Source raide locale implicite (backward_euler_source par niveau, mirroir de AmrSystemCoupler::step :
# SourceFreeModel + AmrImplicitSourceStepper) : bloc raide STABLE sous IMEX la ou l'explicite EXPLOSE
# (disable-and-fail), masse conservee a ~machine (source cellule-locale, hors reflux), multi-blocs
# tout-explicite bit-identique (opt-in), facade time="imex" + masque IMEX partiel (implicit_roles).
# Compile python/amr_system.cpp (la facade).
add_executable(test_amr_multiblock_imex test_amr_multiblock_imex.cpp)
target_link_libraries(test_amr_multiblock_imex PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_multiblock_imex COMMAND test_amr_multiblock_imex)

# --- AMR MULTI-BLOCS REGRID D'UNION DES TAGS (capstone Phase 2, C.6) -----------------------------------
# AmrRuntime::regrid pilote par l'UNION (OU cellule a cellule) des tags de TOUS les blocs (predicat par
# bloc, D1) + tags de phi sur |grad phi| (D4) -> un clustering Berger-Rigoutsos -> un layout fin partage
# applique a TOUS les blocs (y compris stride-tenus, D3) + a l'aux, en maintenant same_layout_or_throw
# (D5 : 2 niveaux). Couvre (a) hierarchie qui evolue, (b)+(c) union A OU B OU |grad phi|, (d) bloc
# stride-tenu re-grille, (e) regrid_every==0 bit-identique. Foncteurs de tag NOMMES (nvcc-safe).
add_executable(test_amr_multiblock_regrid_union test_amr_multiblock_regrid_union.cpp)
target_link_libraries(test_amr_multiblock_regrid_union PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
add_test(NAME test_amr_multiblock_regrid_union COMMAND test_amr_multiblock_regrid_union)

# --- runtime System : B_z partage cable de bout en bout (ensure_aux_width + set_magnetic_field) ---
add_executable(test_aux_runtime_bz test_aux_runtime_bz.cpp)
target_link_libraries(test_aux_runtime_bz PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
add_test(NAME test_aux_runtime_bz COMMAND test_aux_runtime_bz)

# --- runtime System : T_e DERIVE d'un bloc fluide (set_electron_temperature_from, 2e champ aux) ---
add_executable(test_aux_te test_aux_te.cpp)
target_link_libraries(test_aux_te PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
add_test(NAME test_aux_te COMMAND test_aux_te)

# --- descripteur de variables : roles physiques (index_of(role)) ---
adc_add_test(test_variable_role)

# --- roles utilisateurs nommes + fallbacks canoniques stricts (ADC-292) ---
adc_add_test(test_variable_user_role)

# --- flux de Roe (Euler) + eigenvalues() ---
adc_add_test(test_roe_flux)
# Capabilities Riemann (audit 2026-06) : HLLC/Roe generiques par hooks modele (HasHLLCStructure /
# HasRoeDissipation) -- equivalence bit-pour-bit au chemin canonique Euler + HLLC isotherme 3-var
# (contact preserve la ou HLL diffuse).
adc_add_test(test_riemann_capabilities)
# Newton source implicite generalise (audit vague 2) : convergence non-Euler multi-variables sous
# tolerances, damping, fail_policy=throw + cellule fautive, diagnostics = observateur pur.
adc_add_test(test_newton_robustness)

# --- MPI (coeur) : seam comm, reductions, Poisson distribue ---
if(ADC_HAS_MPI)
  # Backfill du label "mpi" (ADC-366). Beaucoup de tests distribues sont enregistres INLINE en
  # add_test(NAME ..._np${np} ...) (np variables, postflags specifiques) sans passer par
  # adc_add_mpi_test, donc SANS label : un `ctest -L mpi` les sautait silencieusement (perte de
  # couverture). On photographie l'ensemble des tests AVANT ce bloc ; tout test ajoute DANS le bloc
  # est par construction un test MPI -> on lui (re)pose LABELS "mpi" + TIMEOUT 600 a la fin du bloc.
  # Difference d'ensemble exacte (aucune selection par motif de nom, fragile) ; idempotent pour les
  # tests deja labellises par adc_add_mpi_test.
  get_property(_adc_tests_before_mpi DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
  function(adc_add_mpi_test name nproc)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE adc::adc)
    add_test(NAME ${name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
    # `ctest -L mpi` pour ne (re)jouer que le sous-ensemble distribue ; TIMEOUT car un deadlock
    # MPI (classe de bug reelle, cf. fill_boundary) bloquerait sinon le runner entier.
    set_tests_properties(${name} PROPERTIES LABELS "mpi" TIMEOUT 600)
  endfunction()
  # Schur PR3 : le solveur de Krylov (BiCGStab) rejoue sous MPI np=1/2/4. Les produits scalaires
  # (dot) sont COLLECTIFS (all_reduce_sum), donc le critere d'arret se declenche a la MEME iteration
  # sur tous les rangs : iterations + convergence INVARIANTES au nombre de rangs (le test all_reduce
  # le nombre d'iters et exige spread nul). np=1 couvre le chemin mono-rang (== serie) ; np=2/4 le
  # chemin distribue (rangs avec local_size()==0 sur ce decoupage mono-box, dot appele partout).
  # Reutilise la cible serie test_krylov_solver (deja construite par adc_add_test, pas de doublon).
  foreach(np 1 2 4)
    add_test(NAME test_krylov_solver_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_krylov_solver> ${MPIEXEC_POSTFLAGS})
  endforeach()

  # Schur PR3 (batisseur) sous MPI np=1/2/4 : l'assemblage (coefficients A_op + RHS) doit etre
  # INVARIANT au nombre de rangs. Les boucles iterent sur local_size() (un rang sans box n'execute
  # aucun kernel, ne deref jamais fab(0)) et les reductions de verification (all_reduce_max) sont
  # COLLECTIVES (appelees sur tous les rangs, y compris vide) : pas d'interblocage. np=1 = chemin
  # mono-rang (== serie) ; np=2/4 = chemin distribue. Reutilise la cible serie test_schur_condensation.
  foreach(np 1 2 4)
    add_test(NAME test_schur_condensation_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_schur_condensation> ${MPIEXEC_POSTFLAGS})
  endforeach()

  # Schur PR4 (etage source) sous MPI np=1/2/4 : l'etage source condense doit etre INVARIANT au nombre
  # de rangs. Le solve de Krylov est COLLECTIF (dot/all_reduce sur tous les rangs, y compris vides) :
  # iterations et resultat invariants ; les verifs (ecarts MAX, normes) sont all_reduce_max/sum. np=1
  # = chemin mono-rang (== serie) ; np=2/4 = chemin distribue (rangs local_size()==0 sur le mono-box).
  foreach(np 1 2 4)
    add_test(NAME test_condensed_schur_source_stepper_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_condensed_schur_source_stepper>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- Schur POLAIRE multi-rang : PolarTensorKrylovSolver / etage Schur polaire sous MPI (decoupage
  # AZIMUTAL). Le solveur polaire etait MONO-RANG (garde-fou dur n_ranks>1) ; il est desormais distribue
  # par decoupage en theta (chaque box couvre la plage radiale complete pour le precond RadialLine), avec
  # produits scalaires collectifs (all_reduce) et projection de jauge (pure-Neumann) all_reduite. Le test
  # verifie, sur une MMS polaire (Dirichlet ET pure-Neumann), que la solution discrete -- et son erreur L2
  # GLOBALE vs l'exact -- est INVARIANTE au decoupage theta ET au nombre de rangs (mono-box vs theta-split,
  # ecart ~ reassociation FP des all_reduce), et que les iterations BiCGStab sont identiques sur tous les
  # rangs (critere d'arret collectif). np=1 = parite multi-box mono-rang ; np=2/4 = cross-rang MPI.
  add_executable(test_mpi_polar_schur test_mpi_polar_schur.cpp)
  target_link_libraries(test_mpi_polar_schur PRIVATE adc::adc)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_polar_schur_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_polar_schur> ${MPIEXEC_POSTFLAGS})
  endforeach()

  adc_add_mpi_test(test_mpi_smoke 4)
  adc_add_mpi_test(test_mpi_array_reduce 4)
  adc_add_mpi_test(test_mpi_fillboundary 4)
  # Cache du schedule de halos (ADC-260) sous MPI : la reutilisation du schedule cache doit garder
  # les listes send/recv ALIGNEES cross-rang (enumeration deterministe identique) et le resultat
  # correct sur K appels. np=1 = chemin local (== serie, deja couvert par adc_add_test) ; np=2/4 =
  # chemin MPI cross-rang. Reutilise la cible serie test_fill_boundary_cache (pas de doublon).
  foreach(np 1 2 4)
    add_test(NAME test_fill_boundary_cache_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_fill_boundary_cache>
                     ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_fill_boundary_cache_np${np} PROPERTIES LABELS "mpi" TIMEOUT 600)
  endforeach()
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

  add_executable(test_mpi_amr_compiled_parity test_mpi_amr_compiled_parity.cpp)
  target_link_libraries(test_mpi_amr_compiled_parity PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_amr_compiled_parity_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_amr_compiled_parity>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- PARITE MPI du capstone AMR MULTI-BLOCS (PR1) : DEUX blocs explicites a schemas DIFFERENTS sur
  # UNE hierarchie partagee (Poisson de systeme somme q0 n0 + q1 n1 co-localise), DISTRIBUES sur
  # n_ranks(). Grossier replique -> masse par bloc + densite + potentiel bit-identiques cross-rang
  # (spread == 0). np=1 multi-box mono-rang, np=2/4 cross-rang MPI. Compile le runtime AmrSystem
  # comme test_mpi_amr_compiled_parity. Un binaire, trois entrees ctest (np=1/2/4 bit-identiques).
  add_executable(test_mpi_amr_twoblock_parity test_mpi_amr_twoblock_parity.cpp)
  target_link_libraries(test_mpi_amr_twoblock_parity PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_amr_twoblock_parity_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_amr_twoblock_parity>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- STRONG-SCALING AMR : grossier REPARTI (distribute_coarse=true) cable dans AmrSystem. Compare
  # le grossier multi-box reparti a l'oracle replique (meme binaire) : meme physique a l'arrondi,
  # cmax bit-identique cross-rang, masse conservee, MG convergent. Compile le runtime AmrSystem comme
  # test_mpi_amr_compiled_parity. Le script ROMEO diff cmax entre np=1/2/4 (bit-identique attendu).
  add_executable(test_mpi_amr_distributed_coarse test_mpi_amr_distributed_coarse.cpp)
  target_link_libraries(test_mpi_amr_distributed_coarse PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_amr_distributed_coarse_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_amr_distributed_coarse>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- PARITE MPI du REGRID D'UNION DES TAGS (T4 du design AMR_REGRID_UNION_TAGS_DESIGN.md, suivi
  # #199). Le SEUL chemin ou la reduction collective des tags (R4, all_reduce_or_inplace) est active :
  # grossier REPARTI + regrid actif. Le MEME scenario regrid (deux blocs ExB, union des tags densite par
  # bloc + tag phi sur |grad phi| cable depuis la facade) est relance en np=1/2/4 et la hierarchie fine
  # resultante doit etre IDENTIQUE : meme n_patches + densites/potentiel bit-identiques cross-rang
  # (spread == 0), masse par bloc conservee a travers les regrids. La CI diffe la ligne AMRREGRID entre
  # np=1/2/4 (layout fin d'union UNIQUE par rang). Compile le runtime AmrSystem comme les autres tests AMR.
  add_executable(test_amr_regrid_mpi_parity test_amr_regrid_mpi_parity.cpp)
  target_link_libraries(test_amr_regrid_mpi_parity PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_amr)
  foreach(np 1 2 4)
    add_test(NAME test_amr_regrid_mpi_parity_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_amr_regrid_mpi_parity>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- runtime System + Poisson + solve_fields() DISTRIBUE : verrou de non-regression du bug HOTE
  # fab(0) (#98). System repartit UNE box en round-robin, donc a np>1 un seul rang la possede ; les
  # autres ont local_size()==0. Avant le fix, le post-traitement par cellule de solve_fields()
  # (derivation phi/grad, apply_te, apply_epsilon*, apply_reaction) appelait fab(0) sans garde et
  # crashait cote hote sur les rangs sans box locale. Le solve elliptique reste COLLECTIF (tous les
  # rangs). np=1 couvre le chemin proprietaire (bit-identique au serie) ; np=2/4 le chemin distribue
  # ou des rangs sont vides. Compile le runtime System comme test_aux_te.
  add_executable(test_mpi_system_solve_fields test_mpi_system_solve_fields.cpp)
  target_link_libraries(test_mpi_system_solve_fields PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
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
  add_executable(test_mpi_system_fft test_mpi_system_fft.cpp)
  target_link_libraries(test_mpi_system_fft PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_system_fft_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_system_fft>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- source COUPLEE generique (System::add_coupled_source, DSL P5) sous MPI np=1/2/4 ---
  # La source generique s'applique par splitting (P->couplings) en iterant local_size() : MPI-safe
  # (no-op sur un rang sans box locale, contrairement aux couplages nommes qui dereferencent fab(0)).
  # On exerce une ionisation 3 especes codee en bytecode ; resultat INVARIANT au nombre de rangs (la box
  # vit sur rang 0). np=1 = chemin proprietaire ; np=2/4 = chemin distribue (rangs vides). Compile le
  # runtime System comme test_mpi_system_solve_fields.
  add_executable(test_mpi_coupled_source test_mpi_coupled_source.cpp)
  target_link_libraries(test_mpi_coupled_source PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_coupled_source_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_coupled_source>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- finding 8 : garde local_size avant fab(0) dans les fermetures rhs_into / advance / max_speed
  # des blocs DSL hote (compile via add_compiled_model). System repartit UNE box en round-robin,
  # donc a np>1 des rangs ont local_size()==0 ; avant le fix, rhs_into / advance / max_speed
  # appelaient copy_state / write_state -> fab(0) hors-bornes -> crash UB silencieux ou segfault.
  # Le fix ajoute "if (U.local_size() == 0) return;" en tete de chaque fermeture (meme pattern que
  # add_poisson). La garde est un no-op unilateral (pas de collectif MPI dans ces fermetures) ->
  # aucun interblocage. np=1 = chemin proprietaire (bit-identique au serie) ; np=2/4 = chemin
  # distribue (rangs vides). Resultat INVARIANT en np (box sur rang 0). Compile le runtime System
  # comme test_mpi_coupled_source.
  add_executable(test_mpi_system_gather_scatter test_mpi_system_gather_scatter.cpp)
  target_link_libraries(test_mpi_system_gather_scatter PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_system_gather_scatter_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_system_gather_scatter>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # --- ADC-257 : gather IO collectif (density_global / state_global / potential_global) sous MPI.
  # Ces accesseurs remplissent un buffer GLOBAL depuis les fabs locaux (aux indices globaux) puis
  # all_reduce_sum : ils sont la base MPI de sim.write / sim.checkpoint, mais n'etaient testes QUE
  # mono-rang (python/tests/test_io_multirank.py). La somme n'est correcte que si chaque cellule est
  # ecrite par EXACTEMENT un rang (mono-box -> rang 0) ; un double-comptage la multiplierait. np=1
  # = identite au serie ; np=2/4 = chemin distribue (rangs vides contribuant 0 a l'all_reduce). Le
  # test pose une densite connue, exige gather == reference bit-a-bit (np-invariant), gather == local
  # apres pas collectifs, et un aller-retour checkpoint/restart bit-identique. Compile le runtime
  # System comme test_mpi_system_gather_scatter.
  add_executable(test_mpi_system_io_gather test_mpi_system_io_gather.cpp)
  target_link_libraries(test_mpi_system_io_gather PRIVATE adc::adc ${CMAKE_DL_LIBS} adc_runtime_system)
  foreach(np 1 2 4)
    add_test(NAME test_mpi_system_io_gather_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_system_io_gather>
                     ${MPIEXEC_POSTFLAGS})
  endforeach()

  # Fin du bloc MPI : tout test ajoute depuis le snapshot ci-dessus est un test distribue. On lui
  # garantit le label "mpi" (selection `ctest -L mpi`) + un TIMEOUT 600 (un deadlock MPI echoue de
  # maniere ciblee au lieu de bloquer le runner). APPEND (et seulement si "mpi" manque) plutot que
  # set_tests_properties LABELS : ne JAMAIS ecraser un autre label deja pose ; TIMEOUT pose uniquement
  # s'il n'en existe pas, pour respecter un timeout choisi explicitement. list(FIND) plutot que
  # IN_LIST : pas de dependance a CMP0057.
  get_property(_adc_tests_after_mpi DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
  foreach(_t ${_adc_tests_after_mpi})
    list(FIND _adc_tests_before_mpi "${_t}" _adc_mpi_seen)
    if(_adc_mpi_seen EQUAL -1)
      get_test_property(${_t} LABELS _adc_mpi_labels)
      if(NOT _adc_mpi_labels MATCHES "(^|;)mpi(;|$)")
        set_property(TEST ${_t} APPEND PROPERTY LABELS "mpi")
      endif()
      get_test_property(${_t} TIMEOUT _adc_mpi_timeout)
      if(NOT _adc_mpi_timeout)
        set_property(TEST ${_t} PROPERTY TIMEOUT 600)
      endif()
    endif()
  endforeach()
endif()

# --- contrats communs de l'etage elliptique (audit D.1) : concepts EllipticOperator /
# LinearSolver / FieldPostProcessor formalises dans elliptic_interface.hpp. Le test est
# ESSENTIELLEMENT statique (static_assert que GeometricMG / TensorKrylovSolver / FFT /
# polaire conforment, ou NON, a chaque concept) ; additif, aucune classe modifiee, donc
# bit-identique. Ligne unique en fin de fichier pour eviter les conflits de fusion.
adc_add_test(test_elliptic_interface)

# --- Couverture AUTOMATIQUE du garde memoire (audit CI : 4 cibles capees a la main sur ~39) ----
# TOUTE cible de ce repertoire qui compile une TU lourde du runtime (python/system.cpp ou
# python/amr_system.cpp -- les deux instancient le produit combinatoire complet du dispatch,
# ~1-2 GB de RSS cc1plus a -O3 sous Kokkos) passe par adc_cap_opt_for_kokkos_ram : -O0 cible
# (correction inchangee, cf. justification en tete de fichier) + pool ninja adc_heavy_tu=1
# (jamais deux TU lourdes concurrentes). La boucle couvre les cibles FUTURES automatiquement :
# plus d'appel manuel a oublier. Benefice annexe : build des tests nettement plus rapide
# (TU lourde a -O0 ~ 40 s contre ~470 s a -O3).
get_property(_adc_dir_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
set(_adc_capped 0)
foreach(_t ${_adc_dir_targets})
  get_target_property(_t_sources ${_t} SOURCES)
  get_target_property(_t_libs ${_t} LINK_LIBRARIES)
  # Cap (-O0 + size-1 pool adc_heavy_tu) every target that EITHER compiles a heavy runtime TU directly
  # (SOURCES) OR links the adc_runtime_{system,amr} object libraries. The second clause is essential:
  # a target that links the object lib still compiles its OWN .cpp, and the parity tests instantiate
  # add_compiled_model (a heavy AMR/System closure build) right there. The object lib's PUBLIC -O0 caps
  # that .cpp, but the POOL is NOT a usage requirement, so without this it escapes adc_heavy_tu -> two
  # such heavy compiles run concurrently under --parallel 2 and OOM the ~7GB runner (exit 143). The pool,
  # not -O0 alone, is the structural anti-OOM guard (see the head-of-file note), so we re-attach it here.
  if(_t_sources MATCHES "python/(amr_)?system\.cpp" OR _t_libs MATCHES "adc_runtime_(system|amr)")
    adc_cap_opt_for_kokkos_ram(${_t})
    math(EXPR _adc_capped "${_adc_capped}+1")
  endif()
endforeach()
message(STATUS "adc tests: ${_adc_capped} cibles a TU lourde capees (-O0 + pool adc_heavy_tu)")

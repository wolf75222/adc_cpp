# cmake/PopsDevTooling.cmake
#
# Outillage QUALITE opt-in (OFF par defaut) : warnings stricts et sanitizers, portes par une cible
# INTERFACE `pops_dev_options` que SEULES les cibles internes lient en PRIVATE (les ~140 executables
# de tests/ via pops_add_test, et le module _pops). Le coeur public `pops::pops` (INTERFACE) n'est JAMAIS
# touche -> aucun flag ne fuit vers `adc_cases` ni vers un consommateur FetchContent / find_package.
#
# Active uniquement par `.github/workflows/quality.yml` (hebdomadaire), via les presets ci-warnings /
# ci-asan. Les builds locaux et le gate CI normal (ci.yml) n'en voient rien : la cible reste vide,
# le codegen est inchange.
#
# Inclus par le CMakeLists.txt racine APRES la resolution de Kokkos et AVANT add_subdirectory(tests)
# / add_subdirectory(python), pour que `pops_dev_options` existe quand ces cibles le lient.

option(POPS_ENABLE_WARNINGS   "Compile les cibles internes (tests + _pops) avec des warnings stricts" OFF)
option(POPS_ENABLE_SANITIZERS "Instrumente les cibles internes (tests + _pops) avec ASan + UBSan"     OFF)
option(POPS_ENABLE_COVERAGE   "Instrumente les cibles internes (tests + _pops) pour gcov/gcovr"       OFF)
option(POPS_ENABLE_TSAN       "Instrumente les cibles internes (tests + _pops) avec ThreadSanitizer (races)" OFF)

# Cible TOUJOURS definie (vide si les deux options sont OFF) : tests/ et python/ la lient
# inconditionnellement, sans garde `if(TARGET ...)`.
if(NOT TARGET pops_dev_options)
  add_library(pops_dev_options INTERFACE)
  add_library(pops::dev_options ALIAS pops_dev_options)
endif()

# --- Warnings stricts ----------------------------------------------------------------------------
# Informatif d'abord : PAS de -Werror. Kokkos / Eigen exposent leurs en-tetes en SYSTEM -> leurs
# propres warnings sont deja filtres, on ne voit que ceux du code pops + des tests.
if(POPS_ENABLE_WARNINGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")  # couvre GNU, Clang et AppleClang
    set(_pops_warn_flags
      -Wall -Wextra -Wpedantic
      -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual
      -Wcast-qual -Wdouble-promotion -Wformat=2
      -Wunused -Wnull-dereference -Wimplicit-fallthrough
      -Wmisleading-indentation)
    # -Wduplicated-cond / -Wduplicated-branches / -Wlogical-op sont specifiques a GCC : clang et
    # AppleClang les rejettent ("unknown warning option"). On ne les ajoute donc que sur GNU.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND _pops_warn_flags
        -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
    endif()
    # Volontairement EXCLUS au demarrage (trop bruyants sur du code numerique generique) :
    # -Wconversion, -Wsign-conversion, -Wold-style-cast. A reintroduire une fois la base assainie.
    target_compile_options(pops_dev_options INTERFACE
      "$<$<COMPILE_LANGUAGE:CXX>:${_pops_warn_flags}>")
  elseif(MSVC)
    target_compile_options(pops_dev_options INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:/W4>")
  endif()
  message(STATUS "pops: POPS_ENABLE_WARNINGS=ON -> warnings stricts sur les cibles internes "
                 "(informatif, pas de -Werror).")
endif()

# --- Sanitizers ASan + UBSan ---------------------------------------------------------------------
# Compile ET link doivent porter -fsanitize. Kokkos (lie comme bibliotheque) n'est PAS instrumente :
# lancer la suite avec ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 pour couper les faux
# positifs de frontiere instrumente / non instrumente (cf. quality.yml).
if(POPS_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_pops_san_flags
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
      -fno-sanitize-recover=undefined)
    target_compile_options(pops_dev_options INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${_pops_san_flags}>")
    target_link_options(pops_dev_options INTERFACE ${_pops_san_flags})
    message(STATUS "pops: POPS_ENABLE_SANITIZERS=ON -> ASan+UBSan sur les cibles internes "
                   "(Kokkos non instrumente ; voir ASAN_OPTIONS dans quality.yml).")
  else()
    message(WARNING "pops: POPS_ENABLE_SANITIZERS ignore (compilateur ${CMAKE_CXX_COMPILER_ID} "
                    "non supporte pour les sanitizers).")
  endif()
endif()

# --- Couverture (gcov) ---------------------------------------------------------------------------
# GCC uniquement (gcovr appelle le gcov assorti au compilateur). --coverage = -fprofile-arcs
# -ftest-coverage en compile ET en link. On instrumente les TU de tests : ce sont elles qui
# INSTANCIENT les en-tetes du coeur, donc c'est bien la couverture de include/pops/ qu'on mesure
# (le rapport gcovr du job coverage filtre sur include/pops/). Combine avec POPS_TESTS_FAST_O0
# (-O0, herite du preset ci-kokkos) : les compteurs de lignes ne sont pas brouilles par l'inlining.
if(POPS_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(pops_dev_options INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:--coverage>")
    target_link_options(pops_dev_options INTERFACE --coverage)
    message(STATUS "pops: POPS_ENABLE_COVERAGE=ON -> instrumentation gcov des cibles internes "
                   "(rapport : gcovr --filter include/pops/).")
  else()
    message(WARNING "pops: POPS_ENABLE_COVERAGE ignore (GCC requis, "
                    "compilateur = ${CMAKE_CXX_COMPILER_ID}).")
  endif()
endif()

# --- ThreadSanitizer (TSan) ----------------------------------------------------------------------
# Detection de data races sur le SEUL backend on-node multi-thread : Kokkos OpenMP (le gate ctest
# tourne en Serial, ou aucune race ne peut apparaitre). Compile ET link portent -fsanitize=thread.
# MUTUELLEMENT EXCLUSIF d'ASan : un binaire ne peut embarquer qu'un seul runtime memoire/thread ->
# erreur claire si les deux options sont ON (presets ci-tsan / ci-asan distincts). Chemin OBLIGATOIRE
# clang + libomp LLVM : libgomp (gcc) n'est PAS TSan-aware (tempete de faux positifs). Kokkos / libomp
# sont lies mais NON instrumentes -> faux positifs benins filtres par tsan-suppressions.txt (chaque
# entree justifiee). Voir le job 'tsan' de quality.yml (OMP_NUM_THREADS, TSAN_OPTIONS).
if(POPS_ENABLE_TSAN)
  if(POPS_ENABLE_SANITIZERS)
    message(FATAL_ERROR "pops: POPS_ENABLE_TSAN et POPS_ENABLE_SANITIZERS (ASan) sont mutuellement "
                        "exclusifs (un binaire ne porte qu'un runtime). Choisir le preset ci-tsan "
                        "OU ci-asan, pas les deux.")
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_pops_tsan_flags -fsanitize=thread -fno-omit-frame-pointer)
    target_compile_options(pops_dev_options INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${_pops_tsan_flags}>")
    target_link_options(pops_dev_options INTERFACE ${_pops_tsan_flags})
    message(STATUS "pops: POPS_ENABLE_TSAN=ON -> TSan sur les cibles internes "
                   "(clang + libomp requis ; Kokkos non instrumente, voir tsan-suppressions.txt).")
  else()
    message(WARNING "pops: POPS_ENABLE_TSAN ignore (compilateur ${CMAKE_CXX_COMPILER_ID} "
                    "non supporte pour TSan).")
  endif()
endif()

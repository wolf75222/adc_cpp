# Harnais libFuzzer (option POPS_BUILD_FUZZING, OFF par defaut -- construit par le job hebdo
# `fuzz` de quality.yml, preset ci-fuzz). Cibles : composants PURS du coeur (Box2D, clustering
# Berger-Rigoutsos, real_eig_minmax) -- pas de Kokkos au runtime, pas d'initialisation : chaque
# harnais est une fonction pure octets -> invariants, ideale pour libFuzzer.
#
# Exige le clang LLVM : libFuzzer (-fsanitize=fuzzer) n'est livre ni par GCC ni par AppleClang
# (CMAKE_CXX_COMPILER_ID "AppleClang" est donc REFUSE aussi : son runtime fuzzer n'existe pas).
# Sous un autre compilateur on AVERTIT et on n'ajoute aucune cible -- le configure reste vert,
# une option non disponible localement ne doit pas casser le reste du build.

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(WARNING "pops: POPS_BUILD_FUZZING exige le clang LLVM (libFuzzer) ; "
                  "compilateur = ${CMAKE_CXX_COMPILER_ID} -> aucune cible fuzz construite.")
  return()
endif()

# fuzzer = moteur libFuzzer ; address,undefined = oracles memoire/UB PENDANT le fuzzing
# (un harnais sans sanitizer ne detecte que les aborts explicites des invariants).
# POPS_FUZZ_SANITIZERS surchargeable : sur macOS, l'ASan du LLVM Homebrew DEADLOCKE avant main
# (init re-entrante : InitializeShadowMemory -> get_dyld_hdr -> Block_copy -> malloc -> re-entre
# AsanInit -> spinlock ; verifie par `sample`, MallocNanoZone=0 ne suffit PAS). En local Mac :
# -DPOPS_FUZZ_SANITIZERS=undefined (fuzzer+UBSan, pas d'ASan). Le CI Linux garde le defaut complet.
set(POPS_FUZZ_SANITIZERS "address,undefined" CACHE STRING
    "Sanitizers combines au moteur libFuzzer (defaut address,undefined ; 'undefined' seul sur macOS)")
if(POPS_FUZZ_SANITIZERS)
  set(_pops_fuzz_san "fuzzer,${POPS_FUZZ_SANITIZERS}")
else()
  set(_pops_fuzz_san "fuzzer")
endif()
set(POPS_FUZZ_FLAGS -fsanitize=${_pops_fuzz_san} -fno-omit-frame-pointer
                   -fno-sanitize-recover=undefined)
message(STATUS "pops: fuzz/ -> -fsanitize=${_pops_fuzz_san}")

function(pops_add_fuzzer name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE pops::pops)
  target_compile_options(${name} PRIVATE ${POPS_FUZZ_FLAGS})
  target_link_options(${name} PRIVATE ${POPS_FUZZ_FLAGS})
endfunction()

pops_add_fuzzer(fuzz_box2d)
pops_add_fuzzer(fuzz_cluster)
pops_add_fuzzer(fuzz_dense_eig)

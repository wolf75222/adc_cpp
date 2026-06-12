# cmake/AdcDevTooling.cmake
#
# Outillage QUALITE opt-in (OFF par defaut) : warnings stricts et sanitizers, portes par une cible
# INTERFACE `adc_dev_options` que SEULES les cibles internes lient en PRIVATE (les ~140 executables
# de tests/ via adc_add_test, et le module _adc). Le coeur public `adc::adc` (INTERFACE) n'est JAMAIS
# touche -> aucun flag ne fuit vers `adc_cases` ni vers un consommateur FetchContent / find_package.
#
# Active uniquement par `.github/workflows/quality.yml` (hebdomadaire), via les presets ci-warnings /
# ci-asan. Les builds locaux et le gate CI normal (ci.yml) n'en voient rien : la cible reste vide,
# le codegen est inchange.
#
# Inclus par le CMakeLists.txt racine APRES la resolution de Kokkos et AVANT add_subdirectory(tests)
# / add_subdirectory(python), pour que `adc_dev_options` existe quand ces cibles le lient.

option(ADC_ENABLE_WARNINGS   "Compile les cibles internes (tests + _adc) avec des warnings stricts" OFF)
option(ADC_ENABLE_SANITIZERS "Instrumente les cibles internes (tests + _adc) avec ASan + UBSan"     OFF)

# Cible TOUJOURS definie (vide si les deux options sont OFF) : tests/ et python/ la lient
# inconditionnellement, sans garde `if(TARGET ...)`.
if(NOT TARGET adc_dev_options)
  add_library(adc_dev_options INTERFACE)
  add_library(adc::dev_options ALIAS adc_dev_options)
endif()

# --- Warnings stricts ----------------------------------------------------------------------------
# Informatif d'abord : PAS de -Werror. Kokkos / Eigen exposent leurs en-tetes en SYSTEM -> leurs
# propres warnings sont deja filtres, on ne voit que ceux du code adc + des tests.
if(ADC_ENABLE_WARNINGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")  # couvre GNU, Clang et AppleClang
    set(_adc_warn_flags
      -Wall -Wextra -Wpedantic
      -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual
      -Wcast-qual -Wdouble-promotion -Wformat=2
      -Wunused -Wnull-dereference -Wimplicit-fallthrough)
    # Volontairement EXCLUS au demarrage (trop bruyants sur du code numerique generique) :
    # -Wconversion, -Wsign-conversion, -Wold-style-cast. A reintroduire une fois la base assainie.
    target_compile_options(adc_dev_options INTERFACE
      "$<$<COMPILE_LANGUAGE:CXX>:${_adc_warn_flags}>")
  elseif(MSVC)
    target_compile_options(adc_dev_options INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:/W4>")
  endif()
  message(STATUS "adc: ADC_ENABLE_WARNINGS=ON -> warnings stricts sur les cibles internes "
                 "(informatif, pas de -Werror).")
endif()

# --- Sanitizers ASan + UBSan ---------------------------------------------------------------------
# Compile ET link doivent porter -fsanitize. Kokkos (lie comme bibliotheque) n'est PAS instrumente :
# lancer la suite avec ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 pour couper les faux
# positifs de frontiere instrumente / non instrumente (cf. quality.yml).
if(ADC_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_adc_san_flags
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
      -fno-sanitize-recover=undefined)
    target_compile_options(adc_dev_options INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${_adc_san_flags}>")
    target_link_options(adc_dev_options INTERFACE ${_adc_san_flags})
    message(STATUS "adc: ADC_ENABLE_SANITIZERS=ON -> ASan+UBSan sur les cibles internes "
                   "(Kokkos non instrumente ; voir ASAN_OPTIONS dans quality.yml).")
  else()
    message(WARNING "adc: ADC_ENABLE_SANITIZERS ignore (compilateur ${CMAKE_CXX_COMPILER_ID} "
                    "non supporte pour les sanitizers).")
  endif()
endif()

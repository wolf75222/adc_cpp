#pragma once

#include <adc/runtime/export.hpp>  // ADC_EXPORT (visibilite defaut a travers le module _adc hidden)

#include <string>

/// @file
/// @brief Cle d'ABI du coeur adc : chaine stable identifiant la combinaison (compilateur,
///        standard C++, signature de l'arbre d'en-tetes) avec laquelle une unite a ete compilee.
///
/// MOTIVATION (chemin DSL "production"). Un loader .so genere par le DSL (cf.
/// dsl.emit_cpp_native_loader) inline le gabarit en-tete adc::add_compiled_model et appelle des
/// methodes hors-ligne de adc::System DEFINIES dans le module _adc deja charge. Le loader et le
/// module DOIVENT partager la meme ABI (memes en-tetes, meme compilateur, meme standard), sinon
/// l'agencement memoire des objets traversant la frontiere (System, GridContext, BlockClosures...)
/// diverge -> comportement indefini SILENCIEUX. On rend l'incompatibilite EXPLICITE : le loader
/// expose adc_native_abi_key() (cle figee a SA compilation) et le System compare a SA propre
/// abi_key() au chargement (add_native_block) ; un ecart leve une erreur claire au lieu d'un UB.
///
/// Construction de la cle (parallele a adc_cases/common/native.py::_abi_key) :
///   - __VERSION__   : identite + version du compilateur (g++/clang++/Apple clang...) ;
///   - __cplusplus   : standard C++ effectif (donc -std= et le mode du compilateur) ;
///   - ADC_HEADER_SIG: signature de l'arbre d'en-tetes du coeur, INJECTEE par le build (CMake cote
///                     module, flag -D cote loader) ; sa valeur change si un en-tete change, donc
///                     la cle change et l'incompatibilite est detectee. Absente (vieux build / build
///                     manuel) -> jeton litteral "unknown" : la cle reste stable et comparable, elle
///                     ne capture alors que compilateur + standard (degradation gracieuse, jamais UB
///                     silencieux car les deux cotes voient le meme "unknown" s'ils sont batis pareil).
///   - kokkos=0|1    : ADC_HAS_KOKKOS de l'unite. Ce macro CHANGE le layout de types traverses a la
///                     frontiere (allocator.hpp : backing Fab malloc vs pool Kokkos::SharedSpace ;
///                     types.hpp : ADC_HD) : un loader serie sur un module Kokkos (ou l'inverse)
///                     etait auparavant ACCEPTE par la cle -> fallback serie muet dans un sens, UB
///                     dans l'autre. Desormais la divergence est rejetee explicitement ; la parite se
///                     retablit via ADC_KOKKOS_ROOT (cf. dsl._native_kokkos_flags) ou un module serie.
///   - stdlib=...    : bibliotheque standard C++ liee (libc++ _LIBCPP_VERSION / libstdc++ __GLIBCXX__).
///                     Deux toolchains peuvent partager __VERSION__ ET __cplusplus mais lier des
///                     stdlib aux ABI INCOMPATIBLES (std::string/std::function traversent la
///                     frontiere du loader). Detecte le panachage clang -stdlib=libc++ vs libstdc++
///                     (mix conda/systeme). Les macros sont visibles ici car <string> est inclus.

#ifndef ADC_HEADER_SIG
#define ADC_HEADER_SIG "unknown"
#endif

// Indirection pour stringifier la valeur d'une macro (et non son nom). Definie AVANT les jetons
// qui l'utilisent (l'expansion se fait a l'usage, mais autant rester lisible).
#define ADC_ABI_STR_(x) #x
#define ADC_ABI_STR(x) ADC_ABI_STR_(x)

// Jeton Kokkos : evalue PAR UNITE (module _adc d'un cote, loader .so genere de l'autre).
#ifdef ADC_HAS_KOKKOS
#define ADC_ABI_KOKKOS "1"
#else
#define ADC_ABI_KOKKOS "0"
#endif

// Jeton stdlib : identite + version de la bibliotheque standard liee. _LIBCPP_VERSION /
// __GLIBCXX__ ne sont definis qu'apres inclusion d'un en-tete de la stdlib (<string> ci-dessus).
#if defined(_LIBCPP_VERSION)
#define ADC_ABI_STDLIB "libc++_" ADC_ABI_STR(_LIBCPP_VERSION)
#elif defined(__GLIBCXX__)
#define ADC_ABI_STDLIB "libstdc++_" ADC_ABI_STR(__GLIBCXX__)
#else
#define ADC_ABI_STDLIB "unknown"
#endif

// Cle d'ABI de l'UNITE DE TRADUCTION courante, en LITTERAL pur concatene par le preprocesseur :
// "compiler=<__VERSION__>;std=<__cplusplus>;headers=<ADC_HEADER_SIG>;kokkos=<0|1>;stdlib=<...>".
// Tous les jetons sont des litteraux de chaine (__VERSION__ et ADC_HEADER_SIG en sont deja), donc
// la cle est figee dans le .rodata de CHAQUE TU au preprocessing -- AUCUN appel de fonction.
//
// POURQUOI un litteral et pas une fonction inline (bug CI reel, ELF/Linux) : un loader .so est
// charge RTLD_GLOBAL et une fonction `inline` (liaison faible, visibilite par defaut) qui
// fabriquerait la cle serait INTERPOSEE par l'editeur de liens dynamique vers la copie du module
// deja charge -- le loader renverrait alors la cle DU MODULE et le garde-fou comparerait la cle du
// module a elle-meme (tautologie : ABI jamais rejetee). Que l'interposition se produise dependait
// du seuil d'inlining du compilateur (la fonction courte etait inlinee -> cle locale correcte ;
// allongee par kokkos=/stdlib=, gcc -O2 a cesse de l'inliner -> interposition -> garde neutralise,
// attrape par test_amr_native_loader). Un litteral ne traverse aucun symbole : pas d'interposition.
// NB : les parsers (dsl._adc_cxx_std_from_module / module_header_signature) scannent par prefixe
// de jeton ("std=", "headers=") -> insensibles a l'AJOUT de jetons en queue.
#define ADC_ABI_KEY_LITERAL                                                          \
  "compiler=" __VERSION__ ";std=" ADC_ABI_STR(__cplusplus) ";headers=" ADC_HEADER_SIG \
  ";kokkos=" ADC_ABI_KOKKOS ";stdlib=" ADC_ABI_STDLIB

namespace adc {
namespace detail {

/// Cle d'ABI de la TU courante (cf. ADC_ABI_KEY_LITERAL). Conservee pour le cote MODULE
/// (abi_key() hors-ligne dans system.cpp) ; un LOADER genere doit renvoyer ADC_ABI_KEY_LITERAL
/// directement (litteral local a sa TU, insensible a l'interposition ELF -- cf. ci-dessus).
inline std::string abi_key_string() { return ADC_ABI_KEY_LITERAL; }

}  // namespace detail

/// Cle d'ABI du module (TU system.cpp). ADC_EXPORT : exportee pour que add_native_block puisse lire
/// la cle du module deja charge et la comparer a celle baked dans le loader .so. Definie hors-ligne
/// (system.cpp) pour figer la cle a la compilation DU MODULE.
ADC_EXPORT std::string abi_key();

}  // namespace adc

#pragma once

#include <adc/core/types.hpp>  // Real, ADC_HD (device-callable)

#include <cstddef>

/// @file
/// @brief RuntimeParams : porteur de PARAMETRES RUNTIME d'un modele DSL (P7-b). Un parametre runtime
///        est declare cote Python via adc.dsl.Param(..., kind="runtime") ; sa valeur peut etre CHANGEE
///        a l'execution SANS recompiler le .so. Les parametres CONST (kind="const"), eux, restent
///        inlines EN DUR dans le .so au codegen (bit-identiques a l'historique : ce chantier ne touche
///        PAS leur chemin).
///
/// MECANIQUE. Le codegen (python/adc/dsl.py) attribue a chaque parametre runtime un INDICE STABLE
/// (ordre trie des noms) et emet, la ou la formule lit ce parametre, `params.get(<indice>)` au lieu
/// d'une constante litterale. Chaque brique generee (hyperbolique / source / elliptique) qui LIT au
/// moins un parametre runtime porte alors un membre `adc::RuntimeParams params{}` initialise a la
/// valeur de DECLARATION (donc, sans appel set au runtime, le bloc se comporte EXACTEMENT comme avec
/// un param const : la valeur de declaration est cuite dans le defaut du membre). Au runtime, l'ABI du
/// .so AOT (compiled_block_abi.hpp) transporte un bloc plat de doubles ; chaque appel reconstruit le
/// modele puis ECRASE `params.values[k]` par la valeur fournie -> le comportement change sans
/// recompilation.
///
/// DEVICE-CLEAN. RuntimeParams est un agregat trivialement copiable (tableau de Real par valeur,
/// aucune allocation, aucun std::), donc copiable sur device et lisible dans un kernel : get() est
/// ADC_HD. La taille est FIXE (kMaxRuntimeParams) pour rester sans allocation ; un modele depassant
/// cette borne est rejete cote Python (codegen), jamais ici.

namespace adc {

/// Nombre MAXIMAL de parametres runtime par bloc DSL. Borne volontairement large (un modele physique
/// raisonnable en a quelques-uns) ; le depassement est diagnostique cote Python au codegen. Garde la
/// structure de taille fixe (pas d'allocation -> device-copiable par valeur).
inline constexpr int kMaxRuntimeParams = 32;

/// Porteur PLAT (taille fixe, par valeur) des valeurs des parametres runtime d'un bloc. `count` = nb de
/// parametres effectivement declares par le modele ; `values[k]` = valeur courante du parametre
/// d'indice k (indice attribue par le codegen, ordre trie des noms). Les indices >= count sont nuls et
/// jamais lus par les briques generees. Agregat trivial : copiable sur device sans cout.
struct RuntimeParams {
  int count = 0;
  Real values[kMaxRuntimeParams] = {};

  /// Valeur du parametre runtime d'indice @p k. Pas de borne dynamique (l'indice est emis par le
  /// codegen, donc statiquement < count) : lecture directe, device-callable.
  ADC_HD Real get(int k) const { return values[k]; }
};

}  // namespace adc

// (P7-b) params runtime DSL : voir test_dsl_runtime_params.py

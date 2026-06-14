#pragma once

#include <string>
#include <vector>

/// @file
/// @brief PODs d'OPTIONS des facades publiques (System / AmrSystem), regroupant les longues familles
///        de parametres HOMOGENES qui posaient un footgun d'ordre (C++ Core Guidelines I.23).
///
/// Couche : `include/adc/runtime`.
/// Role : porter, en un seul agregat nomme, les reglages d'un etage source condense par Schur et la
///   description bytecode d'une source couplee inter-especes. Ces familles etaient auparavant des
///   listes plates de parametres du MEME type (plusieurs `std::string` adjacents, plusieurs
///   `std::vector<int>` paralleles) -- intervertibles silencieusement a l'appel. Les regrouper en POD
///   nomme rend l'appel auto-documente (initialiseurs designes) et supprime le risque d'inversion.
/// Contrat : POD plat traversant les bindings sans friction. Les DEFAUTS in-class reproduisent
///   EXACTEMENT les anciens defauts des parametres a plat -> aucun changement de comportement.
///
/// Invariants :
/// - chaque champ garde le nom, le type et le defaut de l'ancien parametre a plat homonyme ;
/// - ces PODs vivent AU-DESSUS de la couche ABI (compiled_block_abi.hpp / native_loader.hpp) : ils ne
///   traversent jamais la frontiere extern "C" d'un loader .so. L'ABI SEMANTIQUE extern "C"
///   (residual / advance, structs traversant le loader) reste donc INCHANGEE. En revanche le LITTERAL
///   abi_key() CHANGE : il embarque le jeton headers=ADC_HEADER_SIG (sha256 conservateur du chemin et
///   du contenu de CHAQUE en-tete sous include/, cf. abi_key.hpp et python/CMakeLists.txt) ; le seul
///   fait d'AJOUTER cet en-tete et d'EDITER system.hpp / amr_system.hpp deplace ADC_HEADER_SIG. C'est
///   ATTENDU et inoffensif : aucune ABI semantique ne change, mais add_native_block rejettera les .so
///   AOT generes avant ce changement (signature divergente) -> une regeneration unique des .so perimes.

namespace adc {

/// @brief Reglages de l'ETAGE SOURCE condense par Schur (cf. System::set_source_stage /
///        AmrSystem::set_source_stage). Regroupe le solve Krylov et les DESCRIPTEURS de champs --
///        une famille de quatre `std::string` adjacents qui etaient intervertibles a l'appel.
///
/// Usage : construit par la facade (ou par les bindings, a partir des kwargs Python a plat) puis
///   passe a set_source_stage. Tous defauts (POD vide) = comportement historique bit-identique.
/// Contrat : les DEFAUTS reproduisent les anciens defauts des parametres a plat.
///  - krylov_tol / krylov_max_iters : tolerance et budget du solve Krylov (BiCGStab) de l'etage.
///    <= 0 (defaut) = constantes historiques du stepper (1e-10 ; 400 cartesien, 600 polaire).
///  - density / momentum_x / momentum_y / energy : DESCRIPTEURS des champs de l'etage. Chaine VIDE
///    (defaut) = role canonique (Density / MomentumX / MomentumY / Energy optionnel), bit-identique.
///    Sinon : un NOM DE ROLE stable ("density", "momentum_x", ...) ou un NOM DE VARIABLE du bloc.
///    energy == "none" desactive la mise a jour d'energie.
///  - bz_aux_component : composante du canal aux lue comme champ magnetique Omega. < 0 (defaut) =
///    canal canonique B_z (kAuxBaseComps), bit-identique. (Ignore par AmrSystem : etage mono-bloc
///    sur le canal canonique.)
struct SourceStageOptions {
  double krylov_tol = 0.0;
  int krylov_max_iters = 0;
  std::string density = "";
  std::string momentum_x = "";
  std::string momentum_y = "";
  std::string energy = "";
  int bz_aux_component = -1;
};

/// @brief Description BYTECODE d'une SOURCE COUPLEE generique inter-especes (cf.
///        System::add_coupled_source / AmrSystem::add_coupled_source). Regroupe les tableaux PLATS
///        de l'ABI bytecode -- six `std::vector` (quatre de descripteurs bloc/role, deux+ de programme
///        machine a pile) intervertibles a l'appel -- en un seul agregat nomme.
///
/// Usage : construit par la facade (ou par les bindings, a partir des kwargs Python a plat) puis passe
///   a add_coupled_source avec la frequence et le label restes a plat (un double et une chaine, types
///   distincts, hors footgun homogene). Une forme mal formee leve une erreur EXPLICITE a l'ajout.
/// Contrat : ABI PLATE -- aucun objet C++ ne traverse la frontiere ; ce POD n'est qu'un porteur de
///   tableaux cote facade. Les DEFAUTS reproduisent les anciens defauts des parametres a plat (les
///   programmes de frequence par cellule VIDES = frequence constante seule, bit-identique).
///  - in_blocks / in_roles : blocs lus en entree et leurs roles (un par registre d'entree).
///  - consts : constantes (.param()), chargees apres les entrees.
///  - out_blocks / out_roles : bloc cible et role cible de chaque terme de source.
///  - prog_ops / prog_args : opcodes concatenes de TOUS les termes (machine a pile) et leurs arguments
///    paralleles (indice de registre pour PushReg).
///  - prog_lens : longueur du programme de chaque terme (segmente prog_ops / prog_args dans l'ordre).
///  - freq_prog_ops / freq_prog_args : programme OPTIONNEL d'une frequence PAR CELLULE mu(U) (meme
///    machine a pile, MEME table de registres). VIDES (defaut) = frequence constante seule.
struct CoupledSourceProgram {
  std::vector<std::string> in_blocks;
  std::vector<std::string> in_roles;
  std::vector<double> consts;
  std::vector<std::string> out_blocks;
  std::vector<std::string> out_roles;
  std::vector<int> prog_ops;
  std::vector<int> prog_args;
  std::vector<int> prog_lens;
  std::vector<int> freq_prog_ops;
  std::vector<int> freq_prog_args;
};

}  // namespace adc

#pragma once

#include <string>
#include <vector>

/// @file
/// @brief Descripteur des variables d'un modele (Vars). Porte par la brique HYPERBOLIQUE (avec le
///        flux et les conversions), car variables et flux sont physiquement lies ; ce n'est PAS une
///        brique independante combinable librement.
///
/// `Variables` DECRIT les variables (conservatives ou primitives) : nature, noms, taille. C'est une
/// metadonnee HOTE (elle ne pilote pas le calcul, qui travaille par composante via les conversions
/// cons<->prim), mais c'est un CONTRAT OBLIGATOIRE du modele hyperbolique (concept HyperbolicModel) :
/// conservative_vars() et primitive_vars(). Sert a l'introspection, aux diagnostics nommes, a la
/// sortie labellisee.

namespace adc {

enum class VariableKind { Conservative, Primitive };

/// Role PHYSIQUE d'une composante. Permet d'adresser une composante par son SENS
/// (index_of(MomentumX)) plutot que par un indice magique u[1] : une source couplee peut viser
/// "la quantite de mouvement de telle espece" sans coder en dur l'indice. Custom = role non renseigne.
enum class VariableRole {
  Density, MomentumX, MomentumY, MomentumZ, Energy,
  VelocityX, VelocityY, VelocityZ, Pressure, Temperature, Scalar, Custom
};

/// Une variable : nom, role physique, indice de composante dans l'etat.
struct Variable {
  std::string name;
  VariableRole role;
  int component;
};

/// Jeu de variables d'un modele : nature (cons/prim), noms, taille, et roles (optionnels, paralleles
/// a `names` ; absents -> Custom). Les appels existants `{kind, names, size}` restent valides (roles
/// vides). index_of(role) donne l'indice de la composante portant ce role (-1 si absent).
struct VariableSet {
  VariableKind kind;
  std::vector<std::string> names;
  int size;
  std::vector<VariableRole> roles{};  ///< parallele a `names` ; vide = roles non renseignes

  /// Indice de la composante portant @p role (premiere occurrence), -1 si absent.
  int index_of(VariableRole role) const {
    for (int i = 0; i < static_cast<int>(roles.size()); ++i)
      if (roles[i] == role) return i;
    return -1;
  }
  /// Descripteur complet de la composante @p i (role Custom si non renseigne).
  Variable at(int i) const {
    return {names[i], i < static_cast<int>(roles.size()) ? roles[i] : VariableRole::Custom, i};
  }
};

/// Ancien nom (compat) : VariableSet etait `Variables`. Conserve pour le code existant et genere.
using Variables = VariableSet;

}  // namespace adc

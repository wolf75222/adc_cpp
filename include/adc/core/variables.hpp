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

/// Nom lisible d'un role (introspection, binding Python). Stable : sert de cle cote application.
inline const char* role_name(VariableRole r) {
  switch (r) {
    case VariableRole::Density:     return "density";
    case VariableRole::MomentumX:   return "momentum_x";
    case VariableRole::MomentumY:   return "momentum_y";
    case VariableRole::MomentumZ:   return "momentum_z";
    case VariableRole::Energy:      return "energy";
    case VariableRole::VelocityX:   return "velocity_x";
    case VariableRole::VelocityY:   return "velocity_y";
    case VariableRole::VelocityZ:   return "velocity_z";
    case VariableRole::Pressure:    return "pressure";
    case VariableRole::Temperature: return "temperature";
    case VariableRole::Scalar:      return "scalar";
    case VariableRole::Custom:      return "custom";
  }
  return "custom";
}

/// Inverse de role_name : role physique a partir de son nom stable (Custom si inconnu). Sert a
/// reconstruire un VariableSet a roles a partir d'une metadonnee TEXTE (p.ex. la chaine portee par
/// un .so compile / dynamique : l'ABI extern "C" ne transporte que des chaines, pas l'enum).
inline VariableRole role_from_name(const std::string& s) {
  if (s == "density")     return VariableRole::Density;
  if (s == "momentum_x")  return VariableRole::MomentumX;
  if (s == "momentum_y")  return VariableRole::MomentumY;
  if (s == "momentum_z")  return VariableRole::MomentumZ;
  if (s == "energy")      return VariableRole::Energy;
  if (s == "velocity_x")  return VariableRole::VelocityX;
  if (s == "velocity_y")  return VariableRole::VelocityY;
  if (s == "velocity_z")  return VariableRole::VelocityZ;
  if (s == "pressure")    return VariableRole::Pressure;
  if (s == "temperature") return VariableRole::Temperature;
  if (s == "scalar")      return VariableRole::Scalar;
  return VariableRole::Custom;
}

/// CSV des noms d'un VariableSet (separateur ','). Brique de la metadonnee TEXTE qu'un .so genere
/// expose : l'ABI extern "C" ne transporte pas d'objet C++, on serialise donc en chaine.
inline std::string names_csv(const VariableSet& vs) {
  std::string s;
  for (std::size_t i = 0; i < vs.names.size(); ++i) {
    if (i) s += ',';
    s += vs.names[i];
  }
  return s;
}

/// CSV des roles d'un VariableSet (role_name, separateur ','). VIDE si le modele ne renseigne pas
/// ses roles (vs.roles vide) : le consommateur retombe alors sur le fallback indices (retro-compat).
inline std::string roles_csv(const VariableSet& vs) {
  std::string s;
  for (std::size_t i = 0; i < vs.roles.size(); ++i) {
    if (i) s += ',';
    s += role_name(vs.roles[i]);
  }
  return s;
}

/// Metadonnee "noms" d'un modele : "cons_csv|prim_csv" (separateur '|' entre les deux jeux). Lue
/// telle quelle par le consommateur (System) via le symbole optionnel adc_compiled_var_names.
template <class Model>
std::string var_names_meta() {
  return names_csv(Model::conservative_vars()) + "|" + names_csv(Model::primitive_vars());
}

/// Metadonnee "roles" d'un modele : "cons_roles_csv|prim_roles_csv" (cote vide = roles non renseignes).
template <class Model>
std::string roles_meta() {
  return roles_csv(Model::conservative_vars()) + "|" + roles_csv(Model::primitive_vars());
}

/// Ancien nom (compat) : VariableSet etait `Variables`. Conserve pour le code existant et genere.
using Variables = VariableSet;

}  // namespace adc

/// Exporte les metadonnees OPTIONNELLES "noms + roles" d'un bloc .so via des symboles extern "C" lus
/// par dlsym cote System. PARTAGE par les deux backends generes (AOT compiled_block et JIT
/// dynamic_model) : la metadonnee perdue (noms/roles) est transportee sans rompre l'ABI plate.
/// RETRO-COMPATIBLE : un .so ne definissant pas ces symboles (genere avant ce chantier) reste valide
/// -- le System ne trouve pas le symbole et retombe sur son fallback (noms u0.., pas de roles).
/// @p MODEL = type du modele (porte conservative_vars / primitive_vars).
#define ADC_EXPORT_BLOCK_METADATA(MODEL)                                         \
  extern "C" const char* adc_compiled_var_names() {                             \
    static const std::string s = adc::var_names_meta<MODEL>();                  \
    return s.c_str();                                                           \
  }                                                                             \
  extern "C" const char* adc_compiled_roles() {                                 \
    static const std::string s = adc::roles_meta<MODEL>();                      \
    return s.c_str();                                                           \
  }

/// Exporte le gamma (indice adiabatique) du bloc via le symbole optionnel adc_compiled_gamma, lu par
/// les couplages inter-especes du System (collision, echange thermique, T_e). EMIS SEULEMENT si le
/// modele declare un gamma : sinon le symbole reste absent et le System garde son defaut 1.4.
#define ADC_EXPORT_BLOCK_GAMMA(GAMMA) \
  extern "C" double adc_compiled_gamma() { return (GAMMA); }

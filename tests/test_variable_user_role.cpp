// ADC-292 : roles utilisateurs NOMMES + fin des fallbacks canoniques SILENCIEUX.
//
// Verrouille la couche de role string-keyee ajoutee a VariableSet (parallele a l'enum VariableRole,
// label porte par user_roles) et le garde-fou des couplages nommes (coupling_role_index) :
//   (1) un ROLE UTILISATEUR nomme se resout par son label (index_of(string)) -- plus d'ambiguite de
//       premiere-occurrence Custom ;
//   (2) un nom de role CANONIQUE se resout toujours (par nom et par enum), meme en layout NON canonique ;
//   (3) un role/label ABSENT renvoie -1 ;
//   (4) roles_csv emet le label utilisateur et parse_roles_into le reconstruit (aller-retour ABI .so),
//       en restant bit-identique pour un jeu purement canonique (user_roles vide) ;
//   (5) coupling_role_index : bloc SANS roles -> fallback canonique conserve (compat) ; bloc AVEC roles
//       mais sans le role requis -> LEVE en nommant le bloc et le role (plus aucun repli silencieux).
//
// Test PUR (n'inclut que core/variables.hpp) : aucune dependance runtime, lie adc::adc seul.
#include <adc/core/variables.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

using R = adc::VariableRole;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    std::printf("  [%s] %s\n", c ? "OK " : "XX ", w);
    if (!c) ++fails;
  };

  // --- (1)+(2)+(3) index_of(string) : layer role canonique + role utilisateur, layout NON canonique ---
  // Bloc fictif : momentum_x en comp 0, un champ utilisateur "phi" en comp 1, densite en comp 2 (layout
  // NON canonique). roles porte l'enum (Custom pour "phi"), user_roles porte le label parallele.
  adc::VariableSet vs;
  vs.kind = adc::VariableKind::Conservative;
  vs.names = {"mx", "phi", "rho"};
  vs.size = 3;
  vs.roles = {R::MomentumX, R::Custom, R::Density};
  vs.user_roles = {"", "phi", ""};

  chk(vs.index_of("density") == 2, "index_of_string:canonical_name_non_canonical_layout");   // (2)
  chk(vs.index_of(R::Density) == 2, "index_of_enum:canonical_non_canonical_layout");          // (2)
  chk(vs.index_of("momentum_x") == 0, "index_of_string:canonical_name_resolves");
  chk(vs.index_of("phi") == 1, "index_of_string:user_label_resolves");                        // (1)
  chk(vs.index_of("energy") == -1, "index_of_string:absent_canonical_name_is_minus1");        // (3)
  chk(vs.index_of("psi") == -1, "index_of_string:absent_user_label_is_minus1");               // (3)
  // An EMPTY role string is never a valid target: on this MIXED block user_roles[0] == "" (the
  // canonical momentum_x slot), so without a guard index_of("") would wrongly resolve to component 0
  // -- exactly the silent fallback ADC-292 kills. It must return -1.
  chk(vs.index_of("") == -1, "index_of_string:empty_role_is_minus1");                          // (3)

  // --- (4) aller-retour CSV : roles_csv emet le label utilisateur, parse_roles_into le reconstruit ----
  const std::string csv = adc::roles_csv(vs);
  chk(csv == "momentum_x,phi,density", "roles_csv:emits_user_label");
  adc::VariableSet rt;
  rt.kind = adc::VariableKind::Conservative;
  rt.names = vs.names;
  rt.size = vs.size;
  adc::parse_roles_into(rt, csv);
  chk(rt.index_of("phi") == 1, "parse_roles_into:roundtrips_user_label");
  chk(rt.index_of(R::Density) == 2, "parse_roles_into:roundtrips_canonical");
  chk(rt.roles.size() == 3 && rt.roles[1] == R::Custom, "parse_roles_into:custom_enum_for_user_label");

  // Jeu PUREMENT canonique : user_roles reste vide (bit-identique au comportement historique, aucune
  // regression sur les blocs existants ni sur l'ABI .so des blocs sans role utilisateur).
  adc::VariableSet canon;
  canon.kind = adc::VariableKind::Conservative;
  adc::parse_roles_into(canon, "density,momentum_x,energy");
  chk(canon.user_roles.empty(), "parse_roles_into:canonical_csv_leaves_user_roles_empty");
  chk(canon.index_of(R::Energy) == 2, "parse_roles_into:canonical_csv_roles_resolve");

  // --- (5) coupling_role_index : fallback pour bloc SANS roles, LEVE pour bloc AVEC roles sans le role -
  // Bloc SANS roles (legacy / dynamique sans roles declares) -> fallback canonique conserve.
  adc::VariableSet roleless;
  roleless.kind = adc::VariableKind::Conservative;
  roleless.names = {"u0", "u1", "u2"};
  roleless.size = 3;  // roles + user_roles vides
  chk(adc::coupling_role_index(roleless, R::MomentumX, 1, "test", "blk") == 1,
      "coupling_role_index:roleless_block_keeps_fallback");

  // Bloc AVEC roles QUI PORTE le role : on retourne son indice (non canonique : density en comp 2),
  // jamais le fallback (0).
  chk(adc::coupling_role_index(vs, R::Density, 0, "test", "blk") == 2,
      "coupling_role_index:present_role_returns_index");

  // Bloc AVEC roles mais SANS le role requis (vs ne porte pas Energy) -> LEVE en nommant bloc + role.
  bool threw = false, names_block = false, names_role = false;
  try {
    (void)adc::coupling_role_index(vs, R::Energy, 3, "System::add_thermal_exchange", "fluid_a");
  } catch (const std::exception& e) {
    threw = true;
    const std::string m = e.what();
    names_block = m.find("fluid_a") != std::string::npos;
    names_role = m.find("energy") != std::string::npos;
  }
  chk(threw, "coupling_role_index:roles_bearing_missing_role_raises");
  chk(names_block, "coupling_role_index:error_names_block");
  chk(names_role, "coupling_role_index:error_names_role");

  if (fails == 0) {
    std::printf("OK test_variable_user_role : roles utilisateurs nommes + fallbacks stricts\n");
    return 0;
  }
  std::printf("FAIL test_variable_user_role : %d echec(s)\n", fails);
  return 1;
}

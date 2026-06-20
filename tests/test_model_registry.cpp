// Registre UNIQUE des tags de BRIQUES DE MODELE (include/adc/runtime/model_registry.hpp) : source de
// verite partagee par tous les dispatchs modele (detail::dispatch_transport / dispatch_source /
// dispatch_elliptic, le dispatch polaire, et les seams par-transport de python/system.cpp /
// python/amr_system.cpp). Pendant de test_dispatch_tags.cpp (limiteurs + flux) pour l'AXE MODELE.
// Ce test est VOLONTAIREMENT LEGER (il n'inclut QUE model_registry.hpp, aucun System / brique) : il
// verrouille
//   (1) la MATRICE d'appartenance is_transport / is_source / is_elliptic (acceptes + rejets),
//   (2) les helpers de message (transport_tags_csv / *_choices) BYTE-IDENTIQUES aux anciens throws
//       inline ("exb|compressible|isothermal", "'exb' | 'compressible' | 'isothermal'", ...),
//   (3) validate_transport / validate_elliptic : rejet explicite avec le fragment + le tag,
//   (4) les colonnes de capabilite (n_vars, polar_ok, min_vars) et leur variante compile-time
//       (transport_n_vars_ct, utilisee par le static_assert de non-derive de model_factory.hpp),
//   (5) GARDE DE PERIMETRE (ADC-331) : la registry ne contient QUE des briques GENERIQUES, jamais un
//       nom de scenario applicatif (le SET de tags est verrouille -> ajouter "diocotron" echoue ici).
// Le routage effectif (chaque tag builtin atteint bien une branche du dispatch) est verifie cote
// test_config_model_validation.cpp, qui lie la machinerie de dispatch.

#include <adc/runtime/model_registry.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

using namespace adc;

namespace {

// Renvoie true si fn() leve, et capture le message dans @p msg (vide sinon).
template <class Fn>
bool throws(Fn&& fn, std::string& msg) {
  try {
    fn();
    msg.clear();
    return false;
  } catch (const std::exception& e) {
    msg = e.what();
    return true;
  }
}

bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  std::string msg;

  // --- (1) appartenance : matrice d'acceptation / rejet ---------------------------------------------
  chk(is_transport("exb") && is_transport("compressible") && is_transport("isothermal"),
      "is_transport accepte les trois transports builtin");
  chk(!is_transport("bogus") && !is_transport("") && !is_transport("EXB"),
      "is_transport rejette inconnu / vide / casse");
  chk(is_source("none") && is_source("potential") && is_source("gravity") && is_source("magnetic") &&
          is_source("lorentz") && is_source("potential_magnetic") && is_source("potential_lorentz"),
      "is_source accepte les sept orthographes builtin (alias inclus)");
  chk(!is_source("bogus") && !is_source(""), "is_source rejette inconnu / vide");
  chk(is_elliptic("charge") && is_elliptic("background") && is_elliptic("gravity"),
      "is_elliptic accepte les trois elliptiques builtin");
  chk(!is_elliptic("bogus") && !is_elliptic(""), "is_elliptic rejette inconnu / vide");

  // --- (2) helpers de message : BYTE-IDENTIQUES aux anciens throws inline ---------------------------
  chk(transport_tags_csv() == "exb|compressible|isothermal",
      "transport_tags_csv() == 'exb|compressible|isothermal' (message dispatch)");
  chk(transport_tags_csv(/*polar=*/true) == "exb|isothermal",
      "transport_tags_csv(polar) == 'exb|isothermal' (sous-ensemble polaire = colonne polar_ok)");
  chk(elliptic_tags_csv() == "charge|background|gravity",
      "elliptic_tags_csv() == 'charge|background|gravity'");
  chk(transport_choices() == "'exb' | 'compressible' | 'isothermal'",
      "transport_choices() == \"'exb' | 'compressible' | 'isothermal'\" (message validate_model_spec)");
  chk(elliptic_choices() == "'charge' | 'background' | 'gravity'",
      "elliptic_choices() == \"'charge' | 'background' | 'gravity'\"");
  chk(contains(source_choices(), "'none'") && contains(source_choices(), "'potential_lorentz'"),
      "source_choices() liste toutes les orthographes (none .. potential_lorentz)");
  chk(unknown_transport_msg("foo") == "unknown transport 'foo' (exb|compressible|isothermal)",
      "unknown_transport_msg byte-identique a l'ancien throw");
  chk(unknown_elliptic_msg("foo") == "unknown elliptic 'foo' (charge|background|gravity)",
      "unknown_elliptic_msg byte-identique a l'ancien throw");

  // --- (3) validate_transport / validate_elliptic : rejet explicite --------------------------------
  chk(!throws([] { validate_transport("exb"); }, msg), "validate_transport(exb) accepte");
  chk(!throws([] { validate_transport("compressible"); }, msg),
      "validate_transport(compressible) accepte");
  chk(throws([] { validate_transport("bogus"); }, msg), "validate_transport(bogus) rejette");
  chk(contains(msg, "unknown transport") && contains(msg, "bogus") &&
          contains(msg, "exb|compressible|isothermal"),
      "message transport inconnu : fragment + tag + liste valide");
  chk(!throws([] { validate_elliptic("charge"); }, msg), "validate_elliptic(charge) accepte");
  chk(throws([] { validate_elliptic("bogus"); }, msg), "validate_elliptic(bogus) rejette");
  chk(contains(msg, "unknown elliptic") && contains(msg, "bogus") &&
          contains(msg, "charge|background|gravity"),
      "message elliptic inconnu : fragment + tag + liste valide");

  // --- (4) colonnes de capabilite : n_vars / polar_ok / min_vars ------------------------------------
  chk(transport_n_vars("exb") == 1 && transport_n_vars("isothermal") == 3 &&
          transport_n_vars("compressible") == 4,
      "transport_n_vars : exb=1, isothermal=3, compressible=4");
  chk(transport_n_vars("bogus") == -1, "transport_n_vars(inconnu) == -1");
  // variante compile-time (utilisee par les static_assert de non-derive de model_factory.hpp).
  static_assert(transport_n_vars_ct("exb") == 1, "ct exb");
  static_assert(transport_n_vars_ct("compressible") == 4, "ct compressible");
  static_assert(transport_n_vars_ct("isothermal") == 3, "ct isothermal");
  static_assert(transport_n_vars_ct("bogus") == -1, "ct inconnu == -1");
  // polar_ok : exactement {exb, isothermal} sont cables en polaire (compressible : phase ulterieure).
  {
    int n_polar = 0;
    for (const TransportTag& t : kTransports)
      if (t.polar_ok) ++n_polar;
    chk(n_polar == 2, "deux transports polar_ok (exb + isothermal)");
  }
  chk(is_transport("compressible"), "compressible est un transport builtin (cartesien)");
  chk(transport_tags_csv(/*polar=*/true).find("compressible") == std::string::npos,
      "compressible ABSENT du sous-ensemble polaire (pas polar_ok)");
  // min_vars : 'none' neutre (1), les forces fluides exigent >= 3 variables.
  {
    bool ok = true;
    for (const SourceTag& s : kSources) {
      if (std::string(s.name) == "none")
        ok = ok && (s.min_vars == 1);
      else
        ok = ok && (s.min_vars == 3);
    }
    chk(ok, "source min_vars : none=1, forces fluides=3");
  }

  // --- (5) GARDE DE PERIMETRE : que des briques GENERIQUES, aucun scenario applicatif --------------
  // Le SET de tags est verrouille : ModelSpec ne devient PAS une liste implicite de scenarios
  // (ADC-331). Ajouter un nom de scenario (diocotron, ...) ou retirer une brique fait echouer ici.
  {
    constexpr int kNT = static_cast<int>(sizeof(kTransports) / sizeof(kTransports[0]));
    constexpr int kNS = static_cast<int>(sizeof(kSources) / sizeof(kSources[0]));
    constexpr int kNE = static_cast<int>(sizeof(kElliptics) / sizeof(kElliptics[0]));
    chk(kNT == 3 && kNS == 7 && kNE == 3, "cardinalite des tables (3 transports, 7 sources, 3 elliptics)");
    const std::string tr = transport_tags_csv();
    const std::string sr = source_tags_csv();
    const std::string el = elliptic_tags_csv();
    chk(tr == "exb|compressible|isothermal", "set transport verrouille (briques generiques)");
    chk(sr == "none|potential|gravity|magnetic|lorentz|potential_magnetic|potential_lorentz",
        "set source verrouille (briques generiques + alias)");
    chk(el == "charge|background|gravity", "set elliptic verrouille (briques generiques)");
  }

  if (fails == 0) std::printf("OK test_model_registry\n");
  return fails == 0 ? 0 : 1;
}

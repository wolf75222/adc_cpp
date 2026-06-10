#pragma once

#include <stdexcept>
#include <string>

/// @file
/// @brief Registre UNIQUE des tags de schema spatial (limiteurs + flux de Riemann) : source de
///        verite partagee par TOUS les dispatchs (System make_block, AMR dispatch_amr_block /
///        dispatch_amr_compiled, polaire make_block_polar).
///
/// Avant ce header chaque dispatch portait sa PROPRE table de tags (limiteurs x flux) et son PROPRE
/// message d'erreur ; les tables divergeaient en silence (un cas weno5 oublie sur une branche
/// hllc/roe AMR donnait "limiter inconnu" la ou System l'acceptait). Ici : UNE table kLimiters /
/// kRiemanns, des fonctions de VALIDATION partagees (memes messages, ou plus clairs) et
/// limiter_n_ghost (largeur de halo). Les call-sites gardent leur dispatch template if/else (les
/// types Limiter / Flux sont COMPILE-TIME, non tabulables sans X-macro lourde) MAIS valident D'ABORD
/// ici -> rejet centralise ; le throw final du if/else devient une garde "incoherence
/// registry/dispatch" (defense en profondeur, jamais atteinte en pratique).
///
/// Ce header est volontairement LEGER (aucune dependance numerique) : il ne porte que des chaines et
/// des entiers, pas les types Limiter / Flux. Il reste donc inclus tot et sans cout. Les BESOINS de
/// capabilite des flux (hll : ondes signees ; hllc/roe : structure Euler 2D ou capability modele)
/// sont DOCUMENTES dans kRiemanns mais la garde reelle reste un `if constexpr` PAR MODELE au
/// call-site (les capabilites dependent du type Model, indisponible ici).

namespace adc {

/// Tag d'un LIMITEUR de reconstruction : nom expose a l'utilisateur + largeur de halo (n_ghost)
/// requise par son stencil. MIROIR des constantes ::n_ghost des types (NoSlope=1, Minmod=2,
/// VanLeer=2, Weno5=3, cf. numerics/reconstruction.hpp) : un static_assert cote block_builder.hpp
/// (qui voit les deux) verrouille l'absence de derive entre cette table et les types reels.
struct LimiterTag {
  const char* name;
  int n_ghost;
};

/// SOURCE UNIQUE des limiteurs cables (ordre = priorite d'affichage : none < minmod < vanleer < weno5).
inline constexpr LimiterTag kLimiters[] = {
    {"none", 1}, {"minmod", 2}, {"vanleer", 2}, {"weno5", 3}};

/// Tag d'un FLUX de Riemann : nom + besoins de CAPABILITE du modele (DOCUMENTAIRES : la garde reelle
/// est un `if constexpr` par modele au call-site -- ces drapeaux NE pilotent PAS le dispatch, ils
/// documentent le contrat et servent aux tests). polar_ok = cable en geometrie polaire.
///   - needs_wave_speeds : hll exige model.wave_speeds (ondes signees) ;
///   - needs_hllc_struct : hllc exige HasHLLCStructure (ou chemin canonique Euler 2D) ;
///   - needs_roe_diss    : roe  exige HasRoeDissipation (ou chemin canonique Euler 2D).
struct RiemannTag {
  const char* name;
  bool needs_wave_speeds;
  bool needs_hllc_struct;
  bool needs_roe_diss;
  bool polar_ok;
};

/// SOURCE UNIQUE des flux de Riemann cables (ordre = message "(rusanov|hll|hllc|roe)").
inline constexpr RiemannTag kRiemanns[] = {{"rusanov", false, false, false, true},
                                           {"hll", true, false, false, false},
                                           {"hllc", false, true, false, false},
                                           {"roe", false, false, true, false}};

/// Largeur de halo requise par le limiteur @p lim (source : kLimiters). Defaut 2 (MUSCL) pour un
/// limiteur inconnu : c'est l'allocation HISTORIQUE de block_n_ghost -> bit-identique (le dispatch
/// make_block levera de toute facon sur un limiteur invalide). Sert a dimensionner le MultiFab d'etat
/// d'un bloc (stencil large de WENO5 : 5 points, 3 ghosts).
inline int limiter_n_ghost(const std::string& lim) {
  for (const LimiterTag& t : kLimiters)
    if (lim == t.name) return t.n_ghost;
  return 2;  // fallback MUSCL (allocation historique)
}

namespace detail {
/// Egalite de chaines C COMPILE-TIME (pas de <cstring> constexpr garanti partout).
constexpr bool ct_str_eq(const char* a, const char* b) {
  while (*a != '\0' && *b != '\0') {
    if (*a != *b) return false;
    ++a;
    ++b;
  }
  return *a == *b;
}
}  // namespace detail

/// Variante COMPILE-TIME de limiter_n_ghost (litteral const char*) : -1 si inconnu. Sert UNIQUEMENT
/// aux static_assert de non-derive cote block_builder.hpp (cette TU voit ET kLimiters ET les
/// constantes ::n_ghost des types) -- garde que la table ne diverge jamais des types reels.
constexpr int limiter_n_ghost_ct(const char* lim) {
  for (const LimiterTag& t : kLimiters)
    if (detail::ct_str_eq(lim, t.name)) return t.n_ghost;
  return -1;
}

/// Valide un tag de LIMITEUR contre kLimiters. Leve si inconnu, avec le message HISTORIQUE
/// "<ctx> : limiter inconnu '<lim>'" (des tests grepent "limiter inconnu"). @p ctx = prefixe du
/// call-site ("System", "add_block(AmrSystem, multi-blocs)", "add_compiled_model(AmrSystem)",
/// "System (polaire)") -> message STRICTEMENT identique a l'ancien throw inline de chaque dispatch.
inline void validate_limiter(const std::string& lim, const char* ctx = "System") {
  for (const LimiterTag& t : kLimiters)
    if (lim == t.name) return;
  throw std::runtime_error(std::string(ctx) + " : limiter inconnu '" + lim + "'");
}

/// Valide un tag de FLUX de Riemann contre kRiemanns. @p polar : geometrie annulaire (seul rusanov y
/// est cable). Leve si inconnu (cartesien) ou non cable en polaire, avec les messages HISTORIQUES
/// (des tests grepent "flux Riemann", "rusanov", "non supporte"). NE valide PAS les capabilites du
/// modele (hll/hllc/roe sur un transport sans onde signee / sans pression) : ces gardes restent des
/// `if constexpr` PAR MODELE au call-site, avec leurs messages "exige ..." inchanges.
inline void validate_riemann(const std::string& riem, bool polar = false,
                             const char* ctx = "System") {
  if (polar) {
    // Polaire : un seul flux cable (rusanov). Tout autre tag (connu mais non polaire OU inconnu) ->
    // MEME message qu'avant (le dispatch polaire historique rejetait deja tout riem != "rusanov").
    if (riem == "rusanov") return;
    throw std::runtime_error(
        std::string(ctx) + " : flux Riemann '" + riem +
        "' non supporte (polaire -> 'rusanov' ; HLLC/Roe supposent n_vars==4 (Euler avec energie), "
        "sans objet pour l'ExB scalaire ou le fluide isotherme polaire)");
  }
  for (const RiemannTag& t : kRiemanns)
    if (riem == t.name) return;
  throw std::runtime_error(std::string(ctx) + " : flux Riemann inconnu '" + riem +
                           "' (rusanov|hll|hllc|roe)");
}

/// Garde de DEFENSE EN PROFONDEUR : atteinte uniquement si un tag VALIDE (deja accepte par
/// validate_*) n'est route par AUCUNE branche du dispatch if/else -- c'est une incoherence entre le
/// registry (kLimiters/kRiemanns) et le dispatch, donc un bug de programmation, pas une entree
/// utilisateur. Remplace l'ancien `throw "limiter inconnu" / "flux Riemann inconnu"` final, devenu
/// inatteignable depuis que la validation centralisee precede le dispatch. @p kind = "limiteur" ou
/// "flux".
[[noreturn]] inline void throw_registry_dispatch_mismatch(const char* ctx, const char* kind,
                                                          const std::string& tag) {
  throw std::runtime_error(std::string(ctx) + " : incoherence registry/dispatch -- " + kind + " '" +
                           tag + "' valide mais non route (ajouter le cas au dispatch)");
}

}  // namespace adc

// Registre UNIQUE des tags de schema spatial (include/adc/runtime/dispatch_tags.hpp) : source de
// verite partagee par tous les dispatchs (System make_block, AMR dispatch_amr_*, polaire). Ce test
// est VOLONTAIREMENT LEGER (il n'inclut QUE dispatch_tags.hpp, aucun System / modele) : il verrouille
//   (1) la MATRICE d'acceptation/rejet de validate_limiter / validate_riemann (cartesien ET polaire),
//   (2) les n_ghost de limiter_n_ghost (1/2/2/3 + fallback MUSCL 2 sur inconnu),
//   (3) la presence des FRAGMENTS de messages que des tests grepent ("unknown limiter",
//       "unknown Riemann flux", "unsupported" / "polar") + le prefixe de contexte,
//   (4) le contenu des tables kLimiters / kRiemanns (noms, n_ghost, polar_ok).
// Aucune capabilite modele n'est testee ici (hll/hllc/roe sur un transport sans onde / sans pression)
// : ces gardes sont des `if constexpr` PAR MODELE au call-site, hors perimetre du registry.

#include <adc/runtime/dispatch_tags.hpp>

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

  // --- (1) validate_limiter : matrice d'acceptation/rejet -------------------------------------------
  chk(!throws([] { validate_limiter("none"); }, msg), "limiter none accepte");
  chk(!throws([] { validate_limiter("minmod"); }, msg), "limiter minmod accepte");
  chk(!throws([] { validate_limiter("vanleer"); }, msg), "limiter vanleer accepte");
  chk(!throws([] { validate_limiter("weno5"); }, msg), "limiter weno5 accepte");
  // rejets : tag inconnu, casse differente, vide.
  chk(throws([] { validate_limiter("bogus"); }, msg), "limiter bogus rejete");
  chk(contains(msg, "unknown limiter") && contains(msg, "bogus"),
      "message limiter inconnu contient le fragment + le tag");
  chk(contains(msg, "System"), "message limiter porte le contexte par defaut 'System'");
  chk(throws([] { validate_limiter("WENO5"); }, msg), "limiter WENO5 (casse) rejete");
  chk(throws([] { validate_limiter(""); }, msg), "limiter vide rejete");
  // contexte explicite preserve (parite avec les anciens throws inline des dispatchs).
  (void)throws([] { validate_limiter("x", "add_block(AmrSystem, multi-blocs)"); }, msg);
  chk(contains(msg, "add_block(AmrSystem, multi-blocs)") && contains(msg, "unknown limiter"),
      "contexte AMR present dans le message limiter");

  // --- (2) validate_riemann CARTESIEN : matrice -----------------------------------------------------
  chk(!throws([] { validate_riemann("rusanov"); }, msg), "riemann rusanov accepte (cartesien)");
  chk(!throws([] { validate_riemann("hll"); }, msg), "riemann hll accepte (cartesien)");
  chk(!throws([] { validate_riemann("hllc"); }, msg), "riemann hllc accepte (cartesien)");
  chk(!throws([] { validate_riemann("roe"); }, msg), "riemann roe accepte (cartesien)");
  chk(throws([] { validate_riemann("bogus"); }, msg), "riemann bogus rejete (cartesien)");
  chk(contains(msg, "unknown Riemann flux") && contains(msg, "bogus"),
      "message flux Riemann inconnu contient le fragment + le tag");
  chk(contains(msg, "rusanov|hll|hllc|roe"), "message flux liste les tags valides");

  // --- (3) validate_riemann POLAIRE : seul rusanov est cable ---------------------------------------
  chk(!throws([] { validate_riemann("rusanov", /*polar=*/true, "System (polaire)"); }, msg),
      "riemann rusanov accepte (polaire)");
  // hll/hllc/roe sont des tags CONNUS mais NON cables en polaire -> rejet avec le message polaire.
  chk(throws([] { validate_riemann("hllc", /*polar=*/true, "System (polaire)"); }, msg),
      "riemann hllc rejete en polaire");
  chk(contains(msg, "unsupported") && contains(msg, "polar") && contains(msg, "rusanov"),
      "message polaire : unsupported / polar / rusanov");
  chk(!throws([] { validate_riemann("hll", /*polar=*/true, "System (polaire)"); }, msg),
      "riemann hll ACCEPTE en polaire (solde audit : gate wave_speeds au call-site)");
  chk(throws([] { validate_riemann("roe", /*polar=*/true, "System (polaire)"); }, msg),
      "riemann roe rejete en polaire");
  chk(throws([] { validate_riemann("bogus", /*polar=*/true, "System (polaire)"); }, msg),
      "riemann inconnu rejete en polaire (meme message)");
  chk(contains(msg, "unsupported"), "tag inconnu en polaire -> message polaire (parite historique)");

  // --- (4) limiter_n_ghost : largeurs de halo (allocation bit-identique) ----------------------------
  chk(limiter_n_ghost("none") == 1, "n_ghost(none) == 1");
  chk(limiter_n_ghost("minmod") == 2, "n_ghost(minmod) == 2");
  chk(limiter_n_ghost("vanleer") == 2, "n_ghost(vanleer) == 2");
  chk(limiter_n_ghost("weno5") == 3, "n_ghost(weno5) == 3");
  chk(limiter_n_ghost("bogus") == 2, "n_ghost(inconnu) == 2 (fallback MUSCL, historique)");
  // variante compile-time (utilisee par les static_assert de non-derive de block_builder.hpp).
  static_assert(limiter_n_ghost_ct("none") == 1, "ct none");
  static_assert(limiter_n_ghost_ct("weno5") == 3, "ct weno5");
  static_assert(limiter_n_ghost_ct("bogus") == -1, "ct inconnu == -1");

  // --- tables kLimiters / kRiemanns : source de verite ----------------------------------------------
  chk(std::string(kLimiters[0].name) == "none" && kLimiters[0].n_ghost == 1, "kLimiters[0]");
  chk(std::string(kLimiters[3].name) == "weno5" && kLimiters[3].n_ghost == 3, "kLimiters[3]");
  chk(std::string(kRiemanns[0].name) == "rusanov" && kRiemanns[0].polar_ok, "kRiemanns[0] rusanov polar_ok");
  chk(std::string(kRiemanns[1].name) == "hll" && kRiemanns[1].needs_wave_speeds && kRiemanns[1].polar_ok,
      "kRiemanns[1] hll needs_wave_speeds, pas polaire");
  chk(std::string(kRiemanns[2].name) == "hllc" && kRiemanns[2].needs_hllc_struct, "kRiemanns[2] hllc");
  chk(std::string(kRiemanns[3].name) == "roe" && kRiemanns[3].needs_roe_diss, "kRiemanns[3] roe");
  // DEUX flux cables en polaire (rusanov + hll, solde de l'audit) : verrouille polar_ok.
  {
    int n_polar = 0;
    for (const RiemannTag& t : kRiemanns)
      if (t.polar_ok) ++n_polar;
    chk(n_polar == 2, "deux flux polar_ok (rusanov + hll)");
  }

  if (fails == 0) std::printf("OK test_dispatch_tags\n");
  return fails == 0 ? 0 : 1;
}

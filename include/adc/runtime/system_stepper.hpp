#pragma once

#include <adc/core/state.hpp>     // kAuxBaseComps (canal B_z lu par l'etage source condense)
#include <adc/core/types.hpp>     // Real

#include <algorithm>  // std::min, std::max (CFL : pas physique min de la grille, dt min sur les blocs)
#include <cmath>      // std::isfinite, std::ceil (step_cfl / step_adaptive)
#include <limits>     // std::numeric_limits (CFL par bloc : dt = min sur les blocs)
#include <vector>

/// @file
/// @brief SystemStepper : la responsabilite AVANCE EN TEMPS extraite du god-class System::Impl
///        (audit Lot B, continuation de SystemFieldSolver #176). Extrait VERBATIM de python/system.cpp :
///        aucune modification de numerique, de la formule CFL, de la cadence stride/substeps, de la
///        semantique du compteur de macro-pas, des fences, ni de l'ordre (solve_fields ; advance ;
///        etage source ; couplages). STRICTEMENT bit-identique -- le code est deplace tel quel, seul
///        l'acces aux membres PARTAGES de Impl (sp, fields_, aux, couplings, t, macro_step_, geom,
///        pgeom_, polar_) passe par le back-pointer owner_->.
///
/// CONTRAT / INVARIANTS
/// - ORCHESTRE l'avance en temps : step(dt), advance(dt, nsteps), step_cfl(cfl), step_adaptive(cfl),
///   plus les helpers de cadence (stride_due), l'etage source condense (run_source_stage) et les
///   couplages inter-especes (apply_couplings) que les pas invoquent APRES le transport.
/// - LIT (sans posseder) via owner_-> : la liste de blocs (sp) et chaque fermeture d'avance (s.advance),
///   le resolveur elliptique (fields_, pour solve_fields() en tete de pas et fields_.ell_phi() lu par
///   l'etage source), l'aux PARTAGE et sa composante B_z (kAuxBaseComps), la liste de couplages, le
///   temps t et le compteur macro_step_ (qu'il fait avancer), la geometrie (geom cartesien / pgeom_
///   polaire) et le drapeau polar_ pour le pas physique h de la CFL.
/// - PAS PHYSIQUE h DE LA CFL : cartesien = min(dx, dy) ; POLAIRE = min(dr, r_min * dtheta) (le pas
///   azimutal r*dtheta est minimal au rayon interieur r_min de l'anneau -> bord le plus contraignant).
/// - INVARIANT CADENCE MULTIRATE (hold-then-catch-up) : un bloc de cadence M est TENU tant que
///   (macro_step + 1) % M != 0, puis avance d'un pas effectif M*dt au macro-pas ou (macro_step + 1) % M
///   == 0 (FIN de fenetre). macro_step_ est incremente UNE fois par macro-pas, APRES l'avance des blocs
///   et les couplages. NE PAS reordonner solve_fields ; advance ; run_source_stage ; apply_couplings ;
///   t += dt ; ++macro_step_.
/// - FORMULE CFL PAR BLOC (substeps-aware, post-#121) : dt <= cfl * h * substeps_b / (stride_b * w_b) ;
///   le dt global est le min sur les blocs evolutifs. PRESERVEE telle quelle.
///
/// Comme System::Impl reste PRIVE a python/system.cpp, ce helper est un TEMPLATE parametre sur le type
/// Impl reel (meme technique que system_field_solver / native_loader) : python/system.cpp l'instancie
/// avec System::Impl apres avoir defini Impl. owner_ est un Impl* (la duree de vie du helper est
/// sous-jacente a celle de Impl). System::step / advance / step_cfl / step_adaptive deviennent de
/// simples delegations a stepper_.

namespace adc {
namespace stepper {

/// SystemStepper<Impl> : voir contrat ci-dessus. Toutes les methodes sont des MEMBRES car elles
/// partagent l'orchestration de pas ; les acces a l'etat PARTAGE de Impl passent par owner_-> verbatim.
/// Template sur Impl pour rester sans dependance sur la definition (privee) de System::Impl.
template <class Impl>
class SystemStepper {
 public:
  /// @param owner back-pointer vers System::Impl (duree de vie sous-jacente a celle de Impl).
  explicit SystemStepper(Impl* owner) : owner_(owner) {}

  /// Vrai si un bloc de cadence @p stride RATTRAPE a ce macro-pas (FIN de fenetre).
  /// SEMANTIQUE STRIDE = HOLD-THEN-CATCH-UP (rattrapage en FIN de fenetre). Un bloc de cadence M est
  /// TENU (non avance) sur les macro-pas ou (macro_step + 1) % M != 0, puis avance d'un pas effectif
  /// M*dt au macro-pas ou (macro_step + 1) % M == 0, i.e. a la FIN de sa fenetre de M macro-pas. Au
  /// macro-pas k, le temps du systeme est (k+1)*dt et le bloc qui RATTRAPE a alors avance du meme
  /// (k+1)*dt cumule : il est temporellement COHERENT avec les blocs rapides, jamais "dans le futur".
  /// (L'ancienne semantique avancait au DEBUT de fenetre, macro_step % M == 0 : a k=0 le bloc avancait
  /// deja M*dt alors que le systeme n'avancait que dt -> bloc anticipe, couplage Poisson/source faux.)
  static bool stride_due(int macro_step, int stride) { return (macro_step + 1) % stride == 0; }

  /// Sources de COUPLAGE inter-especes : appliquees par SPLITTING (un pas additif explicite de dt)
  /// APRES le transport de chaque bloc. Chaque couplage est un for_each_cell (kernel DEVICE) lisant /
  /// mettant a jour plusieurs blocs au meme point ; ils s'ordonnent apres le transport sur le meme
  /// espace d'execution, donc plus de device_fence prealable (plus d'acces hote).
  void apply_couplings(Real dt) {
    if (owner_->couplings.empty()) return;
    for (auto& c : owner_->couplings) c(dt);
  }

  /// ETAGE SOURCE condense par Schur (OPT-IN, cf. set_source_stage). No-op si le bloc n'a pas d'etage
  /// source (s.schur == nullptr) : le chemin par defaut reste BIT-IDENTIQUE. Sinon, APRES le transport
  /// hyperbolique du bloc (deja joue par s.advance), on joue l'etage source AUTONOME
  /// (CondensedSchurSourceStepper, #126) sur l'etat post-transport :
  ///   - state = s.U (rho gelee dans la source, mom/E mis a jour) ;
  ///   - phi    = le potentiel du Poisson de systeme (ell_phi(), warm start phi^n issu de solve_fields
  ///              en tete de step) -- l'etage resout son PROPRE operateur condense et ECRIT phi^{n+1}
  ///              dedans, il NE rappelle PAS solve_fields (pas de duplication) ;
  ///   - B_z    = canal aux a l'indice kAuxBaseComps (peuple + ghosts remplis par solve_fields).
  /// theta/dt du theta-schema ; dt = eff_dt (facteur stride deja inclus par l'appelant, comme s.advance).
  void run_source_stage(typename Impl::Species& s, Real eff_dt) {
    // DISPATCH GEOMETRIE (Voie A etape 2c) : un bloc porte AU PLUS UN etage source condense (set_source_stage
    // construit le cartesien OU le polaire selon la geometrie du System). Le POLAIRE
    // (PolarCondensedSchurSourceStepper, #212) a la MEME signature step(state, phi, bz, c_bz, theta, dt) que
    // le cartesien (#126) : seul le pointeur change. Le chemin cartesien reste BIT-IDENTIQUE (schur_polar
    // == nullptr en cartesien -> on prend la branche schur d'origine, inchangee).
    if (s.schur_polar) {
      s.schur_polar->step(s.U, owner_->fields_.ell_phi(), owner_->aux, kAuxBaseComps,
                          static_cast<Real>(s.schur_theta), eff_dt);
      return;
    }
    if (!s.schur) return;
    s.schur->step(s.U, owner_->fields_.ell_phi(), owner_->aux, kAuxBaseComps,
                  static_cast<Real>(s.schur_theta), eff_dt);
  }

  /// Un macro-pas de longueur @p dt : solve_fields ; avance de chaque bloc DU (cadence stride honoree) ;
  /// etage source condense ; couplages inter-especes ; t += dt ; ++macro_step. ORDRE INVARIANT.
  void step(double dt) {
    Impl* P = owner_;
    // COUPLAGE / POISSON : solve_fields assemble f = Sum_s elliptic_rhs_s(U_s) sur l'etat COURANT de
    // chaque bloc. Un bloc TENU (cadence M, hors fin de fenetre) y contribue avec son etat PERIME (sa
    // derniere avance, donc figee jusqu'a son prochain rattrapage) : densite / charge stale dans la
    // somme du Poisson tant qu'il n'a pas rattrape. Choix assume du stride (couplage lache du bloc lent).
    P->solve_fields();
    for (auto& s : P->sp) {
      if (!s.evolve) continue;  // bloc gele : non avance
      if (!stride_due(P->macro_step_, s.stride)) continue;  // hold : pas en fin de fenetre stride
      const Real eff_dt = Real(dt) * Real(s.stride);  // catch-up : pas effectif s.stride * dt
      s.advance(s.U, eff_dt, s.substeps);
      run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
    }
    apply_couplings(Real(dt));  // sources couplees inter-especes (splitting), apres transport
    P->t += dt;
    P->macro_step_++;
  }

  /// Avance de @p nsteps macro-pas de longueur @p dt (boucle sur step).
  void advance(double dt, int nsteps) {
    for (int s = 0; s < nsteps; ++s) step(dt);
  }

  /// Un macro-pas a dt CFL : dt = min sur les blocs evolutifs de cfl*h*substeps_b/(stride_b*w_b), puis
  /// avance comme step. @return le dt utilise. SUBSTEPS-AWARE (post-#121) : bit-identique a l'ancienne
  /// formule seulement pour substeps=1 (cf. note retro-compatibilite).
  double step_cfl(double cfl) {
    Impl* P = owner_;
    P->solve_fields();
    // Pas physique MIN de la grille : cartesien = min(dx, dy) ; POLAIRE = min(dr, r_min * dtheta) (le pas
    // physique azimutal r*dtheta est minimal au rayon interieur r_min de l'anneau -> bord le plus
    // contraignant pour la CFL). Le reste de la formule CFL (par bloc, substeps/stride) est inchange.
    const Real h = P->polar_
                       ? std::min(P->pgeom_.dr(), P->pgeom_.r_min * P->pgeom_.dtheta())
                       : std::min(P->geom.dx(), P->geom.dy());
    // CFL PAR BLOC, FACTEUR STRIDE ET SUBSTEPS INCLUS. Un bloc de cadence M avance d'un pas effectif
    // M*dt en substeps_b sous-pas, donc chaque sous-pas vaut stride_b * dt / substeps_b : la condition
    // stable par sous-pas est stride_b * dt / substeps_b <= cfl * h / w_b, soit
    //   dt <= cfl * h * substeps_b / (stride_b * w_b).
    // Le dt GLOBAL est le min sur les blocs evolutifs (le plus contraignant). Sans cela, le pas calcule
    // sur w_max seul puis multiplie par M violerait la CFL d'un facteur M sur le bloc a stride.
    //
    // RETRO-COMPATIBILITE (post-#121). La formule est SUBSTEPS-AWARE : avec substeps_b > 1, le dt
    // retourne est substeps_b fois plus grand que l'ancienne formule dt = cfl*h/(stride*w).
    // bit-identique seulement pour substeps=1 (a tout stride) ; step_cfl est desormais substeps-aware
    // (dt = cfl*h*substeps/(stride*w)), donc un run step_cfl avec substeps>1 avance un dt plus grand
    // qu'avant #121 (pas CFL-maximal, chaque sous-pas est a la limite de stabilite).
    // Pour reproduire un run calibre avec l'ancienne formule, utiliser step(dt) avec le dt historique
    // explicite, PAS step_cfl.
    double dt = std::numeric_limits<double>::infinity();
    for (auto& s : P->sp) {
      if (!s.evolve) continue;  // bloc gele : ne contraint pas le pas
      const Real w = std::max(s.max_speed(s.U), Real(1e-30));
      const double dt_b = cfl * static_cast<double>(h) * static_cast<double>(s.substeps) /
                          (static_cast<double>(s.stride) * static_cast<double>(w));
      dt = std::min(dt, dt_b);
    }
    if (!std::isfinite(dt)) dt = cfl * static_cast<double>(h) / 1e-30;  // tous geles : pas degenere
    for (auto& s : P->sp) {
      if (!s.evolve) continue;
      if (!stride_due(P->macro_step_, s.stride)) continue;  // hold : pas en fin de fenetre stride
      const Real eff_dt = Real(dt) * Real(s.stride);  // catch-up : pas effectif s.stride * dt
      s.advance(s.U, eff_dt, s.substeps);
      run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
    }
    apply_couplings(Real(dt));
    P->t += dt;
    P->macro_step_++;
    return dt;
  }

  /// Un macro-pas MULTIRATE : le macro-pas = pas stable du bloc le plus LENT ; chaque bloc plus rapide
  /// est sous-cycle n_b = ceil(stride_b * w_b / w_min) fois. aux fige sur le macro-pas (couplage
  /// once-per-step). @return le macro-pas.
  double step_adaptive(double cfl) {
    Impl* P = owner_;
    P->solve_fields();
    // Multirate : macro-pas = pas stable du bloc le plus LENT ; chaque bloc plus rapide est
    // sous-cycle n_b. aux fige sur le macro-pas (couplage once-per-step). SEMANTIQUE STRIDE =
    // hold-then-catch-up : un bloc de cadence M est TENU tant que (macro_step + 1) % M != 0, puis
    // avance d'un pas effectif M*macro_dt en fin de fenetre (cf. stride_due).
    Real wmin = Real(1e30);
    std::vector<Real> wb;
    wb.reserve(P->sp.size());
    for (auto& s : P->sp) {
      const Real w = s.evolve ? s.max_speed(s.U) : Real(0);  // bloc gele : hors cadence
      wb.push_back(w);
      if (s.evolve) wmin = std::min(wmin, w);
    }
    if (wmin >= Real(1e30)) wmin = Real(1e-30);  // aucun bloc evolutif (tous geles)
    const Real h = P->polar_
                       ? std::min(P->pgeom_.dr(), P->pgeom_.r_min * P->pgeom_.dtheta())
                       : std::min(P->geom.dx(), P->geom.dy());
    const double macro_dt = cfl * static_cast<double>(h) / static_cast<double>(wmin);
    for (std::size_t b = 0; b < P->sp.size(); ++b) {
      auto& s = P->sp[b];
      if (!s.evolve) continue;  // bloc gele : non avance
      if (!stride_due(P->macro_step_, s.stride)) continue;  // hold : pas en fin de fenetre stride
      // Sous-cyclage stable du pas EFFECTIF M*macro_dt : chaque sous-pas doit verifier
      // M*macro_dt / n <= cfl*h / w_b, i.e. n >= ceil(M * w_b / w_min). Le facteur stride M est donc
      // porte par le nombre de sous-pas (sans lui, n sur w_b/w_min seul violerait la CFL d'un facteur M).
      int n = static_cast<int>(std::ceil(static_cast<double>(s.stride) *
                                         static_cast<double>(wb[b] / wmin)));
      if (n < 1) n = 1;
      const Real eff_dt = Real(macro_dt) * Real(s.stride);  // catch-up : pas effectif M*macro_dt
      s.advance(s.U, eff_dt, n);
      run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
    }
    apply_couplings(Real(macro_dt));
    P->t += macro_dt;
    P->macro_step_++;
    return macro_dt;
  }

 private:
  Impl* owner_;
};

}  // namespace stepper
}  // namespace adc

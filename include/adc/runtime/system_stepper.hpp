#pragma once

#include <adc/core/state.hpp>     // kAuxBaseComps (canal B_z lu par l'etage source condense)
#include <adc/core/types.hpp>     // Real
#include <adc/parallel/comm.hpp>  // all_reduce_min (bornes globales : dt identique sur tous les rangs)
#include <adc/runtime/grid_context.hpp>  // GeometryMode (aiguillage du transport disque)

#include <stdexcept>  // std::runtime_error (mode disque demande sans avance disque sur un bloc)

#include <algorithm>  // std::min, std::max (CFL : pas physique min de la grille, dt min sur les blocs)
#include <cmath>      // std::isfinite, std::ceil (step_cfl / step_adaptive)
#include <limits>     // std::numeric_limits (CFL par bloc : dt = min sur les blocs)
#include <string>     // last_dt_bound (nom de la borne active du dernier step_cfl)
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

/// Politique de SPLITTING en temps du macro-pas (transport hyperbolique H + etage source S).
///  - Lie     : H(dt) ; S(dt) une fois (Godunov, 1er ordre). C'EST LE DEFAUT, bit-identique a
///              l'historique : un seul solve_fields en tete de pas, advance puis run_source_stage
///              entrelaces dans la meme boucle de blocs (cf. step()).
///  - Strang  : H(dt/2) ; S(dt) ; H(dt/2) (symetrique, 2e ordre des que H et S le sont). Necessite
///              de RE-RESOUDRE solve_fields ENTRE les etages (cf. step()) : voir le commentaire de la
///              branche Strang et docs/HOFFART_STEP_SEQUENCE.md (le solve_fields UNIQUE de tete ne
///              suffit pas a la 2e demi-avance, qui lirait sinon un phi perime).
enum class SplitScheme { Lie, Strang };

/// SystemStepper<Impl> : voir contrat ci-dessus. Toutes les methodes sont des MEMBRES car elles
/// partagent l'orchestration de pas ; les acces a l'etat PARTAGE de Impl passent par owner_-> verbatim.
/// Template sur Impl pour rester sans dependance sur la definition (privee) de System::Impl.
template <class Impl>
class SystemStepper {
 public:
  /// @param owner back-pointer vers System::Impl (duree de vie sous-jacente a celle de Impl).
  explicit SystemStepper(Impl* owner) : owner_(owner) {}

  /// Choisit la politique de splitting en temps (defaut Lie = bit-identique). Voir SplitScheme.
  void set_scheme(SplitScheme scheme) { scheme_ = scheme; }
  SplitScheme scheme() const { return scheme_; }

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
      s.schur_polar->step(s.U, owner_->fields_.ell_phi(), owner_->aux, s.schur_bz_comp,
                          static_cast<Real>(s.schur_theta), eff_dt);
      return;
    }
    if (s.schur) {
      s.schur->step(s.U, owner_->fields_.ell_phi(), owner_->aux, s.schur_bz_comp,
                    static_cast<Real>(s.schur_theta), eff_dt);
      return;
    }
    // ETAGE SOURCE GENERIQUE (fallback) : joue UNIQUEMENT si AUCUN etage Schur condense (chemin de
    // production INTOUCHE, bit-identique). Avance l'etage source du bloc EN PLACE sur eff_dt. nullptr
    // (defaut) -> no-op, comme avant. Sert au splitting generique (adc.Strang) et aux tests d'ordre.
    if (s.source_step) s.source_step(s.U, eff_dt);
  }

  /// AVANCE DE TRANSPORT du bloc @p s sur @p dt en @p n sous-pas, AIGUILLEE par le mode de geometrie du
  /// System (chantier T5-PR3). C'est le SEUL point de cablage du disque dans le pas (les 4 pas -- step,
  /// step_strang, step_cfl, step_adaptive -- passent par ici) :
  ///   - None (defaut)             : s.advance (assemble_rhs, plein cartesien). BIT-IDENTIQUE.
  ///   - Staircase, disque fixe    : s.advance_masked (assemble_rhs_masked, masque 0/1).
  ///   - CutCell, disque fixe      : s.advance_eb (assemble_rhs_eb, cut-cell EB).
  /// Un mode disque demande SANS disque fixe (disc_set_ == false) RETOMBE sur s.advance : le mode seul
  /// (sans set_disc_domain) ne doit pas changer le transport. Un mode disque avec disque fixe mais sur
  /// un bloc qui N'A PAS fabrique l'avance disque (p.ex. bloc polaire / charge depuis un .so anterieur)
  /// leve une erreur EXPLICITE plutot que de jouer SILENCIEUSEMENT le chemin plein (le footgun T2 :
  /// croire le disque actif alors que le transport l'ignore). Les avances disque MIMENT s.advance
  /// (meme schema RK / IMEX, meme limiteur / flux) ; seul le residu de transport est aiguille.
  void advance_transport_n(typename Impl::Species& s, Real dt, int n) {
    const GeometryMode mode = owner_->geometry_mode_;
    if (mode == GeometryMode::None || !owner_->disc_set_) {
      s.advance(s.U, dt, n);  // chemin par defaut (ou mode disque sans disque fixe) : BIT-IDENTIQUE
      return;
    }
    if (mode == GeometryMode::Staircase) {
      if (!s.advance_masked)
        throw std::runtime_error(
            "SystemStepper : mode geometrie 'staircase' demande mais le bloc '" + s.name +
            "' n'expose pas d'avance de transport masquee (transport disque non cable pour ce bloc)");
      s.advance_masked(s.U, dt, n);
      return;
    }
    // CutCell
    if (!s.advance_eb)
      throw std::runtime_error(
          "SystemStepper : mode geometrie 'cutcell' demande mais le bloc '" + s.name +
          "' n'expose pas d'avance de transport cut-cell EB (transport disque non cable pour ce bloc)");
    s.advance_eb(s.U, dt, n);
  }

  /// AVANCE DE TRANSPORT du bloc @p s sur @p eff_dt en s.substeps sous-pas, aiguillee par le mode (cf.
  /// advance_transport_n). Reutilise s.substeps comme l'ancien s.advance des pas step / step_cfl.
  void advance_transport(typename Impl::Species& s, Real eff_dt) {
    advance_transport_n(s, eff_dt, s.substeps);
  }

  /// DEMI-avance de transport (Strang) : aiguillage de advance_transport sur @p half_dt = eff_dt/2 --
  /// le mode disque honore AUSSI le chemin Strang (H(dt/2) S(dt) H(dt/2)).
  void advance_transport_half(typename Impl::Species& s, Real eff_dt) {
    advance_transport_n(s, Real(0.5) * eff_dt, s.substeps);
  }

  /// Un macro-pas de longueur @p dt. ORDRE INVARIANT par schema (cf. SplitScheme) :
  ///  - Lie (defaut, bit-identique) : solve_fields ; pour chaque bloc DU (cadence stride honoree)
  ///    advance(dt) puis run_source_stage(dt) entrelaces ; couplages ; t += dt ; ++macro_step.
  ///  - Strang : H(dt/2) ; S(dt) ; H(dt/2), avec un solve_fields RE-RESOLU entre chaque etage
  ///    (cf. step_strang() et docs/HOFFART_STEP_SEQUENCE.md).
  void step(double dt) {
    if (scheme_ == SplitScheme::Strang) { step_strang(dt); return; }
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
      advance_transport(s, eff_dt);  // transport AIGUILLE par le mode geometrie (None : assemble_rhs)
      run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
    }
    apply_couplings(Real(dt));  // sources couplees inter-especes (splitting), apres transport
    P->t += dt;
    P->macro_step_++;
  }

  /// Un macro-pas STRANG (symetrique, 2e ordre) : H(dt/2) ; S(dt) ; H(dt/2). Reutilise s.advance pour
  /// les DEMI-avances de transport et run_source_stage pour l'etage source PLEIN (aucun nouveau stepper).
  ///
  /// CONSISTANCE phi (point critique, cf. docs/HOFFART_STEP_SEQUENCE.md) : le potentiel phi / les champs
  /// aux (grad phi) que LIT le transport sont peuples par solve_fields a partir de la densite COURANTE.
  /// Le solve_fields UNIQUE de tete de pas (suffisant pour le splitting de Lie, ou une seule avance de
  /// transport suit) NE SUFFIT PAS ici : entre la 1re demi-avance et la 2nde, la densite a change (1re
  /// demi-avance + etage source), donc la 2nde demi-avance lirait un phi PERIME. On RE-RESOUT donc
  /// solve_fields AVANT chaque etage qui consomme phi :
  ///   1. solve_fields()  -> phi coherent avec rho^n           (pour H(dt/2))
  ///   2. H(dt/2)         (s.advance sur eff_dt/2)
  ///   3. solve_fields()  -> phi coherent avec rho apres H(dt/2) (pour S(dt) : warm start / champ aux)
  ///   4. S(dt)           (run_source_stage sur eff_dt ; l'etage Schur ECRIT phi^{n+1})
  ///   5. solve_fields()  -> phi coherent avec rho apres S(dt)   (pour la 2nde H(dt/2))
  ///   6. H(dt/2)         (s.advance sur eff_dt/2)
  /// Sans l'etape 5, la 2nde demi-avance lirait soit le phi du Schur (ecrase au pas suivant de toute
  /// facon) soit un phi de rho perime : la symetrie Strang (donc le 2e ordre) serait rompue. Les etapes
  /// 1/3/5 sont des solve_fields SYSTEME (somme sur tous les blocs), hors boucle de blocs.
  ///
  /// CADENCE stride : evaluee UNE fois par macro-pas (stride_due au DEBUT), de sorte que les deux
  /// demi-avances et l'etage source d'un meme macro-pas concernent le MEME ensemble de blocs DUS. Le
  /// pas effectif eff_dt = dt * stride (catch-up) est identique a Lie ; seul le transport est scinde en
  /// deux moities eff_dt/2.
  void step_strang(double dt) {
    Impl* P = owner_;
    // (1) phi coherent avec rho^n, pour la 1re demi-avance de transport.
    P->solve_fields();
    // (2) H(dt/2) : 1re demi-avance de transport de chaque bloc DU. s.substeps sous-pas (inchange).
    // Aiguillee par le mode geometrie (None : assemble_rhs ; Staircase/CutCell : operateur disque).
    for (auto& s : P->sp) {
      if (!s.evolve) continue;
      if (!stride_due(P->macro_step_, s.stride)) continue;
      const Real eff_dt = Real(dt) * Real(s.stride);
      advance_transport_half(s, eff_dt);
    }
    // (3) phi RE-RESOLU sur la densite post-H(dt/2), pour l'etage source.
    P->solve_fields();
    // (4) S(dt) : etage source PLEIN de chaque bloc DU (no-op si pas d'etage Schur, comme Lie).
    for (auto& s : P->sp) {
      if (!s.evolve) continue;
      if (!stride_due(P->macro_step_, s.stride)) continue;
      const Real eff_dt = Real(dt) * Real(s.stride);
      run_source_stage(s, eff_dt);
    }
    // (5) phi RE-RESOLU sur la densite post-source : SANS cette resolution la 2nde demi-avance lirait un
    //     phi perime (cf. docs/HOFFART_STEP_SEQUENCE.md, le solve_fields de tete unique est insuffisant).
    P->solve_fields();
    // (6) H(dt/2) : 2nde demi-avance de transport, fermant le pas symetrique Strang. MEME aiguillage.
    for (auto& s : P->sp) {
      if (!s.evolve) continue;
      if (!stride_due(P->macro_step_, s.stride)) continue;
      const Real eff_dt = Real(dt) * Real(s.stride);
      advance_transport_half(s, eff_dt);
    }
    apply_couplings(Real(dt));  // sources couplees inter-especes (splitting), apres le pas symetrique
    P->t += dt;
    P->macro_step_++;
  }

  /// Avance de @p nsteps macro-pas de longueur @p dt (boucle sur step).
  void advance(double dt, int nsteps) {
    for (int s = 0; s < nsteps; ++s) step(dt);
  }

  /// Un macro-pas a dt CFL : dt = min sur les blocs evolutifs des BORNES de pas du bloc, puis avance
  /// comme step. @return le dt utilise. SUBSTEPS-AWARE (post-#121) : bit-identique a l'ancienne
  /// formule seulement pour substeps=1 (cf. note retro-compatibilite).
  ///
  /// POLITIQUE DE PAS (audit 2026-06, chantier step_cfl) : la borne de TRANSPORT historique
  /// dt <= cfl*h*substeps_b/(stride_b*w_b) reste le socle, mais le pas AGREGE desormais, par bloc :
  ///   - la borne de FREQUENCE DE SOURCE (s.source_frequency, trait HasSourceFrequency) :
  ///     sous-pas effectif stride*dt/substeps <= cfl/mu -> dt <= cfl*substeps/(stride*mu), SANS h
  ///     (une source locale borne en 1/temps, pas en longueur/temps) ;
  ///   - le PAS ADMISSIBLE direct (s.stability_dt, trait HasStabilityDt) :
  ///     stride*dt/substeps <= dt_adm -> dt <= dt_adm*substeps/stride, SANS cfl (le modele declare
  ///     deja un pas admissible) ;
  ///   - la vitesse de CFL elle-meme peut etre la vitesse de STABILITE declaree (trait
  ///     HasStabilitySpeed) : s.max_speed est alors cable sur stability_speed (cf. make_max_speed).
  /// Puis les bornes GLOBALES (P->dt_bounds_ : couplage multi-blocs, Schur/Poisson, AMR/scheduler,
  /// posees par System::add_dt_bound) : dt <= fn() chacune, une evaluation HOTE par pas (pas de
  /// callback par cellule). Un bloc/un systeme SANS bornes optionnelles garde un pas STRICTEMENT
  /// identique a l'historique (les fonctions vides ne sont pas interrogees). La borne ACTIVE du
  /// dernier pas est consultable via last_dt_bound() ("transport:<bloc>", "source_frequency:<bloc>",
  /// "stability_dt:<bloc>", "global:<label>", "degenerate").
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
    std::string reason = "degenerate";
    for (auto& s : P->sp) {
      if (!s.evolve) continue;  // bloc gele : ne contraint pas le pas
      const Real w = std::max(s.max_speed(s.U), kCflSpeedFloor);
      double dt_b = cfl * static_cast<double>(h) * static_cast<double>(s.substeps) /
                    (static_cast<double>(s.stride) * static_cast<double>(w));
      const char* why = "transport";
      // Borne de FREQUENCE DE SOURCE (optionnelle ; mu <= 0 = ne contraint pas).
      if (s.source_frequency) {
        const Real mu = s.source_frequency(s.U);
        if (mu > Real(0)) {
          const double dt_src = cfl * static_cast<double>(s.substeps) /
                                (static_cast<double>(s.stride) * static_cast<double>(mu));
          if (dt_src < dt_b) { dt_b = dt_src; why = "source_frequency"; }
        }
      }
      // PAS ADMISSIBLE direct (optionnel ; <= 0 = ne contraint pas ; cfl NON applique).
      if (s.stability_dt) {
        const Real db = s.stability_dt(s.U);
        if (db > Real(0)) {
          const double dt_adm = static_cast<double>(db) * static_cast<double>(s.substeps) /
                                static_cast<double>(s.stride);
          if (dt_adm < dt_b) { dt_b = dt_adm; why = "stability_dt"; }
        }
      }
      if (dt_b < dt) { dt = dt_b; reason = std::string(why) + ":" + s.name; }
    }
    // Frequences DECLAREES des sources couplees (CoupledSource.frequency) : les couplages
    // s'appliquent UNE fois par MACRO-pas (apply_couplings(dt)), la borne porte donc sur le
    // macro-dt directement : dt <= cfl / mu (PAS de facteur substeps/stride -- ceux-ci ne
    // s'appliquent qu'au transport sous-cycle du bloc, pas au splitting des couplages).
    for (const auto& cs : P->coupled_freqs_) {
      if (!(cs.mu > 0.0)) continue;
      const double dt_cs = cfl / cs.mu;
      if (dt_cs < dt) { dt = dt_cs; reason = "coupled_source:" + cs.label; }
    }
    // Bornes GLOBALES (System::add_dt_bound) : couplage multi-blocs, Schur/Poisson, AMR/scheduler.
    // Une evaluation HOTE par pas et par borne ; <= 0 ou non finie = ne contraint pas ce pas
    // (neutralisee en +inf AVANT le min global). ALL_REDUCE_MIN obligatoire : la callback est
    // evaluee PAR RANG (elle peut lire un etat rank-local) ; sans le min global chaque rang
    // choisirait un dt different -> collectifs du pas desynchronises (Krylov / fill_boundary) ->
    // deadlock MPI. En serie all_reduce_min est l'identite (bit-identique).
    for (const auto& g : P->dt_bounds_) {
      if (!g.fn) continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v)) v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
      if (v < dt) { dt = v; reason = "global:" + g.label; }
    }
    if (!std::isfinite(dt)) {
      dt = cfl * static_cast<double>(h) / static_cast<double>(kCflSpeedFloor);  // tous geles : pas degenere
      reason = "degenerate";
    }
    last_dt_reason_ = std::move(reason);
    for (auto& s : P->sp) {
      if (!s.evolve) continue;
      if (!stride_due(P->macro_step_, s.stride)) continue;  // hold : pas en fin de fenetre stride
      const Real eff_dt = Real(dt) * Real(s.stride);  // catch-up : pas effectif s.stride * dt
      advance_transport(s, eff_dt);  // transport AIGUILLE par le mode geometrie (None : assemble_rhs)
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
  ///
  /// BORNES OPTIONNELLES (audit 2026-06) : comme step_cfl, le macro-pas est ensuite REDUIT par les
  /// bornes de bloc (source_frequency / stability_dt, appliquees au sous-pas effectif
  /// stride_b*macro_dt/n_b -- n_b ne depend pas de dt, donc la clameur est exacte) et par les bornes
  /// GLOBALES (P->dt_bounds_). Sans bornes optionnelles, macro_dt est STRICTEMENT historique.
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
    if (wmin >= Real(1e30)) wmin = kCflSpeedFloor;  // aucun bloc evolutif (tous geles)
    const Real h = P->polar_
                       ? std::min(P->pgeom_.dr(), P->pgeom_.r_min * P->pgeom_.dtheta())
                       : std::min(P->geom.dx(), P->geom.dy());
    double macro_dt = cfl * static_cast<double>(h) / static_cast<double>(wmin);
    // Bornes de bloc OPTIONNELLES : chaque bloc sous-cycle n_b fois son pas effectif
    // stride_b*macro_dt ; le sous-pas stride_b*macro_dt/n_b doit verifier les bornes de source /
    // pas admissible du bloc. n_b (formule identique a la boucle d'avance ci-dessous) ne depend
    // pas de macro_dt : la clameur est faite AVANT l'avance, n_b reste coherent.
    for (std::size_t b = 0; b < P->sp.size(); ++b) {
      auto& s = P->sp[b];
      if (!s.evolve) continue;
      if (!s.source_frequency && !s.stability_dt) continue;
      int n = static_cast<int>(std::ceil(static_cast<double>(s.stride) *
                                         static_cast<double>(wb[b] / wmin)));
      if (n < 1) n = 1;
      if (s.source_frequency) {
        const Real mu = s.source_frequency(s.U);
        if (mu > Real(0))
          macro_dt = std::min(macro_dt, cfl * static_cast<double>(n) /
                                            (static_cast<double>(s.stride) * static_cast<double>(mu)));
      }
      if (s.stability_dt) {
        const Real db = s.stability_dt(s.U);
        if (db > Real(0))
          macro_dt = std::min(macro_dt, static_cast<double>(db) * static_cast<double>(n) /
                                            static_cast<double>(s.stride));
      }
    }
    // Frequences declarees des sources couplees (cf. step_cfl) : borne sur le MACRO-pas.
    for (const auto& cs : P->coupled_freqs_) {
      if (cs.mu > 0.0) macro_dt = std::min(macro_dt, cfl / cs.mu);
    }
    // Bornes GLOBALES (System::add_dt_bound), comme step_cfl (meme all_reduce_min : dt identique
    // sur tous les rangs, cf. le commentaire de step_cfl).
    for (const auto& g : P->dt_bounds_) {
      if (!g.fn) continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v)) v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
      if (v < macro_dt) macro_dt = v;
    }
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
      advance_transport_n(s, eff_dt, n);  // transport AIGUILLE par le mode geometrie (n sous-pas adaptatifs)
      run_source_stage(s, eff_dt);  // OPT-IN : etage source condense par Schur (no-op sinon)
    }
    apply_couplings(Real(macro_dt));
    P->t += macro_dt;
    P->macro_step_++;
    return macro_dt;
  }

  /// Nom de la borne ACTIVE (celle qui a fixe dt) du dernier step_cfl : "transport:<bloc>",
  /// "source_frequency:<bloc>", "stability_dt:<bloc>", "global:<label>", "degenerate", ou "" si
  /// aucun step_cfl n'a encore tourne. Diagnostic (System::last_dt_bound).
  const std::string& last_dt_reason() const { return last_dt_reason_; }

 private:
  Impl* owner_;
  SplitScheme scheme_ = SplitScheme::Lie;  // defaut Lie (Godunov) : bit-identique a l'historique
  std::string last_dt_reason_;             // borne active du dernier step_cfl (diagnostic)
};

}  // namespace stepper
}  // namespace adc

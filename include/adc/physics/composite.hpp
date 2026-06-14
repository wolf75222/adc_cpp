#pragma once

#include <adc/core/physical_model.hpp>  // HyperbolicPhysicalModel : contrat de la brique hyperbolique
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>

/// @file
/// @brief CompositeModel : assemble (hyperbolique, source, elliptic) en un PhysicalModel compile.
///
/// La brique HYPERBOLIQUE porte Vars + flux + vitesses d'onde + conversions (indissociables) ; source
/// et elliptique sont des briques SEPAREES (physics/source.hpp, physics/elliptic.hpp), librement
/// composables. Un scenario nomme est une COMPOSITION choisie depuis l'application (adc_cases).

namespace adc {

/// Modele physique compose : une brique HYPERBOLIQUE + une source + un second membre elliptique.
/// Satisfait le concept PhysicalModel ; les Vars (conversions + descripteur), le flux et les
/// vitesses d'onde viennent de l'hyperbolique ; pressure / wave_speeds sont exposes quand
/// l'hyperbolique les fournit (necessaire au flux HLLC / Roe).
///
/// CONTRAT : CompositeModel est une fonction pure de ses 3 briques, device-callable (ADC_HD).
/// Aucune MultiFab, aucune allocation, aucun acces global. Toute methode deleguee herite de
/// l'invariant device-clean de la brique sous-jacente.
///
/// Propagation n_aux : n_aux = max(aux_comps<Hyperbolic>, aux_comps<Source>, aux_comps<Elliptic>).
/// Le systeme dimensionne le canal aux en consequence. Une brique sans n_aux (defaut kAuxBaseComps=3)
/// ne modifie pas l'historique bit-identique.
template <class Hyperbolic, class Source, class Elliptic>
struct CompositeModel {
  static_assert(HyperbolicPhysicalModel<Hyperbolic>,
                "CompositeModel : la 1ere brique doit etre un modele HYPERBOLIQUE (Vars + "
                "conversions cons<->prim + flux + max_wave_speed), cf. HyperbolicPhysicalModel");
  using State = StateVec<Hyperbolic::n_vars>;
  using Prim = typename Hyperbolic::Prim;
  using Aux = adc::Aux;
  static constexpr int n_vars = Hyperbolic::n_vars;
  // Largeur du canal aux du modele compose = MAX des largeurs de ses briques : si une brique (flux
  // ou source) lit un champ auxiliaire supplementaire (p.ex. une source magnetisee declarant
  // n_aux=4 pour lire B_z), le compose l'expose au systeme (qui dimensionne alors le canal aux).
  // Sans brique a champ extra, n_aux = kAuxBaseComps (3) -> strictement identique a l'historique.
  static constexpr int n_aux = [] {
    int w = aux_comps<Hyperbolic>();
    if (aux_comps<Source>() > w) w = aux_comps<Source>();
    if (aux_comps<Elliptic>() > w) w = aux_comps<Elliptic>();
    return w;
  }();

  Hyperbolic hyp{};
  Source src{};
  Elliptic ell{};

  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return hyp.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return hyp.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State& u, const Aux& a) const { return src.apply(u, a); }
  ADC_HD Real elliptic_rhs(const State& u) const { return ell.rhs(u); }
  ADC_HD Prim to_primitive(const State& u) const { return hyp.to_primitive(u); }
  ADC_HD State to_conservative(const Prim& p) const { return hyp.to_conservative(p); }
  static VariableSet conservative_vars() { return Hyperbolic::conservative_vars(); }
  static VariableSet primitive_vars() { return Hyperbolic::primitive_vars(); }

  ADC_HD Real pressure(const State& u) const
    requires requires(const Hyperbolic h, const State s) { h.pressure(s); }
  {
    return hyp.pressure(u);
  }
  ADC_HD void wave_speeds(const State& u, const Aux& a, int dir, Real& smin, Real& smax) const
    requires requires(const Hyperbolic h, const State s, const Aux aa, int d, Real& lo,
                      Real& hi) { h.wave_speeds(s, aa, d, lo, hi); }
  {
    hyp.wave_speeds(u, a, dir, smin, smax);
  }

  /// CAPABILITIES Riemann (audit vague 3) : hooks HLLC (contact_speed + hllc_star_state) et Roe
  /// (roe_dissipation) forwardes depuis la brique HYPERBOLIQUE quand elle les declare (le DSL les
  /// emet via enable_hllc ; un modele C++ peut les ecrire a la main). Concept-gates comme
  /// pressure / wave_speeds : sans hooks, le compose ne les expose pas (chemins canoniques /
  /// rejets explicites inchanges).
  ADC_HD Real contact_speed(const State& ul, const State& ur, Real pl, Real pr, Real sl, Real sr,
                            int dir) const
    requires requires(const Hyperbolic h, const State a_, const State b_, Real p, Real q, Real x,
                      Real y, int d) { h.contact_speed(a_, b_, p, q, x, y, d); }
  {
    return hyp.contact_speed(ul, ur, pl, pr, sl, sr, dir);
  }
  ADC_HD State hllc_star_state(const State& u, Real p, Real s, Real sStar, int dir) const
    requires requires(const Hyperbolic h, const State a_, Real p_, Real s_, Real ss_, int d) {
      h.hllc_star_state(a_, p_, s_, ss_, d);
    }
  {
    return hyp.hllc_star_state(u, p, s, sStar, dir);
  }
  ADC_HD State roe_dissipation(const State& ul, const Aux& al, const State& ur, const Aux& ar,
                               int dir) const
    requires requires(const Hyperbolic h, const State a_, const Aux x_, const State b_,
                      const Aux y_, int d) { h.roe_dissipation(a_, x_, b_, y_, d); }
  {
    return hyp.roe_dissipation(ul, al, ur, ar, dir);
  }

  /// Terme source GEOMETRIQUE de courbure polaire, delegue a la brique hyperbolique quand elle
  /// l'expose (fluide polaire : IsothermalFluxPolar). Concept-gate comme pressure / wave_speeds :
  /// si l'hyperbolique ne le fournit pas (transport scalaire ExB polaire), CompositeModel ne
  /// l'expose pas -> assemble_rhs_polar retombe sur 0 (bit-identique). Ne touche PAS le cartesien
  /// (assemble_rhs ne l'appelle jamais).
  ADC_HD State polar_geom_source(const State& u, Real r) const
    requires requires(const Hyperbolic h, const State s, Real rr) { h.polar_geom_source(s, rr); }
  {
    return hyp.polar_geom_source(u, r);
  }

  /// BORNES DE PAS optionnelles (audit 2026-06, cf. core/physical_model.hpp) : forwardees
  /// conditionnellement comme pressure / wave_speeds, sinon le compose ne les expose pas et la
  /// politique de pas reste l'historique. stability_speed / stability_dt viennent de la brique
  /// HYPERBOLIQUE (c'est elle que le DSL emet) ; source_frequency vient de la brique SOURCE (c'est
  /// la source qui connait sa frequence de relaxation/collision).
  ADC_HD Real stability_speed(const State& u, const Aux& a, int dir) const
    requires requires(const Hyperbolic h, const State s, const Aux aa, int d) {
      h.stability_speed(s, aa, d);
    }
  {
    return hyp.stability_speed(u, a, dir);
  }
  ADC_HD Real stability_dt(const State& u, const Aux& a) const
    requires requires(const Hyperbolic h, const State s, const Aux aa) { h.stability_dt(s, aa); }
  {
    return hyp.stability_dt(u, a);
  }
  ADC_HD Real source_frequency(const State& u, const Aux& a) const
    requires requires(const Source sc, const State s, const Aux aa) { sc.frequency(s, aa); }
  {
    return src.frequency(u, a);
  }

  /// PROJECTION PONCTUELLE post-pas (ADC-177) : forwardee depuis la brique HYPERBOLIQUE quand elle
  /// declare project(U, aux) (le DSL l'emet via m.projection ; une brique native peut l'ecrire a la
  /// main). Concept-gate comme stability_speed : sans methode, le compose ne l'expose pas et le
  /// stepper n'applique aucune projection (chemin bit-identique).
  ADC_HD State project(const State& u, const Aux& a) const
    requires requires(const Hyperbolic h, const State s, const Aux aa) { h.project(s, aa); }
  {
    return hyp.project(u, a);
  }

  /// JACOBIEN ANALYTIQUE de la source (audit vague 3) : forwarde depuis la brique SOURCE quand
  /// elle declare jacobian(U, aux, J) (J[r][c] = dS_r/dU_c). Le Newton de la source implicite
  /// l'utilise a la place des differences finies (trait HasSourceJacobian) ; sans la methode,
  /// rien n'est expose et le Newton garde les differences finies historiques.
  ADC_HD void source_jacobian(const State& u, const Aux& a, Real (&J)[n_vars][n_vars]) const
    requires requires(const Source sc, const State s, const Aux aa,
                      Real (&JJ)[n_vars][n_vars]) { sc.jacobian(s, aa, JJ); }
  {
    src.jacobian(u, a, J);
  }
};

}  // namespace adc

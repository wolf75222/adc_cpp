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
};

}  // namespace adc

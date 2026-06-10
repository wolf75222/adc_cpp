#pragma once

#include <adc/physics/bricks.hpp>
#include <adc/runtime/model_spec.hpp>

#include <stdexcept>
#include <string>

/// @file
/// @brief Assemble un CompositeModel a partir d'une ModelSpec (briques + parametres).
///
/// Le coeur ne connait que des BRIQUES generiques ; un scenario est une composition,
/// nommee cote application (adc_cases). dispatch_model(spec, visitor) construit le
/// CompositeModel<Hyperbolic, Source, Elliptic> designe par la spec et appelle visitor(model).
/// Les combinaisons invalides (source fluide sur un transport scalaire) sont rejetees.

namespace adc::detail {

/// Construit la brique de transport et appelle v(transport).
template <class Visitor>
void dispatch_transport(const ModelSpec& m, Visitor&& v) {
  if (m.transport == "exb") return v(ExBVelocity{Real(m.B0)});
  if (m.transport == "compressible") return v(CompressibleFlux{Real(m.gamma)});
  if (m.transport == "isothermal") return v(IsothermalFlux{Real(m.cs2)});
  throw std::runtime_error("transport inconnu '" + m.transport +
                           "' (exb|compressible|isothermal)");
}

/// Construit la brique de source et appelle v(source). Les sources fluides (force) exigent
/// >= 3 variables : sur un transport scalaire (exb), seule "none" est valide.
///   - "none"                       : NoSource (neutre) ;
///   - "potential"                  : PotentialForce (q/m) rho E (electrostatique) ;
///   - "gravity"                    : GravityForce rho g ;
///   - "magnetic" | "lorentz"       : MagneticLorentzForce q v x B_z (B_z lu dans l'aux, regime
///                                    EXPLICITE ; le regime raide passe par le Schur condense) ;
///   - "potential_magnetic" | "potential_lorentz" : CompositeSource<PotentialForce, MagneticLorentz>
///                                    = electrostatique + Lorentz somme (force complete du diocotron
///                                    polaire NATIF, sans le contournement centrifuge).
/// qom (q/m, signe inclus) est partage par les deux forces chargees (meme espece). Les briques
/// magnetisees declarent n_aux = 4 -> CompositeModel remonte la largeur aux au systeme (canal B_z).
template <int NV, class Visitor>
void dispatch_source(const ModelSpec& m, Visitor&& v) {
  if (m.source == "none") return v(NoSource{});
  if constexpr (NV >= 3) {
    if (m.source == "potential") return v(PotentialForce{Real(m.qom)});
    if (m.source == "gravity") return v(GravityForce{});
    if (m.source == "magnetic" || m.source == "lorentz")
      return v(MagneticLorentzForce{Real(m.qom)});
    if (m.source == "potential_magnetic" || m.source == "potential_lorentz")
      return v(CompositeSource<PotentialForce, MagneticLorentzForce>{PotentialForce{Real(m.qom)},
                                                                     MagneticLorentzForce{Real(m.qom)}});
  }
  throw std::runtime_error("source '" + m.source +
                           "' invalide ici (exige un transport fluide >= 3 variables, ou 'none')");
}

/// Construit la brique de second membre elliptique et appelle v(elliptic).
template <class Visitor>
void dispatch_elliptic(const ModelSpec& m, Visitor&& v) {
  if (m.elliptic == "charge") return v(ChargeDensity{Real(m.q)});
  if (m.elliptic == "background") return v(BackgroundDensity{Real(m.alpha), Real(m.n0)});
  if (m.elliptic == "gravity")
    return v(GravityCoupling{Real(m.sign), Real(m.four_pi_G), Real(m.rho0)});
  throw std::runtime_error("elliptic inconnu '" + m.elliptic + "' (charge|background|gravity)");
}

/// Resolution AUTOMATIQUE par ROLES (audit §5) : remplit les indices de composantes d'une brique de
/// SOURCE ou d'ELLIPTIQUE (c_rho / c_mx / c_my / c_E) depuis le descripteur conservatif @p cons du
/// TRANSPORT. C'est une resolution TRANSPARENTE, sans aucun parametre utilisateur nouveau : les
/// briques natives s'adaptent au layout du transport (densite/qdm/energie reperees par leur ROLE et
/// non par un indice code en dur). Un index n'est ECRIT que si le role existe dans @p cons ; sinon la
/// brique GARDE son defaut canonique (comportement historique pour un transport sans roles).
///
/// Detection des membres par `requires` (if constexpr) : les briques ont des jeux d'indices
/// HETEROGENES (PotentialForce/GravityForce : rho/mx/my/E ; MagneticLorentzForce : mx/my seulement ;
/// ChargeDensity/Background/GravityCoupling : rho ; NoSource : aucun) ; seuls les membres EXISTANTS
/// sont touches. CompositeSource<A,B> n'a pas d'indices propres : on recurse dans ses deux sous-briques.
///
/// BIT-IDENTIQUE pour les transports NATIFS : Euler (rho=0, m_x=1, m_y=2, E=3), Isothermal
/// (rho=0, m_x=1, m_y=2) et ExB (densite=0) declarent des roles CANONIQUES -> les indices resolus ==
/// les defauts des briques -> aucune valeur ne change. Resolu A LA CONSTRUCTION (hote, std::string) ;
/// jamais en device.
template <class Brick>
void bind_variable_roles(Brick& brk, const VariableSet& cons) {
  const int i_rho = cons.index_of(VariableRole::Density);
  const int i_mx = cons.index_of(VariableRole::MomentumX);
  const int i_my = cons.index_of(VariableRole::MomentumY);
  const int i_E = cons.index_of(VariableRole::Energy);
  if constexpr (requires { brk.c_rho; }) { if (i_rho >= 0) brk.c_rho = i_rho; }
  if constexpr (requires { brk.c_mx; })  { if (i_mx >= 0) brk.c_mx = i_mx; }
  if constexpr (requires { brk.c_my; })  { if (i_my >= 0) brk.c_my = i_my; }
  if constexpr (requires { brk.c_E; })   { if (i_E >= 0) brk.c_E = i_E; }
  if constexpr (requires { brk.a; brk.b; }) {  // CompositeSource<A,B> : recursion dans les sous-briques
    bind_variable_roles(brk.a, cons);
    bind_variable_roles(brk.b, cons);
  }
}

/// Assemble le CompositeModel designe par @p m et appelle `visitor(model)`.
/// @throws std::runtime_error sur tag inconnu ou combinaison invalide.
template <class Visitor>
void dispatch_model(const ModelSpec& m, Visitor&& visitor) {
  dispatch_transport(m, [&](auto tr) {
    using TR = decltype(tr);
    // Roles du transport (hote) : sert a resoudre les indices des briques source / elliptique avant
    // de figer le composite. Transport natif -> roles canoniques -> indices resolus == defauts.
    const VariableSet cons = TR::conservative_vars();
    dispatch_source<TR::n_vars>(m, [&](auto src) {
      dispatch_elliptic(m, [&](auto ell) {
        bind_variable_roles(src, cons);  // resolution AUTOMATIQUE par roles (transparente, bit-identique)
        bind_variable_roles(ell, cons);
        visitor(CompositeModel<TR, decltype(src), decltype(ell)>{tr, src, ell});
      });
    });
  });
}

}  // namespace adc::detail

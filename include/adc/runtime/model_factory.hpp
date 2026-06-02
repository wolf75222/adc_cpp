#pragma once

#include <adc/model/bricks.hpp>
#include <adc/runtime/model_spec.hpp>

#include <stdexcept>
#include <string>

/// @file
/// @brief Assemble un CompositeModel a partir d'une ModelSpec (briques + parametres).
///
/// Le coeur ne connait que des BRIQUES generiques ; un scenario est une composition,
/// nommee cote application (adc_cases). dispatch_model(spec, visitor) construit le
/// CompositeModel<Transport, Source, Elliptic> designe par la spec et appelle visitor(model).
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
template <int NV, class Visitor>
void dispatch_source(const ModelSpec& m, Visitor&& v) {
  if (m.source == "none") return v(NoSource{});
  if constexpr (NV >= 3) {
    if (m.source == "potential") return v(PotentialForce{Real(m.qom)});
    if (m.source == "gravity") return v(GravityForce{});
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

/// Assemble le CompositeModel designe par @p m et appelle `visitor(model)`.
/// @throws std::runtime_error sur tag inconnu ou combinaison invalide.
template <class Visitor>
void dispatch_model(const ModelSpec& m, Visitor&& visitor) {
  dispatch_transport(m, [&](auto tr) {
    using TR = decltype(tr);
    dispatch_source<TR::n_vars>(m, [&](auto src) {
      dispatch_elliptic(m, [&](auto ell) {
        visitor(CompositeModel<TR, decltype(src), decltype(ell)>{tr, src, ell});
      });
    });
  });
}

}  // namespace adc::detail

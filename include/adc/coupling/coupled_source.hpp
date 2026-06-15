/// @file
/// @brief CoupledSourceFor / NoCoupledSource : contrat d'une source de COUPLAGE inter-especes.
///
/// model.source(U, aux) est PUREMENT LOCALE (etat du seul bloc). Un plasma multi-especes echange
/// entre especes (collisions, transfert de charge, friction) : S_e depend de U_i, etc. Ce terme ne
/// rentre PAS dans le PhysicalModel local : c'est une responsabilite du niveau systeme. Une
/// CoupledSource lit l'etat de PLUSIEURS blocs (+ aux = phi, grad phi) et met a jour les blocs sur un
/// pas dt, via le contrat minimal apply(system, aux, dt). Le squelette l'applique par splitting
/// (forward-Euler additif, SystemCoupler::coupled_source_step). Les sources concretes vivent dans
/// adc_cases / les tests (le coeur reste zero-modele).

#pragma once

#include <adc/core/coupled_system.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/multifab.hpp>

// Source de COUPLAGE inter-especes.
//
// model.source(U, aux) est PUREMENT LOCALE : elle ne voit que l'etat du bloc ou
// elle vit. Or un plasma multi-especes echange entre especes (collisions,
// transfert de charge, friction, echange q/m) : S_e depend de U_i, S_i depend de
// U_e, et les deux peuvent dependre de phi (via aux). Ce terme ne rentre PAS dans
// le PhysicalModel local ; c'est une responsabilite du niveau systeme.
//
// Une CoupledSource lit l'etat de PLUSIEURS blocs (+ aux = phi, grad phi) et met a
// jour les blocs sur un pas dt. Le squelette l'applique par splitting (un pas
// additif forward-Euler, via SystemCoupler::coupled_source_step) : c'est le point
// de branchement ou viendront ensuite les sources raides traitees implicitement
// (cf. time/implicit_stepper.hpp) sans changer le contrat.
//
// Le contrat est volontairement minimal : `apply(system, aux, dt)`. Les sources
// concretes (echange lineaire, collisions, ...) vivent dans adc_cases / les tests,
// pas dans le coeur (qui reste zero-modele).

namespace adc {

// Contrat d'une source de couplage pour un systeme donne.
/// Concept : C est une source de couplage valide pour System si System est un CoupledSystem et si C
/// expose apply(System&, const MultiFab& aux, Real dt) (met a jour les blocs sur le pas dt).
template <class C, class System>
concept CoupledSourceFor =
    CoupledSystemLike<System> &&
    requires(const C c, System& s, const MultiFab& aux, Real dt) {
      c.apply(s, aux, dt);
    };

// Defaut : aucune source de couplage. Cas mono-espece, ou couplage assure
// uniquement par Poisson (le champ). No-op, aucun cout.
/// Source de couplage NULLE (defaut) : apply() est un no-op. Cas mono-espece ou couplage assure
/// uniquement par le champ (Poisson). Aucun cout.
struct NoCoupledSource {
  template <CoupledSystemLike System>
  void apply(System&, const MultiFab&, Real) const {}
};

}  // namespace adc

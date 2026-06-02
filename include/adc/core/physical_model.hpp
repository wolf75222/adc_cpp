#pragma once

#include <adc/core/state.hpp>  // Aux : le contrat fixe l'auxiliaire a adc::Aux
#include <adc/core/types.hpp>

#include <concepts>

// Le contrat de la couche physique.
//
// Un PhysicalModel decrit UNE equation : ses formules ponctuelles. Rien de
// plus. C'est le seul axe "quoi calculer" de l'architecture, separe de l'axe
// "ou / comment iterer" (maillage + dispatch) et de l'axe "dans quel ordre"
// (integrateur + coupleur).
//
// Tout est fonction pure d'etats ponctuels :
//   - flux(U, aux, dir)          : le flux physique dans la direction dir
//   - max_wave_speed(U, aux, dir): la plus grande vitesse d'onde (pour le CFL
//                                  et le solveur de Riemann)
//   - source(U, aux)             : le terme source ponctuel
//   - elliptic_rhs(U)            : le second membre de l'equation elliptique
//                                  (densite de charge / de masse selon le modele)
//
// flux ET source prennent aux : c'est le point qui unifie le transport a derive
// (aux dans le flux) et le fluide compressible auto-gravitant (aux dans la
// source) sous un meme operateur spatial.
//
// Contrat Aux (tranche, cf. TODO 4) : l'auxiliaire est FIXE a adc::Aux (phi, grad
// phi). C'est ce que load_aux construit et tout ce que l'operateur spatial fournit ;
// le concept l'exige donc explicitement (M::Aux == adc::Aux) plutot que de laisser
// croire qu'un modele pourrait declarer un auxiliaire arbitraire que le code ne
// remplirait jamais. Generaliser load_aux<Model> a un Model::Aux quelconque reste
// possible plus tard ; en attendant le contrat dit exactement ce que le code donne.

namespace adc {

template <class M>
concept PhysicalModel =
    requires(const M m, const typename M::State u, const typename M::Aux a,
             int dir) {
      typename M::State;
      typename M::Aux;
      requires std::same_as<typename M::Aux, Aux>;
      { M::n_vars } -> std::convertible_to<int>;
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;
      { m.source(u, a) } -> std::same_as<typename M::State>;
      { m.elliptic_rhs(u) } -> std::convertible_to<Real>;
    };

}  // namespace adc

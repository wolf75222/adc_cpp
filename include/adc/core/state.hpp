/// @file
/// @brief Types ponctuels de la couche physique : StateVec<N> (etat conserve) et Aux (champs
///        auxiliaires issus du solveur elliptique). Les indices canoniques de Aux sont definis ici
///        et doivent rester en phase avec AUX_CANONICAL dans python/adc/dsl.py (duplication
///        inherente : Python ne lit pas les en-tetes C++).

#pragma once

#include <adc/core/types.hpp>

// State et Aux : les deux types ponctuels manipules par la couche physique.
//
// Regle d'architecture : ce sont des agregats trivialement copiables (POD).
// Un PhysicalModel ne voit jamais autre chose que ca. Aucune box, aucun rang
// MPI, aucune vue Kokkos n'entre ici. C'est ce qui rend la physique portable :
// le meme code compile pour CPU et device parce qu'il ne touche a aucun
// parallelisme.

namespace adc {

/// Vecteur d'etat conserve de taille fixe, connue a la compilation.
///
/// Exemples : StateVec<1> pour un scalaire (advection), StateVec<4> pour Euler 2D.
/// La valeur N pilote PhysicalModel::n_vars.
///
/// INVARIANT device : tableau C brut (`Real v[N]`), pas std::array; trivialement
/// copiable, device-clean (ADC_HD). Aucun constructeur non-trivial.
template <int N>
struct StateVec {
  Real v[N]{};  // tableau C : trivialement utilisable sur device (pas std::array)

  ADC_HD Real& operator[](int i) { return v[i]; }
  ADC_HD Real operator[](int i) const { return v[i]; }

  ADC_HD static constexpr int size() { return N; }
};

/// @name Arithmetic operators for StateVec (ADC_HD, device-clean).
/// @{
template <int N>
ADC_HD StateVec<N> operator+(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i) a[i] += b[i];
  return a;
}

template <int N>
ADC_HD StateVec<N> operator-(StateVec<N> a, const StateVec<N>& b) {
  for (int i = 0; i < N; ++i) a[i] -= b[i];
  return a;
}

template <int N>
ADC_HD StateVec<N> operator*(Real s, StateVec<N> a) {
  for (int i = 0; i < N; ++i) a[i] *= s;
  return a;
}
/// @}

// Champs auxiliaires derives de la resolution elliptique : le potentiel et
// son gradient au point. C'est le canal unique par lequel le couplage entre
// dans la physique. Il alimente A LA FOIS le flux (transport a derive : la
// vitesse E x B vient de grad phi) et la source (fluide compressible
// auto-gravitant : S = -rho grad phi). Cette dualite est ce qui permet a un seul
// operateur spatial de servir les deux problemes cibles.
//
// Canal aux EXTENSIBLE (cf. aux_comps()/load_aux dans spatial_operator.hpp). Les trois
// premieres composantes sont le contrat de BASE, identique a l'historique :
//   [0] = phi, [1] = grad_x, [2] = grad_y.
// Les suivantes sont des champs auxiliaires SUPPLEMENTAIRES, optionnels, dans un ordre
// canonique fixe ([3] = B_z, [4] = T_e, ...). Un modele declare combien de composantes il lit
// via un membre statique n_aux (defaut kAuxBaseComps = 3) ; un modele sans n_aux ne lit jamais
// les champs extra et reste strictement bit-identique. Les champs extra valent 0 par defaut :
// load_aux ne les ecrase que si le modele les demande.
//
/// SOURCE UNIQUE de la disposition des champs aux EXTRA (X-macro). C'est le SEUL endroit
/// listant {membre, indice} pour les champs au-dela du contrat de base.
/// INVARIANT Python-C++ : les indices ici doivent rester identiques a AUX_CANONICAL dans
/// python/adc/dsl.py ({"phi":0,"grad_x":1,"grad_y":2,"B_z":3,"T_e":4}). Modifier l'un
/// exige de modifier l'autre simultanement. La duplication est inherente : Python ne lit
/// pas les en-tetes C++.
// SOURCE UNIQUE de la disposition des champs aux EXTRA (X-macro). C'est le SEUL endroit
// listant {membre, indice} pour les champs au-dela du contrat de base. load_aux (lecture
// device, spatial_operator.hpp) ET le marshaling hote (python/system.cpp) en sont GENERES,
// donc ajouter un champ aux extra se fait ICI et NULLE PART AILLEURS. Cela ferme le trou
// historique (#51 : T_e ajoute au struct + load_aux mais oublie dans le marshaling JIT ->
// lu comme 0 en silence). Chaque entree : X(membre, indice). L'indice DOIT etre >= 3
// (les composantes 0..2 sont phi/grad_x/grad_y, cablees dans le constructeur de base) et
// suivre la disposition canonique partagee avec AUX_CANONICAL cote DSL (python/adc/dsl.py),
// duplication inherente : Python ne lit pas les en-tetes C++. Pour ajouter un champ :
// 1 ligne ici (et la ligne miroir dans AUX_CANONICAL). Pur preprocesseur -> device-clean
// (nvcc/Kokkos), aucune reflection C++26.
#define ADC_AUX_FIELDS(X) \
  X(B_z, 3)               \
  X(T_e, 4)

struct Aux {
  Real phi{};     // potentiel       (composante aux 0)
  Real grad_x{};  // d phi / d x     (composante aux 1)
  Real grad_y{};  // d phi / d y     (composante aux 2)
  // Membres EXTRA generes depuis ADC_AUX_FIELDS (source unique). B_z = champ B hors-plan
  // fourni par le systeme (comp 3) ; T_e = temperature electronique p/rho d'un bloc fluide
  // (comp 4). Tous optionnels, a 0 par defaut.
#define ADC_AUX_DECL(name, idx) Real name{};
  ADC_AUX_FIELDS(ADC_AUX_DECL)
#undef ADC_AUX_DECL
};

// Largeur du canal aux du contrat de base (phi, grad phi). Un modele lisant des champs
// supplementaires declare un n_aux plus grand ; cf. aux_comps()/load_aux().
inline constexpr int kAuxBaseComps = 3;

// Garde-fou : les indices declares dans ADC_AUX_FIELDS sont strictement EXTRA (>= base) et
// commencent juste apres le contrat de base. Verifie a la compilation que la table reste
// coherente avec kAuxBaseComps (le 1er champ extra est a l'indice kAuxBaseComps).
#define ADC_AUX_IDX_CHECK(name, idx) static_assert((idx) >= kAuxBaseComps, \
    "champ aux extra '" #name "' : indice doit etre >= kAuxBaseComps (3)");
ADC_AUX_FIELDS(ADC_AUX_IDX_CHECK)
#undef ADC_AUX_IDX_CHECK

}  // namespace adc

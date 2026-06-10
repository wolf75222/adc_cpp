/// @file
/// @brief Concepts C++20 definissant le contrat de la couche physique.
///
/// Hierarchy des concepts :
///   PhysicalModel : contrat minimal (flux, source, vitesse d'onde, RHS elliptique).
///   HasPrimitiveVars : extension optionnelle (variables primitives + conversions cons<->prim).
///   HyperbolicPhysicalModel : brique hyperbolique complete (flux + conversions + Variables).
///   HyperbolicModel : alias de compat pour HyperbolicPhysicalModel.
///
/// INVARIANT Aux : tout PhysicalModel recoit un adc::Aux (phi, grad phi, champs extra).
/// Generaliser l'auxiliaire a un Model::Aux quelconque est possible plus tard; aujourd'hui
/// le contrat dit exactement ce que load_aux construit.
///
/// INVARIANT device : les methodes du concept (flux, source, ...) doivent etre ADC_HD si
/// elles sont appelees dans des kernels. Le concept ne le verifie pas -- c'est la responsabilite
/// de l'auteur du modele.

#pragma once

#include <adc/core/state.hpp>  // Aux : le contrat fixe l'auxiliaire a adc::Aux
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>  // Variables : contrat obligatoire du modele hyperbolique

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

/// Largeur du canal aux qu'un modele CONSOMME.
///
/// Retourne `M::n_aux` si le modele le declare (champs extra : B_z, T_e...), sinon
/// `kAuxBaseComps` (= 3 : phi/grad_x/grad_y). Pilote le nombre de composantes que
/// load_aux lit et que le systeme alloue. Un modele sans n_aux -> 3 -> bit-identique
/// a l'historique (champs extra de Aux a 0, jamais lus).
/// Vit dans ce header (contrat) et non dans l'operateur spatial, pour que
/// CompositeModel puisse propager n_aux sans tirer toute la numerique.
// Largeur du canal aux qu'un modele CONSOMME : Model::n_aux s'il le declare (champs auxiliaires
// supplementaires au-dela de phi/grad : B_z, ...), sinon kAuxBaseComps (= 3, le contrat de base
// phi/grad_x/grad_y). Pilote le nombre de composantes que load_aux lit et que le systeme doit
// allouer/peupler. Un modele sans n_aux -> 3 -> strictement identique a l'historique (champs extra
// de Aux a 0, jamais lus). Vit dans le header CONTRAT (et non l'operateur spatial) pour que
// CompositeModel puisse propager n_aux sans tirer toute la numerique.
template <class M>
constexpr int aux_comps() {
  if constexpr (requires { M::n_aux; })
    return M::n_aux;
  else
    return kAuxBaseComps;
}

/// Contrat minimal d'un modele physique.
///
/// Exige : State, Aux == adc::Aux, n_vars, flux(u,a,dir), max_wave_speed(u,a,dir),
/// source(u,a), elliptic_rhs(u). Toutes ces methodes doivent etre ADC_HD si appelees
/// dans des kernels (non verifie par le concept; responsabilite de l'auteur).
/// Ne pas confondre avec HyperbolicPhysicalModel qui ajoute les variables et conversions.
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

// ---------------------------------------------------------------------------------------------
// BORNES DE PAS DE TEMPS OPTIONNELLES du contrat modele (audit 2026-06, chantier "step_cfl").
//
// Historiquement, step_cfl ne connaissait QUE la borne de transport hyperbolique
// dt <= cfl * h / max_wave_speed. Or un modele peut imposer d'autres bornes : frequence de source
// (collision/reaction, mu = eig(dS/dU), unite 1/temps, SANS h), ou directement un pas admissible
// (formule couplee transport-source non reductible). Ces trois traits OPTIONNELS permettent au
// modele de les declarer ; un modele qui n'en declare aucun garde STRICTEMENT le comportement
// historique (fallback max_wave_speed, bit-identique).
//
// SEMANTIQUE (toutes les bornes s'appliquent au SOUS-PAS EFFECTIF stride*dt/substeps du bloc,
// cf. SystemStepper::step_cfl) :
//  - stability_speed(U, aux, dir) : vitesse de stabilite lambda* [longueur/temps] qui REMPLACE
//    max_wave_speed dans la reduction CFL du bloc (dt <= cfl * h / max_cellules(lambda*)). Pour
//    quand la vitesse pertinente pour la STABILITE n'est pas la vitesse d'onde physique (borne
//    conservative declaree, vitesse modifiee par un couplage...). Les solveurs de Riemann, eux,
//    continuent de lire max_wave_speed (precision != stabilite).
//  - source_frequency(U, aux) : frequence locale mu [1/temps] de la source/du couplage local ;
//    impose dt <= cfl / max_cellules(mu) -- PAS de h (la borne de source est sans dimension
//    d'espace). Raccourci pour relaxation/collision/reaction explicites.
//  - stability_dt(U, aux) : pas ADMISSIBLE direct [temps] par cellule ; impose
//    dt <= min_cellules(stability_dt). Le cfl N'EST PAS applique (le modele declare deja un pas
//    admissible ; appliquer cfl en plus melangerait deux marges). C'est la forme la plus generale.
//
// STABILITE vs PRECISION : ces traits declarent des bornes de STABILITE. Une source traitee en
// implicite (SourceImplicit/IMEX) peut ne plus imposer de borne de stabilite tout en gardant une
// contrainte de PRECISION : c'est au modele de choisir ce que stability_dt/source_frequency
// retournent dans ce cas (ou de ne pas les declarer). Les bornes NON locales (couplage
// multi-blocs, Schur/Poisson, AMR/scheduler) ne passent PAS par ces traits cellule-par-cellule :
// elles passent par System::add_dt_bound (borne globale hote, une evaluation par pas).
//
// PRODUCTION GPU/MPI : comme flux/source, ces methodes doivent etre ADC_HD (evaluees dans des
// kernels de reduction) -- une callback Python par cellule n'est pas un chemin de production ;
// le DSL les compile (m.stability_speed(...) / m.stability_dt(...)).
// ---------------------------------------------------------------------------------------------

/// Trait OPTIONNEL : vitesse de stabilite lambda* remplacant max_wave_speed dans la CFL du bloc.
template <class M>
concept HasStabilitySpeed =
    requires(const M m, const typename M::State u, const Aux a, int dir) {
      { m.stability_speed(u, a, dir) } -> std::convertible_to<Real>;
    };

/// Trait OPTIONNEL : frequence locale de source mu [1/s] (borne dt <= cfl / max mu, sans h).
template <class M>
concept HasSourceFrequency = requires(const M m, const typename M::State u, const Aux a) {
  { m.source_frequency(u, a) } -> std::convertible_to<Real>;
};

/// Trait OPTIONNEL : pas admissible direct par cellule (borne dt <= min stability_dt, sans cfl).
template <class M>
concept HasStabilityDt = requires(const M m, const typename M::State u, const Aux a) {
  { m.stability_dt(u, a) } -> std::convertible_to<Real>;
};

/// Extension OPTIONNELLE d'un PhysicalModel : variables primitives + conversions cons<->prim. Permet a
// l'operateur spatial de reconstruire en variables primitives (rho, u, p) plutot que
// conservatives (plus robuste pour Euler : positivite de rho et p), et centralise le
// passage cons <-> prim (la vitesse d'onde, les termes de collision u_a - u_b s'expriment
// naturellement en primitif). Un modele qui ne l'expose pas reconstruit en conservatif.
/// INVARIANT : to_primitive/to_conservative doivent etre inverses l'une de l'autre.
/// L'operateur spatial reconstruit alors en primitif (plus robuste : positivite rho/p).
template <class M>
concept HasPrimitiveVars =
    PhysicalModel<M> &&
    requires(const M m, const typename M::State u, const typename M::Prim p) {
      typename M::Prim;
      { m.to_primitive(u) } -> std::same_as<typename M::Prim>;
      { m.to_conservative(p) } -> std::same_as<typename M::State>;
    };

/// Brique hyperbolique d'un modele : flux + vitesse d'onde + variables + conversions cons<->prim.
///
/// Variables, conversions et flux sont physiquement LIES (un flux est ecrit pour une disposition
/// de variables donnee) : ils forment une seule brique distincte de la source et de l'elliptique.
/// conservative_vars() et primitive_vars() sont OBLIGATOIRES (pas un extra optionnel).
/// Aucune source ni RHS elliptique ici : ce sont d'autres briques de CompositeModel.
// Contrat de la brique HYPERBOLIQUE (la "partie hyperbolique" d'un modele) : elle porte les
// VARIABLES (conservatives U, primitives P + conversions + descripteur Variables), le FLUX et la
// vitesse d'onde. Vars, conversions et flux sont physiquement LIES (un flux est ecrit pour une
// disposition de variables donnee) : on les regroupe dans une seule brique, distincte de la source
// et de l'elliptique. PAS de source ni de second membre elliptique ici (ce sont d'autres briques de
// CompositeModel). conservative_vars() / primitive_vars() sont OBLIGATOIRES (contrat, pas un extra).
template <class M>
concept HyperbolicPhysicalModel =
    requires(const M m, const typename M::State u, const typename M::Prim p, const Aux a, int dir) {
      typename M::State;
      typename M::Prim;
      { M::n_vars } -> std::convertible_to<int>;
      { m.flux(u, a, dir) } -> std::same_as<typename M::State>;
      { m.max_wave_speed(u, a, dir) } -> std::convertible_to<Real>;
      { m.to_primitive(u) } -> std::same_as<typename M::Prim>;
      { m.to_conservative(p) } -> std::same_as<typename M::State>;
      { M::conservative_vars() } -> std::same_as<VariableSet>;
      { M::primitive_vars() } -> std::same_as<VariableSet>;
    };

/// Ancien nom (compat) : HyperbolicPhysicalModel etait `HyperbolicModel`.
template <class M>
concept HyperbolicModel = HyperbolicPhysicalModel<M>;

}  // namespace adc

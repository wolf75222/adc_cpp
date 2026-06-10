#pragma once

#include <adc/core/physical_model.hpp>  // aux_comps<> : propagation du canal aux d'une source composee
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>

/// @file
/// @brief Briques de SOURCE S(U, aux) : terme local, generique sur la taille d'etat (travail sur
///        l'energie si 4 variables). NoSource, PotentialForce ((q/m) rho E), GravityForce (rho g),
///        MagneticLorentzForce (q v x B_z). Composables comme parametre Source d'un CompositeModel
///        (physics/composite.hpp) ; CompositeSource<A, B> SOMME deux sources (electrostatique +
///        Lorentz). Les sources INTER-especes (ionisation, collision) vivent au niveau du systeme
///        (operator-split).

namespace adc {

// CONTRAT OPTIONNEL frequency(U, aux) -> Real (audit 2026-06, chantier step_cfl) : une brique
// source peut declarer sa FREQUENCE locale mu [1/s] (taux de relaxation/collision/reaction). Quand
// la brique l'expose, CompositeModel la forwarde (source_frequency) et System::step_cfl impose la
// borne dt <= cfl * substeps / (stride * max_cellules(mu)) -- la "deuxieme CFL" (source) du
// meeting, distincte de la CFL de transport (pas de h : une source borne en 1/temps). Une brique
// SANS frequency (toutes celles de ce fichier aujourd'hui) ne contraint pas le pas (historique).
// Doit etre ADC_HD (evaluee dans un kernel de reduction).

/// Pas de source : S(U, aux) = 0. Brique neutre (modele sans couplage potentiel/gravite).
/// Device-callable, aucun etat interne.
struct NoSource {
  template <class State>
  ADC_HD State apply(const State&, const Aux&) const {
    return State{};
  }
};

/// Force du potentiel electrostatique (q/m) rho E sur la quantite de mouvement (+ travail
/// sur l'energie si 4 variables). E = -grad phi = -(aux.grad_x, aux.grad_y).
///
/// CONTRAT : brique SOURCE ponctuelle, device-callable (ADC_HD), aucun etat global.
/// Formule : s[1] += qom*rho*Ex, s[2] += qom*rho*Ey, s[3] += qom*(rho_u*Ex + rho_v*Ey)
/// (le terme travail s[3] n'est actif que si State::size() == 4 : Euler compressible).
/// Invariant : u[0] = rho (composante 0 = densite, indice stable entre briques).
struct PotentialForce {
  Real qom = 1;  // q/m (signe inclus)
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    const Real Ex = -a.grad_x, Ey = -a.grad_y;
    State s{};
    s[1] = qom * u[0] * Ex;
    s[2] = qom * u[0] * Ey;
    if constexpr (State::size() == 4) s[3] = qom * (u[1] * Ex + u[2] * Ey);
    return s;
  }
};

/// Force gravitationnelle rho g (+ travail si 4 variables). g = -grad phi.
///
/// CONTRAT : brique SOURCE ponctuelle, device-callable (ADC_HD), aucun etat global.
/// Formule : s[1] += rho*gx, s[2] += rho*gy, s[3] += rho_u*gx + rho_v*gy
/// (le terme travail s[3] n'est actif que si State::size() == 4 : Euler compressible).
/// Pas de coefficient q/m (contrairement a PotentialForce) : g est la gravite directement.
struct GravityForce {
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    const Real gx = -a.grad_x, gy = -a.grad_y;
    State s{};
    s[1] = u[0] * gx;
    s[2] = u[0] * gy;
    if constexpr (State::size() == 4) s[3] = u[1] * gx + u[2] * gy;
    return s;
  }
};

/// Force de Lorentz MAGNETIQUE q (v x B) sur la quantite de mouvement, champ B = B_z z_hat hors-plan.
/// Regime EXPLICITE (omega_c modere) : terme ponctuel ALGEBRIQUE (aucune derivee), code une fois pour
/// les DEUX geometries car il est INVARIANT par orientation du repere local orthonorme (x,y) ou
/// (e_r, e_theta) :
///   (rho v_x, rho v_y) x B_z z_hat = (+B_z rho v_y, -B_z rho v_x) = (+B_z m_y, -B_z m_x)  [cartesien]
///   (rho v_r, rho v_th) x B_z z_hat = (+B_z rho v_th, -B_z rho v_r) = (+B_z m_th, -B_z m_r)  [polaire]
/// Donc s[1] = +qom*B_z*m[2], s[2] = -qom*B_z*m[1] avec m[1]=u[1] (1ere composante de qdm),
/// m[2]=u[2] (2nde). v x B est PERPENDICULAIRE a v : le travail F . v = 0 -> s[3] (energie) reste NUL
/// meme a 4 variables (la force magnetique ne change pas l'energie cinetique). qom = q/m (signe inclus,
/// coherent avec PotentialForce) ; le sens de giration (cyclotron) suit le signe de qom*B_z.
///
/// CONTRAT : brique SOURCE ponctuelle, device-callable (ADC_HD), aucun etat global. Lit B_z dans l'aux
/// (composante canonique 3, comme set_magnetic_field le peuple) -> declare n_aux = 4 pour que le canal
/// aux soit dimensionne et que load_aux remplisse a.B_z. Le regime RAIDE (omega_c grand) passe par le
/// Schur condense (ElectrostaticLorentzCondensation), PAS par cette brique explicite.
/// PRECONDITION : exige un transport fluide >= 3 variables (qdm sur 2 axes) ; sans objet sur scalaire.
struct MagneticLorentzForce {
  Real qom = 1;             // q/m (signe inclus)
  static constexpr int n_aux = 4;  // lit B_z (canal aux extra, indice canonique 3)
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    static_assert(State::size() >= 3,
                  "MagneticLorentzForce : exige un transport fluide >= 3 variables (qdm sur 2 axes)");
    const Real c = qom * a.B_z;
    State s{};
    s[1] = c * u[2];   // +qom B_z m_(y/theta)
    s[2] = -c * u[1];  // -qom B_z m_(x/r)
    // s[3] (energie) reste 0 : v x B est perpendiculaire a v, travail nul.
    return s;
  }
};

/// SOMME de deux briques source : S(U, aux) = A.apply(U, aux) + B.apply(U, aux). Permet de COMPOSER
/// plusieurs forces ponctuelles dans l'UNIQUE slot Source du CompositeModel (p.ex. electrostatique
/// PotentialForce + magnetique MagneticLorentzForce pour le diocotron). Generique sur la taille d'etat
/// (l'addition de StateVec est definie composante a composante, cf. core/state.hpp).
///
/// CONTRAT : brique SOURCE ponctuelle, device-callable (ADC_HD), aucun etat global au-dela des deux
/// sous-briques (elles-memes POD). PROPAGE le canal aux : n_aux = max(aux_comps<A>, aux_comps<B>) ->
/// si une sous-brique lit B_z (n_aux=4), le compose l'expose et CompositeModel le remonte au systeme.
template <class A, class B>
struct CompositeSource {
  A a{};
  B b{};
  static constexpr int n_aux = aux_comps<A>() > aux_comps<B>() ? aux_comps<A>() : aux_comps<B>();
  template <class State>
  ADC_HD State apply(const State& u, const Aux& ax) const {
    return a.apply(u, ax) + b.apply(u, ax);
  }
};

}  // namespace adc

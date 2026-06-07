#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/core/variables.hpp>
#include <adc/physics/euler.hpp>  // Euler : reutilise comme brique hyperbolique CompressibleFlux

#include <cmath>

/// @file
/// @brief Briques HYPERBOLIQUES generiques : Vars (cons U / prim P + conversions + descripteur) +
///        flux + vitesses d'onde. Chacune satisfait le concept HyperbolicPhysicalModel : State, Prim,
///        n_vars, flux, max_wave_speed, to_primitive/to_conservative, conservative_vars/primitive_vars
///        (+ pressure/wave_speeds si flux HLLC). Source et second membre elliptique sont des briques
///        SEPAREES (physics/source.hpp, physics/elliptic.hpp) ; CompositeModel (physics/composite.hpp)
///        les assemble. ExBVelocity (1 var), CompressibleFlux (= Euler, 4 var), IsothermalFlux (3 var).

namespace adc {

/// Advection scalaire par la derive E x B : v = (-d_y phi, d_x phi)/B0 (a divergence nulle).
///
/// Brique HYPERBOLIQUE 1-variable (densite scalaire n). Satisfait HyperbolicPhysicalModel.
/// CONTRAT : fonctions purement ponctuelles, device-callables (ADC_HD). Aucune MultiFab,
/// aucune allocation, aucun acces global. La divergence nulle de la derive E x B assure
/// la conservation exacte (pas de terme de compression dans ce flux).
/// Variables cons = prim = {n} (scalaire : pas de conversion nontriviale).
struct ExBVelocity {
  static constexpr int n_vars = 1;
  using State = StateVec<1>;
  Real B0 = 1;
  ADC_HD Real velocity(const Aux& a, int dir) const {
    return (dir == 0) ? (-a.grad_y / B0) : (a.grad_x / B0);
  }
  ADC_HD StateVec<1> flux(const StateVec<1>& u, const Aux& a, int dir) const {
    StateVec<1> f{};
    f[0] = u[0] * velocity(a, dir);
    return f;
  }
  ADC_HD Real max_wave_speed(const StateVec<1>&, const Aux& a, int dir) const {
    const Real d = velocity(a, dir);
    return d < 0 ? -d : d;
  }
  /// Spectre : une onde, la vitesse de derive dans la direction dir.
  ADC_HD StateVec<1> eigenvalues(const StateVec<1>&, const Aux& a, int dir) const {
    StateVec<1> e{};
    e[0] = velocity(a, dir);
    return e;
  }
  // Scalaire : variables primitives = conservatives (densite transportee).
  using Prim = StateVec<1>;
  ADC_HD Prim to_primitive(const StateVec<1>& u) const { return u; }
  ADC_HD StateVec<1> to_conservative(const Prim& p) const { return p; }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"n"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"n"}, 1, {VariableRole::Density}};
  }
};

/// Advection scalaire par la derive E x B en coordonnees POLAIRES (r, theta) -- chantier "grille
/// polaire annulaire", Phase 1. C'est une brique SEPAREE d'ExBVelocity (cartesienne), pas une
/// modification : le solveur polaire (assemble_rhs_polar) l'utilise sur une PolarGeometry.
///
/// DISPOSITION DU CANAL aux EN POLAIRE (documentee, contrat de cette brique) -- les composantes de
/// base [0..2] portent le champ E dans la BASE LOCALE ORTHONORMEE (e_r, e_theta) :
///   aux.phi    [0] = phi (potentiel ; inutilise par le flux, present pour symetrie)
///   aux.grad_x [1] = grad_r     = d phi / d r            (composante radiale de grad phi)
///   aux.grad_y [2] = grad_theta = (1/r) d phi / d theta  (composante AZIMUTALE PHYSIQUE de grad phi)
/// On REUTILISE les deux emplacements grad_x/grad_y de adc::Aux pour grad_r/grad_theta (pas de
/// nouveau champ aux) ; le SENS est polaire et porte par cette brique seule. grad_theta est la
/// derivee PHYSIQUE (deja divisee par r) : ainsi la vitesse ci-dessous est symetrique de la
/// cartesienne (vr <- -grad_theta/B, vtheta <- grad_r/B) et l'appelant qui remplit aux porte le 1/r.
///
/// VITESSE E x B EN POLAIRE (composantes PHYSIQUES dans la base locale) :
///   v_r     = -(1/(B r)) d phi/d theta = -grad_theta / B   (dir == 0, radiale)
///   v_theta =  (1/B)     d phi/d r     =  grad_r     / B   (dir == 1, azimutale)
/// Le flux rendu (dir 0 = F_r = n v_r ; dir 1 = F_theta = n v_theta) est PHYSIQUE ; la metrique 1/r
/// et la divergence (1/r) d_r(r F_r) + (1/r) d_theta(F_theta) sont portees par assemble_rhs_polar,
/// PAS par cette brique. La brique reste ainsi une physique pure (aucune box, aucun r).
struct ExBVelocityPolar {
  static constexpr int n_vars = 1;
  using State = StateVec<1>;
  Real B0 = 1;
  /// Composante PHYSIQUE de la vitesse de derive dans la direction d'indice dir (0 = r, 1 = theta).
  ADC_HD Real velocity(const Aux& a, int dir) const {
    return (dir == 0) ? (-a.grad_y / B0) : (a.grad_x / B0);
  }
  ADC_HD StateVec<1> flux(const StateVec<1>& u, const Aux& a, int dir) const {
    StateVec<1> f{};
    f[0] = u[0] * velocity(a, dir);
    return f;
  }
  ADC_HD Real max_wave_speed(const StateVec<1>&, const Aux& a, int dir) const {
    const Real d = velocity(a, dir);
    return d < 0 ? -d : d;
  }
  /// Spectre : une onde, la vitesse de derive dans la direction dir.
  ADC_HD StateVec<1> eigenvalues(const StateVec<1>&, const Aux& a, int dir) const {
    StateVec<1> e{};
    e[0] = velocity(a, dir);
    return e;
  }
  // Scalaire : variables primitives = conservatives (densite transportee).
  using Prim = StateVec<1>;
  ADC_HD Prim to_primitive(const StateVec<1>& u) const { return u; }
  ADC_HD StateVec<1> to_conservative(const Prim& p) const { return p; }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"n"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"n"}, 1, {VariableRole::Density}};
  }
};

/// Flux d'Euler compressible 2D (reutilise Euler : gamma, pression, vitesses d'onde signees).
/// Alias de compat : CompressibleFlux == Euler ; la brique hyperbolique complete.
using CompressibleFlux = Euler;

/// Flux d'Euler ISOTHERME (p = cs2 rho), 3 variables (rho, rho u, rho v).
///
/// Brique HYPERBOLIQUE 3-variables (densite + quantites de mouvement). Satisfait
/// HyperbolicPhysicalModel. Loi de fermeture isotherme : p = cs2 * rho (pas d'equation
/// d'energie). CONTRAT : fonctions purement ponctuelles, device-callables (ADC_HD).
/// Aucune MultiFab, aucune allocation, aucun acces global.
/// Invariant : cs2 > 0 pour que la vitesse d'onde sqrt(cs2) soit reelle.
struct IsothermalFlux {
  static constexpr int n_vars = 3;
  using State = StateVec<3>;  ///< variables conservatives (rho, rho u, rho v)
  using Prim = StateVec<3>;   ///< variables primitives (rho, u, v)
  Real cs2 = 1;
  ADC_HD StateVec<3> flux(const StateVec<3>& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vn = (dir == 0 ? u[1] : u[2]) / rho;
    const Real p = cs2 * rho;
    StateVec<3> f{};
    f[0] = (dir == 0 ? u[1] : u[2]);
    f[1] = u[1] * vn + (dir == 0 ? p : Real(0));
    f[2] = u[2] * vn + (dir == 1 ? p : Real(0));
    return f;
  }
  /// Conservatif -> primitif : (rho, rho u, rho v) -> (rho, u, v).
  ADC_HD Prim to_primitive(const StateVec<3>& u) const {
    Prim p{};
    p[0] = u[0];
    p[1] = u[1] / u[0];
    p[2] = u[2] / u[0];
    return p;
  }
  /// Primitif -> conservatif : (rho, u, v) -> (rho, rho u, rho v).
  ADC_HD StateVec<3> to_conservative(const Prim& p) const {
    StateVec<3> u{};
    u[0] = p[0];
    u[1] = p[0] * p[1];
    u[2] = p[0] * p[2];
    return u;
  }
  ADC_HD Real max_wave_speed(const StateVec<3>& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real a = vn < 0 ? -vn : vn;
    return a + std::sqrt(cs2);
  }
  /// Spectre complet : (v_dir - c, v_dir, v_dir + c), c = sqrt(cs2).
  ADC_HD StateVec<3> eigenvalues(const StateVec<3>& u, const Aux&, int dir) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(cs2);
    StateVec<3> e{};
    e[0] = vn - c;
    e[1] = vn;
    e[2] = vn + c;
    return e;
  }
  /// Vitesses signees (HLL/HLLC) : v_dir -+ c_s.
  ADC_HD void wave_speeds(const StateVec<3>& u, const Aux&, int dir, Real& smin,
                          Real& smax) const {
    const Prim p = to_primitive(u);
    const Real vn = (dir == 0 ? p[1] : p[2]);
    const Real c = std::sqrt(cs2);
    smin = vn - c;
    smax = vn + c;
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"rho", "rho_u", "rho_v"}, 3,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::MomentumY}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"rho", "u", "v"}, 3,
            {VariableRole::Density, VariableRole::VelocityX, VariableRole::VelocityY}};
  }
};

/// Flux d'Euler ISOTHERME en geometrie POLAIRE (anneau r, theta), 3 variables (rho, rho v_r,
/// rho v_theta) -- chantier "grille polaire fluide", Voie A etape 1. C'est une brique SEPAREE
/// d'IsothermalFlux (cartesien) : le flux PHYSIQUE et les conversions sont IDENTIQUES (les
/// composantes 1, 2 sont la quantite de mouvement dans la BASE LOCALE ORTHONORMEE (e_r, e_theta) ;
/// dir 0 = radial, dir 1 = azimutal), mais cette brique ajoute le TERME GEOMETRIQUE DE COURBURE
/// porte par la metrique polaire. On herite IsothermalFlux pour ne PAS dupliquer flux /
/// conversions / vitesses d'onde (cartesien strictement intact, bit-identique) et on n'ajoute QUE
/// la methode polar_geom_source.
///
/// POURQUOI UN TERME GEOMETRIQUE EXPLICITE (et non une simple divergence conservative) :
/// l'equation vectorielle de quantite de mouvement d_t(rho v) + div(rho v (x) v) + grad p = 0,
/// projetee sur la base LOCALE polaire (e_r, e_theta) qui TOURNE avec theta, donne pour les
/// composantes PHYSIQUES m_r = rho v_r, m_theta = rho v_theta :
///   d_t m_r     + (1/r) d_r(r (rho v_r^2 + p)) + (1/r) d_theta(rho v_r v_theta)
///                 - (rho v_theta^2 + p)/r            = 0      (terme CENTRIFUGE + pression)
///   d_t m_theta + (1/r) d_r(r rho v_r v_theta)     + (1/r) d_theta(rho v_theta^2 + p)
///                 + (rho v_r v_theta)/r             = 0      (terme de COURBURE croisee)
/// L'operateur assemble_rhs_polar calcule EXACTEMENT -(1/r) d_r(r F_r) - (1/r) d_theta(F_theta)
/// avec F_r, F_theta = IsothermalFlux::flux : il reproduit donc les divergences, mais PAS les
/// termes algebriques -(rho v_theta^2 + p)/r et +(rho v_r v_theta)/r. Ces termes ne sont PAS
/// captures par la divergence conservative (preuve : sur la cellule (rho, v_r=0, v_theta(r)) en
/// equilibre rotatif d_r p = rho v_theta^2/r, la divergence radiale seule rendrait
/// d_t m_r = -(d_r p + p/r) != 0, brisant l'equilibre). Il FAUT donc une SOURCE GEOMETRIQUE
/// explicite, fournie ici et ajoutee en cellule par assemble_rhs_polar (qui seul connait r) :
///   S_geom = ( 0 , (rho v_theta^2 + p)/r , -(rho v_r v_theta)/r ).
/// Avec cette source l'equilibre rotatif est preserve a l'ordre du schema (cf.
/// test_polar_fluid_equilibrium). r > 0 (anneau, r_min > 0) : aucune singularite d'axe.
///
/// CONTRAT : brique PHYSIQUE ponctuelle, device-callable (ADC_HD), aucune box, aucune allocation.
/// polar_geom_source ne prend QUE l'etat et r (pas d'aux) : c'est de la pure metrique.
struct IsothermalFluxPolar : IsothermalFlux {
  /// Terme source GEOMETRIQUE de courbure en cellule de rayon r > 0 (anneau). Voir le bloc @file
  /// ci-dessus pour la derivation. S_geom = (0, (rho v_theta^2 + p)/r, -(rho v_r v_theta)/r),
  /// p = cs2 rho. Composante 0 (masse) nulle : la masse est purement conservative en polaire.
  ADC_HD StateVec<3> polar_geom_source(const StateVec<3>& u, Real r) const {
    const Real rho = u[0];
    const Real inv_rho = Real(1) / rho;
    const Real mr = u[1], mth = u[2];  // rho v_r, rho v_theta (base locale (e_r, e_theta))
    const Real p = cs2 * rho;
    const Real inv_r = Real(1) / r;
    StateVec<3> s{};
    s[0] = Real(0);
    s[1] = (mth * mth * inv_rho + p) * inv_r;   // (rho v_theta^2 + p)/r : centrifuge + pression
    s[2] = -(mr * mth * inv_rho) * inv_r;        // -(rho v_r v_theta)/r : courbure croisee
    return s;
  }
};

}  // namespace adc

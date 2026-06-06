/// @file
/// @brief LorentzEliminator : operateur 2x2 B du schema de Schur pour l'elimination implicite de la vitesse.
///
/// Encode B = [[1, -w], [w, 1]] et son inverse B^{-1} = (1/det)*[[1, w], [-w, 1]],
/// avec w = theta * dt * B_z et det(B) = 1 + w^2 (toujours > 0 : B inversible pour tout w reel).
///
/// Struct POD, zero allocation, trivially copyable. Tous les accesseurs ADC_HD.
/// Aucun appel a std:: : device-safe sous Kokkos/CUDA/HIP sans restriction.

#pragma once

#include <adc/core/types.hpp>

// LorentzEliminator : operateur B du schema de Schur pour l'elimination de la vitesse.
//
// CONVENTION DE SIGNE (a lire avant toute modification) :
//   On travaille en 2D dans le plan (x, y) avec B oriente selon z : B_field = B_z * z_hat.
//   Le terme de Lorentz sur la vitesse v est :
//       F_L = q/m * (v x B_field)
//   Dans le plan 2D, avec v = (v_x, v_y, 0) et B_field = (0, 0, B_z) :
//       v x B_field = (v_y * B_z, -v_x * B_z, 0)
//   Donc la composante x est +v_y*B_z et la composante y est -v_x*B_z.
//
//   L'operateur B encode le terme implicite theta*dt*F_L dans l'avancee en temps :
//       B v = v - theta*dt*(v x B_field)
//   ce qui donne, par la convention ci-dessus :
//       (B v)_x = v_x - theta*dt * v_y * B_z
//       (B v)_y = v_y + theta*dt * v_x * B_z
//   Sous forme matricielle 2x2 :
//       B = [[1, -w], [w, 1]]   avec w = theta*dt*B_z
//
//   ATTENTION : le signe negatif est sur la colonne superieure droite (terme en -w pour v_x)
//   et le signe positif sur la colonne inferieure gauche (terme en +w pour v_y). Ce choix
//   decoule DIRECTEMENT de v x B_field ci-dessus et ne doit pas etre modifie.
//
// Inverse analytique :
//   det(B) = 1 + w^2  (toujours > 0, B est inversible pour tout w reel)
//   B^{-1} = (1/det) * [[1, w], [-w, 1]]
//
// Usage prevu : assemblage de l'operateur de Schur A = rho * B^{-1} dans le solveur
// implicite de la vitesse (cf. PR2/PR3 du chantier Schur). Generique : theta, dt, B_z
// sont des parametres ; aucune dependance physique codee en dur.
//
// Proprietes structurelles :
//   - Struct POD, zero allocation, trivially copyable.
//   - Tous les accesseurs sont ADC_HD (host+device callable, device-safe sous Kokkos/CUDA/HIP).
//   - Aucun appel a std:: (sqrt, etc.) : les quatre operations sont des additions/multiplications.

namespace adc {

/// LorentzEliminator : operateur B = [[1,-w],[w,1]] et son inverse analytique.
///
/// Construit depuis (theta, dt, B_z) ; encode le terme implicite de Lorentz dans un schema
/// Crank-Nicolson ou theta-implicite. Utilise pour assembler l'operateur de Schur
/// A = rho * B^{-1} dans le solveur implicite de la vitesse.
///
/// CONVENTION DE SIGNE : B_field = B_z z_hat dans le plan (x,y).
///   v x B_field = (v_y B_z, -v_x B_z) => B = [[1, -w], [w, 1]] avec w = theta*dt*B_z.
///   Ne pas modifier sans relire la derivation dans le commentaire interne.
///
/// INVARIANT : struct trivialement copiable (static_assert ci-dessous), device-safe,
/// zero allocation, zero appel a std::. Peut etre capture par valeur dans un kernel Kokkos/CUDA.
// LorentzEliminator(theta, dt, B_z) encode B = [[1,-w],[w,1]] et son inverse.
struct LorentzEliminator {
  Real w;    // w = theta * dt * B_z
  Real det;  // det(B) = 1 + w^2

  // Construit l'eliminateur pour un jeu de parametres (theta, dt, B_z).
  // theta : implicite du schema (0 = explicite, 1 = implicite, 0.5 = Crank-Nicolson).
  // dt    : pas de temps.
  // B_z   : composante z du champ magnetique.
  /// Construit depuis (theta, dt, B_z) : w = theta*dt*B_z, det = 1 + w^2. ADC_HD.
  ADC_HD LorentzEliminator(Real theta, Real dt, Real B_z)
      : w(theta * dt * B_z), det(Real(1) + (theta * dt * B_z) * (theta * dt * B_z)) {}

  // apply_B : applique B = [[1,-w],[w,1]] a (vx, vy).
  // Retourne (Bx, By) dans les arguments de sortie.
  /// apply_B : applique B = [[1,-w],[w,1]] a (vx, vy), ecrit (Bx, By). ADC_HD.
  ADC_HD void apply_B(Real vx, Real vy, Real& Bx, Real& By) const {
    Bx = vx - w * vy;
    By = vy + w * vx;
  }

  // apply_Binv : applique B^{-1} = (1/det)*[[1,w],[-w,1]] a (vx, vy).
  // Retourne (vx', vy') dans les arguments de sortie.
  /// apply_Binv : applique B^{-1} = (1/det)*[[1,w],[-w,1]] a (vx, vy), ecrit (vxp, vyp). ADC_HD.
  ADC_HD void apply_Binv(Real vx, Real vy, Real& vxp, Real& vyp) const {
    const Real inv = Real(1) / det;
    vxp = inv * (vx + w * vy);
    vyp = inv * (vy - w * vx);
  }

  // binv_11, binv_12, binv_21, binv_22 : entrees de B^{-1} = (1/det)*[[1,w],[-w,1]].
  // Utiles pour assembler l'operateur de Schur A = rho * B^{-1} cellule par cellule.
  /// @name Entrees scalaires de B^{-1} (pour assembler l'operateur de Schur A = rho * B^{-1}).
  /// @{
  ADC_HD Real binv_11() const { return Real(1) / det; }
  ADC_HD Real binv_12() const { return w / det; }
  ADC_HD Real binv_21() const { return -w / det; }
  ADC_HD Real binv_22() const { return Real(1) / det; }
  /// @}
};

// Verification statique : LorentzEliminator est trivially copyable (POD-like),
// ce qui garantit qu'il peut etre capture dans un kernel Kokkos/CUDA par valeur
// sans allocation cachee.
static_assert(
    __is_trivially_copyable(LorentzEliminator),
    "LorentzEliminator doit etre trivially copyable (zero allocation, device-safe)");

}  // namespace adc

#pragma once

#include <adc/coupling/condensed_schur_source_stepper.hpp>  // CondensedSchurSourceStepper (#126) + detail kernels
#include <adc/numerics/time/amr_subcycling.hpp>              // AmrLevelMP (hierarchie multi-patch)

#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief AmrCondensedSchurSourceStepper : pendant AMR de l'etage SOURCE condense par Schur
///        (CondensedSchurSourceStepper, #126), porte sur une HIERARCHIE de niveaux (AmrLevelMP) plutot
///        que sur une grille uniforme. C'est l'etage source GLOBAL electrostatique/Lorentz du chemin
///        "amr-schur" -- l'equivalent raffine du chemin uniforme
///          System(...).add_equation(time=Strang(hyperbolic=Explicit(ssprk3),
///                                                source=CondensedSchur(theta, alpha)))
///        et NON une source locale cellule-par-cellule (cf. l'IMEX local backward_euler_source du
///        chemin amr-imex, qui n'est PAS quantitativement comparable au papier Hoffart arXiv:2510.11808).
///
/// STRATEGIE (option A, miroir du Poisson AMR existant compute_aux/solve_fields). Le solveur elliptique
/// AMR de ce code resout le Poisson sur le NIVEAU GROSSIER puis injecte grad phi aux niveaux fins (les
/// patchs fins raffinent le TRANSPORT, pas la resolution elliptique). L'etage source condense suit la
/// MEME approche : il assemble et resout l'operateur condense A_op = I + theta^2 dt^2 alpha rho B^{-1}
/// sur le grossier (en COMPOSANT l'etage uniforme #126, bit-pour-bit), puis -- pour une hierarchie
/// multi-niveau -- injecte grad phi^{n+theta} aux fins et y reconstruit les vitesses, en terminant par
/// la cascade fin -> grossier (average_down) qui retablit la coherence des cellules grossieres
/// couvertes (invariant #169). Un etat constant en espace (mono-niveau) degenere EXACTEMENT en l'etage
/// uniforme : c'est le critere de parite (Etape 2).
///
/// PERIMETRE DE CETTE VERSION (Etape 2, parite System d'abord). Le chemin MONO-NIVEAU est complet et
/// bit-identique a l'etage uniforme. Le chemin MULTI-NIVEAU (reconstruction des vitesses fines a partir
/// du grad injecte + cascade average_down) est un suivi dedie (Etape 4) : step() le REFUSE
/// explicitement (erreur claire) plutot que d'appliquer la source au seul grossier en silence (les
/// cellules fines ne sentiraient pas la source -> faux). On valide d'abord la parite mono-niveau contre
/// l'etage uniforme #126, comme demande.
///
/// CYCLE DE VIE / DEVICE / MPI. Construit UNE fois sur le layout GROSSIER (BoxArray + Geometry + CL
/// Poisson) ; tous les tampons de l'etage uniforme grossier sont alloues a la construction et reutilises
/// par step(). Le solve de Krylov grossier est COLLECTIF (dot/all_reduce sur tous les rangs, y compris
/// vides) -- comme l'etage uniforme : pas d'interblocage. theta/dt peuvent changer entre appels.

namespace adc {

/// ETAGE SOURCE condense par Schur sur une hierarchie AMR. GENERIQUE sur tout bloc fluide qui expose
/// les roles Density / MomentumX / MomentumY (+ Energy optionnel), exactement comme l'etage uniforme.
class AmrCondensedSchurSourceStepper {
 public:
  /// @p vars  : descripteur du bloc fluide (DOIT exposer Density / MomentumX / MomentumY ; Energy
  ///            optionnel). Valide ICI (hote) par le ctor de l'etage uniforme grossier.
  /// @p coarse_geom : geometrie du NIVEAU GROSSIER (cartesienne).
  /// @p coarse_ba   : decoupage du niveau grossier (mono-box replique ou multi-box reparti).
  /// @p bcPhi : CL du potentiel phi (memes que le Poisson grossier).
  /// @p alpha : constante de couplage electrostatique.
  /// @p n_precond_vcycles : N V-cycles MG par application du preconditionneur BiCGStab (1 ou 2).
  AmrCondensedSchurSourceStepper(const VariableSet& vars, const Geometry& coarse_geom,
                                 const BoxArray& coarse_ba, const BCRec& bcPhi, Real alpha,
                                 int n_precond_vcycles = 1)
      : coarse_(vars, coarse_geom, coarse_ba, bcPhi, alpha, n_precond_vcycles) {}

  /// true si le modele porte un role Energy (mise a jour d'energie active dans l'etage grossier).
  bool has_energy() const { return coarse_.energy_comp() >= 0; }

  /// ETAGE SOURCE condense, IN-PLACE sur la hierarchie @p levels et le potentiel grossier @p coarse_phi.
  ///   @p levels    : hierarchie multi-patch ; levels[0] = GROSSIER (level 0), levels[k>=1] = FIN
  ///                  (ratio 2). L'etat conservatif de chaque niveau est levels[k].U (rho GELEE,
  ///                  mom/E mis a jour ; meme convention que l'etage uniforme).
  ///   @p coarse_phi: potentiel du niveau grossier. ENTREE phi^n (warm start du solve) ; SORTIE
  ///                  phi^{n+1}. Meme objet que le Poisson grossier (mg_.phi() du coupleur) cote facade.
  ///   @p coarse_bz : champ B_z du niveau grossier (canal aux), composante @p c_bz lue au centre.
  ///   @p theta / @p dt : theta-schema (theta dans (0, 1]) ; dt = pas effectif (facteur stride inclus
  ///                  par l'appelant, comme s.advance / run_source_stage du chemin uniforme).
  void step(std::vector<AmrLevelMP>& levels, MultiFab& coarse_phi, const MultiFab& coarse_bz,
            int c_bz, Real theta, Real dt) {
    if (levels.empty()) return;
    // Un niveau fin EFFECTIVEMENT PEUPLE (>= un patch) signale une hierarchie multi-niveau. NB : le
    // chemin compile (build_amr_compiled) alloue TOUJOURS un niveau fin seed, VIDE apres regrid quand
    // aucun raffinement n'est demande (refine_threshold desactive) -> levels.size() vaut 2 mais la
    // hierarchie est EFFECTIVEMENT mono-niveau. On garde donc sur le NOMBRE DE PATCHS fins, pas sur
    // levels.size(), pour ne pas refuser le cas mono-niveau a niveau fin alloue mais vide.
    int n_fine_patches = 0;
    for (std::size_t k = 1; k < levels.size(); ++k)
      n_fine_patches += static_cast<int>(levels[k].U.box_array().size());
    if (n_fine_patches > 0)
      throw std::runtime_error(
          "AmrCondensedSchurSourceStepper : etage source condense MULTI-NIVEAU (" +
          std::to_string(n_fine_patches) +
          " patch(s) fin(s)) pas encore cable. Cette version porte le chemin MONO-NIVEAU (parite "
          "stricte avec l'etage uniforme #126) ; la reconstruction des vitesses fines + cascade "
          "average_down est un suivi dedie (Etape 4). Demarrer une hierarchie mono-niveau "
          "(refine_threshold desactive).");
    // NIVEAU GROSSIER (= tout le domaine) : etage uniforme COMPLET (assemble + solve + reconstruction +
    // extrapolation + energie + remplissage des ghosts), bit-pour-bit identique a CondensedSchurSourceStepper.
    coarse_.step(levels[0].U, coarse_phi, coarse_bz, c_bz, theta, dt);
  }

  /// Diagnostic du dernier solve de l'etage grossier (iterations BiCGStab, residu relatif, convergence).
  const KrylovResult& last_solve() const { return coarse_.last_solve(); }

  int density_comp() const { return coarse_.density_comp(); }
  int momentum_x_comp() const { return coarse_.momentum_x_comp(); }
  int momentum_y_comp() const { return coarse_.momentum_y_comp(); }
  int energy_comp() const { return coarse_.energy_comp(); }

 private:
  /// Etage source condense uniforme porte sur le NIVEAU GROSSIER. On COMPOSE (pas de duplication de
  /// numerique) : le grossier couvre tout le domaine, donc en mono-niveau cet etage EST l'etage uniforme.
  CondensedSchurSourceStepper coarse_;
};

}  // namespace adc

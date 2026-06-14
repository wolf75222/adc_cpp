#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/runtime/wall_predicate.hpp>  // detail::DiscDomain (en-tete leger : <cmath>/<functional>/<string>)

#include <functional>

/// @file
/// @brief Contexte de grille + fermetures d'un bloc, partages entre System (qui les installe) et
///        block_builder.hpp (qui les fabrique a partir d'un modele compile). En-tete LEGER (mailles +
///        std::function, sans la numerique) pour pouvoir etre inclus dans l'API publique du System
///        sans y tirer assemble_rhs / flux / steppers.

namespace adc {

/// MODE DE GEOMETRIE DE TRANSPORT du macro-pas (chantier T5-PR3, cablage du disque dans System::step).
///  - None      : domaine plein cartesien (defaut). Le transport utilise assemble_rhs (chemin
///                historique). BIT-IDENTIQUE a l'historique tant qu'aucun disque n'est fixe.
///  - Staircase : disque approche par un MASQUE 0/1 cellule-centre (porte de face active/inactive,
///                frontiere crenelee). Le transport utilise assemble_rhs_masked (chantier T2).
///  - CutCell   : disque en cut-cell / embedded-boundary (apertures alpha_f continues + fraction de
///                volume kappa). Le transport utilise assemble_rhs_eb (chantiers T5-PR1/PR2).
/// Le mode est porte par le System (set_disc_domain mode= / set_geometry_mode) et lu par le stepper
/// pour AIGUILLER l'avance de transport de chaque bloc. None reste le chemin de production intouche.
enum class GeometryMode { None, Staircase, CutCell };

/// Maillage + CL transport + aux partages par les fermetures d'un bloc. @c aux n'est PAS possede :
/// il pointe l'aux du System (duree de vie superieure au bloc, adresse stable).
///
/// GEOMETRIE DISQUE (chantier T5-PR3) : @c disc_mask et @c disc pointent (NON possedes) le masque 0/1
/// et le descripteur de disque du System (membres a adresse STABLE). Ils servent UNIQUEMENT a fabriquer
/// les avances de transport disque optionnelles (build_block) ; lues PAR POINTEUR au pas, l'ordre
/// add_block / set_disc_domain est indifferent. nullptr -> aucune avance disque (stepper sur advance,
/// bit-identique). Le masque est materialise / le rayon pose par set_disc_domain.
struct GridContext {
  Box2D dom;                ///< domaine (sans ghost)
  BCRec bc;                 ///< CL de transport
  Geometry geom;            ///< geometrie (dx, dy, bornes)
  MultiFab* aux = nullptr;  ///< aux du System (phi, grad phi) ; NON possede
  const MultiFab* disc_mask = nullptr;        ///< masque 0/1 disque (Impl::disc_mask_) ; NON possede
  const detail::DiscDomain* disc = nullptr;   ///< descripteur disque (Impl::disc_) ; NON possede
};

/// Fermetures compilees d'un bloc, figees a l'ajout.
///
/// advance est l'avance de transport du chemin PAR DEFAUT (assemble_rhs, plein cartesien). Les deux
/// avances DISQUE optionnelles (chantier T5-PR3) miment EXACTEMENT advance (meme schema RK / IMEX,
/// meme limiteur / flux) mais aiguillent le residu de transport vers l'operateur disque :
///   - advance_masked : assemble_rhs_masked (masque 0/1, mode Staircase) ;
///   - advance_eb     : assemble_rhs_eb (cut-cell EB, mode CutCell).
/// Elles lisent le masque / le level set du System PAR POINTEUR au moment du pas (et non a la
/// construction), donc l'ordre add_block / set_disc_domain est indifferent. Vides (defaut) tant que le
/// bloc ne supporte pas le routage disque : le stepper retombe alors sur advance (bit-identique).
struct BlockClosures {
  std::function<void(MultiFab&, Real, int)> advance;  ///< (U, dt, n) : n sous-pas de dt/n
  std::function<void(MultiFab&, Real, int)> advance_masked;  ///< idem, residu via assemble_rhs_masked
  std::function<void(MultiFab&, Real, int)> advance_eb;      ///< idem, residu via assemble_rhs_eb
  std::function<void(MultiFab&, MultiFab&)> rhs_into;  ///< R <- -div F + S (Poisson fige)
  /// Diagnostic dt_hotspot (ADC-182) : (U, w, i, j) -> cellule GLOBALE dominant la CFL de
  /// transport et sa vitesse. OPTIONNELLE (vide = bloc sans diagnostic, p.ex. chemins
  /// historiques non recables) ; jamais appelee par step/step_cfl (hors chemin chaud).
  std::function<void(const MultiFab&, Real&, int&, int&)> hotspot;
  /// PROJECTION PONCTUELLE post-pas (ADC-177) : U <- project(U, aux) sur les cellules VALIDES du
  /// bloc, appliquee par le stepper a la FIN de chaque macro-pas ENTIER (jamais par etage RK).
  /// OPTIONNELLE (vide = bloc sans projection : jamais interrogee, cout nul, bit-identique).
  std::function<void(MultiFab&)> project;
};

}  // namespace adc

#pragma once
// Facade unifiee de production : LevelHierarchy, OwnershipPolicy, advance_amr.

#include <adc/numerics/time/amr_subcycling.hpp>

namespace adc {

// --- Moteur AMR unifie (revue, point 5) ---
// La hierarchie AMR comme OBJET nomme que le moteur fait avancer, plutot qu'une famille de
// fonctions amr_step_* dont le cas (2/N niveaux, mono/multi-box) est encode dans le NOM.
// Entree unifiee : advance_amr(m, LevelHierarchy&, dt), facade fidele du moteur N-niveaux
// multi-patch (verifie 2 ET 3 niveaux, maxdiff = 0, par test_advance_amr). Les ROLES de la
// revue, leurs supports actuels (types nommes, ou code restant a promouvoir) :
//   OwnershipPolicy     = DistributionMapping (qui possede quel patch)        -> alias ci-dessous
//   AmrLevel            = AmrLevelMP (box + donnees + aux + dx d'un niveau)
//   PatchRange          = TYPE NOMME : empreinte grossiere [I0..I1]x[J0..J1] d'un patch fin
//                         (ratio 2), partagee par average_down, couverture et registres
//   CoarseFineGhost     = TYPE/HELPER NOMME : fill_cf_ghost_cell (interp espace+temps par
//                         cellule ghost), partage par les trois mf_fill_fine_ghosts_*
//   CoarseFineInterface = TYPE NOMME : couverture (CoverageMask) + routage bordant du reflux
//                         (route_reflux), partage par subcycle_level_mp et amr_step_2level_multipatch
//   FluxRegister        = TYPE NOMME : buffers avg/ref a index global + all_reduce_sum_inplace
//   SubcyclingSchedule  = TYPE NOMME : cadence Berger-Oliger (ratio r, dt/r, frac s/r) par niveau
//   RegridPolicy        = amr_regrid_finest (Berger-Rigoutsos), cote coupleur
using OwnershipPolicy = DistributionMapping;

struct LevelHierarchy {
  std::vector<AmrLevelMP> levels;    // niveau 0 = grossier, niveaux >0 = patchs fins
  Box2D base_dom;                    // empreinte du niveau de base
  Periodicity base_per{true, true};  // CL du domaine de base
  bool coarse_replicated = true;     // niveau 0 replique (true) ou multi-box reparti (false)
  bool recon_prim = false;           // reconstruction primitive (cf. compute_face_fluxes)
  bool imex = false;                 // source raide implicite (backward_euler) au lieu d'Euler avant
  // OPTIONS NEWTON du pas IMEX (defaut {} = constantes historiques 2 iters / 1e-7 -> bit-identique).
  // Honorees uniquement quand imex==true ; transmises au backward_euler_source par mf_apply_source_treatment.
  NewtonOptions newton_options{};
  // METHODE TEMPORELLE : kEuler (defaut, Euler avant par sous-pas, bit-identique a l'historique) ou
  // kSsprk3 (SSPRK3 ordre 3 + reflux par etage). kSsprk3 exige imex == false (rejet sinon, cf. moteur).
  AmrTimeMethod time_method = AmrTimeMethod::kEuler;
};

// Entree unifiee de production : avance la hierarchie d'un pas dt. Forme "pieces" (le coupleur
// possede son propre stack et passe les vecteurs directement) et forme LevelHierarchy. La porte
// TRANSMET coarse_replicated au moteur ; sans cela un grossier de-replique repasserait en
// replique (mf_find_box au lieu de parallel_copy). coarse_replicated=true (defaut) -> identique
// au comportement historique. recon_prim selectionne la reconstruction primitive (variables
// (rho, u, p)) au lieu de conservative : meme parametre qu'assemble_rhs, fige a l'ajout du bloc ;
// false (defaut) -> conservative, strictement bit-identique a l'historique.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_amr(const Model& m, std::vector<AmrLevelMP>& levels, const Box2D& base_dom, Real dt,
                 Periodicity base_per = Periodicity{true, true}, bool coarse_replicated = true,
                 bool recon_prim = false, bool imex = false, const NewtonOptions& nopts = {},
                 AmrTimeMethod tmethod = AmrTimeMethod::kEuler) {
  detail::amr_step_multilevel_multipatch<Limiter, NumericalFlux>(
      m, levels, base_dom, dt, base_per, coarse_replicated, recon_prim, imex, nopts, tmethod);
}

template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void advance_amr(const Model& m, LevelHierarchy& h, Real dt) {
  advance_amr<Limiter, NumericalFlux>(m, h.levels, h.base_dom, dt, h.base_per, h.coarse_replicated,
                                      h.recon_prim, h.imex, h.newton_options, h.time_method);
}

}  // namespace adc

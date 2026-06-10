#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // BCRec, fill_ghosts, fill_boundary
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator_polar.hpp>  // assemble_rhs_polar (REUTILISE verbatim)
#include <adc/numerics/time/time_steppers.hpp>      // SSPRK2Step / SSPRK3Step (math RK du coeur)
#include <adc/parallel/comm.hpp>                     // all_reduce_max (reduction collective MPI-safe)
#include <adc/physics/bricks.hpp>                    // ExBVelocityPolar, CompositeModel, briques source/elliptic
#include <adc/runtime/dispatch_tags.hpp>             // registry UNIQUE des tags (validate_limiter/riemann)
#include <adc/runtime/grid_context.hpp>              // BlockClosures (en-tete leger)
#include <adc/runtime/model_factory.hpp>             // detail::dispatch_source / dispatch_elliptic (REUTILISES)
#include <adc/runtime/model_spec.hpp>

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief Pendant POLAIRE de block_builder.hpp : fabrique les fermetures d'un bloc (avance en temps +
///        residu + contribution Poisson + vitesse d'onde) sur une grille ANNULAIRE (PolarGeometry), en
///        REUTILISANT assemble_rhs_polar (include/adc/numerics/spatial_operator_polar.hpp). C'est le
///        chemin de Polaire Phase 2b : le transport polaire A TRAVERS System.step.
///
/// SEPARATION STRICTE du chemin cartesien : block_builder.hpp (assemble_rhs cartesien, Geometry) reste
/// INTOUCHE -> un System cartesien est bit-identique. Le chemin polaire est PUREMENT ADDITIF, opt-in
/// (active quand cfg.geometry == "polar", cf. system.cpp). On reutilise les MEMES briques de source /
/// elliptique (dispatch_source / dispatch_elliptic de model_factory.hpp) et le MEME transport ExB, mais
/// porte par la brique ExBVelocityPolar (composantes physiques en base locale (e_r, e_theta)).
///
/// FONCTEURS NOMMES (et non lambdas etendues), comme block_builder.hpp (#64/#97/#133) : emission device
/// robuste si le noyau Model-template est premiere-instancie cross-TU. assemble_rhs_polar et ses noyaux
/// sont deja device-clean (7/7 GH200) ; ces fermetures se contentent de les enchainer.

namespace adc {

/// Maillage POLAIRE + CL transport + aux partages par les fermetures d'un bloc (pendant de
/// GridContext). @c aux n'est PAS possede : il pointe l'aux du System (adresse stable). bc porte la CL
/// radiale (Foextrap = paroi / sortie) ; theta est PERIODIQUE (gere par fill_ghosts via bc.ylo/yhi).
struct PolarGridContext {
  Box2D dom;                ///< domaine d'indices (sans ghost) : nx() = nr, ny() = ntheta
  BCRec bc;                 ///< CL : r (xlo/xhi) physique, theta (ylo/yhi) periodique
  PolarGeometry geom;       ///< anneau (r_min, r_max, dr, dtheta)
  MultiFab* aux = nullptr;  ///< aux du System (phi, grad_r, grad_theta) ; NON possede
};

namespace detail {

/// Construit la brique de transport POLAIRE et appelle v(transport). Deux transports cables :
///   - "exb"        : ExBVelocityPolar, advection scalaire ExB en base locale (e_r, e_theta) ;
///   - "isothermal" : IsothermalFluxPolar (Voie A etape 1), fluide isotherme 3 var (rho, rho v_r,
///                    rho v_theta) en metrique polaire. Le flux PHYSIQUE est celui d'IsothermalFlux
///                    cartesien (reutilise verbatim) ; la metrique 1/r (divergence (1/r) d_r(r F_r) +
///                    (1/r) d_theta(F_theta)) ET le terme GEOMETRIQUE de courbure (centrifuge
///                    -rho v_theta^2/r + courbure croisee) sont portes par assemble_rhs_polar /
///                    IsothermalFluxPolar::polar_geom_source. Couplage electrostatique = Poisson
///                    polaire SCALAIRE existant + source LOCALE (PotentialForce), regime explicite.
/// Le transport "compressible" (Euler 4 var avec energie) en polaire reste hors scope : son flux
/// d'energie et son terme de courbure n'ont pas encore de brique polaire -> erreur EXPLICITE.
template <class Visitor>
void dispatch_transport_polar(const ModelSpec& m, Visitor&& v) {
  if (m.transport == "exb") return v(ExBVelocityPolar{Real(m.B0)});
  if (m.transport == "isothermal") return v(IsothermalFluxPolar{IsothermalFlux{Real(m.cs2)}});
  throw std::runtime_error(
      "transport polaire '" + m.transport +
      "' non supporte (cables : 'exb' = advection scalaire ExB ; 'isothermal' = fluide isotherme "
      "3 var en metrique polaire (Voie A etape 1). 'compressible' (Euler avec energie) en polaire "
      "est une phase ulterieure)");
}

/// Assemble le CompositeModel POLAIRE designe par @p m et appelle visitor(model). REUTILISE
/// dispatch_source / dispatch_elliptic de model_factory.hpp (briques de source / second membre
/// elliptique IDENTIQUES au cartesien : elles ne portent pas de geometrie). Seule la brique de
/// transport change (ExBVelocityPolar ou IsothermalFluxPolar). dispatch_source<TR::n_vars> filtre
/// automatiquement : transport scalaire ExB (1 var) -> seule source 'none' ; transport fluide
/// isotherme (3 var) -> 'none' | 'potential' (-rho grad phi) | 'gravity' | 'magnetic'/'lorentz'
/// (q v x B_z, B_z lu dans l'aux, regime EXPLICITE) | 'potential_magnetic'/'potential_lorentz'
/// (electrostatique + Lorentz somme = force complete du diocotron polaire NATIF) egalement valides.
/// La force de Lorentz est ALGEBRIQUE et INVARIANTE par orientation du repere local orthonorme :
/// la MEME brique MagneticLorentzForce sert les deux geometries (cartesien et polaire), comme
/// PotentialForce / GravityForce. La metrique 1/r et la courbure restent portees par le transport.
template <class Visitor>
void dispatch_model_polar(const ModelSpec& m, Visitor&& visitor) {
  dispatch_transport_polar(m, [&](auto tr) {
    using TR = decltype(tr);
    // Resolution AUTOMATIQUE par roles (audit §5), IDENTIQUE au cartesien (bind_variable_roles de
    // model_factory.hpp). ExBVelocityPolar (densite=0) / IsothermalFluxPolar (rho=0, m_x=1, m_y=2,
    // herite d'IsothermalFlux) declarent des roles canoniques -> indices resolus == defauts ->
    // bit-identique. Resolu a la construction (hote) ; jamais en device.
    const VariableSet cons = TR::conservative_vars();
    dispatch_source<TR::n_vars>(m, [&](auto src) {
      dispatch_elliptic(m, [&](auto ell) {
        bind_variable_roles(src, cons);
        bind_variable_roles(ell, cons);
        visitor(CompositeModel<TR, decltype(src), decltype(ell)>{tr, src, ell});
      });
    });
  });
}

/// Remplit les ghosts d'un MultiFab sur la grille polaire (theta periodique + r physique). fill_ghosts
/// route deja periodique vs physique par BCRec (xlo/xhi physique, ylo/yhi periodique) : on l'appelle
/// VERBATIM. C'est l'analogue du fill_ghosts(U, dom, bc) cartesien de BlockRhsEval.
inline void fill_ghosts_polar(MultiFab& U, const Box2D& dom, const BCRec& bc) {
  fill_ghosts(U, dom, bc);
}

/// Foncteur residu polaire R = -div_polar F + S (fill_ghosts puis assemble_rhs_polar). FONCTEUR NOMME
/// (pendant de detail::BlockRhsEval cartesien) : c'est lui que take_step recoit, declenchant
/// l'instanciation d'assemble_rhs_polar<Limiter, Flux> et de ses noyaux device. @c wall_radial : paroi
/// solide radiale (no-penetration) -> masse conservee a la machine (cf. assemble_rhs_polar).
template <class Limiter, class Flux, class Model>
struct PolarBlockRhsEval {
  Model model;
  const PolarGridContext* ctx;
  bool recon_prim;
  bool wall_radial;
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts_polar(U, ctx->dom, ctx->bc);
    assemble_rhs_polar<Limiter, Flux>(model, U, *ctx->aux, ctx->geom, R, recon_prim, wall_radial);
  }
};

/// Avance EXPLICITE polaire : n sous-pas du stepper @c Stepper (SSPRK2 par defaut, SSPRK3 optionnel)
/// sur le residu transport polaire. Pendant de detail::AdvanceExplicit cartesien.
template <class Limiter, class Flux, class Model, class Stepper = SSPRK2Step>
struct PolarAdvanceExplicit {
  Model m;
  PolarGridContext ctx;
  bool recon_prim;
  bool wall_radial;
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const PolarBlockRhsEval<Limiter, Flux, Model> rhs{m, &ctx, recon_prim, wall_radial};
    for (int s = 0; s < n; ++s) Stepper{}.take_step(rhs, U, h);
  }
};

/// Residu polaire fige (fill_ghosts + assemble_rhs_polar) installe comme rhs_into du bloc (eval_rhs).
template <class Limiter, class Flux, class Model>
struct PolarRhsInto {
  Model m;
  PolarGridContext ctx;
  bool recon_prim;
  bool wall_radial;
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts_polar(U, ctx.dom, ctx.bc);
    assemble_rhs_polar<Limiter, Flux>(m, U, *ctx.aux, ctx.geom, R, recon_prim, wall_radial);
  }
};

/// Foncteur vitesse d'onde max du bloc POLAIRE : reduction sur les cellules valides de
/// max_wave_speed(model, U, aux) dans les deux directions (r, theta). Boucle HOTE pure (pas de kernel
/// device) -- la vitesse polaire vient de l'aux (grad_r, grad_theta) deja resident hote apres
/// solve_fields ; cela suffit pour le pas CFL. Pendant de detail::MaxSpeed cartesien.
template <class Model>
struct PolarMaxSpeed {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const {
    Real wmax = Real(0);
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const ConstArray4 a = aux->fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          const typename Model::State us = load_state<Model>(u, i, j);
          const Aux ac = load_aux<aux_comps<Model>()>(a, i, j);
          for (int dir = 0; dir < 2; ++dir) {
            const Real w = m.max_wave_speed(us, ac, dir);
            if (w > wmax) wmax = w;
          }
        }
    }
    return static_cast<Real>(all_reduce_max(static_cast<double>(wmax)));
  }
};

/// Foncteur contribution Poisson POLAIRE : rhs += elliptic_rhs(U) (boucle HOTE pure). IDENTIQUE au
/// cartesien detail::PoissonRhs : la brique elliptique (charge q n) ne porte pas de geometrie ; la
/// metrique polaire (volume r dr dtheta) est portee par le solveur PolarPoissonSolver, pas par le RHS
/// ponctuel par cellule (le solveur attend f tel quel, comme le solveur FFT cartesien).
template <class Model>
struct PolarPoissonRhs {
  Model m;
  void operator()(const MultiFab& U, MultiFab& rhs) const {
    for (int li = 0; li < rhs.local_size(); ++li) {
      Array4 r = rhs.fab(li).array();
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D b = rhs.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          r(i, j) += m.elliptic_rhs(load_state<Model>(u, i, j));
    }
  }
};

}  // namespace detail

/// Derive l'aux POLAIRE en base locale (e_r, e_theta) a partir du potentiel @p phi resolu par
/// PolarPoissonSolver : aux[0] = phi ; aux[1] = grad_r = d phi/dr ; aux[2] = grad_theta = (1/r) d phi/d
/// theta. C'est la disposition attendue par ExBVelocityPolar (v_r = -grad_theta/B, v_theta = grad_r/B).
///
/// INVARIANT CLE (cause du bug corrige) : @p phi est alloue par le solveur direct SANS ghost (mono-box).
/// On ne lit donc JAMAIS un index radial hors domaine : la derivee radiale est CENTREE a l'interieur et
/// DECENTREE d'ordre 2 aux deux parois (i = lo : avant ; i = hi : arriere), sans toucher phi(lo-1) /
/// phi(hi+1). En theta (PERIODIQUE) on enroule l'indice (j-1 -> jhi, j+1 -> jlo) au lieu de lire le ghost
/// azimutal inexistant. Boucle HOTE pure (phi resident hote apres solve()). Ne remplit PAS les ghosts de
/// l'aux : l'appelant le fait APRES (fill_ghosts : theta periodique, r physique) pour le transport.
/// PRECONDITION nr >= 3 (le stencil decentre d'ordre 2 lit p(i+2)/p(i-2) aux parois) : IMPOSEE en
/// amont par check_geometry (python/system.cpp) et adc.PolarMesh (nr >= 3), pas seulement supposee.
inline void derive_aux_polar(const MultiFab& phi, MultiFab& aux, const PolarGeometry& g) {
  const Real dr = g.dr(), dth = g.dtheta();
  for (int li = 0; li < aux.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    Array4 a = aux.fab(li).array();
    const Box2D v = aux.box(li);
    const int ilo = v.lo[0], ihi = v.hi[0], jlo = v.lo[1], jhi = v.hi[1];
    for (int j = jlo; j <= jhi; ++j) {
      const int jm = (j == jlo) ? jhi : j - 1;  // theta periodique : enroulement de l'indice (pas de ghost)
      const int jp = (j == jhi) ? jlo : j + 1;
      for (int i = ilo; i <= ihi; ++i) {
        const Real ri = g.r_cell(i);
        a(i, j, 0) = p(i, j);
        Real gr;  // grad_r = d phi/dr : centree a l'interieur, decentree ordre 2 aux parois (phi sans ghost en r)
        if (i == ilo)
          gr = (Real(-3) * p(i, j) + Real(4) * p(i + 1, j) - p(i + 2, j)) / (Real(2) * dr);
        else if (i == ihi)
          gr = (Real(3) * p(i, j) - Real(4) * p(i - 1, j) + p(i - 2, j)) / (Real(2) * dr);
        else
          gr = (p(i + 1, j) - p(i - 1, j)) / (Real(2) * dr);
        a(i, j, 1) = gr;
        a(i, j, 2) = (p(i, jp) - p(i, jm)) / (Real(2) * dth * ri);  // grad_theta = (1/r) d phi/d theta (deja /r)
      }
    }
  }
}

/// Fermetures (avance + residu) d'un bloc POLAIRE pour un schema spatial (Limiter x Flux) fige. La math
/// RK vient des TimeStepper du coeur (SSPRK2 / SSPRK3). FONCTEURS NOMMES (cf. namespace detail). Pendant
/// de build_block cartesien, mais sans IMEX (Phase 2b transport ExB scalaire : pas de source raide).
/// @p wall_radial : paroi solide radiale (no-penetration) -> conservation de masse a la machine.
template <class Limiter, class Flux, class Model>
BlockClosures build_block_polar(const Model& m, const PolarGridContext& ctx, bool recon_prim,
                                const std::string& method, bool wall_radial) {
  BlockClosures bc;
  if (method == "ssprk3") {
    bc.advance =
        detail::PolarAdvanceExplicit<Limiter, Flux, Model, SSPRK3Step>{m, ctx, recon_prim, wall_radial};
  } else if (method == "ssprk2") {
    bc.advance =
        detail::PolarAdvanceExplicit<Limiter, Flux, Model, SSPRK2Step>{m, ctx, recon_prim, wall_radial};
  } else {
    throw std::runtime_error("System (polaire) : methode temporelle explicite inconnue '" + method +
                             "' (ssprk2|ssprk3)");
  }
  bc.rhs_into = detail::PolarRhsInto<Limiter, Flux, Model>{m, ctx, recon_prim, wall_radial};
  return bc;
}

/// Dispatch du schema spatial (limiteur fige, flux RusanovFlux) -> fermetures polaires compilees.
/// Seul RusanovFlux est cable en polaire : il ne demande que max_wave_speed (donc valable pour
/// l'ExB scalaire ET le fluide isotherme), tandis que HLLC/Roe supposent n_vars==4 (Euler avec
/// energie), sans objet ici. "weno5" route assemble_rhs_polar sur la reconstruction WENO5-Z (3
/// ghosts) comme le cartesien. @p wall_radial : paroi solide radiale (conservation de masse a la
/// machine ; cf. build_block_polar).
template <class Model>
BlockClosures make_block_polar(const Model& m, const std::string& lim, const std::string& riem,
                               const PolarGridContext& ctx, bool recon_prim,
                               const std::string& method, bool wall_radial) {
  // VALIDATION CENTRALISEE (registry dispatch_tags.hpp) AVANT le dispatch : message polaire IDENTIQUE
  // a l'ancien throw inline (seul rusanov est cable en polaire ; HLLC/Roe supposent n_vars==4). Le
  // dispatch des limiteurs qui suit est INCHANGE ; son throw final devient une garde d'incoherence.
  validate_riemann(riem, /*polar=*/true, "System (polaire)");
  validate_limiter(lim, "System (polaire)");
  if (lim == "none") return build_block_polar<NoSlope, RusanovFlux>(m, ctx, recon_prim, method, wall_radial);
  if (lim == "minmod") return build_block_polar<Minmod, RusanovFlux>(m, ctx, recon_prim, method, wall_radial);
  if (lim == "vanleer") return build_block_polar<VanLeer, RusanovFlux>(m, ctx, recon_prim, method, wall_radial);
  if (lim == "weno5") return build_block_polar<Weno5, RusanovFlux>(m, ctx, recon_prim, method, wall_radial);
  throw_registry_dispatch_mismatch("System (polaire)", "limiteur", lim);
}

/// Fermeture de vitesse d'onde max du bloc POLAIRE (pour le pas CFL). @p aux pointe l'aux du System
/// (adresse stable) : la vitesse ExB polaire vient de grad_r / grad_theta de l'aux.
template <class Model>
std::function<Real(const MultiFab&)> make_max_speed_polar(const Model& m, const MultiFab* aux) {
  return detail::PolarMaxSpeed<Model>{m, aux};
}

namespace detail {
/// Fermetures de BORNES DE PAS optionnelles du bloc POLAIRE (StabilityPolicy, audit vague 3) :
/// memes reductions device que le cartesien (noyaux PONCTUELS sans hypothese de geometrie -- la
/// geometrie n'entre que par le pas physique h du stepper, min(dr, r_min*dtheta)). Foncteurs
/// NOMMES (meme contrat device cross-TU que PolarMaxSpeed).
template <class Model>
struct PolarStabilitySpeed {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const { return max_stability_speed_mf(m, U, *aux); }
};
template <class Model>
struct PolarSourceFreq {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const { return max_source_frequency_mf(m, U, *aux); }
};
template <class Model>
struct PolarStabilityDt {
  Model m;
  const MultiFab* aux;
  Real operator()(const MultiFab& U) const { return min_stability_dt_mf(m, U, *aux); }
};
}  // namespace detail

/// Vitesse de CFL du bloc POLAIRE : lambda* (trait HasStabilitySpeed) si le modele le declare,
/// sinon max_wave_speed (PolarMaxSpeed historique, bit-identique) -- MEME politique que
/// make_max_speed cartesien.
template <class Model>
std::function<Real(const MultiFab&)> make_cfl_speed_polar(const Model& m, const MultiFab* aux) {
  if constexpr (HasStabilitySpeed<Model>)
    return detail::PolarStabilitySpeed<Model>{m, aux};
  else
    return detail::PolarMaxSpeed<Model>{m, aux};
}

/// Frequence de source max du bloc POLAIRE (trait HasSourceFrequency) ; VIDE sans trait (le
/// stepper ne l'interroge pas, politique de pas historique).
template <class Model>
std::function<Real(const MultiFab&)> make_source_frequency_polar(const Model& m,
                                                                 const MultiFab* aux) {
  if constexpr (HasSourceFrequency<Model>)
    return detail::PolarSourceFreq<Model>{m, aux};
  else
    return {};
}

/// Pas admissible min du bloc POLAIRE (trait HasStabilityDt) ; VIDE sans trait.
template <class Model>
std::function<Real(const MultiFab&)> make_stability_dt_polar(const Model& m, const MultiFab* aux) {
  if constexpr (HasStabilityDt<Model>)
    return detail::PolarStabilityDt<Model>{m, aux};
  else
    return {};
}

/// Contribution du bloc au second membre du Poisson POLAIRE : rhs += elliptic_rhs(U) (boucle hote).
template <class Model>
std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs_polar(const Model& m) {
  return detail::PolarPoissonRhs<Model>{m};
}

}  // namespace adc

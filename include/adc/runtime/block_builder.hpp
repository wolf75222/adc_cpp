#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/numerics/time/implicit_stepper.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/runtime/grid_context.hpp>  // GridContext + BlockClosures (en-tete leger partage)

#include <functional>
#include <stdexcept>
#include <string>

/// @file
/// @brief Construit les fermetures d'un bloc (avance en temps + residu + contribution Poisson) a
///        partir d'un modele COMPILE (CompositeModel) et d'un contexte de grille.
///
/// Ce code etait dans System::Impl ; il est extrait en en-tete pour que le MEME chemin template
/// (assemble_rhs<Limiter, Flux>, inlinable et device-ready) soit instanciable depuis une UNITE DE
/// TRADUCTION EXTERNE. C'est la brique qui permettra a un modele genere par le DSL d'etre compile
/// AOT (ahead-of-time) puis branche dans le System par le chemin de PRODUCTION (flux HLLC/Roe,
/// ordre 2, GPU), et non plus seulement par le chemin hote virtuel du bloc dynamique.
///
/// Le System reste l'unique proprietaire du maillage et de l'aux ; GridContext n'en porte que des
/// copies immuables (domaine, CL, geometrie) et un POINTEUR non possedant vers l'aux (adresse stable,
/// duree de vie superieure au bloc).

namespace adc {

// GridContext et BlockClosures : definis dans adc/runtime/grid_context.hpp (en-tete leger, inclus
// aussi par system.hpp pour exposer grid_context() / install_block() sans tirer la numerique).

namespace detail {
/// Foncteur residu -div F + S (fill_ghosts puis assemble_rhs), passe AUX TimeStepper comme RhsEval.
/// FONCTEUR NOMME (et non lambda) : c'est lui que take_step recoit et qui declenche l'instanciation
/// d'assemble_rhs<Limiter, Flux> (et de son AssembleRhsKernel device). Premiere-instancie depuis une
/// TU EXTERNE (add_compiled_model), une lambda a cette place fait buter nvcc sur l'emission du kernel
/// device imbrique (Heisenbug : OK Serial + compute-sanitizer, segfault a l'execution Cuda). Une
/// classe a un contexte d'instanciation stable -> codegen device robuste. Corps identique a l'ancienne
/// lambda -> residu bit-identique a add_block sur CPU (et, vise, sur device).
template <class Limiter, class Flux, class Model>
struct BlockRhsEval {
  Model model;
  const GridContext* ctx;
  bool recon_prim;
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx->dom, ctx->bc);
    assemble_rhs<Limiter, Flux>(model, U, *ctx->aux, ctx->geom, R, recon_prim);
  }
};

/// Avance EXPLICITE : n sous-pas de SSPRK2 sur le residu transport+source.
template <class Limiter, class Flux, class Model>
struct AdvanceExplicit {
  Model m;
  GridContext ctx;
  bool recon_prim;
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEval<Limiter, Flux, Model> rhs{m, &ctx, recon_prim};
    for (int s = 0; s < n; ++s) SSPRK2Step{}.take_step(rhs, U, h);
  }
};

/// Avance IMEX : par sous-pas, demi-pas EXPLICITE (transport sans source) + source IMPLICITE raide.
template <class Limiter, class Flux, class Model>
struct AdvanceImex {
  Model m;
  GridContext ctx;
  bool recon_prim;
  void operator()(MultiFab& U, Real dt, int n) const {
    const Real h = dt / static_cast<Real>(n);
    const BlockRhsEval<Limiter, Flux, SourceFreeModel<Model>> rhs{SourceFreeModel<Model>{m}, &ctx,
                                                                  recon_prim};
    for (int s = 0; s < n; ++s) {
      ForwardEuler{}.take_step(rhs, U, h);     // demi-pas explicite : transport sans source
      backward_euler_source(m, *ctx.aux, U, h);  // source implicite (rappel raide)
    }
  }
};

/// Residu fige (fill_ghosts + assemble_rhs) installe comme rhs_into du bloc.
template <class Limiter, class Flux, class Model>
struct RhsInto {
  Model m;
  GridContext ctx;
  bool recon_prim;
  void operator()(MultiFab& U, MultiFab& R) const {
    fill_ghosts(U, ctx.dom, ctx.bc);
    assemble_rhs<Limiter, Flux>(m, U, *ctx.aux, ctx.geom, R, recon_prim);
  }
};
}  // namespace detail

/// Fermetures (avance + residu) pour un schema spatial (Limiter x Flux) fige. La math RK vient des
/// TimeStepper du coeur : SSPRK2 en explicite ; ForwardEuler + backward_euler_source en IMEX. Les
/// fermetures sont des FONCTEURS NOMMES (cf. namespace detail) et non des lambdas : le chemin
/// add_compiled_model (premiere instanciation depuis une TU externe) s'emet alors proprement sous nvcc.
template <class Limiter, class Flux, class Model>
BlockClosures build_block(const Model& m, const GridContext& ctx, bool imex, bool recon_prim) {
  BlockClosures bc;
  if (imex)
    bc.advance = detail::AdvanceImex<Limiter, Flux, Model>{m, ctx, recon_prim};
  else
    bc.advance = detail::AdvanceExplicit<Limiter, Flux, Model>{m, ctx, recon_prim};
  bc.rhs_into = detail::RhsInto<Limiter, Flux, Model>{m, ctx, recon_prim};
  return bc;
}

/// Dispatch du schema spatial (limiteur x flux Riemann) -> fermetures compilees. HLLC / Roe gardes
/// par requires : exigent un transport a 4 variables exposant pressure (sinon erreur explicite).
template <class Model>
BlockClosures make_block(const Model& m, const std::string& lim, const std::string& riem,
                         const GridContext& ctx, bool imex, bool recon_prim) {
  if (riem == "rusanov") {
    if (lim == "none") return build_block<NoSlope, RusanovFlux>(m, ctx, imex, recon_prim);
    if (lim == "minmod") return build_block<Minmod, RusanovFlux>(m, ctx, imex, recon_prim);
    if (lim == "vanleer") return build_block<VanLeer, RusanovFlux>(m, ctx, imex, recon_prim);
    throw std::runtime_error("System : limiter inconnu '" + lim + "'");
  }
  if (riem == "hllc") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_block<NoSlope, HLLCFlux>(m, ctx, imex, recon_prim);
      if (lim == "minmod") return build_block<Minmod, HLLCFlux>(m, ctx, imex, recon_prim);
      if (lim == "vanleer") return build_block<VanLeer, HLLCFlux>(m, ctx, imex, recon_prim);
      throw std::runtime_error("System : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("System : flux 'hllc' exige un transport compressible "
                               "(4 variables + pression) ; ce transport -> 'rusanov'");
    }
  }
  if (riem == "roe") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_block<NoSlope, RoeFlux>(m, ctx, imex, recon_prim);
      if (lim == "minmod") return build_block<Minmod, RoeFlux>(m, ctx, imex, recon_prim);
      if (lim == "vanleer") return build_block<VanLeer, RoeFlux>(m, ctx, imex, recon_prim);
      throw std::runtime_error("System : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("System : flux 'roe' exige un transport compressible "
                               "(4 variables + pression) ; ce transport -> 'rusanov'");
    }
  }
  throw std::runtime_error("System : flux Riemann inconnu '" + riem + "' (rusanov|hllc|roe)");
}

namespace detail {
/// Foncteur vitesse d'onde max du bloc (max_wave_speed_mf, reduction par le seam). FONCTEUR NOMME :
/// max_wave_speed_mf instancie MaxWaveSpeedKernel (deja un foncteur device) ; l'envelopper dans une
/// classe nommee plutot qu'une lambda preserve le contexte d'instanciation cross-TU sous nvcc.
template <class Model>
struct MaxSpeed {
  Model m;
  GridContext ctx;
  Real operator()(const MultiFab& U) const { return max_wave_speed_mf(m, U, *ctx.aux); }
};

/// Foncteur contribution Poisson : rhs += elliptic_rhs(U) (boucle HOTE pure, pas de kernel device).
template <class Model>
struct PoissonRhs {
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

/// Fermeture de vitesse d'onde max du bloc (pour le pas CFL).
template <class Model>
std::function<Real(const MultiFab&)> make_max_speed(const Model& m, const GridContext& ctx) {
  return detail::MaxSpeed<Model>{m, ctx};
}

/// Contribution du bloc au second membre de Poisson : rhs += elliptic_rhs(U) (boucle hote).
template <class Model>
std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs(const Model& m) {
  return detail::PoissonRhs<Model>{m};
}

}  // namespace adc

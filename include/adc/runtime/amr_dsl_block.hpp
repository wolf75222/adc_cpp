#pragma once

#include <adc/coupling/amr_coupler_mp.hpp>  // AmrCouplerMP, AmrLevelMP
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/refinement.hpp>  // coarsen_index
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/parallel/comm.hpp>  // n_ranks
#include <adc/runtime/amr_system.hpp>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief add_compiled_model cote AmrSystem : branche un modele COMPILE (un CompositeModel, genere
///        par le DSL ou ecrit a la main, connu a la COMPILATION) comme l'unique bloc d'une hierarchie
///        AMR, EXACTEMENT le chemin de production de AmrSystem::add_block mais SANS passer par la
///        dispatch ModelSpec (le modele est deja un type concret).
///
/// Pendant raffine de add_compiled_model(System&, ...) (dsl_block.hpp). La machinerie de build du
/// coupleur AMR (AmrCouplerMP<Model> + reflux conservatif + regrid) est instanciee ICI, depuis l'unite
/// de traduction APPELANTE, sur le type Model concret -- comme block_builder.hpp pour le System plat.
/// Le coupleur, type-erased en std::function (step / mass / max_speed / n_patches / density), entre
/// dans AmrSystem par AmrSystem::set_compiled_block (methode non-template). Le MEME builder partage
/// (detail::build_amr_compiled / dispatch_amr_compiled) sert AUSSI le chemin ModelSpec natif d'add_block
/// (amr_system.cpp), une fois le type Model concret resolu par detail::dispatch_model : un seul build.

namespace adc {

/// Paquet (limiteur, flux Riemann) attendu par AmrCouplerMP::step<Disc>. Unique definition : le
/// chemin natif d'amr_system.cpp passe par ce meme header (plus de DiscLF duplique cote .cpp).
template <class L, class F>
struct AmrDiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

namespace detail {

/// Construit le coupleur AMR pour un Model compose + (Limiter, Flux) concrets et remplit les hooks
/// type-erased. Deux niveaux : grossier + un patch fin seed central, remodele par le regrid. C'est le
/// pendant header de AmrSystem::Impl::build, instancie depuis la TU appelante sur le type Model. Les
/// helpers de grossier (layout, write/read/inject) sont PARTAGES avec le chemin natif via
/// amr_coupler_mp.hpp (detail::coupler_*), donc replique et reparti suivent exactement la meme logique.
template <class Model, class Limiter, class Flux>
AmrCompiledHooks build_amr_compiled(const Model& model, const AmrBuildParams& bp) {
  using Coupler = AmrCouplerMP<Model>;
  const int nc = Model::n_vars;
  const Geometry g{Box2D::from_extents(bp.n, bp.n), 0.0, bp.L, 0.0, bp.L};
  const double dxc = bp.L / bp.n, dxf = dxc / 2;
  // Niveau 0 (grossier) : layout decide par la politique d'ownership (replique mono-box par defaut,
  // multi-box reparti si bp.distribute_coarse). En replique, dmap = my_rank() partout (la box vit
  // sur chaque rang ; un round-robin la poserait sur le seul rang 0 -> fab hors bornes ailleurs,
  // segfault sous np>1). Le seed fin part sur la MEME dmap que le grossier ; le regrid initial le
  // RECONSTRUIT puis le REPARTIT round-robin (DistributionMapping(nfine, n_ranks())) -> distribution
  // multi-GPU des patchs fins. En reparti, le grossier se distribue AUSSI (strong-scaling AMR).
  const auto [bac, dm] =
      coupler_make_coarse_layout(bp.n, bp.distribute_coarse, bp.coarse_max_grid);
  const int ng = Limiter::n_ghost;  // stencil du limiteur (1 NoSlope, 2 MUSCL) : parite du schema
  MultiFab Uc(bac, dm, nc, ng);
  Uc.set_val(Real(0));
  const int I0 = bp.n / 4, I1 = 3 * bp.n / 4 - 1, J0 = bp.n / 4, J1 = 3 * bp.n / 4 - 1;
  Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
  BoxArray baf(std::vector<Box2D>{fb});
  MultiFab Uf(baf, dm, nc, ng);
  Uf.set_val(Real(0));
  std::vector<AmrLevelMP> levels;
  levels.push_back({std::move(Uc), nullptr, dxc, dxc});
  levels.push_back({std::move(Uf), nullptr, dxf, dxf});

  auto cpl = std::make_shared<Coupler>(model, g, bac, bp.poisson_bc, std::move(levels), bp.wall,
                                       !bp.distribute_coarse);
  if (bp.has_density) coupler_write_coarse(cpl->coarse(), bp.density, bp.n, nc, bp.gamma);
  auto& Lv = cpl->levels();
  for (std::size_t k = 1; k < Lv.size(); ++k)
    coupler_inject_coarse_to_fine_mb(cpl->coarse(), Lv[k].U, !bp.distribute_coarse);

  const double thr = bp.refine_threshold;
  auto crit = [thr](const ConstArray4& a, int i, int j) { return a(i, j, 0) > thr; };
  cpl->regrid(crit);
  cpl->update();

  AmrCompiledHooks h;
  h.coupler_holder = cpl;  // duree de vie : les fermetures capturent cpl (shared_ptr)
  const int sub = bp.substeps;
  const bool rprim = bp.recon_prim;
  const int regrid_every = bp.regrid_every;
  auto step_state = std::make_shared<int>(0);  // compteur de pas partage par la fermeture
  h.step = [cpl, crit, sub, rprim, regrid_every, step_state](double dt) {
    if (regrid_every > 0 && *step_state > 0 && *step_state % regrid_every == 0) cpl->regrid(crit);
    const double h2 = dt / sub;
    for (int s = 0; s < sub; ++s) cpl->template step<AmrDiscLF<Limiter, Flux>>(h2, rprim);
    ++*step_state;
  };
  h.max_speed = [cpl] { return static_cast<double>(cpl->max_wave_speed()); };
  h.mass = [cpl] { return static_cast<double>(cpl->mass()); };
  h.n_patches = [cpl] {
    auto& L = cpl->levels();
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
  };
  const int nn = bp.n;
  const bool repl = !bp.distribute_coarse;
  h.density = [cpl, nn, repl] { return coupler_read_coarse(cpl->coarse(), nn, repl); };
  // phi du grossier : on rafraichit (update() = sync_down + compute_aux, donc solve Poisson grossier)
  // puis on lit aux0 composante 0. Pendant de System::potential() qui appelle ensure_elliptic : la
  // valeur est courante meme si aucun step n'a encore tourne. update() est deja appele a chaque step,
  // donc le surcout n'existe que sur un appel hors boucle (diagnostic).
  h.potential = [cpl, nn, repl] {
    cpl->update();
    return coupler_read_coarse_phi(cpl->aux0(), nn, repl);
  };
  return h;
}

/// Dispatch du schema spatial (limiteur x flux Riemann) -> build_amr_compiled. Memes gardes que
/// AmrSystem::add_block (hllc/roe exigent un transport compressible a 4 variables + pression).
template <class Model>
AmrCompiledHooks dispatch_amr_compiled(const Model& m, const std::string& lim,
                                       const std::string& riem, const AmrBuildParams& bp) {
  if (riem == "rusanov") {
    if (lim == "none") return build_amr_compiled<Model, NoSlope, RusanovFlux>(m, bp);
    if (lim == "minmod") return build_amr_compiled<Model, Minmod, RusanovFlux>(m, bp);
    if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, RusanovFlux>(m, bp);
    // WENO5-Z (3 ghosts) : meme mecanisme que System (block_n_ghost(limiter)). Ici les niveaux du
    // coupleur sont alloues a Limiter::n_ghost (build_amr_compiled : ng = Weno5::n_ghost = 3) et le
    // regrid HERITE n_grow() (amr_regrid_finest : ngf = L[fk].U.n_grow()), donc le stencil 5 points
    // ne lit pas hors bornes. Cable sur AMR au MEME titre que none/minmod (rusanov uniquement).
    if (lim == "weno5") return build_amr_compiled<Model, Weno5, RusanovFlux>(m, bp);
    throw std::runtime_error("add_compiled_model(AmrSystem) : limiter inconnu '" + lim + "'");
  }
  if (riem == "hllc") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_amr_compiled<Model, NoSlope, HLLCFlux>(m, bp);
      if (lim == "minmod") return build_amr_compiled<Model, Minmod, HLLCFlux>(m, bp);
      if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, HLLCFlux>(m, bp);
      throw std::runtime_error("add_compiled_model(AmrSystem) : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("add_compiled_model(AmrSystem) : flux 'hllc' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  if (riem == "roe") {
    if constexpr (Model::n_vars == 4 &&
                  requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
      if (lim == "none") return build_amr_compiled<Model, NoSlope, RoeFlux>(m, bp);
      if (lim == "minmod") return build_amr_compiled<Model, Minmod, RoeFlux>(m, bp);
      if (lim == "vanleer") return build_amr_compiled<Model, VanLeer, RoeFlux>(m, bp);
      throw std::runtime_error("add_compiled_model(AmrSystem) : limiter inconnu '" + lim + "'");
    } else {
      throw std::runtime_error("add_compiled_model(AmrSystem) : flux 'roe' exige un transport "
                               "compressible (4 variables + pression)");
    }
  }
  throw std::runtime_error("add_compiled_model(AmrSystem) : flux Riemann inconnu '" + riem +
                           "' (rusanov|hllc|roe)");
}

}  // namespace detail

/// Branche @p model (CompositeModel concret) comme l'unique bloc AMR de @p sys, avec le schema demande.
/// Le build du coupleur est DIFFERE (comme add_block) : la fermeture capturee est invoquee au premier
/// step/mass/density via ensure_built(), apres set_refinement / set_poisson / set_density.
/// @throws std::runtime_error si un bloc est deja defini ou si time != "explicit".
template <class Model>
void add_compiled_model(AmrSystem& sys, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit", double gamma = 1.4, int substeps = 1) {
  (void)name;
  if (substeps < 1) throw std::runtime_error("add_compiled_model(AmrSystem) : substeps >= 1");
  if (time != "explicit")
    throw std::runtime_error("add_compiled_model(AmrSystem) : seul time='explicit' sur AMR");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("add_compiled_model(AmrSystem) : recon inconnu '" + recon +
                             "' (conservative|primitive)");
  const bool recon_prim = (recon == "primitive");
  // Builder type-erase : capture le Model concret + le schema, materialise le coupleur au build
  // paresseux (avec les parametres refine/poisson/density figes a ce moment-la).
  auto builder = [model, limiter, riemann, recon_prim](const AmrBuildParams& bp) {
    AmrBuildParams p = bp;
    p.recon_prim = recon_prim;
    return detail::dispatch_amr_compiled(model, limiter, riemann, p);
  };
  sys.set_compiled_block(Model::n_vars, gamma, substeps, std::move(builder));
}

}  // namespace adc

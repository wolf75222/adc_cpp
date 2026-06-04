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
/// dans AmrSystem par AmrSystem::install_compiled (methode non-template, symetrique d'install_block).

namespace adc {

/// Paquet (limiteur, flux Riemann) attendu par AmrCouplerMP::step<Disc> (identique a amr_system.cpp).
template <class L, class F>
struct AmrDiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

namespace detail {

/// Injecte le grossier (mono-box) dans les cellules valides d'un patch fin (constant par morceaux,
/// ratio 2) : rend la hierarchie coherente avant le 1er sync_down (cf. amr_system.cpp).
inline void amr_inject_coarse_to_fine(const MultiFab& Uc, MultiFab& Uf) {
  device_fence();
  const int nc = Uf.ncomp();
  const ConstArray4 c = Uc.fab(0).const_array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const Box2D v = Uf.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const int ci = coarsen_index(i, 2), cj = coarsen_index(j, 2);
        for (int k = 0; k < nc; ++k) f(i, j, k) = c(ci, cj, k);
      }
  }
}

/// Ecrit une densite initiale (composante 0, n*n row-major) sur le niveau grossier.
inline void amr_write_coarse(MultiFab& U, const std::vector<double>& rho, int n, int ncomp,
                             double gamma) {
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("add_compiled_model(AmrSystem) : densite de taille != n*n");
  const Real gm1 = Real(gamma) - Real(1);
  Array4 u = U.fab(0).array();
  const Box2D v = U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const Real r = rho[static_cast<std::size_t>(j) * n + i];
      u(i, j, 0) = r;
      if (ncomp >= 3) { u(i, j, 1) = 0; u(i, j, 2) = 0; }
      if (ncomp == 4) u(i, j, 3) = r / gm1;
    }
}

inline std::vector<double> amr_read_coarse(const MultiFab& U) {
  device_fence();
  const ConstArray4 u = U.fab(0).const_array();
  const Box2D v = U.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
  return out;
}

/// Construit le coupleur AMR pour un Model compose + (Limiter, Flux) concrets et remplit les hooks
/// type-erased. Deux niveaux : grossier + un patch fin seed central, remodele par le regrid. C'est le
/// pendant header de AmrSystem::Impl::build, instancie depuis la TU appelante sur le type Model.
template <class Model, class Limiter, class Flux>
AmrCompiledHooks build_amr_compiled(const Model& model, const AmrBuildParams& bp) {
  using Coupler = AmrCouplerMP<Model>;
  const int nc = Model::n_vars;
  const Geometry g{Box2D::from_extents(bp.n, bp.n), 0.0, bp.L, 0.0, bp.L};
  const double dxc = bp.L / bp.n, dxf = dxc / 2;
  // Le niveau 0 (grossier mono-box) est REPLIQUE sur tous les rangs : c'est le defaut performant
  // de AmrCouplerMP (replicated_coarse=true) et le layout que GeometricMG construit en interne
  // (dmap = my_rank() partout). Le construire en round-robin DistributionMapping(1, n_ranks())
  // poserait la box sur le seul rang 0 -> coarse().fab(0) hors bornes sur les autres rangs (segfault
  // au premier write/inject/read sous np>1). En serie my_rank()=0 : identique au round-robin, bit a
  // bit. Le seed fin part egalement replique ; le regrid initial le RECONSTRUIT puis le REPARTIT en
  // round-robin (DistributionMapping(nfine, n_ranks()), cf amr_regrid_coupler.hpp) -> distribution
  // multi-GPU des patchs fins. C'est ce qui fait du run un calcul AMR reellement distribue.
  const DistributionMapping dm(std::vector<int>{my_rank()});
  const int ng = Limiter::n_ghost;  // stencil du limiteur (1 NoSlope, 2 MUSCL) : parite du schema
  BoxArray bac(std::vector<Box2D>{Box2D::from_extents(bp.n, bp.n)});
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

  auto cpl = std::make_shared<Coupler>(model, g, bac, bp.poisson_bc, std::move(levels), bp.wall);
  if (bp.has_density) amr_write_coarse(cpl->coarse(), bp.density, bp.n, nc, bp.gamma);
  auto& Lv = cpl->levels();
  for (std::size_t k = 1; k < Lv.size(); ++k) amr_inject_coarse_to_fine(cpl->coarse(), Lv[k].U);

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
  h.density = [cpl] { return amr_read_coarse(cpl->coarse()); };
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

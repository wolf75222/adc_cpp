#include <adc/runtime/amr_system.hpp>

#include <adc/runtime/model_factory.hpp>     // detail::dispatch_model + briques compilees
#include <adc/coupling/amr_coupler_mp.hpp>   // AmrCouplerMP, AmrLevelMP
#include <adc/operator/numerical_flux.hpp>    // RusanovFlux, HLLCFlux
#include <adc/operator/reconstruction.hpp>    // NoSlope, Minmod, VanLeer

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>  // n_ranks

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace adc {

/// Paquet (limiteur, flux Riemann) attendu par AmrCouplerMP::step<Disc>.
template <class L, class F>
struct DiscLF {
  using Limiter = L;
  using NumericalFlux = F;
};

struct AmrSystem::Impl {
  AmrSystemConfig cfg;

  // Specification du bloc (figee a add_block, materialisee au build paresseux).
  bool has_block = false;
  ModelSpec b_spec;
  std::string b_limiter = "minmod", b_riemann = "rusanov";
  int b_substeps = 1;
  int ncomp = 1;
  double gamma = 1.4;

  double refine_threshold = 1e30;  // 1e30 => aucun raffinement par defaut

  std::string p_rhs = "charge_density", p_solver = "geometric_mg", p_bc = "auto",
              p_wall = "none";
  double p_wall_radius = 0.0;

  std::vector<double> pending_density;
  bool has_density = false;

  bool built = false;
  std::shared_ptr<void> coupler_holder;
  std::function<void(double)> step_fn;
  std::function<double()> max_speed_fn;
  std::function<double()> mass_fn;
  std::function<int()> n_patches_fn;
  std::function<std::vector<double>()> density_fn;
  double t = 0;
  int step_count = 0;

  explicit Impl(const AmrSystemConfig& c) : cfg(c) {}

  Geometry geom() const {
    return Geometry{Box2D::from_extents(cfg.n, cfg.n), 0.0, cfg.L, 0.0, cfg.L};
  }
  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
    BCRec b;
    if (mode == "periodic") return b;
    if (mode == "dirichlet") { b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet; return b; }
    if (mode == "neumann") { b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap; return b; }
    throw std::runtime_error("AmrSystem::set_poisson : bc inconnu '" + mode + "'");
  }
  std::function<bool(Real, Real)> wall_active() {
    if (p_wall == "none") return {};
    if (p_wall == "circle") {
      const double cx = 0.5 * cfg.L, cy = 0.5 * cfg.L, R = p_wall_radius;
      return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
    }
    throw std::runtime_error("AmrSystem::set_poisson : wall inconnu '" + p_wall + "'");
  }

  void write_coarse(MultiFab& U, const std::vector<double>& rho) {
    const int n = cfg.n;
    if (static_cast<int>(rho.size()) != n * n)
      throw std::runtime_error("AmrSystem::set_density : taille != n*n");
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

  static std::vector<double> read_coarse(const MultiFab& U) {
    device_fence();
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
    return out;
  }

  // Construit le coupleur AMR pour un Model compose + (Limiter, Flux) concrets et cable les
  // fermetures. Deux niveaux : grossier + un patch fin seed central, remodele par le regrid.
  template <class Model, class L, class F>
  void build(const Model& model) {
    using Coupler = AmrCouplerMP<Model>;
    const int nc = Model::n_vars;
    const Geometry g = geom();
    const double dxc = cfg.L / cfg.n, dxf = dxc / 2;
    DistributionMapping dm(1, n_ranks());
    BoxArray bac(std::vector<Box2D>{Box2D::from_extents(cfg.n, cfg.n)});
    MultiFab Uc(bac, dm, nc, 1);
    Uc.set_val(Real(0));
    const int I0 = cfg.n / 4, I1 = 3 * cfg.n / 4 - 1, J0 = cfg.n / 4, J1 = 3 * cfg.n / 4 - 1;
    Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
    BoxArray baf(std::vector<Box2D>{fb});
    MultiFab Uf(baf, dm, nc, 1);
    Uf.set_val(Real(0));
    std::vector<AmrLevelMP> levels;
    levels.push_back({std::move(Uc), nullptr, dxc, dxc});
    levels.push_back({std::move(Uf), nullptr, dxf, dxf});

    auto cpl = std::make_shared<Coupler>(model, g, bac, poisson_bc(), std::move(levels),
                                         wall_active());
    coupler_holder = cpl;
    if (has_density) write_coarse(cpl->coarse(), pending_density);

    const double thr = refine_threshold;
    auto crit = [thr](const ConstArray4& a, int i, int j) { return a(i, j, 0) > thr; };
    cpl->regrid(crit);
    cpl->update();

    Impl* P = this;
    const int sub = b_substeps;
    step_fn = [P, cpl, crit, sub](double dt) {
      if (P->cfg.regrid_every > 0 && P->step_count > 0 &&
          P->step_count % P->cfg.regrid_every == 0)
        cpl->regrid(crit);
      const double h = dt / sub;
      for (int s = 0; s < sub; ++s) cpl->template step<DiscLF<L, F>>(h);
      ++P->step_count;
    };
    max_speed_fn = [cpl] { return static_cast<double>(cpl->max_wave_speed()); };
    mass_fn = [cpl] { return static_cast<double>(cpl->mass()); };
    n_patches_fn = [cpl] {
      auto& Lv = cpl->levels();
      return Lv.size() >= 2 ? static_cast<int>(Lv[1].U.box_array().size()) : 0;
    };
    density_fn = [cpl] { return read_coarse(cpl->coarse()); };
    built = true;
  }

  // Dispatch du schema spatial (limiteur x flux Riemann) pour un Model compose.
  template <class Model>
  void dispatch_spatial(const Model& m) {
    if (b_riemann == "rusanov") {
      if (b_limiter == "none") return build<Model, NoSlope, RusanovFlux>(m);
      if (b_limiter == "minmod") return build<Model, Minmod, RusanovFlux>(m);
      if (b_limiter == "vanleer") return build<Model, VanLeer, RusanovFlux>(m);
      throw std::runtime_error("AmrSystem : limiter inconnu '" + b_limiter + "'");
    }
    if (b_riemann == "hllc") {
      if constexpr (Model::n_vars == 4 &&
                    requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
        if (b_limiter == "none") return build<Model, NoSlope, HLLCFlux>(m);
        if (b_limiter == "minmod") return build<Model, Minmod, HLLCFlux>(m);
        if (b_limiter == "vanleer") return build<Model, VanLeer, HLLCFlux>(m);
        throw std::runtime_error("AmrSystem : limiter inconnu '" + b_limiter + "'");
      } else {
        throw std::runtime_error("AmrSystem : flux 'hllc' exige un transport compressible "
                                 "(4 variables + pression) ; ce transport -> 'rusanov'");
      }
    }
    throw std::runtime_error("AmrSystem : flux Riemann inconnu '" + b_riemann + "'");
  }

  void ensure_built() {
    if (built) return;
    if (!has_block) throw std::runtime_error("AmrSystem : appeler add_block d'abord");
    detail::dispatch_model(b_spec, [&](auto m) {
      using M = decltype(m);
      ncomp = M::n_vars;
      gamma = b_spec.gamma;
      dispatch_spatial(m);
    });
  }
};

AmrSystem::AmrSystem(const AmrSystemConfig& c) : p_(std::make_unique<Impl>(c)) {}
AmrSystem::~AmrSystem() = default;
AmrSystem::AmrSystem(AmrSystem&&) noexcept = default;
AmrSystem& AmrSystem::operator=(AmrSystem&&) noexcept = default;

void AmrSystem::add_block(const std::string& name, const ModelSpec& model,
                          const std::string& limiter, const std::string& riemann,
                          const std::string& time, int substeps) {
  (void)name;
  if (p_->has_block) throw std::runtime_error("AmrSystem : un seul bloc (AMR mono-modele)");
  if (substeps < 1) throw std::runtime_error("AmrSystem::add_block : substeps >= 1");
  if (time != "explicit")
    throw std::runtime_error("AmrSystem : seul time='explicit' est supporte sur AMR");
  p_->b_spec = model;
  p_->b_limiter = limiter;
  p_->b_riemann = riemann;
  p_->b_substeps = substeps;
  p_->has_block = true;
}

void AmrSystem::set_refinement(double threshold) { p_->refine_threshold = threshold; }

void AmrSystem::set_poisson(const std::string& rhs, const std::string& solver,
                            const std::string& bc, const std::string& wall,
                            double wall_radius) {
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
}

void AmrSystem::set_density(const std::string& name, const std::vector<double>& rho) {
  (void)name;
  p_->pending_density = rho;
  p_->has_density = true;
}

void AmrSystem::step(double dt) {
  p_->ensure_built();
  p_->step_fn(dt);
  p_->t += dt;
}
void AmrSystem::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}
double AmrSystem::step_cfl(double cfl) {
  p_->ensure_built();
  const double h = cfl * (p_->cfg.L / p_->cfg.n) / p_->max_speed_fn();
  p_->step_fn(h);
  p_->t += h;
  return h;
}

int AmrSystem::nx() const { return p_->cfg.n; }
double AmrSystem::time() const { return p_->t; }
int AmrSystem::n_patches() {
  p_->ensure_built();
  return p_->n_patches_fn();
}
double AmrSystem::mass() {
  p_->ensure_built();
  return p_->mass_fn();
}
std::vector<double> AmrSystem::density() {
  p_->ensure_built();
  return p_->density_fn();
}

}  // namespace adc

#include <adc/runtime/system.hpp>

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/integrator/implicit_stepper.hpp>   // backward_euler_source
#include <adc/integrator/time_steppers.hpp>      // ForwardEuler, SSPRK2Step (math RK du coeur)
#include <adc/model/charged_fluid.hpp>           // ChargedEuler, ChargedEulerIsothermal (+ Euler)
#include <adc/model/diocotron.hpp>
#include <adc/model/euler_poisson.hpp>           // EulerPoisson (auto-gravite / Langmuir)
#include <adc/operator/spatial_operator.hpp>     // assemble_rhs, SourceFreeModel, max_wave_speed_mf, load_state

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // saxpy, lincomb, sum
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>  // fill_ghosts, fill_boundary

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <variant>

namespace adc {

struct System::Impl {
  enum class Kind { Diocotron, Euler, Isothermal };

  // Fermetures compilees figees a l'ajout du bloc (modele + schema spatial + temps).
  // Type-erased SEULEMENT au niveau de la liste de blocs ; le noyau reste compile.
  struct Species {
    std::string name;
    MultiFab U;
    double charge;
    Kind kind;
    int ncomp;
    std::function<void(MultiFab&, Real)> advance;             // un macro-pas (sous-pas inclus)
    std::function<void(MultiFab&, MultiFab&)> rhs_into;        // R <- -div F + S (Poisson fige)
    std::function<Real(const MultiFab&)> max_speed;           // max |vitesse d'onde| du bloc
    std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;  // += elliptic_rhs(U)
  };

  SystemConfig cfg;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc_;        // CL transport (periodique ou Foextrap selon cfg.periodic)
  BCRec bc_aux_;    // CL aux (= bc_) pour le cas non periodique
  Box2D dom;
  Periodicity per_;
  bool periodic_;
  MultiFab aux;
  std::vector<Species> sp;
  double t = 0;

  // Configuration Poisson (le solveur elliptique est construit paresseusement).
  std::string p_rhs = "charge_density";
  std::string p_solver = "geometric_mg";
  std::string p_bc = "auto";
  std::string p_wall = "none";
  double p_wall_radius = 0.0;
  std::optional<std::variant<GeometricMG, PoissonFFTSolver>> ell_;

  explicit Impl(const SystemConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        bc_(make_bc(c)),
        bc_aux_(make_bc(c)),
        dom(Box2D::from_extents(c.n, c.n)),
        per_{c.periodic, c.periodic},
        periodic_(c.periodic),
        aux(ba, dm, 3, 1) {}

  static BCRec make_bc(const SystemConfig& c) {
    BCRec b;  // periodique par defaut
    if (!c.periodic) b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }

  Species& find(const std::string& name) {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  const Species& find(const std::string& name) const {
    for (auto& s : sp)
      if (s.name == name) return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }

  // --- solveur elliptique (Poisson de systeme) -----------------------------
  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
    BCRec b;  // periodique par defaut
    if (mode == "periodic") return b;
    if (mode == "dirichlet") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;
      return b;
    }
    if (mode == "neumann") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
      return b;
    }
    throw std::runtime_error("System::set_poisson : bc inconnu '" + mode +
                             "' (auto|periodic|dirichlet|neumann)");
  }
  std::function<bool(Real, Real)> wall_active() {
    if (p_wall == "none") return {};
    if (p_wall == "circle") {
      const double cx = 0.5 * cfg.L, cy = 0.5 * cfg.L, R = p_wall_radius;
      return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
    }
    throw std::runtime_error("System::set_poisson : wall inconnu '" + p_wall +
                             "' (none|circle)");
  }
  void ensure_elliptic() {
    if (ell_) return;
    if (p_rhs != "charge_density")
      throw std::runtime_error("System::set_poisson : rhs '" + p_rhs +
                               "' inconnu (seul 'charge_density')");
    const BCRec pbc = poisson_bc();
    std::function<bool(Real, Real)> active = wall_active();
    if (p_solver == "fft") {
      if (active)
        throw std::runtime_error("System : solver 'fft' incompatible avec une paroi "
                                 "conductrice -> 'geometric_mg'");
      ell_.emplace(std::in_place_type<PoissonFFTSolver>, geom, ba, pbc, active);
    } else if (p_solver == "geometric_mg") {
      ell_.emplace(std::in_place_type<GeometricMG>, geom, ba, pbc, std::move(active));
    } else {
      throw std::runtime_error("System::set_poisson : solver '" + p_solver +
                               "' inconnu (geometric_mg|fft)");
    }
  }
  MultiFab& ell_rhs() {
    return std::visit([](auto& e) -> MultiFab& { return e.rhs(); }, *ell_);
  }
  MultiFab& ell_phi() {
    return std::visit([](auto& e) -> MultiFab& { return e.phi(); }, *ell_);
  }
  void ell_solve() {
    std::visit([](auto& e) { e.solve(); }, *ell_);
  }

  // --- schemas spatiaux compiles -------------------------------------------
  // L'evaluateur methode-des-lignes d'un bloc (L/F/Model figes) : ghosts puis
  // R = -div F + S. La math RK est portee par les TimeStepper du coeur, pas reimplementee.
  template <class Limiter, class Flux, class Model>
  auto rhs_eval(const Model& model) {
    return [this, &model](MultiFab& U, MultiFab& R) {
      fill_ghosts(U, dom, bc_);
      assemble_rhs<Limiter, Flux>(model, U, aux, geom, R);
    };
  }
  // SSPRK2 du coeur sur le RHS du bloc.
  template <class Limiter, class Flux, class Model>
  void ssprk2(const Model& model, MultiFab& U, Real dt) {
    SSPRK2Step{}.take_step(rhs_eval<Limiter, Flux>(model), U, dt);
  }
  // IMEX : transport explicite (modele source-free, Euler avant du coeur) puis source
  // raide implicite (backward-Euler / Newton local).
  template <class Limiter, class Flux, class Model>
  void imex_step(const Model& model, MultiFab& U, Real dt) {
    const SourceFreeModel<Model> sf{model};
    ForwardEuler{}.take_step(rhs_eval<Limiter, Flux>(sf), U, dt);
    backward_euler_source(model, aux, U, dt);
  }

  struct BlockClosures {
    std::function<void(MultiFab&, Real)> advance;
    std::function<void(MultiFab&, MultiFab&)> rhs_into;
  };

  template <class Limiter, class Flux, class Model>
  BlockClosures build(const Model& m, bool imex, int substeps) {
    Impl* P = this;
    BlockClosures bc;
    if (imex)
      bc.advance = [P, m, substeps](MultiFab& U, Real dt) {
        const Real h = dt / static_cast<Real>(substeps);
        for (int s = 0; s < substeps; ++s) P->imex_step<Limiter, Flux>(m, U, h);
      };
    else
      bc.advance = [P, m, substeps](MultiFab& U, Real dt) {
        const Real h = dt / static_cast<Real>(substeps);
        for (int s = 0; s < substeps; ++s) P->ssprk2<Limiter, Flux>(m, U, h);
      };
    // residu nu (Poisson/aux fige par l'appelant) : pour un integrateur custom Python.
    bc.rhs_into = [P, m](MultiFab& U, MultiFab& R) {
      fill_ghosts(U, P->dom, P->bc_);
      assemble_rhs<Limiter, Flux>(m, U, P->aux, P->geom, R);
    };
    return bc;
  }

  template <class Model>
  BlockClosures make_block(const Model& m, const std::string& lim, const std::string& flx,
                           bool imex, int substeps) {
    if (flx == "rusanov") {
      if (lim == "none") return build<NoSlope, RusanovFlux>(m, imex, substeps);
      if (lim == "minmod") return build<Minmod, RusanovFlux>(m, imex, substeps);
      if (lim == "vanleer") return build<VanLeer, RusanovFlux>(m, imex, substeps);
      throw std::runtime_error("System : limiter inconnu '" + lim + "'");
    }
    if (flx == "hllc") {
      if constexpr (Model::n_vars == 4 &&
                    requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
        if (lim == "none") return build<NoSlope, HLLCFlux>(m, imex, substeps);
        if (lim == "minmod") return build<Minmod, HLLCFlux>(m, imex, substeps);
        if (lim == "vanleer") return build<VanLeer, HLLCFlux>(m, imex, substeps);
        throw std::runtime_error("System : limiter inconnu '" + lim + "'");
      } else {
        throw std::runtime_error("System : flux 'hllc' exige un modele Euler complet "
                                 "(4 variables + pression) ; ce modele -> 'rusanov'");
      }
    }
    throw std::runtime_error("System : flux inconnu '" + flx + "' (rusanov|hllc)");
  }

  template <class Model>
  std::function<Real(const MultiFab&)> make_max_speed(const Model& m) {
    Impl* P = this;
    return [P, m](const MultiFab& U) { return max_wave_speed_mf(m, U, P->aux); };
  }

  // Contribution du bloc au second membre de Poisson : rhs += elliptic_rhs(U) sur les
  // cellules valides (boucle hote, comme la derivation de aux). f = Sum_s elliptic_rhs_s.
  template <class Model>
  std::function<void(const MultiFab&, MultiFab&)> make_poisson_rhs(const Model& m) {
    return [m](const MultiFab& U, MultiFab& rhs) {
      for (int li = 0; li < rhs.local_size(); ++li) {
        Array4 r = rhs.fab(li).array();
        const ConstArray4 u = U.fab(li).const_array();
        const Box2D b = rhs.box(li);
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            r(i, j) += m.elliptic_rhs(adc::load_state<Model>(u, i, j));
      }
    };
  }

  void solve_fields() {
    ensure_elliptic();
    MultiFab& rhs = ell_rhs();
    rhs.set_val(Real(0));
    for (auto& s : sp) s.add_poisson_rhs(s.U, rhs);  // f = Sum_s elliptic_rhs_s(u_s)
    ell_solve();
    device_fence();
    const Real dx = geom.dx(), dy = geom.dy();
    const ConstArray4 p = ell_phi().fab(0).const_array();
    Array4 a = aux.fab(0).array();
    const Box2D v = aux.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        a(i, j, 0) = p(i, j);
        a(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
        a(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
      }
    if (periodic_)
      fill_boundary(aux, dom, per_);
    else
      fill_ghosts(aux, dom, bc_aux_);  // extrapolation au bord (paroi / sortie libre)
  }

  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
    return out;
  }
  // Etat complet aplati en ordre composante-majeur : pour c, pour j, pour i.
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    device_fence();
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(ncomp) * v.nx() * v.ny());
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, c));
    return out;
  }
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    const Box2D v = mf.box(0);
    const std::size_t need = static_cast<std::size_t>(ncomp) * v.nx() * v.ny();
    if (in.size() != need)
      throw std::runtime_error("System::set_state : taille != ncomp*n*n");
    Array4 u = mf.fab(0).array();
    std::size_t k = 0;
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) u(i, j, c) = in[k++];
  }
};

System::System(const SystemConfig& c) : p_(std::make_unique<Impl>(c)) {}
System::~System() = default;
System::System(System&&) noexcept = default;
System& System::operator=(System&&) noexcept = default;

void System::add_block(const std::string& name, const std::string& model, double charge,
                       const std::string& limiter, const std::string& flux,
                       const std::string& time, int substeps) {
  using Kind = Impl::Kind;
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("System::add_block : substeps >= 1");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("System::add_block : time 'explicit' | 'imex' (recu '" + time +
                             "')");
  const bool imex = (time == "imex");

  int ncomp = 1;
  Kind kind = Kind::Diocotron;
  Impl::BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  if (model == "diocotron") {
    ncomp = 1; kind = Kind::Diocotron;
    const Diocotron m{Real(P->cfg.B0), Real(P->cfg.n_i0), Real(P->cfg.alpha)};
    clo = P->make_block(m, limiter, flux, imex, substeps);
    max_speed = P->make_max_speed(m);
    add_poisson_rhs = P->make_poisson_rhs(m);
  } else if (model == "electron_euler") {
    ncomp = 4; kind = Kind::Euler;
    const ChargedEuler m{Euler{Real(P->cfg.gamma)}, Real(charge), Real(charge)};
    clo = P->make_block(m, limiter, flux, imex, substeps);
    max_speed = P->make_max_speed(m);
    add_poisson_rhs = P->make_poisson_rhs(m);
  } else if (model == "ion_isothermal") {
    ncomp = 3; kind = Kind::Isothermal;
    const ChargedEulerIsothermal m{Real(P->cfg.cs2), Real(charge), Real(charge)};
    clo = P->make_block(m, limiter, flux, imex, substeps);
    max_speed = P->make_max_speed(m);
    add_poisson_rhs = P->make_poisson_rhs(m);
  } else if (model == "euler_poisson") {
    ncomp = 4; kind = Kind::Euler;  // 4 var, set_density comme Euler ; charge = signe couplage
    EulerPoisson m;
    m.hydro.gamma = Real(P->cfg.gamma);
    m.four_pi_G = Real(P->cfg.four_pi_G);
    m.rho0 = Real(P->cfg.rho0);
    m.coupling_sign = Real(charge);  // +1 auto-gravite, -1 electrostatique (Langmuir)
    clo = P->make_block(m, limiter, flux, imex, substeps);
    max_speed = P->make_max_speed(m);
    add_poisson_rhs = P->make_poisson_rhs(m);
  } else {
    throw std::runtime_error("System::add_block : modele inconnu '" + model +
                             "' (diocotron|electron_euler|ion_isothermal|euler_poisson)");
  }

  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), charge, kind, ncomp,
                                std::move(clo.advance), std::move(clo.rhs_into),
                                std::move(max_speed), std::move(add_poisson_rhs)});
  P->sp.back().U.set_val(Real(0));
}

void System::add_species(const std::string& name, const std::string& model, double charge) {
  add_block(name, model, charge);  // minmod + rusanov + explicite + 1 sous-pas
}

void System::set_poisson(const std::string& rhs, const std::string& solver,
                         const std::string& bc, const std::string& wall,
                         double wall_radius) {
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
  p_->ell_.reset();  // reconstruit au prochain solve_fields
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const int n = p_->cfg.n;
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("System::set_density : taille != n*n");
  const Real gm1 = Real(p_->cfg.gamma) - Real(1);
  Array4 u = s.U.fab(0).array();
  const Box2D v = s.U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const Real r = rho[static_cast<std::size_t>(j) * n + i];
      u(i, j, 0) = r;
      if (s.kind == Impl::Kind::Euler) {
        u(i, j, 1) = 0; u(i, j, 2) = 0;
        u(i, j, 3) = r / gm1;  // E = p/(g-1), p = rho, au repos
      } else if (s.kind == Impl::Kind::Isothermal) {
        u(i, j, 1) = 0; u(i, j, 2) = 0;
      }
    }
}

void System::solve_fields() { p_->solve_fields(); }

void System::step(double dt) {
  p_->solve_fields();
  for (auto& s : p_->sp) s.advance(s.U, Real(dt));
  p_->t += dt;
}
void System::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}
double System::step_cfl(double cfl) {
  p_->solve_fields();
  Real wmax = Real(1e-30);
  for (auto& s : p_->sp) {
    const Real w = s.max_speed(s.U);
    if (w > wmax) wmax = w;
  }
  const Real h = std::min(p_->geom.dx(), p_->geom.dy());
  const double dt = cfl * static_cast<double>(h) / static_cast<double>(wmax);
  for (auto& s : p_->sp) s.advance(s.U, Real(dt));  // aux deja resolu ci-dessus
  p_->t += dt;
  return dt;
}

std::vector<double> System::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  s.rhs_into(s.U, R);  // l'appelant a la charge de solve_fields() au prealable
  return p_->copy_state(R, s.ncomp);
}
std::vector<double> System::get_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  return p_->copy_state(s.U, s.ncomp);
}
void System::set_state(const std::string& name, const std::vector<double>& u) {
  Impl::Species& s = p_->find(name);
  p_->write_state(s.U, s.ncomp, u);
}
int System::n_vars(const std::string& name) const { return p_->find(name).ncomp; }

int System::nx() const { return p_->cfg.n; }
double System::time() const { return p_->t; }
int System::n_species() const { return static_cast<int>(p_->sp.size()); }
double System::mass(const std::string& name) const { return sum(p_->find(name).U, 0); }
std::vector<double> System::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System::potential() {
  p_->ensure_elliptic();
  device_fence();
  const ConstArray4 ph = p_->ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(ph(i, j));
  return out;
}

}  // namespace adc

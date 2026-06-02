#include <adc/runtime/system.hpp>

#include <adc/runtime/model_factory.hpp>  // detail::dispatch_model + briques compilees
#include <adc/elliptic/geometric_mg.hpp>
#include <adc/elliptic/poisson_fft_solver.hpp>
#include <adc/integrator/implicit_stepper.hpp>   // backward_euler_source
#include <adc/integrator/time_steppers.hpp>      // ForwardEuler, SSPRK2Step (math RK du coeur)
#include <adc/operator/spatial_operator.hpp>     // assemble_rhs, SourceFreeModel, max_wave_speed_mf, load_state

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>  // sum
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
  // Fermetures compilees figees a l'ajout du bloc (modele compose + schema spatial + temps).
  // Type-erased SEULEMENT au niveau de la liste de blocs ; le noyau reste compile.
  struct Species {
    std::string name;
    MultiFab U;
    int ncomp;
    int substeps;                                             // sous-pas statiques (add_block)
    bool evolve;                                              // false = espece gelee (fond fixe non avance)
    double gamma;                                             // pour l'energie au repos (4 var)
    std::function<void(MultiFab&, Real, int)> advance;        // (U, dt, n) : n sous-pas de dt/n
    std::function<void(MultiFab&, MultiFab&)> rhs_into;        // R <- -div F + S (Poisson fige)
    std::function<Real(const MultiFab&)> max_speed;           // max |vitesse d'onde| du bloc
    std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;  // += elliptic_rhs(U)
  };

  SystemConfig cfg;
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  BCRec bc_;        // CL transport (periodique ou Foextrap selon cfg.periodic)
  Box2D dom;
  Periodicity per_;
  bool periodic_;
  MultiFab aux;
  std::vector<Species> sp;
  double t = 0;
  std::vector<std::function<void(Real)>> couplings;  // sources couplees inter-especes (splitting)

  // Configuration Poisson (solveur elliptique construit paresseusement).
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
  int index(const std::string& name) const {
    for (std::size_t k = 0; k < sp.size(); ++k)
      if (sp[k].name == name) return static_cast<int>(k);
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }

  // Sources de COUPLAGE inter-especes : appliquees par SPLITTING (un pas additif explicite de
  // dt) APRES le transport de chaque bloc. Passe HOTE (comme set_density) : a revisiter pour
  // Kokkos/GPU. Chaque couplage lit/met a jour l'etat de PLUSIEURS blocs au meme point.
  void apply_couplings(Real dt) {
    if (couplings.empty()) return;
    device_fence();
    for (auto& c : couplings) c(dt);
  }

  // --- solveur elliptique (Poisson de systeme) -----------------------------
  BCRec poisson_bc() {
    std::string mode = p_bc;
    if (mode == "auto") mode = (p_wall == "circle" || !cfg.periodic) ? "dirichlet" : "periodic";
    BCRec b;
    if (mode == "periodic") return b;
    if (mode == "dirichlet") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;
      return b;
    }
    if (mode == "neumann") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
      return b;
    }
    throw std::runtime_error("System::set_poisson : bc inconnu '" + mode + "'");
  }
  std::function<bool(Real, Real)> wall_active() {
    if (p_wall == "none") return {};
    if (p_wall == "circle") {
      const double cx = 0.5 * cfg.L, cy = 0.5 * cfg.L, R = p_wall_radius;
      return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
    }
    throw std::runtime_error("System::set_poisson : wall inconnu '" + p_wall + "'");
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
        throw std::runtime_error("System : solver 'fft' incompatible avec une paroi -> 'geometric_mg'");
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
  // Evaluateur methode-des-lignes d'un bloc (L/F/Model figes) : ghosts puis R = -div F + S.
  // La math RK est portee par les TimeStepper du coeur, pas reimplementee.
  template <class Limiter, class Flux, class Model>
  auto rhs_eval(const Model& model, bool recon_prim) {
    return [this, &model, recon_prim](MultiFab& U, MultiFab& R) {
      fill_ghosts(U, dom, bc_);
      assemble_rhs<Limiter, Flux>(model, U, aux, geom, R, recon_prim);
    };
  }
  template <class Limiter, class Flux, class Model>
  void ssprk2(const Model& model, MultiFab& U, Real dt, bool recon_prim) {
    SSPRK2Step{}.take_step(rhs_eval<Limiter, Flux>(model, recon_prim), U, dt);
  }
  template <class Limiter, class Flux, class Model>
  void imex_step(const Model& model, MultiFab& U, Real dt, bool recon_prim) {
    const SourceFreeModel<Model> sf{model};  // demi-pas explicite : SourceFreeModel sans Prim
    ForwardEuler{}.take_step(rhs_eval<Limiter, Flux>(sf, recon_prim), U, dt);  // -> conservatif
    backward_euler_source(model, aux, U, dt);
  }

  struct BlockClosures {
    std::function<void(MultiFab&, Real, int)> advance;
    std::function<void(MultiFab&, MultiFab&)> rhs_into;
  };

  template <class Limiter, class Flux, class Model>
  BlockClosures build(const Model& m, bool imex, bool recon_prim) {
    Impl* P = this;
    BlockClosures bc;
    if (imex)
      bc.advance = [P, m, recon_prim](MultiFab& U, Real dt, int n) {
        const Real h = dt / static_cast<Real>(n);
        for (int s = 0; s < n; ++s) P->imex_step<Limiter, Flux>(m, U, h, recon_prim);
      };
    else
      bc.advance = [P, m, recon_prim](MultiFab& U, Real dt, int n) {
        const Real h = dt / static_cast<Real>(n);
        for (int s = 0; s < n; ++s) P->ssprk2<Limiter, Flux>(m, U, h, recon_prim);
      };
    bc.rhs_into = [P, m, recon_prim](MultiFab& U, MultiFab& R) {
      fill_ghosts(U, P->dom, P->bc_);
      assemble_rhs<Limiter, Flux>(m, U, P->aux, P->geom, R, recon_prim);
    };
    return bc;
  }

  // Dispatch du schema spatial (limiteur x flux Riemann) -> fermetures compilees. HLLC garde
  // par requires : exige un transport a 4 variables exposant pressure (sinon -> rusanov).
  template <class Model>
  BlockClosures make_block(const Model& m, const std::string& lim, const std::string& riem,
                           bool imex, bool recon_prim) {
    if (riem == "rusanov") {
      if (lim == "none") return build<NoSlope, RusanovFlux>(m, imex, recon_prim);
      if (lim == "minmod") return build<Minmod, RusanovFlux>(m, imex, recon_prim);
      if (lim == "vanleer") return build<VanLeer, RusanovFlux>(m, imex, recon_prim);
      throw std::runtime_error("System : limiter inconnu '" + lim + "'");
    }
    if (riem == "hllc") {
      if constexpr (Model::n_vars == 4 &&
                    requires(const Model mm, typename Model::State s) { mm.pressure(s); }) {
        if (lim == "none") return build<NoSlope, HLLCFlux>(m, imex, recon_prim);
        if (lim == "minmod") return build<Minmod, HLLCFlux>(m, imex, recon_prim);
        if (lim == "vanleer") return build<VanLeer, HLLCFlux>(m, imex, recon_prim);
        throw std::runtime_error("System : limiter inconnu '" + lim + "'");
      } else {
        throw std::runtime_error("System : flux 'hllc' exige un transport compressible "
                                 "(4 variables + pression) ; ce transport -> 'rusanov'");
      }
    }
    throw std::runtime_error("System : flux Riemann inconnu '" + riem + "' (rusanov|hllc)");
  }

  template <class Model>
  std::function<Real(const MultiFab&)> make_max_speed(const Model& m) {
    Impl* P = this;
    return [P, m](const MultiFab& U) { return max_wave_speed_mf(m, U, P->aux); };
  }

  // Contribution du bloc au second membre de Poisson : rhs += elliptic_rhs(U) (boucle hote).
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
      fill_ghosts(aux, dom, bc_);  // extrapolation au bord (paroi / sortie libre)
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

void System::add_block(const std::string& name, const ModelSpec& model,
                       const std::string& limiter, const std::string& riemann,
                       const std::string& recon, const std::string& time, int substeps,
                       bool evolve) {
  Impl* P = p_.get();
  if (substeps < 1) throw std::runtime_error("System::add_block : substeps >= 1");
  if (time != "explicit" && time != "imex")
    throw std::runtime_error("System::add_block : time 'explicit' | 'imex' (recu '" + time +
                             "')");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_block : recon 'conservative' | 'primitive' (recu '" +
                             recon + "')");
  const bool imex = (time == "imex");
  const bool recon_prim = (recon == "primitive");

  int ncomp = 1;
  Impl::BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  // Le modele est compose a partir des briques designees par la spec ; le visiteur cable
  // les fermetures. ncomp = n_vars du modele compose ; set_density s'y adapte.
  detail::dispatch_model(model, [&](auto m) {
    using M = decltype(m);
    ncomp = M::n_vars;
    clo = P->make_block(m, limiter, riemann, imex, recon_prim);
    max_speed = P->make_max_speed(m);
    add_poisson_rhs = P->make_poisson_rhs(m);
  });

  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), ncomp, substeps,
                                evolve, model.gamma, std::move(clo.advance), std::move(clo.rhs_into),
                                std::move(max_speed), std::move(add_poisson_rhs)});
  P->sp.back().U.set_val(Real(0));
}

void System::set_poisson(const std::string& rhs, const std::string& solver,
                         const std::string& bc, const std::string& wall, double wall_radius) {
  p_->p_rhs = rhs;
  p_->p_solver = solver;
  p_->p_bc = bc;
  p_->p_wall = wall;
  p_->p_wall_radius = wall_radius;
  p_->ell_.reset();
}

void System::add_ionization(const std::string& electron, const std::string& ion,
                            const std::string& neutral, double rate) {
  Impl* P = p_.get();
  const int ie = P->index(electron), ii = P->index(ion), ig = P->index(neutral);
  const Real k = static_cast<Real>(rate);
  // Ionisation (operator-split, sur la densite = comp 0) : taux r = k n_e n_g. Un neutre
  // disparait, un ion et un electron apparaissent : n_g -= dt r, n_i += dt r, n_e += dt r. La
  // masse est transferee du neutre vers l'ion (n_i + n_g conserve). Premiere brique de couplage ;
  // le transfert de quantite de mouvement / energie (especes fluides) est un raffinement ulterieur.
  P->couplings.push_back([P, ie, ii, ig, k](Real dt) {
    Impl::Species& e = P->sp[ie];
    Impl::Species& s_i = P->sp[ii];
    Impl::Species& g = P->sp[ig];
    Array4 ue = e.U.fab(0).array();
    Array4 ui = s_i.U.fab(0).array();
    Array4 ug = g.U.fab(0).array();
    const Box2D v = e.U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int x = v.lo[0]; x <= v.hi[0]; ++x) {
        const Real dn = dt * k * ue(x, j, 0) * ug(x, j, 0);
        ug(x, j, 0) -= dn;
        ui(x, j, 0) += dn;
        ue(x, j, 0) += dn;
      }
  });
}

void System::add_collision(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp < 3 || P->sp[ib].ncomp < 3)
    throw std::runtime_error("System::add_collision : les deux blocs doivent porter une quantite "
                             "de mouvement (transport fluide >= 3 variables)");
  const Real k = static_cast<Real>(rate);
  // Friction inter-especes (operator-split) : force F = k (u_a - u_b) sur la quantite de
  // mouvement, opposee sur chaque espece (qte de mvt totale conservee) ; les vitesses relaxent
  // l'une vers l'autre. Sur comp 1 et 2 (qte de mvt). L'echauffement par friction (energie) est
  // un raffinement ulterieur (neglige : convient aux especes isothermes, sans eq. d'energie).
  P->couplings.push_back([P, ia, ib, k](Real dt) {
    Impl::Species& A = P->sp[ia];
    Impl::Species& B = P->sp[ib];
    Array4 ua = A.U.fab(0).array();
    Array4 ub = B.U.fab(0).array();
    const Box2D v = A.U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int x = v.lo[0]; x <= v.hi[0]; ++x)
        for (int c = 1; c <= 2; ++c) {  // composantes de quantite de mouvement (x, y)
          const Real va = ua(x, j, c) / ua(x, j, 0);
          const Real vb = ub(x, j, c) / ub(x, j, 0);
          const Real f = dt * k * (va - vb);
          ua(x, j, c) -= f;
          ub(x, j, c) += f;
        }
  });
}

void System::add_thermal_exchange(const std::string& a, const std::string& b, double rate) {
  Impl* P = p_.get();
  const int ia = P->index(a), ib = P->index(b);
  if (P->sp[ia].ncomp != 4 || P->sp[ib].ncomp != 4)
    throw std::runtime_error("System::add_thermal_exchange : les deux blocs doivent porter une "
                             "energie (Euler compressible, 4 variables)");
  const Real k = static_cast<Real>(rate);
  const Real ga = static_cast<Real>(P->sp[ia].gamma), gb = static_cast<Real>(P->sp[ib].gamma);
  // Echange thermique (operator-split) : flux de chaleur q = k (T_a - T_b) sur l'energie, oppose
  // sur chaque espece (energie totale conservee) ; les temperatures relaxent. T = p/rho (a une
  // constante pres), p = (gamma-1)(E - 1/2 rho |u|^2). Transfere l'energie INTERNE (u inchange).
  P->couplings.push_back([P, ia, ib, k, ga, gb](Real dt) {
    Impl::Species& A = P->sp[ia];
    Impl::Species& B = P->sp[ib];
    Array4 ua = A.U.fab(0).array();
    Array4 ub = B.U.fab(0).array();
    const Box2D v = A.U.box(0);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int x = v.lo[0]; x <= v.hi[0]; ++x) {
        const Real ra = ua(x, j, 0), rb = ub(x, j, 0);
        const Real pa = (ga - Real(1)) * (ua(x, j, 3) -
            Real(0.5) * (ua(x, j, 1) * ua(x, j, 1) + ua(x, j, 2) * ua(x, j, 2)) / ra);
        const Real pb = (gb - Real(1)) * (ub(x, j, 3) -
            Real(0.5) * (ub(x, j, 1) * ub(x, j, 1) + ub(x, j, 2) * ub(x, j, 2)) / rb);
        const Real q = dt * k * (pa / ra - pb / rb);  // k (T_a - T_b), T = p/rho
        ua(x, j, 3) -= q;
        ub(x, j, 3) += q;
      }
  });
}

void System::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const int n = p_->cfg.n;
  if (static_cast<int>(rho.size()) != n * n)
    throw std::runtime_error("System::set_density : taille != n*n");
  const Real gm1 = Real(s.gamma) - Real(1);
  Array4 u = s.U.fab(0).array();
  const Box2D v = s.U.box(0);
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
      const Real r = rho[static_cast<std::size_t>(j) * n + i];
      u(i, j, 0) = r;
      if (s.ncomp >= 3) { u(i, j, 1) = 0; u(i, j, 2) = 0; }  // qte de mvt au repos
      if (s.ncomp == 4) u(i, j, 3) = r / gm1;                // E = p/(g-1), p = rho
    }
}

void System::solve_fields() { p_->solve_fields(); }

void System::step(double dt) {
  p_->solve_fields();
  for (auto& s : p_->sp)
    if (s.evolve) s.advance(s.U, Real(dt), s.substeps);  // bloc gele : non avance
  p_->apply_couplings(Real(dt));  // sources couplees inter-especes (splitting), apres transport
  p_->t += dt;
}
void System::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) step(dt);
}
double System::step_cfl(double cfl) {
  p_->solve_fields();
  Real wmax = Real(1e-30);
  for (auto& s : p_->sp)
    if (s.evolve) wmax = std::max(wmax, s.max_speed(s.U));  // bloc gele : ne contraint pas le pas
  const Real h = std::min(p_->geom.dx(), p_->geom.dy());
  const double dt = cfl * static_cast<double>(h) / static_cast<double>(wmax);
  for (auto& s : p_->sp)
    if (s.evolve) s.advance(s.U, Real(dt), s.substeps);
  p_->apply_couplings(Real(dt));
  p_->t += dt;
  return dt;
}
double System::step_adaptive(double cfl) {
  p_->solve_fields();
  // Multirate : macro-pas = pas stable du bloc le plus LENT ; chaque bloc plus rapide est
  // sous-cycle n_b = ceil(w_b / w_min). aux fige sur le macro-pas (couplage once-per-step).
  Real wmin = Real(1e30);
  std::vector<Real> wb;
  wb.reserve(p_->sp.size());
  for (auto& s : p_->sp) {
    const Real w = s.evolve ? s.max_speed(s.U) : Real(0);  // bloc gele : hors cadence
    wb.push_back(w);
    if (s.evolve) wmin = std::min(wmin, w);
  }
  if (wmin >= Real(1e30)) wmin = Real(1e-30);  // aucun bloc evolutif (tous geles)
  const Real h = std::min(p_->geom.dx(), p_->geom.dy());
  const double macro_dt = cfl * static_cast<double>(h) / static_cast<double>(wmin);
  for (std::size_t b = 0; b < p_->sp.size(); ++b) {
    if (!p_->sp[b].evolve) continue;  // bloc gele : non avance
    int n = static_cast<int>(std::ceil(static_cast<double>(wb[b] / wmin)));
    if (n < 1) n = 1;
    p_->sp[b].advance(p_->sp[b].U, Real(macro_dt), n);
  }
  p_->apply_couplings(Real(macro_dt));
  p_->t += macro_dt;
  return macro_dt;
}

std::vector<double> System::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  s.rhs_into(s.U, R);
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

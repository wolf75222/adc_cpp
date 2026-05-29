#include <adc/solver/euler_poisson_solver.hpp>

#include <adc/coupling/coupler.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/euler_poisson.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>

namespace adc {

static EulerPoisson make_model(const EulerPoissonConfig& c) {
  EulerPoisson m;
  m.hydro.gamma = c.gamma;
  m.four_pi_G = c.four_pi_G;
  m.rho0 = c.rho0;
  return m;
}

struct EulerPoissonSolver::Impl {
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  EulerPoisson model;
  BCRec bcU, bcPhi;
  Coupler<EulerPoisson> cpl;
  MultiFab U;
  double t = 0;
  int n;

  explicit Impl(const EulerPoissonConfig& c)
      : geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        model(make_model(c)),
        cpl(model, geom, ba, bcU, bcPhi),
        U(ba, dm, 4, 2),
        n(c.n) {
    // CI : perturbation acoustique-gravitationnelle au repos (Jeans), au repos.
    constexpr double pi = 3.14159265358979323846;
    const double k = 2 * pi / c.L, cs2 = c.gamma * c.p0 / c.rho0;
    Array4 u = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    auto wrap = [&](int x) { return (x % c.n + c.n) % c.n; };
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const double x = (wrap(i) + 0.5) / c.n * c.L;
        const double drho = c.eps * c.rho0 * std::cos(k * x);
        const double rho = c.rho0 + drho, p = c.p0 + cs2 * drho;
        u(i, j, 0) = rho;
        u(i, j, 1) = 0.0;
        u(i, j, 2) = 0.0;
        u(i, j, 3) = p / (c.gamma - 1);
      }
  }
};

EulerPoissonSolver::EulerPoissonSolver(const EulerPoissonConfig& c)
    : p_(std::make_unique<Impl>(c)) {}
EulerPoissonSolver::~EulerPoissonSolver() = default;
EulerPoissonSolver::EulerPoissonSolver(EulerPoissonSolver&&) noexcept = default;
EulerPoissonSolver& EulerPoissonSolver::operator=(EulerPoissonSolver&&) noexcept =
    default;

void EulerPoissonSolver::step(double dt) {
  p_->cpl.advance<Minmod>(p_->U, dt);
  p_->t += dt;
}
double EulerPoissonSolver::mass() const { return sum(p_->U, 0); }
double EulerPoissonSolver::energy() const { return sum(p_->U, 3); }
double EulerPoissonSolver::total_momentum(int dir) const {
  return sum(p_->U, dir == 0 ? 1 : 2);
}
double EulerPoissonSolver::time() const { return p_->t; }
int EulerPoissonSolver::nx() const { return p_->n; }

std::vector<double> EulerPoissonSolver::density() const {
  const ConstArray4 u = p_->U.fab(0).const_array();
  const Box2D v = p_->U.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.hi[0] - v.lo[0] + 1) *
              (v.hi[1] - v.lo[1] + 1));
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
  return out;
}

}  // namespace adc

#include <adc/solver/diocotron_solver.hpp>

#include <adc/coupling/coupler.hpp>
#include <adc/coupling/coupling_policy.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>

namespace adc {

struct DiocotronSolver::Impl {
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  Diocotron model;
  BCRec bcU, bcPhi;  // periodique (defaut)
  Coupler<Diocotron> cpl;
  MultiFab U;
  double t = 0;
  int n;
  bool per_stage;

  explicit Impl(const DiocotronConfig& c)
      : geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        model{c.B0, c.n_i0, c.alpha},
        cpl(model, geom, ba, bcU, bcPhi),
        U(ba, dm, 1, 2),
        n(c.n),
        per_stage(c.poisson_per_stage) {
    // CI : densite lisse de moyenne n_i0 (second membre de Poisson a moyenne nulle)
    constexpr double pi = 3.14159265358979323846;
    Array4 u = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    auto wrap = [&](int x) { return (x % c.n + c.n) % c.n; };
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const double x = (wrap(i) + 0.5) / c.n, y = (wrap(j) + 0.5) / c.n;
        u(i, j, 0) =
            c.n_i0 * (1 + c.eps * std::sin(2 * pi * x) * std::sin(2 * pi * y));
      }
  }
};

DiocotronSolver::DiocotronSolver(const DiocotronConfig& c)
    : p_(std::make_unique<Impl>(c)) {}
DiocotronSolver::~DiocotronSolver() = default;
DiocotronSolver::DiocotronSolver(DiocotronSolver&&) noexcept = default;
DiocotronSolver& DiocotronSolver::operator=(DiocotronSolver&&) noexcept = default;

void DiocotronSolver::step(double dt) {
  if (p_->per_stage)
    p_->cpl.advance<Minmod, PerStageCoupling>(p_->U, dt);
  else
    p_->cpl.advance<Minmod, OncePerStepCoupling>(p_->U, dt);
  p_->t += dt;
}
double DiocotronSolver::mass() const { return sum(p_->U, 0); }
double DiocotronSolver::time() const { return p_->t; }
int DiocotronSolver::nx() const { return p_->n; }

std::vector<double> DiocotronSolver::density() const {
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

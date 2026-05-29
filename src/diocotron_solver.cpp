#include <adc/solver/diocotron_solver.hpp>

#include <adc/coupling/coupler.hpp>
#include <adc/coupling/coupling_policy.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/operator/reconstruction.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <functional>

namespace adc {

static constexpr double kPi = 3.14159265358979323846;

struct DiocotronSolver::Impl {
  Geometry geom;
  BoxArray ba;
  DistributionMapping dm;
  Diocotron model;
  BCRec bcU, bcPhi;
  std::function<bool(Real, Real)> active;
  Coupler<Diocotron> cpl;
  MultiFab U;
  double t = 0;
  int n;
  bool per_stage;
  double B0;

  // --- helpers statiques (appeles dans la liste d'init) ---

  // n_i0 = fond ionique neutralisant : pour Band il vaut la MOYENNE de la CI (RHS
  // de Poisson a moyenne nulle) ; pour Ring c'est une colonne d'electrons pure (0).
  static double init_ni0(const DiocotronConfig& c) {
    if (c.ic == DiocotronIC::Ring) return 0.0;
    if (c.ic != DiocotronIC::Band) return c.n_i0;  // Smooth : sin*sin de moyenne n_i0
    double mean = 0;
    const double dx = c.L / c.n;
    for (int j = 0; j < c.n; ++j)
      for (int i = 0; i < c.n; ++i) {
        const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
        const double y0 = 0.5 * c.L + c.band_disp * std::cos(2 * kPi * c.band_mode * x / c.L);
        mean += 1.0 + c.band_amp * std::exp(-((y - y0) * (y - y0)) / (c.band_width * c.band_width));
      }
    return mean / (double(c.n) * c.n);
  }
  static BCRec make_bcU(const DiocotronConfig& c) {
    BCRec b;  // periodique par defaut (Smooth/Band)
    if (c.ic == DiocotronIC::Ring)
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;  // sortie libre du fluide
    return b;
  }
  static BCRec make_bcPhi(const DiocotronConfig& c) {
    BCRec b;
    if (c.ic == DiocotronIC::Ring)
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;  // paroi conductrice phi=0
    return b;
  }
  static std::function<bool(Real, Real)> make_active(const DiocotronConfig& c) {
    if (c.ic == DiocotronIC::Ring && c.wall_radius > 0) {
      const double cx = 0.5 * c.L, cy = 0.5 * c.L, R = c.wall_radius;
      return [cx, cy, R](Real x, Real y) { return std::hypot(x - cx, y - cy) < R; };
    }
    return {};
  }

  explicit Impl(const DiocotronConfig& c)
      : geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        dm(1, n_ranks()),
        model{c.B0, init_ni0(c), c.alpha},
        bcU(make_bcU(c)), bcPhi(make_bcPhi(c)), active(make_active(c)),
        cpl(model, geom, ba, bcU, bcPhi, active),
        U(ba, dm, 1, 2),
        n(c.n), per_stage(c.poisson_per_stage), B0(c.B0) {
    fill_ic(c);
    cpl.solve_fields(U);  // phi + aux valides pour le premier max_drift_speed
  }

  // CI ecrite en boucle HOTE (operation unique, memoire unifiee coherente avec les
  // kernels device qui suivent) : pas de for_each_cell ici, pour que la facade
  // compile telle quelle sous nvcc (les lambdas device exigeraient ADC_HD).
  void fill_ic(const DiocotronConfig& c) {
    Array4 u = U.fab(0).array();
    const Box2D d = geom.domain;
    const double cx = 0.5 * c.L, cy = 0.5 * c.L;
    for (int j = d.lo[1]; j <= d.hi[1]; ++j)
      for (int i = d.lo[0]; i <= d.hi[0]; ++i) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        double ne;
        if (c.ic == DiocotronIC::Band) {
          const double y0 = cy + c.band_disp * std::cos(2 * kPi * c.band_mode * x / c.L);
          ne = 1.0 + c.band_amp * std::exp(-((y - y0) * (y - y0)) / (c.band_width * c.band_width));
        } else if (c.ic == DiocotronIC::Ring) {
          const double r = std::hypot(x - cx, y - cy), th = std::atan2(y - cy, x - cx);
          ne = c.ring_floor;
          if (r > c.ring_r0 && r < c.ring_r1)
            ne = 1.0 - c.ring_delta + c.ring_delta * std::sin(c.ring_mode * th);
        } else {  // Smooth : sin*sin de moyenne n_i0
          ne = c.n_i0 * (1 + c.eps * std::sin(2 * kPi * x / c.L) * std::sin(2 * kPi * y / c.L));
        }
        u(i, j, 0) = ne;
      }
  }

  void step(double dt) {
    if (per_stage)
      cpl.advance<Minmod, PerStageCoupling>(U, dt);
    else
      cpl.advance<Minmod, OncePerStepCoupling>(U, dt);
    t += dt;
  }

  double max_drift_speed() const {
    device_fence();  // GPU : barriere avant lecture hote (memoire unifiee)
    const ConstArray4 ax = cpl.aux().fab(0).const_array();
    const Box2D v = U.box(0);
    double vmax = 1e-12;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
        const double gx = ax(i, j, 1), gy = ax(i, j, 2);
        const double s = std::sqrt(gx * gx + gy * gy) / B0;
        if (s > vmax) vmax = s;
      }
    return vmax;
  }

  std::vector<double> copy_field(const MultiFab& mf) const {
    device_fence();
    const ConstArray4 f = mf.fab(0).const_array();
    const Box2D v = U.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(f(i, j, 0));
    return out;
  }
};

DiocotronSolver::DiocotronSolver(const DiocotronConfig& c)
    : p_(std::make_unique<Impl>(c)) {}
DiocotronSolver::~DiocotronSolver() = default;
DiocotronSolver::DiocotronSolver(DiocotronSolver&&) noexcept = default;
DiocotronSolver& DiocotronSolver::operator=(DiocotronSolver&&) noexcept = default;

void DiocotronSolver::step(double dt) { p_->step(dt); }
void DiocotronSolver::step_cfl(double cfl) {
  p_->step(cfl * p_->geom.dx() / p_->max_drift_speed());
}
double DiocotronSolver::max_drift_speed() const { return p_->max_drift_speed(); }
double DiocotronSolver::dx() const { return p_->geom.dx(); }
double DiocotronSolver::mass() const { return sum(p_->U, 0); }
double DiocotronSolver::time() const { return p_->t; }
int DiocotronSolver::nx() const { return p_->n; }
std::vector<double> DiocotronSolver::density() const { return p_->copy_field(p_->U); }
std::vector<double> DiocotronSolver::potential() const {
  return p_->copy_field(p_->cpl.phi());
}

}  // namespace adc

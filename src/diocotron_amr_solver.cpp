#include <adc/solver/diocotron_amr_solver.hpp>

#include <adc/coupling/amr_coupler_mp.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <vector>

namespace adc {

static constexpr double kPi = 3.14159265358979323846;

struct DiocotronAmrSolver::Impl {
  DiocotronAmrConfig cfg;
  Geometry geom;
  BoxArray ba_coarse;
  Diocotron model;
  BCRec bc;  // periodique
  AmrCouplerMP<Diocotron> cpl;
  double t = 0;

  static double init_ni0(const DiocotronAmrConfig& c) {
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

  // densite de la bande au point physique (x, y).
  static double band_ne(const DiocotronAmrConfig& c, double x, double y) {
    const double y0 = 0.5 * c.L + c.band_disp * std::cos(2 * kPi * c.band_mode * x / c.L);
    return 1.0 + c.band_amp * std::exp(-((y - y0) * (y - y0)) / (c.band_width * c.band_width));
  }

  static void fill(MultiFab& U, const Geometry& g, const DiocotronAmrConfig& c, double dx) {
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D b = U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          u(i, j, 0) = band_ne(c, (i + 0.5) * dx, (j + 0.5) * dx);
    }
  }

  // hierarchie initiale : grossier + UN patch fin seed (bande centrale), remplace
  // par regrid des le premier appel.
  static std::vector<AmrLevelMP> init_levels(const DiocotronAmrConfig& c,
                                             const Geometry& geom) {
    const double dxc = c.L / c.n, dxf = dxc / 2;
    DistributionMapping dm(1, n_ranks());
    BoxArray bac(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)});
    MultiFab Uc(bac, dm, 1, 1);
    fill(Uc, geom, c, dxc);
    // seed fin : bande grossiere [2..n-3] x [n/2 - n/8 .. n/2 + n/8 - 1], raffinee x2.
    const int I0 = 2, I1 = c.n - 3;
    const int J0 = c.n / 2 - c.n / 8, J1 = c.n / 2 + c.n / 8 - 1;
    Box2D fb{{2 * I0, 2 * J0}, {2 * I1 + 1, 2 * J1 + 1}};
    BoxArray baf(std::vector<Box2D>{fb});
    MultiFab Uf(baf, dm, 1, 1);
    fill(Uf, geom, c, dxf);
    std::vector<AmrLevelMP> L;
    L.push_back({std::move(Uc), nullptr, dxc, dxc});
    L.push_back({std::move(Uf), nullptr, dxf, dxf});
    return L;
  }

  explicit Impl(const DiocotronAmrConfig& c)
      : cfg(c),
        geom{Box2D::from_extents(c.n, c.n), 0.0, c.L, 0.0, c.L},
        ba_coarse(std::vector<Box2D>{Box2D::from_extents(c.n, c.n)}),
        model{c.B0, init_ni0(c), c.alpha},
        cpl(model, geom, ba_coarse, BCRec{}, init_levels(c, geom)) {
    do_regrid();
    cpl.update();  // champs valides pour le premier max_drift_speed
  }

  // critere : raffiner ou la densite depasse le fond + refine_frac.
  void do_regrid() {
    const double thr = model.n_i0 + cfg.refine_frac;
    cpl.regrid([thr](const ConstArray4& a, int i, int j) { return a(i, j, 0) > thr; });
  }

  void step(double dt) {
    static int s = 0;
    if (cfg.regrid_every > 0 && s > 0 && s % cfg.regrid_every == 0) do_regrid();
    cpl.step(dt);
    ++s;
    t += dt;
  }

  std::vector<double> coarse_density() const {
    device_fence();
    const MultiFab& U = cpl.coarse();
    const ConstArray4 u = U.fab(0).const_array();
    const Box2D v = U.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(u(i, j, 0));
    return out;
  }

  int n_patches() {  // cpl.levels() est non-const ; appele via unique_ptr (pointee mutable)
    auto& L = cpl.levels();
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
  }
};

DiocotronAmrSolver::DiocotronAmrSolver(const DiocotronAmrConfig& c)
    : p_(std::make_unique<Impl>(c)) {}
DiocotronAmrSolver::~DiocotronAmrSolver() = default;
DiocotronAmrSolver::DiocotronAmrSolver(DiocotronAmrSolver&&) noexcept = default;
DiocotronAmrSolver& DiocotronAmrSolver::operator=(DiocotronAmrSolver&&) noexcept = default;

void DiocotronAmrSolver::step(double dt) { p_->step(dt); }
void DiocotronAmrSolver::step_cfl(double cfl) {
  p_->step(cfl * p_->geom.dx() / p_->cpl.max_drift_speed());
}
double DiocotronAmrSolver::max_drift_speed() const { return p_->cpl.max_drift_speed(); }
double DiocotronAmrSolver::dx() const { return p_->geom.dx(); }
double DiocotronAmrSolver::mass() const { return p_->cpl.mass(); }
double DiocotronAmrSolver::time() const { return p_->t; }
int DiocotronAmrSolver::nx() const { return p_->cfg.n; }
int DiocotronAmrSolver::n_patches() const { return p_->n_patches(); }
std::vector<double> DiocotronAmrSolver::density() const { return p_->coarse_density(); }

}  // namespace adc

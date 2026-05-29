#include <adc/solver/two_fluid_ap_solver.hpp>

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/two_fluid_ap.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence

#include <cmath>

namespace adc {

// GeometricMG : elliptique entierement on-device (lisseur GS rb + V-cycle via
// for_each_cell), donc la facade compile telle quelle pour le GPU sous
// -DADC_USE_KOKKOS=ON. Le backend (serie/OpenMP/Kokkos) est herite de la cible adc.
struct TwoFluidAPSolver::Impl {
  TwoFluidAP2D<GeometricMG> d;
  bool stabilize;

  explicit Impl(const TwoFluidAPConfig& c)
      : d(c.n, c.L, c.cse2, c.csi2, c.omega_pe, c.omega_pi), stabilize(c.stabilize) {
    d.upwind_continuity = c.upwind_continuity;
    d.wce = c.omega_ce;
    d.wci = c.omega_ci;
    d.init(c.eps);
  }

  std::vector<double> copy_comp(const MultiFab& mf) const {
    device_fence();  // GPU : barriere avant lecture hote (memoire unifiee)
    const ConstArray4 a = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i) out.push_back(a(i, j, 0));
    return out;
  }

  double max_charge() const {
    device_fence();
    const ConstArray4 fe = d.e.fab(0).const_array(), fi = d.ion.fab(0).const_array();
    const Box2D v = d.e.box(0);
    double m = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m = std::fmax(m, std::fabs(fi(i, j, 0) - fe(i, j, 0)));
    return m;
  }
  double max_dev() const {
    device_fence();
    const ConstArray4 fe = d.e.fab(0).const_array();
    const Box2D v = d.e.box(0);
    double m = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m = std::fmax(m, std::fabs(fe(i, j, 0) - 1.0));
    return m;
  }
};

TwoFluidAPSolver::TwoFluidAPSolver(const TwoFluidAPConfig& c)
    : p_(std::make_unique<Impl>(c)) {}
TwoFluidAPSolver::~TwoFluidAPSolver() = default;
TwoFluidAPSolver::TwoFluidAPSolver(TwoFluidAPSolver&&) noexcept = default;
TwoFluidAPSolver& TwoFluidAPSolver::operator=(TwoFluidAPSolver&&) noexcept = default;

void TwoFluidAPSolver::step(double dt) { p_->d.step(dt, p_->stabilize); }
void TwoFluidAPSolver::advance(double dt, int nsteps) {
  for (int s = 0; s < nsteps; ++s) p_->d.step(dt, p_->stabilize);
}

int TwoFluidAPSolver::nx() const { return p_->d.n; }
double TwoFluidAPSolver::mass_e() const { return sum(p_->d.e, 0); }
double TwoFluidAPSolver::mass_i() const { return sum(p_->d.ion, 0); }
double TwoFluidAPSolver::max_charge() const { return p_->max_charge(); }
double TwoFluidAPSolver::max_dev() const { return p_->max_dev(); }
std::vector<double> TwoFluidAPSolver::density_e() const { return p_->copy_comp(p_->d.e); }
std::vector<double> TwoFluidAPSolver::density_i() const { return p_->copy_comp(p_->d.ion); }

}  // namespace adc

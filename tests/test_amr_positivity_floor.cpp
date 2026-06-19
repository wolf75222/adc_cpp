// Zhang-Shu POSITIVITY FLOOR on the AMR transport path (ADC-259). The single-block uniform System
// floor (reconstruct_pp / zhang_shu_scale, tested in test_positivity_floor.cpp) is here exercised on
// the AMR engine: the floor is threaded facade -> advance_amr -> subcycle_level_mp ->
// compute_face_fluxes, AND a Density-role clamp is applied to the COARSE-FINE fine ghost means
// (fill_cf_ghost_cell, the refined-patch interface that the original Hoffart failure exercised).
//
// Four checks (serial; the np=1/2/4 MPI replay is the Python test, the device path is argued by
// construction -- same ADC_HD kernel, the clamp is host-side and outside every conservation sum):
//  (1) C/F GHOST CLAMP (the highest-risk site): the LIVE multi-box ghost fill mf_fill_fine_ghosts_mb
//      reads a coarse parent whose density dips BELOW the floor in one cell; unclamped the fine ghosts
//      reading it go sub-floor, with the floor every C/F fine-ghost density is >= floor while the
//      momenta stay interpolated (the clamp touches the Density role only).
//  (2) NO-DEFAULT-CHANGE: advance_amr on a SMOOTH positive Euler state is BIT-IDENTICAL with the floor
//      ON vs OFF (the floor never bites a face above it), and stays finite + mass-conserving.
//  (3) EFFECT: on the 1e6-contrast oscillatory top-hat (the profile where weno5 reconstructs a
//      negative face density) advance_amr WITH the floor stays finite and conserves mass, and its
//      trajectory DIFFERS from the unfloored run (the floor was active).
//  (4) REJECT: a model without a Density role (AdvectionDiffusion) + positivity_floor > 0 ->
//      runtime_error mentioning positivity_floor (positivity_comp resolved on the AMR path).
#include <adc/validation/physics/advection_diffusion.hpp>
#include <adc/physics/euler.hpp>

#include <adc/numerics/numerical_flux.hpp>      // RusanovFlux
#include <adc/numerics/reconstruction.hpp>      // Weno5, Minmod
#include <adc/numerics/spatial_operator.hpp>    // reconstruct_pp, positivity_comp
#include <adc/numerics/time/amr_reflux_mf.hpp>  // advance_amr, AmrLevelMP, mf_fill_fine_ghosts_mb

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace adc;

// Euler WITHOUT source (parity with test_positivity_floor.cpp): forwards flux / wave speed and the
// Density-role introspection (conservative_vars, role Density at component 0). advance_amr only needs
// flux / max_wave_speed / source / conservative_vars on the model.
struct EulerNoSrc {
  using State = Euler::State;
  using Aux = Euler::Aux;
  static constexpr int n_vars = Euler::n_vars;
  Euler e{};
  Real gamma = Real(1.4);
  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return e.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return e.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  static VariableSet conservative_vars() { return Euler::conservative_vars(); }
  static VariableSet primitive_vars() { return Euler::primitive_vars(); }
};

static constexpr double kRhoMin = 1e-6, kRhoMax = 1.0;  // contrast 1e6 (Hoffart top-hat)

// Smooth, strictly positive Euler bubble (no face ever undershoots the floor): used for (2).
static void init_smooth(MultiFab& U, const Geometry& geom, const EulerNoSrc& m) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    for_each_cell(U.box(li), [a, geom, m](int i, int j) {
      const Real x = geom.x_cell(i) - Real(0.5), y = geom.y_cell(j) - Real(0.5);
      const Real rho = Real(1.0) + Real(0.4) * std::exp(-(x * x + y * y) / Real(0.02));
      a(i, j, 0) = rho;
      a(i, j, 1) = Real(0.2) * rho;
      a(i, j, 2) = Real(-0.1) * rho;
      const Real ke = Real(0.5) * (a(i, j, 1) * a(i, j, 1) + a(i, j, 2) * a(i, j, 2)) / rho;
      a(i, j, 3) = Real(1.0) / (m.gamma - Real(1)) + ke;  // p = 1, plus kinetic energy
    });
  }
}

// 1e6-contrast top-hat in x with a non-monotone oscillating spike at x ~ 3/4 (the deterministic
// weno5 undershoot of test_positivity_floor.cpp): the EVOLVED contact reconstructs a negative face
// density. p uniform and tiny (quasi-cold, c ~ O(1) on the 1e-6 background). Used for (3).
static void init_spike(MultiFab& U, const Box2D& dom, const EulerNoSrc& m) {
  const int ilo = dom.hi[0] / 3, ihi = 2 * dom.hi[0] / 3;
  const int ks = 3 * dom.hi[0] / 4;
  const Real p0 = Real(1e-6);
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    for_each_cell(U.box(li), [a, ilo, ihi, ks, m, p0](int i, int j) {
      Real rho = (i >= ilo && i <= ihi) ? Real(kRhoMax) : Real(kRhoMin);
      if (i == ks) rho = Real(0.8);
      else if (i == ks + 1) rho = Real(0.5);
      else if (i == ks + 2) rho = Real(kRhoMin);
      else if (i == ks + 3) rho = Real(1.0);
      else if (i == ks + 4) rho = Real(kRhoMin);
      a(i, j, 0) = rho;
      a(i, j, 1) = rho * Real(1.0);  // u = 1: advected contact
      a(i, j, 2) = Real(0);
      a(i, j, 3) = p0 / (m.gamma - Real(1)) + Real(0.5) * rho;
    });
  }
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { if (me == 0) std::printf("FAIL %s\n", w); ++fails; }
  };

  std::printf("=== POSITIVITY FLOOR on AMR (positivity_floor, ADC-259) ===\n");

  const EulerNoSrc model{};
  const Real floor = Real(1e-6);
  const int n = 48;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const BCRec bc;  // periodic

  // ------------------------------------------------------------------------------------------------
  // (1) C/F GHOST CLAMP on the LIVE multi-box ghost fill (mf_fill_fine_ghosts_mb): a coarse parent
  //     with one sub-floor density cell -> the fine ghosts reading it dip below the floor unclamped,
  //     and are brought to >= floor (Density role only, momenta interpolated) with the floor.
  // ------------------------------------------------------------------------------------------------
  {
    const int nc = 8;
    const Box2D cdom = Box2D::from_extents(nc, nc);
    const BoxArray cba(std::vector<Box2D>{cdom});
    const DistributionMapping cdm(1, n_ranks());
    MultiFab Pc(cba, cdm, EulerNoSrc::n_vars, 0);
    Pc.set_val(0.0);
    if (Pc.local_size() > 0) {  // serial: coarse box is local; under np>1 the test is skipped clean
      sync_host();
      Array4 c = Pc.fab(0).array();
      for_each_cell(cdom, [c](int i, int j) {
        c(i, j, 0) = Real(1.0); c(i, j, 1) = Real(0.3); c(i, j, 2) = Real(-0.2); c(i, j, 3) = Real(2.0);
      });
      c(1, 3, 0) = Real(1e-10);  // sub-floor density (coarse means are NOT floored: reachable)
      c(1, 3, 1) = Real(0.5);    // distinctive momentum: the clamp must leave it untouched
    }
    // Fine patch [4,4]-[11,11] (covers coarse [2,2]-[5,5]); ng=2. The left ghosts at fine i in {2,3}
    // coarsen to ci=1; fine (3,6)/(3,7)/(2,6)/(2,7) coarsen to coarse (1,3) -> read the sub-floor cell.
    const Box2D fb{{4, 4}, {11, 11}};
    const BoxArray fba(std::vector<Box2D>{fb});
    const DistributionMapping fdm(1, n_ranks());
    MultiFab Uf(fba, fdm, EulerNoSrc::n_vars, 2);

    auto sub_floor_ghosts = [&](MultiFab& U) {
      int cnt = 0;
      if (U.local_size() > 0) {
        sync_host();
        const ConstArray4 f = U.fab(0).const_array();
        const Box2D v = U.box(0), g = U.fab(0).grown_box();
        for (int j = g.lo[1]; j <= g.hi[1]; ++j)
          for (int i = g.lo[0]; i <= g.hi[0]; ++i)
            if (!v.contains(i, j) && f(i, j, 0) < floor) ++cnt;
      }
      return cnt;
    };

    // unclamped (pos_floor = 0): the sub-floor coarse cell propagates to its fine ghosts.
    Uf.set_val(1.0);
    mf_fill_fine_ghosts_mb(Uf, Pc, Pc, Real(0.5), /*replicated_parent=*/true, Real(0), 0);
    const int n_raw = sub_floor_ghosts(Uf);

    // clamped (pos_floor = floor): every C/F fine-ghost density >= floor; momentum unchanged.
    Uf.set_val(1.0);
    mf_fill_fine_ghosts_mb(Uf, Pc, Pc, Real(0.5), /*replicated_parent=*/true, floor, 0);
    const int n_pp = sub_floor_ghosts(Uf);
    Real ghost_rho = Real(1), ghost_mom = Real(0);
    if (Uf.local_size() > 0) {
      sync_host();
      const ConstArray4 f = Uf.fab(0).const_array();
      ghost_rho = f(3, 6, 0);  // ghost reading coarse (1,3)
      ghost_mom = f(3, 6, 1);
    }
    if (me == 0)
      std::printf("(1) C/F ghost clamp : sub-floor ghosts NU = %d, PP = %d ; ghost(3,6) rho = %.3e "
                  "mom = %.3f\n", n_raw, n_pp, static_cast<double>(ghost_rho),
                  static_cast<double>(ghost_mom));
    if (n_ranks() == 1) {
      chk(n_raw > 0, "1_unclamped_ghost_subfloor");       // the problem exists at the C/F interface
      chk(n_pp == 0, "1_clamped_ghost_floor");            // the clamp fixes every C/F ghost
      chk(ghost_rho >= floor, "1_clamped_ghost_density_floored");
      chk(std::abs(ghost_mom - Real(0.5)) < Real(1e-12), "1_clamped_ghost_momentum_interpolated");
    }
  }

  // one explicit weno5 / rusanov AMR step on a single coarse level (leaf), in place.
  auto amr_step = [&](MultiFab& U, const MultiFab& aux, Real dt, Real pos_floor) {
    std::vector<AmrLevelMP> levels;
    levels.push_back(AmrLevelMP{U, &aux, geom.dx(), geom.dy()});
    advance_amr<Weno5, RusanovFlux>(model, levels, dom, dt, Periodicity{true, true},
                                    /*coarse_replicated=*/true, /*recon_prim=*/false, /*imex=*/false,
                                    NewtonOptions{}, AmrTimeMethod::kEuler, pos_floor);
    U = std::move(levels[0].U);
  };
  auto coarse_mass = [&](const MultiFab& U) {
    sync_host();
    double mass = 0;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 a = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) mass += a(i, j, 0);
    }
    return all_reduce_sum(mass);
  };
  auto finite_min_rho = [&](const MultiFab& U, bool& finite) {
    sync_host();
    Real mn = Real(1e30);
    finite = true;
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 a = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          mn = std::min(mn, a(i, j, 0));
          finite = finite && std::isfinite(static_cast<double>(a(i, j, 0)));
        }
    }
    return mn;
  };
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);

  // ------------------------------------------------------------------------------------------------
  // (2) NO-DEFAULT-CHANGE: smooth positive state -> advance_amr floor ON == floor OFF (bit-identical).
  // ------------------------------------------------------------------------------------------------
  {
    const int ng = Weno5::n_ghost;  // weno5 stencil radius (3)
    MultiFab U0(ba, dm, EulerNoSrc::n_vars, ng);
    init_smooth(U0, geom, model);
    const Real dt = Real(0.1) * geom.dx();
    MultiFab Uoff = U0, Uon = U0;
    amr_step(Uoff, aux, dt, Real(0));      // floor inactive
    amr_step(Uon, aux, dt, floor);         // floor active but never bites a face above it
    sync_host();
    double dmax = 0;
    for (int li = 0; li < Uoff.local_size(); ++li) {
      const ConstArray4 a = Uoff.fab(li).const_array(), b = Uon.fab(li).const_array();
      const Box2D v = Uoff.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int cc = 0; cc < EulerNoSrc::n_vars; ++cc)
            dmax = std::max(dmax, std::abs(static_cast<double>(a(i, j, cc)) - b(i, j, cc)));
    }
    dmax = all_reduce_max(dmax);
    bool finite = true;
    finite_min_rho(Uon, finite);
    const double m0 = coarse_mass(U0), m1 = coarse_mass(Uon);
    if (me == 0)
      std::printf("(2) no-default-change : max|U(floor) - U(no floor)| = %.3e (smooth state)\n", dmax);
    chk(dmax == 0.0, "2_smooth_floor_bit_identical");
    chk(finite, "2_floor_finite");
    chk(std::abs(m1 - m0) < 1e-10, "2_mass_conserved");
  }

  // ------------------------------------------------------------------------------------------------
  // (3) EFFECT on the 1e6-contrast spike: advance_amr WITH the floor stays finite + conserves mass,
  //     and its trajectory differs from the unfloored run (the floor was active on the spike).
  // ------------------------------------------------------------------------------------------------
  {
    const int ng = Weno5::n_ghost;  // weno5 stencil radius (3)
    MultiFab U0(ba, dm, EulerNoSrc::n_vars, ng);
    init_spike(U0, dom, model);
    const Real dt = Real(0.05) * geom.dx() / Real(2.5);  // prudent CFL (u=1, c~1.2 on the background)
    const double m0 = coarse_mass(U0);
    MultiFab Uon = U0, Uoff = U0;
    amr_step(Uon, aux, dt, floor);
    amr_step(Uoff, aux, dt, Real(0));
    bool finite_on = true;
    finite_min_rho(Uon, finite_on);
    const double m1 = coarse_mass(Uon);
    // "active": the floored and unfloored trajectories differ on a valid cell (NaN counts as differ).
    sync_host();
    bool differs = false;
    for (int li = 0; li < Uon.local_size(); ++li) {
      const ConstArray4 a = Uon.fab(li).const_array(), b = Uoff.fab(li).const_array();
      const Box2D v = Uon.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int cc = 0; cc < EulerNoSrc::n_vars; ++cc) {
            const double da = a(i, j, cc), db = b(i, j, cc);
            if (!(da == db)) differs = true;  // NaN != NaN -> differ
          }
    }
    differs = all_reduce_max(differs ? 1.0 : 0.0) > 0.0;
    if (me == 0)
      std::printf("(3) spike + floor : finite = %d, mass drift = %.3e, trajectory differs = %d\n",
                  finite_on ? 1 : 0, m1 - m0, differs ? 1 : 0);
    chk(finite_on, "3_floored_spike_finite");
    chk(std::abs(m1 - m0) < 1e-10, "3_floored_spike_mass_conserved");
    chk(differs, "3_floor_active_on_spike");
  }

  // ------------------------------------------------------------------------------------------------
  // (4) REJECT: a model without a Density role + positivity_floor > 0 -> runtime_error on the AMR path
  //     (positivity_comp resolved at the head of subcycle_level_mp), message mentions positivity_floor.
  // ------------------------------------------------------------------------------------------------
  {
    const adc::validation::AdvectionDiffusion scal{1.0, 0.0, 0.0};
    MultiFab Us(ba, dm, 1, 2), auxs(ba, dm, 3, 1);
    Us.set_val(1.0);
    auxs.set_val(0.0);
    bool threw = false;
    std::string msg;
    try {
      std::vector<AmrLevelMP> levels;
      levels.push_back(AmrLevelMP{Us, &auxs, geom.dx(), geom.dy()});
      advance_amr<Minmod, RusanovFlux>(scal, levels, dom, Real(1e-4), Periodicity{true, true},
                                       /*coarse_replicated=*/true, /*recon_prim=*/false,
                                       /*imex=*/false, NewtonOptions{}, AmrTimeMethod::kEuler,
                                       /*pos_floor=*/floor);
    } catch (const std::runtime_error& e) {
      threw = true;
      msg = e.what();
    }
    if (me == 0)
      std::printf("(4) model without Density : %s (%s)\n", threw ? "REJECTED" : "ACCEPTED (!)",
                  msg.c_str());
    chk(threw, "4_reject_model_without_density");
    chk(msg.find("positivity_floor") != std::string::npos, "4_message_mentions_positivity_floor");
  }

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0) std::printf("OK test_amr_positivity_floor\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

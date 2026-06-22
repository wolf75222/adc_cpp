// Validation DEVICE + MPI multi-GPU (GH200) du B_z par niveau AMR en MULTI-BOX distribue (#59,
// couverture multi-box de AmrSystemCoupler::fill_bz). #59 a fusionne sur master la couverture
// multi-box mono-rang ET MPI np=2/4 (CI Kokkos Serial). On confirme ici le CHEMIN DEVICE distribue :
// grossier 2x2 boites + niveau fin = 2 patchs disjoints, repartis sur n_ranks() GPU (un par rang),
// B_z(x,y) NON CONSTANT pose PAR NIVEAU et PAR BOITE a la resolution du niveau, source S = B_z u lue
// par boite sur device via load_aux<4> (for_each_cell ADC_HD). On valide par advance_amr (le moteur
// device que AmrSystemCoupler appelle ; la facade elle-meme bute sur le concept CoupledSystemLike
// cote nvcc/EDG). B_z etant fonction PURE de la position, les invariants globaux (all_reduce) doivent
// etre INVARIANTS au nombre de rangs : le script compare np=1 (oracle multi-box mono-rang) a np=2/4.
//
// Sortie : reduits GLOBAUX mass/csum/csumsq/cmax (all_reduce sur les boites locales) du bloc B_z apres
// un pas AMR, + un compteur de cellules ou B_z relu != bz(centre du niveau) (all_reduce, doit etre 0).

#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/time/amr_reflux_mf.hpp>  // AmrLevelMP, advance_amr
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

// Modele jouet pilote par B_z : flux nul, source S = B_z u. n_aux=4 -> lit a.B_z par niveau/boite.
struct BzGrow {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};
static_assert(PhysicalModel<BzGrow> && aux_comps<BzGrow>() == 4);

template <class F>
static int fill_bz_level(MultiFab& A, const Geometry& gk, F bz) {
  // Pose B_z par boite a la resolution du niveau ; renvoie le nb de cellules incorrectes (auto-check).
  int bad = 0;
  for (int li = 0; li < A.local_size(); ++li) {
    Fab2D& f = A.fab(li);
    const Box2D g = f.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const Real v = bz(gk.x_cell(i), gk.y_cell(j));
        f(i, j, kAuxBaseComps) = v;
        if (std::fabs(f(i, j, kAuxBaseComps) - v) > Real(1e-12))
          ++bad;
      }
  }
  return bad;
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#endif
  int fails = 0;
  {
    const int me = my_rank(), np = n_ranks();
    const int NC = 16;
    const Box2D dom = Box2D::from_extents(NC, NC);
    const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    const Geometry gf = geom.refine(2);
    const Real dxc = geom.dx(), dyc = geom.dy();

    // grossier MULTI-BOX : 2x2 = 4 boites (max_grid_size=8), reparti sur np rangs (SFC).
    const BoxArray ba_coarse = BoxArray::from_domain(dom, 8);
    const DistributionMapping dm_coarse(ba_coarse.size(), n_ranks());
    // niveau fin : 2 patchs disjoints dans deux boites grossieres differentes.
    const Box2D fb0{{4, 4}, {11, 11}}, fb1{{20, 20}, {27, 27}};
    const BoxArray ba_fine(std::vector<Box2D>{fb0, fb1});
    const DistributionMapping dm_fine(ba_fine.size(), n_ranks());

    auto bz = [](Real x, Real y) {
      return Real(1) + std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
    };

    const Real u0 = Real(2);
    MultiFab Uc(ba_coarse, dm_coarse, 1, 2), Uf(ba_fine, dm_fine, 1, 2);
    Uc.set_val(u0);
    Uf.set_val(u0);
    MultiFab auxc(ba_coarse, dm_coarse, aux_comps<BzGrow>(), 1);
    MultiFab auxf(ba_fine, dm_fine, aux_comps<BzGrow>(), 1);
    auxc.set_val(Real(0));
    auxf.set_val(Real(0));
    int bad = 0;
    bad += fill_bz_level(auxc, geom, bz);  // niveau 0 : centres a dx, par boite locale
    bad += fill_bz_level(auxf, gf, bz);    // niveau 1 : centres a dx/2, par boite locale
    const int bad_global = static_cast<int>(all_reduce_sum(static_cast<double>(bad)));

    std::vector<AmrLevelMP> L;
    L.push_back(AmrLevelMP{std::move(Uc), &auxc, dxc, dyc});
    L.push_back(AmrLevelMP{std::move(Uf), &auxf, dxc / 2, dyc / 2});

    // grossier MULTI-BOX REPARTI (coarse_replicated=false), comme #59 ; halos cross-rang via
    // fill_boundary, source lue par boite sur device.
    advance_amr(BzGrow{}, L, dom, Real(0.05), Periodicity{true, true}, /*coarse_replicated=*/false);
    device_fence();

    // invariants GLOBAUX du grossier (all_reduce sur les boites locales) : INVARIANTS au nb de rangs.
    double lsum = 0, lsumsq = 0, lmax = 0;
    for (int li = 0; li < L[0].U.local_size(); ++li) {
      const ConstArray4 a = L[0].U.fab(li).const_array();
      const Box2D b = L[0].U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
          const double v = a(i, j, 0);
          lsum += v;
          lsumsq += v * v;
          if (std::fabs(v) > lmax)
            lmax = std::fabs(v);
        }
    }
    const double gsum = all_reduce_sum(lsum), gsumsq = all_reduce_sum(lsumsq);
    const double gmax = all_reduce_max(lmax), gmass = gsum * dxc * dyc;

#if defined(ADC_HAS_KOKKOS)
    const char* space = Kokkos::DefaultExecutionSpace::name();
#else
    const char* space = "Serial(host)";
#endif
    if (me == 0) {
      std::printf(
          "AMRBZMPI np=%d exec=%s | coarse mass=%.17e csum=%.17e csumsq=%.17e cmax=%.17e | "
          "bz_bad=%d\n",
          np, space, gmass, gsum, gsumsq, gmax, bad_global);
      if (bad_global != 0) {
        std::printf("FAIL bz mal pose par boite\n");
        ++fails;
      }
      if (!(gmax > u0)) {
        std::printf("FAIL source B_z non lue (cmax <= u0)\n");
        ++fails;
      }
      if (fails == 0)
        std::printf("OK gpu_amr_bz_mpi_validate np=%d (multi-box B_z par niveau, exec=%s)\n", np,
                    space);
    }
  }
#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  comm_finalize();
  return fails ? 1 : 0;
}

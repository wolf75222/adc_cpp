// Chantier "Aux extensible", increment 3 : peuplement B_z cote SystemAssembler MULTI-BLOCS.
// Le canal aux est PARTAGE par tous les blocs ; sa largeur est le MAXIMUM des aux_comps des
// blocs. Un bloc qui declare B_z (n_aux=4) le lit ; un bloc de base (n_aux=3) ignore la
// composante extra, le tout depuis le MEME aux. On verifie :
//   (A) la largeur du canal partage = max sur les blocs (4 ici), B_z peuple a c ;
//   (B) le bloc qui lit B_z voit la source S = B_z u (residu = c) ;
//   (C) le bloc de base, sur le meme aux elargi, reste a residu nul (comp 3 ignoree).

#include <adc/core/model/coupled_system.hpp>
#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/coupling/system/system_coupler.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

// Bloc A : lit B_z. flux nul, source S = B_z u, pas de couplage elliptique.
struct BzGrow {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

// Bloc B : modele de BASE (n_aux=3). flux et source nuls -> residu nul.
struct Scalar {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<BzGrow> && PhysicalModel<Scalar>);

using BlkA = EquationBlock<BzGrow, FirstOrder, ExplicitTime<SSPRK2, 1>>;
using BlkB = EquationBlock<Scalar, FirstOrder, ExplicitTime<SSPRK2, 1>>;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 16;
  const Real c = 0.7;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;  // periodique

  MultiFab UA(ba, dm, 1, 2), UB(ba, dm, 1, 2);
  UA.set_val(Real(1));
  UB.set_val(Real(1));
  BlkA a{"a", BzGrow{}, UA, bc};
  BlkB b{"b", Scalar{}, UB, bc};
  CoupledSystem system{a, b};
  // charges nulles -> f = 0 -> phi = 0 (B_z est independant de l'elliptique).
  ChargeDensityRhs charge{{{Real(0), 0}, {Real(0), 0}}};

  SystemAssembler assembler(system, geom, ba, bc, charge, std::function<bool(Real, Real)>{},
                            [c](Real, Real) { return c; });

  // --- (A) canal partage : largeur = max(4, 3) = 4 ; B_z peuple a c ---
  chk(assembler.aux().fab(0).ncomp() == 4, "shared_aux_width_is_max");
  assembler.solve_fields();
  {
    double maxbz = 0, maxphi = 0;
    const MultiFab& A = assembler.aux();
    for (int li = 0; li < A.local_size(); ++li) {
      const ConstArray4 aa = A.fab(li).const_array();
      const Box2D v = A.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          maxphi = std::max(maxphi, std::fabs(aa(i, j, 0)));
          maxbz = std::max(maxbz, std::fabs(aa(i, j, 3) - c));
        }
    }
    std::printf("  (A) largeur=%d  max|phi|=%.2e  max|B_z - c|=%.2e\n",
                assembler.aux().fab(0).ncomp(), maxphi, maxbz);
    chk(maxphi < 1e-12, "phi_zero");
    chk(maxbz < 1e-14, "Bz_populated_shared");
  }

  // --- (B) le bloc A lit B_z : residu = source = B_z u = c ---
  {
    MultiFab R(ba, dm, 1, 0);
    assembler.block_residual<NoSlope, RusanovFlux>(assembler.system().block<0>(),
                                                   assembler.system().block<0>().U(), R,
                                                   /*recompute_aux=*/false);
    double maxerr = 0;
    for (int li = 0; li < R.local_size(); ++li) {
      const ConstArray4 r = R.fab(li).const_array();
      const Box2D v = R.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          maxerr = std::max(maxerr, std::fabs(r(i, j, 0) - c));
    }
    std::printf("  (B) bloc A : max|R - c| = %.2e\n", maxerr);
    chk(maxerr < 1e-14, "blockA_reads_Bz");
  }

  // --- (C) le bloc B (base) sur le MEME aux elargi ignore la comp 3 : residu nul ---
  {
    MultiFab R(ba, dm, 1, 0);
    assembler.block_residual<NoSlope, RusanovFlux>(assembler.system().block<1>(),
                                                   assembler.system().block<1>().U(), R,
                                                   /*recompute_aux=*/false);
    chk(norm_inf(R) < 1e-14, "blockB_ignores_extra_comp");
    std::printf("  (C) bloc B : ||R||_inf = %.2e\n", norm_inf(R));
  }

  if (fails == 0)
    std::printf("OK test_aux_system_bz\n");
  return fails == 0 ? 0 : 1;
}

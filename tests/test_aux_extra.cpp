// Chantier "Aux extensible", increment 1. Un modele peut declarer un membre statique
// n_aux > 3 pour LIRE des champs auxiliaires SUPPLEMENTAIRES (ici B_z, composante aux 3)
// dans son flux / sa source, sans toucher les modeles existants. On valide au niveau de
// l'operateur spatial (assemble_rhs) :
//   (A) un modele a n_aux=4 voit bien B_z : flux nul -> R = source = B_z * u ;
//   (B) RETRO-COMPAT : un modele de base (n_aux=3, defaut) ignore STRICTEMENT la
//       composante 3 du canal aux -- changer sa valeur ne change rien au residu.
// Le coeur ne connait aucune physique : les deux modeles sont des jouets inline.

#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/storage/fab2d.hpp>
#include <adc/mesh/geometry/geometry.hpp>
#include <adc/mesh/storage/multifab.hpp>
#include <adc/mesh/boundary/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace adc;

// Modele jouet qui LIT un champ aux supplementaire : source S = B_z * u (flux nul).
// n_aux = 4 -> load_aux remplit a.B_z depuis la composante aux 3.
struct MagSource {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // phi, grad_x, grad_y, B_z
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

// Modele de BASE (pas de n_aux) : source S = grad_x * u. Ne doit JAMAIS lire la comp 3.
struct GradSource {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.grad_x * u[0];
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<MagSource>, "MagSource modele PhysicalModel");
static_assert(PhysicalModel<GradSource>, "GradSource modele PhysicalModel");
static_assert(aux_comps<MagSource>() == 4, "MagSource declare n_aux = 4");
static_assert(aux_comps<GradSource>() == 3, "GradSource = contrat de base (3 comps)");

// Remplit une composante c du canal aux a la valeur v sur la boite ETENDUE (ghosts inclus).
static void fill_aux_comp(MultiFab& aux, int c, Real v) {
  for (int li = 0; li < aux.local_size(); ++li) {
    Fab2D& f = aux.fab(li);
    const Box2D g = f.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        f(i, j, c) = v;
  }
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 16;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique

  MultiFab U(ba, dm, 1, 1);
  U.set_val(1.0);  // u = 1 partout (ghosts inclus)

  // --- (A) modele a n_aux=4 : la source lit bien B_z (composante aux 3) ---
  {
    const Real Bz = 0.7;
    MultiFab aux(ba, dm, 4, 1);  // canal aux elargi a 4 composantes
    aux.set_val(0.0);
    fill_aux_comp(aux, 3, Bz);  // B_z partout
    MultiFab R(ba, dm, 1, 0);
    R.set_val(-12345.0);
    MagSource m;
    assemble_rhs<NoSlope>(m, U, aux, geom, R);  // flux nul -> R = source = B_z * u = 0.7
    double maxerr = 0;
    for (int li = 0; li < R.local_size(); ++li) {
      const ConstArray4 r = R.fab(li).const_array();
      const Box2D v = R.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          maxerr = std::max(maxerr, std::fabs(r(i, j, 0) - Bz));
    }
    std::printf("  (A) n_aux=4 : max|R - B_z| = %.2e\n", maxerr);
    chk(maxerr < 1e-14, "extra_aux_Bz_read");
  }

  // --- (B) retro-compat : un modele de base ignore STRICTEMENT la composante 3 ---
  // Meme grad_x impose ; comp 3 = 0 puis 999. R doit etre IDENTIQUE (B_z jamais lu).
  {
    const Real gx = 0.3;
    auto run = [&](Real comp3) {
      MultiFab aux(ba, dm, 4, 1);
      aux.set_val(0.0);
      fill_aux_comp(aux, 1, gx);     // grad_x
      fill_aux_comp(aux, 3, comp3);  // B_z parasite : ne doit pas etre lu
      MultiFab R(ba, dm, 1, 0);
      GradSource m;
      assemble_rhs<NoSlope>(m, U, aux, geom, R);
      return R;
    };
    MultiFab R0 = run(0.0);
    MultiFab R1 = run(999.0);
    double maxdiff = 0, val = 0;
    for (int li = 0; li < R0.local_size(); ++li) {
      const ConstArray4 r0 = R0.fab(li).const_array();
      const ConstArray4 r1 = R1.fab(li).const_array();
      const Box2D v = R0.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
          maxdiff = std::max(maxdiff, std::fabs(r0(i, j, 0) - r1(i, j, 0)));
          val = r0(i, j, 0);
        }
    }
    std::printf("  (B) base n_aux=3 : R(c3=0) vs R(c3=999) max diff = %.2e (R=%.3f)\n", maxdiff,
                val);
    chk(maxdiff == 0.0, "base_model_ignores_extra_comp");
    chk(std::fabs(val - gx) < 1e-14, "base_model_reads_grad_x");
  }

  if (fails == 0)
    std::printf("OK test_aux_extra\n");
  return fails == 0 ? 0 : 1;
}

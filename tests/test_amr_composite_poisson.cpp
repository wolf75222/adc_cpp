// Poisson AMR COMPOSITE branche dans AmrCouplerMP (set_composite_poisson) : test de fidelite.
// Cf. include/adc/coupling/amr_coupler_mp.hpp (compute_aux_composite) + composite_fac_poisson.hpp.
//
// On construit une hierarchie AMR 2 niveaux (grossier + UN patch fin central) portant un modele
// SCALAIRE dont elliptic_rhs(U) = U : le coupleur resout donc Lap phi = U. On pose U = f_rhs =
// Lap(u_exact) avec u_exact = sin(3 pi x) sin(3 pi y) (manufacturee, nulle au bord) -> phi = u_exact.
// On compare le grad phi POSE DANS L'AUX FIN (la quantite que le transport ExB consomme) :
//   - chemin Option A (compute_aux par defaut) : aux fin = grad GROSSIER injecte (constant par morceaux) ;
//   - chemin COMPOSITE (set_composite_poisson) : aux fin = grad du phi FIN resolu par FAC (diff centree fine).
// CRITERE : le composite donne un grad phi NETTEMENT plus precis que l'injection Option A dans la zone
// raffinee -- c'est le verrou de fidelite AMR (les patchs raffinent le couplage elliptique, pas seulement
// le transport). On verifie aussi que le chemin Option A reste bit-identique (composite OFF par defaut).
//
// Serie (Kokkos OFF) : grossier mono-box replique, 1 patch mono-box. Multi-patch / MPI = phases ulterieures.

#include <adc/coupling/amr/amr_coupler_mp.hpp>

#include <adc/core/state.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

static double u_exact(double x, double y) {
  return std::sin(3 * kPi * x) * std::sin(3 * kPi * y);
}
static double f_rhs(double x, double y) {
  return -18.0 * kPi * kPi * u_exact(x, y);
}  // Lap u

// Modele SCALAIRE minimal : aucune dynamique (flux/source nuls) ; elliptic_rhs(U) = U -> Lap phi = U.
struct ScalarCharge {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

// Pose U(i,j,0) = f_rhs(x_cell, y_cell) sur les cellules valides (selon la geometrie @p g du niveau).
static void set_state_f(MultiFab& U, const Geometry& g) {
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    const Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        a(i, j, 0) = f_rhs(g.x_cell(i), g.y_cell(j));
  }
}

// erreur MAX du grad phi de l'aux fin vs le grad analytique, dans la zone INTERIEURE du patch.
static double aux_grad_err(const MultiFab& aux_f, const Geometry& gf, int Ic0, int Ic1, int guard,
                           int r) {
  device_fence();
  const int iIc0 = Ic0 + guard, iIc1 = Ic1 - guard;
  double e = 0;
  const ConstArray4 A = aux_f.fab(0).const_array();
  for (int J = iIc0; J <= iIc1; ++J)
    for (int I = iIc0; I <= iIc1; ++I)
      for (int tj = 0; tj < r; ++tj)
        for (int ti = 0; ti < r; ++ti) {
          const int iff = r * I + ti, jff = r * J + tj;
          const double xf = gf.x_cell(iff), yf = gf.y_cell(jff);
          const double gxa = 3 * kPi * std::cos(3 * kPi * xf) * std::sin(3 * kPi * yf);
          const double gya = 3 * kPi * std::sin(3 * kPi * xf) * std::cos(3 * kPi * yf);
          e = std::fmax(
              e, std::fmax(std::fabs(A(iff, jff, 1) - gxa), std::fabs(A(iff, jff, 2) - gya)));
        }
  return all_reduce_max(e);
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      if (me == 0)
        std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 48, r = 2;
  const Real dxc = Real(1) / n, dxf = dxc / 2;
  Geometry g{Box2D::from_extents(n, n), 0.0, 1.0, 0.0, 1.0};
  Geometry gf = g.refine(r);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  auto [bac, dm] =
      detail::coupler_make_coarse_layout(n, /*distribute=*/false, 0);  // mono-box replique
  const int Ic0 = n / 4, Ic1 = 3 * n / 4 - 1;
  Box2D fb{{r * Ic0, r * Ic0}, {r * Ic1 + r - 1, r * Ic1 + r - 1}};
  BoxArray baf(std::vector<Box2D>{fb});

  MultiFab Uc(bac, dm, 1, 1);
  Uc.set_val(Real(0));
  MultiFab Uf(baf, dm, 1, 1);
  Uf.set_val(Real(0));
  std::vector<AmrLevelMP> levels;
  levels.push_back({std::move(Uc), nullptr, dxc, dxc});
  levels.push_back({std::move(Uf), nullptr, dxf, dxf});

  ScalarCharge model;
  AmrCouplerMP<ScalarCharge> cpl(model, g, bac, bc, std::move(levels), {},
                                 /*replicated_coarse=*/true);
  set_state_f(cpl.coarse(), g);
  set_state_f(cpl.levels()[1].U, gf);

  // --- (1) Option A (chemin par defaut) ---
  cpl.compute_aux();
  const double e_optA = aux_grad_err(*cpl.levels()[1].aux, gf, Ic0, Ic1, /*guard=*/3, r);

  // --- (2) COMPOSITE FAC ---
  cpl.set_composite_poisson(true);
  cpl.compute_aux();
  const double e_comp = aux_grad_err(*cpl.levels()[1].aux, gf, Ic0, Ic1, /*guard=*/3, r);

  if (me == 0)
    std::printf("  grad phi aux fin : e_optionA=%.3e  e_composite=%.3e  (x%.2f)\n", e_optA, e_comp,
                e_optA / std::fmax(e_comp, 1e-30));

  chk(std::isfinite(e_optA) && std::isfinite(e_comp), "erreurs finies");
  // CRITERE : le patch fin raffine VRAIMENT l'elliptique -> grad phi fin nettement plus precis.
  chk(e_comp < 0.5 * e_optA,
      "(fidelite) composite plus precis que Option A sur grad phi (e_comp < 0.5 e_optA)");

  // --- (3) non-regression : composite OFF -> Option A inchange (bit-identique a un coupleur neuf) ---
  {
    MultiFab Uc2(bac, dm, 1, 1);
    Uc2.set_val(Real(0));
    MultiFab Uf2(baf, dm, 1, 1);
    Uf2.set_val(Real(0));
    std::vector<AmrLevelMP> lv2;
    lv2.push_back({std::move(Uc2), nullptr, dxc, dxc});
    lv2.push_back({std::move(Uf2), nullptr, dxf, dxf});
    AmrCouplerMP<ScalarCharge> ref(model, g, bac, bc, std::move(lv2), {}, true);
    set_state_f(ref.coarse(), g);
    set_state_f(ref.levels()[1].U, gf);
    ref.compute_aux();  // Option A (composite OFF par defaut)
    const double e_ref = aux_grad_err(*ref.levels()[1].aux, gf, Ic0, Ic1, 3, r);
    chk(std::fabs(e_ref - e_optA) < 1e-12,
        "(non-regression) Option A inchange (composite OFF par defaut)");
  }

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0)
    std::printf("OK test_amr_composite_poisson\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

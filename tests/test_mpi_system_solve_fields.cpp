// Bug HOTE MPI de System::solve_fields() (#98) : System repartit UNE box unique en round-robin
// (DistributionMapping(1, n_ranks())), donc a np>1 un seul rang possede la box ; les autres ont
// local_size()==0. Le post-traitement par cellule de solve_fields() (derivation phi/grad,
// apply_te, apply_epsilon*, apply_reaction) appelait fab(0) SANS tester local_size() -> les rangs
// sans box locale dereferencaient un fab inexistant et segfaultaient cote hote.
//
// Ce test exerce System + Poisson + solve_fields() (mono-box) sous mpirun -np {1,2,4} et exige que
// l'appel TOURNE (pas de crash) sur tous les rangs et donne un resultat sense. Le solve elliptique
// est COLLECTIF (tous les rangs y participent) ; seul le post-traitement par cellule est local au
// rang proprietaire. Les I/O par cellule (set_density / density / get_state) ne touchent QUE le rang
// proprietaire ; mass() est une reduction collective (sum -> all_reduce) appelee par tous les rangs.
//
// AVANT le fix : segfault a np=2/4 sur le(s) rang(s) sans box locale. APRES : np=1/2/4 verts, et le
// resultat (potentiel, masse) est invariant au nombre de rangs (la box unique vit toujours sur rang 0).

#include <adc/physics/composite.hpp>
#include <adc/physics/euler.hpp>       // Euler (bloc fluide source de T_e)
#include <adc/physics/hyperbolic.hpp>  // ExBVelocity
#include <adc/physics/source.hpp>      // NoSource
#include <adc/runtime/dsl_block.hpp>   // add_compiled_model
#include <adc/runtime/system.hpp>

#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#ifdef ADC_HAS_MPI
#include <mpi.h>
#endif

using namespace adc;

// Source qui lit T_e : exerce le canal aux derive (apply_te) dans solve_fields() (composante 4).
struct TeSource {
  static constexpr int n_aux = 5;
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.T_e * u[0];
    return s;
  }
};
// Bloc de CHARGE : alimente le second membre du Poisson (elliptic_rhs = densite de charge q n).
struct ChargeEll {
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return u[0];
  }  // rho = comp 0
};
struct NoEll {
  template <class State>
  ADC_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using ProbeModel = CompositeModel<ExBVelocity, TeSource, ChargeEll>;  // lit T_e + charge le Poisson
using GasModel = CompositeModel<Euler, NoSource, NoEll>;              // fournit p/rho

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int me = my_rank(), np = n_ranks();

  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("[rank %d/%d] FAIL %s\n", me, np, w);
      ++fails;
    }
  };

  const int n = 16;
  const double gamma = 1.4, rho_gas = 1.0, p_gas = 3.0;
  const double Te = p_gas / rho_gas;  // T = p / rho = 3

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  System sys(cfg);
  add_compiled_model(sys, "gas", GasModel{Euler{gamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", gamma);
  add_compiled_model(sys, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_poisson("composite",
                  "geometric_mg");  // f = somme des briques elliptiques (ici la charge)

  // La box unique vit sur rang 0 (round-robin de 1 box). set_state / set_density ecrivent la box ;
  // on ne les appelle donc QUE sur le rang proprietaire. set_electron_temperature_from cable un
  // INDICE (pas d'acces par cellule), inoffensif sur tous les rangs. La densite de charge a moyenne
  // nulle (rho - rho0) pour que le Poisson periodique soit soluble : on met un creneau symetrique.
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const bool owns = (me == 0);  // box 0 -> rang 0 sous DistributionMapping(1, np)
  if (owns) {
    std::vector<double> Ug(4 * nn, 0.0);
    for (std::size_t k = 0; k < nn; ++k) {
      Ug[0 * nn + k] = rho_gas;
      Ug[3 * nn + k] = p_gas / (gamma - 1.0);
    }
    sys.set_state("gas", Ug);
    // charge a moyenne nulle : +1 sur la moitie gauche, -1 sur la moitie droite (somme = 0).
    std::vector<double> q(nn, 0.0);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        q[static_cast<std::size_t>(j) * n + i] = (i < n / 2) ? 1.0 : -1.0;
    sys.set_density("probe", q);
  }
  sys.set_electron_temperature_from("gas");  // T_e <- p/rho du gaz, recalcule a chaque solve

  // L'APPEL CRITIQUE : sur tous les rangs. Le solve elliptique est collectif ; sans le fix, les
  // rangs sans box locale crashaient ici (fab(0)). On l'enchaine deux fois (ensure_elliptic puis
  // resolve) pour couvrir le chemin construit-puis-reutilise.
  sys.solve_fields();
  sys.solve_fields();

  // Resultat sense : potentiel + masse, lus sur le rang proprietaire (par cellule). mass() est une
  // reduction COLLECTIVE : tous les rangs l'appellent (sinon interblocage de l'all_reduce).
  if (owns) {
    const std::vector<double> phi = sys.potential();
    chk(phi.size() == nn, "potential_size");
    double maxabs = 0, sumphi = 0;
    bool finite = true;
    for (double v : phi) {
      if (!std::isfinite(v))
        finite = false;
      maxabs = std::fmax(maxabs, std::fabs(v));
      sumphi += v;
    }
    chk(finite, "potential_finite");
    chk(maxabs > 0.0, "potential_nonzero");  // une charge non nulle -> un potentiel non trivial

    // solve_fields() a peuple le canal aux (phi, grad phi, T_e) : eval_rhs(probe) = -div F + S lit
    // ce canal sans crash et reste fini (la derivation par cellule a bien tourne sur le rang
    // proprietaire ; la correction T_e exacte est verifiee a part par test_aux_te en serie).
    const std::vector<double> R = sys.eval_rhs("probe");
    bool rfin = !R.empty();
    for (double r : R)
      rfin = rfin && std::isfinite(r);
    chk(rfin, "rhs_finite");

    std::printf("[rank %d/%d] np=%d  |phi|max=%.3e  sum(phi)=%.3e\n", me, np, np, maxabs, sumphi);
  }

  const double mtot = sys.mass("gas");  // collectif (sum -> all_reduce) : appele par TOUS les rangs
  chk(std::isfinite(mtot), "mass_finite");
  // masse du gaz = rho_gas * n*n (la box vit sur rang 0, repliquee logiquement par l'all_reduce).
  chk(std::fabs(mtot - rho_gas * static_cast<double>(nn)) < 1e-9, "mass_value");

#ifdef ADC_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_system_solve_fields (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

// Source COUPLEE generique (System::add_coupled_source, DSL P5 phase 1) sous MPI np={1,2,4}.
//
// System repartit UNE box unique en round-robin (DistributionMapping(1, n_ranks())), donc a np>1 un
// seul rang la possede ; les autres ont local_size()==0. La source couplee s'applique APRES le transport
// (P->couplings, splitting explicite). Les couplages NOMMES historiques (add_ionization/...) dereferencent
// fab(0) EN DUR : ils crasheraient cote hote sur un rang sans box locale. La source GENERIQUE itere sur
// local_size() (no-op sur un rang vide) -> ce test verrouille la SAFETY MPI du nouveau chemin et l'INVARIANCE
// du resultat au nombre de rangs.
//
// On code l'IONISATION en bytecode (d_t n_e = +k n_e n_g, d_t n_i = +k n_e n_g, d_t n_g = -k n_e n_g) :
//   regs : ne=0, ni=1, ng=2, k=3 (constante)
//   e/i : PushReg(k) PushReg(ne) Mul PushReg(ng) Mul          -> k*ne*ng
//   g   : ... Neg                                              -> -(k*ne*ng)
// Densites UNIFORMES -> transport nul -> seule la source agit ; on compare aux invariants physiques
// (n_i+n_g conserve, n_e-n_i conserve, e/i croissent, g decroit) et a la valeur attendue, INVARIANTE en np.

#include <adc/physics/composition/composite.hpp>
#include <adc/physics/bricks/hyperbolic.hpp>  // ExBVelocity (scalaire 1 var, role Density)
#include <adc/physics/bricks/source.hpp>      // NoSource
#include <adc/runtime/builders/compiled/dsl_block.hpp>   // add_compiled_model
#include <adc/runtime/system.hpp>

#include <adc/coupling/source/coupled_source_program.hpp>  // CsOp (opcodes, miroir Python)
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

struct NoEll {
  template <class State>
  ADC_HD Real rhs(const State&) const {
    return Real(0);
  }  // pas de charge -> phi=0 -> derive nulle
};
using Dens = CompositeModel<ExBVelocity, NoSource, NoEll>;  // densite scalaire, transport E x B

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
  const double k = 0.7, dt = 0.01;
  const int nsteps = 25;
  const double ne0 = 0.30, ni0 = 0.10, ng0 = 1.00;

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  System sys(cfg);
  add_compiled_model(sys, "electrons", Dens{}, "none", "rusanov", "conservative", "explicit");
  add_compiled_model(sys, "ions", Dens{}, "none", "rusanov", "conservative", "explicit");
  add_compiled_model(sys, "neutrals", Dens{}, "none", "rusanov", "conservative", "explicit");
  sys.set_poisson("composite", "geometric_mg");

  // Init des densites UNIFORMES sur le rang proprietaire (set_density n'ecrit que la box locale).
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const bool owns = (me == 0);
  if (owns) {
    sys.set_density("electrons", std::vector<double>(nn, ne0));
    sys.set_density("ions", std::vector<double>(nn, ni0));
    sys.set_density("neutrals", std::vector<double>(nn, ng0));
  }

  // --- Source couplee generique (bytecode ionisation) : appelee sur TOUS les rangs (l'enregistrement
  //     resout des indices, aucun acces par cellule ; l'application itere local_size() -> no-op si vide).
  const std::vector<std::string> in_blocks = {"electrons", "ions", "neutrals"};
  const std::vector<std::string> in_roles = {"density", "density", "density"};
  const std::vector<double> consts = {k};
  const std::vector<std::string> out_blocks = {"electrons", "ions", "neutrals"};
  const std::vector<std::string> out_roles = {"density", "density", "density"};
  const int PUSH = static_cast<int>(CsOp::PushReg), MUL = static_cast<int>(CsOp::Mul),
            NEG = static_cast<int>(CsOp::Neg);
  // e : k*ne*ng ; i : k*ne*ng ; g : -(k*ne*ng). reg ne=0, ng=2, k=3.
  std::vector<int> ops = {PUSH, PUSH, MUL, PUSH, MUL,        // electrons (len 5)
                          PUSH, PUSH, MUL, PUSH, MUL,        // ions      (len 5)
                          PUSH, PUSH, MUL, PUSH, MUL, NEG};  // neutrals  (len 6)
  std::vector<int> args = {3, 0, 0, 2, 0, 3, 0, 0, 2, 0, 3, 0, 0, 2, 0, 0};
  std::vector<int> lens = {5, 5, 6};
  // ADC-214 : la description bytecode est regroupee dans un POD CoupledSourceProgram (initialiseurs
  // designes -> appel auto-documente, plus de liste de vecteurs du meme type intervertibles).
  adc::CoupledSourceProgram prog;
  prog.in_blocks = in_blocks;
  prog.in_roles = in_roles;
  prog.consts = consts;
  prog.out_blocks = out_blocks;
  prog.out_roles = out_roles;
  prog.prog_ops = ops;
  prog.prog_args = args;
  prog.prog_lens = lens;
  sys.add_coupled_source(prog);

  // REFERENCE forward-Euler (etat uniforme -> scalaire) : MEME recurrence que l'etage C++.
  double ne = ne0, ni = ni0, ng = ng0;
  for (int s = 0; s < nsteps; ++s) {
    sys.step(dt);  // transport (nul, etat uniforme) + apply_couplings (la source)
    const double r = k * ne * ng;
    ne += dt * r;
    ni += dt * r;
    ng += dt * (-r);
  }

  if (owns) {
    const std::vector<double> de = sys.density("electrons");
    const std::vector<double> di = sys.density("ions");
    const std::vector<double> dg = sys.density("neutrals");
    chk(de.size() == nn && di.size() == nn && dg.size() == nn, "density_size");
    double ge = 0, gi = 0, gg = 0, sde = 0, sdi = 0, sdg = 0;
    bool finite = true;
    for (std::size_t q = 0; q < nn; ++q) {
      finite = finite && std::isfinite(de[q]) && std::isfinite(di[q]) && std::isfinite(dg[q]);
      sde += de[q];
      sdi += di[q];
      sdg += dg[q];
    }
    ge = sde / nn;
    gi = sdi / nn;
    gg = sdg / nn;
    chk(finite, "density_finite");
    // etat reste uniforme (transport nul) : min == max == moyenne.
    bool uniform = true;
    for (std::size_t q = 0; q < nn; ++q)
      uniform = uniform && std::fabs(de[q] - ge) < 1e-12 && std::fabs(dg[q] - gg) < 1e-12;
    chk(uniform, "etat_uniforme");
    // == reference ODE (meme recurrence). Invariant en np : la box vit sur rang 0, la source l'avance
    // identiquement quel que soit le nombre de rangs (les rangs vides ne touchent rien).
    chk(std::fabs(ge - ne) < 1e-10, "n_e == ref");
    chk(std::fabs(gi - ni) < 1e-10, "n_i == ref");
    chk(std::fabs(gg - ng) < 1e-10, "n_g == ref");
    // physique + conservation
    chk(ge > ne0 + 1e-6 && gi > ni0 + 1e-6 && gg < ng0 - 1e-6, "e/i croissent, g decroit");
    chk(std::fabs((gi + gg) - (ni0 + ng0)) < 1e-9, "n_i+n_g conserve");
    chk(std::fabs((ge - gi) - (ne0 - ni0)) < 1e-9, "n_e-n_i conserve");
    std::printf("[rank %d/%d] np=%d  n_e=%.8f n_i=%.8f n_g=%.8f\n", me, np, np, ge, gi, gg);
  }

  // mass() est COLLECTIVE (sum -> all_reduce) : appelee par TOUS les rangs (sinon interblocage). La masse
  // lourde (ions + neutres) est conservee a 1e-9 et INVARIANTE en np.
  const double mi = sys.mass("ions"), mg = sys.mass("neutrals");
  chk(std::isfinite(mi) && std::isfinite(mg), "mass_finite");
  chk(std::fabs((mi + mg) - (ni0 + ng0) * static_cast<double>(nn)) < 1e-7,
      "masse_lourde_conservee");

#ifdef ADC_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_coupled_source (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

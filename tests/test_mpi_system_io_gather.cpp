// ADC-257 : couverture MULTI-RANGS des accesseurs GLOBAUX collectifs de l'IO v1
// (System::density_global / state_global / potential_global) -- le gather all_reduce_sum
// utilise par sim.write / sim.checkpoint -- + un aller-retour checkpoint/restart.
//
// CONTEXTE. System repartit UNE box couvrant tout le domaine en round-robin
// (DistributionMapping(1, n_ranks())) : a np>1 seul le rang 0 la possede, les autres ont
// local_size()==0. Les accesseurs _global remplissent un buffer GLOBAL a partir des fabs LOCAUX
// (aux indices GLOBAUX) puis font un all_reduce_sum : un rang sans box n'ecrit rien (boucle
// local_size() vide -> 0), donc la somme = le champ exact. La correction n'est valide que si
// CHAQUE cellule est ecrite par EXACTEMENT un rang ; un double-comptage (cellule ecrite par >1
// rang, p.ex. box dupliquee/decoupee) MULTIPLIERAIT le gather par le nombre d'ecrivains.
//
// CE TEST verrouille le chemin COLLECTIF du gather a np>1, jamais teste avant ADC-257 (le test
// Python python/tests/test_io_multirank.py tourne en MONO-RANG ; tests/test_mpi_system_gather_scatter.cpp
// couvre la garde fab(0) des fermetures, PAS le gather IO). PORTEE : System ne decoupe pas la box
// aujourd'hui, donc le double-comptage proprement dit n'est pas declenchable par cette topologie ;
// ce que le test verrouille, c'est que (a) les rangs SANS box contribuent EXACTEMENT 0 a
// l'all_reduce (buffer zero-init + boucle local_size() vide), (b) le gather est BIT-A-BIT invariant
// au nombre de rangs, (c) l'aller-retour checkpoint/restart est correct a np=2/4. Un futur decoupage
// /duplication de box qui introduirait un double-comptage casserait T1 (gather = writers x ref).
// Sous mpirun -np {1,2,4} :
//   T1 - GATHER == REFERENCE CONNUE (np-invariant) : sur une densite analytique posee, density_global
//        et state_global egalent BIT-A-BIT la reference calculee identiquement par chaque rang.
//   T2 - GATHER == LOCAL apres pas COLLECTIFS : sur le rang proprietaire, density_global == density,
//        state_global == get_state, potential_global == potential (bit-a-bit). Tous les rangs
//        appellent les 3 accesseurs (collectifs) + mass() ; masse conservee == reference (np-invariant).
//   T3 - CHECKPOINT/RESTART : gather global de l'etat + du potentiel (collectif), restauration dans un
//        System FRAIS (scatter box-proprietaire + horloge), puis K pas identiques sur les deux ->
//        get_state bit-a-bit identique (parite restart), potentiel et masse a la tolerance solveur.
//
// CHOIX DU MODELE : Euler + brique de charge, source NoSource -> phi N'AGIT PAS sur le gaz. Donc (i)
// la masse est conservee par le transport FV conservatif seul (a l'arrondi, ~1e-13), independamment
// du Poisson, ce qui justifie la tolerance 1e-9 de T2 ; (ii) l'etat evolue de maniere DETERMINISTE
// (pur Euler, sans le solveur iteratif) -> la parite restart de get_state est bit-a-bit exacte. Le
// potentiel, lui, sort d'un MG iteratif : on le compare a une tolerance serree (robuste a l'ordre de
// reduction sous OpenMP), pas bit-a-bit.
//
// DISCIPLINE COLLECTIVE MPI : density_global / state_global / potential_global et mass() font un
// all_reduce_sum -> TOUS les rangs DOIVENT les appeler (sinon interblocage). set_state ecrit la box
// du proprietaire : appele UNIQUEMENT sur le rang proprietaire (convention des tests MPI System ;
// write_state est en fait no-op sur un rang vide, mais on garde la garde owns). set_potential et
// set_clock sont MPI-safe (no-op sur les rangs vides). step() / step_cfl() sont collectifs.

#include <adc/physics/composition/composite.hpp>
#include <adc/physics/fluids/euler.hpp>      // Euler (bloc fluide a 4 composantes, etat conservatif riche)
#include <adc/physics/bricks/source.hpp>     // NoSource
#include <adc/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
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

// Brique elliptique de CHARGE : alimente le second membre du Poisson (charge q n = rho = comp 0),
// pour que potential_global() renvoie un potentiel non trivial. Identique a test_mpi_system_solve_fields.
struct ChargeEll {
  template <class State>
  ADC_HD Real rhs(const State& u) const {
    return u[0];
  }
};

using GasModel = CompositeModel<Euler, NoSource, ChargeEll>;  // Euler + charge le Poisson

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
  const double gamma = 1.4;
  const std::size_t nn = static_cast<std::size_t>(n) * n;

  // Etat de REFERENCE, calcule a l'IDENTIQUE par chaque rang (fonction pure, aucun MPI). Bosse
  // gaussienne lisse en densite (p = rho -> le gaz evolue en acoustique sous Euler), quantite de
  // mouvement au repos, E = p/(gamma-1). Layout component-major (c*ny + j)*nx + i, EXACTEMENT celui
  // de state_global / get_state. Sert de cible bit-a-bit np-invariante a T1.
  auto build_ref = [&]() {
    std::vector<double> U(4 * nn, 0.0);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) / n, y = (j + 0.5) / n;
        const double r =
            1.0 + 0.4 * std::exp(-50.0 * ((x - 0.4) * (x - 0.4) + (y - 0.5) * (y - 0.5)));
        const std::size_t k = static_cast<std::size_t>(j) * n + i;
        U[0 * nn + k] = r;                  // rho
        U[3 * nn + k] = r / (gamma - 1.0);  // E = p/(gamma-1), p = rho ; momentum (comp 1,2) = 0
      }
    }
    return U;
  };
  const std::vector<double> Uref = build_ref();
  std::vector<double> rho_ref(nn);
  for (std::size_t k = 0; k < nn; ++k)
    rho_ref[k] = Uref[0 * nn + k];
  double mass_ref = 0.0;
  for (double r : rho_ref)
    mass_ref += r;

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  // Construction COLLECTIVE (repliquee sur tous les rangs).
  System sys(cfg);
  add_compiled_model(sys, "gas", GasModel{Euler{gamma}, NoSource{}, ChargeEll{}}, "minmod",
                     "rusanov", "conservative", "explicit", gamma);
  sys.set_poisson("composite", "geometric_mg");  // f = somme des briques elliptiques (la charge)

  const bool owns = (me == 0);  // box 0 -> rang 0 sous DistributionMapping(1, np)
  if (owns)
    sys.set_state("gas", Uref);

  // === T1 : gather == reference connue (np-invariant), sur le champ fraichement pose ===========
  // Tous les rangs appellent les accesseurs collectifs ; le resultat egale BIT-A-BIT la reference.
  {
    const std::vector<double> dG = sys.density_global("gas");
    const std::vector<double> sG = sys.state_global("gas");
    chk(dG.size() == nn, "T1_density_global_size");
    chk(sG.size() == 4 * nn, "T1_state_global_size");
    chk(dG == rho_ref, "T1_density_global_eq_ref_no_double_count");
    chk(sG == Uref, "T1_state_global_eq_ref_no_double_count");
  }

  // === T2 : apres des pas COLLECTIFS, gather == accesseur local sur le proprietaire ============
  const double dt = 0.01;
  for (int s = 0; s < 5; ++s)
    sys.step(dt);  // collectif : solve_fields (Poisson) + advance

  const std::vector<double> dG = sys.density_global("gas");  // collectif (tous les rangs)
  const std::vector<double> sG = sys.state_global("gas");    // collectif
  const std::vector<double> pG = sys.potential_global();     // collectif (resout le Poisson)
  chk(dG.size() == nn && sG.size() == 4 * nn && pG.size() == nn, "T2_global_sizes");
  bool gfin = true;
  for (double v : sG)
    gfin = gfin && std::isfinite(v);
  for (double v : pG)
    gfin = gfin && std::isfinite(v);
  chk(gfin, "T2_globals_finite");
  // Sur le rang proprietaire UNIQUEMENT : le gather GLOBAL (deja calcule collectivement ci-dessus)
  // egale BIT-A-BIT l'accesseur LOCAL (density / get_state / potential, lecture fab(0) NON collective).
  // On NE rappelle PAS les accesseurs _global ici : ce sont des all_reduce -> les rappeler sous le
  // garde owns ferait diverger les collectifs entre rangs (MPI_ERR_TRUNCATE).
  if (owns) {
    chk(dG == sys.density("gas"), "T2_density_global_eq_local");
    chk(sG == sys.get_state("gas"), "T2_state_global_eq_local");
    chk(pG == sys.potential(), "T2_potential_global_eq_local");
  }
  const double mtot = sys.mass("gas");  // collectif (sum -> all_reduce)
  chk(std::isfinite(mtot), "T2_mass_finite");
  chk(std::fabs(mtot - mass_ref) < 1e-9, "T2_mass_conserved_eq_ref");

  // === T3 : aller-retour checkpoint/restart (parite bit-a-bit, np-invariant) ====================
  // Instantane GLOBAL (collectif) de l'etat + du potentiel + de l'horloge.
  const std::vector<double> ckpt_state = sys.state_global("gas");
  const std::vector<double> ckpt_phi = sys.potential_global();
  const double ckpt_t = sys.time();
  const int ckpt_ms = sys.macro_step();

  // System FRAIS, construit a l'identique sur tous les rangs.
  System rs(cfg);
  add_compiled_model(rs, "gas", GasModel{Euler{gamma}, NoSource{}, ChargeEll{}}, "minmod",
                     "rusanov", "conservative", "explicit", gamma);
  rs.set_poisson("composite", "geometric_mg");
  // Restauration : scatter box-proprietaire (set_state/set_potential ecrivent la box du rang 0) +
  // horloge (set_clock est MPI-safe, scalaires). state_global et get_state partagent le MEME layout
  // mono-box, donc set_state(ckpt_state) sur le proprietaire reproduit exactement l'etat.
  if (owns) {
    rs.set_state("gas", ckpt_state);
    rs.set_potential(ckpt_phi);
  }
  rs.set_clock(ckpt_t, ckpt_ms);

  // K pas identiques sur les DEUX systemes (collectifs). L'etat (pur Euler, sans retro-action de phi)
  // evolue de maniere deterministe -> get_state bit-a-bit identique. Le warm-start MG restaure
  // (set_potential) ramene le potentiel a la tolerance du solveur.
  for (int s = 0; s < 4; ++s) {
    sys.step(dt);
    rs.step(dt);
  }
  if (owns) {
    chk(rs.get_state("gas") == sys.get_state("gas"), "T3_restart_state_bit_identical");
    // Potentiel : tolerance serree (MG iteratif -> ordre de reduction non bit-stable sous OpenMP).
    const std::vector<double> p_rs = rs.potential(), p_sys = sys.potential();
    double dphi = 0.0, phimax = 0.0;
    for (std::size_t k = 0; k < p_sys.size(); ++k) {
      dphi = std::fmax(dphi, std::fabs(p_rs[k] - p_sys[k]));
      phimax = std::fmax(phimax, std::fabs(p_sys[k]));
    }
    chk(dphi <= 1e-10 * (1.0 + phimax), "T3_restart_potential_within_tol");
  }
  const double m_sys = sys.mass("gas"), m_rs = rs.mass("gas");  // collectifs (tous les rangs)
  chk(std::fabs(m_rs - m_sys) < 1e-12, "T3_restart_mass_parity");

  if (owns)
    std::printf("[rank %d/%d] np=%d  mass=%.12f  |phi|gathered=%zu  OK gather+restart\n", me, np,
                np, m_sys, pG.size());

#ifdef ADC_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_system_io_gather (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

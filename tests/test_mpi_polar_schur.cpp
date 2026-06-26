// PARITE / CONVERGENCE MPI MULTI-RANG du solveur elliptique POLAIRE TENSORIEL (PolarTensorKrylovSolver)
// et, par extension, de l'etage Schur polaire (PolarCondensedSchurSourceStepper) dont c'est le verrou.
//
// CONTEXTE. Le solveur polaire etait MONO-RANG (garde-fou dur si n_ranks()>1). Ce PR le rend MULTI-RANG
// MPI par decoupage AZIMUTAL (theta seul) : chaque box couvre la plage RADIALE complete (le sweep Thomas
// du preconditionneur RadialLine reste box-local), les produits scalaires sont collectifs (all_reduce),
// la projection de jauge (operateur pure-Neumann singulier) est all_reduit sur tous les rangs.
//
// PROPRIETE TESTEE. Pour un decoupage en THETA SEUL, le preconditionneur RadialLine est INVARIANT au
// decoupage (chaque ligne radiale a theta fixe est resolue INDEPENDAMMENT, qu'elle vive seule dans une
// box ou parmi d'autres) ; la matvec est correcte via les halos (fill_ghosts) ; les normes/dot sont
// GLOBALES (all_reduce). La BiCGStab est donc MATHEMATIQUEMENT IDENTIQUE quel que soit le decoupage, a
// la SEULE reassociation FP de l'all_reduce pres. On verifie, sur une MMS polaire (solution manufacturee
// exacte, anneau r in [r_min, r_max] > 0, theta periodique) :
//
//   (a) PARITE MONO-RANG vs DECOUPE A 1 RANG : a np=1, une reference MONO-BOX et un decoupage THETA
//       multi-box (8 boites) donnent la MEME solution convergee (ecart L2 == 0 a la tolerance FP). Le
//       preconditionneur RadialLine est mathematiquement invariant au decoupage theta ; seule la
//       REASSOCIATION FP des produits scalaires (somme par box dans un ordre different) perturbe la
//       BiCGStab pres de la convergence, ce qui peut decaler le nombre d'iters de quelques unites pour
//       atteindre la MEME tolerance relative. On verifie donc la PARITE DE LA SOLUTION (ecart L2 ~ FP)
//       et la convergence, pas l'egalite exacte des iters entre decoupages distincts.
//
//   (b) PARITE MULTI-RANG : a np=2 (et np=4), le decoupage THETA reparti sur les rangs converge, et son
//       erreur L2 GLOBALE vs l'exact (all_reduit sur tous les rangs) colle a celle de la reference
//       MONO-BOX (boite unique sur le rang 0 sous le round-robin du solveur, les autres rangs vides
//       contribuant 0 aux collectifs) a la tolerance du solveur. L'operateur + le preconditionneur
//       RadialLine sont invariants au decoupage theta, donc la solution discrete -- et son erreur vs
//       l'exact -- sont les MEMES quel que soit le decoupage, a la reassociation FP des all_reduce pres.
//       Le nombre d'iterations est IDENTIQUE sur tous les rangs (critere d'arret collectif).
//
//   (c) CAS PURE-NEUMANN (operateur SINGULIER, jauge) : meme parite, en exercant project_mean ALL-REDUIT
//       sur les rangs (les deux bords radiaux Neumann -> constante dans le noyau, jauge fixee par
//       projection de moyenne nulle GLOBALE). Sans l'all_reduce de la moyenne, la jauge differerait par
//       rang et la solution rassemblee serait incoherente.
//
// Lance via ctest a np=1/2/4 (ntheta divisible par np). Independant du backend (header-only, propriete
// algebrique). Le verrou Schur leve = ce solveur multi-rang ; le test cible directement le solveur.

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;

// ---------------------------------------------------------------------------------------------------
// MMS Dirichlet : phi(r, theta) = S(r) + H(r) cos(m theta), H s'annule aux bords radiaux (modes m != 0
// a reflexion Dirichlet homogene exacte) ; S porte la donnee Dirichlet non triviale. Tenseur A = I
// (isotrope) : exerce la metrique polaire + la BiCGStab + le preconditionneur RadialLine distribue.
// ---------------------------------------------------------------------------------------------------
static double aS() {
  return kPiL / (kRmax - kRmin);
}
static double Sf(double r) {
  return 1.0 + 0.5 * (r - kRmin);
}
static double Hf(double r) {
  return std::sin(aS() * (r - kRmin));
}
static double Hp(double r) {
  return aS() * std::cos(aS() * (r - kRmin));
}
static double Hpp(double r) {
  return -aS() * aS() * std::sin(aS() * (r - kRmin));
}
static double phi_dir(double r, double th, int m) {
  return Sf(r) + Hf(r) * std::cos(m * th);
}
static double f_dir(double r, double th,
                    int m) {  // A = I : div(grad phi) = phi_rr + phi_r/r + phi_tt/r^2
  const double rad = 0.0 + 0.5 / r + (Hpp(r) + Hp(r) / r) * std::cos(m * th);
  const double azi = -(double)m * m * Hf(r) * std::cos(m * th) / (r * r);
  return rad + azi;
}

// MMS pure-Neumann (operateur singulier, jauge) : phi = G(r) + K(r) cos(m theta), G'=K'=0 aux bords.
static double bN() {
  return kPiL / (kRmax - kRmin);
}
static double Gf(double r) {
  return std::cos(bN() * (r - kRmin));
}
static double Gp(double r) {
  return -bN() * std::sin(bN() * (r - kRmin));
}
static double Gpp(double r) {
  return -bN() * bN() * std::cos(bN() * (r - kRmin));
}
static double Kf(double r) {
  const double u = r - kRmin, w = r - kRmax;
  return u * u * w * w;
}
static double Kp(double r) {
  const double u = r - kRmin, w = r - kRmax;
  return 2 * u * w * w + 2 * u * u * w;
}
static double Kpp(double r) {
  const double u = r - kRmin, w = r - kRmax;
  return 2 * w * w + 8 * u * w + 2 * u * u;
}
static double phi_neu(double r, double th, int m) {
  return Gf(r) + Kf(r) * std::cos(m * th);
}
static double f_neu(double r, double th, int m) {  // A = I
  const double rad = Gpp(r) + Gp(r) / r + (Kpp(r) + Kp(r) / r) * std::cos(m * th);
  const double azi = -(double)m * m * Kf(r) * std::cos(m * th) / (r * r);
  return rad + azi;
}

enum class Problem { Dirichlet, Neumann };

// Construit un BoxArray decoupe en THETA SEUL : decoupe [0, nth-1] en nseg segments egaux, chaque box
// couvre la plage radiale complete [0, nr-1]. nseg==1 -> boite unique (reference mono-box).
static BoxArray theta_split(int nr, int nth, int nseg) {
  std::vector<Box2D> boxes;
  int base = nth / nseg, rem = nth % nseg, cur = 0;
  for (int k = 0; k < nseg; ++k) {
    int len = base + (k < rem ? 1 : 0);
    boxes.push_back(Box2D{{0, cur}, {nr - 1, cur + len - 1}});
    cur += len;
  }
  return BoxArray(std::move(boxes));
}

// Remplit les cellules valides locales de mf avec une fonction analytique de (r, theta).
template <class F>
static void fill_local(MultiFab& mf, const PolarGeometry& g, F fn) {
  for (int li = 0; li < mf.local_size(); ++li) {
    const Box2D vb = mf.box(li);
    Array4 a = mf.fab(li).array();
    for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
      for (int i = vb.lo[0]; i <= vb.hi[0]; ++i)
        a(i, j, 0) = fn(g.r_cell(i), g.theta_cell(j));
  }
}

// Resout la MMS sur (nr x nth) avec un BoxArray donne (mono-box ou decoupe theta). Le solveur repartit
// le BoxArray en ROUND-ROBIN sur les rangs (DistributionMapping interne ; cf. PolarTensorKrylovSolver).
// Rend (iters, converged) ; ECRIT la solution locale dans phi_out (DOIT etre alloue avec le MEME
// round-robin que le solveur pour que les boxes locales coincident). Le tenseur est A = I.
static PolarKrylovResult solve_mms(const PolarGeometry& g, const BoxArray& ba, Problem prob, int m,
                                   MultiFab& phi_out) {
  BCRec bc;
  bc.ylo = bc.yhi = BCType::Periodic;  // theta periodique
  if (prob == Problem::Dirichlet) {
    bc.xlo = bc.xhi = BCType::Dirichlet;
    bc.xlo_val = Sf(kRmin);
    bc.xhi_val = Sf(kRmax);
  } else {
    bc.xlo = bc.xhi = BCType::Foextrap;  // Neumann homogene radial -> operateur singulier (jauge)
  }

  PolarTensorKrylovSolver solver(g, ba, bc, PolarPrecond::RadialLine);
  // A = I (coefficients internes a 1) : on n'appelle pas set_coefficients avec des champs externes ; on
  // doit toutefois remplir les ghosts du store interne -> set_coefficients(nullptr,...) reactive le store.
  solver.set_coefficients(nullptr, nullptr);

  // RHS = f analytique (= div(A grad phi_exact)) sur les cellules valides locales.
  if (prob == Problem::Dirichlet)
    fill_local(solver.rhs(), g, [&](double r, double th) { return f_dir(r, th, m); });
  else
    fill_local(solver.rhs(), g, [&](double r, double th) { return f_neu(r, th, m); });

  solver.phi().set_val(0.0);  // depart froid
  PolarKrylovResult kr = solver.solve(1e-11, 4000);

  // recopie la solution locale dans phi_out (memes ba/dm -> memes boxes locales).
  for (int li = 0; li < phi_out.local_size(); ++li) {
    const Box2D vb = phi_out.box(li);
    Array4 d = phi_out.fab(li).array();
    const ConstArray4 s = solver.phi().fab(li).const_array();
    for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
      for (int i = vb.lo[0]; i <= vb.hi[0]; ++i)
        d(i, j, 0) = s(i, j, 0);
  }
  return kr;
}

// Erreur L2 GLOBALE (ponderee volume r dr dtheta) entre phi distribue et l'exact (collectif all_reduce).
// Pour le cas pure-Neumann (jauge), on retire d'abord la moyenne FV GLOBALE de (phi - exact) -> erreur
// MODULO LA CONSTANTE (la solution est definie a une constante pres).
static double err_l2_global(const MultiFab& phi, const PolarGeometry& g, Problem prob, int m) {
  const double dr = g.dr(), dth = g.dtheta();
  // 1) moyenne globale de (phi - exact) si Neumann (jauge), sinon 0.
  double mean = 0.0;
  if (prob == Problem::Neumann) {
    double sum = 0, vol = 0;
    for (int li = 0; li < phi.local_size(); ++li) {
      const Box2D vb = phi.box(li);
      const ConstArray4 p = phi.fab(li).const_array();
      for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
        for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
          const double w = g.r_cell(i) * dr * dth;
          sum += (p(i, j, 0) - phi_neu(g.r_cell(i), g.theta_cell(j), m)) * w;
          vol += w;
        }
    }
    sum = all_reduce_sum(sum);
    vol = all_reduce_sum(vol);
    mean = vol > 0 ? sum / vol : 0.0;
  }
  // 2) erreur L2 ponderee (collectif).
  double l2 = 0, vol = 0;
  for (int li = 0; li < phi.local_size(); ++li) {
    const Box2D vb = phi.box(li);
    const ConstArray4 p = phi.fab(li).const_array();
    for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
      for (int i = vb.lo[0]; i <= vb.hi[0]; ++i) {
        const double w = g.r_cell(i) * dr * dth;
        const double ex = (prob == Problem::Dirichlet) ? phi_dir(g.r_cell(i), g.theta_cell(j), m)
                                                       : phi_neu(g.r_cell(i), g.theta_cell(j), m);
        const double e = p(i, j, 0) - ex - mean;
        l2 += e * e * w;
        vol += w;
      }
  }
  l2 = all_reduce_sum(l2);
  vol = all_reduce_sum(vol);
  return std::sqrt(l2 / vol);
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  // Code de sortie : declare AVANT le scope accolade Kokkos (il est lu par le return APRES la
  // fermeture du scope). Le declarer dedans ne compile pas des que POPS_HAS_KOKKOS est defini
  // (cas MPI + Kokkos) -- defaut invisible tant que le job MPI compilait sans Kokkos.
  long fails = 0;
#if defined(POPS_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
  {
#endif
    const int me = my_rank(), np = n_ranks();
    const int nr = 48, nth = 96, m = 3;  // nth divisible par np <= 4 et par 8 (decoupage theta)
    Box2D dom = Box2D::from_extents(nr, nth);
    PolarGeometry g{dom, kRmin, kRmax};

    auto chk = [&](bool c, const char* w) {
      if (!c) {
        if (me == 0)
          std::printf("  ECHEC %s (np=%d)\n", w, np);
        ++fails;
      }
    };

    // round-robin (le MEME que le solveur construit en interne) pour aligner les boxes locales de phi_out.
    auto rr = [&](const BoxArray& ba) { return DistributionMapping(ba.size(), np); };

    for (Problem prob : {Problem::Dirichlet, Problem::Neumann}) {
      const char* pname = (prob == Problem::Dirichlet) ? "Dirichlet" : "Neumann(jauge)";

      // -----------------------------------------------------------------------------------------------
      // REFERENCE MONO-BOX : boite unique. Sous le round-robin du solveur, elle vit sur le rang 0 ; les
      // autres rangs ont local_size()==0 (et contribuent 0 aux collectifs dot/project_mean/err). L'erreur
      // L2 GLOBALE vs l'exact est all_reduite -> identique sur tous les rangs. C'est l'oracle.
      // -----------------------------------------------------------------------------------------------
      BoxArray ba_mono(std::vector<Box2D>{dom});
      MultiFab phi_ref(ba_mono, rr(ba_mono), 1, 1);
      PolarKrylovResult kr_ref = solve_mms(g, ba_mono, prob, m, phi_ref);
      const double err_ref = err_l2_global(phi_ref, g, prob, m);  // collectif
      chk(kr_ref.converged, "ref_mono_converge");

      // -----------------------------------------------------------------------------------------------
      // DECOUPAGE THETA reparti sur les rangs (round-robin). nseg boites en theta (chaque box = plage r
      // complete). np=1 : 8 boites mono-rang (parite multi-box). np>=2 : 2 boites par rang (exerce le
      // multi-box intra-rang ET le cross-rang MPI). La solution discrete est invariante au decoupage theta
      // -> son erreur L2 vs l'exact colle a l'oracle a la reassociation FP des all_reduce pres.
      // -----------------------------------------------------------------------------------------------
      const int nseg = (np >= 2) ? np * 2 : 8;
      BoxArray bad = theta_split(nr, nth, nseg);
      MultiFab phid(bad, rr(bad), 1, 1);
      PolarKrylovResult krd = solve_mms(g, bad, prob, m, phid);
      const double errd = err_l2_global(phid, g, prob, m);  // collectif
      const double derr = std::fabs(errd - err_ref);
      if (me == 0)
        std::printf(
            "[np=%d %-14s] mono: iters=%d err=%.6e | theta(%d box): iters=%d err=%.6e | d=%.3e\n",
            np, pname, kr_ref.iters, err_ref, nseg, krd.iters, errd, derr);
      chk(krd.converged, "dist_converge");
      // PARITE de la solution : meme operateur + meme precond (invariant au decoupage theta) -> meme erreur
      // discrete vs l'exact, a la tolerance du solveur (1e-11 rel) + reassociation FP des all_reduce.
      chk(derr <= 1e-8, "dist_err_matches_mono");

      // iterations IDENTIQUES sur TOUS les rangs du MEME decoupage (critere d'arret base sur des scalaires
      // GLOBAUX all_reduits -> il se declenche a la meme iteration partout). spread == 0.
      long it_min = krd.iters, it_max = krd.iters;
#ifdef POPS_HAS_MPI
      if (np > 1) {
        long lo = krd.iters, hi = krd.iters;
        MPI_Allreduce(&lo, &it_min, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&hi, &it_max, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      }
#endif
      chk(it_min == it_max, "dist_same_iters_all_ranks");
    }

#ifdef POPS_HAS_MPI
    if (np > 1) {
      long g = 0;
      MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      fails = g;
    }
#endif
    if (me == 0) {
      if (fails == 0)
        std::printf("OK test_mpi_polar_schur (np=%d)\n", np);
      else
        std::printf("FAIL test_mpi_polar_schur (np=%d) : %ld echecs\n", np, fails);
    }
#if defined(POPS_HAS_KOKKOS)
  }
  Kokkos::finalize();
#endif
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

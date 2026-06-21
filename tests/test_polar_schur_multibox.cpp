// PARITE MULTI-BOX (plusieurs boites PAR RANG) du solveur elliptique POLAIRE TENSORIEL
// (PolarTensorKrylovSolver) -- extension de B2 (multi-rang MPI, decoupage theta seul, une box couvrant
// l'anneau par rang) vers le cas ou UN rang porte PLUSIEURS boites.
//
// CONTEXTE. B2 a rendu le Schur polaire multi-rang par decoupage AZIMUTAL : chaque box couvre la plage
// radiale complete, les produits scalaires sont collectifs (all_reduce), la jauge est all_reduite. Mais
// le test B2 (test_mpi_polar_schur) n'exerce que A = I (tenseur ISOTROPE). Le risque MULTI-BOX reel est
// le STENCIL A 9 POINTS des TERMES CROISES a_rt / a_tr : polar_radial_div / polar_azimuthal_div lisent
// les voisins DIAGONAUX p(i+-1, j+-1) (cf. dth_p, dr_p). Avec UNE box (anneau complet, theta periodique)
// ces ghosts diagonaux sont remplis par le repli periodique ; avec PLUSIEURS boites, ils doivent venir
// d'une box VOISINE DIAGONALE via fill_boundary (echange de halos intra-niveau, coins inclus). Ce chemin
// (coin du 9-points cross-box) n'avait JAMAIS ete exerce en polaire. Ce test le couvre.
//
// PROPRIETES TESTEES (mono-rang, plusieurs boites locales ; cf. test_mpi_polar_schur pour le cross-rang) :
//
//   (A) PARITE TENSEUR (TERMES CROISES) THETA-SPLIT vs MONO-BOX. Tenseur PLEIN NON SYMETRIQUE
//       (a_rt != a_tr, comme la rotation B^{-1} du Schur). Decoupage en THETA seul (chaque box = plage r
//       complete, contrainte du precond RadialLine). La solution convergee multi-box DOIT coller a la
//       reference mono-box (ecart L2 ~ reassociation FP des produits scalaires par box). Si le coin du
//       9-points n'etait PAS rempli cross-box, le terme croise serait FAUX au bord de box -> ecart O(1).
//
//   (B) PARITE 2D TILING (COUPE r ET theta) avec precond JACOBI vs MONO-BOX. Jacobi est par cellule
//       (M^{-1} diagonal), sans contrainte de layout : on peut couper r ET theta (vrai pavage 2D). C'est
//       le chemin "multi-box COMPLET" du contrat (cf. check_radial_columns : seul RadialLine exige des
//       colonnes radiales completes). On verifie que le pavage 2D (4 boites : 2 en r x 2 en theta) donne
//       la MEME solution que la reference mono-box (Jacobi aussi), a la reassociation FP pres. Exerce les
//       quatre coins internes du pavage (halos de box adjacentes en r, en theta, ET en diagonale).
//
//   (C) GARDE-FOU DE LAYOUT. Un decoupage qui COUPE r sous precond RadialLine DOIT lever (le sweep Thomas
//       en r ne peut pas franchir une frontiere de box). On verifie que la construction throw -- le repli
//       documente etant Jacobi (cas B). C'est un resultat CLAIR, pas un faux silencieux.
//
//   (D) PARITE NEUMANN (jauge) THETA-SPLIT vs MONO-BOX. Operateur singulier pure-Neumann radial : la
//       projection de moyenne nulle (project_mean) accumule sur TOUTES les boites locales puis all_reduit.
//       Multi-box mono-rang : la somme locale parcourt plusieurs boites -> verifie que la jauge globale
//       est coherente quel que soit le decoupage.
//
// CIBLE le MULTI-BOX intra-rang (a np=1 : 8 boites theta / pavage 2x2 sur UN rang). Les solves et les
// normes L2 sont COLLECTIFS (all_reduce sur tous les rangs) -> le test est aussi CORRECT sous MPI (la
// parite tient cross-rang ; le cross-rang A=I est par ailleurs couvert par test_mpi_polar_schur).
// Enregistre serie (adc_add_test). Independant du backend (header-only, propriete algebrique ; tous les
// kernels du chemin sont des foncteurs nommes device-clean).

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/elliptic/polar_tensor_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;

// ------------------------------------------------------------------------------------------------
// MMS tenseur (cf. test_polar_tensor_elliptic_mms) : phi = S(r) + H(r) cos(m theta), H s'annule aux
// bords radiaux (reflexion Dirichlet homogene exacte des modes m != 0). f = div(A grad phi) pour A
// CONSTANT, avec terme croise (a_rt + a_tr) phi_rt / r.
// ------------------------------------------------------------------------------------------------
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
static double f_tensor(double r, double th, int m, double arr, double art, double atr, double att) {
  const double rad =
      0.0 + 0.5 / r + (Hpp(r) + Hp(r) / r) * std::cos(m * th);  // S''+S'/r + (H''+H'/r) cos
  const double azi = -(double)m * m * Hf(r) * std::cos(m * th) / (r * r);  // -m^2 H cos / r^2
  const double cross = -(double)m * Hp(r) * std::sin(m * th) / r;          // -m H' sin / r
  return arr * rad + att * azi + (art + atr) * cross;
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

static DistributionMapping rr_dm(const BoxArray& ba) {
  return DistributionMapping(ba.size(), n_ranks());
}

// BoxArray decoupe en THETA seul : nseg segments egaux en theta, chaque box couvre [0, nr-1] en r.
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

// BoxArray pave en 2D (nr_seg en r x nth_seg en theta) : COUPE r ET theta -> precond Jacobi obligatoire.
static BoxArray tile_2d(int nr, int nth, int nr_seg, int nth_seg) {
  std::vector<Box2D> boxes;
  int rbase = nr / nr_seg, rrem = nr % nr_seg;
  int tbase = nth / nth_seg, trem = nth % nth_seg;
  int rcur = 0;
  for (int a = 0; a < nr_seg; ++a) {
    int rlen = rbase + (a < rrem ? 1 : 0);
    int tcur = 0;
    for (int b = 0; b < nth_seg; ++b) {
      int tlen = tbase + (b < trem ? 1 : 0);
      boxes.push_back(Box2D{{rcur, tcur}, {rcur + rlen - 1, tcur + tlen - 1}});
      tcur += tlen;
    }
    rcur += rlen;
  }
  return BoxArray(std::move(boxes));
}

// Remplit un champ scalaire CONSTANT sur les cellules valides locales (ghosts remplis par le solveur).
static void fill_const_local(MultiFab& mf, double val) {
  for (int li = 0; li < mf.local_size(); ++li) {
    const Box2D vb = mf.box(li);
    Array4 a = mf.fab(li).array();
    for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
      for (int i = vb.lo[0]; i <= vb.hi[0]; ++i)
        a(i, j, 0) = val;
  }
}

// Resout la MMS (tenseur arr/art/atr/att) sur un BoxArray donne ; ecrit la solution dans phi_out (memes
// ba/dm -> memes boites locales). Rend le resultat BiCGStab.
static PolarKrylovResult solve_mb(const PolarGeometry& g, const BoxArray& ba, Problem prob, int m,
                                  double arr, double art, double atr, double att, PolarPrecond pc,
                                  MultiFab& phi_out) {
  BCRec bc;
  bc.ylo = bc.yhi = BCType::Periodic;
  if (prob == Problem::Dirichlet) {
    bc.xlo = bc.xhi = BCType::Dirichlet;
    bc.xlo_val = Sf(kRmin);
    bc.xhi_val = Sf(kRmax);
  } else {
    bc.xlo = bc.xhi = BCType::Foextrap;  // Neumann homogene radial -> singulier (jauge)
  }

  PolarTensorKrylovSolver solver(g, ba, bc, pc);

  // Coefficients du tenseur (champs au centre, CONSTANTS ici, alloues sur le MEME layout).
  MultiFab arr_mf(ba, rr_dm(ba), 1, 1), att_mf(ba, rr_dm(ba), 1, 1);
  MultiFab art_mf(ba, rr_dm(ba), 1, 1), atr_mf(ba, rr_dm(ba), 1, 1);
  fill_const_local(arr_mf, arr);
  fill_const_local(att_mf, att);
  const bool cross = (art != 0.0) || (atr != 0.0);
  if (cross) {
    fill_const_local(art_mf, art);
    fill_const_local(atr_mf, atr);
    solver.set_coefficients(&arr_mf, &att_mf, &art_mf, &atr_mf);
  } else {
    solver.set_coefficients(&arr_mf, &att_mf);
  }

  // RHS analytique = f sur les cellules valides locales.
  for (int li = 0; li < solver.rhs().local_size(); ++li) {
    const Box2D vb = solver.rhs().box(li);
    Array4 rhs = solver.rhs().fab(li).array();
    for (int j = vb.lo[1]; j <= vb.hi[1]; ++j)
      for (int i = vb.lo[0]; i <= vb.hi[0]; ++i)
        rhs(i, j, 0) = (prob == Problem::Dirichlet)
                           ? f_tensor(g.r_cell(i), g.theta_cell(j), m, arr, art, atr, att)
                           : f_neu(g.r_cell(i), g.theta_cell(j), m);
  }

  solver.phi().set_val(0.0);
  PolarKrylovResult kr = solver.solve(1e-11, 4000);

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

// Erreur L2 GLOBALE (ponderee volume) vs l'exact. Somme sur toutes les boites locales PUIS all_reduit
// (COLLECTIF : appele sur chaque rang, y compris vide) -> meme valeur globale partout. Ce test cible le
// MULTI-BOX intra-rang (exerce a np=1 : 8 boites / 2x2 tiles sur un rang) ; les reductions restent
// collectives par robustesse si lance sous MPI. Neumann : retire d'abord la moyenne FV globale (jauge).
static double err_l2(const MultiFab& phi, const PolarGeometry& g, Problem prob, int m) {
  const double dr = g.dr(), dth = g.dtheta();
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
  // `ok` declare AVANT le bloc Kokkos-garde : sous ADC_HAS_KOKKOS la paire init/finalize ouvre
  // un bloc { } autour du corps. Le `return ok ? 0 : 1;` final est APRES la fermeture de ce bloc,
  // donc `ok` doit vivre dans la portee de main() pour rester visible dans LES DEUX builds.
  bool ok = true;
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
  {
#endif
    const int me = my_rank();
    // CIBLE : MULTI-BOX intra-rang (a np=1 : 8 boites / 2x2 tiles sur UN rang). Les SOLVES et err_l2 sont
    // COLLECTIFS (appeles sur tous les rangs) -> le test reste correct si lance sous MPI (la parite tient
    // aussi cross-rang, le cross-term multi-box etant alors reparti). Les affichages et le verdict sont
    // GATES sur le rang 0 (les checks restent evalues partout : meme valeurs globales -> meme decision).
    if (me == 0)
      std::printf(
          "=== PARITE MULTI-BOX (plusieurs boites par rang) du Schur polaire tensoriel ===\n"
          "Anneau r in [%.2f, %.2f], theta in [0, 2pi). Cible : plusieurs boites LOCALES (np=1).\n",
          kRmin, kRmax);
    auto chk = [&](bool c, const char* w) {
      if (!c) {
        if (me == 0)
          std::printf("  ECHEC %s\n", w);
        ok = false;
      }
    };
    // Sortie GATEE sur le rang 0 (les checks restent evalues partout : memes scalaires GLOBAUX -> meme
    // decision). Macro plutot que lambda variadique pour eviter -Wformat-security (printf indirect).
#define SAY(...)                \
  do {                          \
    if (me == 0)                \
      std::printf(__VA_ARGS__); \
  } while (0)

    const int nr = 48, nth = 96, m = 3;
    Box2D dom = Box2D::from_extents(nr, nth);
    PolarGeometry g{dom, kRmin, kRmax};

    // Tenseur PLEIN NON SYMETRIQUE (cf. test_polar_tensor_elliptic_mms : defini positif en partie sym).
    const double arr = 1.4, att = 0.8, art = 0.5, atr = -0.3;

    // ---------------------------------------------------------------------------------------------
    // (A) PARITE TENSEUR (TERMES CROISES) THETA-SPLIT (8 boites) vs MONO-BOX, precond RadialLine.
    //     Exerce le COIN du 9-points cross-box (les ghosts diagonaux p(i+-1, j+-1) des termes croises
    //     traversent une frontiere de box theta -> remplis par fill_boundary, coins inclus).
    // ---------------------------------------------------------------------------------------------
    SAY("\n--- (A) Tenseur croise : theta-split (8 boites) vs mono-box [RadialLine] ---\n");
    {
      BoxArray ba_mono(std::vector<Box2D>{dom});
      MultiFab phi_ref(ba_mono, rr_dm(ba_mono), 1, 1);
      PolarKrylovResult kr_ref = solve_mb(g, ba_mono, Problem::Dirichlet, m, arr, art, atr, att,
                                          PolarPrecond::RadialLine, phi_ref);
      const double err_ref = err_l2(phi_ref, g, Problem::Dirichlet, m);
      chk(kr_ref.converged, "A_mono_converge");

      BoxArray ba8 = theta_split(nr, nth, 8);
      MultiFab phi8(ba8, rr_dm(ba8), 1, 1);
      PolarKrylovResult kr8 = solve_mb(g, ba8, Problem::Dirichlet, m, arr, art, atr, att,
                                       PolarPrecond::RadialLine, phi8);
      const double err8 = err_l2(phi8, g, Problem::Dirichlet, m);
      const double derr = std::fabs(err8 - err_ref);
      SAY("  mono : iters=%d err=%.6e | 8 boites theta : iters=%d err=%.6e | d=%.3e\n",
          kr_ref.iters, err_ref, kr8.iters, err8, derr);
      chk(kr8.converged, "A_multibox_converge");
      chk(derr <= 1e-8, "A_multibox_err_matches_mono");  // coin 9-points cross-box correct
    }

    // ---------------------------------------------------------------------------------------------
    // (B) PARITE 2D TILING (coupe r ET theta) avec precond JACOBI vs mono-box (Jacobi aussi). Vrai pavage
    //     2D (2 en r x 2 en theta = 4 boites) : exerce les halos en r, en theta, ET en diagonale. C'est le
    //     chemin "multi-box COMPLET" (Jacobi = par cellule, sans contrainte de layout). Tenseur croise.
    // ---------------------------------------------------------------------------------------------
    SAY("\n--- (B) Pavage 2D (coupe r ET theta, 4 boites) vs mono-box [Jacobi] ---\n");
    {
      BoxArray ba_mono(std::vector<Box2D>{dom});
      MultiFab phi_ref(ba_mono, rr_dm(ba_mono), 1, 1);
      PolarKrylovResult kr_ref = solve_mb(g, ba_mono, Problem::Dirichlet, m, arr, art, atr, att,
                                          PolarPrecond::Jacobi, phi_ref);
      const double err_ref = err_l2(phi_ref, g, Problem::Dirichlet, m);
      chk(kr_ref.converged, "B_mono_jacobi_converge");

      BoxArray tile = tile_2d(nr, nth, 2, 2);  // 4 boites : coupe r ET theta
      MultiFab phit(tile, rr_dm(tile), 1, 1);
      PolarKrylovResult krt =
          solve_mb(g, tile, Problem::Dirichlet, m, arr, art, atr, att, PolarPrecond::Jacobi, phit);
      const double errt = err_l2(phit, g, Problem::Dirichlet, m);
      const double derr = std::fabs(errt - err_ref);
      SAY("  mono : iters=%d err=%.6e | 2x2 tiles : iters=%d err=%.6e | d=%.3e\n", kr_ref.iters,
          err_ref, krt.iters, errt, derr);
      chk(krt.converged, "B_tile2d_converge");
      chk(derr <= 1e-8, "B_tile2d_err_matches_mono");  // halos r/theta/diagonale corrects
    }

    // ---------------------------------------------------------------------------------------------
    // (C) GARDE-FOU DE LAYOUT : un pavage qui COUPE r sous RadialLine DOIT lever (Thomas en r ne franchit
    //     pas une frontiere de box). Resultat CLAIR (pas de faux silencieux). Le repli est Jacobi (cas B).
    // ---------------------------------------------------------------------------------------------
    SAY("\n--- (C) Garde-fou : layout coupant r sous RadialLine -> doit lever ---\n");
    {
      BCRec bc;
      bc.xlo = bc.xhi = BCType::Dirichlet;
      bc.ylo = bc.yhi = BCType::Periodic;
      BoxArray tile = tile_2d(nr, nth, 2, 1);  // coupe r seulement (2 boites en r)
      bool threw = false;
      try {
        PolarTensorKrylovSolver bad(g, tile, bc, PolarPrecond::RadialLine);
        (void)bad;
      } catch (const std::runtime_error&) {
        threw = true;
      }
      SAY("  RadialLine + coupe-r : %s\n", threw ? "throw (attendu)" : "PAS de throw (BUG)");
      chk(threw, "C_radialline_rcut_throws");

      // Le MEME layout coupant r sous JACOBI ne doit PAS lever (sans contrainte de layout).
      bool threw_jac = false;
      try {
        PolarTensorKrylovSolver good(g, tile, bc, PolarPrecond::Jacobi);
        (void)good;
      } catch (const std::runtime_error&) {
        threw_jac = true;
      }
      SAY("  Jacobi + coupe-r     : %s\n", threw_jac ? "throw (BUG)" : "OK (pas de contrainte)");
      chk(!threw_jac, "C_jacobi_rcut_ok");
    }

    // ---------------------------------------------------------------------------------------------
    // (D) PARITE NEUMANN (jauge, project_mean multi-box) theta-split (8 boites) vs mono-box [RadialLine].
    //     L'operateur pure-Neumann est singulier : project_mean accumule sum/vol sur TOUTES les boites
    //     locales puis all_reduit. Multi-box mono-rang : verifie la jauge coherente sur plusieurs boites.
    // ---------------------------------------------------------------------------------------------
    SAY("\n--- (D) Neumann (jauge) : theta-split (8 boites) vs mono-box [RadialLine] ---\n");
    {
      const int mN = 2;
      BoxArray ba_mono(std::vector<Box2D>{dom});
      MultiFab phi_ref(ba_mono, rr_dm(ba_mono), 1, 1);
      PolarKrylovResult kr_ref = solve_mb(g, ba_mono, Problem::Neumann, mN, 1.0, 0.0, 0.0, 1.0,
                                          PolarPrecond::RadialLine, phi_ref);
      const double err_ref = err_l2(phi_ref, g, Problem::Neumann, mN);
      chk(kr_ref.converged, "D_mono_neumann_converge");

      BoxArray ba8 = theta_split(nr, nth, 8);
      MultiFab phi8(ba8, rr_dm(ba8), 1, 1);
      PolarKrylovResult kr8 = solve_mb(g, ba8, Problem::Neumann, mN, 1.0, 0.0, 0.0, 1.0,
                                       PolarPrecond::RadialLine, phi8);
      const double err8 = err_l2(phi8, g, Problem::Neumann, mN);
      const double derr = std::fabs(err8 - err_ref);
      SAY("  mono : iters=%d err=%.6e | 8 boites theta : iters=%d err=%.6e | d=%.3e\n",
          kr_ref.iters, err_ref, kr8.iters, err8, derr);
      chk(kr8.converged, "D_multibox_neumann_converge");
      chk(derr <= 1e-8, "D_multibox_neumann_err_matches_mono");  // jauge multi-box coherente
    }

    SAY("\n=== VERDICT : %s ===\n", ok ? "SUCCESS" : "ECHEC");
    if (ok)
      SAY("OK test_polar_schur_multibox\n");
#if defined(ADC_HAS_KOKKOS)
  }
  Kokkos::finalize();
#endif
  comm_finalize();
  return ok ? 0 : 1;
}
#undef SAY

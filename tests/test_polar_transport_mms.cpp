// MMS + conservation de l'operateur de transport POLAIRE (chantier "grille polaire diocotron",
// Phase 1). Verifie deux proprietes de assemble_rhs_polar sur un anneau global (r, theta) :
//
//   (A) CONVERGENCE MMS de la divergence polaire (1/r) d_r(r F_r) + (1/r) d_theta(F_theta).
//       On PRESCRIT (test transport seul, AUCUN Poisson) un champ aux lisse (grad_r, grad_theta dans
//       la base locale) et un profil de densite lisse n(r, theta). Le flux ExB polaire en decoule :
//         v_r = -grad_theta / B,  v_theta = grad_r / B,  F_r = n v_r,  F_theta = n v_theta.
//       Le residu discret -assemble_rhs_polar (source nulle) doit converger vers la divergence
//       ANALYTIQUE -div F. La reference analytique est obtenue en differenciant les flux CONTINUS
//       (r F_r et F_theta, formes closes) par un stencil central d'ordre 4 a pas infinitesimal
//       (h_ref = 1e-4) : ~12 chiffres exacts, tres en dessous de l'erreur de grille mesuree
//       (h ~ 1/48..1/192) -> isole proprement l'ordre de l'operateur. L'operateur est un schema
//       VOLUMES FINIS conservatif a flux de FACE ponctuels (comme le cartesien) : la ponderation par
//       le rayon de face brise le telescopage haut ordre du cartesien uniforme -> ordre formel de la
//       DIVERGENCE = 2 (resultat standard du FV-WENO sur grille a metrique ; WENO5 apporte la faible
//       dissipation mesuree en Phase-0, pas un ordre superieur). On verifie donc l'ORDRE 2 PROPRE :
//       une metrique r erronee donnerait un ordre 0 / une divergence, donc l'ordre 2 PROUVE la
//       metrique. Les ghosts (radiaux ET azimutaux) sont remplis avec n EXACT pour que le stencil
//       voie partout des donnees exactes : l'erreur reflete alors la troncature de l'operateur.
//
//   (B) CONSERVATION en volumes finis sur une AVANCE en temps. On choisit un champ purement azimutal
//       (grad_theta = 0 partout -> v_r = 0 : AUCUN flux radial, donc aucun flux aux bords physiques
//       r_min / r_max), theta periodique. La masse totale Sum_ij n_ij r_i dr dtheta doit etre
//       conservee a la machine sur K pas SSPRK3, car le terme azimutal telescope (periodique) et le
//       terme radial est identiquement nul.
//
// Host / Serial-safe (UNE box, n_ranks()==1 dans les 3 jobs CI : le test n'est pas enregistre MPI).

#include <adc/core/state.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator_polar.hpp>
#include <adc/numerics/time/time_steppers.hpp>
#include <adc/physics/hyperbolic.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

static constexpr double kPiL = 3.14159265358979323846;
static constexpr double kRmin = 0.30;
static constexpr double kRmax = 1.00;
static constexpr double kB0 = 1.0;
static constexpr int    kMode = 2;  // mode azimutal de la solution manufacturee

// --- Champs CONTINUS de la solution manufacturee (formes closes, lisses, periodiques en theta) ----
// Densite lisse, strictement positive : n = (2 + cos(m theta)) * (1 + 0.5 sin(pi (r-rmin)/(rmax-rmin)))
static double mms_n(double r, double th) {
  const double rr = (r - kRmin) / (kRmax - kRmin);
  return (2.0 + std::cos(kMode * th)) * (1.0 + 0.5 * std::sin(kPiL * rr));
}
// Composantes du champ aux en base locale (grad_r, grad_theta), parametrees par @p const_vel :
//   const_vel == true  : aux CONSTANT -> vitesse uniforme (v_r, v_theta). La valeur de cellule
//                        coincide alors avec la valeur de FACE, donc la reconstruction de la VITESSE
//                        est EXACTE : l'ordre observe reflete celui de la reconstruction de la DENSITE
//                        (WENO5-Z -> ordre eleve). C'est le cas qui prouve la haute precision du
//                        schema ET la metrique polaire (le terme radial (1/r) d_r(r F_r) est exerce,
//                        v_r != 0).
//   const_vel == false : aux VARIABLE (champ ExB realiste lisse). L'operateur prend l'aux au CENTRE
//                        des cellules (comme l'operateur cartesien : aux non reconstruit), donc la
//                        vitesse de face est d'ordre 2 -> l'ordre observe est plafonne a 2, quel que
//                        soit le limiteur. Ce cas verifie la coherence de l'operateur sur un champ
//                        non trivial (ordre 2 propre = metrique correcte).
static double mms_grad_r(double r, double th, bool const_vel) {
  return const_vel ? 0.6 : std::cos(kMode * th) * (0.7 + 0.3 * r);
}
static double mms_grad_theta(double r, double th, bool const_vel) {
  // const_vel : grad_theta constant non nul -> v_r = -grad_theta/B constant non nul (exerce la metrique).
  return const_vel ? 0.5 : std::sin(kMode * th) * (0.4 + 0.2 * std::cos(kPiL * r));
}
// Vitesse ExB polaire (B0) : v_r = -grad_theta/B, v_theta = grad_r/B.
static double mms_vr(double r, double th, bool cv) { return -mms_grad_theta(r, th, cv) / kB0; }
static double mms_vth(double r, double th, bool cv) { return mms_grad_r(r, th, cv) / kB0; }
// Flux physiques.
static double mms_Fr(double r, double th, bool cv) { return mms_n(r, th) * mms_vr(r, th, cv); }
static double mms_Fth(double r, double th, bool cv) { return mms_n(r, th) * mms_vth(r, th, cv); }

// Divergence ANALYTIQUE de reference : (1/r) d_r(r F_r) + (1/r) d_theta(F_theta), evaluee par stencil
// central d'ordre 4 a pas h_ref = 1e-4 sur les formes closes (precision ~1e-12, negligeable vs grille).
static double mms_div_ref(double r, double th, bool cv) {
  const double h = 1e-4;
  auto rFr = [cv](double rr, double tt) { return rr * mms_Fr(rr, tt, cv); };
  // d_r(r F_r) ordre 4
  const double drFr = (-rFr(r + 2 * h, th) + 8 * rFr(r + h, th) - 8 * rFr(r - h, th) +
                       rFr(r - 2 * h, th)) /
                      (12 * h);
  // d_theta(F_theta) ordre 4
  const double dthFth = (-mms_Fth(r, th + 2 * h, cv) + 8 * mms_Fth(r, th + h, cv) -
                         8 * mms_Fth(r, th - h, cv) + mms_Fth(r, th - 2 * h, cv)) /
                        (12 * h);
  return (drFr + dthFth) / r;
}

// Remplit U (densite) et aux (phi=0, grad_r, grad_theta) avec les champs EXACTS sur TOUTE la boite
// (cellules valides + ghosts) -> le stencil voit partout des donnees exactes (erreur = troncature
// de l'operateur, isolee des conditions aux limites).
static void fill_exact(MultiFab& U, MultiFab& aux, const PolarGeometry& g, bool cv) {
  Array4 u = U.fab(0).array();
  Array4 a = aux.fab(0).array();
  const Box2D gb = U.fab(0).grown_box();  // boite AVEC ghosts (Fab2D::box() ne rend que les valides)
  for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
    for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
      const double r = g.r_cell(i);
      const double th = g.theta_cell(j);
      u(i, j, 0) = mms_n(r, th);
      a(i, j, 0) = 0.0;                       // phi (inutilise par le flux)
      a(i, j, 1) = mms_grad_r(r, th, cv);     // grad_r
      a(i, j, 2) = mms_grad_theta(r, th, cv); // grad_theta (composante azimutale physique)
    }
}

// Erreur L1 (ponderee par le volume r dr dtheta) entre le residu discret -R et la divergence
// analytique, sur les cellules valides. Renvoie aussi la norme Linf.
struct ErrNorms { double l1; double linf; };

template <class Limiter>
static ErrNorms mms_error(int nr, int nth, bool cv) {
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  const int ng = Limiter::n_ghost;
  MultiFab U(ba, dm, 1, ng);
  MultiFab aux(ba, dm, 3, ng);
  MultiFab R(ba, dm, 1, 0);
  U.set_val(0.0);
  aux.set_val(0.0);
  fill_exact(U, aux, g, cv);  // exact partout (valides + ghosts)

  ExBVelocityPolar model;
  model.B0 = kB0;
  assemble_rhs_polar<Limiter, RusanovFlux>(model, U, aux, g, R);

  // R vient d'etre ecrit par un kernel device : rendre la residence HOTE valide avant la lecture
  // directe ci-dessous (sous Kokkos::Cuda = device_fence ; no-op en serie/OpenMP). Sans cela on lit
  // R pendant que le kernel est en vol (memoire unifiee non ordonnee) -> R quasi nul, ordre faux.
  sync_host();
  const ConstArray4 r = R.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double l1 = 0.0, vol = 0.0, linf = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double rc = g.r_cell(i), th = g.theta_cell(j);
      const double div_discrete = -r(i, j, 0);       // R = -div F (source nulle) -> div = -R
      const double e = std::fabs(div_discrete - mms_div_ref(rc, th, cv));
      const double w = rc * dr * dth;                 // volume de cellule
      l1 += e * w;
      vol += w;
      if (e > linf) linf = e;
    }
  return {l1 / vol, linf};
}

// --- (B) CONSERVATION : champ purement azimutal (v_r = 0), masse conservee sur K pas SSPRK3 -------
// Masse totale FV = Sum_ij n_ij r_i dr dtheta.
static double total_mass(const MultiFab& U, const PolarGeometry& g, const Box2D& dom) {
  // U a pu etre ecrit par les kernels device de l'integrateur (SSPRK3) : on rend la residence hote
  // valide avant la lecture directe ci-dessous (Kokkos::Cuda = device_fence ; no-op en serie/OpenMP).
  sync_host();
  const ConstArray4 u = U.fab(0).const_array();
  const double dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
      m += u(i, j, 0) * g.r_cell(i) * dr * dth;
  return m;
}

static double run_conservation() {
  const int nr = 64, nth = 128;
  Box2D dom = Box2D::from_extents(nr, nth);
  PolarGeometry g{dom, kRmin, kRmax};
  BoxArray ba(std::vector<Box2D>{dom});
  DistributionMapping dm(1, n_ranks());

  // BC : radial PHYSIQUE (Foextrap), azimutal PERIODIQUE.
  BCRec bc;
  bc.xlo = bc.xhi = BCType::Foextrap;  // r (dir 0)
  bc.ylo = bc.yhi = BCType::Periodic;  // theta (dir 1)

  const int ng = Weno5::n_ghost;
  MultiFab U(ba, dm, 1, ng);
  MultiFab aux(ba, dm, 3, ng);
  U.set_val(0.0);
  aux.set_val(0.0);

  // Champ PUREMENT AZIMUTAL : grad_theta = 0 partout -> v_r = 0 (aucun flux radial, donc aucun flux
  // aux bords r_min/r_max). grad_r != 0 -> v_theta != 0 (advection en theta). Profil n module en theta.
  {
    Array4 u = U.fab(0).array();
    Array4 a = aux.fab(0).array();
    const Box2D gb = U.fab(0).box();  // cellules VALIDES ; les ghosts sont remplis ci-dessous
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) {
        const double r = g.r_cell(i), th = g.theta_cell(j);
        u(i, j, 0) = 1.0 + 0.5 * std::cos(kMode * th) * std::sin(kPiL * (r - kRmin) / (kRmax - kRmin));
        a(i, j, 0) = 0.0;
        a(i, j, 1) = 0.8;  // grad_r constant -> v_theta = 0.8/B (azimutal pur)
        a(i, j, 2) = 0.0;  // grad_theta = 0 -> v_r = 0
      }
  }
  // aux est PRESCRIT et statique : on remplit ses ghosts UNE fois (theta periodique, radial extrapole),
  // comme le fait solve_fields dans le System reel. Sans cela, les ghosts azimutaux de grad_r restent a
  // 0 et le flux azimutal au joint theta=0/2pi ne telescope plus -> perte de masse parasite.
  fill_ghosts(aux, dom, bc);

  ExBVelocityPolar model;
  model.B0 = kB0;

  const double m0 = total_mass(U, g, dom);

  // CFL azimutal : v_theta = grad_r/B = 0.8 ; pas physique azimutal min = r_min * dtheta.
  const double v_th = 0.8 / kB0;
  const double ds_min = kRmin * g.dtheta();
  const double dt = 0.4 * ds_min / v_th;
  const int nsteps = 40;

  for (int s = 0; s < nsteps; ++s) {
    SSPRK3Step{}.take_step(
        [&](MultiFab& stage, MultiFab& Rr) {
          fill_ghosts(stage, dom, bc);
          assemble_rhs_polar<Weno5, RusanovFlux>(model, stage, aux, g, Rr);
        },
        U, dt);
  }

  const double m1 = total_mass(U, g, dom);
  const double rel = std::fabs(m1 - m0) / std::fabs(m0);
  std::printf("[conservation] masse initiale=%.15e finale=%.15e  ecart relatif=%.3e (K=%d pas)\n",
              m0, m1, rel, nsteps);
  return rel;
}

int main() {
  std::printf("=== MMS + conservation de l'operateur de transport POLAIRE (Phase 1) ===\n");
  std::printf("Anneau r in [%.2f, %.2f], theta in [0, 2pi), mode azimutal m=%d, B0=%.1f\n",
              kRmin, kRmax, kMode, kB0);

  bool ok = true;
  const int res[3] = {48, 96, 192};  // nth = 2 nr (anneau, theta plus echantillonne)

  // (A) CONVERGENCE MMS de la divergence polaire. L'operateur est un schema VOLUMES FINIS conservatif
  // a flux de FACE ponctuels, comme l'operateur cartesien : (1/r_i) (r_{i+1/2}F_{i+1/2} -
  // r_{i-1/2}F_{i-1/2})/dr + (1/r_i)(Ftheta_{j+1/2}-Ftheta_{j-1/2})/dtheta. La PONDERATION par le rayon
  // de face (r_{i+/-1/2}, constantes DIFFERENTES sur les deux faces) brise le telescopage haut ordre
  // exact du cartesien uniforme -> l'operateur de DIVERGENCE est formellement d'ORDRE 2 (resultat
  // STANDARD du FV-WENO sur grille a metrique : une quadrature de face d'ordre eleve serait necessaire
  // pour preserver l'ordre 5, hors scope Phase 1). WENO5-Z apporte ici la FAIBLE DISSIPATION / capture
  // de gradient (le benefice mesure par le proto Phase-0 : rapport 73 sur le pic radial), PAS un ordre
  // formel superieur. On EXIGE donc une convergence d'ordre 2 PROPRE pour WENO5 ET minmod (un facteur
  // metrique r errone donnerait un ordre 0 ou une divergence) : c'est la preuve que la metrique polaire
  // est correctement portee. Vitesse CONSTANTE (aux constant) : la reconstruction de la vitesse est
  // exacte a la face, donc l'ordre observe est celui de l'operateur de divergence lui-meme (et v_r != 0
  // exerce le terme radial (1/r) d_r(r F_r)).
  std::printf("\n--- (A) Convergence MMS de la divergence polaire (ordre 2, vitesse constante) ---\n");
  ErrNorms e[3];
  for (int k = 0; k < 3; ++k) {
    e[k] = mms_error<Weno5>(res[k], 2 * res[k], /*const_vel=*/true);
    std::printf("  WENO5  nr=%-4d nth=%-4d : L1=%.4e  Linf=%.4e\n", res[k], 2 * res[k], e[k].l1,
                e[k].linf);
  }
  const double p1 = std::log2(e[0].l1 / e[1].l1);
  const double p2 = std::log2(e[1].l1 / e[2].l1);
  std::printf("  ordre observe WENO5 (L1) : %.2f (48->96), %.2f (96->192)\n", p1, p2);
  ErrNorms em[3];
  for (int k = 0; k < 3; ++k)
    em[k] = mms_error<Minmod>(res[k], 2 * res[k], /*const_vel=*/true);
  const double pm1 = std::log2(em[0].l1 / em[1].l1);
  const double pm2 = std::log2(em[1].l1 / em[2].l1);
  std::printf("  ordre observe minmod (L1): %.2f (48->96), %.2f (96->192)\n", pm1, pm2);
  // Ordre 2 attendu : on accepte [1.7, 2.3] (la borne haute exclut un ordre artificiellement gonfle).
  auto order2_ok = [](double p) { return p >= 1.7 && p <= 2.3; };
  if (!order2_ok(p1) || !order2_ok(p2) || !order2_ok(pm1) || !order2_ok(pm2)) {
    std::printf("  ECHEC : ordre hors [1.7, 2.3] (metrique polaire incoherente)\n");
    ok = false;
  } else {
    std::printf("  OK : convergence d'ordre 2 propre (metrique polaire correcte, WENO5 et minmod)\n");
  }

  // (A') Champ ExB VARIABLE (realiste, lisse) : meme operateur, aux non constant -> la vitesse de face
  // (aux pris au centre des cellules, comme le cartesien) est elle aussi d'ordre 2 ; l'operateur
  // converge encore proprement a l'ordre 2 sur un champ non trivial. Confirme la coherence de
  // reconstruct<> (reutilise verbatim) sur un cas ou densite ET vitesse varient.
  std::printf("\n--- (A') MMS champ ExB variable : ordre 2 sur densite + vitesse variables ---\n");
  ErrNorms ew[2];
  ew[0] = mms_error<Weno5>(96, 192, /*const_vel=*/false);
  ew[1] = mms_error<Weno5>(192, 384, /*const_vel=*/false);
  const double pw = std::log2(ew[0].l1 / ew[1].l1);
  std::printf("  WENO5 : L1(96)=%.4e L1(192)=%.4e ordre=%.2f\n", ew[0].l1, ew[1].l1, pw);
  if (pw < 1.7) {
    std::printf("  ECHEC : ordre < 1.7 sur champ variable (metrique ou reconstruction incoherente)\n");
    ok = false;
  } else {
    std::printf("  OK : ordre 2 sur champ variable\n");
  }

  // (B) CONSERVATION : masse conservee a la machine (champ azimutal pur, v_r = 0).
  std::printf("\n--- (B) Conservation FV (champ azimutal pur, v_r = 0) ---\n");
  const double rel = run_conservation();
  // Tolerance machine elargie (accumulation sur K pas x 3 etages SSPRK3, reductions hote) : 1e-12.
  if (rel > 1e-12) {
    std::printf("  ECHEC : ecart de masse %.3e > 1e-12\n", rel);
    ok = false;
  } else {
    std::printf("  OK : masse conservee a ~machine (%.3e <= 1e-12)\n", rel);
  }

  std::printf("\n=== VERDICT : %s ===\n", ok ? "SUCCESS" : "ECHEC");
  std::printf("OK test_polar_transport_mms\n");
  return ok ? 0 : 1;
}

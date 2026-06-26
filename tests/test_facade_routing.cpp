// Chantier T5-PR3 : CABLAGE du transport disque (staircase + cut-cell EB) dans System::step.
//
// CONTEXTE (cf. docs/HOFFART_FIDELITY.md ligne 39, verrou "bords d'anneau cartesiens" ; le footgun T2 :
// set_disc_domain materialisait un masque MAIS System::step ne le consultait jamais -> le disque etait
// INERTE). Ce PR aiguille l'avance de transport de step() vers l'operateur disque selon un MODE explicite
// (none | staircase | cutcell), porte par set_disc_domain(mode=) / set_geometry_mode et lu par le stepper.
//
// On valide (vraies assertions, pas de no-op) :
//   (a) NO-DISC PAR DEFAUT : un pas avec set_disc_domain(mode='none') est BYTE-IDENTIQUE a un pas SANS
//       set_disc_domain (le masque est materialise mais le transport l'ignore) -> diff EXACTEMENT 0.
//   (b) ROUTING-LIVE (staircase) : mode='staircase' produit un etat DIFFERENT du carre sur le MEME init
//       (max|diff| > 0 : le routage N'EST PAS inerte) ET la masse sur les cellules ACTIVES du disque est
//       conservee a la machine (aucun flux ne franchit la frontiere du masque) -> propre au schema masque.
//   (c) CUTCELL : mode='cutcell' tourne, etat FINI partout (aucun NaN/Inf), DIFFERENT du carre ; et sur
//       un disque ENGLOBANT (rayon > diagonale, aucune cellule coupee) un pas est BIT-IDENTIQUE au carre.
//   (d) MODE HONORE SOUS LIE ET STRANG : le mode disque change l'etat dans les DEUX schemas de splitting
//       (set_time_scheme 'lie' / 'strang') -> les deux chemins (step / step_strang) consultent le mode.
//
// Modele : transport scalaire ExB (add_block transport='exb', source='none', elliptic='charge') -- le
// transport DIOCOTRON de production. La vitesse derive de grad phi (Poisson sur la densite) : champ a
// divergence nulle -> la masse est conservee par les schemas masque / EB. Compile python/system.cpp.

#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

// Densite initiale : anneau lisse (recouvre le disque interieur), perturbe en azimut pour casser la
// symetrie -> grad phi non trivial -> vitesse ExB non nulle. n*n row-major (j lent, i rapide).
std::vector<double> ring_density(int n, double L) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n, 1e-3);
  const double cx = 0.5 * L, cy = 0.5 * L;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) * L / n, y = (j + 0.5) * L / n;
      const double r = std::hypot(x - cx, y - cy);
      // anneau gaussien centre sur r0 = 0.18 L, module en sin(3 theta) (perturbation azimutale l=3).
      const double r0 = 0.18 * L, w = 0.05 * L;
      const double th = std::atan2(y - cy, x - cx);
      const double g = std::exp(-((r - r0) * (r - r0)) / (2 * w * w));
      rho[static_cast<std::size_t>(j) * n + i] = 1e-3 + g * (1.0 + 0.3 * std::sin(3 * th));
    }
  return rho;
}

// Construit un System scalaire ExB diocotron pret a stepper. Le disque/mode est pose par l'appelant.
void build_exb(System& s, double R_wall) {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = 1.0;
  spec.B0 = 1.0;
  // minmod + rusanov + SSPRK2 explicite : meme schema spatial dans tous les modes (seul le residu de
  // transport est aiguille par le mode -- c'est ce qu'on veut isoler).
  s.add_block("n", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
  // Poisson sur la densite de charge, mur conducteur circulaire concentrique (comme le diocotron) :
  // donne un phi non trivial -> vitesse ExB. Le mur elliptique et le disque de transport partagent le
  // meme centre (L/2, L/2) et la meme convention de level set.
  s.set_poisson("charge_density", "geometric_mg", "dirichlet", "circle", R_wall, 1.0);
}

// max|diff| composante a composante entre deux champs de meme taille.
double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}

bool all_finite(const std::vector<double>& a) {
  for (double v : a)
    if (!std::isfinite(v))
      return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 48;
  const double L = 1.0;
  const double R_wall = 0.45 * L;  // mur conducteur de Poisson (rayon < L/2)
  const double R_disc =
      0.30 * L;  // disque de transport (plus petit : de vraies cellules inactives)
  const double cx = 0.5 * L, cy = 0.5 * L;
  const double dt = 2e-4;  // pas court, transport ExB sous-CFL
  const int n_steps = 12;
  const std::vector<double> rho0 = ring_density(n, L);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // ----------------------------------------------------------------------
  // (a) NO-DISC PAR DEFAUT : mode='none' (disque materialise) == jamais set_disc_domain (byte a byte).
  // ----------------------------------------------------------------------
  std::vector<double>
      ref_state;  // etat de reference (chemin plein cartesien), reutilise par (b)/(c)/(d)
  {
    System base(SystemConfig{n, L, false});
    build_exb(base, R_wall);
    base.set_density("n", rho0);
    for (int k = 0; k < n_steps; ++k)
      base.step(dt);
    ref_state = base.get_state("n");

    System none(SystemConfig{n, L, false});
    build_exb(none, R_wall);
    none.set_density("n", rho0);
    none.set_disc_domain(cx, cy, R_disc, "none");  // disque pose, mode none : doit rester inerte
    for (int k = 0; k < n_steps; ++k)
      none.step(dt);
    const std::vector<double> none_state = none.get_state("n");

    const double d = max_abs_diff(ref_state, none_state);
    std::printf("  (a) mode='none' vs sans disque : max|diff| = %.3e (attendu 0)\n", d);
    // Egalite BYTE A BYTE : mode none emprunte exactement assemble_rhs, le disque materialise n'a AUCUN
    // effet sur le transport. Pas une tolerance -- l'invariant "inerte par defaut".
    chk(d == 0.0,
        "(a) mode='none' BIT-IDENTIQUE au chemin sans disque (routage inerte sauf opt-in)");
    chk(all_finite(ref_state) && ref_state.size() == static_cast<std::size_t>(n) * n,
        "(a) etat de reference fini et de taille n*n (le pas plein a bien tourne)");
  }

  // ----------------------------------------------------------------------
  // (b) ROUTING-LIVE (staircase) : etat DIFFERENT du carre + masse active conservee a la machine.
  // ----------------------------------------------------------------------
  {
    System sc(SystemConfig{n, L, false});
    build_exb(sc, R_wall);
    sc.set_density("n", rho0);
    sc.set_disc_domain(cx, cy, R_disc, "staircase");

    // Masse initiale sur les cellules ACTIVES (masque 0/1 du System) AVANT les pas.
    const std::vector<double> mask = sc.disc_mask();  // (ny, nx) row-major, 1.0 actif
    const std::vector<double> dens0 = sc.density("n");
    const double dx2 = (L / n) * (L / n);
    int n_active = 0, n_inactive = 0;
    double mass0 = 0.0;
    for (std::size_t k = 0; k < mask.size(); ++k) {
      if (mask[k] >= 0.5) {
        ++n_active;
        mass0 += dens0[k] * dx2;
      } else
        ++n_inactive;
    }
    chk(n_active > 0 && n_inactive > 0,
        "(b) le disque partitionne la grille en cellules actives ET inactives (test non vide)");

    for (int k = 0; k < n_steps; ++k)
      sc.step(dt);
    const std::vector<double> sc_state = sc.get_state("n");

    // Masse active APRES les pas (meme masque : le disque est statique).
    const std::vector<double> dens1 = sc.density("n");
    double mass1 = 0.0;
    for (std::size_t k = 0; k < mask.size(); ++k)
      if (mask[k] >= 0.5)
        mass1 += dens1[k] * dx2;

    const double d_vs_square = max_abs_diff(ref_state, sc_state);
    const double rel_drift = std::fabs(mass1 - mass0) / std::fabs(mass0);
    std::printf(
        "  (b) staircase vs carre : max|diff| = %.3e (attendu > 0) ; masse active drift = %.3e\n",
        d_vs_square, rel_drift);

    // Le routage N'EST PAS inerte : l'operateur masque ferme les faces a la frontiere du disque, donc
    // l'etat diverge du chemin plein cartesien. C'est la preuve directe contre le footgun T2.
    chk(d_vs_square > 1e-10,
        "(b) staircase produit un etat DIFFERENT du carre (le transport disque est REELLEMENT "
        "cable)");
    chk(all_finite(sc_state), "(b) etat staircase fini partout (aucun NaN/Inf)");
    // La masse sur les cellules actives est conservee a la machine (flux normal nul aux faces
    // active/inactive). Borne juste au-dessus du bruit flottant des sommes telescopiques de flux.
    chk(rel_drift < 1e-12,
        "(b) masse sur les cellules actives conservee a la machine (schema masque conservatif)");
  }

  // ----------------------------------------------------------------------
  // (c) CUTCELL : tourne, FINI partout, DIFFERENT du carre ; disque ENGLOBANT == carre (bit a bit).
  // ----------------------------------------------------------------------
  {
    // (c1) disque coupant : etat fini + different du carre.
    System cc(SystemConfig{n, L, false});
    build_exb(cc, R_wall);
    cc.set_density("n", rho0);
    cc.set_disc_domain(cx, cy, R_disc, "cutcell");
    for (int k = 0; k < n_steps; ++k)
      cc.step(dt);
    const std::vector<double> cc_state = cc.get_state("n");
    const double d_vs_square = max_abs_diff(ref_state, cc_state);
    std::printf("  (c1) cutcell vs carre : max|diff| = %.3e (attendu > 0) ; fini = %d\n",
                d_vs_square, all_finite(cc_state) ? 1 : 0);
    chk(all_finite(cc_state),
        "(c1) etat cutcell fini partout (clamp small-cell -> pas de NaN/Inf)");
    chk(d_vs_square > 1e-10,
        "(c1) cutcell produit un etat DIFFERENT du carre (transport EB cable)");

    // (c2) disque ENGLOBANT (rayon > demi-diagonale) : TOUTE cellule est active, AUCUNE face coupee ->
    // assemble_rhs_eb == assemble_rhs (kappa=1, alpha=1 partout, cf. test_eb_transport bit-identite).
    // Un pas cutcell doit alors etre BIT-IDENTIQUE au pas carre sur le meme init.
    const double R_big = 10.0 * L;         // englobe largement la boite
    System sq(SystemConfig{n, L, false});  // reference 1 pas plein
    build_exb(sq, R_wall);
    sq.set_density("n", rho0);
    sq.step(dt);
    const std::vector<double> sq1 = sq.get_state("n");

    System eb(SystemConfig{n, L, false});
    build_exb(eb, R_wall);
    eb.set_density("n", rho0);
    eb.set_disc_domain(cx, cy, R_big, "cutcell");
    eb.step(dt);
    const std::vector<double> eb1 = eb.get_state("n");

    const double d_enclosing = max_abs_diff(sq1, eb1);
    std::printf("  (c2) cutcell disque englobant vs carre (1 pas) : max|diff| = %.3e (attendu 0)\n",
                d_enclosing);
    chk(d_enclosing == 0.0,
        "(c2) cutcell sans coupe BIT-IDENTIQUE au carre (kappa=1, alpha=1 partout)");
  }

  // ----------------------------------------------------------------------
  // (d) MODE HONORE SOUS LIE ET STRANG : le mode disque change l'etat dans les deux splittings.
  // ----------------------------------------------------------------------
  {
    auto run = [&](const char* scheme, const char* mode) {
      System s(SystemConfig{n, L, false});
      build_exb(s, R_wall);
      s.set_density("n", rho0);
      s.set_time_scheme(scheme);
      if (std::string(mode) != "none")
        s.set_disc_domain(cx, cy, R_disc, mode);
      for (int k = 0; k < n_steps; ++k)
        s.step(dt);
      return s.get_state("n");
    };

    // LIE : staircase != none. (Couvert aussi par (b), mais on l'asserte sous le tag 'lie' explicite.)
    const std::vector<double> lie_none = run("lie", "none");
    const std::vector<double> lie_sc = run("lie", "staircase");
    const double d_lie = max_abs_diff(lie_none, lie_sc);
    std::printf("  (d) LIE : staircase vs none = %.3e (attendu > 0)\n", d_lie);
    chk(d_lie > 1e-10, "(d) sous LIE, le mode disque change l'etat (step consulte le mode)");

    // STRANG : meme exigence sur le chemin step_strang (H(dt/2) S(dt) H(dt/2)).
    const std::vector<double> str_none = run("strang", "none");
    const std::vector<double> str_sc = run("strang", "staircase");
    const double d_str = max_abs_diff(str_none, str_sc);
    std::printf("  (d) STRANG : staircase vs none = %.3e (attendu > 0)\n", d_str);
    chk(all_finite(str_none) && all_finite(str_sc), "(d) etats Strang finis (les deux modes)");
    chk(d_str > 1e-10,
        "(d) sous STRANG, le mode disque change l'etat (step_strang consulte le mode)");
    // Et la cutcell aussi est honoree sous Strang (chemin step_strang -> advance_transport_half EB).
    const std::vector<double> str_cc = run("strang", "cutcell");
    chk(all_finite(str_cc) && max_abs_diff(str_none, str_cc) > 1e-10,
        "(d) sous STRANG, le mode cutcell est aussi cable (fini + different de none)");
  }

  if (fails == 0)
    std::printf("OK test_facade_routing\n");
  return fails == 0 ? 0 : 1;
}

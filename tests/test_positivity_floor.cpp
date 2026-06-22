// LIMITEUR DE POSITIVITE Zhang-Shu (ADC-76) : plancher de densite des etats de face reconstruits,
// cable a travers reconstruct_pp / les kernels de flux / make_block (pos_floor). Motivation : au saut
// top-hat contraste 1e6 du diocotron Hoffart (adc_cases ADC-62/ADC-74), WENO5 reconstruit une densite
// de face NEGATIVE -> 1/rho et la source Lorentz detonent -> NaN. Le scaling conservatif de l'etat de
// face vers la moyenne de cellule garantit rho_face >= floor sans toucher la moyenne.
//
// Quatre verifications :
//  (1) MECANIQUE zhang_shu_scale (unitaire, deterministe) : REPLI A L'ORDRE 1 LOCAL (etat de face =
//      moyenne de la cellule source quand la face viole le plancher -- la variante theta-scaling
//      colineaire du papier original fabrique des vitesses de face m/rho divergentes au bord du
//      quasi-vide, cf. doc de zhang_shu_scale), inactif si floor <= 0 / face deja >= floor.
//  (2) NO-DEFAULT-CHANGE : assemble_rhs / make_block avec pos_floor = 0 est BIT-IDENTIQUE au chemin
//      historique sur un top-hat raide (le court-circuit ne change pas un bit).
//  (3) EFFET + GARANTIE DE FACE : sur le top-hat 1e6, WENO5 nu reconstruit une face rho < floor ;
//      reconstruct_pp ramene TOUTES les faces a rho >= floor (ou a la moyenne si elle est sous le
//      plancher), et le residu assemble_rhs reste conservatif (somme de masse nulle en periodique).
//  (4) REJET CLAIR : un modele sans role Density (AdvectionDiffusion scalaire) + pos_floor > 0 ->
//      runtime_error explicite (jamais un scaling muet d'une composante arbitraire).
#include <adc/validation/physics/advection_diffusion.hpp>
#include <adc/physics/euler.hpp>
#include <adc/runtime/builders/block_builder.hpp>

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace adc;

// Euler SANS terme source : le modele physique Euler n'expose pas source() (make_block l'exige).
// Forward minimal flux / vitesse d'onde + INTROSPECTION (positivity_comp lit conservative_vars,
// role Density en composante 0). Pendant local du SourceFreeModel (qui exige elliptic_rhs).
struct EulerNoSrc {
  using State = Euler::State;
  using Aux = Euler::Aux;
  static constexpr int n_vars = Euler::n_vars;
  Euler e{};
  Real gamma = Real(1.4);
  ADC_HD State flux(const State& u, const Aux& a, int dir) const { return e.flux(u, a, dir); }
  ADC_HD Real max_wave_speed(const State& u, const Aux& a, int dir) const {
    return e.max_wave_speed(u, a, dir);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{}; }
  static VariableSet conservative_vars() { return Euler::conservative_vars(); }
  static VariableSet primitive_vars() { return Euler::primitive_vars(); }
};

static constexpr double kRhoMin = 1e-6, kRhoMax = 1.0;  // contraste 1e6 du papier Hoffart

// Top-hat raide en x ADVECTE (u = 1) : rho = kRhoMax dans la bande centrale, kRhoMin ailleurs ;
// pression uniforme. Sur le profil STATIQUE, weno5z ne sous-shoote pas (ses poids non lineaires
// retombent sur le stencil lisse d'un cote du saut) ; le sous-shoot apparait sur le profil EVOLUE
// (oscillations residuelles au contact advecte) -- exactement le mecanisme du run Hoffart (la
// densite passait negative au bord d'anneau apres quelques pas, pas a t=0).
static void init_tophat(MultiFab& U, const Box2D& dom, const EulerNoSrc& m, bool spike = true) {
  const int ilo = dom.hi[0] / 3, ihi = 2 * dom.hi[0] / 3;
  // p UNIFORME MINUSCULE (regime quasi froid, comme le diocotron Hoffart cs2 = 1e-4) : la vitesse
  // du son au FOND c = sqrt(gamma p / kRhoMin) reste O(1) (p = 1e-6 -> c ~ 1.2), sinon le pas CFL
  // s'effondre et le test passe a cote du mecanisme (advection du contact, pas l'acoustique).
  const Real p0 = Real(1e-6);
  // ECHARDE oscillante non monotone a x ~ 3/4 : rho = [0.8, 0.5, 1e-6, 1.0, 1e-6]. Sur un saut
  // top-hat MONOTONE statique, weno5z ne sous-shoote pas (ses poids retombent sur le stencil lisse) ;
  // sur ce motif oscillant -- le genre de profil que la dynamique du diocotron cree au bord d'anneau
  // (cf. la sonde ADC-62 : rho < 0 des le pas 1 sur l'anneau perturbe) -- la face+ de la cellule de
  // fond reconstruit ~ -0.22 (sous-shoot deterministe, verifie sur l'implementation weno5z exacte).
  const int ks = spike ? 3 * dom.hi[0] / 4 : -100;
  for (int li = 0; li < U.local_size(); ++li) {
    Array4 a = U.fab(li).array();
    for_each_cell(U.box(li), [a, ilo, ihi, ks, m, p0](int i, int j) {
      Real rho = (i >= ilo && i <= ihi) ? Real(kRhoMax) : Real(kRhoMin);
      if (i == ks)
        rho = Real(0.8);
      else if (i == ks + 1)
        rho = Real(0.5);
      else if (i == ks + 2)
        rho = Real(kRhoMin);
      else if (i == ks + 3)
        rho = Real(1.0);
      else if (i == ks + 4)
        rho = Real(kRhoMin);
      a(i, j, 0) = rho;
      a(i, j, 1) = rho * Real(1.0);  // u = 1 : advection du contact
      a(i, j, 2) = Real(0);
      a(i, j, 3) = p0 / (m.gamma - Real(1)) + Real(0.5) * rho;  // E = p/(gamma-1) + cinetique
    });
  }
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

  std::printf("=== LIMITEUR DE POSITIVITE Zhang-Shu (pos_floor, ADC-76) ===\n");

  const int n = 48;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);
  const EulerNoSrc model{};
  const GridContext ctx{dom, bc, geom, &aux};

  const int ng = block_n_ghost("weno5");
  MultiFab U(ba, dm, EulerNoSrc::n_vars, ng);
  init_tophat(U, dom, model);
  fill_ghosts(U, dom, bc);

  // ----------------------------------------------------------------------------------------------
  // (1) MECANIQUE zhang_shu_scale : etat de face fabrique a la main contre la moyenne de (i, j).
  // ----------------------------------------------------------------------------------------------
  {
    sync_host();
    const ConstArray4 u = U.fab(0).const_array();
    const int i = dom.hi[0] / 3, j = n / 2;  // cellule de la bande haute (rho = 1)
    const Real floor = Real(1e-8);
    // face fabriquee : rho NEGATIF (comme un sous-shoot WENO au saut), moments arbitraires
    typename EulerNoSrc::State s{};
    s[0] = Real(-2e-3);
    s[1] = Real(0.3);
    s[2] = Real(-0.1);
    s[3] = u(i, j, 3) + Real(0.5);
    const typename EulerNoSrc::State s0 = s;
    zhang_shu_scale<EulerNoSrc>(s, u, i, j, floor, 0);
    for (int c = 0; c < EulerNoSrc::n_vars; ++c)  // REPLI ordre 1 : etat de face = moyenne ENTIERE
      chk(s[c] == u(i, j, c), "1_repli_face_egale_moyenne");
    // vitesse de face BORNEE par construction : v_face = m_face/rho_face = v_cellule
    chk(std::abs(s[1] / s[0] - u(i, j, 1) / u(i, j, 0)) < 1e-14, "1_vitesse_face_bornee");
    // floor <= 0 : INACTIF strict
    typename EulerNoSrc::State t = s0;
    zhang_shu_scale<EulerNoSrc>(t, u, i, j, Real(0), 0);
    chk(t[0] == s0[0] && t[3] == s0[3], "1_floor0_inactif");
    // face deja au-dessus du plancher : INACTIF strict
    typename EulerNoSrc::State w{};
    w[0] = Real(0.5);
    w[1] = Real(1);
    w[2] = Real(2);
    w[3] = Real(3);
    const typename EulerNoSrc::State w0 = w;
    zhang_shu_scale<EulerNoSrc>(w, u, i, j, floor, 0);
    chk(w[0] == w0[0] && w[1] == w0[1], "1_face_positive_inactif");
    // moyenne elle-meme sous le plancher : repli identique (face = moyenne, pas de masse creee)
    const int ilow = 1;  // cellule du fond (rho = 1e-6 < floor 1e-5)
    typename EulerNoSrc::State v{};
    v[0] = Real(-1e-7);
    v[1] = Real(0.1);
    zhang_shu_scale<EulerNoSrc>(v, u, ilow, j, Real(1e-5), 0);
    chk(v[0] == u(ilow, j, 0) && v[1] == u(ilow, j, 1), "1_moyenne_sous_plancher_pente_nulle");
    if (me == 0)
      std::printf("(1) mecanique : repli ordre 1 (face = moyenne), plancher %.1e\n",
                  static_cast<double>(floor));
  }

  // ----------------------------------------------------------------------------------------------
  // (2) NO-DEFAULT-CHANGE : pos_floor = 0 -> residu make_block BIT-IDENTIQUE a l'historique.
  // ----------------------------------------------------------------------------------------------
  {
    MultiFab R0(ba, dm, EulerNoSrc::n_vars, 0), R1(ba, dm, EulerNoSrc::n_vars, 0);
    BlockClosures c0 = make_block(model, "weno5", "rusanov", ctx, false, false);
    BlockClosures c1 = make_block(model, "weno5", "rusanov", ctx, false, false, "ssprk2", {}, {},
                                  nullptr, Real(0));
    c0.rhs_into(U, R0);
    c1.rhs_into(U, R1);
    sync_host();
    double dmax = 0;
    for (int li = 0; li < R0.local_size(); ++li) {
      const ConstArray4 a = R0.fab(li).const_array(), b = R1.fab(li).const_array();
      const Box2D v = R0.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < EulerNoSrc::n_vars; ++c)
            dmax = std::max(dmax, std::abs(static_cast<double>(a(i, j, c)) - b(i, j, c)));
    }
    dmax = all_reduce_max(dmax);
    if (me == 0)
      std::printf("(2) no-default-change : max|R(floor=0) - R(historique)| = %.3e\n", dmax);
    chk(dmax == 0.0, "2_floor0_bit_identique");
  }

  // ----------------------------------------------------------------------------------------------
  // (3) EFFET + GARANTIE DE FACE sur le top-hat 1e6 ADVECTE (weno5) : le profil EVOLUE sous-shoote.
  // ----------------------------------------------------------------------------------------------
  {
    const Real floor = Real(1e-8);
    // SCAN STATIQUE sur l'IC a echarde : la face+ de la cellule de fond de l'echarde reconstruit
    // ~ -0.22 en weno5z nu (sous-shoot deterministe) ; reconstruct_pp doit la ramener >= floor.
    sync_host();
    const ConstArray4 u = U.fab(0).const_array();
    const Weno5 lim{};
    Real worst_raw = Real(1e30), worst_pp = Real(1e30);
    int n_under_raw = 0, n_under_pp = 0;
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0] + ng; i <= dom.hi[0] - ng; ++i)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          const auto raw = reconstruct<EulerNoSrc>(model, u, i, j, 0, Real(sgn), lim, false);
          const auto pp =
              reconstruct_pp<EulerNoSrc>(model, u, i, j, 0, Real(sgn), lim, false, floor, 0);
          worst_raw = std::min(worst_raw, raw[0]);
          worst_pp = std::min(worst_pp, pp[0]);
          if (raw[0] < floor)
            ++n_under_raw;
          // garantie : rho_face >= floor (a 1 ulp pres : le scaling rend EXACTEMENT floor en
          // arithmetique exacte, l'arrondi flottant peut laisser floor - 1 ulp) OU la moyenne
          // elle-meme est <= floor (pente nulle).
          if (pp[0] < floor * Real(0.999999) && u(i, j, 0) > floor)
            ++n_under_pp;
        }
    if (me == 0)
      std::printf(
          "(3) top-hat 1e6 + echarde, weno5 : min rho_face NU = %.3e (%d faces < floor) ; "
          "PP = %.3e (%d violations)\n",
          static_cast<double>(worst_raw), n_under_raw, static_cast<double>(worst_pp), n_under_pp);
    chk(worst_raw < Real(0), "3_weno5_nu_sousshoote_negatif");  // le probleme existe (face < 0)
    chk(n_under_pp == 0, "3_pp_garantit_le_plancher");          // le limiteur le corrige partout
    // CONSERVATION : la divergence de flux telescope en periodique -> somme de masse du residu nulle,
    // floor actif ou non (le scaling ne touche que les etats de FACE, jamais la moyenne).
    MultiFab R(ba, dm, EulerNoSrc::n_vars, 0);
    assemble_rhs<Weno5, RusanovFlux>(model, U, aux, geom, R, false, floor);
    sync_host();
    double mass = 0;
    for (int li = 0; li < R.local_size(); ++li) {
      const ConstArray4 a = R.fab(li).const_array();
      const Box2D v = R.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          mass += a(i, j, 0);
    }
    mass = all_reduce_sum(mass);
    if (me == 0)
      std::printf("    somme de masse du residu (periodique) = %.3e\n", mass);
    chk(std::abs(mass) < 1e-10, "3_residu_conservatif");
    // DYNAMIQUE AVEC floor : advance du top-hat SANS echarde (le profil du cas reel) -> etat fini.
    // NB : le scaling PAR FACE garantit les ETATS DE FACE >= floor, pas les moyennes mises a jour
    // (le theoreme Zhang-Shu complet exige le theta commun par cellule + CFL SSP) ; sur l'echarde
    // extreme en Euler quasi froid, la PRESSION (variable independante, p = (gamma-1)(E - cinetique))
    // peut passer negative quoi qu'il arrive a rho -- physique plus fragile que la cible ISOTHERME
    // du ticket (p = cs2 rho, asservie a rho). La validation dynamique de reference est le cas
    // Hoffart isotherme end-to-end (adc_cases).
    const Real dt = Real(0.2) * geom.dx() / Real(2.5);  // CFL prudent (u=1, c~1.2 au fond)
    const int nsteps = 30;
    MultiFab Uf(ba, dm, EulerNoSrc::n_vars, ng);
    init_tophat(Uf, dom, model, /*spike=*/false);
    fill_ghosts(Uf, dom, bc);
    BlockClosures cpp =
        make_block(model, "weno5", "rusanov", ctx, false, false, "ssprk2", {}, {}, nullptr, floor);
    cpp.advance(Uf, dt * nsteps, nsteps);
    sync_host();
    Real min_pp = Real(1e30);
    bool finite = true;
    const ConstArray4 uf = Uf.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        min_pp = std::min(min_pp, uf(i, j, 0));
        finite = finite && std::isfinite(static_cast<double>(uf(i, j, 0)));
      }
    if (me == 0)
      std::printf("    advance %d pas AVEC floor : min rho moyenne = %.3e (fini = %d)\n", nsteps,
                  static_cast<double>(min_pp), finite ? 1 : 0);
    chk(finite, "3_floor_etat_fini");
  }

  // ----------------------------------------------------------------------------------------------
  // (4) REJET CLAIR : modele sans role Density + pos_floor > 0 -> runtime_error explicite.
  // ----------------------------------------------------------------------------------------------
  {
    const adc::validation::AdvectionDiffusion scal{1.0, 0.0, 0.0};
    MultiFab Us(ba, dm, 1, 2), Rs(ba, dm, 1, 0);
    Us.set_val(1.0);
    bool threw = false;
    std::string msg;
    try {
      BlockClosures c = make_block(scal, "minmod", "rusanov", ctx, false, false, "ssprk2", {}, {},
                                   nullptr, Real(1e-8));
      c.rhs_into(Us, Rs);
    } catch (const std::runtime_error& e) {
      threw = true;
      msg = e.what();
    }
    if (me == 0)
      std::printf("(4) modele sans Density : %s (%s)\n", threw ? "REJETE" : "ACCEPTE (!)",
                  msg.c_str());
    chk(threw, "4_rejet_modele_sans_density");
    // AdvectionDiffusion n'a pas d'introspection VariableSet du tout -> message "sans introspection" ;
    // un modele introspectable sans role Density donnerait le message "role Density". Les deux
    // mentionnent positivity_floor (la cause).
    chk(msg.find("positivity_floor") != std::string::npos, "4_message_mentionne_positivity_floor");
  }

  fails = static_cast<long>(all_reduce_max(static_cast<double>(fails)));
  if (me == 0 && fails == 0)
    std::printf("OK test_positivity_floor\n");
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

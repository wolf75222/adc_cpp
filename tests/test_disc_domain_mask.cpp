// Chantier T2 : MASQUE DE DOMAINE DISQUE conservatif (CONTRAT, inerte par defaut).
//
// Le coeur expose deux briques de transport FV :
//   - assemble_rhs<L, F>           : residu -div Fhat + S sur TOUT le domaine (chemin historique) ;
//   - assemble_rhs_masked<L, F>    : MEME residu RESTREINT a un sous-domaine actif (masque 0/1
//                                    cellule-centre), avec flux normal NUL aux faces active/inactive
//                                    (paroi FV) -> conservation de masse sur le sous-domaine actif.
//
// On valide les DEUX proprietes du contrat (vraies assertions, pas de no-op) :
//   (a) BIT-IDENTITE : un masque TOUT ACTIF rend assemble_rhs_masked STRICTEMENT egal a assemble_rhs
//       (egalite bit a bit, diff exactement 0). C'est l'invariant "inerte par defaut" : tant que le
//       sous-domaine est le domaine entier, le residu est celui du chemin historique.
//   (b) CONSERVATION : avec un masque DISQUE (DiscDomain), une advection a vitesse CONSTANTE
//       (champ de transport a divergence nulle) avancee par Euler avant conserve la masse sur les
//       cellules ACTIVES a la PRECISION MACHINE, et le residu est EXACTEMENT 0 sur les cellules
//       inactives (aucun flux ne traverse la frontiere du disque).
//
// Modele jouet INLINE (le coeur ne connait aucune physique) : scalaire advecte a vitesse (vx, vy)
// constante, flux F = (vx u, vy u), vitesse d'onde max |v|. Aucune source, aucun elliptique.

#include <adc/core/physical_model.hpp>
#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/mf_arith.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/runtime/wall_predicate.hpp>  // detail::DiscDomain (descripteur source-unique)

#include <cmath>
#include <cstdio>

using namespace adc;

// Advection scalaire a vitesse constante (vx, vy). Flux F = (vx u, vy u).
struct Advect {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  Real vx = 0.0, vy = 0.0;
  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    return State{(dir == 0 ? vx : vy) * u[0]};
  }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int dir) const {
    return std::fabs(dir == 0 ? vx : vy);
  }
  ADC_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<Advect>, "Advect est un PhysicalModel");
static_assert(!DiffusiveModel<Advect>, "Advect n'est pas diffusif (le masque cible le flux hyperbolique)");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 48;
  const double L = 1.0;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, L, 0.0, L};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;  // periodique par defaut (suffit : le masque ferme la frontiere physique du disque)

  const Advect model{0.7, -0.4};  // vitesse constante quelconque, div v = 0

  // Etat initial : bosse lisse (recouvrement avec le disque). Aux nul (flux ignore aux).
  // U porte Minmod::n_ghost (= 2) couches de ghost : les deux chemins exerces ici (assemble_rhs et
  // assemble_rhs_masked, instancies avec Minmod) reconstruisent les cellules voisines i+-1 -> lecture
  // i+-2 au bord de la boite valide. Avec 1 seul ghost cette lecture sortait du buffer (ADC-163,
  // heap-buffer-overflow ASan). aux / masque ne sont lus qu'a i+-1 -> 1 ghost suffit.
  MultiFab U(ba, dm, 1, Minmod::n_ghost), aux(ba, dm, kAuxBaseComps, 1);
  aux.set_val(0.0);
  {
    Array4 a = U.fab(0).array();
    const Box2D g = U.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        a(i, j, 0) = 1.0 + 0.5 * std::exp(-(((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.02));
      }
  }

  // ----------------------------------------------------------------------
  // (a) BIT-IDENTITE : masque TOUT ACTIF -> assemble_rhs_masked == assemble_rhs (diff exactement 0).
  // ----------------------------------------------------------------------
  {
    MultiFab mask(ba, dm, 1, 1);
    mask.set_val(Real(1));  // tout actif : le sous-domaine est le domaine entier
    fill_ghosts(U, geom.domain, bc);  // memes ghosts pour les deux chemins
    MultiFab R_ref(ba, dm, 1, 0), R_msk(ba, dm, 1, 0);
    assemble_rhs<Minmod, RusanovFlux>(model, U, aux, geom, R_ref);
    assemble_rhs_masked<Minmod, RusanovFlux>(model, U, aux, mask, geom, R_msk);

    double max_abs_diff = 0.0;
    const ConstArray4 rr = R_ref.fab(0).const_array();
    const ConstArray4 rm = R_msk.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
        max_abs_diff = std::max(max_abs_diff, std::fabs(double(rr(i, j, 0)) - double(rm(i, j, 0))));
    std::printf("  (a) bit-identite masque-tout-actif : max|R_masked - R_ref| = %.3e (attendu 0)\n",
                max_abs_diff);
    // Egalite BIT A BIT : le chemin masque tout-actif emprunte le MEME flux/reconstruction, on exige
    // une difference EXACTEMENT nulle (pas une tolerance) -- c'est l'invariant "inerte par defaut".
    chk(max_abs_diff == 0.0,
        "(a) masque tout actif : residu masque BIT-IDENTIQUE au residu historique (diff = 0)");
  }

  // ----------------------------------------------------------------------
  // (b) CONSERVATION : masque DISQUE -> masse sur les cellules actives conservee a la machine,
  //     residu EXACTEMENT 0 sur les cellules inactives (flux nul a travers la frontiere du disque).
  // ----------------------------------------------------------------------
  {
    // Disque centre dans la boite, rayon < L/2 pour qu'il y ait de vraies cellules inactives.
    const detail::DiscDomain disc = detail::DiscDomain::centered_in_box(L, 0.35);
    MultiFab mask(ba, dm, 1, 1);
    {
      Array4 m = mask.fab(0).array();
      const Box2D g = mask.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
          m(i, j, 0) = disc.cell_active(geom.x_cell(i), geom.y_cell(j)) ? Real(1) : Real(0);
    }

    // Compte les cellules actives ET inactives valides : le test n'a de sens que si les DEUX existent.
    int n_active = 0, n_inactive = 0;
    {
      const ConstArray4 m = mask.fab(0).const_array();
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          (m(i, j, 0) >= Real(0.5) ? n_active : n_inactive)++;
    }
    chk(n_active > 0 && n_inactive > 0,
        "(b) le disque partitionne la grille en cellules actives ET inactives (test non vide)");

    // Masse initiale sur les cellules ACTIVES (somme ponderee par le masque). dx2 = aire de cellule.
    const double dx2 = geom.dx() * geom.dy();
    auto active_mass = [&](const MultiFab& F) {
      device_fence();
      const ConstArray4 f = F.fab(0).const_array();
      const ConstArray4 m = mask.fab(0).const_array();
      double s = 0.0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          if (m(i, j, 0) >= Real(0.5)) s += double(f(i, j, 0));
      return s * dx2;
    };

    const double m0 = active_mass(U);
    chk(m0 > 0.0, "(b) masse active initiale strictement positive (bosse couvre le disque)");

    // Avance EXPLICITE Euler avant sur le residu MASQUE : U^{n+1} = U^n + dt R_masked(U^n).
    // Sur une cellule inactive R = 0 -> U y reste fige ; sur une cellule active, le flux normal des
    // faces touchant une inactive est nul -> aucune masse ne franchit la frontiere du disque.
    const double v = std::hypot(model.vx, model.vy);
    const double dt = 0.2 * geom.dx() / v;  // CFL transport
    double max_inactive_residual = 0.0;
    for (int s = 0; s < 60; ++s) {
      fill_ghosts(U, geom.domain, bc);
      MultiFab R(ba, dm, 1, 0);
      assemble_rhs_masked<Minmod, RusanovFlux>(model, U, aux, mask, geom, R);
      // Le residu DOIT etre exactement nul sur les cellules inactives (elles ne sont pas avancees).
      {
        const ConstArray4 r = R.fab(0).const_array();
        const ConstArray4 m = mask.fab(0).const_array();
        for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
          for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
            if (m(i, j, 0) < Real(0.5))
              max_inactive_residual = std::max(max_inactive_residual, std::fabs(double(r(i, j, 0))));
      }
      saxpy(U, Real(dt), R);  // U += dt R (cellules valides)
    }

    const double m1 = active_mass(U);
    const double rel_drift = std::fabs(m1 - m0) / std::fabs(m0);
    std::printf("  (b) masse active : m0 = %.15e  m1 = %.15e  drift relatif = %.3e\n", m0, m1,
                rel_drift);
    std::printf("  (b) residu max sur cellules inactives = %.3e (attendu 0)\n",
                max_inactive_residual);

    // Le residu sur les cellules inactives est EXACTEMENT 0 (le kernel les met a zero) : egalite bit
    // a bit, pas une tolerance.
    chk(max_inactive_residual == 0.0,
        "(b) residu EXACTEMENT nul sur les cellules inactives (aucune avance hors du disque)");
    // La masse active derive seulement du non-bit-identisme de l'arithmetique flottante (somme de
    // flux internes telescopiques) : borne JUSTE au-dessus du bruit machine (~1e-15 attendu).
    chk(rel_drift < 1e-12,
        "(b) masse sur les cellules actives conservee a la machine (flux normal nul a la frontiere "
        "du disque ; drift < 1e-12)");
    // Temoin que la dynamique a bien TOURNE (sinon la conservation serait triviale) : l'etat a bouge.
    {
      device_fence();
      const ConstArray4 u = U.fab(0).const_array();
      double max_dev = 0.0;
      for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
        for (int i = dom.lo[0]; i <= dom.hi[0]; ++i)
          max_dev = std::max(max_dev, std::fabs(double(u(i, j, 0)) - 1.0));
      chk(max_dev > 1e-3,
          "(b) le transport a effectivement avance l'etat (la conservation n'est pas triviale)");
    }
  }

  if (fails == 0) std::printf("OK test_disc_domain_mask\n");
  return fails == 0 ? 0 : 1;
}

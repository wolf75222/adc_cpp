// CAPABILITIES Riemann (audit 2026-06, generalisation long terme) : HLLC et Roe ne sont plus des
// algorithmes Euler-only deguises -- la structure physique (onde de contact, dissipation de Roe)
// peut etre fournie par le MODELE via les traits HasHLLCStructure / HasRoeDissipation, et le coeur
// applique alors l'algorithme GENERIQUE sans hypothese de layout.
//
// Verifications :
//  (1) DETECTION : adc::Euler (sans hooks) ne satisfait PAS les capabilities (il prend le chemin
//      canonique historique) ; HookedEuler (hooks Euler explicites) les satisfait.
//  (2) EQUIVALENCE : le chemin GENERIQUE (HookedEuler, hooks reproduisant les formules Euler) rend
//      le MEME flux que le chemin canonique (adc::Euler) -- HLLC et Roe, subsonique et
//      supersonique, x et y. C'est la preuve que l'algorithme generique est l'algorithme
//      historique, la structure en moins.
//  (3) NON-EULER : un modele ISOTHERME 3 VARIABLES (la ou hllc canonique est impossible,
//      n_vars != 4) fournit ses hooks HLLC -> le solveur contact-resolving marche : consistance
//      F*(U,U) == flux(U), et un CISAILLEMENT STATIONNAIRE (un = 0, saut tangentiel) est preserve
//      EXACTEMENT (flux tangentiel nul) la ou HLL le diffuse. C'est exactement ce qu'apporte la
//      resolution de l'onde intermediaire, prouvee hors Euler.
#include <adc/numerics/fv/numerical_flux.hpp>
#include <adc/physics/fluids/euler.hpp>

#include <cmath>
#include <cstdio>

using adc::Aux;
using adc::Real;

// ---------------------------------------------------------------------------------------------
// HookedEuler : adc::Euler + capabilities HLLC/Roe reproduisant EXACTEMENT les formules du chemin
// canonique (Toro 10.37 pour le contact, moyenne de Roe + Harten pour la dissipation).
// ---------------------------------------------------------------------------------------------
struct HookedEuler : adc::Euler {
  ADC_HD Real contact_speed(const State& UL, const State& UR, Real pL, Real pR, Real sL, Real sR,
                            int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
           (rL * (sL - unL) - rR * (sR - unR));
  }
  ADC_HD State hllc_star_state(const State& U, Real p, Real s, Real sStar, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real r = U[0];
    const Real un = U[in] / r;
    const Real fac = r * (s - un) / (s - sStar);
    State Us{};
    Us[0] = fac;
    Us[in] = fac * sStar;
    Us[it] = fac * (U[it] / r);
    Us[3] = fac * (U[3] / r + (sStar - un) * (sStar + p / (r * (s - un))));
    return Us;
  }
  ADC_HD State roe_dissipation(const State& UL, const Aux&, const State& UR, const Aux&,
                               int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    const Real utL = UL[it] / rL, utR = UR[it] / rR;
    const Real pL = pressure(UL), pR = pressure(UR);
    const Real HL = (UL[3] + pL) / rL, HR = (UR[3] + pR) / rR;
    const Real sqL = std::sqrt(rL), sqR = std::sqrt(rR), den = sqL + sqR;
    const Real un = (sqL * unL + sqR * unR) / den;
    const Real ut = (sqL * utL + sqR * utR) / den;
    const Real H = (sqL * HL + sqR * HR) / den;
    const Real rho = sqL * sqR;
    const Real q2 = un * un + ut * ut;
    const Real gm1 = pL / (UL[3] - Real(0.5) * rL * (unL * unL + utL * utL));
    const Real c2 = gm1 * (H - Real(0.5) * q2);
    const Real c = std::sqrt(c2);
    const Real dr = rR - rL, dp = pR - pL, dun = unR - unL, dut = utR - utL;
    const Real a1 = (dp - rho * c * dun) / (Real(2) * c2);
    const Real a2 = dr - dp / c2;
    const Real a3 = rho * dut;
    const Real a5 = (dp + rho * c * dun) / (Real(2) * c2);
    const Real eps = adc::kRoeEntropyFixFraction * c;
    auto absfix = [eps](Real l) {
      const Real al = l < 0 ? -l : l;
      return al < eps ? Real(0.5) * (l * l / eps + eps) : al;
    };
    const Real al1 = absfix(un - c), al2 = (un < 0 ? -un : un), al5 = absfix(un + c);
    State d{};
    d[0] = al1 * a1 + al2 * a2 + al5 * a5;
    d[in] = al1 * a1 * (un - c) + al2 * a2 * un + al5 * a5 * (un + c);
    d[it] = al1 * a1 * ut + al2 * (a2 * ut + a3) + al5 * a5 * ut;
    d[3] =
        al1 * a1 * (H - un * c) + al2 * (a2 * Real(0.5) * q2 + a3 * ut) + al5 * a5 * (H + un * c);
    return d;
  }
};

// ---------------------------------------------------------------------------------------------
// IsoHLLC : fluide ISOTHERME 3 variables (rho, m_x, m_y) avec capability HLLC -- le cas que le
// chemin canonique (n_vars == 4) ne peut PAS traiter. p = cs2 * rho ; ondes u -+ c, c = sqrt(cs2).
// Etat etoile : rho* = rho (s - un)/(s - s*) ; m_n* = rho* s* ; m_t* = rho* u_t.
// ---------------------------------------------------------------------------------------------
struct IsoHLLC {
  using State = adc::StateVec<3>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 3;
  Real cs2 = 0.5;

  ADC_HD State flux(const State& u, const Aux&, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real un = u[in] / u[0];
    State F{};
    F[0] = u[in];
    F[in] = u[in] * un + cs2 * u[0];
    F[it] = u[it] * un;
    return F;
  }
  ADC_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real un = u[in] / u[0];
    const Real c = std::sqrt(cs2);
    const Real a = un < 0 ? -un : un;
    return a + c;
  }
  ADC_HD void wave_speeds(const State& u, const Aux&, int dir, Real& smin, Real& smax) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real un = u[in] / u[0];
    const Real c = std::sqrt(cs2);
    smin = un - c;
    smax = un + c;
  }
  ADC_HD Real pressure(const State& u) const { return cs2 * u[0]; }
  ADC_HD Real contact_speed(const State& UL, const State& UR, Real pL, Real pR, Real sL, Real sR,
                            int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const Real rL = UL[0], rR = UR[0];
    const Real unL = UL[in] / rL, unR = UR[in] / rR;
    return (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) /
           (rL * (sL - unL) - rR * (sR - unR));
  }
  ADC_HD State hllc_star_state(const State& U, Real /*p*/, Real s, Real sStar, int dir) const {
    const int in = (dir == 0) ? 1 : 2;
    const int it = (dir == 0) ? 2 : 1;
    const Real r = U[0];
    const Real un = U[in] / r;
    const Real fac = r * (s - un) / (s - sStar);
    State Us{};
    Us[0] = fac;
    Us[in] = fac * sStar;
    Us[it] = fac * (U[it] / r);
    return Us;
  }
};

using State4 = adc::StateVec<4>;
using State3 = adc::StateVec<3>;

static State4 cons(double rho, double u, double v, double p, double gamma) {
  State4 U{};
  U[0] = rho;
  U[1] = rho * u;
  U[2] = rho * v;
  U[3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
  return U;
}

template <int N>
static double maxdiff(const adc::StateVec<N>& a, const adc::StateVec<N>& b) {
  double m = 0;
  for (int c = 0; c < N; ++c)
    m = std::fmax(m, std::fabs(a[c] - b[c]));
  return m;
}

int main() {
  // (1) DETECTION compile-time des capabilities.
  static_assert(!adc::HasHLLCStructure<adc::Euler>,
                "Euler sans hooks ne doit PAS satisfaire HasHLLCStructure (chemin canonique)");
  static_assert(!adc::HasRoeDissipation<adc::Euler>,
                "Euler sans hooks ne doit PAS satisfaire HasRoeDissipation (chemin canonique)");
  static_assert(adc::HasHLLCStructure<HookedEuler>, "HookedEuler doit satisfaire HasHLLCStructure");
  static_assert(adc::HasRoeDissipation<HookedEuler>,
                "HookedEuler doit satisfaire HasRoeDissipation");
  static_assert(adc::HasHLLCStructure<IsoHLLC>, "IsoHLLC doit satisfaire HasHLLCStructure");
  std::printf("OK  detection des capabilities (Euler canonique, Hooked/Iso capability)\n");

  adc::Euler e;
  e.gamma = 1.4;
  HookedEuler he;
  he.gamma = 1.4;
  adc::HLLCFlux hllc;
  adc::RoeFlux roe;
  adc::HLLFlux hll;
  Aux a{};

  // (2) EQUIVALENCE chemin generique == chemin canonique (memes formules fournies en hooks).
  const State4 pairs[][2] = {
      {cons(1.2, 0.3, -0.1, 1.5, 1.4), cons(0.7, -0.2, 0.4, 0.9, 1.4)},   // subsonique
      {cons(1.0, 8.0, 0.5, 1.0, 1.4), cons(1.5, 12.0, -0.3, 1.3, 1.4)},   // supersonique +
      {cons(0.9, -7.0, 0.2, 1.1, 1.4), cons(1.1, -9.0, -0.4, 0.8, 1.4)},  // supersonique -
  };
  for (const auto& pr : pairs)
    for (int dir = 0; dir < 2; ++dir) {
      const double dh =
          maxdiff(hllc(he, pr[0], a, pr[1], a, dir), hllc(e, pr[0], a, pr[1], a, dir));
      if (dh > 1e-13) {
        std::printf("FAIL HLLC generique != canonique (dir %d) : %.3e\n", dir, dh);
        return 1;
      }
      const double dr = maxdiff(roe(he, pr[0], a, pr[1], a, dir), roe(e, pr[0], a, pr[1], a, dir));
      if (dr > 1e-13) {
        std::printf("FAIL Roe generique != canonique (dir %d) : %.3e\n", dir, dr);
        return 1;
      }
    }
  std::printf("OK  HLLC/Roe generiques == canoniques (hooks Euler, 3 paires x 2 directions)\n");

  // (3) NON-EULER : HLLC capability sur l'isotherme 3 variables.
  IsoHLLC iso;
  // (3a) consistance : F*(U, U) == flux(U).
  {
    State3 U{};
    U[0] = 1.3;
    U[1] = 0.4;
    U[2] = -0.7;
    for (int dir = 0; dir < 2; ++dir) {
      const double d = maxdiff(hllc(iso, U, a, U, a, dir), iso.flux(U, a, dir));
      if (d > 1e-13) {
        std::printf("FAIL consistance HLLC isotherme (dir %d) : %.3e\n", dir, d);
        return 1;
      }
    }
  }
  std::printf("OK  HLLC isotherme consistant : F*(U,U) == flux(U)\n");
  // (3b) cisaillement stationnaire (un = 0, rho egal, saut tangentiel) : l'onde intermediaire est
  // resolue -> flux tangentiel EXACTEMENT nul (HLLC), la ou HLL le diffuse (terme sL sR dU != 0).
  {
    State3 UL{}, UR{};
    UL[0] = 1.0;
    UL[1] = 0.0;
    UL[2] = 2.0;  // u_t = +2
    UR[0] = 1.0;
    UR[1] = 0.0;
    UR[2] = -3.0;  // u_t = -3
    const State3 Fc = hllc(iso, UL, a, UR, a, 0);
    const State3 Fh = hll(iso, UL, a, UR, a, 0);
    if (std::fabs(Fc[2]) > 1e-14) {
      std::printf("FAIL HLLC isotherme : cisaillement stationnaire diffuse (F_t = %.3e)\n",
                  static_cast<double>(Fc[2]));
      return 1;
    }
    if (std::fabs(Fh[2]) < 1e-2) {
      std::printf("FAIL temoin HLL : le cisaillement devrait etre diffuse (F_t = %.3e)\n",
                  static_cast<double>(Fh[2]));
      return 1;
    }
    std::printf(
        "OK  HLLC isotherme preserve le cisaillement stationnaire (F_t = 0 ; "
        "HLL temoin F_t = %.3e)\n",
        static_cast<double>(Fh[2]));
  }

  std::printf("OK  test_riemann_capabilities : tout est vert\n");
  return 0;
}

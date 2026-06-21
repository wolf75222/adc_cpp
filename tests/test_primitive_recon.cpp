// Variables primitives (rho, u, p) : conversions cons <-> prim sur les briques de
// transport, leur usage dans max_wave_speed, et le concept HasPrimitiveVars satisfait par
// les modeles composes. La reconstruction primitive de l'operateur spatial s'appuie sur ces
// conversions ; on verifie ici leur exactitude (round-trip) et la centralisation du calcul
// des variables primitives. Imprime "OK test_primitive_recon" en cas de succes.

#include <adc/core/physical_model.hpp>
#include <adc/physics/bricks.hpp>
#include <adc/physics/euler.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

static int fails = 0;
static void chk(bool c, const char* label) {
  std::printf("  [%s] %s\n", c ? "OK " : "XX ", label);
  if (!c)
    ++fails;
}
static bool close(Real a, Real b, Real tol = 1e-12) {
  const Real d = a - b;
  return (d < 0 ? -d : d) < tol;
}

int main() {
  std::printf("== variables primitives : conversions + concept ==\n");

  // Euler : round-trip conservatif -> primitif -> conservatif.
  Euler e;
  e.gamma = 1.4;
  Euler::State U{};
  U[0] = 1.2;
  U[1] = 0.36;
  U[2] = -0.6;
  U[3] = 3.0;
  const Euler::Prim P = e.to_primitive(U);
  chk(close(P[0], 1.2) && close(P[1], 0.36 / 1.2) && close(P[2], -0.6 / 1.2),
      "Euler to_primitive : (rho, u, v)");
  chk(P[3] > 0, "Euler to_primitive : pression positive");
  const Euler::State U2 = e.to_conservative(P);
  bool rt = true;
  for (int c = 0; c < 4; ++c)
    rt = rt && close(U[c], U2[c]);
  chk(rt, "Euler round-trip cons->prim->cons == identite");

  // max_wave_speed calcule via le primitif, coherent avec |u| + c.
  Aux a{};
  const Real c = std::sqrt(1.4 * e.pressure(U) / U[0]);
  chk(close(e.max_wave_speed(U, a, 0), std::fabs(0.36 / 1.2) + c),
      "Euler max_wave_speed (via primitif) == |u| + c");

  // Isotherme : round-trip + vitesses primitives.
  IsothermalFlux is;
  is.cs2 = 0.5;
  StateVec<3> Ui{};
  Ui[0] = 2.0;
  Ui[1] = 0.8;
  Ui[2] = -0.2;
  const auto Pi = is.to_primitive(Ui);
  const auto Ui2 = is.to_conservative(Pi);
  bool rti = true;
  for (int k = 0; k < 3; ++k)
    rti = rti && close(Ui[k], Ui2[k]);
  chk(rti && close(Pi[1], 0.4) && close(Pi[2], -0.1), "isotherme round-trip + (u, v) primitifs");

  // Scalaire (ExB) : conversions = identite.
  ExBVelocity exb;
  StateVec<1> n{};
  n[0] = 0.7;
  chk(close(exb.to_primitive(n)[0], 0.7) && close(exb.to_conservative(n)[0], 0.7),
      "scalaire : prim == cons (identite)");

  // Concept : les modeles composes exposent les variables primitives.
  using Mc = CompositeModel<CompressibleFlux, NoSource, ChargeDensity>;
  using Mi = CompositeModel<IsothermalFlux, NoSource, ChargeDensity>;
  using Ms = CompositeModel<ExBVelocity, NoSource, BackgroundDensity>;
  static_assert(HasPrimitiveVars<Mc>, "compose Euler doit exposer les variables primitives");
  static_assert(HasPrimitiveVars<Mi>, "compose isotherme doit exposer les variables primitives");
  static_assert(HasPrimitiveVars<Ms>, "compose scalaire : conversions identite");
  chk(true, "HasPrimitiveVars : Euler / isotherme / scalaire (static_assert)");

  std::printf(fails == 0 ? "OK test_primitive_recon\n" : "%d ECHEC(S)\n", fails);
  return fails == 0 ? 0 : 1;
}

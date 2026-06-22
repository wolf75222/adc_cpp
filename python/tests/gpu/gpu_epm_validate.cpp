// Validation DEVICE (GH200, Kokkos Cuda) des operateurs elliptiques composables de GeometricMG
// merges sur master apres #48 : (2) Poisson ECRANTE / Helmholtz div(eps grad phi) - kappa phi = f
// (#44, set_reaction) et (3) Poisson ANISOTROPE div(diag(eps_x, eps_y) grad phi) = f (#52,
// set_epsilon_anisotropic). Les noyaux kappa/eps_y vivent dans les for_each_cell ADC_HD du
// smoother red-black, du residu et de l'apply (cf. numerics/elliptic/poisson_operator.hpp) : ils
// s'executent donc sur le device sous le backend Cuda, EXACTEMENT comme le Poisson de base
// (phase 3). Ce harness rejoue les MMS de tests/test_screened_poisson.cpp et
// tests/test_anisotropic_epsilon.cpp et :
//   - imprime exec= (espace Kokkos actif), le nombre de V-cycles, l'erreur L-inf MMS et les ratios
//     de convergence (ordre 2 attendu) ;
//   - ecrit un dump binaire de phi (toutes les cellules valides) -> comparaison bit-a-bit Cuda vs
//     Serial par diff_bin (dmax sur CHAQUE cellule, pas seulement une reduction).
// Portable seriel / Kokkos+CUDA : le MEME .cpp donne l'oracle Serial (exec=Serial) et le run device.

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/elliptic/mg/geometric_mg.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;
static constexpr double KAPPA = 50.0;  // 1/lambda_D^2 (ecrantage de Debye modere)

static double phi_exact(double x, double y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}
static double eps_x_field(double x, double /*y*/) {
  return 1.0 + 0.5 * x;
}
static double eps_y_field(double /*x*/, double y) {
  return 1.0 + 0.3 * y;
}

// f pour l'operateur ECRANTE : div(eps grad phi) - kappa phi, eps = eps_x_field (composable eps+kappa).
static double rhs_screened(double x, double y) {
  const double s = std::sin(kPi * x) * std::sin(kPi * y);
  const double div_eps_grad =
      -(1.0 + 0.5 * x) * 2.0 * kPi * kPi * s + 0.5 * kPi * std::cos(kPi * x) * std::sin(kPi * y);
  return div_eps_grad - KAPPA * s;
}
// f pour l'operateur ANISOTROPE : div(diag(eps_x, eps_y) grad phi).
static double rhs_aniso(double x, double y) {
  const double s = std::sin(kPi * x) * std::sin(kPi * y);
  return 0.5 * kPi * std::cos(kPi * x) * std::sin(kPi * y) +
         0.3 * kPi * std::sin(kPi * x) * std::cos(kPi * y) -
         kPi * kPi * s * ((1.0 + 0.5 * x) + (1.0 + 0.3 * y));
}

// Resout le probleme demande sur n x n (Dirichlet exact). Renvoie l'erreur L-inf et, par reference,
// le nombre de V-cycles et la solution phi (cellules valides, ordre lexicographique j puis i).
enum class Op { Screened, Aniso };
static double solve_mms(Op op, int n, int& cycles, std::vector<double>* phi_out) {
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  BoxArray ba = BoxArray::from_domain(dom, n);
  BCRec bc;
  bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;

  GeometricMG mg(geom, ba, bc);
  if (op == Op::Screened) {
    mg.set_epsilon([](Real x, Real y) { return Real(eps_x_field(x, y)); });
    mg.set_reaction([](Real, Real) { return Real(KAPPA); });
  } else {
    mg.set_epsilon_anisotropic([](Real x, Real y) { return Real(eps_x_field(x, y)); },
                               [](Real x, Real y) { return Real(eps_y_field(x, y)); });
  }

  // Remplissage du second membre f : ecriture HOTE en memoire unifiee (comme phase3_poisson.cpp).
  // On n'utilise PAS for_each_cell ici car rhs_screened/rhs_aniso sont des fonctions HOTE pures
  // (un appel hote depuis une lambda __device__ etendue serait rejete par nvcc) ; le solve, lui,
  // tourne entierement sur le device. Memoire unifiee -> f est visible du device sans copie.
  {
    Array4 af = mg.rhs().fab(0).array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double x = geom.x_cell(i), y = geom.y_cell(j);
        af(i, j, 0) = (op == Op::Screened) ? rhs_screened(x, y) : rhs_aniso(x, y);
      }
  }
  mg.phi().set_val(0.0);

  // boucle V-cycle EXACTEMENT comme les tests CPU (meme critere, meme plafond) -> meme nb de cycles.
  const Real r0 = mg.current_residual();
  Real rn = r0;
  cycles = 0;
  for (int c = 0; c < 80 && rn > 1e-11 * r0; ++c) {
    mg.vcycle();
    rn = mg.current_residual();
    ++cycles;
  }
  device_fence();  // for_each_cell est ASYNC sous Cuda : fence avant toute lecture HOTE

  Fab2D& p = mg.phi().fab(0);
  ConstArray4 pc = p.const_array();
  double eInf = 0;
  if (phi_out)
    phi_out->clear();
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      const double v = pc(i, j, 0);
      eInf = std::max(eInf, std::fabs(v - phi_exact(geom.x_cell(i), geom.y_cell(j))));
      if (phi_out)
        phi_out->push_back(v);
    }
  return eInf;
}

static void dump_bin(const std::string& path, const std::vector<double>& v) {
  if (path.empty())
    return;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::printf("WARN: cannot open %s\n", path.c_str());
    return;
  }
  const std::uint64_t nbytes = static_cast<std::uint64_t>(v.size()) * sizeof(double);
  std::fwrite(v.data(), 1, nbytes, f);
  std::fclose(f);
  std::printf("  dump %s (%zu doubles)\n", path.c_str(), v.size());
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::initialize(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  std::string dump_prefix;
  for (int k = 1; k < argc; ++k)
    if (std::strncmp(argv[k], "--dump=", 7) == 0)
      dump_prefix = argv[k] + 7;

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };
#if defined(ADC_HAS_KOKKOS)
  const char* space = Kokkos::DefaultExecutionSpace::name();
#else
  const char* space = "Serial(host)";
#endif

  // ---- (2) ECRANTE / Helmholtz (#44) : eps(x) + kappa, MMS ordre 2 -----------------------------
  {
    int c32 = 0, c64 = 0, c128 = 0;
    std::vector<double> phi64;
    const double e32 = solve_mms(Op::Screened, 32, c32, nullptr);
    const double e64 = solve_mms(Op::Screened, 64, c64, &phi64);
    const double e128 = solve_mms(Op::Screened, 128, c128, nullptr);
    const double r1 = e32 / e64, r2 = e64 / e128;
    std::printf(
        "[EPM screened] exec=%s cycles(32/64/128)=%d/%d/%d Linf=%.17g/%.17g/%.17g "
        "ratios=%.4f/%.4f\n",
        space, c32, c64, c128, e32, e64, e128, r1, r2);
    chk(r1 > 3.5 && r1 < 4.5, "screened_ordre2_32_64");
    chk(r2 > 3.5 && r2 < 4.5, "screened_ordre2_64_128");
    dump_bin(dump_prefix.empty() ? "" : dump_prefix + "_screened64.bin", phi64);
  }

  // ---- (3) ANISOTROPE (#52) : div(diag(eps_x, eps_y) grad phi), MMS ordre 2 ---------------------
  {
    int c32 = 0, c64 = 0, c128 = 0;
    std::vector<double> phi64;
    const double e32 = solve_mms(Op::Aniso, 32, c32, nullptr);
    const double e64 = solve_mms(Op::Aniso, 64, c64, &phi64);
    const double e128 = solve_mms(Op::Aniso, 128, c128, nullptr);
    const double r1 = e32 / e64, r2 = e64 / e128;
    std::printf(
        "[EPM aniso]    exec=%s cycles(32/64/128)=%d/%d/%d Linf=%.17g/%.17g/%.17g "
        "ratios=%.4f/%.4f\n",
        space, c32, c64, c128, e32, e64, e128, r1, r2);
    chk(r1 > 3.5 && r1 < 4.5, "aniso_ordre2_32_64");
    chk(r2 > 3.5 && r2 < 4.5, "aniso_ordre2_64_128");
    dump_bin(dump_prefix.empty() ? "" : dump_prefix + "_aniso64.bin", phi64);
  }

  if (fails == 0)
    std::printf("OK gpu_epm_validate (exec=%s)\n", space);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return fails == 0 ? 0 : 1;
}

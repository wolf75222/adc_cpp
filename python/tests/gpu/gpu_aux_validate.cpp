// Validation DEVICE (GH200, Kokkos Cuda) de la LECTURE de T_e (composante aux 4) via load_aux<5>
// sur le device. Le portage precedent n'avait valide que load_aux<4> (B_z, comp 3) ; on ajoute ici
// la comp 4 (T_e). Un modele declarant n_aux=5 lit a.T_e dans sa source S = T_e * u ; le residu
// passe par assemble_rhs -> load_aux<aux_comps<Model>()=5>(a,i,j) (a.T_e = a(i,j,4)) dans le
// FONCTEUR NOMME AssembleRhsKernel (for_each_cell ADC_HD) -> chemin device sous Cuda.
//
// On valide ce chemin par assemble_rhs HEADER-ONLY (et NON via System+add_compiled_model). Raison
// d'honnetete : add_compiled_model instancie des lambdas etendues __host__ __device__ dans la TU
// appelante -> limite nvcc connue qui SEGFAUTE a l'execution sur Cuda (documentee dans
// runtime/dsl_block.hpp et tests/test_compiled_model_parity.cpp), independamment de T_e. assemble_rhs
// passe au contraire par des FONCTEURS NOMMES, device-robustes (cf. spatial_operator.hpp), c'est donc
// le chemin device REEL de la lecture de T_e -- le meme que pour B_z dans le harness AMR.
//
// T_e est pose au centre des cellules (comp 4 du canal aux, ecriture hote en memoire unifiee) avec un
// profil NON CONSTANT T_e(x,y) = 1 + x + 2 y pour que la source depende vraiment de la cellule.
// Imprime exec=, max|R - T_e*u|, et dump binaire de R -> diff_bin Cuda vs Serial (dmax par cellule).

#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>  // device_fence
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/numerical_flux.hpp>    // RusanovFlux
#include <adc/numerics/reconstruction.hpp>    // NoSlope
#include <adc/numerics/spatial_operator.hpp>  // assemble_rhs, load_aux, aux_comps
#include <adc/parallel/comm.hpp>

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

// Modele jouet qui LIT T_e (composante aux 4). flux nul, pas de couplage elliptique ; source
// S = T_e * u. n_aux=5 -> assemble_rhs instancie load_aux<5> et remplit a.T_e = a(i,j,4).
struct TeProbe {
  using State = StateVec<1>;
  using Aux = adc::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 5;  // phi, grad_x, grad_y, B_z, T_e
  ADC_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  ADC_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  ADC_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.T_e * u[0];  // lit la composante aux 4 (T_e)
    return s;
  }
  ADC_HD Real elliptic_rhs(const State&) const { return Real(0); }
};
static_assert(PhysicalModel<TeProbe>);
static_assert(aux_comps<TeProbe>() == 5, "TeProbe lit T_e (composante aux 4)");

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

  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());

  // u = 2 partout (flux nul -> R = source = T_e * u). aux a 5 comp : T_e (comp 4) = 1 + x + 2 y,
  // phi/grad/B_z = 0. Ecriture HOTE en memoire unifiee (T_e pose une fois).
  const Real u0 = Real(2);
  MultiFab U(ba, dm, 1, 2);
  U.set_val(u0);
  MultiFab aux(ba, dm, aux_comps<TeProbe>(), 1);
  aux.set_val(Real(0));
  chk(aux.ncomp() == 5, "aux_width_5");
  auto te = [](Real x, Real y) { return Real(1) + x + 2 * y; };
  for (int li = 0; li < aux.local_size(); ++li) {
    Fab2D& f = aux.fab(li);
    const Box2D g = f.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        f(i, j, 4) = te(geom.x_cell(i), geom.y_cell(j));
  }

  // assemble_rhs : R = -div F + S ; flux nul -> R = S = T_e * u. La lecture de a.T_e = a(i,j,4)
  // se fait dans load_aux<5> sur le device (for_each_cell sous Cuda).
  MultiFab R(ba, dm, 1, 0);
  assemble_rhs<NoSlope, RusanovFlux>(TeProbe{}, U, aux, geom, R);
  device_fence();

  double err = 0, rmin = 1e300, rmax = -1e300;
  std::vector<double> Rd;
  {
    const ConstArray4 r = R.fab(0).const_array();
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double got = r(i, j, 0);
        const double expect = te(geom.x_cell(i), geom.y_cell(j)) * u0;  // T_e * u
        err = std::fmax(err, std::fabs(got - expect));
        rmin = std::fmin(rmin, got);
        rmax = std::fmax(rmax, got);
        Rd.push_back(got);
      }
  }
  std::printf("[aux T_e] exec=%s  R=T_e*u in [%.17g, %.17g]  max|R - T_e*u| = %.3e\n", space, rmin,
              rmax, err);
  chk(err < 1e-12, "Te_comp4_read_via_load_aux5_assemble_rhs");
  chk(rmax > 1e-3, "Te_actually_read_nonzero");
  chk(std::fabs(rmax - rmin) > 1e-3, "Te_varies_per_cell");  // profil non constant lu

  if (!dump_prefix.empty()) {
    const std::string path = dump_prefix + "_te.bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f) {
      std::fwrite(Rd.data(), 1, Rd.size() * sizeof(double), f);
      std::fclose(f);
      std::printf("  dump %s (%zu doubles)\n", path.c_str(), Rd.size());
    }
  }

  if (fails == 0)
    std::printf("OK gpu_aux_validate (exec=%s)\n", space);
#if defined(ADC_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return fails == 0 ? 0 : 1;
}

// Diocotron AMR (AmrCouplerMP : Poisson grossier + injection d'aux + pas multi-patch +
// regrid Berger-Rigoutsos) execute SOUS LE SEAM KOKKOS. C'est la de-risk locale de l'etape 1
// du hero-run AMR (cf. docs/HERO_RUN_AMR.md) : le MEME for_each_cell qui partira sur le GPU
// (espace d'execution Cuda sur GH200) tourne ici sur l'espace HOTE (OpenMP), ce qui prouve
// que tout le pas AMR (flux, reflux coverage-aware, multigrille, regrid) compile et s'execute
// correctement par la voie GPU, sans materiel CUDA. Seuls la memoire device reelle et le MPI
// CUDA-aware restent a valider sur ROMEO.
//
// Gate : masse grossiere conservee a l'arrondi sous regrid dynamique, etat fini. Affiche aussi
// l'espace d'execution effectif et une somme de controle reproductible du grossier.

#include <adc/coupling/amr_coupler_mp.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef ADC_HAS_KOKKOS
#include <Kokkos_Core.hpp>
#endif

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main(int argc, char** argv) {
#ifdef ADC_HAS_KOKKOS
  Kokkos::initialize(argc, argv);
  const char* exec = Kokkos::DefaultExecutionSpace::name();
#else
  (void)argc; (void)argv;
  const char* exec = "serie-cpu";
#endif
  int rc = 0;
  {
    const int nc = 64;
    Box2D dom = Box2D::from_extents(nc, nc);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    const double dxc = geom.dx(), dyc = geom.dy(), dxf = dxc / 2, dyf = dyc / 2;
    BoxArray ba(std::vector<Box2D>{dom});
    DistributionMapping dm(1, n_ranks());
    BCRec bc;

    Diocotron model;
    model.B0 = 1.0; model.alpha = 1.0; model.n_i0 = 1.0;

    auto blob = [&](double x, double y) {
      return 1.0 + 0.6 * std::exp(-((x - 0.32) * (x - 0.32) + (y - 0.5) * (y - 0.5)) / 0.004) +
             0.6 * std::exp(-((x - 0.68) * (x - 0.68) + (y - 0.5) * (y - 0.5)) / 0.004);
    };
    auto crit = [&](const ConstArray4& a, int i, int j) { return a(i, j, 0) > model.n_i0 + 0.05; };
    Box2D seed{{2 * (nc / 4), 2 * (nc / 4)}, {2 * (3 * nc / 4) - 1, 2 * (3 * nc / 4) - 1}};
    MultiFab Uc(ba, dm, 1, 1), Uf(BoxArray(std::vector<Box2D>{seed}), dm, 1, 1);
    {
      Array4 u = Uc.fab(0).array();
      const Box2D g = Uc.fab(0).grown_box();
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = blob((i + 0.5) * dxc, (j + 0.5) * dyc);
      Array4 uf = Uf.fab(0).array();
      const Box2D b = Uf.box(0);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i) uf(i, j, 0) = blob((i + 0.5) * dxf, (j + 0.5) * dyf);
    }
    std::vector<AmrLevelMP> LP;
    LP.push_back({std::move(Uc), nullptr, dxc, dyc});
    LP.push_back({std::move(Uf), nullptr, dxf, dyf});
    AmrCouplerMP<Diocotron> sim(model, geom, ba, bc, std::move(LP));

    sim.regrid(crit);
    sim.update();
    const double m0 = sim.mass();
    const double dt = 0.4 * dxc / sim.max_drift_speed();
    bool finite = true;
    int npatch = 0;
    for (int s = 0; s < 80; ++s) {
      if (s % 10 == 0) sim.regrid(crit);
      sim.step(dt);
      if (!std::isfinite(sim.mass())) finite = false;
      npatch = sim.levels()[1].U.local_size();
    }
    const double drift = std::fabs(sim.mass() - m0);
    device_fence();
    double checksum = 0;  // somme de controle du grossier (acces hote apres fence)
    const ConstArray4 c = sim.coarse().fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) checksum += c(i, j, 0) * (i + 1) * (j + 1);

    std::printf("diocotron AMR sous le seam (%s) : npatch=%d derive_masse=%.3e checksum=%.6f %s\n",
                exec, npatch, drift, checksum, finite ? "fini" : "NON-FINI");
    if (!finite || drift > 1e-9) {
      std::printf("FAIL diocotron_amr_kokkos\n");
      rc = 1;
    } else {
      std::printf("OK diocotron_amr_kokkos\n");
    }
  }
#ifdef ADC_HAS_KOKKOS
  Kokkos::finalize();
#endif
  return rc;
}

// Validation DEVICE (GH200, Kokkos Cuda) du peuplement de B_z PAR NIVEAU dans le chemin AMR (#53).
// Le coupleur de systeme (AmrSystemCoupler::fill_bz) pose B_z(x,y) aux centres DE CHAQUE NIVEAU
// (dx = dx_coarse / 2^k) sur la composante kAuxBaseComps du canal aux PARTAGE, puis le modele lit
// a.B_z via load_aux<4> dans le noyau source AMR (for_each_cell POPS_HD -> device). On valide ICI
// ce CHEMIN DEVICE par le moteur header-only advance_amr, qui est exactement la primitive qui
// consomme B_z sur le device a chaque niveau (le meme que celui qu'appelle le coupleur niveau par
// niveau). NB : la facade AmrSystemCoupler ENTIERE est, elle, validee sous nvcc par le harness frere
// gpu_amrsys_facade_validate.cpp -- la limite "le concept CoupledSystemLike (requires
// s.for_each_block(...)) ne s'instancie pas sous nvcc/EDG" a ete LEVEE en passant la sonde du concept
// a un foncteur nomme (detail::ForEachBlockProbe), meme recette que les foncteurs nommes (#64).
//
// Le harness : B_z(x,y) = 1 + sin(2 pi x) cos(2 pi y) NON CONSTANT (depend vraiment du niveau via
// dx), pose par niveau comme fill_bz ; un modele BzGrow (n_aux=4, source S = B_z u) sur grossier +
// fin -> u(t+dt) depend de B_z LU AU BON NIVEAU. Imprime exec=, B_z relu par niveau, et les valeurs
// finales ; dump binaire de U(grossier+fin) -> diff_bin Cuda vs Serial (dmax sur chaque cellule).

#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>  // device_fence
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP, advance_amr
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

// Modele jouet pilote par B_z : flux nul, pas de couplage elliptique, source S = B_z * u.
// Declare n_aux=4 -> lit a.B_z (dependant du niveau via fill_bz / le peuplement par niveau ici).
struct BzGrow {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;
  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};
static_assert(PhysicalModel<BzGrow>);
static_assert(aux_comps<BzGrow>() == 4, "BzGrow declare n_aux=4");

// Peuple B_z(x,y) PAR NIVEAU sur la composante 3 du canal aux, exactement comme
// AmrSystemCoupler::fill_bz : centres DU NIVEAU (gk = geometrie du niveau, dx = dx_coarse / 2^k),
// valides + ghosts. Ecriture HOTE en memoire unifiee (B_z statique, pose une fois).
template <class F>
static void fill_bz_level(MultiFab& A, const Geometry& gk, F bz) {
  for (int li = 0; li < A.local_size(); ++li) {
    Fab2D& f = A.fab(li);
    const Box2D g = f.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        f(i, j, kAuxBaseComps) = bz(gk.x_cell(i), gk.y_cell(j));
  }
}
static void collect(const MultiFab& U, std::vector<double>& out) {
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 a = U.fab(li).const_array();
    const Box2D b = U.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        out.push_back(a(i, j, 0));
  }
}

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
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
#if defined(POPS_HAS_KOKKOS)
  const char* space = Kokkos::DefaultExecutionSpace::name();
#else
  const char* space = "Serial(host)";
#endif

  const int NC = 16;
  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};  // niveau 0
  const Geometry geomF = geom.refine(2);         // niveau 1 (dx/2)
  const BoxArray ba_coarse(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const Real dxc = geom.dx(), dyc = geom.dy();
  const Box2D fbox{{8, 8}, {23, 23}};  // patch fin sur [4..11]^2 grossier
  const BoxArray ba_fine(std::vector<Box2D>{fbox});

  auto bz = [](Real x, Real y) { return Real(1) + std::sin(2 * kPi * x) * std::cos(2 * kPi * y); };

  const Real u0 = Real(2);
  MultiFab Uc(ba_coarse, dm, 1, 2), Uf(ba_fine, dm, 1, 2);
  Uc.set_val(u0);
  Uf.set_val(u0);

  // canal aux ALLOUE A 4 comp (aux_comps<BzGrow>()) a CHAQUE niveau ; B_z pose PAR NIVEAU.
  MultiFab auxc(ba_coarse, dm, aux_comps<BzGrow>(), 1);
  MultiFab auxf(ba_fine, dm, aux_comps<BzGrow>(), 1);
  auxc.set_val(Real(0));
  auxf.set_val(Real(0));
  chk(auxc.ncomp() == 4 && auxf.ncomp() == 4, "amr_aux_width_4_each_level");
  fill_bz_level(auxc, geom, bz);   // niveau 0 : centres a dx
  fill_bz_level(auxf, geomF, bz);  // niveau 1 : centres a dx/2 -> valeurs DIFFERENTES

  // verifie que B_z a bien ete pose PAR NIVEAU : meme indice (i,j) -> valeur != entre niveaux.
  device_fence();
  {
    const ConstArray4 a0 = auxc.fab(0).const_array();
    const ConstArray4 a1 = auxf.fab(0).const_array();
    const Real bz0 = a0(4, 4, 3), bz1 = a1(8, 8, 3);
    const Real e0 = bz(geom.x_cell(4), geom.y_cell(4)), e1 = bz(geomF.x_cell(8), geomF.y_cell(8));
    std::printf(
        "[AMR Bz] exec=%s  bz_lvl0(4,4)=%.17g (exp %.17g)  bz_lvl1(8,8)=%.17g (exp %.17g)\n", space,
        bz0, e0, bz1, e1);
    chk(std::fabs(bz0 - e0) < 1e-12, "bz_level0_at_level0_centers");
    chk(std::fabs(bz1 - e1) < 1e-12, "bz_level1_at_level1_centers");
    chk(std::fabs(bz0 - bz1) > 1e-6, "bz_differs_between_levels");  // sinon le niveau n'est pas lu
  }

  std::vector<AmrLevelMP> L;
  L.push_back(AmrLevelMP{std::move(Uc), &auxc, dxc, dyc});
  L.push_back(AmrLevelMP{std::move(Uf), &auxf, dxc / 2, dyc / 2});

  // flux nul -> mf_advance_faces no-op ; mf_apply_source ajoute dt*B_z*u (Euler avant par sous-pas).
  // Grossier : 1 pas dt ; fin : 2 sous-pas dt/2. B_z lu AU NIVEAU via load_aux<4> dans le for_each
  // device. La source consomme B_z(niveau) -> u final depend du niveau.
  const Real dt = Real(0.05);
  advance_amr(BzGrow{}, L, dom, dt);
  device_fence();

  const Real bz0 = bz(geom.x_cell(0), geom.y_cell(0));
  const Real expect_c = u0 * (Real(1) + dt * bz0);  // coin (0,0) non couvert par le fin
  const Real half = dt / 2;
  const Real bzf = bz(geomF.x_cell(fbox.lo[0]), geomF.y_cell(fbox.lo[1]));
  const Real expect_f = u0 * (Real(1) + half * bzf) * (Real(1) + half * bzf);
  {
    const ConstArray4 uc = L[0].U.fab(0).const_array();
    const ConstArray4 uf = L[1].U.fab(0).const_array();
    const Real gotc = uc(0, 0, 0), gotf = uf(fbox.lo[0], fbox.lo[1], 0);
    std::printf("[AMR Bz] exec=%s  coarse(0,0)=%.17g (exp %.17g)  fine(lo)=%.17g (exp %.17g)\n",
                space, gotc, expect_c, gotf, expect_f);
    chk(std::fabs(gotc - expect_c) < 1e-12, "coarse_source_reads_level0_Bz");
    chk(std::fabs(gotf - expect_f) < 1e-12, "fine_source_reads_level1_Bz");
    chk(std::fabs(gotc - u0) > 1e-3, "Bz_actually_read");
  }

  if (!dump_prefix.empty()) {
    std::vector<double> v;
    collect(L[0].U, v);
    collect(L[1].U, v);
    const std::string path = dump_prefix + "_amrbz.bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f) {
      std::fwrite(v.data(), 1, v.size() * sizeof(double), f);
      std::fclose(f);
      std::printf("  dump %s (%zu doubles)\n", path.c_str(), v.size());
    }
  }

  if (fails == 0)
    std::printf("OK gpu_amr_bz_validate (exec=%s)\n", space);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::finalize();
#endif
  return fails == 0 ? 0 : 1;
}

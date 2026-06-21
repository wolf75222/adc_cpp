// AMR MULTI-BLOCS SOURCES COUPLEES (capstone vi) : la FACADE RUNTIME (AmrSystem -> AmrRuntime) expose
// une SOURCE COUPLEE inter-especes (DSL CoupledSource, bytecode P5) sur la hierarchie AMR PARTAGEE, en
// mirroir du moteur compile-time AmrSystemCoupler::coupled_source_step (splitting forward-Euler par
// niveau + cascade fin -> grossier #169) et du chemin non-AMR System::add_coupled_source (ABI plate).
//
// Ce que le test verrouille (cf. tache capstone vi) :
//   (1) ECHANGE CONSERVATIF PAR CELLULE : deux blocs ("ions" + "neutrals") sur une hierarchie 2 niveaux,
//       avec une source ionisation-like construite a la add_pair (+k*ni*ng sur un bloc, EXACTEMENT
//       -k*ni*ng sur l'autre, MEME cellule). La source SEULE (coupled_source_step direct, sans transport)
//       conserve n_ions + n_neutrals A LA CELLULE pres a ~machine ; l'etat reste FINI (rejet nan/inf
//       AVANT toute tolerance).
//   (2) ACTIVE : la source CHANGE l'etat (les deux blocs evoluent) vs un run sans source -> non no-op.
//   (3) CONSERVATION GLOBALE sous macro-pas complets : apres K step() (transport + source), la masse
//       composite n_ions + n_neutrals (somme leaf-aware par bloc) est conservee globalement a ~machine.
//   (4) COUVERTURE (#169) : apres la source, une cellule grossiere COUVERTE par le patch fin reste la
//       moyenne 2x2 de ses enfants fins (la cascade average_down a tourne).
//   (5) MULTIRATE : la conservation globale composite tient AUSSI avec un bloc stride=2 (la source doit
//       rester conservative sous le hold-then-catch-up).
//   (6) OPT-IN BIT-IDENTIQUE : un run multi-blocs SANS source couplee est bit-identique a avant
//       (dmax==0 entre deux runs ; et identique a un runtime equivalent jamais touche par la feature).
//
// On travaille au niveau du MOTEUR AmrRuntime + build_amr_block (les briques de cette PR), ou l'on
// construit la hierarchie partagee et l'on accede aux densites/masses par bloc et au RHS Poisson.

#include <adc/coupling/coupled_source_program.hpp>  // CsOp (opcodes du bytecode P5)
#include <adc/runtime/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <adc/runtime/amr_runtime.hpp>    // AmrRuntime, AmrRuntimeBlock
#include <adc/runtime/model_factory.hpp>  // detail::dispatch_model
#include <adc/runtime/model_spec.hpp>
#include <adc/mesh/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(ADC_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace adc;

// Spec ExB scalaire (1 var) a charge q : advection pilotee par grad phi (le couplage Poisson lit q*n).
// q=0 -> bloc neutre (ne contribue PAS au Poisson), advecte par le MEME phi que les autres.
static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// densite a moyenne nulle (solvable en periodique) : un creneau centre +/- amplitude, n*n row-major.
static std::vector<double> bump(int n, double base, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool in = (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4);
      r[static_cast<std::size_t>(j) * n + i] = base + (in ? amp : -amp / 3.0);
    }
  return r;
}

// tout fini (ni nan ni inf) : garde AVANT toute comparaison de tolerance (un nan passerait une borne).
static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

// ecart L-inf entre deux champs n*n (pour "A change" ou "X inchange").
static double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

// max sur les cellules de |a[c] + b[c] - (a0[c] + b0[c])| : ecart de la SOMME des deux blocs (par
// cellule) entre l'etat courant et un etat de reference. ~0 <=> echange conservatif par cellule.
static double dmax_pair_sum(const std::vector<double>& a, const std::vector<double>& b,
                            const std::vector<double>& a0, const std::vector<double>& b0) {
  double d = 0;
  const std::size_t nn = a.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs((a[i] + b[i]) - (a0[i] + b0[i])));
  return d;
}

// Construit un AmrRuntime a DEUX blocs ExB ("ions" q=+1, "neutrals" q=0) sur une hierarchie 2 niveaux
// figee N x N (un patch fin central), avec stride par bloc. Densites initiales rho0/rho1 sur le grossier.
static AmrRuntime make_two_block(int N, double L, double B0, const std::vector<double>& rho_ions,
                                 const std::vector<double>& rho_neut, int stride_ions,
                                 int stride_neut) {
  AmrBuildParams bp;
  bp.n = N;
  bp.L = L;
  bp.regrid_every = 0;      // hierarchie figee (multi-blocs)
  bp.poisson_bc = BCRec{};  // periodique
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_charge(+1.0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "ions", rho_ions,
                                                /*has_density=*/true, 1.4, 1, false, false,
                                                stride_ions));
  });
  detail::dispatch_model(exb_charge(0.0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "neutrals", rho_neut,
                                                /*has_density=*/true, 1.4, 1, false, false,
                                                stride_neut));
  });
  return AmrRuntime(S.geom, S.ba_coarse, S.poisson_bc, std::move(blocks), S.base_per,
                    S.replicated_coarse, S.wall);
}

// Enregistre une source ionisation-like CONSERVATIVE entre "ions" (gagne) et "neutrals" (perd) sur le
// role density, terme S = k * n_ions * n_neutrals. Bytecode postfixe construit A LA MAIN, EXACTEMENT
// comme l'emettrait adc.dsl.CoupledSource.add_pair (gain = +S ; perte = Neg(gain) = -S, MEME programme
// + un Neg) -> les deux contributions par cellule sont opposees au signe pres (echange conservatif).
//   registres : r0 = n_ions (entree), r1 = n_neutrals (entree), r2 = k (constante)
//   gain  : PushReg 0, PushReg 1, Mul, PushReg 2, Mul        -> k * n_ions * n_neutrals
//   perte : <gain> puis Neg                                  -> -(k * n_ions * n_neutrals)
static void register_ionization(AmrRuntime& rt, double k) {
  const std::vector<std::string> in_blocks = {"ions", "neutrals"};
  const std::vector<std::string> in_roles = {"density", "density"};
  const std::vector<double> consts = {k};
  const std::vector<std::string> out_blocks = {"ions", "neutrals"};
  const std::vector<std::string> out_roles = {"density", "density"};
  // programme du gain (5 opcodes) puis du gain+Neg (6 opcodes), concatenes.
  const int P = static_cast<int>(CsOp::PushReg), MUL = static_cast<int>(CsOp::Mul),
            NEG = static_cast<int>(CsOp::Neg);
  std::vector<int> prog_ops = {P, P, MUL, P, MUL,        // gain : k*ni*ng
                               P, P, MUL, P, MUL, NEG};  // perte : -(k*ni*ng)
  std::vector<int> prog_args = {0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0};
  std::vector<int> prog_lens = {5, 6};
  rt.add_coupled_source(in_blocks, in_roles, consts, out_blocks, out_roles, prog_ops, prog_args,
                        prog_lens);
}

int main(int argc, char** argv) {
#if defined(ADC_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    std::printf("  [%s] %s\n", c ? "OK " : "XX ", w);
    if (!c)
      ++fails;
  };

  const int N = 32;
  const double L = 1.0, B0 = 1.0, k = 0.5;
  const std::vector<double> rho_ions = bump(N, 1.0, 0.40);
  const std::vector<double> rho_neut = bump(N, 1.0, 0.20);
  const Real dt = Real(0.01);
  const int K = 6;  // macro-pas

  // ============================================================================================
  // (1) ECHANGE CONSERVATIF PAR CELLULE : la SOURCE SEULE (sans transport) conserve n_ions + n_neutrals
  //     a la cellule pres. On appelle coupled_source_step(dt) DIRECTEMENT (pas step) : seul l'echange
  //     additif agit, donc la somme des deux blocs ne doit PAS bouger, cellule par cellule. (2) ACTIVE :
  //     chaque bloc, lui, CHANGE (la source n'est pas un no-op).
  // ============================================================================================
  {
    AmrRuntime rt =
        make_two_block(N, L, B0, rho_ions, rho_neut, /*stride_ions=*/1, /*stride_neut=*/1);
    register_ionization(rt, k);
    const std::vector<double> ni0 = rt.density(0);  // ions, grossier
    const std::vector<double> ng0 = rt.density(1);  // neutrals, grossier
    rt.coupled_source_step(dt);                     // SOURCE SEULE (pas de transport)
    const std::vector<double> ni1 = rt.density(0);
    const std::vector<double> ng1 = rt.density(1);

    chk(all_finite(ni1) && all_finite(ng1), "cell_state_finite");  // AVANT toute tolerance
    // echange conservatif : (n_ions + n_neutrals) inchange PAR CELLULE (+S sur l'un, -S sur l'autre).
    chk(dmax_pair_sum(ni1, ng1, ni0, ng0) < 1e-12, "cell_pair_sum_conserved");
    // ACTIVE : ions GAGNENT (+k ni ng > 0), neutrals PERDENT -> les deux changent (non no-op).
    chk(dmax_field(ni1, ni0) > 1e-9, "source_active_ions_change");
    chk(dmax_field(ng1, ng0) > 1e-9, "source_active_neutrals_change");
  }

  // ============================================================================================
  // (3) CONSERVATION GLOBALE sous macro-pas complets : apres K step() (transport AMR + source apres le
  //     transport), la masse composite (somme leaf-aware par bloc) n_ions + n_neutrals est conservee
  //     globalement a ~machine. Chaque masse de bloc, elle, DERIVE (la source transfere entre blocs).
  // ============================================================================================
  {
    AmrRuntime rt =
        make_two_block(N, L, B0, rho_ions, rho_neut, /*stride_ions=*/1, /*stride_neut=*/1);
    register_ionization(rt, k);
    const Real tot0 = rt.mass(0) + rt.mass(1);
    const Real mi0 = rt.mass(0);
    for (int s = 0; s < K; ++s)
      rt.step(dt);
    const std::vector<double> ni = rt.density(0);
    const std::vector<double> ng = rt.density(1);
    const Real tot1 = rt.mass(0) + rt.mass(1);
    chk(all_finite(ni) && all_finite(ng), "global_state_finite");
    chk(std::fabs(tot1 - tot0) < 1e-9, "global_composite_mass_conserved");
    // la source TRANSFERE vraiment entre blocs : la masse des ions a sensiblement augmente.
    chk(rt.mass(0) > mi0 + Real(1e-6), "global_source_transfers_to_ions");
  }

  // ============================================================================================
  // (4) COUVERTURE (#169) : apres la source, une cellule grossiere COUVERTE par le patch fin reste la
  //     moyenne 2x2 de ses enfants fins. On lit le niveau 0 (grossier) et le niveau 1 (patch fin) du
  //     bloc ions, et on verifie l'invariant average_down sur les cellules couvertes. Le patch fin
  //     central couvre les cellules grossieres [N/4 .. 3N/4-1]^2 (cf. make_shared_amr_layout).
  // ============================================================================================
  {
    AmrRuntime rt =
        make_two_block(N, L, B0, rho_ions, rho_neut, /*stride_ions=*/1, /*stride_neut=*/1);
    register_ionization(rt, k);
    rt.coupled_source_step(
        dt);  // SOURCE SEULE -> l'invariant de couverture vient de la cascade post-source
    const MultiFab& Uc = rt.levels(0)[0].U;  // grossier (ions)
    const MultiFab& Uf = rt.levels(0)[1].U;  // patch fin (ions)
    // moyenne 2x2 des enfants fins vs valeur grossiere couverte, sur la box fine (mono-box replique).
    const Box2D vf = Uf.box(0);
    const ConstArray4 uf = Uf.fab(0).const_array();
    // grossier : trouver le fab couvrant la cellule grossiere ic,jc. Mono-box replique -> fab(0).
    const ConstArray4 uc = Uc.fab(0).const_array();
    double dmax_cover = 0.0;
    bool covered_seen = false;
    for (int jf = vf.lo[1]; jf <= vf.hi[1]; jf += 2)
      for (int if_ = vf.lo[0]; if_ <= vf.hi[0]; if_ += 2) {
        const int ic = if_ / 2, jc = jf / 2;  // cellule grossiere parente (ratio 2)
        const double avg = 0.25 * (uf(if_, jf, 0) + uf(if_ + 1, jf, 0) + uf(if_, jf + 1, 0) +
                                   uf(if_ + 1, jf + 1, 0));
        dmax_cover = std::max(dmax_cover, std::fabs(uc(ic, jc, 0) - avg));
        covered_seen = true;
      }
    chk(covered_seen, "cover_has_fine_patch");
    chk(dmax_cover < 1e-12, "cover_coarse_is_avg_of_fine_after_source");
  }

  // ============================================================================================
  // (5) MULTIRATE : la source reste conservative GLOBALEMENT avec un bloc neutrals stride=2 (tenu au
  //     mac0, rattrape au mac1). La source est appliquee a chaque macro-pas APRES le transport ; meme
  //     quand neutrals est tenu, l'echange +S/-S par cellule conserve la somme composite globale.
  // ============================================================================================
  {
    AmrRuntime rt =
        make_two_block(N, L, B0, rho_ions, rho_neut, /*stride_ions=*/1, /*stride_neut=*/2);
    register_ionization(rt, k);
    const Real tot0 = rt.mass(0) + rt.mass(1);
    for (int s = 0; s < K; ++s)
      rt.step(dt);
    chk(all_finite(rt.density(0)) && all_finite(rt.density(1)), "multirate_state_finite");
    chk(std::fabs((rt.mass(0) + rt.mass(1)) - tot0) < 1e-9, "multirate_composite_mass_conserved");
  }

  // ============================================================================================
  // (6) OPT-IN BIT-IDENTIQUE : un run multi-blocs SANS source couplee est INCHANGE. Deux runs sans
  //     source coincident au bit pres (determinisme), et un run AVEC source DIFFERE d'un run sans
  //     source -> la feature est strictement opt-in (aucun effet tant qu'aucune source n'est ajoutee).
  // ============================================================================================
  {
    auto run_no_source = [&]() {
      AmrRuntime rt = make_two_block(N, L, B0, rho_ions, rho_neut, 1, 1);
      for (int s = 0; s < K; ++s)
        rt.step(dt);
      return rt.density(0);
    };
    const std::vector<double> a = run_no_source();
    const std::vector<double> b = run_no_source();
    chk(dmax_field(a, b) == 0.0, "no_source_bit_identical");

    AmrRuntime rt_src = make_two_block(N, L, B0, rho_ions, rho_neut, 1, 1);
    register_ionization(rt_src, k);
    for (int s = 0; s < K; ++s)
      rt_src.step(dt);
    chk(dmax_field(rt_src.density(0), a) > 1e-9, "source_differs_from_no_source");
    chk(rt_src.n_coupled_sources() == 1, "one_coupled_source_registered");
  }

  if (fails == 0)
    std::printf("OK test_amr_multiblock_coupled_source\n");
  else
    std::printf("FAIL test_amr_multiblock_coupled_source : %d echec(s)\n", fails);
  return fails == 0 ? 0 : 1;
}

// AMR SOURCES COUPLEES : RESOLUTION DE ROLE STRICTE (P0, mirroir de System #181 / fb02e1f).
//
// Le chemin DSL coupled-source du moteur runtime AMR (AmrRuntime::add_coupled_source, in/out_blocks +
// in/out_roles) resout chaque (bloc, role) en (indice de bloc, composante) via le descripteur conservatif
// du bloc (cons_vars.index_of(role)). Le faux-silencieux corrige : un role CANONIQUE mais NON EXPOSE par
// le bloc retombait sur la composante 0 (comp = ix >= 0 ? ix : 0), donc la source s'appliquait au MAUVAIS
// champ EN SILENCE. La resolution devient STRICTE (mirroir exact de System::add_coupled_source, #181) :
// index_of(role) < 0 -> throw explicite nommant le bloc ET le role.
//
// Ce que ce test verrouille (les trois cas demandes) :
//   (A) role VALIDE expose par le bloc (density sur un bloc scalaire ExB) -> add_coupled_source REUSSIT.
//   (B) role ABSENT (canonique mais non expose : momentum_x sur un bloc scalaire) -> LEVE, avec le NOM du
//       bloc ET le NOM du role dans le message (plus aucun repli silencieux sur la composante 0).
//   (C) role INCONNU (non canonique, p.ex. "bogus_role") -> LEVE (comportement preexistant conserve).
//   (D) role UTILISATEUR nomme ("charge", label ajoute sur la density) -> resolu de bout en bout par
//       add_coupled_source via la couche string (index_of(string), ADC-292).
//   (E) LABEL utilisateur ABSENT -> LEVE en nommant le role (pas de repli silencieux sur la comp 0).
//
// On travaille au niveau du moteur AmrRuntime + build_amr_block, comme test_amr_multiblock_coupled_source :
// deux blocs ExB scalaires (1 var, role density) sur la hierarchie partagee, puis on enregistre des sources
// couplees minimales et on assertit le succes / la levee. Backend CPU/Kokkos (ctor sous ScopeGuard).

#include <pops/coupling/source/coupled_source_program.hpp>  // CsOp (opcodes du bytecode P5)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <pops/runtime/amr/amr_runtime.hpp>    // AmrRuntime, AmrRuntimeBlock
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model
#include <pops/runtime/config/model_spec.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// Spec ExB scalaire (1 var, role density) a charge q. Bloc scalaire : conservative_vars() == {density},
// donc momentum_x est CANONIQUE mais NON EXPOSE -> support du cas (B).
static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// densite a moyenne nulle (solvable en periodique) : creneau centre, n*n row-major.
static std::vector<double> bump(int n, double base, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool in = (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4);
      r[static_cast<std::size_t>(j) * n + i] = base + (in ? amp : -amp / 3.0);
    }
  return r;
}

// Construit un AmrRuntime a DEUX blocs ExB scalaires ("ions" q=+1, "neutrals" q=0) sur une hierarchie
// 2 niveaux figee N x N (un patch fin central). Calque de test_amr_multiblock_coupled_source.
static AmrRuntime make_two_block(int N, double L, double B0, const std::vector<double>& rho_ions,
                                 const std::vector<double>& rho_neut,
                                 const char* ions_user_role = nullptr) {
  AmrBuildParams bp;
  bp.n = N;
  bp.L = L;
  bp.regrid_every = 0;      // hierarchie figee (multi-blocs)
  bp.poisson_bc = BCRec{};  // periodique
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_charge(+1.0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "ions", rho_ions,
                                                /*has_density=*/true, 1.4, 1, false, false, 1));
  });
  detail::dispatch_model(exb_charge(0.0, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "neutrals", rho_neut,
                                                /*has_density=*/true, 1.4, 1, false, false, 1));
  });
  // ADC-292: optionally ADD a USER-DEFINED role label on the "ions" density component (keeping the
  // canonical Density role, so the ExB charge coupling is untouched). The coupled-source resolver must
  // then go through the string user-role layer (index_of(string)) to map this label to component 0.
  // Pure host METADATA: the numerics still act on component 0, only (block, role) resolution changes.
  if (ions_user_role != nullptr)
    blocks[0].cons_vars.user_roles = {ions_user_role};
  return AmrRuntime(S.geom, S.ba_coarse, S.poisson_bc, std::move(blocks), S.base_per,
                    S.replicated_coarse, S.wall);
}

// Source minimale a UN terme : out_role recu sur le bloc "ions", lisant density des deux blocs.
// terme = k * n_ions * n_neutrals (5 opcodes : PushReg 0, PushReg 1, Mul, PushReg 2, Mul). On fait varier
// SEULEMENT out_roles[0] entre les cas (valide / absent / inconnu) ; les entrees restent valides (density).
static void add_source_with_out_role(AmrRuntime& rt, const std::string& out_role) {
  const std::vector<std::string> in_blocks = {"ions", "neutrals"};
  const std::vector<std::string> in_roles = {"density", "density"};
  const std::vector<double> consts = {0.5};
  const std::vector<std::string> out_blocks = {"ions"};
  const std::vector<std::string> out_roles = {out_role};
  const int P = static_cast<int>(CsOp::PushReg), MUL = static_cast<int>(CsOp::Mul);
  const std::vector<int> prog_ops = {P, P, MUL, P, MUL};  // k * n_ions * n_neutrals
  const std::vector<int> prog_args = {0, 1, 0, 2, 0};
  const std::vector<int> prog_lens = {5};
  rt.add_coupled_source(in_blocks, in_roles, consts, out_blocks, out_roles, prog_ops, prog_args,
                        prog_lens);
}

// Comme add_source_with_out_role mais l'entree lue sur "ions" ET la sortie sont adressees par @p io_role
// (un LABEL de role utilisateur). "neutrals" lit toujours sa density canonique. Sert les cas (D)/(E) :
// resolution d'un role utilisateur nomme de bout en bout via add_coupled_source -> index_of(string).
static void add_source_with_io_role(AmrRuntime& rt, const std::string& io_role) {
  const std::vector<std::string> in_blocks = {"ions", "neutrals"};
  const std::vector<std::string> in_roles = {io_role, "density"};
  const std::vector<double> consts = {0.5};
  const std::vector<std::string> out_blocks = {"ions"};
  const std::vector<std::string> out_roles = {io_role};
  const int P = static_cast<int>(CsOp::PushReg), MUL = static_cast<int>(CsOp::Mul);
  const std::vector<int> prog_ops = {P, P, MUL, P, MUL};  // k * n_ions * n_neutrals
  const std::vector<int> prog_args = {0, 1, 0, 2, 0};
  const std::vector<int> prog_lens = {5};
  rt.add_coupled_source(in_blocks, in_roles, consts, out_blocks, out_roles, prog_ops, prog_args,
                        prog_lens);
}

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
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
  const double L = 1.0, B0 = 1.0;
  const std::vector<double> rho_ions = bump(N, 1.0, 0.40);
  const std::vector<double> rho_neut = bump(N, 1.0, 0.20);

  // ============================================================================================
  // (A) role VALIDE (density, EXPOSE par le bloc scalaire) -> add_coupled_source REUSSIT.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, B0, rho_ions, rho_neut);
    bool threw = false;
    try {
      add_source_with_out_role(rt, "density");
    } catch (const std::exception&) {
      threw = true;
    }
    chk(!threw, "valid_role_density_resolves");
    chk(rt.n_coupled_sources() == 1, "valid_role_source_registered");
  }

  // ============================================================================================
  // (B) role ABSENT (momentum_x : CANONIQUE mais NON EXPOSE par un bloc scalaire) -> LEVE, avec le NOM
  //     du bloc ET le NOM du role dans le message (plus de repli silencieux sur la composante 0).
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, B0, rho_ions, rho_neut);
    bool threw = false;
    bool names_block = false, names_role = false;
    try {
      add_source_with_out_role(rt, "momentum_x");
    } catch (const std::exception& e) {
      threw = true;
      const std::string msg = e.what();
      names_block = msg.find("ions") != std::string::npos;       // le bloc cible
      names_role = msg.find("momentum_x") != std::string::npos;  // le role absent
    }
    chk(threw, "absent_role_raises");
    chk(names_block, "absent_role_message_names_block");
    chk(names_role, "absent_role_message_names_role");
    // STRICT : aucune source enregistree (pas de repli silencieux qui aurait stocke comp 0).
    chk(rt.n_coupled_sources() == 0, "absent_role_no_source_registered");
  }

  // ============================================================================================
  // (C) role INCONNU (non canonique, "bogus_role") -> LEVE (comportement preexistant conserve).
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, B0, rho_ions, rho_neut);
    bool threw = false;
    try {
      add_source_with_out_role(rt, "bogus_role");
    } catch (const std::exception&) {
      threw = true;
    }
    chk(threw, "unknown_role_raises");
    chk(rt.n_coupled_sources() == 0, "unknown_role_no_source_registered");
  }

  // ============================================================================================
  // (D) role UTILISATEUR nomme ("charge", ajoute en label sur la density de "ions") resolu DE BOUT EN
  //     BOUT par add_coupled_source via la couche string (index_of(string), ADC-292) -> REUSSIT.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, B0, rho_ions, rho_neut, /*ions_user_role=*/"charge");
    bool threw = false;
    try {
      add_source_with_io_role(rt, "charge");
    } catch (const std::exception&) {
      threw = true;
    }
    chk(!threw, "user_role_label_resolves_end_to_end");
    chk(rt.n_coupled_sources() == 1, "user_role_source_registered");
  }

  // ============================================================================================
  // (E) LABEL utilisateur ABSENT (le bloc declare "charge", pas "not_a_label") -> LEVE en nommant le
  //     role, sans repli silencieux sur la composante 0.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, B0, rho_ions, rho_neut, /*ions_user_role=*/"charge");
    bool threw = false, names_role = false;
    try {
      add_source_with_io_role(rt, "not_a_label");
    } catch (const std::exception& e) {
      threw = true;
      names_role = std::string(e.what()).find("not_a_label") != std::string::npos;
    }
    chk(threw, "absent_user_label_raises");
    chk(names_role, "absent_user_label_message_names_role");
    chk(rt.n_coupled_sources() == 0, "absent_user_label_no_source_registered");
  }

  if (fails == 0)
    std::printf("OK test_amr_coupled_source_role_strict\n");
  else
    std::printf("FAIL test_amr_coupled_source_role_strict : %d echec(s)\n", fails);
  return fails == 0 ? 0 : 1;
}

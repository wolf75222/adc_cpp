// Coupled multi-block field solve (Spec 3 section 12.3, criterion 24; ADC-457).
//
// SystemFieldSolver::assemble_poisson_rhs_from_blocks assembles the system Poisson RHS as
// f = Sum_s elliptic_rhs_s(U_s) reading EVERY block's stage state at once (indexed by block index;
// a nullptr entry uses the block's live state) -- the SIMULTANEOUS multi-target counterpart of the
// single-target assemble_poisson_rhs. System::solve_fields_from_blocks wraps it (solve + aux derive),
// the seam the compiled-Program codegen lowers P.solve_fields_from_blocks([...]) to (ProgramContext).
//
// This test exercises that path through the PUBLIC System API (assemble_poisson_rhs_* are private to
// the templated field solver). Two charge (ExB) blocks with distinct mean-zero densities:
//   (a) SUM over all blocks: solve_fields_from_blocks({&U0, &U1}) (every block at its LIVE state) gives
//       a potential matching the historical solve_fields() to round-off -- both assemble Sum_s
//       elliptic_rhs_s(s.U). This is the "RHS == sum of the per-block contributions" assertion. (Not
//       bit-for-bit: the GeometricMG is iterative + warm-started, so a redundant solve stops on a
//       relative tolerance -- the result matches to ~ulp*|phi|, not exactly.)
//   (b) PER-BLOCK stage override: with block 1's slot pointing at a stage state whose charge is ZEROED,
//       the potential matches (to round-off) a reference where block 1's LIVE density is zero (only
//       block 0 contributes) -- so block 1 read its STAGE override, not its live state (the per-block
//       sum is honored per slot, the coupled commit_many guarantee), and differs from the all-live
//       solve by far more than the tolerance.
//   (c) SIZE guard: a wrong-sized U_stages vector throws (a stale binding cannot silently mis-route).
//
// Serial (host) test: the elliptic solve runs on the local box; no MPI/Kokkos device path is exercised
// beyond the standard System build. The compiled .so running this coupled solve in a step loop is
// validated on ROMEO (Kokkos-only AOT).

#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/storage/fab2d.hpp>

#include "test_harness.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

// A charge density with ZERO mean (a periodic Poisson is solvable only for a mean-zero RHS): a smooth
// cosine bump pattern, distinct per block via the @p phase shift so the two blocks carry different
// charge. n*n row-major (j slow, i fast).
std::vector<double> charge_density(int n, double amp, double phase) {
  std::vector<double> q(static_cast<std::size_t>(n) * n, 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n;
      const double y = (j + 0.5) / n;
      q[static_cast<std::size_t>(j) * n + i] =
          amp * std::cos(2.0 * test::kPi * (x + phase)) * std::cos(2.0 * test::kPi * y);
    }
  return q;  // cos over a full period has zero mean
}

// Builds a 2-block charge System (both ExB transport, charge elliptic brick) ready for solve_fields.
// Blocks declared "n0" then "n1" -> runtime indices 0 and 1 (the order the coupled vector expects).
void build_two_charge_blocks(System& s) {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = 1.0;
  spec.B0 = 1.0;
  s.add_block("n0", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
  s.add_block("n1", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
  s.set_poisson("composite", "geometric_mg");  // f = sum of the per-block elliptic bricks
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    return 1e300;  // a size mismatch is a hard failure (compared against 0 by the caller)
  double d = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  test::Checker chk(test::Checker::Style::Verbose);

  const int n = 32;
  const SystemConfig cfg{n, 1.0, true};  // periodic Cartesian
  const std::vector<double> q0 = charge_density(n, 1.0, 0.0);
  const std::vector<double> q1 = charge_density(n, 0.6, 0.25);  // distinct from block 0

  // (a) SUM over all blocks: a coupled solve from the LIVE states == historical solve_fields ----------
  System s(cfg);
  build_two_charge_blocks(s);
  s.set_density("n0", q0);
  s.set_density("n1", q1);
  chk(s.n_blocks() == 2, "two blocks installed");

  s.solve_fields();  // historical: f = elliptic_rhs(n0.U) + elliptic_rhs(n1.U)
  const std::vector<double> phi_ref = s.potential();

  MultiFab& U0 = s.block_state(0);
  MultiFab& U1 = s.block_state(1);
  std::vector<const MultiFab*> stages_live{&U0, &U1};
  s.solve_fields_from_blocks(stages_live);  // coupled: every block at its live state
  const std::vector<double> phi_blocks = s.potential();

  chk(!phi_ref.empty() && phi_ref.size() == static_cast<std::size_t>(n) * n, "potential size");
  bool finite = true;
  for (double v : phi_blocks)
    finite = finite && std::isfinite(v);
  chk(finite, "coupled potential is finite");
  double maxabs = 0.0;
  for (double v : phi_ref)
    maxabs = std::fmax(maxabs, std::fabs(v));
  // The two solves assemble the SAME RHS (Sum_s elliptic_rhs_s(s.U)); the GeometricMG is iterative and
  // WARM-STARTED, so the second solve resumes from the first's converged phi and the V-cycle stops on a
  // RELATIVE tolerance -- the result matches to round-off, not bit-for-bit (a redundant iterative solve
  // is rarely a true no-op). We assert a tight relative agreement (~few ulp * |phi|): proves the
  // from-blocks RHS == the historical sum, the field-solve numerics are identical.
  const double d_sum = max_abs_diff(phi_blocks, phi_ref);
  const double tol = 1e-12 * std::fmax(maxabs, 1.0);
  std::printf("  d_sum=%.3e (tol=%.3e, |phi|max=%.3e)\n", d_sum, tol, maxabs);
  chk(d_sum <= tol,
      "coupled solve from live states matches solve_fields to round-off (RHS == sum of blocks)");
  // The potential is non-trivial (a genuine solve, not a zero field) -- guards against a vacuous pass.
  chk(maxabs > 0.0, "the coupled solve produces a non-trivial potential");

  // (b) PER-BLOCK stage override: block 1 reads a ZEROED stage state (drops its contribution) ----------
  // Reference: only block 0 contributes (block 1's live density zeroed), via the historical path.
  System ref(cfg);
  build_two_charge_blocks(ref);
  ref.set_density("n0", q0);
  ref.set_density("n1", std::vector<double>(static_cast<std::size_t>(n) * n, 0.0));  // block 1 = 0
  ref.solve_fields();
  const std::vector<double> phi_only0 = ref.potential();

  // Coupled solve on the original system: block 0 at its live state, block 1 at a ZEROED stage copy.
  MultiFab stage1 = s.block_state(1);  // deep copy of block 1's live state (same ba/dm layout)
  stage1.set_val(Real(0));             // zero the stage charge
  std::vector<const MultiFab*> stages_override{&s.block_state(0), &stage1};
  s.solve_fields_from_blocks(stages_override);
  const std::vector<double> phi_override = s.potential();

  const double d_override = max_abs_diff(phi_override, phi_only0);
  const double d_vs_all = max_abs_diff(phi_override, phi_blocks);
  std::printf("  d_override=%.3e (tol=%.3e)  d_vs_all=%.3e\n", d_override, tol, d_vs_all);
  // Same RHS as the only-block-0 reference (block 1 contributes zero), warm-started GeometricMG -> a
  // round-off match (see the d_sum note), not bit-for-bit.
  chk(d_override <= tol,
      "block 1's ZEROED stage override drops its charge (== only-block-0 reference, to round-off)");
  // And it differs from the all-blocks solve by FAR more than the tolerance (block 1's live charge
  // mattered) -- the override was honored, not ignored.
  chk(d_vs_all > 1e-4, "the stage override changes the potential vs the all-live coupled solve");

  // (b2) ALL-nullptr U_stages: every slot falls back to its block's LIVE state == the all-live solve --
  std::vector<const MultiFab*> stages_null{nullptr, nullptr};
  s.solve_fields_from_blocks(stages_null);
  const std::vector<double> phi_null = s.potential();
  chk(max_abs_diff(phi_null, phi_blocks) <= tol,
      "an all-nullptr U_stages falls back to every block's live state (== the all-live coupled solve)");

  // (d) eps != 1: the coupled path scales the system RHS by 1/eps just like solve_fields -------------
  // The constant-permittivity branch (p_eps_ != 1) is the one RHS-scaling branch the eps=1 cases above
  // never touch; assert the coupled solve honors it identically to the historical single-target solve.
  System se(cfg);
  {
    ModelSpec spec;
    spec.transport = "exb";
    spec.source = "none";
    spec.elliptic = "charge";
    spec.q = 1.0;
    spec.B0 = 1.0;
    se.add_block("n0", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
    se.add_block("n1", spec, "minmod", "rusanov", "conservative", "explicit", 1, true);
    se.set_poisson("composite", "geometric_mg", "auto", "none", 0.0, 2.0, 0.0);  // eps = 2
  }
  se.set_density("n0", q0);
  se.set_density("n1", q1);
  se.solve_fields();  // historical, with eps = 2
  const std::vector<double> phi_eps_ref = se.potential();
  std::vector<const MultiFab*> stages_eps{&se.block_state(0), &se.block_state(1)};
  se.solve_fields_from_blocks(stages_eps);
  const std::vector<double> phi_eps_blocks = se.potential();
  double eps_maxabs = 0.0;
  for (double v : phi_eps_ref)
    eps_maxabs = std::fmax(eps_maxabs, std::fabs(v));
  const double eps_tol = 1e-12 * std::fmax(eps_maxabs, 1.0);
  chk(eps_maxabs > 0.0, "the eps != 1 coupled solve produces a non-trivial potential");
  chk(max_abs_diff(phi_eps_blocks, phi_eps_ref) <= eps_tol,
      "with eps != 1 the coupled solve scales the RHS by 1/eps like solve_fields (to round-off)");

  // (c) SIZE guard: a U_stages not sized to n_blocks() throws (fail-loud on a stale binding) ----------
  bool threw = false;
  try {
    std::vector<const MultiFab*> bad{&s.block_state(0)};  // size 1 != 2 blocks
    s.solve_fields_from_blocks(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  chk(threw, "a U_stages not sized to n_blocks() throws std::invalid_argument");

  if (!chk.failed())
    std::printf("OK test_coupled_fieldsolve\n");
  return chk.failed();
}

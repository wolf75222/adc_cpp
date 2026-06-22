// ADC-291: the JIT host-residual path (native_loader.hpp host_residual, used by eval_rhs on a
// dynamic/JIT block) must transport model-NAMED aux fields (extra[k] = aux component
// kAuxNamedBase+k), not only the canonical extras (B_z/T_e from ADC_AUX_FIELDS). Before the fix the
// per-cell aux marshaling looped ONLY over ADC_AUX_FIELDS, so a model reading aux.extra_field(0) on
// the host path read 0 SILENTLY (#51-class gap, but for named fields). This pins the marshaling.

#include <adc/core/state.hpp>
#include <adc/runtime/detail/dynamic_model.hpp>
#include <adc/runtime/builders/native_loader.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

// Scalar model, flux = 0, source S = extra_field(0) * u. Reads ONE model-named aux field, so
// n_aux = kAuxNamedBase + 1 (= 6). With flux 0, host_residual returns exactly the source.
struct KappaSrc final : IModel<1> {
  using State = StateVec<1>;
  State flux(const State&, const Aux&, int) const override { return State{Real(0)}; }
  Real max_wave_speed(const State&, const Aux&, int) const override { return Real(0); }
  State source(const State& u, const Aux& a) const override {
    return State{a.extra_field(0) * u[0]};
  }
  int n_aux() const override { return kAuxNamedBase + 1; }
};

int main() {
  int fails = 0;
  const int n = 8;
  const double dx = 1.0 / n;
  const std::size_t nn = static_cast<std::size_t>(n) * n;

  KappaSrc m;
  std::vector<double> U(nn, 1.0);  // u = 1 everywhere

  // (A) wide channel: named field at component kAuxNamedBase (=5) = 0.7 -> residual = 0.7 * u = 0.7.
  const double kappa = 0.7;
  std::vector<double> AUX(static_cast<std::size_t>(kAuxNamedBase + 1) * nn, 0.0);
  for (std::size_t c = 0; c < nn; ++c)
    AUX[static_cast<std::size_t>(kAuxNamedBase) * nn + c] = kappa;
  std::vector<double> R = native_loader::host_residual<1>(m, U, AUX, n, dx, /*recon=*/0);
  double maxerr = 0.0;
  for (double v : R)
    maxerr = std::fmax(maxerr, std::fabs(v - kappa));
  std::printf("  (A) host_residual named aux : max|R - kappa| = %.2e\n", maxerr);
  if (maxerr > 1e-14) {
    std::printf("FAIL named_aux_marshaled_host\n");
    ++fails;
  }

  // (B) narrow channel (base 3 only): the named field is absent -> read as 0 -> residual 0
  // (documented: a field the channel does not carry stays 0, no out-of-bounds read).
  std::vector<double> AUXnarrow(3 * nn, 0.0);
  std::vector<double> R0 = native_loader::host_residual<1>(m, U, AUXnarrow, n, dx, 0);
  double maxz = 0.0;
  for (double v : R0)
    maxz = std::fmax(maxz, std::fabs(v));
  if (maxz != 0.0) {
    std::printf("FAIL narrow_channel_named_zero (max=%.2e)\n", maxz);
    ++fails;
  }

  if (fails == 0)
    std::printf("OK test_native_aux_named\n");
  return fails == 0 ? 0 : 1;
}

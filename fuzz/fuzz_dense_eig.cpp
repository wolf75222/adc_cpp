/// @file
/// @brief Harnais libFuzzer : robustesse + invariants de real_eig_minmax (spectre d'un petit
/// bloc dense, consommateur : codegen DSL wave_speeds_from_jacobian).
///
/// Deux modes pilotes par le premier octet :
///  - fini : entrees bornees (toujours finies, amplitude 1e3) -> invariants verifies quand
///    converged : lmin <= lmax, finitude, spectre dans l'encadrement de Gershgorin (tolerance) ;
///  - brut : 8 octets -> double tel quel (NaN/Inf/denormaux compris) -> seul le no-crash/no-UB
///    est exige (ASan + UBSan sont lies au harnais, -fno-sanitize-recover rend l'UB fatal).

#include <adc/numerics/dense_eig.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "fuzz_util.hpp"

namespace {

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "INVARIANT VIOLE : %s\n", what);
    std::abort();
  }
}

template <int N>
void run(ByteReader& br, bool raw_mode) {
  adc::Real a[N][N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      a[i][j] = raw_mode ? br.raw() : br.finite(1.0e3);
    }
  }

  const adc::EigBounds e = adc::real_eig_minmax(a);

  if (raw_mode) {
    return;  // NaN/Inf en entree : le contrat est "pas de crash, pas d'UB", rien de plus.
  }

  if (e.converged) {
    expect(std::isfinite(e.lmin) && std::isfinite(e.lmax), "lmin/lmax finis quand converged");
    expect(e.lmin <= e.lmax, "lmin <= lmax quand converged");
    expect(e.max_im >= adc::Real(0), "max_im >= 0 quand converged");

    // Gershgorin contient mathematiquement le spectre ; tolerance relative generouse pour
    // l'arithmetique flottante de l'iteration QR (sinon faux positifs).
    adc::Real lo = adc::Real(0), hi = adc::Real(0);
    adc::detail::gershgorin_bounds(a, lo, hi);
    const adc::Real tol =
        adc::Real(1e-6) * (adc::Real(1) + std::fabs(lo) + std::fabs(hi));
    expect(e.lmin >= lo - tol && e.lmax <= hi + tol,
           "spectre dans l'encadrement de Gershgorin (a tolerance pres)");
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  ByteReader br{data, size};
  const bool raw_mode = br.boolean();
  switch (br.range(0, 3)) {
    case 0: run<2>(br, raw_mode); break;
    case 1: run<3>(br, raw_mode); break;
    case 2: run<4>(br, raw_mode); break;
    default: run<8>(br, raw_mode); break;
  }
  return 0;
}

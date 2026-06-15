/// @file
/// @brief Mini consommateur d'octets pour les harnais libFuzzer (equivalent minimal de
/// FuzzedDataProvider, vendorise pour ne dependre d'aucun en-tete interne LLVM).
///
/// Deterministe et total : une entree epuisee rend des zeros (jamais d'UB cote harnais, la
/// reproduction d'un crash depuis son artefact est exacte).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ByteReader {
  const std::uint8_t* p = nullptr;
  std::size_t n = 0;
  std::uint32_t bitbuf = 0;
  int bitcnt = 0;

  /// Octet suivant (0 si l'entree est epuisee).
  std::uint8_t byte() { return n > 0 ? (--n, *p++) : std::uint8_t(0); }

  /// Entier dans [lo, hi] (bornes incluses) ; consomme 4 octets. Le biais du modulo est
  /// negligeable pour des plages petites devant 2^32 (toutes celles des harnais).
  int range(int lo, int hi) {
    if (hi <= lo) {
      return lo;
    }
    std::uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      v = (v << 8) | byte();
    }
    const std::uint32_t span = std::uint32_t(hi - lo) + 1u;
    return lo + int(v % span);
  }

  /// Un bit (pool separe du flux range/raw : 8 marqueurs par octet consomme).
  bool bit() {
    if (bitcnt == 0) {
      bitbuf = byte();
      bitcnt = 8;
    }
    const bool b = (bitbuf & 1u) != 0;
    bitbuf >>= 1;
    --bitcnt;
    return b;
  }

  bool boolean() { return (byte() & 1u) != 0; }

  /// double TOUJOURS FINI dans [-amp, amp] (reconstruit d'un int16 : jamais NaN/Inf).
  double finite(double amp) {
    const int v = range(-32768, 32767);
    return amp * (double(v) / 32768.0);
  }

  /// double BRUT (bit-pattern de 8 octets : NaN/Inf/denormaux compris).
  double raw() {
    std::uint64_t v = 0;
    for (int k = 0; k < 8; ++k) {
      v = (v << 8) | byte();
    }
    double d = 0.0;
    std::memcpy(&d, &v, sizeof d);
    return d;
  }
};

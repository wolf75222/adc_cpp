// Utilitaire de comparaison BIT-A-BIT de deux dumps binaires (double[]) ecrits par les harness GPU.
// Sert a obtener le dmax = max |a - b| sur CHAQUE cellule entre un run device (exec=Cuda) et son
// oracle hote (exec=Serial), au lieu de se contenter d'une reduction scalaire. dmax = 0 => les deux
// runs produisent EXACTEMENT le meme champ. Compilation hote pure (g++), aucune dependance Kokkos.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static std::vector<double> load(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f) / static_cast<long>(sizeof(double));
  std::fseek(f, 0, SEEK_SET);
  std::vector<double> v(static_cast<std::size_t>(n));
  if (std::fread(v.data(), sizeof(double), v.size(), f) != v.size()) {
    std::fprintf(stderr, "short read %s\n", path);
    std::exit(2);
  }
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: diff_bin a.bin b.bin\n");
    return 2;
  }
  const std::vector<double> a = load(argv[1]), b = load(argv[2]);
  if (a.size() != b.size()) {
    std::printf("SIZE MISMATCH %zu vs %zu (%s vs %s)\n", a.size(), b.size(), argv[1], argv[2]);
    return 1;
  }
  double dmax = 0;
  std::size_t imax = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (d > dmax) {
      dmax = d;
      imax = i;
    }
  }
  std::printf("DIFF %s vs %s : n=%zu  dmax=%.3e  @i=%zu (a=%.17g b=%.17g)\n", argv[1], argv[2],
              a.size(), dmax, imax, a.empty() ? 0 : a[imax], b.empty() ? 0 : b[imax]);
  return dmax == 0.0 ? 0 : 1;
}

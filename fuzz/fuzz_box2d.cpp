/// @file
/// @brief Harnais libFuzzer : invariants de l'arithmetique d'index Box2D.
///
/// Entrees BORNEES (coordonnees +/-2^20, ratios [1,8], grow [0,16]) : on fuzze la LOGIQUE
/// (intersection, coarsen/refine sur coordonnees negatives, contains), pas l'overflow `int`
/// trivial par construction -- les coordonnees d'un vrai maillage tiennent largement dans ces
/// bornes, et refine(8) y reste loin de INT_MAX.

#include <pops/mesh/index/box2d.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "fuzz_util.hpp"

namespace {

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "INVARIANT VIOLE : %s\n", what);
    std::abort();  // abort -> libFuzzer enregistre l'entree fautive (artefact crash-*)
  }
}

pops::Box2D make_box(ByteReader& br) {
  const int B = 1 << 20;
  const int lx = br.range(-B, B), ly = br.range(-B, B);
  // Largeurs dans [-2, 64] : ~1 boite sur 20 est VIDE (hi < lo), exactement la classe de bord
  // (boites degenerees) qui mord dans le code geometrique.
  const int w = br.range(-2, 64), h = br.range(-2, 64);
  return pops::Box2D{{lx, ly}, {lx + w, ly + h}};
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  ByteReader br{data, size};
  const pops::Box2D a = make_box(br);
  const pops::Box2D b = make_box(br);
  const int r = br.range(1, 8);
  const int g = br.range(0, 16);
  const int d = br.range(0, 1);
  const int s = br.range(-1000, 1000);

  // intersect : symetrique ; non vide -> contenue dans chaque operande.
  const pops::Box2D i1 = a.intersect(b), i2 = b.intersect(a);
  expect(i1 == i2, "a.intersect(b) == b.intersect(a)");
  if (!i1.empty()) {
    expect(a.contains(i1) && b.contains(i1), "intersection contenue dans chaque operande");
  }

  if (!a.empty()) {
    // refine puis coarsen : aller-retour EXACT ; coarsen puis refine : englobant (division
    // entiere au sol -- c'est precisement le helper de division floor qu'on exerce en negatif).
    expect(a.refine(r).coarsen(r) == a, "refine(r) puis coarsen(r) = identite");
    expect(a.coarsen(r).refine(r).contains(a), "coarsen(r) puis refine(r) englobe la box");

    // grow / shift : inverses exacts.
    expect(a.grow(g).grow(-g) == a, "grow(g) puis grow(-g) = identite");
    expect(a.shift(d, s).shift(d, -s) == a, "shift(d,s) puis shift(d,-s) = identite");

    // contains <-> intersect : a contient b (non vide) ssi l'intersection EST b.
    if (!b.empty()) {
      expect(a.contains(b) == (a.intersect(b) == b), "contains coherent avec intersect");
    }
  }
  return 0;
}

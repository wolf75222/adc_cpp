/// @file
/// @brief Harnais libFuzzer : invariants du clustering Berger-Rigoutsos.
///
/// Construit une TagBox (<= 48x48, origine decalable y compris negative) dont les marqueurs
/// viennent des bits de l'entree, clusterise, puis verifie le CONTRAT de berger_rigoutsos :
/// toute cellule taguee est couverte par au moins une box ; toute box rendue est non vide,
/// contenue dans le domaine et de dimensions <= max_box_size. La recursion (trous, inflexions,
/// chop final) est exactement la classe de code geometrique ou les bords degeneres mordent
/// (boites 1x1, lignes vides, domaines plats 1xN).

#include <adc/amr/tagging/cluster.hpp>
#include <adc/amr/tagging/tag_box.hpp>
#include <adc/mesh/index/box2d.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fuzz_util.hpp"

namespace {

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "INVARIANT VIOLE : %s\n", what);
    std::abort();
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  ByteReader br{data, size};

  const int w = br.range(1, 48);
  const int h = br.range(1, 48);
  const int ox = br.range(-64, 64);
  const int oy = br.range(-64, 64);
  const adc::Box2D dom{{ox, oy}, {ox + w - 1, oy + h - 1}};

  adc::TagBox tags(dom);
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j) {
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      if (br.bit()) {
        tags(i, j) = 1;
      }
    }
  }

  adc::ClusterParams p;
  p.min_efficiency = 0.05 + 0.9 * (double(br.range(0, 100)) / 100.0);
  p.min_box_size = br.range(1, 4);
  p.max_box_size = br.range(p.min_box_size, 32);

  const std::vector<adc::Box2D> boxes = adc::berger_rigoutsos(tags, p);

  // Contrat cote boxes : non vides, dans le domaine, chopees a max_box_size.
  for (const adc::Box2D& b : boxes) {
    expect(!b.empty(), "box rendue non vide");
    expect(dom.contains(b), "box rendue contenue dans le domaine");
    expect(b.hi[0] - b.lo[0] + 1 <= p.max_box_size && b.hi[1] - b.lo[1] + 1 <= p.max_box_size,
           "box rendue de dimensions <= max_box_size");
  }

  // Contrat cote couverture : aucune cellule taguee orpheline.
  for (int j = dom.lo[1]; j <= dom.hi[1]; ++j) {
    for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
      if (!tags.tagged(i, j)) {
        continue;
      }
      bool covered = false;
      for (const adc::Box2D& b : boxes) {
        if (b.contains(i, j)) {
          covered = true;
          break;
        }
      }
      expect(covered, "toute cellule taguee couverte par au moins une box");
    }
  }
  return 0;
}

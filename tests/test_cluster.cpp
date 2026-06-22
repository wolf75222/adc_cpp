// Berger-Rigoutsos : bloc plein -> une box, deux blocs separes -> deux boxes,
// gros bloc -> chop par max_box_size, et couverture complete d'une forme en L.

#include <adc/amr/tagging/cluster.hpp>
#include <adc/amr/tagging/tag_box.hpp>
#include <adc/mesh/box2d.hpp>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace adc;

static void tag_block(TagBox& tb, const Box2D& b) {
  for (int j = b.lo[1]; j <= b.hi[1]; ++j)
    for (int i = b.lo[0]; i <= b.hi[0]; ++i)
      tb(i, j) = 1;
}

// toutes les cellules taguees sont-elles couvertes par au moins une box ?
static bool covers_all_tags(const TagBox& tb, const std::vector<Box2D>& boxes) {
  for (int j = tb.box.lo[1]; j <= tb.box.hi[1]; ++j)
    for (int i = tb.box.lo[0]; i <= tb.box.hi[0]; ++i)
      if (tb(i, j)) {
        bool in = false;
        for (const auto& b : boxes)
          if (b.contains(i, j)) {
            in = true;
            break;
          }
        if (!in)
          return false;
      }
  return true;
}

static bool has_box(const std::vector<Box2D>& v, const Box2D& b) {
  return std::find(v.begin(), v.end(), b) != v.end();
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // bloc plein
  {
    TagBox tb(Box2D::from_extents(10, 10));
    tag_block(tb, Box2D{{2, 3}, {5, 6}});
    auto boxes = berger_rigoutsos(tb, ClusterParams{});
    chk(boxes.size() == 1, "solid_count");
    chk(has_box(boxes, Box2D{{2, 3}, {5, 6}}), "solid_box");
  }

  // deux blocs separes
  {
    TagBox tb(Box2D::from_extents(10, 10));
    tag_block(tb, Box2D{{1, 1}, {3, 3}});
    tag_block(tb, Box2D{{6, 6}, {8, 8}});
    auto boxes = berger_rigoutsos(tb, ClusterParams{});
    chk(boxes.size() == 2, "two_count");
    chk(has_box(boxes, Box2D{{1, 1}, {3, 3}}), "two_box1");
    chk(has_box(boxes, Box2D{{6, 6}, {8, 8}}), "two_box2");
    chk(covers_all_tags(tb, boxes), "two_cover");
  }

  // gros bloc + chop max_box_size
  {
    TagBox tb(Box2D::from_extents(16, 16));
    tag_block(tb, Box2D::from_extents(16, 16));
    ClusterParams p;
    p.max_box_size = 8;
    auto boxes = berger_rigoutsos(tb, p);
    chk(boxes.size() == 4, "chop_count");
    for (const auto& b : boxes)
      chk(b.nx() <= 8 && b.ny() <= 8, "chop_size");
    chk(covers_all_tags(tb, boxes), "chop_cover");
  }

  // forme en L : couverture complete, boxes dans le domaine
  {
    Box2D dom = Box2D::from_extents(8, 8);
    TagBox tb(dom);
    tag_block(tb, Box2D{{0, 0}, {1, 7}});  // colonne gauche
    tag_block(tb, Box2D{{0, 0}, {7, 1}});  // ligne basse
    auto boxes = berger_rigoutsos(tb, ClusterParams{});
    chk(covers_all_tags(tb, boxes), "L_cover");
    for (const auto& b : boxes)
      chk(dom.contains(b), "L_in_domain");
  }

  if (fails == 0)
    std::printf("OK test_cluster\n");
  return fails == 0 ? 0 : 1;
}

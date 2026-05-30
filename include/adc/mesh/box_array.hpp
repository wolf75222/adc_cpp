#pragma once

#include <adc/mesh/box2d.hpp>

#include <utility>
#include <vector>

// BoxArray : l'ensemble des boxes qui pavent un niveau (disjointes, couvrantes).
// Equivalent du BoxArray d'AMReX. from_domain decoupe un domaine en
// tuiles d'au plus max_grid_size par direction, reparties le plus egalement
// possible (meilleur equilibrage que des chunks gloutons).

namespace adc {

class BoxArray {
 public:
  BoxArray() = default;
  explicit BoxArray(std::vector<Box2D> boxes) : boxes_(std::move(boxes)) {}

  static BoxArray from_domain(const Box2D& domain, int max_grid_size) {
    auto sx = split_range(domain.lo[0], domain.hi[0], max_grid_size);
    auto sy = split_range(domain.lo[1], domain.hi[1], max_grid_size);
    std::vector<Box2D> boxes;
    boxes.reserve(sx.size() * sy.size());
    for (auto [ylo, yhi] : sy)
      for (auto [xlo, xhi] : sx) boxes.push_back(Box2D{{xlo, ylo}, {xhi, yhi}});
    return BoxArray{std::move(boxes)};
  }

  int size() const { return static_cast<int>(boxes_.size()); }
  const Box2D& operator[](int i) const { return boxes_[i]; }
  const std::vector<Box2D>& boxes() const { return boxes_; }

  long num_cells() const {
    long n = 0;
    for (const auto& b : boxes_) n += b.num_cells();
    return n;
  }

  Box2D bounding_box() const {
    if (boxes_.empty()) return Box2D{};
    Box2D b = boxes_[0];
    for (const auto& o : boxes_) {
      b.lo[0] = std::min(b.lo[0], o.lo[0]);
      b.lo[1] = std::min(b.lo[1], o.lo[1]);
      b.hi[0] = std::max(b.hi[0], o.hi[0]);
      b.hi[1] = std::max(b.hi[1], o.hi[1]);
    }
    return b;
  }

 private:
  // Decoupe [lo, hi] en segments de longueur <= m, repartis egalement :
  // n = ceil(len/m) segments, les premiers `rem` d'un cran plus longs.
  static std::vector<std::pair<int, int>> split_range(int lo, int hi, int m) {
    std::vector<std::pair<int, int>> segs;
    int len = hi - lo + 1;
    if (len <= 0 || m <= 0) return segs;
    int n = (len + m - 1) / m;
    int base = len / n, rem = len % n;
    int cur = lo;
    for (int k = 0; k < n; ++k) {
      int l = base + (k < rem ? 1 : 0);
      segs.push_back({cur, cur + l - 1});
      cur += l;
    }
    return segs;
  }

  std::vector<Box2D> boxes_{};
};

}  // namespace adc

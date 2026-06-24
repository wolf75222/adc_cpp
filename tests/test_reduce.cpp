// Contrat des reductions device sum / norm_inf (point 4 de la revue : vraies
// reductions device via for_each_cell_reduce_*, plus de boucle hote sous fence).
//
// Le contrat est FP-conscient et le meme pour les trois backends :
//   - sum est EXACT (egalite stricte) la ou la reassociation est neutre : champ
//     constant (toutes valeurs egales) et champ symetrique a somme nulle. Sur un
//     champ a valeurs variees, sum n'est plus bit-identique a la boucle hote sous
//     Kokkos (somme par tuile) : on n'exige qu'un ecart RELATIF petit vs une
//     reference hote sequentielle (1e-10), pas l'egalite stricte.
//   - sum est IDEMPOTENT : deux appels sur le meme MultiFab inchange rendent
//     exactement le meme bit (verifie le determinisme du reducteur, cle pour
//     test_fill_boundary/sum_unchanged ; Kokkos::Sum est deterministe par tuile,
//     pas d'atomics flottants).
//   - norm_inf est EXACT partout (max et fabs sans arrondi, max associatif et
//     commutatif en IEEE754) : egalite stricte avec la reference hote.

#include <adc/mesh/index/box2d.hpp>
#include <adc/mesh/layout/box_array.hpp>
#include <adc/mesh/layout/distribution_mapping.hpp>
#include <adc/mesh/execution/for_each.hpp>
#include <adc/mesh/storage/mf_arith.hpp>
#include <adc/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace adc;

// Reference hote sequentielle, lecture directe des fabs locaux (ordre
// lexicographique fixe), sans passer par for_each_cell_reduce_*.
static double host_sum(const MultiFab& mf, int comp) {
  double s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i)
        s += f(i, j, comp);
  }
  return s;
}

static double host_norm_inf(const MultiFab& mf, int comp) {
  double m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
        const double a = f(i, j, comp);
        m = std::max(m, a < 0 ? -a : a);
      }
  }
  return m;
}

// Host references over ALL components (the contract of reduce_*_all / norm_inf_all): sum / signed
// extrema / max|.| accumulated across every component, lexicographic per component.
static double host_sum_all(const MultiFab& mf) {
  double s = 0;
  for (int c = 0; c < mf.ncomp(); ++c)
    s += host_sum(mf, c);
  return s;
}
static double host_max_all(const MultiFab& mf) {
  double m = -std::numeric_limits<double>::infinity();
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int c = 0; c < mf.ncomp(); ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          m = std::max(m, double(f(i, j, c)));
  }
  return m;
}
static double host_min_all(const MultiFab& mf) {
  double m = std::numeric_limits<double>::infinity();
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& f = mf.fab(li);
    const Box2D b = f.box();
    for (int c = 0; c < mf.ncomp(); ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          m = std::min(m, double(f(i, j, c)));
  }
  return m;
}
static double host_norm_inf_all(const MultiFab& mf) {
  double m = 0;
  for (int c = 0; c < mf.ncomp(); ++c)
    m = std::max(m, host_norm_inf(mf, c));
  return m;
}

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // Domaine 256x256 decoupe en boites 32x32 (64 fabs), une composante.
  Box2D dom = Box2D::from_extents(256, 256);
  BoxArray ba = BoxArray::from_domain(dom, 32);
  DistributionMapping dm(ba.size(), n_ranks());

  // 1. champ constant : sum exact (reassociation neutre, valeurs egales).
  {
    MultiFab mf(ba, dm, 1, 0);
    mf.set_val(2.0);
    const double expect = 2.0 * dom.num_cells();
    chk(std::fabs(sum(mf) - expect) <= 1e-15 * expect, "sum_constant_exact");
  }

  // 2. champ a valeurs variees : ecart RELATIF device vs reference hote < 1e-10
  //    (PAS d'egalite stricte ; la somme par tuile reassocie sous Kokkos).
  {
    MultiFab mf(ba, dm, 1, 0);
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 a = mf.fab(li).array();
      for_each_cell(mf.box(li), [a] ADC_HD(int i, int j) { a(i, j, 0) = i + 100.0 * j; });
    }
    device_fence();
    const double ref = host_sum(mf, 0);
    const double dev = sum(mf, 0);
    const double rel = std::fabs(dev - ref) / std::max(1.0, std::fabs(ref));
    chk(rel < 1e-10, "sum_varied_relative");
  }

  // 3. champ symetrique a somme nulle : valeurs +v / -v par parite de (i+j).
  //    Toute reassociation appariant +v et -v donne 0 ; ici on tolere quand
  //    meme un bruit relatif a la magnitude des termes.
  {
    MultiFab mf(ba, dm, 1, 0);
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 a = mf.fab(li).array();
      for_each_cell(mf.box(li), [a] ADC_HD(int i, int j) {
        const Real v = i + 100.0 * j;
        a(i, j, 0) = ((i + j) & 1) ? -v : v;
      });
    }
    device_fence();
    const double dev = sum(mf, 0);
    chk(std::fabs(dev) < 1e-6, "sum_antisymmetric_small");
  }

  // 4. norm_inf EXACT : signe alterne, max |.| connu. Egalite stricte vs hote.
  {
    MultiFab mf(ba, dm, 1, 0);
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 a = mf.fab(li).array();
      for_each_cell(mf.box(li), [a] ADC_HD(int i, int j) {
        const Real v = i + 100.0 * j;
        a(i, j, 0) = ((i + j) & 1) ? -v : v;
      });
    }
    device_fence();
    const double ref = host_norm_inf(mf, 0);
    const double dev = norm_inf(mf, 0);
    chk(dev == ref, "norm_inf_exact");
    // max |i + 100 j| sur [0..255]^2 = 255 + 100*255 = 25755.
    chk(dev == 255.0 + 100.0 * 255.0, "norm_inf_value");
  }

  // 5. idempotence : deux sum / deux norm_inf sur le meme champ inchange rendent
  //    exactement le meme bit (determinisme du reducteur).
  {
    MultiFab mf(ba, dm, 1, 0);
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 a = mf.fab(li).array();
      for_each_cell(mf.box(li), [a] ADC_HD(int i, int j) {
        a(i, j, 0) = std::sin(0.1 * i) + std::cos(0.07 * j);
      });
    }
    device_fence();
    chk(sum(mf, 0) == sum(mf, 0), "sum_idempotent");
    chk(norm_inf(mf, 0) == norm_inf(mf, 0), "norm_inf_idempotent");
  }

  // 6. FULL-component reductions (ADC-432): dot_all / reduce_sum_all / reduce_max_all / reduce_min_all /
  //    norm_inf_all reduce over EVERY component. A 3-component field with DISTINCT per-component ranges
  //    (comp 1 carries the global min, comp 2 the global max / max|.|) -- a comp-0-only reduction would
  //    miss them. Max / min / norm_inf are EXACT (strict equality vs host); sum / dot are within the
  //    per-tile Kokkos::Sum relative tolerance.
  {
    MultiFab mf(ba, dm, 3, 0);
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 a = mf.fab(li).array();
      for_each_cell(mf.box(li), [a] ADC_HD(int i, int j) {
        a(i, j, 0) = 1.0 + 0.001 * (i + j);      // small positive
        a(i, j, 1) = -1000.0 - (i + 100.0 * j);  // the global min, negative (drives norm_inf)
        a(i, j, 2) = 2000.0 + (i + 100.0 * j);   // the global max
      });
    }
    device_fence();
    const double rsum = host_sum_all(mf), rmax = host_max_all(mf), rmin = host_min_all(mf);
    const double rninf = host_norm_inf_all(mf);
    chk(reduce_max_all(mf) == rmax, "reduce_max_all_exact");
    chk(reduce_min_all(mf) == rmin, "reduce_min_all_exact");
    chk(norm_inf_all(mf) == rninf, "norm_inf_all_exact");
    const double dsum = reduce_sum_all(mf);
    chk(std::fabs(dsum - rsum) / std::max(1.0, std::fabs(rsum)) < 1e-10, "reduce_sum_all_relative");
    // dot_all(mf, mf) = Sum_{cells, c} f^2 over all comps -- the full-state ||.||^2.
    double rdotsq = 0;
    for (int c = 0; c < mf.ncomp(); ++c)
      for (int li = 0; li < mf.local_size(); ++li) {
        const Fab2D& f = mf.fab(li);
        const Box2D b = f.box();
        for (int jj = b.lo[1]; jj <= b.hi[1]; ++jj)
          for (int ii = b.lo[0]; ii <= b.hi[0]; ++ii)
            rdotsq += double(f(ii, jj, c)) * f(ii, jj, c);
      }
    const double ddot = dot_all(mf, mf);
    chk(std::fabs(ddot - rdotsq) / std::max(1.0, rdotsq) < 1e-10, "dot_all_relative");
    // The full-component reductions must DIFFER from the comp-0-only values (proving they cover c1, c2).
    chk(reduce_max_all(mf) != reduce_max(mf, 0), "reduce_max_all_covers_all_comps");
    chk(reduce_min_all(mf) != reduce_min(mf, 0), "reduce_min_all_covers_all_comps");
    chk(norm_inf_all(mf) != norm_inf(mf, 0), "norm_inf_all_covers_all_comps");
  }

  // 7. ncomp==1 BIT-IDENTITY (the no-regression guard): each _all helper loops the one component, so it
  //    is bit-identical to its per-component counterpart on a single-component field.
  {
    MultiFab mf(ba, dm, 1, 0);
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 a = mf.fab(li).array();
      for_each_cell(mf.box(li), [a] ADC_HD(int i, int j) {
        const Real v = i + 100.0 * j;
        a(i, j, 0) = ((i + j) & 1) ? -v : v;
      });
    }
    device_fence();
    chk(reduce_sum_all(mf) == reduce_sum(mf, 0), "reduce_sum_all_ncomp1_bit_identical");
    chk(reduce_max_all(mf) == reduce_max(mf, 0), "reduce_max_all_ncomp1_bit_identical");
    chk(reduce_min_all(mf) == reduce_min(mf, 0), "reduce_min_all_ncomp1_bit_identical");
    chk(norm_inf_all(mf) == norm_inf(mf, 0), "norm_inf_all_ncomp1_bit_identical");
    chk(dot_all(mf, mf) == dot(mf, mf, 0), "dot_all_ncomp1_bit_identical");
  }

  if (fails == 0)
    std::printf("OK test_reduce\n");
  return fails == 0 ? 0 : 1;
}

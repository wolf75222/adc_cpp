// Cache du schedule de halos de fill_boundary (ADC-260).
//   - reutilisation BIT-IDENTIQUE a une reconstruction a chaque appel (cache ON == cache OFF) ;
//   - le schedule est construit UNE fois puis reutilise sur K appels (engagement, compteur de builds) ;
//   - il est reconstruit (invalide) quand la periodicite, le domaine, n_grow, ou la layout changent,
//     et quand le MultiFab entier est reassigne (style regrid AMR).
// Invariant au nombre de rangs : couvre le chemin local (np=1) comme le chemin MPI (np>1).

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fill_boundary.hpp>
#include <adc/mesh/halo_schedule.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>
#include <adc/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>

using namespace adc;

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL[rank %d] %s\n", me, w);
      ++fails;
    }
  };

  const int L = 64, ng = 1, ncomp = 2;
  const Box2D dom = Box2D::from_extents(L, L);
  auto wrap = [&](int x) { return ((x % L) + L) % L; };
  auto val = [&](int i, int j, int c) {
    return double(wrap(i)) + 0.001 * double(wrap(j)) + 100.0 * c;
  };
  const BoxArray ba = BoxArray::from_domain(dom, 16);  // 16x16 boxes -> 4x4 = 16 boxes
  const DistributionMapping dm = make_sfc_distribution(ba, np);

  // remplit les cellules VALIDES avec la valeur periodiquement repliee (les ghosts d'un fill
  // periodique correct doivent ensuite valoir la meme chose).
  auto set_valid = [&](MultiFab& mf) {
    for (int li = 0; li < mf.local_size(); ++li) {
      Fab2D& F = mf.fab(li);
      const Box2D b = F.box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            F(i, j, c) = val(i, j, c);
    }
  };
  // nombre de cellules (valides + ghosts) != valeur repliee, reduit sur tous les rangs.
  auto count_wrong = [&](const MultiFab& mf) {
    long w = 0;
    for (int li = 0; li < mf.local_size(); ++li) {
      const Fab2D& F = mf.fab(li);
      const Box2D g = F.box().grow(mf.n_grow());
      for (int c = 0; c < ncomp; ++c)
        for (int j = g.lo[1]; j <= g.hi[1]; ++j)
          for (int i = g.lo[0]; i <= g.hi[0]; ++i)
            if (std::fabs(F(i, j, c) - val(i, j, c)) > 1e-12)
              ++w;
    }
    return all_reduce_sum(w);
  };

  const Periodicity per{true, true};
  const int K = 5;

  // (1)+(2) cache ON : K fills sur une layout stable. Schedule construit UNE fois, ghosts corrects.
  {
    MultiFab mf(ba, dm, ncomp, ng);
    set_valid(mf);
    reset_halo_schedule_build_count();
    for (int k = 0; k < K; ++k)
      fill_boundary(mf, dom, per);
    chk(count_wrong(mf) == 0, "cache_on_correct");
    chk(halo_schedule_build_count() == 1, "cache_built_once");
    chk(mf.halo_cache().size() == 1, "cache_one_entry");
  }

  // (1') cache ON == cache OFF (clear() force la reconstruction a chaque appel) : BIT-IDENTIQUE.
  {
    MultiFab a(ba, dm, ncomp, ng), b(ba, dm, ncomp, ng);
    set_valid(a);
    set_valid(b);
    for (int k = 0; k < K; ++k)
      fill_boundary(a, dom, per);  // cache reutilise
    reset_halo_schedule_build_count();
    for (int k = 0; k < K; ++k) {  // reconstruit a chaque appel
      b.halo_cache().clear();
      fill_boundary(b, dom, per);
    }
    chk(halo_schedule_build_count() == K, "cache_off_rebuilds_each_call");
    long diff = 0;
    for (int li = 0; li < a.local_size(); ++li) {
      const Fab2D& FA = a.fab(li);
      const Fab2D& FB = b.fab(li);
      const Box2D g = FA.box().grow(ng);
      for (int c = 0; c < ncomp; ++c)
        for (int j = g.lo[1]; j <= g.hi[1]; ++j)
          for (int i = g.lo[0]; i <= g.hi[0]; ++i)
            if (FA(i, j, c) != FB(i, j, c))
              ++diff;  // egalite EXACTE (0 ulp)
    }
    chk(all_reduce_sum(diff) == 0, "cache_on_equals_rebuild_bit_identical");
  }

  // (3) invalidation : une (Periodicite, domaine) differente sur le MEME mf construit un nouveau
  // schedule ; la meme reutilisee n'en construit pas.
  {
    MultiFab mf(ba, dm, ncomp, ng);
    set_valid(mf);
    reset_halo_schedule_build_count();
    fill_boundary(mf, dom, Periodicity{true, true});  // build #1
    fill_boundary(mf, dom, Periodicity{true, true});  // reutilise
    chk(halo_schedule_build_count() == 1, "same_per_domain_reused");
    fill_boundary(mf, dom, Periodicity{false, false});  // periodicite differente -> build #2
    chk(halo_schedule_build_count() == 2, "diff_periodicity_rebuilds");
    const Box2D dom2 = Box2D::from_extents(2 * L, 2 * L);
    fill_boundary(mf, dom2, Periodicity{false, false});  // domaine different -> build #3
    chk(halo_schedule_build_count() == 3, "diff_domain_rebuilds");
    chk(mf.halo_cache().size() == 3, "three_distinct_entries");
  }

  // (3') invalidation par n_grow : un MultiFab d'une autre largeur de ghosts a son propre schedule.
  {
    MultiFab mf2(ba, dm, ncomp, 2);  // ng = 2
    set_valid(mf2);
    reset_halo_schedule_build_count();
    fill_boundary(mf2, dom, per);
    chk(halo_schedule_build_count() == 1, "diff_ng_builds_own_schedule");
    chk(count_wrong(mf2) == 0, "diff_ng_correct");
  }

  // (3'') invalidation par reassignation de l'objet entier (style regrid AMR) : reassigner le
  // MultiFab abandonne le cache, donc le prochain fill reconstruit sur la NOUVELLE layout.
  {
    MultiFab mf(ba, dm, ncomp, ng);
    set_valid(mf);
    reset_halo_schedule_build_count();
    fill_boundary(mf, dom, per);  // build pour la layout #1
    chk(halo_schedule_build_count() == 1, "regrid_pre");
    const BoxArray ba2 = BoxArray::from_domain(dom, 32);  // autre decoupage (2x2 = 4 boxes)
    const DistributionMapping dm2 = make_sfc_distribution(ba2, np);
    mf = MultiFab(ba2, dm2, ncomp, ng);  // move-assign d'un MultiFab frais (regrid)
    set_valid(mf);
    fill_boundary(mf, dom, per);  // cache abandonne -> reconstruction pour la layout #2
    chk(halo_schedule_build_count() == 2, "regrid_invalidates_cache");
    chk(count_wrong(mf) == 0, "regrid_new_layout_correct");
  }

  // (3''') copy-sharing : a MultiFab copy (the RK-stage pattern `U1 = U`) shares the cache via its
  // shared_ptr, so filling the copy on the SAME (Periodicity, domain) reuses the schedule (no
  // rebuild) and is correct -- the jobs carry global indices resolved against the copy's identical
  // layout.
  {
    MultiFab mf(ba, dm, ncomp, ng);
    set_valid(mf);
    fill_boundary(mf, dom, per);  // populate mf's cache
    MultiFab cp = mf;             // copy: shares the shared_ptr cache (same layout)
    set_valid(cp);
    reset_halo_schedule_build_count();
    fill_boundary(cp, dom, per);  // shared cache HIT -> no rebuild
    chk(halo_schedule_build_count() == 0, "copy_shares_cache_no_rebuild");
    chk(count_wrong(cp) == 0, "copy_correct");
  }

  const long gfails = all_reduce_sum(fails);
  if (me == 0) {
    if (gfails == 0)
      std::printf("OK test_fill_boundary_cache (np=%d)\n", np);
    else
      std::printf("FAIL test_fill_boundary_cache : %ld checks (np=%d)\n", gfails, np);
  }
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}

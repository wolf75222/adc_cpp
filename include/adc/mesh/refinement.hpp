#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/box_hash.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/fill_boundary.hpp>  // detail::copy_shifted
#include <adc/mesh/multifab.hpp>

#include <utility>
#include <vector>

// Operateurs de transfert entre niveaux AMR (ratio entier r).
//
//   average_down : fin -> grossier, moyenne conservative sur les blocs r x r
//                  (une cellule grossiere = moyenne des r^2 cellules fines).
//   interpolate  : grossier -> fin, injection constante par morceaux (chaque
//                  cellule fine recoit la valeur de sa cellule grossiere).
//
// Les deux passent par un MultiFab temporaire "fin coarsen" partageant la
// DistributionMapping du fin : le calcul par bloc est alors local (pas de
// croisement de fabs), suivi d'un parallel_copy vers la cible. C'est le schema
// d'AMReX (average_down via une copie parallele), et parallel_copy resservira
// pour le regrid.

namespace adc {

inline int coarsen_index(int a, int r) {
  int q = a / r, rem = a % r;
  return (rem != 0 && ((rem < 0) != (r < 0))) ? q - 1 : q;
}

inline BoxArray coarsen(const BoxArray& ba, int r) {
  std::vector<Box2D> b;
  b.reserve(ba.size());
  for (int i = 0; i < ba.size(); ++i) b.push_back(ba[i].coarsen(r));
  return BoxArray{std::move(b)};
}

// Copie des regions valides qui se recouvrent : src -> dst (memes indices, pas
// de decalage). Redistribution generale entre deux MultiFab sur le meme domaine
// mais a decompositions (BoxArray + DistributionMapping) eventuellement
// differentes (ex. tuiles 2D <-> bandes FFT). C'est le ParallelCopy d'AMReX.
//
// Mono-rang : copies memoire directes. Multi-rang : metadonnees repliquees ->
// chaque rang enumere la meme liste de jobs (gd, gs, region) dans le meme ordre
// (hash spatial sur la src, candidats tries), classe par appartenance, et agrege
// un message par voisin (MPI_Isend/Irecv, tag 1). Tampons alignes sans handshake.
inline void parallel_copy(MultiFab& dst, const MultiFab& src) {
  const int nc = std::min(dst.ncomp(), src.ncomp());
  const BoxArray& sba = src.box_array();
  const BoxArray& dba = dst.box_array();
  const BoxHash shash(sba, suggest_bin(sba));

  // --- copies locales (dst locale ET src locale) ---
  for (int ld = 0; ld < dst.local_size(); ++ld) {
    Fab2D& D = dst.fab(ld);
    const Box2D vd = D.box();
    for (int gs : shash.query(vd)) {
      const int ls = src.local_index_of(gs);
      if (ls < 0) continue;  // src non locale -> MPI ci-dessous
      const Box2D region = vd.intersect(sba[gs]);
      if (region.empty()) continue;
      detail::copy_shifted(D, src.fab(ls), region, 0, 0, nc);
    }
  }

#ifdef ADC_HAS_MPI
  if (n_ranks() <= 1) return;
  const int me = my_rank(), np = n_ranks();
  const DistributionMapping& sdm = src.dmap();
  const DistributionMapping& ddm = dst.dmap();

  struct Job {
    int gs, gd;
    Box2D region;
  };
  std::vector<std::vector<Job>> send(np), recv(np);
  for (int gd = 0; gd < dba.size(); ++gd) {
    const int od = ddm[gd];
    const Box2D vd = dba[gd];
    for (int gs : shash.query(vd)) {
      const int os = sdm[gs];
      if (od != me && os != me) continue;  // ne nous concerne pas
      if (od == me && os == me) continue;  // local, deja fait
      const Box2D region = vd.intersect(sba[gs]);
      if (region.empty()) continue;
      if (os == me)
        send[od].push_back({gs, gd, region});  // je possede la src
      else
        recv[os].push_back({gs, gd, region});  // je possede la dst
    }
  }

  auto bufsz = [&](const std::vector<Job>& js) {
    long n = 0;
    for (const auto& j : js) n += j.region.num_cells() * nc;
    return n;
  };
  std::vector<std::vector<Real>> sbuf(np), rbuf(np);
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < np; ++r) {
    if (!send[r].empty()) {
      sbuf[r].resize(bufsz(send[r]));
      long k = 0;
      for (const auto& j : send[r]) {
        const ConstArray4 s = src.fab(src.local_index_of(j.gs)).const_array();
        for (int c = 0; c < nc; ++c)
          for (int jj = j.region.lo[1]; jj <= j.region.hi[1]; ++jj)
            for (int ii = j.region.lo[0]; ii <= j.region.hi[0]; ++ii)
              sbuf[r][k++] = s(ii, jj, c);
      }
      reqs.emplace_back();
      MPI_Isend(sbuf[r].data(), static_cast<int>(sbuf[r].size()), MPI_DOUBLE, r,
                1, MPI_COMM_WORLD, &reqs.back());
    }
    if (!recv[r].empty()) {
      rbuf[r].resize(bufsz(recv[r]));
      reqs.emplace_back();
      MPI_Irecv(rbuf[r].data(), static_cast<int>(rbuf[r].size()), MPI_DOUBLE, r,
                1, MPI_COMM_WORLD, &reqs.back());
    }
  }
  if (!reqs.empty())
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for (int r = 0; r < np; ++r) {
    long k = 0;
    for (const auto& j : recv[r]) {
      Array4 d = dst.fab(dst.local_index_of(j.gd)).array();
      for (int c = 0; c < nc; ++c)
        for (int jj = j.region.lo[1]; jj <= j.region.hi[1]; ++jj)
          for (int ii = j.region.lo[0]; ii <= j.region.hi[0]; ++ii)
            d(ii, jj, c) = rbuf[r][k++];
    }
  }
#endif
}

inline void average_down(const MultiFab& fine, MultiFab& coarse, int r) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  const Real inv = Real(1) / (r * r);
  for (int li = 0; li < fine.local_size(); ++li) {
    const Fab2D& F = fine.fab(li);
    Fab2D& C = cfine.fab(li);
    const Box2D cb = C.box();
    for (int c = 0; c < nc; ++c)
      for (int J = cb.lo[1]; J <= cb.hi[1]; ++J)
        for (int I = cb.lo[0]; I <= cb.hi[0]; ++I) {
          Real s = 0;
          for (int b = 0; b < r; ++b)
            for (int a = 0; a < r; ++a) s += F(r * I + a, r * J + b, c);
          C(I, J, c) = s * inv;
        }
  }
  parallel_copy(coarse, cfine);
}

inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  parallel_copy(cfine, coarse);  // amene les valeurs grossieres sur la grille fine-coarsen
  for (int li = 0; li < fine.local_size(); ++li) {
    Fab2D& F = fine.fab(li);
    const Fab2D& C = cfine.fab(li);
    const Box2D fb = F.box();
    for (int c = 0; c < nc; ++c)
      for (int j = fb.lo[1]; j <= fb.hi[1]; ++j)
        for (int i = fb.lo[0]; i <= fb.hi[0]; ++i)
          F(i, j, c) = C(coarsen_index(i, r), coarsen_index(j, r), c);
  }
}

}  // namespace adc

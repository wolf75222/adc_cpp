#pragma once

#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_hash.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/parallel/comm.hpp>

#include <utility>
#include <vector>

// fill_boundary : echange de halos intra-niveau. Remplit les ghosts de chaque
// Fab depuis les regions valides des boxes voisines de la meme MultiFab, avec
// wrapping periodique optionnel.
//
// Mono-rang : copies memoire directes (paires dst-locale / src-locale).
//
// Multi-rang (ADC_HAS_MPI + n_ranks()>1) : les metadonnees etant repliquees
// (chaque rang connait tout le BoxArray + DistributionMapping), chaque rang
// enumere DETERMINISTIQUEMENT la meme liste de jobs (src, dst, shift, region),
// classe chacun par appartenance, et agrege un message par rang voisin
// (MPI_Isend/Irecv, tag 0). Comme les deux rangs d'une paire enumerent dans le
// meme ordre, la disposition du tampon coincide sans negocier les tailles. Le
// pack lit la cellule source decalee S(i-sx, j-sy) ; l'unpack ecrit D(i, j).
//
// Les ghosts qui tombent hors du domaine sans wrapping periodique ne sont pas
// touches ici : ce sont les conditions aux limites physiques (etape suivante).

namespace adc {

struct Periodicity {
  bool x = false;
  bool y = false;
};

namespace detail {

// dst(i, j, c) = src(i - sx, j - sy, c) pour (i, j) dans region.
inline void copy_shifted(Fab2D& dst, const Fab2D& src, const Box2D& region,
                         int sx, int sy, int ncomp) {
  Array4 d = dst.array();
  ConstArray4 s = src.const_array();
  for (int c = 0; c < ncomp; ++c)
    for_each_cell(region,
                  [=](int i, int j) { d(i, j, c) = s(i - sx, j - sy, c); });
}

}  // namespace detail

inline void fill_boundary(MultiFab& mf, const Box2D& domain,
                          Periodicity per = {}) {
  const int ng = mf.n_grow();
  if (ng == 0) return;
  const int nc = mf.ncomp();
  const int Lx = domain.nx();
  const int Ly = domain.ny();
  const BoxArray& ba = mf.box_array();

  std::vector<int> sxv = {0};
  if (per.x) {
    sxv.push_back(Lx);
    sxv.push_back(-Lx);
  }
  std::vector<int> syv = {0};
  if (per.y) {
    syv.push_back(Ly);
    syv.push_back(-Ly);
  }
  std::vector<std::pair<int, int>> shifts;
  for (int sx : sxv)
    for (int sy : syv) shifts.push_back({sx, sy});

  // hash spatial : restreint la recherche des boxes voisines (sect. 3.3). Une
  // src vB (eventuellement decalee de shift) intersecte la box requete gbox ssi
  // vB intersecte gbox decalee de -shift -> on interroge le hash avec ce Q.
  const BoxHash hash(ba, suggest_bin(ba));

  // --- copies locales (dst locale ET src locale) ---
  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const int gF = mf.global_index(li);
    const Box2D gbox = F.box().grow(ng);

    for (auto [sx, sy] : shifts) {
      const Box2D Q = gbox.shift(0, -sx).shift(1, -sy);
      for (int gB : hash.query(Q)) {
        if (gB == gF && sx == 0 && sy == 0) continue;  // soi-meme, sans decalage
        const int srcLocal = mf.local_index_of(gB);
        if (srcLocal < 0) continue;  // src non locale -> MPI ci-dessous
        const Box2D region = gbox.intersect(ba[gB].shift(0, sx).shift(1, sy));
        if (region.empty()) continue;
        detail::copy_shifted(F, mf.fab(srcLocal), region, sx, sy, nc);
      }
    }
  }

#ifdef ADC_HAS_MPI
  if (n_ranks() <= 1) return;
  const int me = my_rank(), np = n_ranks();
  const DistributionMapping& dm = mf.dmap();

  struct Job {
    int src, dst, sx, sy;
    Box2D region;
  };
  std::vector<std::vector<Job>> send(np), recv(np);  // par rang voisin

  // enumeration globale deterministe : (dst gF) x shifts x candidats hash (gB
  // tries). Identique sur tous les rangs (ba/dm repliques), donc les listes
  // send/recv d'une paire coincident en ordre -> tampons alignes.
  for (int gF = 0; gF < ba.size(); ++gF) {
    const int od = dm[gF];
    const Box2D gbox = ba[gF].grow(ng);
    for (auto [sx, sy] : shifts) {
      const Box2D Q = gbox.shift(0, -sx).shift(1, -sy);
      for (int gB : hash.query(Q)) {
        if (gB == gF && sx == 0 && sy == 0) continue;
        const int os = dm[gB];
        if (od != me && os != me) continue;  // ne nous concerne pas
        if (od == me && os == me) continue;  // deja traite en local
        const Box2D region = gbox.intersect(ba[gB].shift(0, sx).shift(1, sy));
        if (region.empty()) continue;
        if (os == me)
          send[od].push_back({gB, gF, sx, sy, region});  // je possede la src
        else
          recv[os].push_back({gB, gF, sx, sy, region});  // je possede la dst
      }
    }
  }

  auto buf_size = [&](const std::vector<Job>& js) {
    long n = 0;
    for (const auto& j : js) n += j.region.num_cells() * nc;
    return n;
  };

  std::vector<std::vector<Real>> sbuf(np), rbuf(np);
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < np; ++r) {
    if (!send[r].empty()) {
      sbuf[r].resize(buf_size(send[r]));
      long k = 0;
      for (const auto& j : send[r]) {
        const ConstArray4 s = mf.fab(mf.local_index_of(j.src)).const_array();
        for (int c = 0; c < nc; ++c)
          for (int jj = j.region.lo[1]; jj <= j.region.hi[1]; ++jj)
            for (int ii = j.region.lo[0]; ii <= j.region.hi[0]; ++ii)
              sbuf[r][k++] = s(ii - j.sx, jj - j.sy, c);
      }
      reqs.emplace_back();
      MPI_Isend(sbuf[r].data(), static_cast<int>(sbuf[r].size()), MPI_DOUBLE, r,
                0, MPI_COMM_WORLD, &reqs.back());
    }
    if (!recv[r].empty()) {
      rbuf[r].resize(buf_size(recv[r]));
      reqs.emplace_back();
      MPI_Irecv(rbuf[r].data(), static_cast<int>(rbuf[r].size()), MPI_DOUBLE, r,
                0, MPI_COMM_WORLD, &reqs.back());
    }
  }
  if (!reqs.empty())
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(),
                MPI_STATUSES_IGNORE);

  for (int r = 0; r < np; ++r) {
    long k = 0;
    for (const auto& j : recv[r]) {
      Array4 d = mf.fab(mf.local_index_of(j.dst)).array();
      for (int c = 0; c < nc; ++c)
        for (int jj = j.region.lo[1]; jj <= j.region.hi[1]; ++jj)
          for (int ii = j.region.lo[0]; ii <= j.region.hi[0]; ++ii)
            d(ii, jj, c) = rbuf[r][k++];
    }
  }
#endif
}

}  // namespace adc

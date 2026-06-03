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
    for_each_cell(region, [=] ADC_HD(int i, int j) {
      d(i, j, c) = s(i - sx, j - sy, c);
    });
}

}  // namespace detail

// Handle d'echange de halos en deux phases (recouvrement calcul/comm, sect. 4.3).
// fill_boundary_begin fait les copies locales et poste les Isend/Irecv ; entre
// begin et end on peut avancer l'INTERIEUR (qui ne depend pas des ghosts
// distants) pendant que les halos transitent ; fill_boundary_end attend la fin
// des transferts et deballe le bord. Les tampons vivent dans le handle (les
// Isend/Irecv les referencent jusqu'a end ; le move du handle preserve les
// adresses des buffers heap).
struct HaloExchange {
#ifdef ADC_HAS_MPI
  struct Job {
    int src, dst, sx, sy;
    Box2D region;
  };
  std::vector<std::vector<Job>> recv;         // jobs de reception (deballage)
  // Tampons en MEMOIRE UNIFIEE (fab_allocator = SharedSpace sous Kokkos, std::allocator sinon) :
  // pack/unpack en for_each (device sous Kokkos), et on passe le pointeur unifie DIRECTEMENT a MPI ;
  // une MPI CUDA-aware detecte l'UVM et fait du device-to-device (GPUDirect), sans rebond hote.
  std::vector<std::vector<Real, fab_allocator<Real>>> sbuf, rbuf;  // vivants jusqu'a end
  std::vector<MPI_Request> reqs;
  int nc = 0;
#endif
};

// Phase 1 : copies locales + postage des echanges distants (non-bloquant).
inline HaloExchange fill_boundary_begin(MultiFab& mf, const Box2D& domain,
                                        Periodicity per = {}) {
  HaloExchange h;
  const int ng = mf.n_grow();
  if (ng == 0) return h;
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

  // hash spatial : restreint la recherche des boxes voisines (sect. 3.3).
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
  if (n_ranks() <= 1) return h;
  const int me = my_rank(), np = n_ranks();
  const DistributionMapping& dm = mf.dmap();
  h.nc = nc;
  using Job = HaloExchange::Job;
  std::vector<std::vector<Job>> send(np);
  h.recv.assign(np, {});

  // enumeration globale deterministe : (dst gF) x shifts x candidats hash (gB
  // tries). Identique sur tous les rangs -> listes send/recv alignees.
  for (int gF = 0; gF < ba.size(); ++gF) {
    const int od = dm[gF];
    const Box2D gbox = ba[gF].grow(ng);
    for (auto [sx, sy] : shifts) {
      const Box2D Q = gbox.shift(0, -sx).shift(1, -sy);
      for (int gB : hash.query(Q)) {
        if (gB == gF && sx == 0 && sy == 0) continue;
        const int os = dm[gB];
        if (od != me && os != me) continue;
        if (od == me && os == me) continue;
        const Box2D region = gbox.intersect(ba[gB].shift(0, sx).shift(1, sy));
        if (region.empty()) continue;
        if (os == me)
          send[od].push_back({gB, gF, sx, sy, region});
        else
          h.recv[os].push_back({gB, gF, sx, sy, region});
      }
    }
  }

  auto buf_size = [&](const std::vector<Job>& js) {
    long n = 0;
    for (const auto& j : js) n += j.region.num_cells() * nc;
    return n;
  };
  h.sbuf.assign(np, {});
  h.rbuf.assign(np, {});
  // PACK device (for_each, parallele sous Kokkos) dans les tampons unifies. Disposition par job :
  // c-majeur puis (jj, ii), IDENTIQUE a l'ancien ordre k++ -> tampon bit-identique au chemin hote
  // (les ctests MPI CPU restent bit-identiques np=1/2/4). Le rang pair enumere dans le meme ordre,
  // donc sbuf[A->B] et rbuf[B<-A] s'alignent sans negocier les tailles.
  for (int r = 0; r < np; ++r) {
    if (send[r].empty()) continue;
    h.sbuf[r].resize(buf_size(send[r]));
    Real* sb = h.sbuf[r].data();
    long base = 0;
    for (const auto& jb : send[r]) {
      const ConstArray4 s = mf.fab(mf.local_index_of(jb.src)).const_array();
      const int lo0 = jb.region.lo[0], lo1 = jb.region.lo[1], rnx = jb.region.nx();
      const long rsz = static_cast<long>(rnx) * jb.region.ny();
      const int sx = jb.sx, sy = jb.sy, ncl = nc;
      const long b0 = base;
      for_each_cell(jb.region, [=] ADC_HD(int i, int jc) {
        const long off = static_cast<long>(jc - lo1) * rnx + (i - lo0);
        for (int c = 0; c < ncl; ++c) sb[b0 + static_cast<long>(c) * rsz + off] = s(i - sx, jc - sy, c);
      });
      base += rsz * nc;
    }
  }
  for (int r = 0; r < np; ++r)  // allouer les tampons de reception
    if (!h.recv[r].empty()) h.rbuf[r].resize(buf_size(h.recv[r]));
  device_fence();  // les kernels de pack (et les copies locales) doivent finir avant que MPI ne lise sbuf
  for (int r = 0; r < np; ++r) {  // postage non-bloquant ; MPI recoit des pointeurs UNIFIES (GPUDirect si CUDA-aware)
    if (!h.sbuf[r].empty()) {
      h.reqs.emplace_back();
      MPI_Isend(h.sbuf[r].data(), static_cast<int>(h.sbuf[r].size()), MPI_DOUBLE, r, 0,
                MPI_COMM_WORLD, &h.reqs.back());
    }
    if (!h.rbuf[r].empty()) {
      h.reqs.emplace_back();
      MPI_Irecv(h.rbuf[r].data(), static_cast<int>(h.rbuf[r].size()), MPI_DOUBLE, r, 0,
                MPI_COMM_WORLD, &h.reqs.back());
    }
  }
#endif
  return h;
}

// Phase 2 : attend la fin des transferts et deballe (no-op en serie).
inline void fill_boundary_end(MultiFab& mf, HaloExchange& h) {
#ifdef ADC_HAS_MPI
  if (h.reqs.empty()) return;
  MPI_Waitall(static_cast<int>(h.reqs.size()), h.reqs.data(), MPI_STATUSES_IGNORE);
  // UNPACK device (for_each) depuis les tampons unifies recus. Waitall garantit le transfert
  // termine ; le kernel lance ensuite lit la memoire unifiee (coherente). Pas de rebond hote.
  for (std::size_t r = 0; r < h.recv.size(); ++r) {
    if (h.rbuf[r].empty()) continue;
    const Real* rb = h.rbuf[r].data();
    long base = 0;
    for (const auto& jb : h.recv[r]) {
      Array4 d = mf.fab(mf.local_index_of(jb.dst)).array();
      const int lo0 = jb.region.lo[0], lo1 = jb.region.lo[1], rnx = jb.region.nx();
      const long rsz = static_cast<long>(rnx) * jb.region.ny();
      const int ncl = h.nc;
      const long b0 = base;
      for_each_cell(jb.region, [=] ADC_HD(int i, int jc) {
        const long off = static_cast<long>(jc - lo1) * rnx + (i - lo0);
        for (int c = 0; c < ncl; ++c) d(i, jc, c) = rb[b0 + static_cast<long>(c) * rsz + off];
      });
      base += rsz * ncl;
    }
  }
#else
  (void)mf;
  (void)h;
#endif
}

// Version bloquante : begin puis end (comportement historique, inchange).
inline void fill_boundary(MultiFab& mf, const Box2D& domain,
                          Periodicity per = {}) {
  HaloExchange h = fill_boundary_begin(mf, domain, per);
  fill_boundary_end(mf, h);
}

}  // namespace adc

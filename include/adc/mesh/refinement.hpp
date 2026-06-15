/// @file
/// @brief Operateurs de transfert entre niveaux AMR (ratio entier r) + parallel_copy.
///
/// average_down (fin -> grossier) : moyenne CONSERVATIVE sur les blocs r x r (une cellule grossiere
/// = moyenne des r^2 cellules fines). interpolate (grossier -> fin) : injection CONSTANTE par
/// morceaux (chaque cellule fine recoit la valeur de sa cellule grossiere). Les deux passent par un
/// MultiFab temporaire "fin coarsen" partageant la DistributionMapping du fin (calcul par bloc LOCAL),
/// suivi d'un parallel_copy vers la cible (schema d'AMReX). parallel_copy : redistribution generale
/// entre deux MultiFab sur le MEME domaine a decompositions eventuellement differentes (le ParallelCopy
/// d'AMReX) ; mono-rang = copies directes, multi-rang = jobs enumeres deterministiquement (tag 1).

#pragma once
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

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

/// Indice de la cellule grossiere contenant la cellule fine a (division PLANCHER par r, gere a < 0).
/// ADC_HD : appele dans les kernels d'interpolation / injection coarse->fine. Mince adaptateur vers
/// floor_div (box2d.hpp) : meme division plancher, resultat bit-identique.
ADC_HD inline int coarsen_index(int a, int r) { return floor_div(a, r); }

/// Grossit chaque box du BoxArray d'un ratio r (coarsen box a box, ordre preserve).
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
/// Copie les regions valides qui SE RECOUVRENT de src vers dst (memes indices, sans decalage).
/// Redistribution generale entre deux MultiFab sur le meme domaine a decompositions differentes.
/// Copie min(ncomp) composantes. Une cellule de dst non couverte par src est laissee intacte.
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
    std::int64_t n = 0;
    for (const auto& j : js) n += j.region.num_cells() * nc;
    return n;
  };
  device_fence();  // GPU : les copies locales (device) precedent le pack HOTE
  std::vector<std::vector<Real>> sbuf(np), rbuf(np);
  std::vector<MPI_Request> reqs;
  for (int r = 0; r < np; ++r) {
    if (!send[r].empty()) {
      sbuf[r].resize(bufsz(send[r]));
      std::int64_t k = 0;
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

  device_fence();  // GPU : barriere avant l'ecriture HOTE des cellules recues
  for (int r = 0; r < np; ++r) {
    std::int64_t k = 0;
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

namespace detail {

// FONCTEURS NOMMES (et non lambdas ADC_HD) pour les kernels de transfert AMR. Memes raisons que le
// reste du chemin maillage/elliptique (cf. fill_boundary.hpp) : refinement est premiere-instancie depuis
// le V-cycle MG tire d'une TU externe ; une lambda etendue y fait buter l'emission du kernel device sous
// nvcc. Corps strictement identique aux anciennes lambdas -> bit-identique CPU et device.

/// Moyenne CONSERVATIVE d'un bloc r x r : C(I, J, c) = (somme des r^2 cellules fines) * inv.
struct AverageDownKernel {
  ConstArray4 F;
  Array4 C;
  Real inv;
  int r, c;
  ADC_HD void operator()(int I, int J) const {
    Real s = 0;
    for (int b = 0; b < r; ++b)
      for (int a = 0; a < r; ++a) s += F(r * I + a, r * J + b, c);
    C(I, J, c) = s * inv;
  }
};

}  // namespace detail

/// Moyenne CONSERVATIVE fin -> grossier (ratio r) : coarse(I, J) = moyenne des r^2 cellules fines du
/// bloc. Ecrit les cellules de coarse couvertes par fine (via parallel_copy depuis une grille
/// fine-coarsen locale) ; copie min(ncomp). Conserve l'integrale (somme * dV) du fin sur la zone couverte.
// Variante a TAMPON FOURNI : @p cfine est la grille "fin coarsen" (layout coarsen(fine.box_array(), r),
// dmap = fine.dmap(), >= min(ncomp) composantes, 0 ghost) ALLOUEE par l'appelant et reutilisee a chaque
// appel (chemin chaud du V-cycle MG : evite une allocation MultiFab par restriction). Calcul STRICTEMENT
// identique a la variante allouante ci-dessous.
inline void average_down(const MultiFab& fine, MultiFab& coarse, int r, MultiFab& cfine) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  assert(cfine.box_array().size() == fine.box_array().size() && cfine.ncomp() >= nc &&
         "average_down(scratch) : cfine doit etre coarsen(fine, r) sur la dmap du fin");
  const Real inv = Real(1) / (r * r);
  for (int li = 0; li < fine.local_size(); ++li) {
    const ConstArray4 F = fine.fab(li).const_array();
    Array4 C = cfine.fab(li).array();
    const Box2D cb = cfine.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(cb, detail::AverageDownKernel{F, C, inv, r, c});
  }
  parallel_copy(coarse, cfine);
}
inline void average_down(const MultiFab& fine, MultiFab& coarse, int r) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  average_down(fine, coarse, r, cfine);
}

namespace detail {

/// Injection CONSTANTE par morceaux : F(i, j, c) recoit la valeur de sa cellule grossiere couvrante.
struct InterpolateKernel {
  Array4 F;
  ConstArray4 C;
  int r, c;
  ADC_HD void operator()(int i, int j) const {
    F(i, j, c) = C(coarsen_index(i, r), coarsen_index(j, r), c);
  }
};

}  // namespace detail

/// Interpolation grossier -> fin (ratio r) par injection CONSTANTE par morceaux : chaque cellule fine
/// (y compris ghosts de la box) recoit la valeur de sa cellule grossiere (coarsen_index). Copie
/// min(ncomp). Amene d'abord les valeurs grossieres sur une grille fine-coarsen locale (parallel_copy).
// Variante a TAMPON FOURNI : @p cfine est la grille "fin coarsen" (meme contrat de layout que
// average_down ci-dessus) allouee par l'appelant et reutilisee (chemin chaud du V-cycle MG : evite une
// allocation par prolongation). Calcul STRICTEMENT identique a la variante allouante.
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r, MultiFab& cfine) {
  const int nc = std::min(fine.ncomp(), coarse.ncomp());
  assert(cfine.box_array().size() == fine.box_array().size() && cfine.ncomp() >= nc &&
         "interpolate(scratch) : cfine doit etre coarsen(fine, r) sur la dmap du fin");
  parallel_copy(cfine, coarse);  // amene les valeurs grossieres sur la grille fine-coarsen
  for (int li = 0; li < fine.local_size(); ++li) {
    Array4 F = fine.fab(li).array();
    const ConstArray4 C = cfine.fab(li).const_array();
    const Box2D fb = fine.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(fb, detail::InterpolateKernel{F, C, r, c});
  }
}
inline void interpolate(const MultiFab& coarse, MultiFab& fine, int r) {
  MultiFab cfine(coarsen(fine.box_array(), r), fine.dmap(), fine.ncomp(), 0);
  interpolate(coarse, fine, r, cfine);
}

}  // namespace adc

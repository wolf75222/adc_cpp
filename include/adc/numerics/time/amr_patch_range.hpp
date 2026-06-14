#pragma once
// Types coarse-fine multi-patch : PatchRange, FluxRegister, CoverageMask, SubcyclingSchedule,
// CoarseFineInterface + helpers fill/avgdown multi-box et fill periodique local.

#include <adc/numerics/time/amr_flux_helpers.hpp>
#include <adc/parallel/comm.hpp>  // all_reduce_sum_inplace (reflux multi-patch distribue)

#include <algorithm>

namespace adc {

// PatchRange (revue, point 5 : role promu en type). Empreinte GROSSIERE [I0..I1]x[J0..J1]
// d'un patch fin sous ratio 2 : I0 = lo/2, I1 = (hi-1)/2 (patch aligne, lo pair / hi impair).
// Centralise le calcul d'empreinte repete inline dans average_down, la couverture et l'init
// des registres de reflux. NB : ce n'est PAS Box2D::coarsen (floor des deux bornes) mais la
// borne haute (hi-1)/2 historique ; on conserve l'arithmetique exacte (bit-identique).
struct PatchRange {
  int I0, I1, J0, J1;
  explicit PatchRange(const Box2D& fine)
      : I0(fine.lo[0] / 2), I1((fine.hi[0] - 1) / 2),
        J0(fine.lo[1] / 2), J1((fine.hi[1] - 1) / 2) {}
  Box2D box() const { return Box2D{{I0, J0}, {I1, J1}}; }  // empreinte grossiere (cellules)
};

// ghosts fins multi-box depuis le grossier (interp espace+temps), PUIS fill_boundary
// (fin-fin) ecrasera les ghosts couverts par une box voisine. coarse mono-box.
inline void mf_fill_fine_ghosts_multi(MultiFab& Uf, const MultiFab& Uc_old,
                                      const MultiFab& Uc_new, Real frac) {
  device_fence();
  const int nc = Uf.ncomp();
  const ConstArray4 co = Uc_old.fab(0).const_array();
  const ConstArray4 cn = Uc_new.fab(0).const_array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 f = Uf.fab(li).array();
    const Box2D v = Uf.box(li), g = Uf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (!v.contains(i, j)) fill_cf_ghost_cell(f, co, cn, i, j, nc, frac);
  }
}

namespace detail {
// Foncteur NOMME device-clean (lambda etendue -> bute nvcc en cross-TU) : moyenne fin -> grossier
// (ratio 2) d'une box fine sur l'empreinte grossiere PatchRange. Corps bit-identique a l'ancienne
// lambda de mf_average_down_multi.
struct AvgDownMultiKernel {
  ConstArray4 f;
  Array4 c;
  int nc;
  ADC_HD void operator()(int I, int J) const {
    for (int k = 0; k < nc; ++k)
      c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                 f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
  }
};
}  // namespace detail

// moyenne fin -> grossier sur l'empreinte de CHAQUE box fine (multi-box).
inline void mf_average_down_multi(const MultiFab& Uf, MultiFab& Uc) {
  const int nc = Uc.ncomp();
  Array4 c = Uc.fab(0).array();
  for (int li = 0; li < Uf.local_size(); ++li) {
    const ConstArray4 f = Uf.fab(li).const_array();
    const PatchRange pr(Uf.box(li));
    for_each_cell(pr.box(), detail::AvgDownMultiKernel{f, c, nc});
  }
}

// Remplissage periodique PUREMENT LOCAL des ghosts d'un grossier mono-box (auto-repli).
// Equivalent a fill_boundary periodique pour une seule box, mais SANS le plan MPI : sert
// au grossier REPLIQUE (copie par-rang), dont le DistributionMapping par-rang violerait
// l'hypothese de metadonnees repliquees de fill_boundary. Lit des cellules valides (indices
// repliees dans [0,N)) et n'ecrit que des ghosts : pas de course lecture/ecriture.
inline void fill_periodic_local(MultiFab& mf, const Box2D& dom) {
  device_fence();
  const int nc = mf.ncomp(), NX = dom.nx(), NY = dom.ny();
  auto wrap = [](int x, int n) { return (x % n + n) % n; };
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    const Box2D g = mf.fab(li).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        if (i < 0 || i >= NX || j < 0 || j >= NY)
          for (int k = 0; k < nc; ++k) a(i, j, k) = a(wrap(i, NX), wrap(j, NY), k);
  }
}

// FluxRegister (revue, point 2 : role promu en type). Registre grossier a indexation GLOBALE
// sur une REGION (box, avec origine), pour remonter average_down (ecrasement des cellules
// couvertes, set) et reflux (addition aux cellules bordantes, add borne) a travers les rangs.
// Chaque rang remplit ses contributions LOCALES (0 ailleurs), gather() les somme par
// all_reduce_sum_inplace, puis chaque rang lit le total via at(). En serie all_reduce est
// l'identite -> bit a bit identique. Region a origine (0,0) = grille grossiere pleine ; region
// = boite englobante = chemin average_down multi-box. Memes formules d'index qu'avant.
struct FluxRegister {
  int I0, J0, NX, NY, nc;
  std::vector<Real> buf;
  FluxRegister(const Box2D& region, int ncomp)
      : I0(region.lo[0]), J0(region.lo[1]),
        NX(region.hi[0] - region.lo[0] + 1), NY(region.hi[1] - region.lo[1] + 1), nc(ncomp),
        buf(static_cast<std::size_t>(NX) * NY * ncomp, Real(0)) {}
  std::size_t idx(int I, int J, int k) const {
    return (static_cast<std::size_t>(J - J0) * NX + (I - I0)) * nc + k;
  }
  bool in(int I, int J) const { return I >= I0 && I < I0 + NX && J >= J0 && J < J0 + NY; }
  void set(int I, int J, int k, Real v) { buf[idx(I, J, k)] = v; }  // ecrasement (average_down)
  void add(int I, int J, int k, Real v) {                          // addition bordante (reflux)
    if (in(I, J)) buf[idx(I, J, k)] += v;
  }
  Real at(int I, int J, int k) const { return buf[idx(I, J, k)]; }
  void gather() { all_reduce_sum_inplace(buf.data(), static_cast<int>(buf.size())); }
};

// CoverageMask (revue, point 2 : part "couverture" de CoarseFineInterface). Masque grossier
// sur une REGION disant quelles cellules sont OMBRAGEES par un patch fin. Bati sur le
// box_array GLOBAL (connu de tous les rangs) -> MPI-safe. mark(box) marque l'empreinte
// grossiere d'un patch (intersectee a la region) ; covered(I,J) est borne (faux hors region).
// C'est ce qui empeche le double-reflux d'un joint fin-fin. Memes cellules qu'avant.
struct CoverageMask {
  int I0, J0, NX, NY;
  std::vector<char> cov;
  explicit CoverageMask(const Box2D& region)
      : I0(region.lo[0]), J0(region.lo[1]),
        NX(region.hi[0] - region.lo[0] + 1), NY(region.hi[1] - region.lo[1] + 1),
        cov(static_cast<std::size_t>(NX) * NY, 0) {}
  void mark(const Box2D& b) {  // marque les cellules de b intersectees a la region
    const int i0 = std::max(b.lo[0], I0), i1 = std::min(b.hi[0], I0 + NX - 1);
    const int j0 = std::max(b.lo[1], J0), j1 = std::min(b.hi[1], J0 + NY - 1);
    for (int J = j0; J <= j1; ++J)
      for (int I = i0; I <= i1; ++I) cov[(static_cast<std::size_t>(J - J0) * NX) + (I - I0)] = 1;
  }
  bool covered(int I, int J) const {
    if (I < I0 || I >= I0 + NX || J < J0 || J >= J0 + NY) return false;
    return cov[(static_cast<std::size_t>(J - J0) * NX) + (I - I0)] != 0;
  }
};

// SubcyclingSchedule (revue, point 5 : role promu en type). Cadence Berger-Oliger d'un
// niveau : ratio de raffinement temporel r, sous-pas dt/r, et position temporelle frac(s)
// = s/r du sous-pas s dans le pas parent. Centralise le `const int r = 2`, `dt / r` et
// `Real(s) / r` epars dans les boucles de sous-cyclage. Arithmetique strictement preservee :
// dt_sub(dt) == dt / r et frac(s) == Real(s) / r aux memes types, donc bit-identique.
struct SubcyclingSchedule {
  int r;
  explicit SubcyclingSchedule(int ratio = 2) : r(ratio) {}
  int count() const { return r; }                       // nombre de sous-pas
  Real dt_sub(Real dt) const { return dt / r; }         // pas fin = pas parent / r
  Real frac(int s) const { return Real(s) / r; }        // position temporelle du sous-pas s
};

// CoarseFineInterface (revue, point 2). L'interface grossier-fin d'un niveau : couverture
// (quelles cellules grossieres sont ombragees par un patch fin, via CoverageMask) + ROUTAGE
// bordant du reflux (quelle cellule grossiere borde quelle face de patch fin, et la
// correction conservative qu'on y verse). Centralise les deux logiques inline auparavant
// dupliquees dans amr_step_2level_multipatch et subcycle_level_mp. Construit le masque sur le
// box_array() GLOBAL des patchs fins (MPI-safe). route_reflux est un template sur le type de
// registre (Reg / RegMP, meme disposition de champs) : fonction nommee (pas de lambda
// generique), donc sure sous nvcc. Arithmetique bit-identique aux corps precedents.
struct CoarseFineInterface {
  CoverageMask cmask;
  // region = empreinte grossiere du niveau (origine (0,0), dims NX x NY) ; fine_ba = patchs
  // fins GLOBAUX (toutes les boxes, connues de tous les rangs). On marque l'empreinte
  // grossiere PatchRange de chaque patch.
  CoarseFineInterface(const Box2D& coarse_region, const BoxArray& fine_ba)
      : cmask(coarse_region) {
    for (int g = 0; g < fine_ba.size(); ++g) cmask.mark(PatchRange(fine_ba[g]).box());
  }
  bool covered(int I, int J) const { return cmask.covered(I, J); }

  // Verse la correction de reflux d'UN patch fin (registre g, coords parentes) dans le
  // registre grossier ref : sur chaque cellule grossiere BORDANTE non couverte par un autre
  // patch, (flux fin time-integre - flux grossier x dt) / dx|dy. Memes formules, meme ordre
  // (gauche/droite en x, bas/haut en y) que les corps inline d'origine.
  template <class Reg>
  void route_reflux(const Reg& g, Real dx, Real dy, Real dt, FluxRegister& ref, int nc) const {
    for (int J = g.J0; J <= g.J1; ++J)
      for (int k = 0; k < nc; ++k) {
        if (!covered(g.I0 - 1, J))
          ref.add(g.I0 - 1, J, k, -(g.fL[(J - g.J0) * nc + k] - g.cL[(J - g.J0) * nc + k] * dt) / dx);
        if (!covered(g.I1 + 1, J))
          ref.add(g.I1 + 1, J, k, +(g.fR[(J - g.J0) * nc + k] - g.cR[(J - g.J0) * nc + k] * dt) / dx);
      }
    for (int I = g.I0; I <= g.I1; ++I)
      for (int k = 0; k < nc; ++k) {
        if (!covered(I, g.J0 - 1))
          ref.add(I, g.J0 - 1, k, -(g.fB[(I - g.I0) * nc + k] - g.cB[(I - g.I0) * nc + k] * dt) / dy);
        if (!covered(I, g.J1 + 1))
          ref.add(I, g.J1 + 1, k, +(g.fT[(I - g.I0) * nc + k] - g.cT[(I - g.I0) * nc + k] * dt) / dy);
      }
  }
};

}  // namespace adc

#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>  // PolarGeometry
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>  // reconstruct<>, load_state/load_aux (REUTILISES verbatim)

#include <concepts>
#include <utility>
#include <vector>

/// @file
/// @brief Operateur spatial POLAIRE additif : R = -div_polar F + S sur une grille annulaire (r, theta).
///
/// Chantier "grille polaire diocotron", Phase 1 (TRANSPORT seul). C'est un assemble_rhs SEPARE de
/// l'historique cartesien (adc/numerics/spatial_operator.hpp) : ce dernier reste STRICTEMENT INTOUCHE,
/// donc un run sur maillage cartesien est bit-identique. Le chemin polaire est PUREMENT ADDITIF,
/// opt-in (l'appelant choisit assemble_rhs_polar avec une PolarGeometry).
///
/// CONVENTION D'AXES (cf. PolarGeometry) : direction d'indice 0 = RADIALE (r), direction d'indice 1 =
/// AZIMUTALE (theta). Domaine r in [r_min, r_max] (BC physique en r), theta in [0, 2pi) (periodique).
///
/// DIVERGENCE EN POLAIRE (forme conservative, volumes finis) :
///   div F = (1/r) d_r(r F_r) + (1/r) d_theta(F_theta)
/// Discretisation par cellule (i, j), avec r_i = r_cell(i), r_{i+/-1/2} = r_face(i+1)/r_face(i),
/// dr = PolarGeometry::dr(), dtheta = PolarGeometry::dtheta() :
///   -div = -(1/r_i) (r_{i+1/2} Fr_{i+1/2} - r_{i-1/2} Fr_{i-1/2}) / dr
///          -(1/r_i) (Ftheta_{j+1/2} - Ftheta_{j-1/2}) / dtheta
/// Fr, Ftheta sont les flux numeriques PHYSIQUES aux faces (le modele rend la composante physique
/// dans la base locale ; cf. ExBVelocityPolar). Le facteur metrique r aux faces radiales (et 1/r en
/// cellule) est porte ICI. CONSERVATION : la masse Sum_ij n_ij r_i dr dtheta voit le terme radial
/// telescoper (le poids r_{i+1/2} d'une face est partage par les deux cellules voisines) et le terme
/// azimutal telescoper exactement (periodique). Seuls les flux radiaux aux bords physiques r_min /
/// r_max comptent : une paroi (flux radial nul) conserve la masse a la machine.
///
/// FONCTEURS NOMMES (et non lambdas etendues), comme spatial_operator.hpp (#64/#97) : emission device
/// robuste si le noyau Model-template est instancie cross-TU. La RECONSTRUCTION (weno5z, minmod via
/// reconstruct<>) et le FLUX numerique (RusanovFlux...) sont REUTILISES verbatim depuis l'operateur
/// cartesien : direction-generiques (dir 0/1), ils s'appliquent tels quels aux axes (r, theta).

namespace adc {

namespace detail {

// Le terme source est OPTIONNEL en transport polaire (Phase 1) : une brique de transport pure
// (ExBVelocityPolar) n'expose pas source(), seul un modele compose (CompositeModel) le fait. On le
// detecte et on retombe sur 0 sinon -> l'operateur polaire marche aussi bien sur la brique seule que
// sur un modele compose, sans imposer source() (le chemin cartesien, lui, suppose un modele compose).
template <class M>
concept PolarHasSource = requires(const M m, const typename M::State u, const Aux a) {
  { m.source(u, a) } -> std::convertible_to<typename M::State>;
};

template <class Model>
ADC_HD inline typename Model::State polar_source(const Model& m, const typename Model::State& u,
                                                 const Aux& a) {
  if constexpr (PolarHasSource<Model>) return m.source(u, a);
  else return typename Model::State{};
}

// Noyau de FLUX DE FACE RADIALE (dir 0) : flux numerique a la face radiale i (entre i-1 et i), DEJA
// pondere par le rayon de face r_face(i). On stocke r_{i-1/2} Fr_{i-1/2} pour que la divergence soit
// une simple difference. FONCTEUR NOMME (device-clean cross-TU).
template <class Limiter, class NumericalFlux, class Model>
struct PolarFaceFluxRKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fr;        // sortie : r_face(i) * Fr a la face radiale i (ncomp composantes)
  Real r_min, dr;   // geometrie radiale (r_face(i) = r_min + i*dr)
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  ADC_HD void operator()(int i, int j) const {
    // Etats reconstruits de part et d'autre de la face radiale i (REUTILISE reconstruct<> cartesien,
    // dir == 0). L = extrapolation depuis la cellule i-1 vers sa face + ; R = depuis la cellule i
    // vers sa face -.
    const auto L = reconstruct<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim);
    const auto Rr = reconstruct<Model>(model, u, i, j, 0, -1, lim, recon_prim);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i - 1, j), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 0);
    const Real rf = r_min + i * dr;  // r_face(i) (positif sur l'anneau : r_min >= 0, i >= 0)
    for (int c = 0; c < Model::n_vars; ++c) fr(i, j, c) = rf * F[c];
  }
};

// Noyau de FLUX DE FACE AZIMUTALE (dir 1) : flux numerique a la face theta j (entre j-1 et j). PAS de
// ponderation par r (la metrique azimutale 1/r est appliquee en cellule, cf. PolarAssembleRhsKernel).
template <class Limiter, class NumericalFlux, class Model>
struct PolarFaceFluxThetaKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 ft;        // sortie : Ftheta a la face azimutale j (ncomp composantes)
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  ADC_HD void operator()(int i, int j) const {
    const auto L = reconstruct<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim);
    const auto Rr = reconstruct<Model>(model, u, i, j, 1, -1, lim, recon_prim);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i, j - 1), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 1);
    for (int c = 0; c < Model::n_vars; ++c) ft(i, j, c) = F[c];
  }
};

// Noyau d'assemblage du residu polaire en cellule (i, j) :
//   R = S - (1/r_i) (Fr_pondere_{i+1} - Fr_pondere_i) / dr - (1/r_i) (Ftheta_{j+1} - Ftheta_j) / dtheta
// Fr_pondere = r_face * Fr (deja produit par PolarFaceFluxRKernel). FONCTEUR NOMME.
template <class Model>
struct PolarAssembleRhsKernel {
  Model model;
  ConstArray4 u, ax, fr, ft;  // etat, aux, flux radial pondere par r, flux azimutal
  Array4 r;                    // sortie : residu
  Real r_min, dr, dtheta;
  ADC_HD void operator()(int i, int j) const {
    const Real ri = r_min + (i + Real(0.5)) * dr;  // r_cell(i)
    const Real inv_r = Real(1) / ri;
    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const auto S = polar_source<Model>(model, load_state<Model>(u, i, j), Ac);
    for (int c = 0; c < Model::n_vars; ++c) {
      const Real div_r = (fr(i + 1, j, c) - fr(i, j, c)) / dr;       // d_r(r Fr) discret
      const Real div_t = (ft(i, j + 1, c) - ft(i, j, c)) / dtheta;   // d_theta(Ftheta) discret
      r(i, j, c) = S[c] - inv_r * (div_r + div_t);
    }
  }
};

// Boites de FACE radiale / azimutale (cf. xface_box/yface_box cartesiennes) : faces radiales = nr+1 x
// ntheta, faces azimutales = nr x ntheta+1. Reutilise les helpers cartesiens (memes conventions
// d'indices de face) pour ne pas dupliquer.
}  // namespace detail

/// assemble_rhs_polar<Limiter, NumericalFlux> : R = -div_polar F* + S sur une PolarGeometry. Le
/// limiteur (reconstruction) et le flux numerique sont des parametres de template, comme l'operateur
/// cartesien. Calcule d'abord les flux de FACE (radial pondere par r, azimutal) dans des MultiFab
/// temporaires, puis differencie. Tous les noyaux sont device-callable (foncteurs nommes).
///
/// CONDITIONS AUX LIMITES (Phase 1, transport seul) : theta PERIODIQUE (l'appelant remplit les ghosts
/// azimutaux par fill_boundary periodique). r PHYSIQUE : l'appelant remplit les ghosts radiaux (paroi
/// / sortie). Les flux radiaux aux faces r_min (i = lo) et r_max (i = hi+1) sont calcules a partir des
/// etats de ghost ; pour une PAROI (flux radial nul, conservation stricte) l'appelant impose des
/// ghosts radiaux qui annulent la vitesse radiale de derive (ou, plus simplement, le test verifie la
/// conservation sur un champ ou v_r = 0 aux bords -- cf. test_polar_transport_mms).
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model>
void assemble_rhs_polar(const Model& model, const MultiFab& U, const MultiFab& aux,
                        const PolarGeometry& geom, MultiFab& R, bool recon_prim = false) {
  const Real r_min = geom.r_min, dr = geom.dr(), dtheta = geom.dtheta();
  const Limiter lim{};
  const NumericalFlux nflux{};
  // BoxArrays de FACE (cf. compute_face_fluxes cartesien) : faces radiales = surroundingNodes en x
  // (xface_box), faces azimutales en y (yface_box). Memes conventions d'indices de face -> fr(i, .)
  // est la face radiale entre i-1 et i, ft(., j) la face azimutale entre j-1 et j.
  std::vector<Box2D> rfaces, tfaces;
  rfaces.reserve(U.box_array().size());
  tfaces.reserve(U.box_array().size());
  for (const Box2D& b : U.box_array().boxes()) {
    rfaces.push_back(xface_box(b));
    tfaces.push_back(yface_box(b));
  }
  MultiFab Fr(BoxArray(std::move(rfaces)), U.dmap(), Model::n_vars, 0);
  MultiFab Ft(BoxArray(std::move(tfaces)), U.dmap(), Model::n_vars, 0);
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fr = Fr.fab(li).array();
    Array4 ft = Ft.fab(li).array();
    const Box2D v = R.box(li);
    // Faces radiales : i dans [lo..hi+1], j dans [lo..hi] (cf. xface_box).
    for_each_cell(xface_box(v), detail::PolarFaceFluxRKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, fr, r_min, dr, lim, nflux, recon_prim});
    // Faces azimutales : i dans [lo..hi], j dans [lo..hi+1] (cf. yface_box).
    for_each_cell(yface_box(v), detail::PolarFaceFluxThetaKernel<Limiter, NumericalFlux, Model>{
                                    model, u, ax, ft, lim, nflux, recon_prim});
  }
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 fr = Fr.fab(li).const_array();
    const ConstArray4 ft = Ft.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::PolarAssembleRhsKernel<Model>{model, u, ax, fr, ft, r, r_min, dr,
                                                           dtheta});
  }
}

}  // namespace adc

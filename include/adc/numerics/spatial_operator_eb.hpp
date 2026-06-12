#pragma once

#include <adc/core/state.hpp>
#include <adc/core/types.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/numerics/elliptic/cut_fraction.hpp>  // detail::cut_fraction, CutFraction (PR1)
#include <adc/numerics/numerical_flux.hpp>
#include <adc/numerics/reconstruction.hpp>
#include <adc/numerics/spatial_operator.hpp>  // reconstruct<>, load_state/load_aux, *_face_box (REUTILISES verbatim)
#include <adc/runtime/wall_predicate.hpp>     // detail::DiscDomain (level set source-unique du disque)

#include <utility>
#include <vector>

/// @file
/// @brief Operateur spatial CUT-CELL / EMBEDDED BOUNDARY (EB) : R = -div_eb F + S sur un disque, en
///        volumes finis CONSERVATIFS aux ouvertures de face (apertures alpha_f) et fraction de volume
///        kappa derivees de detail::cut_fraction (chantier T5-PR1).
///
/// CONTEXTE (cf. docs/HOFFART_FIDELITY.md ligne 39, verrou "bords d'anneau cartesiens"). Le chemin T2
/// (spatial_operator.hpp : assemble_rhs_masked) approche le disque par un MASQUE en ESCALIER : une face
/// active/inactive est une PORTE 0/1 (flux normal mis a zero), la frontiere est crenelee. Cet operateur
/// GENERALISE la porte 0/1 a une OUVERTURE alpha_f in [0, 1] (la fraction lineaire de face dans le
/// disque, EXACTEMENT la cut_distance/h du mur elliptique -> coherence bit-a-bit avec Poisson) ET divise
/// le residu par la FRACTION DE VOLUME kappa de la cellule (volume de cellule coupee). C'est un schema EB
/// du 2e ordre pour le transport lisse a l'interieur du disque, qui rend la croissance du mode diocotron
/// sans le sur-taux l-dependant structurel du baseline cartesien.
///
/// NOTE SUR alpha_f ET LE MODELE D'ACTIVITE. L'activite de cellule suit le critere centre-dans-le-disque
/// (ls(centre) < 0), comme GeometricMG et le masque T2. Avec ce critere, l'ouverture geometrique
/// cut_distance(lc, ln, h)/h vaut EXACTEMENT 1 sur une face entre deux cellules ACTIVES (lc < 0 et
/// ln < 0 -> branche "voisin interieur" de cut_distance -> distance = h) et FRACTIONNAIRE in (0, 1) sur
/// la jambe d'une cellule active vers un voisin INACTIF (ln >= 0 : crossing lineaire). La frontiere
/// immergee est donc portee par DEUX quantites continues issues de cut_fraction : (i) la jambe
/// fractionnaire (qui ferme la face vers l'inactif, cf. paroi no-penetration ci-dessous) et SURTOUT
/// (ii) la FRACTION DE VOLUME kappa in (0, 1] de la cellule coupee, qui corrige la divergence par le vrai
/// volume immerge. C'est kappa (et non l'ouverture des faces internes) qui de-crenelle la frontiere et
/// restaure l'ordre 2 du transport interieur, ce que le masque T2 (volume implicite = 1) ne fait pas.
///
/// FORME CONSERVATIVE EB (cellule (i, j), volume kappa dx dy) :
///   kappa dx dy d_t U = - [ alpha_xp Fx_{i+1} - alpha_xm Fx_i ] dy
///                       - [ alpha_yp Fy_{j+1} - alpha_ym Fy_i ] dx
///                       - alpha_wall |wall| F_wall                       (terme de PAROI immergee)
///                       + kappa dx dy S
/// soit, apres division par kappa dx dy (avec kappa CLAMPE, cf. stabilite petite cellule) :
///   R = S - (1/kappa) [ (alpha_xp Fx_{i+1} - alpha_xm Fx_i) / dx
///                     + (alpha_yp Fy_{j+1} - alpha_ym Fy_i) / dy ]
///         - (1/kappa) (alpha_wall |wall| / (dx dy)) F_wall
/// Le flux de PAROI immergee est un flux de NO-PENETRATION (zero-normal-flux) : F_wall = 0. C'est le
/// pendant FV du mur conducteur (l'elliptique applique Dirichlet ; le transport applique une paroi
/// solide a la MEME frontiere geometrique). Le terme de paroi est donc IDENTIQUEMENT NUL ; il est ecrit
/// explicitement (et garde a zero) pour que le contrat soit lisible et qu'un futur flux de paroi non nul
/// (injection, glissement) ait son point d'accroche unique.
///
/// CONSERVATION DE LA MASSE. Sur deux cellules actives voisines, l'ouverture de la FACE PARTAGEE est la
/// MEME des deux cotes (alpha_f geometrique, fonction du seul level set : alpha_xp(i) == alpha_xm(i+1),
/// cf. cut_fraction symetrique par face) -> le flux de cette face telescope EXACTEMENT dans la somme
/// Sum_cells kappa dx dy R. Une face dont la cellule voisine est INACTIVE est FERMEE (alpha_f force a 0)
/// ET le flux de paroi est nul -> aucune masse ne franchit la frontiere immergee. La masse totale sur
/// les cellules actives Sum n_ij kappa_ij dx dy est donc conservee A LA MACHINE. C'est l'analogue EB
/// (apertures continues) de la conservation T2 (portes 0/1).
///
/// STABILITE PETITE CELLULE (small-cell problem). Le facteur 1/kappa amplifie le residu quand kappa
/// devient petit sur la couche de cisaillement r0/r1 ; a pas dt FIXE calibre sur les cellules pleines,
/// une amplification non bornee fait exploser (Inf/NaN) le pas explicite sur une cellule fortement
/// coupee. Deux gardes EMPILEES, complementaires :
///   1. Le plancher de la primitive PR1 : cut_distance bute chaque demi-face a theta >= 1e-3 (garde
///      anti-division HERITEE du mur elliptique). kappa = produit de moyennes de demi-faces ne peut donc
///      jamais valoir EXACTEMENT 0 -> pas de division par zero stricte. MAIS le minorant induit est
///      ~ (1e-3)^2 / 4 ~ 2.5e-7 : 1/kappa peut atteindre ~4e6, ce qui suffit a deborder le pas fixe.
///   2. SCHEMA RETENU = CLAMP DE VOLUME (ce header) : kappa_eff = max(kappa, kappa_min), kappa_min par
///      defaut 1e-2 (1% du volume plein). Garde de NIVEAU SCHEMA, INDEPENDANTE du plancher elliptique :
///      elle borne l'amplification 1/kappa a 1/kappa_min = 100, valeur CALIBREE pour que le pas explicite
///      fixe reste stable quel que soit le degre de coupe. C'est le "volume merging" implicite le plus
///      simple et le plus robuste pour un pas FIXE, au prix d'une legere NON-conservation LOCALE sur les
///      cellules les plus coupees (volume effectif > volume reel).
/// La masse GLOBALE reste conservee A LA MACHINE car le clamp n'agit que sur le DENOMINATEUR (volume),
/// PAS sur les flux de face (numerateur) : la somme telescopique des flux est inchangee (la masse
/// discrete coherente avec le schema utilise le MEME kappa_eff, cf. test conservation). Alternative
/// documentee (hors PR2) : la redistribution de flux (flux redistribution d'AMReX-EB) repartit l'exces
/// de divergence des petites cellules sur les voisines pleines -> conservation LOCALE exacte mais stencil
/// non local ; le clamp suffit pour la cible (pas fixe calibre, MMS lisse, conservation globale).
///
/// FONCTEURS NOMMES (et non lambdas etendues), comme spatial_operator.hpp / _polar.hpp (#64/#97) :
/// emission device robuste si le noyau Model-template est instancie cross-TU. La RECONSTRUCTION et le
/// FLUX numerique sont REUTILISES verbatim depuis l'operateur cartesien (reconstruct<>, RusanovFlux).
/// Le level set est passe PAR VALEUR (callable ADC_HD, p.ex. detail::DiscDomain capture) : device-safe.
///
/// INVARIANT : ce header est PUREMENT ADDITIF et OPT-IN. L'operateur cartesien (assemble_rhs) et le
/// chemin masque T2 (assemble_rhs_masked) restent STRICTEMENT INTOUCHES ; un run sans disque EB est
/// bit-identique. assemble_rhs_eb n'est appele que sur opt-in explicite (geometry = cutcell).

namespace adc {

namespace detail {

/// Aperture par defaut sous laquelle une face est traitee comme FERMEE (paroi immergee). En dessous de
/// ce seuil l'ouverture lineaire est numeriquement nulle (la face ne croise quasiment pas le disque) ;
/// le clamp anti-division de cut_distance bute deja a 1e-3, ce seuil l'aligne sur la fermeture FV.
constexpr Real kEbFaceOpenEps = Real(1e-6);

/// Plancher de fraction de volume (small-cell clamp). kappa_eff = max(kappa, kEbKappaMin) borne
/// l'amplification 1/kappa a 1/kEbKappaMin = 100 -> residu fini sur une cellule arbitrairement coupee,
/// pas explicite fixe stable. Voir la note STABILITE PETITE CELLULE du @file.
constexpr Real kEbKappaMin = Real(1e-2);

/// Adaptateur device-safe : rend DiscDomain (qui expose level_set/cell_active, PAS operator()) utilisable
/// comme callable Real(Real, Real) attendu par cut_fraction et l'operateur EB. FONCTEUR NOMME (capture
/// le DiscDomain PAR VALEUR : trois doubles, device-safe), pas une lambda etendue. operator() forwarde
/// EXACTEMENT DiscDomain::level_set -> meme geometrie de coupe que le mur elliptique (coherence bit).
struct DiscLevelSet {
  DiscDomain disc;
  ADC_HD Real operator()(Real x, Real y) const { return disc.level_set(x, y); }
};

/// Construit le callable level set du disque depuis un DiscDomain (sucre : disc_level_set(d)).
ADC_HD inline DiscLevelSet disc_level_set(const DiscDomain& d) { return DiscLevelSet{d}; }

/// Indicateur d'activite (centre dans le disque, ls < 0) depuis un level set callable. ADC_HD.
template <class LevelSet>
ADC_HD inline bool eb_cell_active(const LevelSet& ls, Real xc, Real yc) {
  return ls(xc, yc) < Real(0);
}

/// Ouverture (aperture) d'UNE face entre la cellule active (xc, yc) et son voisin a (xn, yn), pas h.
///
/// Convention FV EB :
///   - voisin INACTIF (ls(xn,yn) >= 0) : la face touche le mur immerge -> FERMEE (alpha = 0,
///     no-penetration). C'est la generalisation de la porte 0/1 de T2 : l'inactif ferme la face.
///   - voisin ACTIF (ln < 0) : l'ouverture reutilise A L'IDENTIQUE la primitive partagee cut_distance
///     (donc bit-coherente avec le mur elliptique), alpha = cut_distance(lc, ln, h) / h. Mais pour un
///     voisin actif cut_distance prend la branche "voisin interieur" et rend h -> alpha = 1 EXACTEMENT,
///     loin comme pres du bord : la face partagee de deux cellules actives est toujours PLEINE. Les
///     ouvertures de face sont donc BINAIRES {0, 1} ; c'est la FRACTION DE VOLUME kappa in (0, 1], et
///     non l'ouverture des faces internes, qui porte la geometrie de coupe (cf. NOTE alpha_f du @file).
/// SYMETRIE (cle de la conservation) : cut_distance(lc, ln, h) ne depend que de (lc, ln) ; la face
/// partagee vue de la cellule i (centre lc, voisin ln) et vue de la cellule i+1 (centre ln, voisin lc)
/// donne la MEME ouverture des que les deux sont actives (ln < 0 : la branche "voisin interieur" rend
/// h des deux cotes -> alpha = 1). La frontiere active/active interne est donc traitee SYMETRIQUEMENT.
ADC_HD inline Real eb_face_aperture(Real lc, Real ln, Real h) {
  if (ln >= Real(0)) return Real(0);          // voisin inactif : face fermee (paroi, no-penetration)
  return cut_distance(lc, ln, h) / h;          // voisin actif : ouverture lineaire (== mur elliptique)
}

/// Noyau de FLUX DE FACE x (dir 0) pour le transport EB : flux numerique a la face entre (i-1, j) et
/// (i, j), PONDERE par l'ouverture alpha_x de cette face. On stocke alpha_x * Fx pour que la divergence
/// EB soit une simple difference (comme la ponderation par r du polaire). FONCTEUR NOMME (device-clean).
///
/// L'ouverture de la face i est calculee DU COTE de la cellule active : si (i, j) est active, on prend
/// l'ouverture vue de (i, j) vers son voisin (i-1) ; si (i, j) est inactive mais (i-1, j) active, on
/// prend l'ouverture vue de (i-1) vers (i). Si les DEUX sont inactives, la face est hors domaine actif
/// (alpha = 0). Cette symetrie garantit l'unicite de alpha sur la face partagee (conservation).
template <class Limiter, class NumericalFlux, class Model, class LevelSet>
struct EbFaceFluxXKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fx;            // sortie : alpha_x * Fx a la face entre i-1 et i (ncomp composantes)
  Real dx;
  Geometry geom;        // centres de cellule (level set evalue au centre, comme cut_fraction)
  LevelSet ls;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  int pos_comp = 0;          ///< composante du role Density (resolue par l'appelant hote)
  ADC_HD void operator()(int i, int j) const {
    const Real xL = geom.x_cell(i - 1), xR = geom.x_cell(i), yc = geom.y_cell(j);
    const Real lL = ls(xL, yc), lR = ls(xR, yc);
    const bool aL = lL < Real(0), aR = lR < Real(0);
    Real alpha;
    if (aL && aR) {
      alpha = eb_face_aperture(lL, lR, dx);     // face interne active/active : ouverture symetrique
    } else if (aR) {
      alpha = eb_face_aperture(lR, lL, dx);     // (i) active, (i-1) inactive : face fermee (0)
    } else if (aL) {
      alpha = eb_face_aperture(lL, lR, dx);     // (i-1) active, (i) inactive : face fermee (0)
    } else {
      alpha = Real(0);                           // les deux inactives : face hors domaine actif
    }
    if (alpha < kEbFaceOpenEps) {                // face fermee (paroi immergee) : flux normal nul
      for (int c = 0; c < Model::n_vars; ++c) fx(i, j, c) = Real(0);
      return;
    }
    const auto L = reconstruct_pp<Model>(model, u, i - 1, j, 0, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 0, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i - 1, j), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 0);
    for (int c = 0; c < Model::n_vars; ++c) fx(i, j, c) = alpha * F[c];
  }
};

/// Noyau de FLUX DE FACE y (dir 1) pour le transport EB : analogue de EbFaceFluxXKernel en j. Stocke
/// alpha_y * Fy a la face entre (i, j-1) et (i, j). FONCTEUR NOMME (device-clean).
template <class Limiter, class NumericalFlux, class Model, class LevelSet>
struct EbFaceFluxYKernel {
  Model model;
  ConstArray4 u, ax;
  Array4 fy;
  Real dy;
  Geometry geom;
  LevelSet ls;
  Limiter lim;
  NumericalFlux nflux;
  bool recon_prim;
  Real pos_floor = Real(0);  ///< limiteur de positivite Zhang-Shu (<= 0 : inactif, bit-identique)
  int pos_comp = 0;          ///< composante du role Density (resolue par l'appelant hote)
  ADC_HD void operator()(int i, int j) const {
    const Real xc = geom.x_cell(i), yL = geom.y_cell(j - 1), yR = geom.y_cell(j);
    const Real lL = ls(xc, yL), lR = ls(xc, yR);
    const bool aL = lL < Real(0), aR = lR < Real(0);
    Real alpha;
    if (aL && aR) {
      alpha = eb_face_aperture(lL, lR, dy);
    } else if (aR) {
      alpha = eb_face_aperture(lR, lL, dy);
    } else if (aL) {
      alpha = eb_face_aperture(lL, lR, dy);
    } else {
      alpha = Real(0);
    }
    if (alpha < kEbFaceOpenEps) {
      for (int c = 0; c < Model::n_vars; ++c) fy(i, j, c) = Real(0);
      return;
    }
    const auto L = reconstruct_pp<Model>(model, u, i, j - 1, 1, +1, lim, recon_prim, pos_floor, pos_comp);
    const auto Rr = reconstruct_pp<Model>(model, u, i, j, 1, -1, lim, recon_prim, pos_floor, pos_comp);
    const auto F = nflux(model, L, load_aux<aux_comps<Model>()>(ax, i, j - 1), Rr,
                         load_aux<aux_comps<Model>()>(ax, i, j), 1);
    for (int c = 0; c < Model::n_vars; ++c) fy(i, j, c) = alpha * F[c];
  }
};

/// Noyau d'assemblage du residu EB en cellule (i, j) :
///   cellule INACTIVE -> residu 0 (non avancee, comme T2) ;
///   cellule ACTIVE -> R = S - (1/kappa_eff) [ (fx_{i+1} - fx_i)/dx + (fy_{j+1} - fy_i)/dy ] - terme_paroi.
/// fx/fy contiennent DEJA alpha_f * F (produits par les noyaux de face). kappa_eff = max(kappa, min)
/// (clamp petite cellule). Le terme de PAROI immergee est un no-penetration F_wall = 0 -> nul, ecrit
/// explicitement (garde a zero) comme point d'accroche unique. FONCTEUR NOMME (device-clean).
template <class Model, class LevelSet>
struct EbAssembleRhsKernel {
  Model model;
  ConstArray4 u, ax, fx, fy;  // etat, aux, flux x pondere par alpha, flux y pondere par alpha
  Array4 r;                    // sortie : residu
  Real dx, dy;
  Geometry geom;
  LevelSet ls;
  Real kappa_min;
  ADC_HD void operator()(int i, int j) const {
    const Real xc = geom.x_cell(i), yc = geom.y_cell(j);
    if (!eb_cell_active(ls, xc, yc)) {            // hors disque : residu nul, non avancee (cf. T2)
      for (int c = 0; c < Model::n_vars; ++c) r(i, j, c) = Real(0);
      return;
    }
    // Fraction de volume kappa derivee EXACTEMENT de la meme cut_fraction que le mur elliptique : la
    // geometrie de coupe est l'unique source de verite (apertures de face ET volume). kappa in (0, 1].
    const CutFraction cf = cut_fraction(ls, xc, yc, dx, dy);
    // CLAMP PETITE CELLULE : borne 1/kappa a 1/kappa_min -> residu fini, pas fixe stable. N'agit QUE sur
    // le denominateur (volume) ; les flux (numerateur) sont inchanges -> conservation GLOBALE preservee.
    const Real kappa_eff = cf.kappa > kappa_min ? cf.kappa : kappa_min;
    const Real inv_kappa = Real(1) / kappa_eff;

    const Aux Ac = load_aux<aux_comps<Model>()>(ax, i, j);
    const auto S = model.source(load_state<Model>(u, i, j), Ac);

    // Flux de PAROI immergee (no-penetration) : F_wall = 0 -> terme nul. Point d'accroche unique pour un
    // futur flux de paroi non nul. Reste a 0 dans PR2 (paroi solide, comme le mur Dirichlet elliptique).
    constexpr Real wall_flux = Real(0);

    for (int c = 0; c < Model::n_vars; ++c) {
      const Real div_x = (fx(i + 1, j, c) - fx(i, j, c)) / dx;  // d_x(alpha Fx) discret
      const Real div_y = (fy(i, j + 1, c) - fy(i, j, c)) / dy;  // d_y(alpha Fy) discret
      // Accumulation TERME A TERME (et non inv_kappa*(div_x + div_y)) : quand kappa_eff = 1 et toutes
      // alpha = 1 (cas SANS coupe), inv_kappa = 1 et chaque inv_kappa*div_* = div_* a l'identite IEEE
      // (x*1.0 == x), donc r = S - div_x - div_y - 0 reproduit BIT A BIT l'operateur cartesien
      // (S - (Fxp-Fxm)/dx - (Fyp-Fym)/dy), terme par terme. Un regroupement (div_x + div_y) casserait
      // l'associativite flottante et la bit-identite du chemin par defaut.
      r(i, j, c) = S[c] - inv_kappa * div_x - inv_kappa * div_y - inv_kappa * wall_flux;
    }
  }
};

}  // namespace detail

/// assemble_rhs_eb<Limiter, NumericalFlux> : residu R = -div_eb F + S sur un DISQUE en cut-cell / EB,
/// avec ouvertures de face alpha_f in [0, 1] et fraction de volume kappa derivees de detail::cut_fraction
/// (T5-PR1). C'est la generalisation EB du chemin masque T2 (assemble_rhs_masked, portes 0/1) : la
/// frontiere immergee n'est plus crenelee, le schema est du 2e ordre pour le transport lisse a
/// l'interieur du disque, et la masse est conservee A LA MACHINE (aucun flux ne franchit le mur).
///
/// @tparam Limiter        reconstruction (NoSlope / Minmod / VanLeer / Weno5), comme l'operateur cartesien.
/// @tparam NumericalFlux  politique de flux (RusanovFlux par defaut).
/// @param  ls             level set callable ADC_HD (p.ex. detail::DiscDomain) : ls < 0 a l'interieur.
/// @param  kappa_min      plancher de fraction de volume (clamp petite cellule), defaut kEbKappaMin.
///
/// MISE EN OEUVRE en DEUX PASSES (structure REUTILISEE de l'operateur polaire) : passe 1 calcule les
/// flux de FACE ponderes par alpha_f dans des MultiFab temporaires ; passe 2 differencie et divise par
/// kappa_eff. AUCUN MultiFab de masque n'est requis : l'activite des cellules ET les ouvertures de face
/// derivent toutes du SEUL level set (source unique geometrique), comme cut_fraction.
///
/// CONDITIONS AUX LIMITES : l'appelant remplit les ghosts (fill_ghosts) avant l'appel, comme pour
/// assemble_rhs. Les faces touchant une cellule inactive sont FERMEES par le kernel (paroi immergee),
/// donc la BC physique de la boite carree n'influe pas le disque interieur (l'EB la masque).
///
/// INVARIANT : point d'entree SEPARE ; le chemin par defaut (assemble_rhs) reste bit-identique tant
/// qu'il n'appelle PAS cette surcharge.
template <class Limiter = NoSlope, class NumericalFlux = RusanovFlux, class Model, class LevelSet>
void assemble_rhs_eb(const Model& model, const MultiFab& U, const MultiFab& aux, const LevelSet& ls,
                     const Geometry& geom, MultiFab& R, bool recon_prim = false,
                     Real kappa_min = detail::kEbKappaMin, Real pos_floor = Real(0)) {
  const Real dx = geom.dx(), dy = geom.dy();
  const Limiter lim{};
  const NumericalFlux nflux{};
  const int pos_comp = detail::positivity_comp<Model>(pos_floor);
  // BoxArrays de FACE (cf. compute_face_fluxes / operateur polaire) : faces x = surroundingNodes en x
  // (xface_box), faces y en y (yface_box). fx(i, .) est la face entre i-1 et i, fy(., j) entre j-1 et j.
  std::vector<Box2D> xfaces, yfaces;
  xfaces.reserve(U.box_array().size());
  yfaces.reserve(U.box_array().size());
  for (const Box2D& b : U.box_array().boxes()) {
    xfaces.push_back(xface_box(b));
    yfaces.push_back(yface_box(b));
  }
  MultiFab Fx(BoxArray(std::move(xfaces)), U.dmap(), Model::n_vars, 0);
  MultiFab Fy(BoxArray(std::move(yfaces)), U.dmap(), Model::n_vars, 0);
  // PASSE 1 : flux de face ponderes par alpha_f.
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    Array4 fx = Fx.fab(li).array();
    Array4 fy = Fy.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(xface_box(v),
                  detail::EbFaceFluxXKernel<Limiter, NumericalFlux, Model, LevelSet>{
                      model, u, ax, fx, dx, geom, ls, lim, nflux, recon_prim, pos_floor, pos_comp});
    for_each_cell(yface_box(v),
                  detail::EbFaceFluxYKernel<Limiter, NumericalFlux, Model, LevelSet>{
                      model, u, ax, fy, dy, geom, ls, lim, nflux, recon_prim, pos_floor, pos_comp});
  }
  // PASSE 2 : divergence EB / kappa_eff + source ; cellule inactive -> residu 0.
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 ax = aux.fab(li).const_array();
    const ConstArray4 fx = Fx.fab(li).const_array();
    const ConstArray4 fy = Fy.fab(li).const_array();
    Array4 r = R.fab(li).array();
    const Box2D v = R.box(li);
    for_each_cell(v, detail::EbAssembleRhsKernel<Model, LevelSet>{model, u, ax, fx, fy, r, dx, dy,
                                                                  geom, ls, kappa_min});
  }
}

}  // namespace adc

#pragma once

#include <adc/integrator/amr_reflux.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/fab2d.hpp>

#include <vector>

// AMR multi-niveaux (N niveaux emboites, ratio 2) : sous-cyclage Berger-Oliger
// recursif avec reflux a chaque interface coarse-fine. Generalise
// amr_step_2level : chaque niveau qui possede un enfant joue le role du
// "grossier" de l'etape 2-niveaux vis-a-vis de cet enfant.
//
// Hypothese (single-box par niveau) : chaque niveau l>=1 est une seule box
// rectangulaire, raffinement ratio 2 du sous-domaine [rCI0..rCI1]x[rCJ0..rCJ1]
// du niveau parent l-1, strictement interieure (>=1 cellule grossiere de marge)
// pour que les ghosts du fin, coarsenises, retombent dans des cellules valides
// du parent. Le niveau 0 couvre le domaine periodique.
//
// Invariants conserves de amr_step_2level, appliques recursivement :
//   - flux grossier sauve aux 4 faces de l'enfant (x dt du niveau courant) ;
//   - flux fins accumules dans le registre du parent (x dt de l'enfant), a
//     chaque sous-pas ;
//   - average_down enfant -> parent sur la region couverte ;
//   - reflux : la cellule grossiere adjacente recoit (somme flux fins - flux
//     grossier x dt) / dx.
// Resultat : conservation a l'arrondi sur toute la hierarchie, stabilite via le
// FillPatch espace-temps (ghosts du fin interpoles entre parent ancien/nouveau).

namespace adc {

// Un niveau de la hierarchie. aux est detenu ailleurs (recalcule par pas) ;
// on n'en garde qu'un pointeur. rC* decrit la region (en coords de CE niveau)
// raffinee par l'enfant l+1 ; valable seulement si has_fine.
struct AmrLevel {
  Fab2D U;                 // etat, 1 composante, 1 ghost
  const Fab2D* aux;        // [phi, gx, gy], ghosts remplis
  double dx, dy;           // pas d'espace de ce niveau
  int rCI0, rCI1, rCJ0, rCJ1;  // region raffinee par l'enfant (coords locales)
  bool has_fine;           // existe-t-il un niveau l+1 ?
};

// Applique U -= dt div(F) a partir de flux deja calcules (ghosts non touches).
inline void apply_flux_div_1c(Fab2D& U, const Fab2D& fx, const Fab2D& fy,
                              double dx, double dy, double dt) {
  Array4 uu = U.array();
  const ConstArray4 FX = fx.const_array();
  const ConstArray4 FY = fy.const_array();
  const Box2D v = U.box();
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      uu(i, j) -= dt * ((FX(i + 1, j) - FX(i, j)) / dx +
                        (FY(i, j + 1) - FY(i, j)) / dy);
}

// Avance recursivement le niveau lev de dt. pOld/pNew = etats parent
// (ancien/nouveau) bornant le pas parent ; frac = position temporelle du
// sous-pas courant dans ce pas (s/r). Le quadruplet preg = registre du parent
// (accumulateur des flux fins de CE niveau) ; nul si lev == 0.
template <class Model>
void subcycle_level(const Model& m, std::vector<AmrLevel>& L, int lev,
                    double dt, const Box2D& dom, const Fab2D* pOld,
                    const Fab2D* pNew, double frac, std::vector<double>* pregL,
                    std::vector<double>* pregR, std::vector<double>* pregB,
                    std::vector<double>* pregT) {
  const int r = 2;
  AmrLevel& lv = L[lev];

  // --- 1. ghosts : periodiques (niveau 0) ou injection espace-temps ---
  if (lev == 0)
    fill_periodic_fab(lv.U, dom);
  else
    fill_fine_ghosts_t(lv.U, *pOld, *pNew, frac);

  // --- 2. flux de ce niveau (etat de debut de sous-pas) ---
  Fab2D fx(xface_box(lv.U.box()), 1, 0), fy(yface_box(lv.U.box()), 1, 0);
  compute_fluxes_1c(m, lv.U, *lv.aux, fx, fy);
  const ConstArray4 FX = fx.const_array();
  const ConstArray4 FY = fy.const_array();

  // --- 3. contribution au registre du parent (flux fins x dt de ce niveau) ---
  if (lev > 0) {
    const AmrLevel& par = L[lev - 1];
    const int pI0 = par.rCI0, pI1 = par.rCI1, pJ0 = par.rCJ0, pJ1 = par.rCJ1;
    for (int J = pJ0; J <= pJ1; ++J) {
      (*pregL)[J - pJ0] += 0.5 * (FX(2 * pI0, 2 * J) + FX(2 * pI0, 2 * J + 1)) * dt;
      (*pregR)[J - pJ0] +=
          0.5 * (FX(2 * pI1 + 2, 2 * J) + FX(2 * pI1 + 2, 2 * J + 1)) * dt;
    }
    for (int I = pI0; I <= pI1; ++I) {
      (*pregB)[I - pI0] += 0.5 * (FY(2 * I, 2 * pJ0) + FY(2 * I + 1, 2 * pJ0)) * dt;
      (*pregT)[I - pI0] +=
          0.5 * (FY(2 * I, 2 * pJ1 + 2) + FY(2 * I + 1, 2 * pJ1 + 2)) * dt;
    }
  }

  if (!lv.has_fine) {
    apply_flux_div_1c(lv.U, fx, fy, lv.dx, lv.dy, dt);  // feuille : simple avance
    return;
  }

  // --- 4. niveau avec enfant : registre local + flux grossier sauve ---
  const int cI0 = lv.rCI0, cI1 = lv.rCI1, cJ0 = lv.rCJ0, cJ1 = lv.rCJ1;
  const int nI = cI1 - cI0 + 1, nJ = cJ1 - cJ0 + 1;
  std::vector<double> cL(nJ), cR(nJ), cB(nI), cT(nI);  // flux grossier (sans dt)
  for (int J = cJ0; J <= cJ1; ++J) {
    cL[J - cJ0] = FX(cI0, J);
    cR[J - cJ0] = FX(cI1 + 1, J);
  }
  for (int I = cI0; I <= cI1; ++I) {
    cB[I - cI0] = FY(I, cJ0);
    cT[I - cI0] = FY(I, cJ1 + 1);
  }
  std::vector<double> fL(nJ, 0), fR(nJ, 0), fB(nI, 0), fT(nI, 0);  // flux fins

  const Fab2D U_old = lv.U;  // etat t (pour l'interp temporelle de l'enfant)
  apply_flux_div_1c(lv.U, fx, fy, lv.dx, lv.dy, dt);  // lv.U devient l'etat t+dt

  // --- 5. sous-cyclage de l'enfant : r sous-pas de dt/r ---
  for (int s = 0; s < r; ++s)
    subcycle_level(m, L, lev + 1, dt / r, dom, &U_old, &lv.U, double(s) / r, &fL,
                   &fR, &fB, &fT);

  average_down_fab(L[lev + 1].U, lv.U, cI0, cI1, cJ0, cJ1);  // sync couvert

  // --- 6. reflux : flux grossier (x dt) remplace par somme des flux fins ---
  Array4 c = lv.U.array();
  for (int J = cJ0; J <= cJ1; ++J) {
    c(cI0 - 1, J) -= (fL[J - cJ0] - cL[J - cJ0] * dt) / lv.dx;
    c(cI1 + 1, J) += (fR[J - cJ0] - cR[J - cJ0] * dt) / lv.dx;
  }
  for (int I = cI0; I <= cI1; ++I) {
    c(I, cJ0 - 1) -= (fB[I - cI0] - cB[I - cI0] * dt) / lv.dy;
    c(I, cJ1 + 1) += (fT[I - cI0] - cT[I - cI0] * dt) / lv.dy;
  }
}

// Driver : un pas dt de la hierarchie complete (niveau 0 = grossier).
template <class Model>
void amr_step_multilevel(const Model& m, std::vector<AmrLevel>& L,
                         const Box2D& dom, double dt) {
  subcycle_level(m, L, 0, dt, dom, nullptr, nullptr, 0.0, nullptr, nullptr,
                 nullptr, nullptr);
}

}  // namespace adc

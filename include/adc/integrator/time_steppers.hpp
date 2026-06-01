#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/mf_arith.hpp>  // saxpy, lincomb
#include <adc/mesh/multifab.hpp>

#include <utility>

// Integrateurs en temps comme OBJETS de premier plan (`take_step`), retour tuteur §8.2 A.
//
// Le schema mathematique (SSPRK, Euler avant, ...) vit ICI, dans le coeur, et le coupleur
// l'APPELLE — au lieu d'inliner SSPRK dans chaque coupleur. Un integrateur est agnostique
// du modele et de la discretisation : il ne voit QUE `rhs_eval(U_stage, R)` — la fleche
// methode-des-lignes qui remplit le residu R = −div F + S — et les operations MultiFab
// (saxpy/lincomb). C'est exactement le contrat « donner un TimeIntegrator au coupleur,
// comme on donne un PhysicalModel ». L'utilisateur peut fournir le sien : meme signature
// `take_step(rhs_eval, U, dt)`.
//
// Le scratch (R, etages U1/U2/U3) est alloue depuis le layout de U : aucun etat porte.
// Ces objets remplacent les copies SSPRK de SystemCoupler / ssprk.hpp (dedup).

namespace adc {

// Contrat : un integrateur sait avancer U de dt via un evaluateur de residu.
template <class I>
concept TimeStepper = requires(const I integ, MultiFab& U, Real dt) {
  integ.take_step([](MultiFab&, MultiFab&) {}, U, dt);
};

// Euler avant (ordre 1) : U <- U + dt R(U).
struct ForwardEuler {
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);
    rhs(U, R);
    saxpy(U, dt, R);
  }
};

// SSP-RK2 (Shu-Osher, 2 etages, ordre 2). Memes operations que l'ancien
// SystemCoupler::advance_explicit_ssprk2 / ssprk.hpp::advance_ssprk2 (bit-identique).
struct SSPRK2Step {
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);
    rhs(U, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);
    rhs(U1, R);
    saxpy(U1, dt, R);
    lincomb(U, Real(0.5), U, Real(0.5), U1);
  }
};

// SSP-RK3 (Shu-Osher, 3 etages, ordre 3). Memes operations que l'ancien
// SystemCoupler::advance_explicit_ssprk3 (bit-identique).
struct SSPRK3Step {
  template <class RhsEval>
  void take_step(RhsEval&& rhs, MultiFab& U, Real dt) const {
    MultiFab R(U.box_array(), U.dmap(), U.ncomp(), 0);
    rhs(U, R);
    MultiFab U1 = U;
    saxpy(U1, dt, R);

    rhs(U1, R);
    MultiFab U2 = U1;
    saxpy(U2, dt, R);
    lincomb(U2, Real(3) / 4, U, Real(1) / 4, U2);

    rhs(U2, R);
    MultiFab U3 = U2;
    saxpy(U3, dt, R);
    lincomb(U, Real(1) / 3, U, Real(2) / 3, U3);
  }
};

}  // namespace adc

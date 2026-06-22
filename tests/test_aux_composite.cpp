// Chantier "Aux extensible", increment 4 : un MODELE COMPOSE propage la largeur aux de ses
// briques. Si une brique (flux ou source) lit un champ auxiliaire supplementaire (ici une
// source magnetisee qui declare n_aux=4 pour lire B_z), CompositeModel::n_aux l'expose au
// systeme (max sur les briques). C'est ce qui permet au runtime System / a la DSL de
// dimensionner le canal aux pour un modele genere/compose. On verifie au niveau de l'operateur
// spatial :
//   (A) CompositeModel<ExBVelocity, BzSource, NoEll>::n_aux == 4 (propagation) ;
//   (B) le modele compose lit bien B_z dans assemble_rhs (grad nul -> flux nul -> R = B_z u) ;
//   (C) un compose SANS brique a champ extra reste a n_aux == 3 (bit-identique).

#include <adc/core/model/physical_model.hpp>
#include <adc/core/state/state.hpp>
#include <adc/core/foundation/types.hpp>
#include <adc/physics/composition/composite.hpp>
#include <adc/physics/bricks/hyperbolic.hpp>  // ExBVelocity (brique hyperbolique 1 var)
#include <adc/physics/bricks/source.hpp>      // NoSource
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/numerics/spatial_operator.hpp>
#include <adc/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace adc;

// Brique de SOURCE magnetisee : S = B_z u (composante 0). Declare n_aux=4 pour lire a.B_z.
struct BzSource {
  static constexpr int n_aux = 4;
  template <class State>
  ADC_HD State apply(const State& u, const Aux& a) const {
    State s{};
    s[0] = a.B_z * u[0];
    return s;
  }
};

// Brique de SECOND MEMBRE elliptique nulle (pas de couplage Poisson).
struct NoEll {
  template <class State>
  ADC_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using MagModel = CompositeModel<ExBVelocity, BzSource, NoEll>;    // lit B_z (via la source)
using PlainModel = CompositeModel<ExBVelocity, NoSource, NoEll>;  // contrat de base

static_assert(PhysicalModel<MagModel> && PhysicalModel<PlainModel>);
// (A) propagation : la source magnetisee remonte n_aux=4 au compose ; le compose nu reste a 3.
static_assert(MagModel::n_aux == 4, "CompositeModel propage n_aux d'une brique");
static_assert(aux_comps<MagModel>() == 4, "aux_comps voit le n_aux du compose");
static_assert(PlainModel::n_aux == 3, "compose sans champ extra = contrat de base");

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 16;
  const Real L = 1.0, Bz = 0.4;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());

  MultiFab U(ba, dm, 1, 1);
  U.set_val(1.0);

  // aux a la largeur du compose (4). phi/grad nuls -> flux ExB nul ; comp 3 = B_z.
  MultiFab aux(ba, dm, MagModel::n_aux, 1);
  aux.set_val(0.0);
  for (int li = 0; li < aux.local_size(); ++li) {
    Fab2D& f = aux.fab(li);
    const Box2D g = f.grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        f(i, j, 3) = Bz;
  }

  // (B) le compose lit B_z : flux nul (grad=0) -> R = source = B_z u = Bz.
  MultiFab R(ba, dm, 1, 0);
  MagModel model{};
  assemble_rhs<NoSlope>(model, U, aux, geom, R);
  double maxerr = 0;
  for (int li = 0; li < R.local_size(); ++li) {
    const ConstArray4 r = R.fab(li).const_array();
    const Box2D v = R.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        maxerr = std::max(maxerr, std::fabs(r(i, j, 0) - Bz));
  }
  std::printf("  composite n_aux=%d : max|R - B_z| = %.2e\n", MagModel::n_aux, maxerr);
  chk(maxerr < 1e-14, "composite_reads_Bz");

  if (fails == 0)
    std::printf("OK test_aux_composite\n");
  return fails == 0 ? 0 : 1;
}

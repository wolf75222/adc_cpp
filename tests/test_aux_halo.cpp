// ADC-369: per-field aux halo/ghost BC. fill_physical_bc gains a COMPONENT-SCOPED overload so a single
// aux component (a model-named field) can carry its OWN boundary policy (foextrap / dirichlet),
// overriding the shared aux BC for that component only, while the all-component overload stays
// strictly bit-identical. This is the mechanism the System / polar / AMR aux-ghost paths reuse to
// honor a per-field halo declared via adc.AuxHalo (Refs ADC-291).

#include <adc/core/types.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  const int n = 8;
  const Box2D dom = Box2D::from_extents(n, n);  // {{0,0},{n-1,n-1}}
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  const int nc = 6;  // base (0..2) + canonical extras (3,4) + one model-named field at comp 5
  MultiFab mf(ba, dm, nc, 1);

  // Distinct, known valid-cell values per component, varying in x so a foextrap ghost (copy of the
  // interior) differs from a dirichlet ghost (2 v - interior): a(i, j, c) = c*100 + i.
  auto seed = [&]() {
    for (int li = 0; li < mf.local_size(); ++li) {
      Array4 a = mf.fab(li).array();
      const Box2D v = mf.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            a(i, j, c) = Real(c * 100 + i);
    }
  };
  seed();

  // NON-periodic domain: shared aux BC = Foextrap on all faces (ghost = mirror interior).
  BCRec bc_foe;
  bc_foe.xlo = bc_foe.xhi = bc_foe.ylo = bc_foe.yhi = BCType::Foextrap;

  // (1) shared fill over ALL components (existing overload, must stay bit-identical).
  fill_physical_bc(mf, dom, bc_foe);
  device_fence();
  {
    const ConstArray4 a = mf.fab(0).const_array();
    // x-low ghost at (lo-1, j) with foextrap -> a(lo, j, c) = c*100 + 0.
    chk(std::fabs(a(-1, 0, 0) - Real(0)) < 1e-12, "foextrap comp0 xlo ghost == interior");
    chk(std::fabs(a(-1, 0, 5) - Real(500)) < 1e-12, "foextrap comp5 xlo ghost == interior");
  }

  // (2) per-field override: ONLY component 5 gets a Dirichlet value V (ghost = 2 V - interior).
  // RED before ADC-369: the component-scoped fill_physical_bc overload does not exist.
  const Real V = 7.0;
  BCRec bc_dir;
  bc_dir.xlo = bc_dir.xhi = bc_dir.ylo = bc_dir.yhi = BCType::Dirichlet;
  bc_dir.xlo_val = bc_dir.xhi_val = bc_dir.ylo_val = bc_dir.yhi_val = V;
  fill_physical_bc(mf, dom, bc_dir, /*comp=*/5);
  device_fence();
  {
    const ConstArray4 a = mf.fab(0).const_array();
    // comp 5 xlo ghost: dirichlet mirror at 2*lo - i - 1 = 0 -> 2 V - a(0, j, 5) = 14 - 500 = -486.
    chk(std::fabs(a(-1, 0, 5) - (2 * V - Real(500))) < 1e-12,
        "dirichlet comp5 xlo ghost == 2V - interior");
    // comp 0 must be UNCHANGED by the comp-5 override (still foextrap from step 1).
    chk(std::fabs(a(-1, 0, 0) - Real(0)) < 1e-12, "comp0 unchanged by comp-5 override");
    // and the named field's halo now genuinely differs from the shared one.
    chk(std::fabs(a(-1, 0, 5) - a(-1, 0, 0)) > 1e-9, "comp5 halo differs from comp0");
  }

  // (3) aux_halo_override keeps PERIODIC faces periodic (polar theta / periodic domain), and replaces
  // only the NON-PERIODIC faces with the field policy. This is what makes a per-field policy safe on
  // polar (radial physical, theta periodic) and on a periodic Cartesian domain.
  {
    BCRec shared;  // polar-like: radial physical (Foextrap), theta periodic
    shared.xlo = shared.xhi = BCType::Foextrap;
    shared.ylo = shared.yhi = BCType::Periodic;
    const BCRec ov = aux_halo_override(shared, AuxHaloPolicy{BCType::Dirichlet, Real(3)});
    chk(ov.xlo == BCType::Dirichlet && std::fabs(ov.xlo_val - Real(3)) < 1e-12,
        "override: xlo -> dirichlet(3)");
    chk(ov.xhi == BCType::Dirichlet && std::fabs(ov.xhi_val - Real(3)) < 1e-12,
        "override: xhi -> dirichlet(3)");
    chk(ov.ylo == BCType::Periodic && ov.yhi == BCType::Periodic, "override: theta stays periodic");
  }

  if (fails == 0)
    std::printf("OK test_aux_halo\n");
  return fails == 0 ? 0 : 1;
}

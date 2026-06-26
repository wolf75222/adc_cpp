// API memoire explicite : sync_host() / sync_device() (cf. for_each.hpp,
// multifab.hpp). Encodent l'intention de residence des donnees. Sous memoire
// unifiee (Kokkos::SharedSpace) sync_host() est un device_fence() cible et
// sync_device() un no-op ; le comportement doit rester BIT-IDENTIQUE a un acces
// hote nu. Ce test verifie :
//   1) les seams libres pops::sync_host()/sync_device() s'appellent (idempotents,
//      sans effet observable sur les donnees) ;
//   2) les methodes MultiFab::sync_host()/sync_device() sont idempotentes : un
//      sync repete ne change AUCUN bit des fabs ;
//   3) une lecture/ecriture hote encadree par sync_host() donne exactement le
//      meme resultat qu'aujourd'hui (set_val + sum inchanges) ;
//   4) sync_device() avant un kernel for_each_cell ne perturbe pas le calcul.

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cstdio>

using namespace pops;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // 1) Les seams libres existent et sont des appels surs et repetables. Sous
  // SharedSpace : sync_host == fence cible, sync_device == no-op. Aucun effet
  // observable, on verifie juste qu'ils s'enchainent sans planter.
  sync_host();
  sync_host();
  sync_device();
  sync_device();
  chk(true, "free_seams_callable");

  Box2D dom = Box2D::from_extents(8, 8);
  BoxArray ba = BoxArray::from_domain(dom, 4);  // 4 boxes
  DistributionMapping dm(ba.size(), n_ranks());
  MultiFab mf(ba, dm, /*ncomp=*/1, /*ngrow=*/1);

  // 2) set_val passe deja par sync_host() en interne. La somme doit etre exacte.
  mf.set_val(3.0);
  const Real s0 = sum(mf);
  chk(s0 == 3.0 * 64, "set_val_sum_exact");

  // 3) IDEMPOTENCE : sync_host()/sync_device() repetes ne touchent aucune
  // donnee. La somme apres N sync est BIT-IDENTIQUE (==, pas une tolerance).
  mf.sync_host();
  mf.sync_device();
  mf.sync_host();
  const Real s1 = sum(mf);
  chk(s1 == s0, "sync_idempotent_sum");

  // ecrire un champ via for_each_cell, encadre par les sync explicites comme le
  // ferait un appelant qui declare son intention de residence.
  mf.sync_device();  // intention : un kernel va ecrire (no-op sous unifiee)
  for (int li = 0; li < mf.local_size(); ++li) {
    Array4 a = mf.fab(li).array();
    for_each_cell(mf.box(li), [a] POPS_HD(int i, int j) { a(i, j, 0) = i + 100.0 * j; });
  }
  mf.sync_host();  // intention : on va relire cote hote (fence cible)

  Real expected = 0;
  for (int j = 0; j < 8; ++j)
    for (int i = 0; i < 8; ++i)
      expected += i + 100.0 * j;
  const Real sf = sum(mf);
  chk(sf == expected, "field_after_sync_exact");

  // 4) re-sync apres lecture : toujours bit-identique (aucune migration).
  mf.sync_host();
  chk(sum(mf) == sf, "resync_no_drift");

  // une cellule precise : la valeur lue cote hote apres sync_host() est exacte.
  bool found = false;
  for (int li = 0; li < mf.local_size(); ++li) {
    if (mf.box(li).contains(5, 6)) {
      found = true;
      mf.sync_host();
      chk(mf.fab(li)(5, 6, 0) == 5 + 600.0, "cell_value_after_sync");
    }
  }
  chk(found, "cell_located");

  if (fails == 0)
    std::printf("OK test_sync_residence\n");
  return fails == 0 ? 0 : 1;
}

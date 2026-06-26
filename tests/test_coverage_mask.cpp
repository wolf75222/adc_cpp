// Contrat du type CoverageMask (revue, point 2 : part "couverture" de CoarseFineInterface).
// Verifie le marquage par box (avec clipping a la region), la requete covered bornee (faux
// hors region), et qu'une cellule non marquee reste non couverte. C'est le masque qui empeche
// le double-reflux d'un joint fin-fin ; son integration AMR est couverte par les tests de
// reflux (np=1/2/4 bit-identiques), ici on fige les mecaniques locales.

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // pops::CoverageMask
#include <pops/mesh/index/box2d.hpp>

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

  // region a origine non nulle : [10,5] x [20,8].
  CoverageMask m(Box2D{{10, 5}, {20, 8}});
  chk(m.NX == 11 && m.NY == 4, "dims");

  // rien marque au depart.
  chk(!m.covered(15, 6), "vide_au_depart");

  // marque une box interieure ; couvert dedans, pas dehors.
  m.mark(Box2D{{12, 6}, {14, 7}});
  chk(m.covered(12, 6) && m.covered(14, 7) && m.covered(13, 6), "marque_dedans");
  chk(!m.covered(11, 6) && !m.covered(15, 6) && !m.covered(13, 5), "non_marque_reste_faux");

  // requete hors region : toujours faux, jamais de crash.
  chk(!m.covered(5, 5) && !m.covered(25, 8) && !m.covered(15, 4) && !m.covered(15, 9),
      "hors_region_faux");

  // marque une box debordant la region : clippee a la region.
  m.mark(Box2D{{18, 7}, {30, 20}});
  chk(m.covered(18, 7) && m.covered(20, 8), "clip_dedans");
  chk(!m.covered(21, 8) && !m.covered(20, 9), "clip_pas_de_debordement");

  if (fails == 0)
    std::printf("OK test_coverage_mask\n");
  return fails == 0 ? 0 : 1;
}

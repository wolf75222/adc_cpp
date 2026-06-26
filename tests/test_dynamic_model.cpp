// Test de l'interface de modele TYPE-ERASED (pops::IModel / ModelAdapter) : un modele statique
// (Euler) enrobe dans IModel<4> et dispatche par vtable doit donner exactement le meme flux et la
// meme vitesse d'onde que l'appel direct. C'est le mecanisme qui permet d'utiliser, a l'execution,
// une brique generee/JIT dont le type n'est pas connu a la compilation (cf. dynamic_model.hpp).
#include <pops/physics/fluids/euler.hpp>
#include <pops/runtime/dynamic/dynamic_model.hpp>

#include <cmath>
#include <cstdio>

using State = pops::StateVec<4>;

static_assert(std::is_base_of_v<pops::IModel<4>, pops::ModelAdapter<pops::Euler>>,
              "ModelAdapter<Euler> doit deriver de IModel<4>");

int main() {
  pops::Euler ref;
  ref.gamma = 1.4;
  auto dyn = pops::make_dynamic(ref);  // std::unique_ptr<pops::IModel<4>>
  const pops::IModel<4>& m = *dyn;

  pops::Aux a{};
  const double S[][4] = {
      {1.0, 0.2, -0.1, 2.5}, {2.0, 0.5, 0.3, 6.0}, {0.5, -0.2, 0.1, 1.8}, {1.5, 0.0, 0.0, 3.0}};
  double maxdiff = 0.0;
  for (const auto& s : S) {
    State u{};
    for (int i = 0; i < 4; ++i)
      u[i] = s[i];
    for (int dir = 0; dir < 2; ++dir) {
      State fr = ref.flux(u, a, dir);
      State fg = m.flux(u, a, dir);  // dispatch virtuel (type efface)
      for (int i = 0; i < 4; ++i)
        maxdiff = std::fmax(maxdiff, std::fabs(fr[i] - fg[i]));
      maxdiff = std::fmax(maxdiff,
                          std::fabs(ref.max_wave_speed(u, a, dir) - m.max_wave_speed(u, a, dir)));
    }
  }
  if (maxdiff > 1e-14) {
    std::printf("FAIL test_dynamic_model : IModel != direct (maxdiff=%.3e)\n", maxdiff);
    return 1;
  }
  std::printf("OK test_dynamic_model : IModel<4>(Euler) == Euler direct (maxdiff=%.1e)\n", maxdiff);
  return 0;
}

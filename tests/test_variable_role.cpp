// Test des roles de variables : adresser une composante par son SENS (index_of(role)) plutot que
// par un indice magique. Verifie Euler / isotherme / ExB.
#include <adc/physics/bricks/bricks.hpp>
#include <adc/physics/fluids/euler.hpp>

#include <cstdio>

using R = adc::VariableRole;

int main() {
  const adc::VariableSet c = adc::Euler::conservative_vars();
  if (c.index_of(R::Density) != 0 || c.index_of(R::MomentumX) != 1 ||
      c.index_of(R::MomentumY) != 2 || c.index_of(R::Energy) != 3) {
    std::printf("FAIL : roles conservatifs Euler\n");
    return 1;
  }
  if (c.index_of(R::Pressure) != -1) {  // la pression n'est pas une variable conservative
    std::printf("FAIL : Pressure devrait etre absente des conservatives\n");
    return 1;
  }
  const adc::VariableSet p = adc::Euler::primitive_vars();
  if (p.index_of(R::Pressure) != 3 || p.index_of(R::VelocityX) != 1) {
    std::printf("FAIL : roles primitifs Euler\n");
    return 1;
  }
  const adc::Variable v = p.at(1);
  if (v.name != "u" || v.role != R::VelocityX || v.component != 1) {
    std::printf("FAIL : Variable::at\n");
    return 1;
  }
  // isotherme (3 var) + ExB (1 var)
  if (adc::IsothermalFlux::conservative_vars().index_of(R::MomentumY) != 2) {
    std::printf("FAIL : roles isotherme\n");
    return 1;
  }
  if (adc::ExBVelocity::conservative_vars().index_of(R::Density) != 0) {
    std::printf("FAIL : role ExB\n");
    return 1;
  }
  std::printf(
      "OK test_variable_role : composantes indexees par role (Density/Momentum/Energy/...)\n");
  return 0;
}

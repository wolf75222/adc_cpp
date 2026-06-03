"""Test des ROLES physiques portes par une brique generee (adc.dsl.emit_cpp_brick).

Une brique generee DECLARE desormais le SENS de ses composantes (densite, qte de mvt, energie...)
via adc::VariableSet::roles, et non plus seulement leurs noms. Les couplages inter-especes du System
resolvent ainsi une composante par index_of(role) au lieu d'un indice litteral.

Ce test verifie :
(1) FORME (sans compilateur) : Euler (noms standards) emet les roles CANONIQUES (Density, MomentumX,
    MomentumY, Energy / Pressure) ; un layout NON STANDARD (qte de mvt avant densite) avec roles=
    explicites emet ces roles dans l'ordre demande ; un modele aux noms inconnus n'emet PAS de roles
    (retro-compat stricte : le 4e champ VariableSet::roles reste absent, fallback indices historiques).
(2) RESOLUTION (si compilateur + en-tetes adc) : la brique au layout non standard compile, satisfait
    adc::HyperbolicModel, et index_of(MomentumX/MomentumY/Density/Energy) retrouve la BONNE composante
    QUELLE QUE SOIT sa position -- c'est exactement ce dont depend la resolution par role des couplages.
Lance avec python3.
"""
import os
import shutil
import subprocess
import tempfile

from adc import dsl

GAMMA = 1.4
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def build_euler_brick():
    """Euler standard : noms usuels (rho, rho_u, rho_v, E) -> roles deduits par mapping canonique."""
    e = dsl.HyperbolicModel("euler")
    rho, rhou, rhov, E = e.conservative_vars("rho", "rho_u", "rho_v", "E")
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    return e


def build_shuffled_brick():
    """Euler au layout NON STANDARD : composantes rangees (mom_y, E, mom_x, rho). Les noms ne suivent
    pas la convention, donc on impose les roles explicitement via roles=. La physique reste Euler ;
    seule la POSITION des composantes change. Sert a prouver que index_of(role) resout par le SENS."""
    e = dsl.HyperbolicModel("euler_shuf")
    # ordre des conservatives : [rho_v(my), E, rho_u(mx), rho]
    my, E, mx, rho = e.conservative_vars(
        "my", "ee", "mx", "rho",
        roles=["MomentumY", "Energy", "MomentumX", "Density"])
    u = e.primitive("u", mx / rho)
    v = e.primitive("v", my / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = dsl.sqrt(GAMMA * p / rho)
    # flux Euler reordonne pour suivre le layout [my, E, mx, rho]
    e.set_flux(x=[mx * v, rho * H * u, mx * u + p, mx],
               y=[my * v + p, rho * H * v, my * u, my])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    # Prim au layout primitif STANDARD (rho, u, v, p) avec ses roles ; to_conservative produit
    # ensuite le layout conservatif SHUFFLE [my, E, mx, rho] a partir de ces primitives.
    e.set_primitive_state(rho, u, v, p,
                          roles=["Density", "VelocityX", "VelocityY", "Pressure"])
    e.set_conservative_from([rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v),
                             rho * u, rho])
    return e


def build_scalar_brick():
    """Modele a NOM inconnu (q) : aucun role canonique -> brique sans roles (retro-compat stricte)."""
    e = dsl.HyperbolicModel("scal")
    (q,) = e.conservative_vars("q")
    e.set_flux(x=[q], y=[q])
    e.set_eigenvalues(x=[q], y=[q])
    e.set_primitive_state(q)
    e.set_conservative_from([q])
    return e


HARNESS = r"""
#include <adc/physics/euler.hpp>
#include <adc/core/physical_model.hpp>
%s
#include <cstdio>

using R = adc::VariableRole;

static_assert(adc::HyperbolicModel<adc_generated::ShufGen>, "brique non standard non conforme au concept");

int main() {
  const adc::VariableSet c = adc_generated::ShufGen::conservative_vars();
  // layout = [my, E, mx, rho] : index_of(role) doit retrouver la composante par son SENS.
  if (c.index_of(R::MomentumY) != 0) { printf("FAIL MomentumY=%%d\n", c.index_of(R::MomentumY)); return 1; }
  if (c.index_of(R::Energy)    != 1) { printf("FAIL Energy=%%d\n",    c.index_of(R::Energy));    return 1; }
  if (c.index_of(R::MomentumX) != 2) { printf("FAIL MomentumX=%%d\n", c.index_of(R::MomentumX)); return 1; }
  if (c.index_of(R::Density)   != 3) { printf("FAIL Density=%%d\n",   c.index_of(R::Density));   return 1; }
  if (c.index_of(R::Pressure)  != -1){ printf("FAIL Pressure devrait etre absente\n");          return 1; }
  printf("OK\n");
  return 0;
}
"""


def main():
    # (1) FORME : roles emis pour Euler standard ----------------------------------------------
    euler = build_euler_brick().emit_cpp_brick(name="EulerGen")
    assert ("conservative_vars() { return {adc::VariableKind::Conservative, "
            '{"rho", "rho_u", "rho_v", "E"}, 4, {adc::VariableRole::Density, '
            "adc::VariableRole::MomentumX, adc::VariableRole::MomentumY, "
            "adc::VariableRole::Energy}}; }") in euler, "roles conservatifs Euler absents/incorrects"
    assert ("primitive_vars() { return {adc::VariableKind::Primitive, "
            '{"rho", "u", "v", "p"}, 4, {adc::VariableRole::Density, '
            "adc::VariableRole::VelocityX, adc::VariableRole::VelocityY, "
            "adc::VariableRole::Pressure}}; }") in euler, "roles primitifs Euler absents/incorrects"
    print("OK  Euler (noms standards) : roles canoniques emis (Density/Momentum/Energy/Pressure)")

    # layout non standard : roles dans l'ordre demande
    shuf = build_shuffled_brick().emit_cpp_brick(name="ShufGen")
    assert ("{adc::VariableRole::MomentumY, adc::VariableRole::Energy, "
            "adc::VariableRole::MomentumX, adc::VariableRole::Density}") in shuf, \
        "roles du layout non standard incorrects"
    print("OK  layout non standard : roles explicites emis dans l'ordre du layout")

    # retro-compat stricte : noms inconnus -> AUCUN champ roles (init 3 champs comme avant)
    scal = build_scalar_brick().emit_cpp_brick(name="ScalGen")
    assert ('conservative_vars() { return {adc::VariableKind::Conservative, {"q"}, 1}; }') in scal, \
        "modele a roles Custom devrait emettre l'init historique 3 champs (retro-compat)"
    assert "VariableRole" not in scal, "modele a roles Custom ne doit emettre aucun role"
    print("OK  noms inconnus : aucun role emis (retro-compat bit-exacte, fallback indices)")

    # (2) RESOLUTION par role a travers le C++ (si compilateur dispo) --------------------------
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  compilateur ou en-tetes adc absents -> resolution sautee (%s)" % INCLUDE)
        print("test_dsl_roles : OK (forme des roles seulement)")
        return

    prog = HARNESS % shuf
    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "roles.cpp")
        exe = os.path.join(tmp, "roles")
        with open(cpp, "w") as f:
            f.write(prog)
        subprocess.run([cxx, "-std=c++20", "-O2", "-I", INCLUDE, cpp, "-o", exe], check=True)
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "OK", "index_of(role) n'a pas retrouve la bonne composante : %s" % out.strip()
    print("OK  index_of(role) retrouve la composante par son SENS dans un layout non standard")
    print("test_dsl_roles : tout est vert")


if __name__ == "__main__":
    main()

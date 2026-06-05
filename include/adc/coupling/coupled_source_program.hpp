#pragma once

#include <adc/core/types.hpp>
#include <adc/mesh/fab2d.hpp>  // Array4 (POD device-copyable)

#include <cmath>

// Interprete de SOURCE COUPLEE generique (P5 phase 1, splitting EXPLICITE).
//
// Les couplages NOMMES (add_ionization / add_collision / add_thermal_exchange) figent leur formule
// EN DUR dans une lambda C++. Pour un couplage ARBITRAIRE decrit cote Python (adc.dsl.CoupledSource),
// on ne veut NI generer/compiler un .so par couplage (codegen + ABI lourds), NI un callback Python par
// cellule. On transporte donc l'expression symbolique (arbre adc.dsl.Expr) sous forme de BYTECODE
// postfixe (pile) a evaluer dans le MEME for_each_cell device que les couplages nommes.
//
// Le bytecode est un POD a capacite fixe : il se capture par valeur dans un kernel device (comme un
// Array4). L'evaluateur est un foncteur NOMME (pas de lambda etendue) -> device-clean (nvcc/Kokkos).
//
// Registres lus par le programme (charges UNE fois par cellule) :
//   r[0 .. n_in-1]      : champs d'entree = u_block(i, j, comp) pour chaque (bloc, role) declare ;
//   r[n_in .. n_in+n_const-1] : constantes (parametres .param(), inlinees comme reels).
// Les opcodes manipulent une pile de Real ; le sommet final est la valeur de source S du terme.

namespace adc {

// Opcodes de la machine a pile. Valeurs stables (transportees telles quelles par l'ABI Python->C++).
enum class CsOp : int {
  PushReg = 0,  // empile r[arg]  (entree ou constante)
  Add = 1,      // b = pop ; a = pop ; push a + b
  Sub = 2,      // b = pop ; a = pop ; push a - b
  Mul = 3,      // b = pop ; a = pop ; push a * b
  Div = 4,      // b = pop ; a = pop ; push a / b
  Neg = 5,      // a = pop ; push -a
  Pow = 6,      // b = pop ; a = pop ; push pow(a, b)
  Sqrt = 7,     // a = pop ; push sqrt(a)
};

// Capacites fixes (bornent la pile et le nombre de registres dans le kernel device, ou l'allocation
// dynamique est proscrite). Genereuses pour des formules de couplage realistes ; depassees, l'API
// Python leve une erreur EXPLICITE avant d'atteindre le device.
inline constexpr int kCsMaxReg = 32;    // entrees + constantes
inline constexpr int kCsMaxStack = 32;  // profondeur de pile postfixe
inline constexpr int kCsMaxProg = 256;  // longueur d'un programme (opcodes)
inline constexpr int kCsMaxTerms = 16;  // termes de source (un par .add)

// Un programme postfixe : suite d'(opcode, argument). arg n'est lu que par PushReg (indice de registre).
struct CsProgram {
  int len = 0;
  int op[kCsMaxProg] = {};
  int arg[kCsMaxProg] = {};

  // Evalue le programme sur les registres @p reg (deja charges pour la cellule courante). Pile locale
  // a capacite fixe : aucun heap, device-callable. Un programme bien forme laisse exactement une valeur
  // au sommet (verifie cote Python a la construction). Pile insuffisante -> on borne (sp ne deborde pas).
  ADC_HD Real eval(const Real* reg) const {
    Real st[kCsMaxStack];
    int sp = 0;
    for (int k = 0; k < len; ++k) {
      switch (static_cast<CsOp>(op[k])) {
        case CsOp::PushReg:
          if (sp < kCsMaxStack) st[sp++] = reg[arg[k]];
          break;
        case CsOp::Add: { Real b = st[--sp]; Real a = st[--sp]; st[sp++] = a + b; } break;
        case CsOp::Sub: { Real b = st[--sp]; Real a = st[--sp]; st[sp++] = a - b; } break;
        case CsOp::Mul: { Real b = st[--sp]; Real a = st[--sp]; st[sp++] = a * b; } break;
        case CsOp::Div: { Real b = st[--sp]; Real a = st[--sp]; st[sp++] = a / b; } break;
        case CsOp::Neg: { Real a = st[--sp]; st[sp++] = -a; } break;
        case CsOp::Pow: { Real b = st[--sp]; Real a = st[--sp]; st[sp++] = std::pow(a, b); } break;
        case CsOp::Sqrt: { Real a = st[--sp]; st[sp++] = std::sqrt(a); } break;
      }
    }
    return sp > 0 ? st[sp - 1] : Real(0);
  }
};

// Foncteur device d'application d'UNE source couplee sur UNE box (un fab local). Capture par VALEUR des
// POD (Array4 d'entree, Array4 de sortie, programmes, constantes) : aucun pointeur vers un objet hote,
// device-clean. operator()(i, j) :
//   1. charge les registres : r[c] = in[c](i, j, in_comp[c]) pour les entrees, puis les constantes ;
//   2. pour chaque terme t : S_t = prog[t].eval(r) ; out[t](i, j, out_comp[t]) += dt * S_t.
// Les ecritures de sortie sont ADDITIVES (forward-Euler split) ; plusieurs termes peuvent viser le meme
// (bloc, comp) -- les sources s'additionnent, cohérent avec une somme de termes sources.
struct CoupledSourceKernel {
  Array4 in[kCsMaxReg];   // champs d'entree (un par (bloc, role) lu) ; seuls n_in premiers sont valides
  int in_comp[kCsMaxReg];
  int n_in = 0;

  Real consts[kCsMaxReg];  // constantes (parametres), chargees dans r[n_in ..]
  int n_const = 0;

  Array4 out[kCsMaxTerms];  // cible de chaque terme (peut aliaser une entree : meme fab)
  int out_comp[kCsMaxTerms];
  CsProgram prog[kCsMaxTerms];
  int n_terms = 0;

  Real dt = Real(0);

  ADC_HD void operator()(int i, int j) const {
    Real reg[kCsMaxReg];
    for (int c = 0; c < n_in; ++c) reg[c] = in[c](i, j, in_comp[c]);
    for (int c = 0; c < n_const; ++c) reg[n_in + c] = consts[c];
    // On evalue TOUS les termes sur l'etat AU DEBUT du pas (reg fige), puis on ecrit : un terme ecrivant
    // une cible qui est aussi une entree ne perturbe pas l'evaluation des autres termes (splitting additif
    // explicite coherent, ordre des .add sans importance pour le resultat au 1er ordre).
    Real sval[kCsMaxTerms];
    for (int t = 0; t < n_terms; ++t) sval[t] = prog[t].eval(reg);
    for (int t = 0; t < n_terms; ++t) out[t](i, j, out_comp[t]) += dt * sval[t];
  }
};

}  // namespace adc

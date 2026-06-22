/// @file
/// @brief Generic COUPLED SOURCE interpreter: postfix bytecode evaluated on device (P5 phase 1).
///
/// For an ARBITRARY coupling described on the Python side (adc.dsl.CoupledSource), we want NEITHER to
/// generate one .so per coupling, NOR one Python callback per cell. So we transport the symbolic
/// expression as postfix (stack) BYTECODE to evaluate inside the SAME for_each_cell device kernel as
/// the named couplings. The bytecode (CsProgram) is a FIXED-CAPACITY POD: capturable by value in a
/// device kernel (like an Array4); the evaluator (CoupledSourceKernel) is a NAMED FUNCTOR (no extended
/// lambda) -> device-clean. Per-cell registers: r[0..n_in-1] = input fields, r[n_in..] = constants.
/// Output writes are ADDITIVE (forward-Euler split). The capacities (kCsMaxReg, kCsMaxStack, kCsMaxProg,
/// kCsMaxTerms) bound the stack/registers; an overflow raises an error on the Python side BEFORE the
/// device.

#pragma once

#include <adc/core/foundation/types.hpp>
#include <adc/mesh/storage/fab2d.hpp>  // Array4 (POD device-copyable)

#include <cassert>
#include <cmath>

namespace adc {

/// Opcodes of the postfix stack machine. STABLE integer values (part of the Python -> C++ ABI):
/// do not reorder or reassign.
enum class CsOp : int {
  PushReg = 0,  // push r[arg]  (input or constant)
  Add = 1,      // b = pop; a = pop; push a + b
  Sub = 2,      // b = pop; a = pop; push a - b
  Mul = 3,      // b = pop; a = pop; push a * b
  Div = 4,      // b = pop; a = pop; push a / b
  Neg = 5,      // a = pop; push -a
  Pow = 6,      // b = pop; a = pop; push pow(a, b)
  Sqrt = 7,     // a = pop; push sqrt(a)
};

// Fixed capacities (bound the stack and the number of registers in the device kernel, where dynamic
// allocation is forbidden). Generous for realistic coupling formulas; if exceeded, the Python API
// raises an EXPLICIT error before reaching the device.
inline constexpr int kCsMaxReg = 32;    // inputs + constants
inline constexpr int kCsMaxStack = 32;  // postfix stack depth
inline constexpr int kCsMaxProg = 256;  // length of a program (opcodes)
inline constexpr int kCsMaxTerms = 16;  // source terms (one per .add)

/// Fixed-capacity postfix program (POD device-copyable): len opcodes, arg read only by PushReg
/// (register index). Capturable by value in a device kernel.
struct CsProgram {
  int len = 0;
  int op[kCsMaxProg] = {};
  int arg[kCsMaxProg] = {};

  /// Evaluate the program on the registers @p reg (loaded for the cell). LOCAL fixed-capacity stack,
  /// no heap -> ADC_HD device-callable. PRECONDITION: well-formed program (checked on the Python
  /// side); returns the final top, or Real(0) if the stack is empty.
  ///
  /// DEFENSIVE stack bounds (program assumed well-formed on the Python side, but we never read out of
  /// bounds on device): a PushReg pushes only if sp < kCsMaxStack (bounded overflow); a binary (resp.
  /// unary) operator requires sp >= 2 (resp. >= 1) before popping, otherwise it is skipped (bounded
  /// underflow, no st[sp<0] access). In debug, an assert flags the ill-formed program.
  ADC_HD Real eval(const Real* reg) const {
    Real st[kCsMaxStack];
    int sp = 0;
    for (int k = 0; k < len; ++k) {
      switch (static_cast<CsOp>(op[k])) {
        case CsOp::PushReg:
          if (sp < kCsMaxStack)
            st[sp++] = reg[arg[k]];
          break;
        case CsOp::Add: {
          assert(sp >= 2);
          if (sp < 2)
            break;
          Real b = st[--sp];
          Real a = st[--sp];
          st[sp++] = a + b;
        } break;
        case CsOp::Sub: {
          assert(sp >= 2);
          if (sp < 2)
            break;
          Real b = st[--sp];
          Real a = st[--sp];
          st[sp++] = a - b;
        } break;
        case CsOp::Mul: {
          assert(sp >= 2);
          if (sp < 2)
            break;
          Real b = st[--sp];
          Real a = st[--sp];
          st[sp++] = a * b;
        } break;
        case CsOp::Div: {
          assert(sp >= 2);
          if (sp < 2)
            break;
          Real b = st[--sp];
          Real a = st[--sp];
          st[sp++] = a / b;
        } break;
        case CsOp::Neg: {
          assert(sp >= 1);
          if (sp < 1)
            break;
          Real a = st[--sp];
          st[sp++] = -a;
        } break;
        case CsOp::Pow: {
          assert(sp >= 2);
          if (sp < 2)
            break;
          Real b = st[--sp];
          Real a = st[--sp];
          st[sp++] = std::pow(a, b);
        } break;
        case CsOp::Sqrt: {
          assert(sp >= 1);
          if (sp < 1)
            break;
          Real a = st[--sp];
          st[sp++] = std::sqrt(a);
        } break;
      }
    }
    return sp > 0 ? st[sp - 1] : Real(0);
  }
};

/// Device functor applying ONE coupled source over a box: captures the PODs by VALUE
/// (input/output Array4, programs, constants) -> device-clean. operator()(i, j) loads the
/// registers, evaluates each term and WRITES additively out[t] += dt * S_t (forward-Euler
/// split). All terms are evaluated on the state AT THE START of the step (frozen reg) before the writes.
struct CoupledSourceKernel {
  Array4 in[kCsMaxReg];  // input fields (one per (block, role) read); only the first n_in are valid
  int in_comp[kCsMaxReg];
  int n_in = 0;

  Real consts[kCsMaxReg];  // constants (parameters), loaded into r[n_in ..]
  int n_const = 0;

  Array4 out[kCsMaxTerms];  // target of each term (may alias an input: same fab)
  int out_comp[kCsMaxTerms];
  CsProgram prog[kCsMaxTerms];
  int n_terms = 0;

  Real dt = Real(0);

  ADC_HD void operator()(int i, int j) const {
    Real reg[kCsMaxReg];
    for (int c = 0; c < n_in; ++c)
      reg[c] = in[c](i, j, in_comp[c]);
    for (int c = 0; c < n_const; ++c)
      reg[n_in + c] = consts[c];
    // We evaluate ALL terms on the state AT THE START of the step (frozen reg), then write: a term writing
    // a target that is also an input does not perturb the evaluation of the other terms (explicit additive
    // splitting consistent, order of the .add irrelevant to the result at 1st order).
    Real sval[kCsMaxTerms];
    for (int t = 0; t < n_terms; ++t)
      sval[t] = prog[t].eval(reg);
    for (int t = 0; t < n_terms; ++t)
      out[t](i, j, out_comp[t]) += dt * sval[t];
  }
};

// MAX REDUCTION functor of a PER-CELL COUPLED FREQUENCY (CoupledSource.frequency with an Expr,
// refinement of the declared CONSTANT frequency). mu(i, j) = prog.eval(cell registers): same
// vocabulary as the source terms ((block, role) input fields + .param() constants). Modeled on
// CoupledSourceKernel but READ-ONLY (no writes); it reduces the MAX instead of writing outputs.
// Captures the PODs by VALUE (input Array4, program, constants) -> device-clean (nvcc/Kokkos);
// the signature (i, j, Real& acc) is the one expected by reduce_max_cell. Registers loaded
// EXACTLY like the source kernel: r[0 .. n_in-1] = inputs, r[n_in ..] = constants -> the register
// indices of the program (emitted on the Python side against the SAME table) are consistent. The step
// bound that derives from it is dt <= cfl / max(mu) (global max aggregated by step_cfl, all_reduce_max
// over all ranks).
struct CoupledFreqKernel {
  Array4 in[kCsMaxReg];  // input fields (one per (block, role) read); only the first n_in are valid
  int in_comp[kCsMaxReg];
  int n_in = 0;

  Real consts[kCsMaxReg];  // constants (parameters), loaded into r[n_in ..]
  int n_const = 0;

  CsProgram prog;  // postfix program of the frequency (final top = mu of the cell)

  ADC_HD void operator()(int i, int j, Real& acc) const {
    Real reg[kCsMaxReg];
    for (int c = 0; c < n_in; ++c)
      reg[c] = in[c](i, j, in_comp[c]);
    for (int c = 0; c < n_const; ++c)
      reg[n_in + c] = consts[c];
    const Real mu = prog.eval(reg);
    if (mu > acc)
      acc = mu;
  }
};

}  // namespace adc

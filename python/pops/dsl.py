"""pops.dsl : TRANSITIONAL re-export shim for the symbolic mini-DSL.

The model-authoring layer moved to :mod:`pops.physics` (Spec 4). This module is a
thin RE-EXPORT shim kept so the historical surface (``from pops import dsl``,
``dsl.Model``, ``from pops.dsl import HyperbolicModel/Param/CoupledSource/...``)
keeps working during the migration. It owns nothing: every name is re-exported
from its new home -- the symbolic IR (:mod:`pops.ir`), the build/compile engine
(:mod:`pops.codegen`) and the authoring facade (:mod:`pops.physics`).

The flat ``dsl.py`` is DELETED in a later Spec-4 step (PR-E), once every consumer
migrates to the package APIs (``pops.physics.Model`` / ``pops.compile_problem`` /
``pops.codegen``). ``dsl.Model`` is the PDE facade (:class:`pops.physics.facade.Model`),
distinct from the blackboard ``pops.physics.Model`` (the board facade).
"""
# --- Symbolic IR (single source of truth: pops.ir) ---------------------------
from pops.ir.expr import (Expr, _wrap, Const, Var, _Bin, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Abs, Sign)  # noqa: F401
from pops.ir.ops import sqrt, abs_, sign, eig_max_im, eig_lmin, eig_lmax, eig_all_real, left, right  # noqa: F401
from pops.ir.values import _EIG_FIELDS, _EIG_PREDICATES, EigWitness, StateRef, RuntimeParamRef  # noqa: F401
from pops.ir.visitors import _children, _expr_uses_cons_or_prim, _key  # noqa: F401
from pops.ir.lowering import _is_const, _s_add, _s_neg, _s_sub, _s_mul, _s_div, _s_pow, diff  # noqa: F401

# --- codegen infrastructure (single source of truth: pops.codegen) -----------
from pops.codegen.toolchain import (  # noqa: F401
    pops_header_signature, pops_include, loader_cxx_std, _pops_cxx_std_from_module, _pops_module,
    loader_cxx_compiler, resolve_auto_backend, _check_headers_match_module, _default_cxx,
    _STD_ALIAS, _run_compile, _probe_cache, _probe_cxx_std,
    _native_kokkos_root, _libomp_prefix, _native_feature_key,
    _warn_kokkos_parity, _env_truthy, _native_kokkos_compiler, _pops_import_lib,
    _native_kokkos_flags, pops_loader_build_flags,
)
from pops.codegen.cache import (  # noqa: F401
    pops_cache_dir, _cache_so_path, _process_so_backend, _backend_distinct_so_path,
    _record_so_backend, _native_mpi_flags, _platform_cache_key, _dsl_optflags,
    _DSL_OPTFLAGS_DEFAULT,
)
from pops.codegen.abi import (  # noqa: F401
    module_header_signature, check_compiled_matches_module, _abi_key_python,
)
from pops.codegen.cpp_writer import (  # noqa: F401
    _cpp_expand, _cpp_cse, _cse_emit, _collect_eig_witnesses, _eig_witness_helpers,
    _count_cons_denoms, _recip_rewrite, _dir_key, _roe_validate, _cpp_roe,
)
from pops.codegen.compile import (  # noqa: F401
    _BACKEND_CAPS, _BACKENDS,
    model_hash as _cg_model_hash,
    adder_for as _cg_adder_for,
    emit_cpp_so_source as _cg_emit_cpp_so_source,
    emit_cpp_aot_source as _cg_emit_cpp_aot_source,
    emit_cpp_native_loader as _cg_emit_cpp_native_loader,
    compile_so as _cg_compile_so,
    compile_aot as _cg_compile_aot,
    compile_native as _cg_compile_native,
    compile_or_jit as _cg_compile_or_jit,
    compile_model as _cg_compile_model,
    compile_problem, _module_to_model,
)
from pops.codegen.loader import CompiledModel, CompiledProblem  # noqa: F401

# --- model authoring layer (single source of truth: pops.physics) ------------
# dsl.Model is the PDE facade (pops.physics.facade.Model); the blackboard board
# facade (pops.physics.Model) is a DIFFERENT class.
from pops.physics.facade import Model  # noqa: F401
from pops.physics.model import HyperbolicModel, Param, RuntimeParam  # noqa: F401
from pops.physics.aux import (  # noqa: F401
    AUX_CANONICAL, AUX_BASE_COMPS, AUX_NAMED_BASE, AUX_NAMED_MAX, CANONICAL_ROLES,
    _K_MAX_RUNTIME_PARAMS, aux_n_aux, aux_total_n_aux, role_of, roles_for)
from pops.physics.bricks import (  # noqa: F401
    NativeBrick, CompiledBrick, CompiledHyperbolicBrick, CompiledSourceBrick,
    CompiledEllipticBrick, HyperbolicBrick, SourceBrick, EllipticBrick)
from pops.physics.hybrid import HybridModel  # noqa: F401
from pops.physics.multispecies import (  # noqa: F401
    CoupledSource, CompiledCoupledSource, _CsField, _CsBlock, _role_canonical,
    _ROLE_TO_CANONICAL, _CS_PUSHREG, _CS_ADD, _CS_SUB, _CS_MUL, _CS_DIV, _CS_NEG,
    _CS_POW, _CS_SQRT, _CS_MAX_REG, _CS_MAX_TERMS, _CS_MAX_PROG)

"""Spec 4 architecture test: static acyclic-layer check over python/adc.

Uses ast to parse each .py file and inspects cross-package import edges.
No modules are executed; this test is pure stdlib (pathlib, ast, sys).

Layer rules (a pkg may only import from listed peers):
    ir:      {}                       (no adc.* imports allowed)
    model:   {ir}
    physics: {ir, model}
    time:    {ir, model}
    lib:     {ir, model, time, physics}
    codegen: {ir, model, time, physics, lib}
    runtime: {}   (runtime talks to _adc directly; no adc.<pkg> imports from runtime)

For packages that do not yet exist, the test is silently skipped (forward-looking).

Expected to FAIL against the current (pre-Spec-4) tree.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADC_PKG = REPO_ROOT / "python" / "adc"

# Allowed cross-package dependencies per source package.
# Key = source pkg name under python/adc/; value = set of adc pkgs it may import.
ALLOWED_EDGES: dict[str, frozenset[str]] = {
    "ir":      frozenset(),
    "model":   frozenset({"ir"}),
    "physics": frozenset({"ir", "model"}),
    "time":    frozenset({"ir", "model"}),
    "lib":     frozenset({"ir", "model", "time", "physics"}),
    "codegen": frozenset({"ir", "model", "time", "physics", "lib"}),
    "runtime": frozenset(),  # must not import other adc packages
}


def _collect_adc_imports(tree: ast.AST) -> list[str]:
    """Return a list of top-level adc package names imported in the AST.

    Collects targets from:
      import adc.X[.Y...]   -> X
      from adc.X[.Y...] import ...  -> X
      from adc import X  -> X  (only if X is a known pkg name)
    Ignores relative imports (from . import ...).
    """
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("adc."):
                    # adc.X.Y... -> X
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        targets.append(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import — intra-package, skip
            mod = node.module or ""
            if mod == "adc":
                # from adc import X — collect only known pkg names
                for alias in node.names:
                    if alias.name in ALLOWED_EDGES:
                        targets.append(alias.name)
            elif mod.startswith("adc."):
                parts = mod.split(".")
                if len(parts) >= 2:
                    targets.append(parts[1])
    return targets


def _check_package_layer(src_pkg: str) -> list[str]:
    """Return violation strings for all illegal cross-package imports in src_pkg."""
    pkg_dir = ADC_PKG / src_pkg
    if not pkg_dir.is_dir():
        return []  # not yet present — skip

    allowed = ALLOWED_EDGES.get(src_pkg, frozenset())
    violations: list[str] = []

    for py_file in sorted(pkg_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue

        source = py_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:
            violations.append(f"{py_file.relative_to(REPO_ROOT)}: SyntaxError: {exc}")
            continue

        rel = str(py_file.relative_to(REPO_ROOT))
        for imported_pkg in _collect_adc_imports(tree):
            # Ignore self-imports (intra-package)
            if imported_pkg == src_pkg:
                continue
            # Ignore unknown tokens (e.g. "adc" itself, or top-level aliases)
            if imported_pkg not in ALLOWED_EDGES:
                continue
            if imported_pkg not in allowed:
                violations.append(
                    f"{rel}: illegal edge {src_pkg} -> {imported_pkg} "
                    f"(allowed: {{{', '.join(sorted(allowed)) or 'none'}}})"
                )

    return violations


# One test function per checked package so pytest shows granular failures.

def test_ir_layer():
    """python/adc/ir must not import any other adc package."""
    v = _check_package_layer("ir")
    assert not v, "Layer violation in ir:\n" + "\n".join(f"  {x}" for x in v)


def test_model_layer():
    """python/adc/model may only import from adc.ir."""
    v = _check_package_layer("model")
    assert not v, "Layer violation in model:\n" + "\n".join(f"  {x}" for x in v)


def test_physics_layer():
    """python/adc/physics may only import from adc.{ir, model}."""
    v = _check_package_layer("physics")
    assert not v, "Layer violation in physics:\n" + "\n".join(f"  {x}" for x in v)


def test_time_layer():
    """python/adc/time may only import from adc.{ir, model}."""
    v = _check_package_layer("time")
    assert not v, "Layer violation in time:\n" + "\n".join(f"  {x}" for x in v)


def test_lib_layer():
    """python/adc/lib may only import from adc.{ir, model, time, physics}."""
    v = _check_package_layer("lib")
    assert not v, "Layer violation in lib:\n" + "\n".join(f"  {x}" for x in v)


def test_codegen_layer():
    """python/adc/codegen may only import from adc.{ir, model, time, physics, lib}."""
    v = _check_package_layer("codegen")
    assert not v, "Layer violation in codegen:\n" + "\n".join(f"  {x}" for x in v)


def test_runtime_layer():
    """python/adc/runtime must not import other adc packages (it talks to _adc directly)."""
    v = _check_package_layer("runtime")
    assert not v, "Layer violation in runtime:\n" + "\n".join(f"  {x}" for x in v)


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_ir_layer,
        test_model_layer,
        test_physics_layer,
        test_time_layer,
        test_lib_layer,
        test_codegen_layer,
        test_runtime_layer,
    ]
    for t in tests:
        try:
            t()
            print(f"OK  {t.__name__}")
        except AssertionError as exc:
            print(f"FAIL {t.__name__}: {exc}")

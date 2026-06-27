"""Spec 4: the intra-pops import graph is acyclic and respects the layering.

The seven sub-packages form a directed acyclic dependency stack:

    ir       imports nothing else in pops
    model    -> ir
    physics  -> ir, model
    time     -> ir, model
    lib      -> ir, model, time, physics
    codegen  -> ir, model, physics, time, lib   (lowering, no _pops at module scope)
    runtime  -> everything, and is the ONLY layer allowed to import _pops

This test builds the cross-layer edges from module-scope imports (``ast``,
``col_offset == 0``) between sub-packages and asserts (a) the graph has no cycle and
(b) every edge points to an allowed lower layer. The flat root files and
``pops/__init__.py`` (the runtime facade, which re-exports everything) are not layered
sub-packages and are excluded from the graph.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"

# Allowed downstream targets for each layer (what it MAY import within pops).
ALLOWED = {
    "ir": set(),
    "model": {"ir"},
    "physics": {"ir", "model"},
    "time": {"ir", "model"},
    "lib": {"ir", "model", "time", "physics"},
    "codegen": {"ir", "model", "physics", "time", "lib"},
    "runtime": {"ir", "model", "physics", "time", "lib", "codegen"},
}
LAYERS = set(ALLOWED)


def _layer_of(modname):
    """Return the sub-package layer for a dotted ``pops.<layer>...`` name, else None."""
    parts = modname.split(".")
    if len(parts) >= 2 and parts[0] == "pops" and parts[1] in LAYERS:
        return parts[1]
    return None


def _module_name(path):
    rel = path.relative_to(POPS.parent).with_suffix("")
    return ".".join(rel.parts)


def _intra_targets(tree):
    """Yield module-scope (col_offset==0) import targets that name some pops module."""
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if node.col_offset != 0:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pops" or alias.name.startswith("pops."):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and (
                node.module == "pops" or node.module.startswith("pops.")
            ):
                yield node.module


def _build_edges():
    """Return {src_layer: {(dst_layer, "src_module -> dst_target"), ...}}."""
    edges = {}
    for path in sorted(POPS.rglob("*.py")):
        src_layer = _layer_of(_module_name(path))
        if src_layer is None:
            continue  # root facade / flat files are not layered sub-packages.
        tree = ast.parse(path.read_text(), str(path))
        for target in _intra_targets(tree):
            dst_layer = _layer_of(target)
            if dst_layer is None or dst_layer == src_layer:
                continue
            why = "%s -> %s" % (_module_name(path), target)
            edges.setdefault(src_layer, set()).add((dst_layer, why))
    return edges


def test_layering_respected():
    edges = _build_edges()
    violations = []
    for src_layer, deps in edges.items():
        for dst_layer, why in sorted(deps):
            if dst_layer not in ALLOWED[src_layer]:
                violations.append("%s may not import %s (%s)" % (src_layer, dst_layer, why))
    assert not violations, "layering violations:\n  " + "\n  ".join(sorted(violations))


def test_graph_is_acyclic():
    edges = _build_edges()
    adjacency = {layer: {d for d, _ in deps} for layer, deps in edges.items()}

    # Iterative DFS with three-color marking; record the back-edge that closes a cycle.
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {layer: WHITE for layer in LAYERS}
    cycle_edge = []

    def visit(start):
        stack = [(start, iter(sorted(adjacency.get(start, ()))))]
        color[start] = GRAY
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if color[child] == GRAY:
                    cycle_edge.append("%s -> %s" % (node, child))
                    return True
                if color[child] == WHITE:
                    color[child] = GRAY
                    stack.append((child, iter(sorted(adjacency.get(child, ())))))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
        return False

    for layer in sorted(LAYERS):
        if color[layer] == WHITE and visit(layer):
            break
    assert not cycle_edge, "import cycle through edge(s): " + ", ".join(cycle_edge)

"""Configuration Sphinx pour la documentation Python de PoPS."""

from __future__ import annotations

import sys
from pathlib import Path

# Rendre le module compile `pops` importable s'il a ete construit
# (-DPOPS_BUILD_PYTHON=ON) : autodoc en a besoin pour reference/python-api.md. Non
# fatal s'il manque (les classes apparaissent alors sans signature). On n'ajoute QUE
# des dossiers de BUILD (qui contiennent l'extension `_pops*.so` a cote du paquet) :
# surtout PAS le `python/` source, dont le paquet pops/ n'a pas de _pops.so -> il
# masquerait le build et `import pops` echouerait sur `from ._pops import ...`.
_repo = Path(__file__).parent.parent.parent
for _cand in (_repo / "build-py" / "python", _repo / "build" / "python",
              _repo / "build-master" / "python"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

project = "PoPS"
author = "Romain Despoullains"
copyright = "2026, Romain Despoullains"
def _version_from_cmake(_path: Path) -> str | None:
    import re

    try:
        _txt = _path.read_text(encoding="utf-8")
    except OSError:
        return None
    _m = re.search(r"project\s*\(\s*PoPS\b.*?VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", _txt, re.S)
    return _m.group(1) if _m else None


# Single source of the version: project(VERSION) in CMakeLists.txt (see docs/VERSIONING.md).
release = _version_from_cmake(_repo / "CMakeLists.txt") or "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# API de reference CUREE : on ne documente que les membres POSSEDANT une docstring
# (pas d'undoc-members) pour eviter de deverser la surface interne et ses warnings.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Reference C++ embarquee (doxysphinx) : scripts/build_docs.sh (mode complet) genere
# docs/sphinx/doxygen/ (doxygen puis doxysphinx, repertoire gitignore) AVANT le build
# sphinx ; les pages .rst generees sont toutes :orphan: et l'entree de navigation vit
# dans index.md (toctree "doxygen/index"). En mode rapide (--sphinx ou sphinx-build
# direct) le repertoire peut etre absent : on exclut alors le sous-arbre et on supprime
# UNIQUEMENT les warnings de toctree vers ce document absent/exclu, pour que -W reste
# vert dans les deux modes.
# Autodoc renders the codebase's Doxygen-style docstrings (@param / @p, hanging-indent
# continuations) as reStructuredText; docutils then emits benign "unexpected indentation" /
# "block quote ends without a blank line" messages that -W (warnings = errors) would turn into
# a build failure. Suppress the docutils category in BOTH modes (the meaningful -W checks --
# broken cross-references, missing toctree entries -- stay strict). See ADC-272.
suppress_warnings = ["docutils"]
if not (Path(__file__).parent / "doxygen" / "index.rst").is_file():
    exclude_patterns.append("doxygen")
    suppress_warnings += ["toc.excluded", "toc.not_readable"]

html_theme = "furo"
html_title = f"PoPS {release}"
html_theme_options = {
    "source_repository": "https://github.com/wolf75222/adc_cpp",
    "source_branch": "master",
    "source_directory": "docs/sphinx/",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "dollarmath",
]

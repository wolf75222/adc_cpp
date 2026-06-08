"""Configuration Sphinx pour la documentation Python d'adc_cpp."""

from __future__ import annotations

import sys
from pathlib import Path

# Rendre le module compile `adc` importable s'il a ete construit
# (-DADC_BUILD_PYTHON=ON) : autodoc en a besoin pour reference/api_python.md. Non
# fatal s'il manque (les classes apparaissent alors sans signature). On n'ajoute QUE
# des dossiers de BUILD (qui contiennent l'extension `_adc*.so` a cote du paquet) :
# surtout PAS le `python/` source, dont le paquet adc/ n'a pas de _adc.so -> il
# masquerait le build et `import adc` echouerait sur `from ._adc import ...`.
_repo = Path(__file__).parent.parent.parent
for _cand in (_repo / "build-py" / "python", _repo / "build" / "python",
              _repo / "build-master" / "python"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

project = "adc_cpp"
author = "Romain Despoullains"
copyright = "2026, Romain Despoullains"
release = "0.1.0"
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

html_theme = "furo"
html_title = f"adc_cpp {release}"
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

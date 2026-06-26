"""Robust handle on the :mod:`pops.model` package for the authoring layer.

``pops.physics`` lowers a model to the operator-first typed view
(:class:`pops.model.Module` and friends). Most of the time a plain
``from pops import model`` resolves it. But some tests load an authoring module
standalone (``spec_from_file_location`` with no parent package, e.g.
``test_projection_eig``); the relative import then has no parent. We fall back to
loading the sibling ``model/`` PACKAGE by path -- it is stdlib-only, so this pulls
in neither ``pops`` nor ``_pops``. Mirrors the historical ``dsl.py`` fallback.
"""
import os

try:
    from pops import model as model  # noqa: F401  -- normal package path
except ImportError:  # pragma: no cover - exercised only by the standalone-import path
    import importlib.util as _ilu
    import sys

    _mdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    _mspec = _ilu.spec_from_file_location(
        "pops_model", os.path.join(_mdir, "__init__.py"),
        submodule_search_locations=[_mdir])
    model = _ilu.module_from_spec(_mspec)
    sys.modules["pops_model"] = model  # so the package's relative imports resolve
    _mspec.loader.exec_module(model)

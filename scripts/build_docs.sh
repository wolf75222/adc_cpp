#!/usr/bin/env bash
# Construit TOUTE la doc (lint + Sphinx + Doxygen + site combine) en une commande.
# Source unique des etapes : ce script est utilise tel quel par la CI (docs.yml) ET en local.
#
#   bash scripts/build_docs.sh            # tout : lint, module, sphinx, doxygen, _site
#   bash scripts/build_docs.sh --sphinx   # lint + sphinx seulement (iteration rapide)
#
# Dependances : sphinx/furo/myst-parser + numpy (docs/sphinx/requirements.txt), doxygen,
# graphviz. Sous l'env conda du depot :
#   conda install -c conda-forge doxygen graphviz && pip install -r docs/sphinx/requirements.txt
# Sortie : docs/_site (Sphinx a la racine, Doxygen sous /cpp/), comme le site publie.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
ONLY_SPHINX="${1:-}"

missing=()
python -c "import sphinx" 2>/dev/null || missing+=("sphinx (pip install -r docs/sphinx/requirements.txt)")
python -c "import numpy" 2>/dev/null || missing+=("numpy (pip/conda)")
if [ "$ONLY_SPHINX" != "--sphinx" ]; then
  command -v doxygen >/dev/null || missing+=("doxygen (conda install -c conda-forge doxygen)")
  command -v dot >/dev/null || missing+=("graphviz (conda install -c conda-forge graphviz)")
fi
if [ ${#missing[@]} -gt 0 ]; then
  printf 'Dependance manquante : %s\n' "${missing[@]}" >&2
  exit 1
fi

echo "--- 1/4 lint documentaire (docs/check_docs.py) ---"
python docs/check_docs.py

# Le module _adc doit etre importable par autodoc (sphinx -W echoue sinon). On reutilise un
# build existant, sinon on construit le minimum (preset python = build-py).
echo "--- 2/4 module Python _adc (autodoc) ---"
PYMOD=""
for d in build-py build-py-kokkos build; do
  if ls "$d"/python/adc/_adc.*.so >/dev/null 2>&1; then PYMOD="$PWD/$d/python"; break; fi
done
if [ -z "$PYMOD" ]; then
  # preset `python` si l'env conda est actif (pin interpreteur de l'env), sinon `ci-python`
  # (sans condition conda : l'interpreteur vient du PATH -- en CI, celui de setup-python).
  PRESET=python
  [ -z "${CONDA_PREFIX:-}" ] && PRESET=ci-python
  echo "    (aucun build existant : cmake --preset $PRESET)"
  cmake --preset "$PRESET"
  cmake --build --preset "$PRESET"
  PYMOD="$PWD/build-py/python"
fi
echo "    module : $PYMOD"

echo "--- 3/4 Sphinx (-W : warnings = erreurs) ---"
PYTHONPATH="$PYMOD" python -m sphinx -W --keep-going -b html docs/sphinx docs/_build/sphinx

if [ "$ONLY_SPHINX" = "--sphinx" ]; then
  echo "OK (sphinx seul) : docs/_build/sphinx/index.html"
  exit 0
fi

echo "--- 4/4 Doxygen + site combine ---"
mkdir -p docs/_build
doxygen docs/Doxyfile
# Doxygen rend le README avec <img src="docs/..."> mais ne copie pas les images.
mkdir -p docs/_build/doxygen/html/docs
cp docs/*.png docs/*.gif docs/_build/doxygen/html/docs/ 2>/dev/null || true
mkdir -p docs/_site
cp -r docs/_build/sphinx/. docs/_site/
mkdir -p docs/_site/cpp
cp -r docs/_build/doxygen/html/. docs/_site/cpp/

echo "OK : docs/_site/index.html (Sphinx) et docs/_site/cpp/ (Doxygen)"

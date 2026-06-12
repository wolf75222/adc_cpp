#!/usr/bin/env bash
# Construit TOUTE la doc (lint + Sphinx + Doxygen + site combine) en une commande.
# Source unique des etapes : ce script est utilise tel quel par la CI (docs.yml) ET en local.
#
#   bash scripts/build_docs.sh            # tout : lint, module, doxysphinx, sphinx, doxygen, _site
#   bash scripts/build_docs.sh --sphinx   # lint + sphinx seulement (iteration rapide)
#
# Reference C++ embarquee (doxysphinx) : le mode complet regenere docs/sphinx/doxygen/
# (doxygen avec surcharges stdin, puis doxysphinx) AVANT le build sphinx ; les pages
# Doxygen sont alors servies DANS le site Sphinx sous /doxygen/, en plus du site Doxygen
# brut inchange sous /cpp/. Le mode --sphinx ne lance NI doxygen NI doxysphinx : conf.py
# exclut proprement le sous-arbre docs/sphinx/doxygen/ quand il est absent (sphinx -W
# reste vert) et le reutilise tel quel s'il reste d'un build complet precedent.
#
# Dependances : sphinx/furo/myst-parser + numpy + doxysphinx (docs/sphinx/requirements.txt),
# doxygen, graphviz. Sous l'env conda du depot :
#   conda install -c conda-forge doxygen graphviz && pip install -r docs/sphinx/requirements.txt
# Sortie : docs/_site (Sphinx a la racine, ref C++ embarquee sous /doxygen/, Doxygen brut
# sous /cpp/), comme le site publie.
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
  command -v doxysphinx >/dev/null || missing+=("doxysphinx (pip install -r docs/sphinx/requirements.txt)")
fi
if [ ${#missing[@]} -gt 0 ]; then
  printf 'Dependance manquante : %s\n' "${missing[@]}" >&2
  exit 1
fi

echo "--- 1/5 lint documentaire (docs/check_docs.py) ---"
python docs/check_docs.py

# Le module _adc doit etre importable par autodoc (sphinx -W echoue sinon). On reutilise un
# build existant, sinon on construit le minimum (preset python = build-py).
echo "--- 2/5 module Python _adc (autodoc) ---"
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

if [ "$ONLY_SPHINX" != "--sphinx" ]; then
  echo "--- 3/5 reference C++ embarquee (doxygen + doxysphinx) ---"
  # Un SEUL Doxyfile : la variante embarquee surcharge par stdin les cles exigees par
  # doxysphinx (sortie sous la racine sphinx, pas de treeview, pas de sous-repertoires ;
  # la recherche est celle de Sphinx). Regeneration complete a chaque build (le
  # repertoire est gitignore) pour ne pas trainer de pages orphelines perimees.
  rm -rf docs/sphinx/doxygen
  ( cat docs/Doxyfile
    echo "OUTPUT_DIRECTORY=docs/sphinx"
    echo "HTML_OUTPUT=doxygen"
    echo "GENERATE_TREEVIEW=NO"
    echo "CREATE_SUBDIRS=NO"
    echo "SEARCHENGINE=NO"
    echo "QUIET=YES"
  ) | doxygen -
  # doxysphinx genere un .rst par page html (tous :orphan:) sous docs/sphinx/doxygen/ et
  # copie les ressources (css/js/images) vers la sortie sphinx. Doit preceder sphinx.
  doxysphinx build docs/sphinx docs/_build/sphinx docs/sphinx/doxygen
  # Meme rustine d'images que le site /cpp/ : le README embarque reference docs/*.png.
  mkdir -p docs/_build/sphinx/doxygen/docs
  cp docs/*.png docs/*.gif docs/_build/sphinx/doxygen/docs/ 2>/dev/null || true
fi

echo "--- 4/5 Sphinx (-W : warnings = erreurs) ---"
PYTHONPATH="$PYMOD" python -m sphinx -W --keep-going -b html docs/sphinx docs/_build/sphinx

if [ "$ONLY_SPHINX" = "--sphinx" ]; then
  echo "OK (sphinx seul) : docs/_build/sphinx/index.html"
  exit 0
fi

echo "--- 5/5 Doxygen + site combine ---"
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

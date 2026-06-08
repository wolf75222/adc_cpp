#!/usr/bin/env python3
"""Doc-lint adc_cpp : garde-fous executes en CI (docs.yml), AVANT sphinx-build -W.

Echoue (exit 1) sur :
  1. em-dash U+2014 dans n'importe quel .md de docs/ (convention adc_cpp = ASCII, jamais d'em-dash) ;
  2. caracteres non-ASCII dans les pages Sphinx user-facing docs/sphinx/**.md (ASCII strict) ;
  3. termes interdits / anciennes API confirmees fausses, UNIQUEMENT dans les docs NORMATIVES
     (guide Sphinx + README + docs canoniques) -- PAS dans les docs d'audit/historique/design qui
     DISCUTENT legitimement ces termes (ex. DOC_REFONTE_AUDIT.md liste les claims faux) ;
  4. image referencee (![](...) / <img src=...>) introuvable sur le disque (lien casse).

Lancement : python3 docs/check_docs.py   (depuis la racine). 0 = OK, 1 = violations.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EM_DASH = "—"

# Docs NORMATIVES (doivent etre exactes pour l'utilisateur) : les checks de termes interdits ne
# s'appliquent qu'ici. Les pages Sphinx (docs/sphinx/**.md) sont toujours normatives.
NORMATIVE_ROOT = {
    "README.md", "ARCHITECTURE.md", "ALGORITHMS.md", "BACKEND_COVERAGE.md",
    "DSL_API.md", "DSL_MODEL_DESIGN.md", "CHOICES.md", "CONSERVATION_SUMMARY.md",
}

# (motif regex, message). HIGH-CONFIDENCE : choses UNAMBIGUMENT fausses. Verifie sur une seule ligne.
FORBIDDEN: list[tuple[str, str]] = [
    (r"ADC_USE_EIGEN", "option CMake fantome (pas d'option(), pas de cible adc_eigen)"),
    (r"--target\s+adc_py", "mauvaise cible : la cible pybind est `_adc` (python/CMakeLists.txt)"),
    (r"limiter\s*=\s*['\"]mc['\"]", "limiteur 'mc' inexistant (NoSlope/Minmod/VanLeer/Weno5)"),
    (r"monotonized[- ]central", "limiteur MC / monotonized-central inexistant"),
    (r"AmrSystem[^\n]{0,40}mono-bloc", "fausse limitation : AmrSystem est mono- ET multi-bloc"),
    (r"mono-bloc[^\n]{0,40}AmrSystem", "fausse limitation : AmrSystem est mono- ET multi-bloc"),
]


def md_files() -> list[pathlib.Path]:
    return sorted(set(DOCS.glob("*.md")) | set(DOCS.glob("sphinx/**/*.md")))


def is_sphinx_page(p: pathlib.Path) -> bool:
    return "sphinx" in p.relative_to(ROOT).parts


def is_normative(p: pathlib.Path) -> bool:
    if is_sphinx_page(p):
        return True
    rel = p.relative_to(ROOT)
    return str(rel) == "README.md" or (rel.parent == DOCS.relative_to(ROOT) and p.name in NORMATIVE_ROOT)


def check() -> int:
    # README.md vit a la racine, pas sous docs/ : l'ajouter au scan.
    files = md_files()
    readme = ROOT / "README.md"
    if readme.exists():
        files = sorted(set(files) | {readme})
    violations: list[str] = []
    img_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)[\"']")
    for p in files:
        rel = p.relative_to(ROOT)
        text = p.read_text(encoding="utf-8")
        if EM_DASH in text:
            violations.append(f"{rel}: {text.count(EM_DASH)} em-dash (U+2014) interdits (utiliser '--')")
        if is_sphinx_page(p):
            bad = sorted({c for c in text if ord(c) > 127})
            if bad:
                show = " ".join(f"U+{ord(c):04X}({c})" for c in bad[:10])
                violations.append(f"{rel}: non-ASCII dans une page Sphinx (ASCII strict) : {show}")
        if is_normative(p):
            for pat, msg in FORBIDDEN:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    line = text[: m.start()].count("\n") + 1
                    violations.append(f"{rel}:{line}: terme interdit '{m.group(0)[:36]}' -- {msg}")
        for m in img_re.finditer(text):
            target = (m.group(1) or m.group(2) or "").split()[0].split("#")[0]
            if (not target or target.startswith(("http://", "https://", "data:")) or "..." in target):
                continue
            if not (p.parent / target).resolve().exists():
                line = text[: m.start()].count("\n") + 1
                violations.append(f"{rel}:{line}: image referencee introuvable : {target}")
    if violations:
        print(f"DOC-LINT : {len(violations)} violation(s)", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        return 1
    print(f"DOC-LINT : OK ({len(files)} fichiers .md verifies)")
    return 0


if __name__ == "__main__":
    sys.exit(check())

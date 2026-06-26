#!/usr/bin/env python3
"""Doc-lint adc_cpp : garde-fous executes en CI (docs.yml), AVANT sphinx-build -W.

Echoue (exit 1) sur :
  1. em-dash U+2014 dans n'importe quel .md de docs/ (convention adc_cpp = ASCII, jamais d'em-dash) ;
  2. caracteres non-ASCII dans les pages Sphinx user-facing docs/sphinx/**.md (ASCII strict) ;
  3. termes interdits / anciennes API confirmees fausses, UNIQUEMENT dans les docs NORMATIVES
     (guide Sphinx + README + docs canoniques) -- PAS dans les docs d'audit/historique/design qui
     DISCUTENT legitimement ces termes (ex. DOC_REFONTE_AUDIT.md liste les claims faux) ;
  4. image referencee (![](...) / <img src=...>) introuvable sur le disque (lien casse) ;
  5. lien markdown relatif vers un fichier introuvable sur le disque ;
  6. PRESENCE docmap : page Sphinx non mappee dans docs/docmap.toml, ou chemin depends_on / tested_by
     inexistant (anti-rot de la carte elle-meme) ;
  7. FRAICHEUR (git) : doc en mode strict dont un depends_on a ete commite apres sa relecture.

Avertissements (n'affectent pas l'exit code) : fraicheur en mode warning, et bloc python >= 10 lignes
avec `import adc` dans une page testable (ADC-155 : preferer un literalinclude).

Lancement : python3 docs/check_docs.py   (depuis la racine). 0 = OK, 1 = violations.
Options   : --freshness-warn-only (toute la fraicheur devient avertissement), --selftest (auto-test).
"""
from __future__ import annotations

import argparse
import fnmatch
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 (non supporte ici, message clair)
    tomllib = None

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCMAP = DOCS / "docmap.toml"
EM_DASH = "—"

# Docs NORMATIVES (doivent etre exactes pour l'utilisateur) : les checks de termes interdits ne
# s'appliquent qu'ici. Les pages Sphinx (docs/sphinx/**.md) sont toujours normatives.
NORMATIVE_ROOT = {
    "README.md", "ARCHITECTURE.md", "ALGORITHMS.md", "BACKEND_COVERAGE.md",
    "DSL_API.md", "DSL_MODEL_DESIGN.md", "CHOICES.md", "CONSERVATION_SUMMARY.md",
}

# (motif regex, message). HIGH-CONFIDENCE : choses UNAMBIGUMENT fausses. Verifie sur une seule ligne.
FORBIDDEN: list[tuple[str, str]] = [
    (r"POPS_USE_EIGEN", "option CMake fantome (pas d'option(), pas de cible pops_eigen)"),
    (r"--target\s+pops_py", "mauvaise cible : la cible pybind est `_pops` (python/CMakeLists.txt)"),
    (r"limiter\s*=\s*['\"]mc['\"]", "limiteur 'mc' inexistant (NoSlope/Minmod/VanLeer/Weno5)"),
    (r"monotonized[- ]central", "limiteur MC / monotonized-central inexistant"),
    (r"AmrSystem[^\n]{0,40}mono-bloc", "fausse limitation : AmrSystem est mono- ET multi-bloc"),
    (r"mono-bloc[^\n]{0,40}AmrSystem", "fausse limitation : AmrSystem est mono- ET multi-bloc"),
    # ADC-255 : garde-fous de regression sur les claims re-verifies contre le code puis corriges.
    (r"regrid_every > 0` is refused", "fausse limitation : multi-blocs + regrid_every > 0 est SUPPORTE (capstone Phase 2)"),
    (r"reject(ed)? on the Python AMR facade", "fausse limitation : HLLC/Roe/primitive sont CABLES sur la facade AMR (garde pression p)"),
    (r"add_equation[^\n]{0,40}(reject|rejette)[^\n]{0,30}(Split|Strang)", "fausse limitation : AmrSystem.add_equation ACCEPTE Split/Strang (Schur AMR mono-bloc)"),
    # ADC-298/293 : HLLC/Roe ne sont PAS Euler-only ; chemin generique via HasHLLCStructure /
    # HasRoeDissipation (numerical_flux.hpp), Euler 2D n'est que le fallback sans hooks.
    (r"(2D Euler|Euler[ -]?2D) only", "fausse limitation : HLLC/Roe sont generiques via HasHLLCStructure / HasRoeDissipation ; Euler 2D n'est que le fallback (numerical_flux.hpp)"),
    # ADC-298 : le transport polaire couvre ExB ET le fluide isotherme (IsothermalFluxPolar, riemann hll).
    (r"scalar `?ExB`?( transport)? only", "fausse limitation : le transport polaire couvre scalar ExB ET le fluide isotherme (IsothermalFluxPolar, riemann hll), pas ExB only"),
]


def md_files(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    docs = root / "docs"
    files = sorted(set(docs.glob("*.md")) | set(docs.glob("sphinx/**/*.md")))
    readme = root / "README.md"
    if readme.exists():
        files = sorted(set(files) | {readme})
    return files


def is_sphinx_page(p: pathlib.Path, root: pathlib.Path = ROOT) -> bool:
    return "sphinx" in p.relative_to(root).parts


def is_normative(p: pathlib.Path, root: pathlib.Path = ROOT) -> bool:
    if is_sphinx_page(p, root):
        return True
    rel = p.relative_to(root)
    return str(rel) == "README.md" or (rel.parent == pathlib.Path("docs") and p.name in NORMATIVE_ROOT)


def mask_code(text: str) -> str:
    """Remplace les blocs/spans de code par des blancs de MEME longueur (newlines preservees) :
    les liens markdown a l'interieur d'un code ne sont pas des liens, et le masquage garde les
    offsets intacts pour des numeros de ligne corrects."""
    def blank(m: re.Match) -> str:
        return "".join(c if c == "\n" else " " for c in m.group(0))
    text = re.sub(r"```.*?```", blank, text, flags=re.DOTALL)
    text = re.sub(r"~~~.*?~~~", blank, text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]*`", blank, text)
    return text


# --- docmap + git helpers --------------------------------------------------------------------------

def load_docmap(path: pathlib.Path) -> dict:
    if tomllib is None:
        raise RuntimeError("tomllib indisponible (Python >= 3.11 requis pour docs/check_docs.py)")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _git(args: list[str], root: pathlib.Path) -> str | None:
    """Renvoie la sortie git (str, possiblement vide) ou None si git echoue / absent."""
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    except (OSError, ValueError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state(root: pathlib.Path) -> str:
    res = _git(["rev-parse", "--is-shallow-repository"], root)
    if res is None:
        return "nogit"
    return "shallow" if res == "true" else "ok"


def last_commit(doc_rel: str, root: pathlib.Path) -> str | None:
    return _git(["log", "-1", "--format=%H", "--", doc_rel], root)


def commits_touching(ref: str, deps: list[str], root: pathlib.Path) -> list[str] | None:
    # --no-merges : un merge ne modifie pas un depends_on, il propage du contenu deja committe
    # (les commits porteurs sont detectes par ailleurs) ; sans ce filtre, la simplification
    # d'historique de rev-list peut remonter des commits de merge en faux positifs, notamment
    # quand `ref` acquitte un commit d'une branche soeur pas encore fusionnee.
    res = _git(["rev-list", "--no-merges", f"{ref}..HEAD", "--", *deps], root)
    if res is None:
        return None
    return [c for c in res.splitlines() if c]


def excluded(rel_sphinx: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel_sphinx, pat) for pat in patterns)


# --- checks ----------------------------------------------------------------------------------------

def check_presence(data: dict, root: pathlib.Path, violations: list[str]) -> None:
    docs_map = data.get("docs", {})
    patterns = data.get("exclude", {}).get("patterns", [])
    sphinx_root = root / "docs" / "sphinx"
    for p in sorted(sphinx_root.glob("**/*.md")):
        rel_repo = str(p.relative_to(root))
        rel_sphinx = str(p.relative_to(sphinx_root))
        if excluded(rel_sphinx, patterns):
            continue
        if rel_repo not in docs_map:
            violations.append(f"{rel_repo}: page Sphinx non mappee dans docs/docmap.toml "
                              f"(ajouter une entree ou l'exclure)")
    for doc, meta in docs_map.items():
        for kind in ("depends_on", "tested_by"):
            for dep in meta.get(kind, []) or []:
                if not (root / dep).exists():
                    violations.append(f"{doc}: {kind} introuvable sur le disque : {dep}")


def check_freshness(data: dict, root: pathlib.Path, violations: list[str],
                    warnings: list[str], notices: list[str], warn_only: bool) -> None:
    state = git_state(root)
    if state == "nogit":
        notices.append("fraicheur ignoree : pas de depot git accessible (presence maintenue)")
        return
    if state == "shallow":
        notices.append("fraicheur ignoree : depot git shallow (presence maintenue)")
        return
    for doc, meta in data.get("docs", {}).items():
        deps = meta.get("depends_on") or []
        if not deps:
            continue
        reviewed = meta.get("reviewed")
        if reviewed:
            ref = reviewed
        else:
            ref = last_commit(doc, root)
            if not ref:
                notices.append(f"{doc}: jamais commite, fraicheur ignoree")
                continue
        bad = commits_touching(ref, deps, root)
        if bad is None:
            # ref ne resout pas. Un reviewed explicite qui pointe hors historique (sha d'une
            # branche soeur non fusionnee, ou ramasse par gc) desactiverait SILENCIEUSEMENT la
            # fraicheur, y compris en mode strict sur les pages porteuses. On le rend visible et
            # fail-closed : meme escalade qu'un doc suspect, pour qu'un acquittement casse ne
            # puisse jamais neutraliser la garantie sans bruit.
            if reviewed:
                msg = (f"{doc}: reviewed '{ref[:12]}' ne resout pas (hors historique) ; "
                       f"acquittement invalide, fraicheur non verifiee -- corriger reviewed "
                       f"avec un sha de master dans docs/docmap.toml")
                if warn_only or meta.get("mode") == "warning":
                    warnings.append(msg)
                else:
                    violations.append(msg)
            else:
                notices.append(f"{doc}: ref de fraicheur '{ref[:12]}' inconnue, fraicheur ignoree")
            continue
        if not bad:
            continue
        short = ", ".join(c[:12] for c in bad[:3])
        more = "" if len(bad) <= 3 else f" (+{len(bad) - 3} autre(s))"
        msg = (f"{doc}: doc suspect -- un depends_on a ete commite depuis la relecture "
               f"({short}{more}) ; relire le doc puis le committer, ou acquitter via "
               f"reviewed = <sha> dans docs/docmap.toml")
        if warn_only or meta.get("mode") == "warning":
            warnings.append(msg)
        else:
            violations.append(msg)


LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")
PY_BLOCK_RE = re.compile(r"```[ \t]*python\b[^\n]*\n(.*?)```", re.DOTALL)


def check_links_and_inline(p: pathlib.Path, text: str, rel: str, testable: bool,
                           violations: list[str], warnings: list[str]) -> None:
    masked = mask_code(text)
    for m in LINK_RE.finditer(masked):
        raw = m.group(1).strip()
        if raw.startswith("<") and ">" in raw:
            raw = raw[1:raw.index(">")]
        parts = raw.split()
        target = parts[0] if parts else ""
        path = target.split("#")[0]
        if (not path or target.startswith(("http://", "https://", "mailto:", "data:", "#"))):
            continue
        # URL externe sans schema explicite : tout autre schema (ftp://, ssh://...) ou cible de forme
        # hote (www.x, domaine.tld/...) n'est PAS un chemin de fichier du depot. Sans ce garde-fou un
        # futur lien externe ecrit sans http(s):// casserait la lane PR (faux "fichier introuvable").
        first = path.split("/", 1)[0]
        looks_host = "/" in path and "." in first and first.rsplit(".", 1)[-1].isalpha()
        if "://" in target or first.startswith("www.") or looks_host:
            continue
        # ne traiter que ce qui ressemble a un chemin de fichier (extension ou separateur),
        # pour ne pas confondre une notation type out[l-1](c) avec un lien.
        if "/" not in path and "." not in pathlib.PurePosixPath(path).name:
            continue
        if not (p.parent / path).exists():
            line = masked[: m.start()].count("\n") + 1
            violations.append(f"{rel}:{line}: lien relatif introuvable : {target}")
    if testable:
        for m in PY_BLOCK_RE.finditer(text):
            body = m.group(1)
            nlines = len(body.splitlines())
            if nlines >= 10 and "import adc" in body:
                line = text[: m.start()].count("\n") + 1
                warnings.append(f"{rel}:{line}: bloc python de {nlines} lignes avec 'import adc' "
                                f"dans une page testable -- preferer un literalinclude (ADC-155)")


def check(freshness_warn_only: bool = False, root: pathlib.Path = ROOT) -> int:
    files = md_files(root)
    violations: list[str] = []
    warnings: list[str] = []
    notices: list[str] = []

    data: dict = {}
    docs_map: dict = {}
    mappath = root / "docs" / "docmap.toml"
    if mappath.exists():
        data = load_docmap(mappath)
        docs_map = data.get("docs", {})
    else:
        violations.append("docs/docmap.toml manquant (carte de documentation requise)")

    # ADC-311 : les patterns [exclude] du docmap (_generated/**, doxygen/**, ...) sont honores AUSSI
    # par les lints de CONTENU (em-dash / non-ASCII / termes interdits / liens / images), pas seulement
    # par le check de presence -- sinon une page generee (gitignoree) presente sur le disque au moment du
    # build (docs.yml genere docs/sphinx/doxygen/ avant sphinx) serait lintee a tort et casserait le
    # deploy sous set -e.
    exclude_patterns = data.get("exclude", {}).get("patterns", [])
    sphinx_root = root / "docs" / "sphinx"

    img_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)[\"']")
    for p in files:
        rel = str(p.relative_to(root))
        if is_sphinx_page(p, root) and excluded(str(p.relative_to(sphinx_root)), exclude_patterns):
            continue
        text = p.read_text(encoding="utf-8")
        if EM_DASH in text:
            violations.append(f"{rel}: {text.count(EM_DASH)} em-dash (U+2014) interdits (utiliser '--')")
        if is_sphinx_page(p, root):
            bad = sorted({c for c in text if ord(c) > 127})
            if bad:
                show = " ".join(f"U+{ord(c):04X}({c})" for c in bad[:10])
                violations.append(f"{rel}: non-ASCII dans une page Sphinx (ASCII strict) : {show}")
        if is_normative(p, root):
            for pat, msg in FORBIDDEN:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    line = text[: m.start()].count("\n") + 1
                    violations.append(f"{rel}:{line}: terme interdit '{m.group(0)[:36]}' -- {msg}")
        for m in img_re.finditer(text):
            tgt = (m.group(1) or m.group(2) or "").split()[0].split("#")[0]
            if (not tgt or tgt.startswith(("http://", "https://", "data:")) or "..." in tgt):
                continue
            if not (p.parent / tgt).resolve().exists():
                line = text[: m.start()].count("\n") + 1
                violations.append(f"{rel}:{line}: image referencee introuvable : {tgt}")
        meta = docs_map.get(rel, {})
        check_links_and_inline(p, text, rel, bool(meta.get("testable")), violations, warnings)

    if data:
        check_presence(data, root, violations)
        check_freshness(data, root, violations, warnings, notices, freshness_warn_only)

    for n in notices:
        print(f"DOC-LINT (info) : {n}")
    if warnings:
        print(f"DOC-LINT : {len(warnings)} avertissement(s)", file=sys.stderr)
        for w in warnings:
            print("  " + w, file=sys.stderr)
    if violations:
        print(f"DOC-LINT : {len(violations)} violation(s)", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        return 1
    print(f"DOC-LINT : OK ({len(files)} fichiers .md verifies, {len(docs_map)} entrees docmap)")
    return 0


# --- selftest --------------------------------------------------------------------------------------

def _git_run(root: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@test",
                    "-C", str(root), *args], check=True, capture_output=True, text=True)


def _write(p: pathlib.Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _selftest_case(name: str, build, flags: list[str], expect_exit: int) -> bool:
    with tempfile.TemporaryDirectory(prefix="docmap-selftest-") as td:
        root = pathlib.Path(td)
        (root / "docs" / "sphinx").mkdir(parents=True)
        shutil.copy(__file__, root / "docs" / "check_docs.py")
        build(root)
        proc = subprocess.run([sys.executable, str(root / "docs" / "check_docs.py"), *flags],
                              capture_output=True, text=True)
        ok = proc.returncode == expect_exit
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} : exit {proc.returncode} (attendu {expect_exit})")
        if not ok:
            sys.stderr.write(proc.stdout + proc.stderr)
        return ok


PAGE = "# Page\n\nContenu ASCII propre.\n"


def selftest() -> int:
    results: list[bool] = []

    # cas 1 : depends_on inexistant -> violation de presence -> exit 1 (pas besoin de git).
    def build_missing_dep(root: pathlib.Path) -> None:
        _write(root / "docs" / "sphinx" / "p.md", PAGE)
        _write(root / "docs" / "docmap.toml",
               '[exclude]\npatterns = []\n\n'
               '[docs."docs/sphinx/p.md"]\nowner = "t"\nmode = "warning"\n'
               'depends_on = ["src/absent.txt"]\ntested_by = []\ntestable = false\n')
    results.append(_selftest_case("depends_on inexistant -> exit 1", build_missing_dep, [], 1))

    # cas 2 : page Sphinx non mappee -> exit 1.
    def build_unmapped(root: pathlib.Path) -> None:
        _write(root / "docs" / "sphinx" / "p.md", PAGE)
        _write(root / "docs" / "sphinx" / "q.md", PAGE)   # absente du docmap
        _write(root / "docs" / "docmap.toml",
               '[exclude]\npatterns = []\n\n'
               '[docs."docs/sphinx/p.md"]\nowner = "t"\nmode = "warning"\n'
               'depends_on = []\ntested_by = []\ntestable = false\n')
    results.append(_selftest_case("page Sphinx non mappee -> exit 1", build_unmapped, [], 1))

    def _stale_repo(mode: str):
        def build(root: pathlib.Path) -> None:
            _write(root / "src" / "dep.txt", "v1\n")
            _write(root / "docs" / "sphinx" / "p.md", PAGE)
            _write(root / "docs" / "docmap.toml",
                   '[exclude]\npatterns = []\n\n'
                   '[docs."docs/sphinx/p.md"]\nowner = "t"\nmode = "' + mode + '"\n'
                   'depends_on = ["src/dep.txt"]\ntested_by = []\ntestable = false\n')
            _git_run(root, "init", "-q")
            _git_run(root, "add", "src/dep.txt", "docs/sphinx/p.md", "docs/docmap.toml")
            _git_run(root, "commit", "-q", "-m", "doc + dep")
            _write(root / "src" / "dep.txt", "v2\n")        # dep modifie APRES la relecture du doc
            _git_run(root, "add", "src/dep.txt")
            _git_run(root, "commit", "-q", "-m", "touche le dep")
        return build

    # cas 3 : stale en mode strict -> exit 1.
    results.append(_selftest_case("stale strict -> exit 1", _stale_repo("strict"), [], 1))
    # cas 4 : stale en mode warning -> exit 0 (avertissement seulement).
    results.append(_selftest_case("stale warning -> exit 0", _stale_repo("warning"), [], 0))
    # cas 5 : stale strict + --freshness-warn-only -> exit 0.
    results.append(_selftest_case("stale strict + --freshness-warn-only -> exit 0",
                                  _stale_repo("strict"), ["--freshness-warn-only"], 0))

    # cas 6 (ADC-311) : page Sphinx GENEREE sous un pattern [exclude], avec du non-ASCII -> NON
    # lintee (exit 0). Sans le honoring des patterns par les lints de contenu, le non-ASCII de
    # _generated/g.md ferait echouer (exit 1) : c'est exactement la classe de violations qui cassait
    # le deploy sur du cruft genere.
    def build_excluded_generated(root: pathlib.Path) -> None:
        _write(root / "docs" / "sphinx" / "p.md", PAGE)
        _write(root / "docs" / "sphinx" / "_generated" / "g.md",
               "# G\n\ncafe \u00e9\u00e8 non-ASCII genere.\n")
        _write(root / "docs" / "docmap.toml",
               '[exclude]\npatterns = ["_generated/**"]\n\n'
               '[docs."docs/sphinx/p.md"]\nowner = "t"\nmode = "warning"\n'
               'depends_on = []\ntested_by = []\ntestable = false\n')
    results.append(_selftest_case("page generee exclue non content-lintee -> exit 0",
                                  build_excluded_generated, [], 0))

    passed = sum(results)
    print(f"SELFTEST : {passed}/{len(results)} PASS")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Doc-lint adc_cpp (presence/fraicheur/liens docmap).")
    ap.add_argument("--freshness-warn-only", action="store_true",
                    help="toute la fraicheur devient avertissement (presence et lints restent durs)")
    ap.add_argument("--selftest", action="store_true",
                    help="auto-test sur des depots git jetables (PASS/FAIL par cas)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    sys.exit(check(freshness_warn_only=args.freshness_warn_only))

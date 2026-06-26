#!/usr/bin/env python3
"""Rejoue les scripts d'exemple de la documentation (champ tested_by de docs/docmap.toml).

Pour chaque script unique reference par un tested_by, lance :

    <python> <script> --quick --outdir <tmp> avec MPLBACKEND=Agg, cwd = racine du depot

Le module `pops` est localise par la variable POPS_PYMOD si elle est posee, sinon par balayage des
emplacements de build usuels (build-py/python, build-py-kokkos/python, build/python,
build-master/python) relatifs a la racine. Le chemin trouve prefixe PYTHONPATH.

Critere de succes : exit 0 du script (aucune assertion numerique ici ; les scripts portent leurs
propres verifications). En cas d'echec, la sortie du script est dump, les suivants continuent, et
l'exit final est 1 si au moins un script a echoue.

Lancement : python3 docs/run_doc_examples.py [--timeout 600]   (depuis n'importe ou). 0 = OK, 1 = echec.
Stdlib uniquement.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import time

try:
    import tomllib
except ModuleNotFoundError:
    print("ERREUR : tomllib indisponible (Python >= 3.11 requis).", file=sys.stderr)
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCMAP = ROOT / "docs" / "docmap.toml"
MODULE_CANDIDATES = ("build-py/python", "build-py-kokkos/python", "build/python", "build-master/python")


def locate_module() -> pathlib.Path:
    """Renvoie le dossier contenant le paquet `pops` (a prefixer a PYTHONPATH)."""
    env = os.environ.get("POPS_PYMOD")
    if env:
        cand = pathlib.Path(env).resolve()
        if (cand / "pops" / "__init__.py").exists():
            return cand
        print(f"ERREUR : POPS_PYMOD={env} ne contient pas pops/__init__.py.", file=sys.stderr)
        sys.exit(1)
    for rel in MODULE_CANDIDATES:
        cand = (ROOT / rel).resolve()
        if (cand / "pops" / "__init__.py").exists():
            return cand
    print("ERREUR : module `pops` introuvable. Construire le module (preset python) ou poser "
          "POPS_PYMOD vers le dossier contenant pops/__init__.py.\n"
          f"  emplacements balayes : {', '.join(MODULE_CANDIDATES)} (relatifs a {ROOT}).",
          file=sys.stderr)
    sys.exit(1)


def collect_scripts() -> list[str]:
    if not DOCMAP.exists():
        print(f"ERREUR : {DOCMAP} introuvable.", file=sys.stderr)
        sys.exit(1)
    with DOCMAP.open("rb") as fh:
        data = tomllib.load(fh)
    scripts: list[str] = []
    for meta in data.get("docs", {}).values():
        for s in meta.get("tested_by", []) or []:
            if s not in scripts:
                scripts.append(s)
    return scripts


def run_one(script: str, module_dir: pathlib.Path, timeout: int) -> tuple[bool, float, str]:
    script_path = (ROOT / script).resolve()
    if not script_path.exists():
        return False, 0.0, f"script introuvable : {script_path}"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(module_dir) + (os.pathsep + existing if existing else "")
    with tempfile.TemporaryDirectory(prefix="adc-doc-example-") as td:
        cmd = [sys.executable, str(script_path), "--quick", "--outdir", td]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, time.time() - t0, f"timeout apres {timeout}s"
        dt = time.time() - t0
        if proc.returncode == 0:
            return True, dt, ""
        return False, dt, (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Rejoue les exemples de doc (tested_by du docmap).")
    ap.add_argument("--timeout", type=int, default=600, help="timeout par script en secondes (defaut 600)")
    args = ap.parse_args()

    scripts = collect_scripts()
    if not scripts:
        print("Aucun script tested_by dans docs/docmap.toml : rien a rejouer.")
        return 0

    module_dir = locate_module()
    print(f"module pops localise : {module_dir}")
    print(f"scripts a rejouer ({len(scripts)}) : {', '.join(scripts)}\n")

    results: list[tuple[str, bool, float]] = []
    for script in scripts:
        print(f"=== {script} (--quick) ===")
        ok, dt, output = run_one(script, module_dir, args.timeout)
        if ok:
            print(f"  OK ({dt:.1f}s)")
        else:
            print(f"  FAIL ({dt:.1f}s)")
            sys.stderr.write(output.rstrip() + "\n")
        results.append((script, ok, dt))

    print("\n--- resume ---")
    for script, ok, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'} {script} ({dt:.1f}s)")
    failures = [s for s, ok, _ in results if not ok]
    if failures:
        print(f"\n{len(failures)} echec(s) sur {len(results)}.", file=sys.stderr)
        return 1
    print(f"\n{len(results)} script(s) OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

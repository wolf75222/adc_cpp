#!/usr/bin/env python3
"""Compte les tests de la suite adc_cpp directement depuis les sources.

Source de verite pour les totaux affiches dans docs/BACKEND_COVERAGE.md.
Aucun nombre code en dur : on parse tests/CMakeLists.txt et on liste
python/tests/test_*.py. Lance-le apres tout ajout/retrait de test :

    python3 docs/gen_test_counts.py

Definitions (alignees sur la matrice BACKEND_COVERAGE.md) :
  - pops_add_test          : cibles ctest C++ hors-MPI declarees via la
                            fonction helper pops_add_test(<nom>).
  - add_executable runtime: cibles ctest C++ hors-MPI qui lient un runtime
                            (system.cpp / amr_system.cpp) via add_executable,
                            HORS le bloc if(POPS_HAS_MPI) et HORS la ligne
                            generique add_executable(${name} ...) des helpers.
  - MPI                   : binaires add_executable declares DANS le bloc
                            if(POPS_HAS_MPI) (hors ligne generique du helper) ;
                            chacun rejoue typiquement np=1/2/4.
  - Python                : fichiers python/tests/test_*.py.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKE = ROOT / "tests" / "CMakeLists.txt"
PYTEST_DIR = ROOT / "python" / "tests"
BACKEND_COVERAGE = ROOT / "docs" / "BACKEND_COVERAGE.md"

# Ligne generique des fonctions helper (pops_add_test / pops_add_mpi_test) :
# add_executable(${name} ${name}.cpp). Ce n'est PAS un test, a exclure.
GENERIC_ADD_EXEC = re.compile(r"add_executable\(\$\{name\}")
ADD_EXEC = re.compile(r"^\s*add_executable\(")
POPS_ADD_TEST = re.compile(r"\badc_add_test\(")
MPI_GUARD_OPEN = re.compile(r"if\(POPS_HAS_MPI\)")
ENDIF = re.compile(r"^endif\(\)")


def count() -> dict[str, int]:
    lines = CMAKE.read_text().splitlines()

    pops_add_test = 0
    add_exec_runtime = 0   # non-MPI add_executable, hors ligne generique
    add_exec_mpi = 0       # add_executable dans le bloc POPS_HAS_MPI, hors generique

    in_mpi = False
    for line in lines:
        if MPI_GUARD_OPEN.search(line):
            in_mpi = True
        elif ENDIF.match(line) and in_mpi:
            in_mpi = False

        if POPS_ADD_TEST.search(line):
            pops_add_test += 1

        if ADD_EXEC.match(line) and not GENERIC_ADD_EXEC.search(line):
            if in_mpi:
                add_exec_mpi += 1
            else:
                add_exec_runtime += 1

    python = len(list(PYTEST_DIR.glob("test_*.py")))

    cpp_non_mpi = pops_add_test + add_exec_runtime
    return {
        "pops_add_test": pops_add_test,
        "add_executable_runtime": add_exec_runtime,
        "cpp_non_mpi_total": cpp_non_mpi,
        "mpi_binaries": add_exec_mpi,
        "python": python,
    }


def _disk_test_stems() -> tuple[list[str], list[str]]:
    """Noms (stems) des tests sur disque : tests/test_*.cpp et python/tests/test_*.py
    (top-level ; les harnais python/tests/gpu/ sont la section 3 de la matrice, traites a part)."""
    cpp = sorted(p.stem for p in (ROOT / "tests").glob("test_*.cpp"))
    py = sorted(p.stem for p in PYTEST_DIR.glob("test_*.py"))
    return cpp, py


def check_matrix() -> int:
    """Liste les tests presents sur disque mais ABSENTS de docs/BACKEND_COVERAGE.md.

    La matrice se veut la source unique de couverture ; un test sans ligne y est un trou
    (et la derive principale qui la rend perimee). Renvoie le nombre de tests absents
    (0 = matrice complete)."""
    text = BACKEND_COVERAGE.read_text()
    cpp, py = _disk_test_stems()
    miss_cpp = [t for t in cpp if t not in text]
    miss_py = [t for t in py if t not in text]
    for t in miss_cpp:
        print(f"MISSING (C++)    : {t}")
    for t in miss_py:
        print(f"MISSING (Python) : {t}")
    n = len(miss_cpp) + len(miss_py)
    print(
        f"\nmatrix coverage : {len(cpp) - len(miss_cpp)}/{len(cpp)} C++ + "
        f"{len(py) - len(miss_py)}/{len(py)} Python listed ; "
        f"{n} test(s) absent(s) de docs/BACKEND_COVERAGE.md"
    )
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="adc_cpp : totaux de la suite de tests + verification de couverture de la matrice."
    )
    ap.add_argument(
        "--check-matrix",
        action="store_true",
        help="liste les tests disque absents de docs/BACKEND_COVERAGE.md ; exit 1 si au moins un manque",
    )
    args = ap.parse_args()

    if args.check_matrix:
        raise SystemExit(1 if check_matrix() else 0)

    c = count()
    print(f"pops_add_test (ctest C++ hors-MPI)       : {c['pops_add_test']}")
    print(f"add_executable runtime (hors-MPI)        : {c['add_executable_runtime']}")
    print(f"  -> total cibles ctest C++ hors-MPI     : {c['cpp_non_mpi_total']}")
    print(f"add_executable bloc POPS_HAS_MPI          : {c['mpi_binaries']}")
    print(f"tests Python (python/tests/test_*.py)    : {c['python']}")


if __name__ == "__main__":
    main()

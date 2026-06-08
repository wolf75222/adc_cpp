#!/usr/bin/env python3
"""Compte les tests de la suite adc_cpp directement depuis les sources.

Source de verite pour les totaux affiches dans docs/BACKEND_COVERAGE.md.
Aucun nombre code en dur : on parse tests/CMakeLists.txt et on liste
python/tests/test_*.py. Lance-le apres tout ajout/retrait de test :

    python3 docs/gen_test_counts.py

Definitions (alignees sur la matrice BACKEND_COVERAGE.md) :
  - adc_add_test          : cibles ctest C++ hors-MPI declarees via la
                            fonction helper adc_add_test(<nom>).
  - add_executable runtime: cibles ctest C++ hors-MPI qui lient un runtime
                            (system.cpp / amr_system.cpp) via add_executable,
                            HORS le bloc if(ADC_HAS_MPI) et HORS la ligne
                            generique add_executable(${name} ...) des helpers.
  - MPI                   : binaires add_executable declares DANS le bloc
                            if(ADC_HAS_MPI) (hors ligne generique du helper) ;
                            chacun rejoue typiquement np=1/2/4.
  - Python                : fichiers python/tests/test_*.py.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKE = ROOT / "tests" / "CMakeLists.txt"
PYTEST_DIR = ROOT / "python" / "tests"

# Ligne generique des fonctions helper (adc_add_test / adc_add_mpi_test) :
# add_executable(${name} ${name}.cpp). Ce n'est PAS un test, a exclure.
GENERIC_ADD_EXEC = re.compile(r"add_executable\(\$\{name\}")
ADD_EXEC = re.compile(r"^\s*add_executable\(")
ADC_ADD_TEST = re.compile(r"\badc_add_test\(")
MPI_GUARD_OPEN = re.compile(r"if\(ADC_HAS_MPI\)")
ENDIF = re.compile(r"^endif\(\)")


def count() -> dict[str, int]:
    lines = CMAKE.read_text().splitlines()

    adc_add_test = 0
    add_exec_runtime = 0   # non-MPI add_executable, hors ligne generique
    add_exec_mpi = 0       # add_executable dans le bloc ADC_HAS_MPI, hors generique

    in_mpi = False
    for line in lines:
        if MPI_GUARD_OPEN.search(line):
            in_mpi = True
        elif ENDIF.match(line) and in_mpi:
            in_mpi = False

        if ADC_ADD_TEST.search(line):
            adc_add_test += 1

        if ADD_EXEC.match(line) and not GENERIC_ADD_EXEC.search(line):
            if in_mpi:
                add_exec_mpi += 1
            else:
                add_exec_runtime += 1

    python = len(list(PYTEST_DIR.glob("test_*.py")))

    cpp_non_mpi = adc_add_test + add_exec_runtime
    return {
        "adc_add_test": adc_add_test,
        "add_executable_runtime": add_exec_runtime,
        "cpp_non_mpi_total": cpp_non_mpi,
        "mpi_binaries": add_exec_mpi,
        "python": python,
    }


def main() -> None:
    c = count()
    print(f"adc_add_test (ctest C++ hors-MPI)       : {c['adc_add_test']}")
    print(f"add_executable runtime (hors-MPI)        : {c['add_executable_runtime']}")
    print(f"  -> total cibles ctest C++ hors-MPI     : {c['cpp_non_mpi_total']}")
    print(f"add_executable bloc ADC_HAS_MPI          : {c['mpi_binaries']}")
    print(f"tests Python (python/tests/test_*.py)    : {c['python']}")


if __name__ == "__main__":
    main()

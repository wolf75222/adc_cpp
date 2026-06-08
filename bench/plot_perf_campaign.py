#!/usr/bin/env python3
"""Trace les graphes de la campagne perf ADC a partir de CSV.

Le script ne lance aucun bench et n'invente aucune valeur. Il lit les fichiers
CSV decrits dans docs/PERF_SCALING_FRONTENDS_AUDIT_2026-06-08.md et produit les
PNG disponibles. Les graphes dont les donnees manquent sont sautes.

Usage:
  python3 bench/plot_perf_campaign.py --results-dir bench/results --out-dir docs/perf_figures

Fichiers lus si presents:
  perf_scaling*.csv
  perf_frontends*.csv
  perf_phases*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict


def _need_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "matplotlib est requis pour tracer les graphes: %s" % exc
        )
    return plt


def _read_many(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                row = dict(row)
                row.setdefault("_source", os.path.basename(path))
                rows.append(row)
    return rows


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else default
    except ValueError:
        return default


def _s(row: dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else str(value)


def _resource(row: dict[str, str]) -> float:
    gpus = _f(row, "gpus", 0.0)
    if gpus > 0:
        return gpus
    np = _f(row, "np", 0.0)
    if np > 1:
        return np
    threads = _f(row, "threads", 0.0)
    return threads if threads > 0 else 1.0


def _norm_backend(row: dict[str, str]) -> str:
    return _s(row, "backend").replace("-pic", "")


def _thread_note(row: dict[str, str]) -> str:
    m = re.search(r"(?:^|\s)threads=([^\s,]+)", _s(row, "notes"))
    return m.group(1) if m else ""


def _label(row: dict[str, str], fields: tuple[str, ...]) -> str:
    parts = [_s(row, f) for f in fields if _s(row, f)]
    return " / ".join(parts) if parts else "series"


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print("wrote %s" % path)


def _filter_scaling(rows: list[dict[str, str]], kind: str) -> list[dict[str, str]]:
    with_kind = [r for r in rows if _s(r, "scaling").lower() == kind]
    if with_kind:
        return with_kind
    if kind == "strong":
        # Backward-compatible fallback: if no scaling column exists, plot all
        # perf_scaling rows as strong scaling. Weak scaling then stays skipped.
        if rows and all("scaling" not in r or not _s(r, "scaling") for r in rows):
            return rows
    return []


def plot_strong(rows: list[dict[str, str]], out_dir: str) -> None:
    rows = _filter_scaling(rows, "strong")
    if not rows:
        print("skip strong scaling: no rows")
        return
    plt = _need_matplotlib()
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        label = _label(r, ("case", "backend", "machine"))
        x = _resource(r)
        ms = _f(r, "per_step_ms", math.nan)
        if math.isfinite(ms) and ms > 0:
            groups[label].append((x, ms))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, pts in sorted(groups.items()):
        pts = sorted(pts)
        if not pts:
            continue
        base = pts[0][1]
        ax.plot([p[0] for p in pts], [base / p[1] for p in pts], marker="o", label=label)
    ax.set_xlabel("resources")
    ax.set_ylabel("speedup vs first point")
    ax.set_title("Strong scaling speedup")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "strong_scaling_speedup.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, pts in sorted(groups.items()):
        pts = sorted(pts)
        if not pts:
            continue
        base_x, base_ms = pts[0]
        ys = []
        for x, ms in pts:
            ideal = x / base_x if base_x > 0 else x
            ys.append((base_ms / ms) / ideal if ideal > 0 else math.nan)
        ax.plot([p[0] for p in pts], ys, marker="o", label=label)
    ax.set_xlabel("resources")
    ax.set_ylabel("efficiency")
    ax.set_title("Strong scaling efficiency")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "strong_scaling_efficiency.png")
    plt.close(fig)


def plot_weak(rows: list[dict[str, str]], out_dir: str) -> None:
    rows = _filter_scaling(rows, "weak")
    if not rows:
        print("skip weak scaling: no rows")
        return
    plt = _need_matplotlib()
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        label = _label(r, ("case", "backend", "machine"))
        x = _resource(r)
        ms = _f(r, "per_step_ms", math.nan)
        if math.isfinite(ms) and ms > 0:
            groups[label].append((x, ms))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, pts in sorted(groups.items()):
        pts = sorted(pts)
        if not pts:
            continue
        base = pts[0][1]
        ax.plot([p[0] for p in pts], [base / p[1] for p in pts], marker="o", label=label)
    ax.set_xlabel("resources")
    ax.set_ylabel("weak efficiency")
    ax.set_title("Weak scaling efficiency")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "weak_scaling_efficiency.png")
    plt.close(fig)


def plot_phases(rows: list[dict[str, str]], out_dir: str) -> None:
    if not rows:
        print("skip phase breakdown: no rows")
        return
    plt = _need_matplotlib()
    phases = sorted({r.get("phase", "") for r in rows if r.get("phase") and r.get("phase") != "TOTAL"})
    labels = []
    by_bar: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        phase = _s(r, "phase")
        if not phase or phase == "TOTAL":
            continue
        label = _label(r, ("case", "backend", "np", "threads"))
        if label not in labels:
            labels.append(label)
        by_bar[label][phase] += _f(r, "per_step_ms", 0.0)
    if not labels or not phases:
        print("skip phase breakdown: incomplete rows")
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5.2))
    bottom = [0.0] * len(labels)
    xs = list(range(len(labels)))
    for phase in phases:
        vals = [by_bar[label].get(phase, 0.0) for label in labels]
        ax.bar(xs, vals, bottom=bottom, label=phase)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("per step ms")
    ax.set_title("Phase breakdown")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "phase_breakdown_stacked.png")
    plt.close(fig)


def plot_frontends(rows: list[dict[str, str]], out_dir: str) -> None:
    if not rows:
        print("skip frontends: no rows")
        return
    rows = [r for r in rows if _f(r, "advance_ms", 0.0) > 0 or _f(r, "total_ms", 0.0) > 0]
    if not rows:
        print("skip frontends: no timed rows")
        return
    plt = _need_matplotlib()

    # Ratio vs first cpp-native row with same case/backend if ratio_vs_cpp is missing.
    cpp: dict[tuple[str, str, str], float] = {}
    cpp_fallback: dict[tuple[str, str], float] = {}
    for r in rows:
        if _s(r, "frontend") == "cpp-native":
            value = _f(r, "advance_ms", _f(r, "total_ms", math.nan))
            key = (_s(r, "case"), _norm_backend(r), _thread_note(r))
            cpp[key] = value
            cpp_fallback.setdefault((_s(r, "case"), _norm_backend(r)), value)

    labels = []
    ratios = []
    for r in rows:
        label = _label(r, ("case", "frontend", "cache_state"))
        labels.append(label)
        ratio = _f(r, "ratio_vs_cpp", math.nan)
        if not math.isfinite(ratio) or ratio <= 0:
            base = cpp.get(
                (_s(r, "case"), _norm_backend(r), _thread_note(r)),
                cpp_fallback.get((_s(r, "case"), _norm_backend(r)), math.nan),
            )
            val = _f(r, "advance_ms", _f(r, "total_ms", math.nan))
            ratio = val / base if math.isfinite(base) and base > 0 else math.nan
        ratios.append(ratio)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.8))
    ax.bar(range(len(labels)), ratios)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ratio vs cpp")
    ax.set_title("Frontend hot-loop ratios")
    ax.axhline(1.0, color="black", linewidth=1)
    _save(fig, out_dir, "frontend_ratios.png")
    plt.close(fig)

    dsl = [r for r in rows if "dsl" in _s(r, "frontend")]
    if dsl:
        labels = [_label(r, ("frontend", "cache_state")) for r in dsl]
        compile_ms = [_f(r, "compile_ms", 0.0) for r in dsl]
        total_ms = [_f(r, "total_ms", _f(r, "compile_ms", 0.0) + _f(r, "advance_ms", 0.0)) for r in dsl]
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.8))
        xs = list(range(len(labels)))
        ax.bar(xs, total_ms, label="total_ms")
        ax.bar(xs, compile_ms, label="compile_ms")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("ms")
        ax.set_title("DSL cold/warm compile impact")
        ax.legend()
        _save(fig, out_dir, "dsl_cold_warm.png")
        plt.close(fig)

    diag_rows = [r for r in rows if _f(r, "extract_ms", 0.0) > 0]
    if diag_rows:
        labels = [_label(r, ("case", "frontend", "cache_state")) for r in diag_rows]
        impact = [
            _f(r, "extract_ms", 0.0) / max(_f(r, "advance_ms", 0.0), 1e-300)
            for r in diag_rows
        ]
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.8))
        ax.bar(range(len(labels)), impact)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("extract_ms / advance_ms")
        ax.set_title("Diagnostics and extraction impact")
        _save(fig, out_dir, "diagnostics_io_impact.png")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="bench/results")
    ap.add_argument("--out-dir", default="docs/perf_figures")
    args = ap.parse_args()

    scaling = _read_many(os.path.join(args.results_dir, "perf_scaling*.csv"))
    phases = _read_many(os.path.join(args.results_dir, "perf_phases*.csv"))
    frontends = _read_many(os.path.join(args.results_dir, "perf_frontends*.csv"))

    print("loaded scaling=%d phases=%d frontends=%d" % (len(scaling), len(phases), len(frontends)))
    plot_strong(scaling, args.out_dir)
    plot_weak(scaling, args.out_dir)
    plot_phases(phases, args.out_dir)
    plot_frontends(frontends, args.out_dir)


if __name__ == "__main__":
    main()

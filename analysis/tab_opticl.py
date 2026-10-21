#!/usr/bin/env python3
"""The convex-hull trust region against SRC's correction, one row per hull setting.

Reads derived/opticl_hull.csv and writes tables/tab_opticl.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up, exact                                   # noqa: E402
from _paths import DERIVED, TABLES                                # noqa: E402

BENCH = {"methane": "Methane", "williams": "Williams--Otto", "hanoi": "Hanoi"}

rows = list(csv.DictReader(open(DERIVED / "opticl_hull.csv")))

lines, prev = [], None
for x in rows:
    b = BENCH[x["benchmark"]]
    key = (b, x["seed"])
    head = (f"{b if (prev is None or prev[0] != b) else ''} & "
            f"{x['seed'] if key != prev else ''}")
    if prev and prev[0] != b:
        lines.append("\\addlinespace[3pt]")
    p = float(x["mcnemar_p_exact"])
    p_cell = f"{p:.1e}" if p < 0.001 else half_up(exact(p))
    lines.append(f"{head} & {half_up(exact(x['hull_cost']), 4)} & "
                 f"{x['hull_violations']}/{x['n_objectives']} & "
                 f"{half_up(exact(x['method_cost']), 4)} & "
                 f"{x['method_violations']}/{x['n_objectives']} & {p_cell} \\\\")
    prev = key

tex = r"""{\scriptsize
\begin{longtable}{llccccc}
\caption{The convex-hull trust region of OptiCL against SRC's correction. For each of the
hull's three settings per seed, SRC's correction is taken at its most expensive setting
that costs no more than the hull. The $p$-value is from an exact McNemar test on the 200
shared objectives.}
\label{tab:opticl}\\
\toprule
 & & \multicolumn{2}{c}{convex hull} & \multicolumn{2}{c}{SRC correction} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
benchmark & seed & cost & violations & cost & violations & $p$ \\
\midrule
\endfirsthead
\toprule
benchmark & seed & cost & violations & cost & violations & $p$ \\
\midrule
\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}}
"""
(TABLES / "tab_opticl.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_opticl.tex'}  ({len(rows)} hull settings)")

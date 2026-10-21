#!/usr/bin/env python3
"""The coverage diagnostic summarised over the five repetitions of each configuration.

Reads derived/coverage_diagnostic_fits.csv and writes tables/tab_fcs_summary.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up                                          # noqa: E402
from _paths import DERIVED, TABLES                                # noqa: E402

ORDER = [
    ("methane / symbolic / scalar",          "Methane / symbolic / scalar"),
    ("methane / MLP / scalar",               "Methane / MLP / scalar"),
    ("methane / MLP / normalized",           "Methane / MLP / normalized"),
    ("methane / symbolic / normalized",      "Methane / symbolic / normalized"),
    ("Williams--Otto / symbolic / scalar",   "Williams--Otto / symbolic / scalar"),
    ("Williams--Otto / MLP / scalar",        "Williams--Otto / MLP / scalar"),
]

rows = list(csv.DictReader(open(DERIVED / "coverage_diagnostic_fits.csv")))


def defined(x):
    return x["matched_coverage"] not in ("", "undefined", "-")


def rng(values, places=2):
    lo, hi = min(values), max(values)
    return half_up(lo, places) if lo == hi else f"{half_up(lo, places)}--{half_up(hi, places)}"


lines = []
for key, label in ORDER:
    reps = [x for x in rows if x["configuration"] == key]
    if len(reps) != 5:
        raise SystemExit(f"{key}: expected 5 repetitions, found {len(reps)}")
    random_cov = [F(x["random_coverage"]) for x in reps]
    xstar = [F(x["xstar_coverage"]) for x in reps]
    below = sum(1 for v in xstar if v < F(9, 10))
    matched = [F(x["matched_coverage"]) for x in reps if defined(x)]
    drop = [F(x["drop_matched_minus_xstar"]) for x in reps if defined(x)]
    matched_cell = rng(matched, 3) + (f" ({len(matched)} of 5)" if len(matched) < 5 else "")
    lines.append(f"{label} & {rng(random_cov, 3)} & {rng(xstar)} & {below} of 5 & "
                 f"{matched_cell} & {rng(drop, 3)} \\\\")

tex = r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{The coverage diagnostic, summarized over the five repetitions of each
configuration: the range of coverage on random inputs, at the returned decisions
and at matched-boundary points, the number of repetitions in which coverage at the
returned decisions is below $0.90$, and the range of the drop from matched-boundary
points to returned decisions. Where matched-boundary points were found in fewer
than five repetitions, the number is given in brackets.}
\label{tab:fcs-fits}
\begin{tabular}{lccccc}
\toprule
 & random & returned & below & matched & \\
configuration & inputs & decisions & $0.90$ & boundary & drop \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(TABLES / "tab_fcs_summary.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_fcs_summary.tex'}  ({len(ORDER)} configurations)")

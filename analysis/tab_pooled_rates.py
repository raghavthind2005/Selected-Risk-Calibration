#!/usr/bin/env python3
"""Pooled violation rates in the matched-cost comparison, with intervals and seed ranges.

Reads derived/matched_cost_pooled.csv and writes tables/tab_pooled_rates.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up, exact                                   # noqa: E402
from _paths import DERIVED, TABLES                                # noqa: E402

BENCH = {"methane": "Methane", "williams": "Williams--Otto", "hanoi": "Hanoi"}
BUDGET = {"cap025": "0.25", "cap055": "0.55", "cap100": "1.0"}
NAMES = {"law_blind_envelope": "SRC correction$^{\\ast}$",
         "cmicl": "C-MICL",
         "cmicl_refit": "C-MICL refit",
         "wmicl_p5": "W-MICL ($P{=}5$)",
         "cqr_scaled": "Locally weighted",
         "scalar": "Scalar padding",
         "alpha_tightening": "$\\alpha$-tightening",
         "trust_region_box": "Trust region (box)"}
# The rows come from the CSV, so the roster is whatever the config produced.

rows = list(csv.DictReader(open(DERIVED / "matched_cost_pooled.csv")))

lines, prev = [], None
for x in rows:
    key = (x["benchmark"], x["budget"])
    if prev and key != prev:
        lines.append("\\addlinespace[2pt]")
    head = (f"{BENCH[x['benchmark']]} & {BUDGET[x['budget']]}" if key != prev else " & ")
    if not x["pooled_rate_exact"]:
        lines.append(f"{head} & {NAMES[x['arm']]} & -- & 0 & -- & -- \\\\")
    else:
        lines.append(
            f"{head} & {NAMES[x['arm']]} & {half_up(F(x['pooled_rate_exact']))} & "
            f"{x['n_admissible_seeds']} & "
            f"[{half_up(exact(x['cp95_lo']))}, {half_up(exact(x['cp95_hi']))}] & "
            f"{half_up(F(x['min_seed_rate']))}--{half_up(F(x['max_seed_rate']))} \\\\")
    prev = key

tex = r"""{\scriptsize
\begin{longtable}{llllccc}
\caption{Violation rates in the matched-cost comparison. The rate is the mean of the
per-seed rates over the seeds where the method can be compared with SRC's correction
(all five seeds for SRC's correction itself); the interval is a two-sided 95\%
Clopper--Pearson interval on the pooled objectives, and the range runs over seeds.}
\label{tab:pooled-rates}\\
\toprule
benchmark & budget & method & rate & seeds & 95\% interval & seed range \\
\midrule
\endfirsthead
\toprule
benchmark & budget & method & rate & seeds & 95\% interval & seed range \\
\midrule
\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}}
"""
(TABLES / "tab_pooled_rates.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_pooled_rates.tex'}  ({len(rows)} rows)")

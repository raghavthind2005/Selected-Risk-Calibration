#!/usr/bin/env python3
"""All 40 margins of the certification study, each certified twice.

Reads results/certification_anchors.csv and writes tables/tab_cert_all.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up, exact                                   # noqa: E402
from _paths import DERIVED, TABLES                                # noqa: E402

BENCH = {"methane": "Methane", "williams": "Williams--Otto"}
ORDER = {"methane": 0, "williams": 1}
ANCHOR = {"a0p25": "0.25", "a0p50": "0.50", "a0p75": "0.75",
          "a0p333": "1/3", "a0p667": "2/3"}
ANCHOR_VALUE = {"a0p25": 0.25, "a0p50": 0.50, "a0p75": 0.75,
                "a0p333": 1 / 3, "a0p667": 2 / 3}

rows = list(csv.DictReader(open(DERIVED / "certification_anchors.csv")))
rows.sort(key=lambda x: (ORDER[x["benchmark"]], int(x["seed"]), ANCHOR_VALUE[x["anchor"]]))

lines, prev = [], None
for x in rows:
    b = BENCH[x["benchmark"]]
    if prev and b != prev:
        lines.append("\\addlinespace[3pt]")
    theta = half_up(F(x["selected_lambda"]), 2)
    first = "yes" if x["first_decision"] == "certified" else "no"
    second = "yes" if x["second_decision"] == "certified" else "no"
    no_fresh = (x["seed"] == "10" or not x["fresh_violations"]
                or x["fresh_source"] in ("", "no fresh data"))
    fresh = "--" if no_fresh else f"{int(float(x['fresh_violations']))}/{x['fresh_n']}"
    lines.append(
        f"{b if b != prev else ''} & {x['seed']} & {ANCHOR[x['anchor']]} & {theta} & "
        f"{x['first_violations']}/{x['first_n']} & {half_up(exact(x['first_cp_upper']))} & {first} & "
        f"{x['second_violations']}/{x['second_n']} & {half_up(exact(x['second_cp_upper']))} & {second} & "
        f"{fresh} \\\\")
    prev = b

tex = r"""\begin{table}[p]
\centering
\scriptsize
\caption{All 40 margins of the certification study. Each margin is certified twice,
on two independent sets of 300 objectives; ``cert.'' says whether $\bar U\le\gamma=0.12$
at level $\delta/K$ with $K=40$, which holds exactly when $v\le 19$. The last column
counts violations on fresh objectives; seeds~4 and~10 have no fresh evaluation of
the certified margins.}
\label{tab:cert-all}
\begin{tabular}{llcc ccc ccc c}
\toprule
 & & & & \multicolumn{3}{c}{first certification} & \multicolumn{3}{c}{second certification} & \\
\cmidrule(lr){5-7}\cmidrule(lr){8-10}
benchmark & seed & anchor & $\theta$ & $v/300$ & $\bar U$ & cert. & $v/300$ & $\bar U$ & cert. & fresh \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(TABLES / "tab_cert_all.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_cert_all.tex'}  ({len(rows)} margins)")

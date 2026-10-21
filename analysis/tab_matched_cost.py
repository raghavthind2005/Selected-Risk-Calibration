#!/usr/bin/env python3
"""Violation rates at matched cost, one column per benchmark and budget.

Reads derived/matched_cost_pooled.csv and writes tables/tab_matched_conv.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up                                          # noqa: E402
from _paths import DERIVED, TABLES                                # noqa: E402
import _config                                                    # noqa: E402

LABELS = {"law_blind_envelope": "SRC correction$^{\\ast}$",
          "cmicl": "C-MICL~\\cite{bib12}",
          "cmicl_refit": "C-MICL refit$^{\\ddagger}$",
          "wmicl_p5": "W-MICL~\\cite{bib12}",
          "cqr_scaled": "Locally weighted~\\cite{bib31}",
          "scalar": "Scalar padding",
          "alpha_tightening": "$\\alpha$-tightening",
          "trust_region_box": "Box trust region$^{\\ddagger}$"}
_CFG = _config.study("matched_cost_table")
ARMS = [(a, LABELS[a]) for a in _config.roster("matched_cost_table", "method", "baselines")]

CELLS = [(b, c) for b in _CFG["benchmarks"] for c in _CFG["budgets"]]

rows = list(csv.DictReader(open(DERIVED / "matched_cost_pooled.csv")))
R = {(x["benchmark"], x["budget"], x["arm"]): x for x in rows}
# An arm the config no longer names has no row here, and no column in the table.
ARMS = [(a, lab) for a, lab in ARMS if any(k[2] == a for k in R)]

# Bold the lowest rate in each column, among the arms compared on all five seeds.
best = {}
for cell in CELLS:
    vals = [(half_up(F(R[(cell[0], cell[1], a)]["pooled_rate_exact"]), 2), a)
            for a, _ in ARMS
            if R[(cell[0], cell[1], a)]["pooled_rate_exact"]
            and int(R[(cell[0], cell[1], a)]["n_admissible_seeds"]) == 5]
    lowest = min(float(v) for v, _ in vals)
    best[cell] = {a for v, a in vals if float(v) == lowest}

body = []
for arm, label in ARMS:
    out = []
    for cell in CELLS:
        x = R[(cell[0], cell[1], arm)]
        if not x["pooled_rate_exact"]:
            out.append("--")
            continue
        v = half_up(F(x["pooled_rate_exact"]), 2)
        if arm in best[cell]:
            out.append(f"\\textbf{{{v}}}")
        elif int(x["n_admissible_seeds"]) < 5:
            out.append(f"\\textit{{{v}}}")
        else:
            out.append(v)
    body.append(label + " & " + " & ".join(out) + " \\\\")
    if arm == "law_blind_envelope":
        body.append("\\midrule")

tex = r"""\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Violation rates at matched cost (lower is better). Each rate is averaged
over the seeds on which the method could be compared with SRC's correction; rates in
italics use fewer than five seeds. Bold marks the lowest rate in each column among
methods compared on all five seeds. Budgets are fractions of C-MICL's own cost
$C^{\star}$. $^{\ast}$Sized to each budget without the certification step.
$^{\ddagger}$Our controls rather than published methods.}
\label{tab:matched}
\begin{tabular}{l ccc ccc ccc}
\toprule
 & \multicolumn{3}{c}{Methane} & \multicolumn{3}{c}{Williams--Otto} & \multicolumn{3}{c}{Hanoi} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
budget $/C^{\star}$ & 0.25 & 0.55 & 1.0 & 0.25 & 0.55 & 1.0 & 0.25 & 0.55 & 1.0 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
(TABLES / "tab_matched_conv.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_matched_conv.tex'}")

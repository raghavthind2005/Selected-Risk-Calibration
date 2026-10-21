#!/usr/bin/env python3
"""Fresh violation rates at equal cost in the certification study, pooled over anchors.

Reads derived/equal_cost_panel.csv and writes tables/tab_cert_panel.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up                                          # noqa: E402
from _paths import DERIVED, TABLES                                # noqa: E402

NAMES = [("pure_scalar", "Constant margin ($\\theta=0$)"),
         ("floorscale_lam0.25", "Fixed shape $\\theta=0.25$"),
         ("floorscale_lam0.5", "Fixed shape $\\theta=0.5$"),
         ("floorscale_lam0.75", "Fixed shape $\\theta=0.75$"),
         ("floorscale_lam1", "Fixed shape $\\theta=1$"),
         ("scaled_cmicl", "Scaled C-MICL"),
         ("alpha_tightening", "$\\alpha$-tightening$^{\\S}$"),
         ("trust_region_box", "Trust region (box)"),
         ("trust_region_knn", "Trust region (nearest neighbour)"),
         ("law_blind_envelope", "Correction of Section~\\ref{sec:exp-cost}")]

rows = list(csv.DictReader(open(DERIVED / "equal_cost_panel.csv")))


def get(subset, comparator):
    return [x for x in rows if x["subset"] == subset and x["comparator"] == comparator][0]


def cells(x):
    return (f"{half_up(F(x['pooled_risk_aggregate']))} & {x['more']} & "
            f"{x['fewer']} & {x['equal']} & {x['rows_compared']}")


body = []
for comparator, label in NAMES:
    allanch = get("all anchors", comparator)
    certified = get("certified anchors only", comparator)
    body.append(f"{label} & {cells(allanch)} & {cells(certified)} \\\\")

ref_all = get("all anchors", "trust_region_box")
ref_cert = get("certified anchors only", "trust_region_box")
top = (f"SRC's locked margin & {half_up(F(ref_all['primary_pooled_aggregate']))} & & & & "
       f"{ref_all['n_anchors']} & "
       f"{half_up(F(ref_cert['primary_pooled_aggregate']))} & & & & "
       f"{ref_cert['n_anchors']} \\\\")

tex = r"""\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\caption{Fresh violation rates at equal cost in the certification study, pooled over
anchors, for all 30 anchors with a fresh evaluation (seed~10 excluded) and for the 22
of them whose margin was certified. ``more'', ``fewer'' and ``equal'' count the anchors
at which the comparator had more, fewer or as many violations as SRC's locked margin;
$n$ is the number of anchors at which the comparator could be matched. At the methane
anchors where SRC locked $\theta=0$, the constant margin is the same margin, so those
anchors count as equal. $^{\S}$$\alpha$-tightening cannot hit a cost exactly and is
randomized between two neighbouring settings, so its counts are expected values.}
\label{tab:cert-panel}
\begin{tabular}{l ccccc ccccc}
\toprule
 & \multicolumn{5}{c}{all anchors} & \multicolumn{5}{c}{certified anchors} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
 & rate & more & fewer & equal & $n$ & rate & more & fewer & equal & $n$ \\
\midrule
""" + top + "\n\\midrule\n" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(TABLES / "tab_cert_panel.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_cert_panel.tex'}  ({len(NAMES)} comparators)")

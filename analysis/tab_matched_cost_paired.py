#!/usr/bin/env python3
"""The matched-cost comparisons on the same seeds, as baseline minus SRC's correction.

Reads derived/matched_cost_pooled.csv and derived/matched_cost_paired.csv and
writes tables/tab_matched.tex.
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
ARMS = [(a, LABELS[a]) for a in _CFG["baselines"]]
CELLS = [(b, c) for b in _CFG["benchmarks"] for c in _CFG["budgets"]]
METHOD_ARM = _CFG["method"]

pooled = list(csv.DictReader(open(DERIVED / "matched_cost_pooled.csv")))
paired = list(csv.DictReader(open(DERIVED / "matched_cost_paired.csv")))
METHOD = {(x["benchmark"], x["budget"]): x for x in pooled
          if x["arm"] == METHOD_ARM}
P = {(x["benchmark"], x["budget"], x["baseline"]): x for x in paired}
# A baseline the config no longer names has no paired row, and no table row.
ARMS = [(a, lab) for a, lab in ARMS if any(k[2] == a for k in P)]

lines = ["SRC correction$^{\\ast}$, rate & "
         + " & ".join(half_up(F(METHOD[c]["pooled_rate_exact"])) for c in CELLS)
         + " \\\\", "\\midrule"]
for arm, label in ARMS:
    out = []
    for cell in CELLS:
        x = P[(cell[0], cell[1], arm)]
        if not x["method_rate_exact"]:
            out.append("--")
            continue
        d = F(x["baseline_rate_exact"]) - F(x["method_rate_exact"])
        s = half_up(d)
        if d > 0:
            s = "+" + s
        sig = "^{\\ast\\ast}" if x["sig_bonferroni"] else ("^{\\ast}" if x["sig_p05"] else "")
        n = int(x["n_seeds"])
        sub = f"_{{{n}}}" if n < 5 else ""
        out.append(f"${s}{sig}{sub}$")
    lines.append(label + " & " + " & ".join(out) + " \\\\")

tex = r"""\begin{table*}[p]
\centering
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\caption{The comparisons of Table~\ref{tab:matched} on the same seeds. The first row is
the violation rate of SRC's correction$^{\ast}$ over five seeds. Every other entry is a baseline's violation
rate minus that of SRC's correction, on the seeds where the two are compared, so a
positive entry means SRC's correction was safer. A subscript gives the number of
seeds when it is below five, and ``--'' means no seed could be compared. Exact
McNemar test on the same objectives: $^{\ast}p<0.05$; $^{\ast\ast}$significant
after Bonferroni correction for all 62 tests. Budgets are fractions of C-MICL's own
cost $C^{\star}$. $^{\ast}$SRC's correction sized to each budget without the
certification step. $^{\ddagger}$Our controls rather than published methods.}
\label{tab:matched-diff}
\begin{tabular}{l ccc ccc ccc}
\toprule
 & \multicolumn{3}{c}{Methane} & \multicolumn{3}{c}{Williams--Otto} & \multicolumn{3}{c}{Hanoi} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
budget $/C^{\star}$ & 0.25 & 0.55 & 1.0 & 0.25 & 0.55 & 1.0 & 0.25 & 0.55 & 1.0 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
(TABLES / "tab_matched.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_matched.tex'}")

#!/usr/bin/env python3
"""Violations with no margin and with the conformal margin, per benchmark and seed.

Reads results/no_margin_vs_conformal.csv and writes tables/tab_nomargin.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up, exact                                   # noqa: E402
from _paths import RESULTS, TABLES                                # noqa: E402

BENCH = {"methane": "Methane", "williams": "Williams--Otto", "hanoi": "Hanoi"}
ORDER = {"methane": 0, "williams": 1, "hanoi": 2}

SRC = RESULTS / "no_margin_vs_conformal.csv"
if not SRC.exists():
    raise SystemExit(f"{SRC} is not present; this table is not built.")
rows = list(csv.DictReader(open(SRC)))
rows.sort(key=lambda x: (ORDER[x["benchmark"]], int(x["seed"])))

lines, prev = [], None
for x in rows:
    b = BENCH[x["benchmark"]]
    if prev and b != prev:
        lines.append("\\addlinespace[2pt]")
    q = float(x["q_hat"])
    worse = "$^{\\dagger}$" if x["conformal_worse_than_no_margin"] == "True" else ""
    lines.append(f"{b if b != prev else ''} & {x['seed']} & ${q:+.3f}$ & "
                 f"{x['no_margin_v_over_n']} & {x['conformal_v_over_n']}{worse} & "
                 f"{half_up(exact(x['heldout_coverage']))} \\\\")
    prev = b

tex = r"""\begin{table}[t]
\centering
\small
\caption{The surrogate deployed without a margin ($\hat h\le 0$) and with the
conformal margin ($\hat h+\hat q\le 0$): violations among 200 objectives per seed,
and the margin's coverage of 800 random held-out inputs. $^{\dagger}$Runs where the
margin gives more violations than no margin.}
\label{tab:nomargin}
\begin{tabular}{llcccc}
\toprule
benchmark & seed & $\hat q$ & no margin & conformal margin & coverage \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(TABLES / "tab_nomargin.tex").write_text(tex)
print(f"wrote {TABLES / 'tab_nomargin.tex'}  ({len(rows)} runs)")

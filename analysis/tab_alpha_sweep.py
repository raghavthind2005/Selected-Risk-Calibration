#!/usr/bin/env python3
"""The alpha sweep on methane, as the minimum and maximum over five repetitions.

Reads derived/alpha_sweep.csv and writes tables/tab_alpha_sweep_v2.tex.
"""
from __future__ import annotations
import csv, sys, pathlib
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up                                          # noqa: E402
from _paths import DERIVED, TABLES                                # noqa: E402

SURROGATE = {"symbolic": "symbolic", "mlp": "MLP"}

rows = list(csv.DictReader(open(DERIVED / "alpha_sweep.csv")))


def signed(value, places=3):
    s = half_up(F(value), places)
    return s if s.startswith("-") else "+" + s


lines = [r"\begin{table}[t]\centering", r"\small",
         r"\caption{The $\alpha$ sweep on methane over five repetitions, reported as the "
         r"minimum and maximum across them. The gap is $(1-\alpha)$ minus coverage at "
         r"$x^{\star}$, so a positive gap is coverage promised but not delivered, and the "
         r"mean objective is minimised, so a larger value is worse.}",
         r"\label{tab:alpha-sweep}",
         r"\begin{tabular}{l r rr rr rr rr}", r"\toprule",
         r"surrogate & $\alpha$ & \multicolumn{2}{c}{random inputs} & "
         r"\multicolumn{2}{c}{at $x^{\star}$} & \multicolumn{2}{c}{gap} & "
         r"\multicolumn{2}{c}{mean objective} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r" & & min & max & min & max & min & max & min & max \\", r"\midrule"]
prev = None
for r in rows:
    if prev is not None and r["surrogate"] != prev:
        lines.append(r"\midrule")
    name = SURROGATE[r["surrogate"]] if r["surrogate"] != prev else ""
    prev = r["surrogate"]
    lines.append(
        f"{name} & {half_up(F(r['alpha']), 2)} & "
        f"{half_up(F(r['sanity_min']), 3)} & {half_up(F(r['sanity_max']), 3)} & "
        f"{half_up(F(r['selected_min']), 3)} & {half_up(F(r['selected_max']), 3)} & "
        f"{signed(r['gap_min'])} & {signed(r['gap_max'])} & "
        f"{half_up(F(r['objective_min']), 3)} & {half_up(F(r['objective_max']), 3)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

(TABLES / "tab_alpha_sweep_v2.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {TABLES / 'tab_alpha_sweep_v2.tex'}  ({len(rows)} rows)")

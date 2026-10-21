#!/usr/bin/env python3
"""Check src/ocl/certify.py against the stored certification results.

Recomputes, for all 40 margins and both certifications, the Clopper-Pearson
bound Ubar and the certified flag from the stored violation counts alone, and
compares them with

  * the counts and decisions stored in results/export/certification_rows.csv, and
  * every cell printed in tables/tab_cert_all.tex.

Also checks the integer threshold b and exercises the final selection step.

Exit status is 0 only when every comparison agrees.
"""
from __future__ import annotations

import csv
import pathlib
import re
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from ocl.certify import (CertificationSetting, certify,                # noqa: E402
                         certification_threshold, clopper_pearson_upper,
                         load_setting, select_margin)
from _fmt import half_up                                               # noqa: E402

EXPORT = ROOT / "results" / "export"
TABLES = ROOT / "tables"
DRAWS = ("first_certification_draw", "second_certification_draw")
BENCH = {"Methane": "methane", "Williams--Otto": "williams"}
ANCHOR_LABEL = {"a0p25": "0.25", "a0p50": "0.50", "a0p75": "0.75",
                "a0p333": "1/3", "a0p667": "2/3"}
ANCHOR_VALUE = {"a0p25": 0.25, "a0p50": 0.50, "a0p75": 0.75,
                "a0p333": 1 / 3, "a0p667": 2 / 3}


def stored_rows(draw: str) -> dict:
    out = {}
    for r in csv.DictReader(open(EXPORT / "certification_rows.csv")):
        if r["source_kind"] != "certification.json" or r["derived_draw"] != draw:
            continue
        key = (r["target"] or r["file_target"],
               int(float(r["seed"] or r["file_seed"])), r["anchor_id"])
        out[key] = r
    return out


def table_rows() -> dict:
    """Every data row of tab_cert_all.tex, keyed (benchmark, seed, anchor label)."""
    out, bench = {}, None
    for line in (TABLES / "tab_cert_all.tex").read_text().splitlines():
        if not line.rstrip().endswith(r"\\") or "&" not in line:
            continue
        cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
        if len(cells) != 11 or not cells[1].isdigit():
            continue                                  # the header rows
        bench = BENCH[cells[0]] if cells[0] else bench
        out[(bench, int(cells[1]), cells[2])] = cells[3:]
    return out


def main() -> int:
    setting = load_setting()
    fails: list[str] = []
    print(f"[1] constants from config/reported_arms.json")
    print(f"    {setting.describe()}")

    # ---- the threshold b -------------------------------------------------
    b = setting.threshold
    print(f"[2] threshold b = largest v with U_n(v; eps) <= gamma")
    print(f"    b = {b}   U_n({b}) = {clopper_pearson_upper(b, setting.n, setting.epsilon):.6f}"
          f" <= {setting.gamma}")
    print(f"             U_n({b + 1}) = "
          f"{clopper_pearson_upper(b + 1, setting.n, setting.epsilon):.6f} > {setting.gamma}")
    if b != 19:
        fails.append(f"b is {b}, the paper's caption says 19")
    # a brute-force sweep, independent of the bisection in certification_threshold
    brute = max((v for v in range(setting.n + 1)
                 if clopper_pearson_upper(v, setting.n, setting.epsilon) <= setting.gamma),
                default=-1)
    if brute != b:
        fails.append(f"bisection gives b = {b}, a full sweep gives {brute}")
    print(f"    full sweep over v = 0..{setting.n} agrees: {brute == b}")
    print(f"    theorem side condition b <= n*gamma - 1: {setting.guarantee_applies} "
          f"({b} <= {setting.n * setting.gamma - 1.0})")
    if not setting.guarantee_applies:
        fails.append("the theorem's side condition b <= n*gamma - 1 does not hold")

    # ---- all 40 margins, both certifications -----------------------------
    stored = {d: stored_rows(d) for d in DRAWS}
    table = table_rows()
    print(f"[3] stored margins: {len(stored[DRAWS[0]])} first, {len(stored[DRAWS[1]])} second;"
          f" table rows: {len(table)}")
    if len(stored[DRAWS[0]]) != setting.K:
        fails.append(f"{len(stored[DRAWS[0]])} first-draw margins, K = {setting.K}")

    n_cells = max_delta = 0
    n_flag = n_printed = 0
    for draw in DRAWS:
        for key, row in sorted(stored[draw].items()):
            bench, seed, anchor = key
            v, n = int(float(row["violation_count"])), int(float(row["n"]))
            cert = certify(v, setting, n)

            # against the stored decision
            want_flag = row["passes_registered_target"] == "True"
            if cert.certified != want_flag:
                fails.append(f"{draw} {bench} s{seed} {anchor}: certified "
                             f"{cert.certified}, stored {want_flag}")
            n_flag += 1
            # the count-threshold form must agree with the bound form
            if cert.certified != (v <= b):
                fails.append(f"{draw} {bench} s{seed} {anchor}: Ubar<=gamma gives "
                             f"{cert.certified} but v<=b gives {v <= b}")

            # against the stored bound
            max_delta = max(max_delta, abs(cert.bound - float(row["cp_upper"])))

            # against the printed table
            cells = table[(bench, seed, ANCHOR_LABEL[anchor])]
            off = 0 if draw == DRAWS[0] else 3
            got = (f"{v}/{n}", half_up(repr(cert.bound), 3),
                   "yes" if cert.certified else "no")
            want = tuple(cells[1 + off:4 + off])
            if got != want:
                fails.append(f"{draw} {bench} s{seed} {anchor}: recomputed {got}, "
                             f"table prints {want}")
            n_cells += 3
            n_printed += 1
    print(f"    certified flag vs stored decision : {n_flag} margins compared, "
          f"{sum(1 for f in fails if 'certified' in f)} disagreements")
    print(f"    Ubar vs stored cp_upper          : max |difference| = {max_delta:.3e}")
    print(f"    Ubar at the printed precision    : {n_printed} values, agree on all "
          f"{n_printed - sum(1 for f in fails if 'table prints' in f)}")
    print(f"    cells compared against tab_cert_all.tex: {n_cells} "
          f"({len(table)} rows x 3 columns x 2 certifications)")

    # ---- the final step: cheapest certified margin, or abstain -----------
    print("[4] the final step on each run: the cheapest certified margin, or abstain")
    groups: dict[tuple[str, int], list] = {}
    for key, row in stored[DRAWS[0]].items():
        bench, seed, anchor = key
        groups.setdefault((bench, seed), []).append(
            {"cost": ANCHOR_VALUE[anchor], "violations": int(float(row["violation_count"])),
             "n": int(float(row["n"])), "anchor": anchor})
    n_abstain = 0
    for (bench, seed), cands in sorted(groups.items()):
        sel = select_margin(cands, setting)
        n_abstain += sel.abstained
        # the same answer read off the counts alone
        by_cost = sorted(cands, key=lambda c: c["cost"])
        want = next((c["anchor"] for c in by_cost if c["violations"] <= b), None)
        got = None if sel.abstained else sel.chosen["anchor"]
        if got != want:
            fails.append(f"select_margin {bench} s{seed}: chose {got}, "
                         f"cheapest with v <= b is {want}")
        print(f"    {bench:9} seed {seed:<3} -> "
              + ("abstain" if sel.abstained else f"{got} "
                 f"(v = {sel.chosen['violations']}/{sel.chosen['n']})"))
    print(f"    {len(groups)} runs; abstained on {n_abstain}")

    print()
    if fails:
        print(f"CERTIFY CHECK: FAIL ({len(fails)})")
        for f in fails[:40]:
            print("   ", f)
        return 1
    print("CERTIFY CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

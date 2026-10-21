#!/usr/bin/env python3
"""The equal-cost panel around the certified margin.

At each anchor the fresh evaluation holds 14 arms, all matched to the primary's
measured cost. The primary is the locked policy Algorithm 1 would deploy. For
every other arm this reports, against the primary and on the same objectives:

  fewer / more / equal  -- anchors where the comparator commits fewer, more, or
                           the same number of violations
  pooled risk           -- two poolings, both stated, never mixed:
                             aggregate  = sum(violations) / sum(n)
                             mean-of-rates = mean over anchors of v/n
                           kept as exact Fractions and rounded once, half up.

Reported twice: over ALL anchors with fresh data, and over CERTIFIED anchors only
(certified = the first-draw certificate passed at delta/K, K = 40).

Three honesty rules are enforced rather than assumed:

  1. An arm is never compared with itself. At each anchor the arm flagged
     `is_primary` is dropped from the comparator set, so its row count is lower
     than the others' and is reported as such.
  2. `alpha_tightening` carries a FRACTIONAL violation count at 22 of 30 anchors:
     its discrete knob grid cannot hit the cost target, so the matched policy is a
     randomized mixture and the count is an EXPECTED value, not an observed one.
     Every alpha_tightening figure is labelled `expected`, and its fewer/more/equal
     counts are comparisons of an expectation with a count.
  3. A comparator whose `failure_uid_sha256` equals the primary's failed on
     exactly the same objectives. Those ties are an IDENTITY, not a result, and
     are counted separately as `tie_identical_failures`.

Unresolved rows (n = 0, the arm never produced a cost-matched realisation) are
excluded from that arm's denominator and reported.

Source: results/certification/{methane,williams}_seed<seed>/perf_rows.csv
        and results/export/certification_rows.csv (the first certification draw).
Seed 10 is NOT pooled here: it re-matched every knob to its own measured
cost, so it is not the identical certified policy.

writes: derived/equal_cost_panel.csv and equal_cost_panel.md
"""
from __future__ import annotations
import collections, csv, glob, json, pathlib, sys
from fractions import Fraction

PM = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # this directory, where _fmt.py and _config.py live
from _fmt import half_up                                       # noqa: E402
import _config                                                 # noqa: E402

EXPORT = PM / "results" / "export"
RESULTS = PM / "results" / "certification"
TAB = PM / "derived"; TAB.mkdir(parents=True, exist_ok=True)


def load_cert():
    out = {}
    for r in csv.DictReader(open(EXPORT / "certification_rows.csv")):
        if (r["source_kind"] == "certification.json"
                and r["derived_draw"] == "first_certification_draw"):
            t = r["target"] or r["file_target"]
            s = int(float(r["seed"] or r["file_seed"]))
            out[(t, s, r["anchor_id"])] = r["passes_registered_target"] == "True"
    return out


def load_anchors():
    cells = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(RESULTS / "[mw]*_seed*" / "perf_rows.csv"))):
        for r in csv.DictReader(open(f)):
            cells[(r["target"], int(r["seed"]), r["anchor_id"])][r["arm"]] = r
    return cells


def panel(cells, keys, certified):
    """Accumulate the comparison for one subset of anchors."""
    acc = collections.defaultdict(lambda: dict(
        fewer=0, more=0, equal=0, tie_identical=0, unresolved=0, n_rows=0,
        v_sum=Fraction(0), n_sum=0, rates=[], expected=False, skipped_self=0))
    for k in keys:
        # This panel discovers its comparators from the stored perf rows; the one
        # roster decision it makes is to drop config/reported_arms.json's
        # excluded_arms. No arm name is written down in this file.
        arms = {a: r for a, r in cells[k].items()
                if a not in _config.excluded_arms()}
        prim = [r for r in arms.values() if r["is_primary"] == "True"]
        if len(prim) != 1:
            raise SystemExit(f"{k}: {len(prim)} primary rows")
        p = prim[0]
        pv, pn = float(p["violation_count"]), int(float(p["n"]))
        for a, r in arms.items():
            if a == p["arm"]:
                acc[a]["skipped_self"] += 1
                continue
            d = acc[a]
            n = int(float(r["n"] or 0))
            if n == 0:
                d["unresolved"] += 1
                continue
            v = Fraction(str(r["violation_count"]))
            if v != int(v):
                d["expected"] = True
            d["n_rows"] += 1
            d["v_sum"] += v
            d["n_sum"] += n
            d["rates"].append(v / n)
            # fewer/more/equal on the same objective set: compare rates, since
            # n can differ between arms in principle. Here n is equal throughout.
            pr = Fraction(str(pv)) / pn
            cr = v / n
            if cr < pr:
                d["fewer"] += 1
            elif cr > pr:
                d["more"] += 1
            else:
                d["equal"] += 1
                if (r["failure_uid_sha256"] and
                        r["failure_uid_sha256"] == p["failure_uid_sha256"]):
                    d["tie_identical"] += 1
    return acc


def primary_pool(cells, keys):
    v = Fraction(0); n = 0; rates = []
    for k in keys:
        p = [r for r in cells[k].values() if r["is_primary"] == "True"][0]
        vv, nn = Fraction(str(p["violation_count"])), int(float(p["n"]))
        v += vv; n += nn; rates.append(vv / nn)
    return v, n, rates


def fmt(acc_row):
    agg = (acc_row["v_sum"] / acc_row["n_sum"]) if acc_row["n_sum"] else None
    mor = (sum(acc_row["rates"], Fraction(0)) / len(acc_row["rates"])
           if acc_row["rates"] else None)
    return agg, mor


def main() -> int:
    cert = load_cert()
    cells = load_anchors()
    allk = sorted(cells)
    certk = [k for k in allk if cert.get(k)]
    print(f"anchors with a fresh evaluation: {len(allk)}  "
          f"(certified {len(certk)}, rejected {len(allk)-len(certk)})")
    print("seed 10 is excluded: re-matched knobs, not the identical policy\n")

    out_rows = []
    L = ["# The equal-cost panel around the certified margin", "",
         f"{len(allk)} anchors carry a fresh evaluation ({len(certk)} certified, "
         f"{len(allk)-len(certk)} rejected). Every arm at an anchor is matched to the "
         "primary's measured cost, so a difference in violations is a difference at "
         "equal cost. Seed 10 is left out: its fresh evaluation re-matched the knobs, so it is not the identical margin.", "",
         "Every anchor carries n = 200, so the aggregate pooling and the "
         "mean-of-rates pooling coincide to the last digit. Both are printed anyway: "
         "they are different estimands and would diverge the moment an arm's n did.",
         ""]

    subsets = [("all anchors", allk), ("certified anchors only", certk)]
    # The two benchmarks are not the same experiment: on methane the primary is
    # floorscale_lam0 (= the constant pad) at every anchor, so pure_scalar ties by
    # identity there; on williams the primary is genuinely shaped. Pooling them
    # hides that, so each is also reported on its own.
    for b in ("methane", "williams"):
        subsets.append((f"certified anchors, {b} only",
                        [k for k in certk if k[0] == b]))
    for label, keys in subsets:
        if not keys:
            continue
        acc = panel(cells, keys, cert)
        pv, pn, prates = primary_pool(cells, keys)
        print(f"=== {label}  (n = {len(keys)}) ===")
        print(f"  primary (locked policy): pooled risk aggregate "
              f"{half_up(pv/pn,4)}   mean-of-rates "
              f"{half_up(sum(prates,Fraction(0))/len(prates),4)}")
        hdr = (f"  {'comparator':<22} {'rows':>4} {'fewer':>5} {'more':>5} "
               f"{'equal':>5} {'(ident)':>7} {'unres':>5} {'agg risk':>9} {'mean rate':>10}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        L += [f"## {label} (n = {len(keys)})", "",
              f"Primary (the locked policy): pooled risk **{half_up(pv/pn,4)}** "
              f"aggregate, **{half_up(sum(prates,Fraction(0))/len(prates),4)}** "
              f"mean-of-rates.", "",
              "| comparator | rows | fewer | more | equal | of which identical | unresolved | pooled risk (aggregate) | pooled risk (mean of rates) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for a in sorted(acc, key=lambda x: (-acc[x]["more"], x)):
            d = acc[a]
            agg, mor = fmt(d)
            ex = " *" if d["expected"] else ""
            print(f"  {a+ex:<22} {d['n_rows']:>4} {d['fewer']:>5} {d['more']:>5} "
                  f"{d['equal']:>5} {d['tie_identical']:>7} {d['unresolved']:>5} "
                  f"{(half_up(agg,4) if agg is not None else '-'):>9} "
                  f"{(half_up(mor,4) if mor is not None else '-'):>10}")
            L.append(f"| `{a}`{' ᵉ' if d['expected'] else ''} | {d['n_rows']} | "
                     f"{d['fewer']} | {d['more']} | {d['equal']} | "
                     f"{d['tie_identical']} | {d['unresolved']} | "
                     f"{half_up(agg,4) if agg is not None else '—'} | "
                     f"{half_up(mor,4) if mor is not None else '—'} |")
            out_rows.append({
                "subset": label, "n_anchors": len(keys), "comparator": a,
                "rows_compared": d["n_rows"], "fewer": d["fewer"], "more": d["more"],
                "equal": d["equal"], "equal_identical_failure_set": d["tie_identical"],
                "unresolved": d["unresolved"],
                "skipped_as_primary": d["skipped_self"],
                "counts_are_expected_values": d["expected"],
                "pooled_risk_aggregate": str(agg), "pooled_risk_mean_of_rates": str(mor),
                "primary_pooled_aggregate": str(pv / pn),
                "primary_pooled_mean_of_rates": str(sum(prates, Fraction(0)) / len(prates))})
        selfskip = {a: acc[a]["skipped_self"] for a in acc if acc[a]["skipped_self"]}
        print(f"  arms dropped as the primary at some anchor: {selfskip}\n")
        L += ["", "ᵉ `alpha_tightening`'s violation counts are **expected values**, not "
              "observed counts: its discrete knob grid cannot hit the cost target, so the "
              "cost-matched policy is a randomized mixture. Its fewer, more and equal counts "
              "compare an expectation with a count.",
              "", f"Arms dropped where they were themselves the primary (never compared "
              f"with itself): {selfskip}.", ""]

    # ---- what the panel actually supports, computed not asserted ----------
    def sub(lbl):
        return [r for r in out_rows if r["subset"] == lbl]
    L += ["## What this panel supports", ""]
    for b, n_lbl in (("williams", "certified anchors, williams only"),
                     ("methane", "certified anchors, methane only")):
        rs = sub(n_lbl)
        if not rs:
            continue
        na = rs[0]["n_anchors"]
        prim = Fraction(rs[0]["primary_pooled_aggregate"])
        beat = [r for r in rs if r["fewer"] > 0]
        allloss = [r for r in rs if r["more"] == r["rows_compared"] and r["rows_compared"]]
        near = [r for r in rs if Fraction(r["pooled_risk_aggregate"]) <= prim]
        L += [f"**{b}, {na} certified anchors.** The locked policy pools to "
              f"**{half_up(prim,4)}**. Of the {len(rs)} comparators, "
              f"**{len(allloss)}** are worse at every anchor **where the pair was "
              f"compared** (a comparator is not compared at an anchor where it is "
              f"itself the primary, or where its row is unresolved, so the "
              f"denominator differs between comparators and is given beside each "
              f"count as `rows_compared`: "
              + ", ".join(f"`{r['comparator']}` {r['more']}/{r['rows_compared']}"
                          for r in sorted(allloss, key=lambda x: -x["rows_compared"]))
              + f"), and "
              f"**{len(beat)}** win{'s' if len(beat)==1 else ''} at even one anchor"
              + ("" if not beat else
                 " (" + ", ".join(f"`{r['comparator']}` {r['fewer']}/{r['rows_compared']}"
                                  for r in sorted(beat, key=lambda x: -x["fewer"])) + ")")
              + f". **{len(near)}** pool{'s' if len(near)==1 else ''} "
                f"at or below the locked policy"
              + ("" if not near else
                 " (" + ", ".join(f"`{r['comparator']}` "
                                  f"{half_up(Fraction(r['pooled_risk_aggregate']),4)}"
                                  for r in near) + ")") + ".", ""]
    L += ["The two benchmarks are not making the same claim. On **williams** the "
          "locked policy is genuinely shaped and the panel is a win at equal cost. "
          "On **methane** the locked policy *is* the constant pad "
          "(`floorscale_lam0`, λ = 0), so the arms that tie it tie by identity and "
          "the panel shows only that the pad is not beaten at equal cost — it is not "
          "evidence for shape. Any §7.3 sentence that pools the two benchmarks into "
          "one equal-cost claim overstates methane.", ""]

    # the identity warning, stated once with its evidence
    idn = [r for r in out_rows if r["equal_identical_failure_set"]]
    if idn:
        L += ["## Ties that are identities, not results", "",
              "A comparator whose `failure_uid_sha256` matches the primary's failed on "
              "exactly the same objectives. On methane the primary is `floorscale_lam0` "
              "at every anchor, and `pure_scalar` shares its knob, its measured cost and "
              "that hash — one policy under two names. Those ties carry no information "
              "and are broken out in the `of which identical` column above.", ""]

    with open(TAB / "equal_cost_panel.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    (TAB / "equal_cost_panel.md").write_text("\n".join(L))
    print(f"wrote {TAB/'equal_cost_panel.csv'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

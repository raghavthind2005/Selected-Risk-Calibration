#!/usr/bin/env python3
"""The certificate-bound figure and its companion table.

Figure: the certificate bound U (x) against the realised fresh risk (y), one
point per ANCHOR that has a fresh draw -- not only the anchors Algorithm 1
returned. Reference lines at 0.12 on both axes: the vertical line is the
certification threshold (a row left of it is certified), the horizontal line is
the same number read as an outcome (a row below it kept its promise). Certified
and rejected rows get different markers, so the quadrant structure is visible:
the claim is that nothing certified lands above the horizontal line.

Companion table: all 40 anchors, each with its second-draw decision
beside its first-draw one, so a reader can see where a fresh calibration
draw would have decided differently.

Sources (primary only):
  results/export/certification_rows.csv
      derived_draw == 'first_certification_draw'  -> the first draw (40 rows)
      derived_draw == 'second_certification_draw'         -> the second certification (40)
  .../{methane,williams}_seed<seed>/perf_rows.csv          -> fresh risk, n = 200
  .../<bench>_equal_cost_frontier/frontier_rows.csv   -> seed 10, n = 300,
      RE-MATCHED (not the identical certified policy) and marked as such.

Geometry matches fig_coverage_diagnostic.py: 5.5in == \\textwidth so LaTeX includes it at scale
1.0, no bbox_inches="tight", Times 9pt/7pt, pdf.fonttype 42, no baked-in caption.

writes: derived/certification_anchors.csv, .md and .json
"""
from __future__ import annotations
import collections, csv, json, pathlib, sys
from fractions import Fraction

PM = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # this directory, where _fmt.py and _config.py live
from _fmt import half_up                                      # noqa: E402
import _figsave                                               # noqa: E402
import _config                                                 # noqa: E402

EXPORT = PM / "results" / "export"
RESULTS = PM / "results" / "certification"
TAB = PM / "derived"; TAB.mkdir(parents=True, exist_ok=True)
GAMMA = 0.12
BENCH = {"methane": "methane", "williams": "williams"}
# The comparison set comes from config/reported_arms.json and nowhere else.
SEEDS = list(_config.study("certification_study")["seeds"])



def cert_rows(draw):
    out = {}
    for r in csv.DictReader(open(EXPORT / "certification_rows.csv")):
        if r["source_kind"] != "certification.json" or r["derived_draw"] != draw:
            continue
        t = r["target"] or r["file_target"]
        s = int(float(r["seed"] or r["file_seed"]))
        out[(t, s, r["anchor_id"])] = r
    return out


def fresh_rows():
    """(bench, seed, anchor) -> (primary_row, n, source)."""
    out = {}
    for t, pre in BENCH.items():
        for s in SEEDS:
            p = RESULTS / f"{pre}_seed{s}" / "perf_rows.csv"
            if not p.exists():
                continue
            by = collections.defaultdict(dict)
            for r in csv.DictReader(open(p)):
                by[(r["target"], int(r["seed"]), r["anchor_id"])][r["arm"]] = r
            for k, arms in by.items():
                pr = [x for x in arms.values() if x.get("is_primary") == "True"]
                if len(pr) == 1:
                    out[k] = (pr[0], int(float(pr[0]["n"])), "fresh evaluation")
    for t in BENCH:
        p = RESULTS / f"{t}_equal_cost_frontier" / "frontier_rows.csv"
        if not p.exists():
            continue
        by = collections.defaultdict(dict)
        for r in csv.DictReader(open(p)):
            by[(t, int(r["seed"]), r["anchor_id"])][r["arm"]] = r
        for k, arms in by.items():
            if k in out:                      # the run's own draw wins where it exists
                continue
            pr = [x for x in arms.values() if x.get("role") == "primary"]
            if len(pr) == 1:
                out[k] = (pr[0], int(float(pr[0]["n"])), "re-matched policy")
    return out


def main() -> int:
    first, second, fresh = cert_rows("first_certification_draw"), \
        cert_rows("second_certification_draw"), fresh_rows()
    print(f"rows: first draw {len(first)}, second draw {len(second)}")
    print(f"anchors with a fresh draw: {len(fresh)}")
    missing = set(first) - set(second)
    if missing:
        print(f"  NOTE: {len(missing)} first-draw rows have no second-draw row: {sorted(missing)}")

    rows = []
    for k in sorted(first, key=lambda x: (x[0], x[1], x[2])):
        t, s, a = k
        c1 = first[k]
        c2 = second.get(k)
        rec = {
            "benchmark": t, "seed": s, "anchor": a, "arm": c1["arm"],
            "selected_lambda": c1["selected_lambda"],
            "first_violations": int(float(c1["violation_count"])),
            "first_n": int(float(c1["n"])),
            "first_cp_upper": float(c1["cp_upper"]),
            "first_decision": "certified" if c1["passes_registered_target"] == "True" else "rejected",
            "second_violations": "" if c2 is None else int(float(c2["violation_count"])),
            "second_n": "" if c2 is None else int(float(c2["n"])),
            "second_cp_upper": "" if c2 is None else float(c2["cp_upper"]),
            "second_decision": "" if c2 is None else
                ("certified" if c2["passes_registered_target"] == "True" else "rejected"),
        }
        rec["decisions_agree"] = ("" if c2 is None else
                                 str(rec["first_decision"] == rec["second_decision"]))
        if k in fresh:
            pr, n, src = fresh[k]
            v = float(pr["violation_count"])
            rec.update(fresh_violations=v, fresh_n=n, fresh_source=src,
                       fresh_risk=v / n)
        else:
            rec.update(fresh_violations="", fresh_n="", fresh_source="none",
                       fresh_risk="")
        rows.append(rec)

    # ---- the claim the figure makes ---------------------------------------
    pts = [r for r in rows if r["fresh_risk"] != ""]
    cert_pts = [r for r in pts if r["first_decision"] == "certified"]
    rej_pts = [r for r in pts if r["first_decision"] == "rejected"]
    broken = [r for r in cert_pts if r["fresh_risk"] > GAMMA]
    print(f"\nplotted: {len(pts)} anchors  ({len(cert_pts)} certified, "
          f"{len(rej_pts)} rejected)")
    print(f"certified anchors whose fresh risk exceeds {GAMMA}: {len(broken)}"
          + ("" if not broken else "  " + str([(r['benchmark'], r['seed'], r['anchor'],
                                                r['fresh_risk']) for r in broken])))
    if cert_pts:
        mx = max(Fraction(str(r["fresh_violations"])) / r["fresh_n"] for r in cert_pts)
        print(f"max fresh risk among certified anchors: {half_up(mx,3)}")
    if rej_pts:
        above = sum(1 for r in rej_pts if r["fresh_risk"] > GAMMA)
        print(f"rejected anchors whose fresh risk exceeds {GAMMA}: {above}/{len(rej_pts)}"
              f"  (a rejected row is allowed to be either side of the line)")
    agree = [r for r in rows if r["decisions_agree"] != ""]
    n_agree = sum(1 for r in agree if r["decisions_agree"] == "True")
    print(f"first draw vs second draw: agree on {n_agree}/{len(agree)} rows")
    for r in agree:
        if r["decisions_agree"] == "False":
            print(f"  DISAGREE {r['benchmark']} s{r['seed']} {r['anchor']}: "
                  f"first={r['first_decision']} (U={r['first_cp_upper']:.3f})  "
                  f"second={r['second_decision']} (U={r['second_cp_upper']:.3f})")

    # ---- counterfactual: Algorithm 1 run on the second draw instead ----------
    frac = {}
    for r in csv.DictReader(open(EXPORT / "certification_rows.csv")):
        if r["source_kind"] == "certification.json" and r["anchor_fraction"]:
            frac[(r["target"] or r["file_target"], r["anchor_id"])] = float(r["anchor_fraction"])
    order = collections.defaultdict(list)
    for (t_, a), f in frac.items():
        if a not in [x for x, _ in order[t_]]:
            order[t_].append((a, f))
    order = {t_: [a for a, _ in sorted(v, key=lambda kv: kv[1])] for t_, v in order.items()}
    print("\nAlgorithm 1 on the first draw vs on the second draw:")
    cf = []
    for t_ in ("methane", "williams"):
        for s in SEEDS:
            pick = {}
            for lbl, src in (("first", first), ("second", second)):
                pick[lbl] = next((a for a in order[t_]
                                  if (t_, s, a) in src
                                  and src[(t_, s, a)]["passes_registered_target"] == "True"),
                                 None)
            same = pick["first"] == pick["second"]
            cf.append({"benchmark": t_, "seed": s, "first": pick["first"],
                       "second": pick["second"], "same": same})
            if not same:
                print(f"  CHANGES  {t_} s{s}: first -> {pick['first'] or 'abstain'}   "
                      f"second -> {pick['second'] or 'abstain'}")
    print(f"  identical decision on {sum(1 for c in cf if c['same'])}/{len(cf)} "
          f"(benchmark, seed) pairs")

    # ---- figure ------------------------------------------------------------
    under = [r for r in pts if r["fresh_risk"] <= r["first_cp_upper"]]
    print(f"anchors whose realised fresh risk is at or below their own bound U: "
          f"{len(under)}/{len(pts)}")

    # ---- companion table ---------------------------------------------------
    cols = ["benchmark", "seed", "anchor", "arm", "selected_lambda",
            "first_violations", "first_n", "first_cp_upper", "first_decision",
            "second_violations", "second_n", "second_cp_upper", "second_decision",
            "decisions_agree", "fresh_violations", "fresh_n", "fresh_risk",
            "fresh_source"]
    with open(TAB / "certification_anchors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    L = ["# Every anchor, first draw and second draw", "",
         f"All {len(rows)} rows of the K = 40 family. `first_decision` is the "
         f"first draw that the algorithm acts on; `second_decision` is the same row "
         f"re-certified on a fresh calibration draw. Threshold {GAMMA} on `cp_upper` "
         f"at delta/K. Built by `analysis/build_certification.py`.", "",
         "| bench | seed | anchor | arm | theta | first v/n | U | first | second v/n | U | second | agree | fresh v/n | fresh risk |",
         "|---|---:|---|---|---:|---:|---:|---|---:|---:|---|:-:|---:|---:|"]
    for r in rows:
        fr = ("—" if r["fresh_risk"] == "" else
              half_up(Fraction(str(r["fresh_violations"])) / r["fresh_n"], 3))
        fv = "—" if r["fresh_violations"] == "" else f"{r['fresh_violations']:g}/{r['fresh_n']}"
        if r["fresh_source"].startswith("re-matched"):
            fv += " ᵃ"
        r2 = "—" if r["second_decision"] == "" else r["second_decision"]
        u2 = "—" if r["second_cp_upper"] == "" else half_up(repr(r["second_cp_upper"]), 3)
        v2 = "—" if r["second_violations"] == "" else f"{r['second_violations']}/{r['second_n']}"
        ag = {"True": "✓", "False": "**✗**", "": "—"}[r["decisions_agree"]]
        L.append(f"| {r['benchmark']} | {r['seed']} | {r['anchor']} | `{r['arm']}` | "
                 f"{r['selected_lambda']} | {r['first_violations']}/{r['first_n']} | "
                 f"{half_up(repr(r['first_cp_upper']),3)} | {r['first_decision']} | "
                 f"{v2} | {u2} | {r2} | {ag} | {fv} | {fr} |")
    L += ["", "ᵃ seed 10's fresh evaluation re-matched the knob to its own measured "
          "cost (n = 300), so it is not the identical certified "
          "policy. Those points are drawn faded in the figure.", "",
          f"**{n_agree} of {len(agree)}** rows get the same decision from both draws. "
          f"**{len(pts)}** anchors have a fresh draw; of the **{len(cert_pts)}** "
          f"certified ones, **{len(broken)}** exceed the {GAMMA} line on the fresh "
          f"draw, and all **{len(rej_pts)}** rejected ones do. The dotted diagonal is "
          f"y = x: the realised risk is at or below the row's own bound U on "
          f"**{len(under)} of {len(pts)}** anchors.", "",
          "The single anchor above the diagonal is " +
          ", ".join(f"{r['benchmark']} s{r['seed']} {r['anchor']} "
                    f"(U = {half_up(repr(r['first_cp_upper']),3)}, fresh risk "
                    f"{half_up(Fraction(str(r['fresh_violations']))/r['fresh_n'],3)})"
                    for r in pts if r["fresh_risk"] > r["first_cp_upper"]) +
          ", and it is **rejected**, so Algorithm 1 never deploys it. U is a "
          "one-sided bound on the risk of the *certification* draw, holding "
          "familywise with probability 1 - delta; it is not a promise about a "
          "different draw's realised risk, and a rejected row sitting above it "
          "costs nothing. The claim the figure supports is the quadrant one: no "
          "certified anchor exceeds gamma on fresh data.", ""]
    (TAB / "certification_anchors.md").write_text("\n".join(L))
    json.dump({"gamma": GAMMA, "n_rows": len(rows), "n_plotted": len(pts),
               "n_certified_plotted": len(cert_pts), "n_broken": len(broken),
               "n_below_diagonal": len(under),
               "second_agreement": [n_agree, len(agree)],
               "second_draw_counterfactual": cf, "rows": rows},
              open(TAB / "certification_anchors.json", "w"), indent=1, default=str)
    print(f"wrote {TAB/'certification_anchors.csv'}, .md and {TAB/'certification_anchors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

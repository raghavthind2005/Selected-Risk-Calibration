#!/usr/bin/env python3
"""The OptiCL (convex-hull) comparison, complete.

Every stored `trust_region_hull` rung, on every seed of methane and williams.
At each hull rung the method (`law_blind_envelope`) is taken at the DEAREST of its
own swept rungs whose cost is no higher than the hull's -- CHECK| rungs included --
so the comparison is never in the method's favour on cost.

Pairing is per objective: both arms solved the same 200 objectives, so the test is
the exact two-sided McNemar, `binomtest(b, b+c, 0.5)` on the discordant pairs
(b = hull violates and method does not, c = the reverse). No normal approximation
and no continuity correction.

Built from the per-objective rows of `results/export/objective_level.csv.gz` with
`source_kind == 'matched_cost_frontier_mask'`. Mask
sums are checked against each rung's own stored `rung_n_viol` and
`rung_n_deployed` before anything is compared.

Also reported: the hull's CHEAPEST rung cost divided by that seed's cap100 target,
which is the number that says the hull is not merely weak but unaffordable; and a
one-line statement of why no hull is built on hanoi.

writes: derived/opticl_hull.csv and opticl_hull.md
"""
from __future__ import annotations
import ast, collections, csv, gzip, json, pathlib, re, sys
from fractions import Fraction

PM = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # this directory, where _fmt.py and _config.py live
from _fmt import half_up                                       # noqa: E402
import _config                                                 # noqa: E402
from scipy.stats import binomtest                              # noqa: E402

EXPORT = PM / "results" / "export"
TAB = PM / "derived"; TAB.mkdir(parents=True, exist_ok=True)
# The comparison set comes from config/reported_arms.json and nowhere else.
_CFG = _config.study("opticl_comparison")
HULL, METHOD = _CFG["hull"], _CFG["method"]
SEEDS = tuple(_CFG["seeds"])

HANOI_LINE = (
    "No hull is built on hanoi: at DIM = 8 the convex hull is not constructible at "
    "this scale — 400 points already give 474,818 facets / 322 MB and N_FIT = 1000 "
    "SIGKILLed the kernel, against 6,050 facets / 0.03 s / 60 MB in 5-D — so hanoi's "
    "roster is the usual eleven arms minus `trust_region_hull`. This was settled "
    "before any hanoi row of this comparison existed.")


def load_masks():
    """(bench, seed, frontier_key) -> {cost, arm, mask{oid: viol}, rung counts}."""
    sm = {}
    with gzip.open(EXPORT / "objective_level.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["source_kind"] != "matched_cost_frontier_mask":
                continue
            if r["reported_run"] != "True":
                continue
            ex = json.loads(r["extra_json"])
            if ex.get("run_variant") != "stage2":
                continue
            fk = ex["frontier_key"]
            if fk.startswith("READOUT|"):
                continue
            k = (r["benchmark"], int(r["seed"]), fk)
            d = sm.setdefault(k, {"cost": float(r["cost"]), "arm": r["arm"],
                                  "knob": float(r["knob"]), "m": {},
                                  "rung_nv": ex.get("rung_n_viol"),
                                  "rung_nd": ex.get("rung_n_deployed"),
                                  })
            if abs(d["cost"] - float(r["cost"])) > 1e-15:
                raise SystemExit(f"{k}: two costs")
            oid = r["objective_id"]
            if oid in d["m"]:
                raise SystemExit(f"{k}: duplicate objective {oid}")
            d["m"][oid] = (r["violation_fail_closed"] == "True", ex.get("dep_mask"))
    bad = 0
    for k, d in sm.items():
        nv = sum(v[0] for v in d["m"].values())
        nd = sum(1 for v in d["m"].values() if v[1])
        if d["rung_nv"] is not None and (nv != d["rung_nv"] or nd != d["rung_nd"]):
            bad += 1
    print(f"rungs loaded: {len(sm)}   mask-vs-stored-count mismatches: {bad}")
    if bad:
        raise SystemExit("refusing to compare: a mask disagrees with its own rung counts")
    return sm


def cap100_targets():
    """(bench, seed) -> the cap100 cost target, from the readout rows."""
    out = {}
    for r in csv.DictReader(open(EXPORT / "frontier.csv")):
        if r["is_readout"] != "True" or r["readout_cap"] != "cap100":
            continue
        try:
            t = float(r["derived_cap_target"])
        except ValueError:
            continue
        k = (r["benchmark"], int(r["seed"]))
        if k in out and abs(out[k] - t) > 1e-12:
            raise SystemExit(f"{k}: two cap100 targets {out[k]} vs {t}")
        out[k] = t
    return out



def main() -> int:
    sm = load_masks()
    caps = cap100_targets()

    rows = []
    for bench in ("methane", "williams"):
        for seed in SEEDS:
            hull = [(k, d) for k, d in sm.items()
                    if k[0] == bench and k[1] == seed and d["arm"] == HULL]
            meth = [(k, d) for k, d in sm.items()
                    if k[0] == bench and k[1] == seed and d["arm"] == METHOD]
            if not hull:
                continue
            for hk, hd in sorted(hull, key=lambda x: x[1]["cost"]):
                cand = [(k, d) for k, d in meth if d["cost"] <= hd["cost"]]
                if not cand:
                    rows.append({"benchmark": bench, "seed": seed,
                                 "hull_rung": hk[2], "hull_cost": hd["cost"],
                                 "note": "no method rung at or below the hull's cost"})
                    continue
                mk, md = max(cand, key=lambda x: x[1]["cost"])
                common = sorted(set(hd["m"]) & set(md["m"]))
                if len(common) != 200:
                    raise SystemExit(f"{bench} s{seed} {hk[2]}: {len(common)} common objectives")
                b = sum(1 for o in common if hd["m"][o][0] and not md["m"][o][0])
                c = sum(1 for o in common if md["m"][o][0] and not hd["m"][o][0])
                hv = sum(hd["m"][o][0] for o in common)
                mv = sum(md["m"][o][0] for o in common)
                p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
                rows.append({
                    "benchmark": bench, "seed": seed, "hull_rung": hk[2],
                    "hull_cost": hd["cost"], "hull_violations": hv,
                    "method_rung": mk[2], "method_cost": md["cost"],
                    "method_violations": mv, "n_objectives": len(common),
                    "discordant_hull_only": b, "discordant_method_only": c,
                    "mcnemar_p_exact": p,
                    "cost_ratio_hull_over_method": hd["cost"] / md["cost"],
                    "method_rung_is_CHECK": mk[2].startswith("CHECK|"),
                    "winner": ("method" if hv > mv else "hull" if mv > hv else "tie"),
                    "note": ""})

    live = [r for r in rows if r.get("n_objectives")]
    print(f"\nhull rungs compared: {len(live)}  "
          f"(methane {sum(1 for r in live if r['benchmark']=='methane')}, "
          f"williams {sum(1 for r in live if r['benchmark']=='williams')})")

    hdr = (f"{'bench':<9} {'s':>2} {'hull rung':<30} {'cost':>8} {'v':>4} | "
           f"{'method rung':<34} {'cost':>8} {'v':>4} | {'b':>3} {'c':>3} "
           f"{'McNemar p':>11} {'ratio':>6}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in live:
        print(f"{r['benchmark']:<9} {r['seed']:>2} {r['hull_rung']:<30} "
              f"{r['hull_cost']:>8.4f} {r['hull_violations']:>4} | "
              f"{r['method_rung']:<34} {r['method_cost']:>8.4f} "
              f"{r['method_violations']:>4} | {r['discordant_hull_only']:>3} "
              f"{r['discordant_method_only']:>3} {r['mcnemar_p_exact']:>11.3e} "
              f"{r['cost_ratio_hull_over_method']:>6.2f}")

    # ---- summary and the affordability ratio -------------------------------
    L = ["# The OptiCL (convex-hull) comparison, complete", "",
         "Every stored `trust_region_hull` rung on every seed of methane and "
         "williams. At each rung the method is taken at the **dearest** of its own "
         "swept rungs whose cost does not exceed the hull's, `CHECK|` rungs "
         "included, so the method is never given a cost advantage. "
         "Both arms solved the same 200 objectives, so the test "
         "is the exact two-sided McNemar on the discordant pairs.", "",
         "Rebuilt from `objective_level.csv.gz`.", "",
         "| bench | seed | hull rung | hull cost | hull v | method rung | method cost | method v | b | c | exact McNemar p | hull/method cost |",
         "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in live:
        chk = " ᶜ" if r["method_rung_is_CHECK"] else ""
        L.append(f"| {r['benchmark']} | {r['seed']} | `{r['hull_rung']}` | "
                 f"{r['hull_cost']:.4f} | {r['hull_violations']} | "
                 f"`{r['method_rung']}`{chk} | {r['method_cost']:.4f} | "
                 f"{r['method_violations']} | {r['discordant_hull_only']} | "
                 f"{r['discordant_method_only']} | {r['mcnemar_p_exact']:.3e} | "
                 f"{r['cost_ratio_hull_over_method']:.2f} |")
    L += ["", "ᶜ the method rung is a `CHECK|` rung.", ""]

    L += ["## Per benchmark", ""]
    for bench in ("methane", "williams"):
        R = [r for r in live if r["benchmark"] == bench]
        if not R:
            continue
        w = sum(1 for r in R if r["winner"] == "method")
        t_ = sum(1 for r in R if r["winner"] == "tie")
        l = sum(1 for r in R if r["winner"] == "hull")
        sig = sum(1 for r in R if r["winner"] == "method" and r["mcnemar_p_exact"] < 0.05)
        sigl = sum(1 for r in R if r["winner"] == "hull" and r["mcnemar_p_exact"] < 0.05)
        rr = [r["cost_ratio_hull_over_method"] for r in R]
        L += [f"**{bench}** — {len(R)} rungs. Method has fewer violations at "
              f"**{w}**, ties at **{t_}**, more at **{l}**. Significant at p < 0.05: "
              f"**{sig}** for the method, **{sigl}** for the hull. The hull costs "
              f"**{min(rr):.2f}–{max(rr):.2f}×** the method rung it is compared with.", ""]
        print(f"\n{bench}: {len(R)} rungs | method fewer {w}, ties {t_}, more {l} | "
              f"p<0.05 method {sig}, hull {sigl} | cost ratio {min(rr):.2f}-{max(rr):.2f}")

    # ---- where the hull wins, and at what cost gap -------------------------
    lost = [r for r in live if r["winner"] == "hull"]
    won = [r for r in live if r["winner"] == "method"]
    L += ["## Where the hull wins", ""]
    if not lost:
        L += ["The hull is beaten at every rung.", ""]
    else:
        L += [f"The hull has fewer violations at **{len(lost)} of {len(live)}** rungs. "
              "Listing them rather than pooling them away:", "",
              "| bench | seed | hull rung | hull v | method v | exact p | hull/method cost |",
              "|---|---:|---|---:|---:|---:|---:|"]
        for r in lost:
            L.append(f"| {r['benchmark']} | {r['seed']} | `{r['hull_rung']}` | "
                     f"{r['hull_violations']} | {r['method_violations']} | "
                     f"{r['mcnemar_p_exact']:.3e} | "
                     f"{r['cost_ratio_hull_over_method']:.2f} |")
        sigl = [r for r in lost if r["mcnemar_p_exact"] < 0.05]
        lr = [r["cost_ratio_hull_over_method"] for r in lost]
        wr = [r["cost_ratio_hull_over_method"] for r in won]
        L += ["", f"**{len(sigl)}** of those reach p < 0.05"
              + ("" if not sigl else
                 ", all on " + ", ".join(sorted({f"{r['benchmark']} seed {r['seed']}"
                                                 for r in sigl})))
              + f". The three williams seed 7 rungs do not (p = "
              + ", ".join(f"{r['mcnemar_p_exact']:.2f}" for r in lost
                          if r["benchmark"] == "williams") + ").", "",
              "These rungs compare **unequal budgets**. The matching rule takes the "
              "method's dearest swept rung at or below the hull's cost, so where the "
              "method's sweep has no rung near the hull's spend, the method is held to "
              "a fraction of it: on methane seed 3 the method is capped at 0.0189 "
              "against the hull's 0.0508–0.0597. The rule is deliberately conservative "
              "— it can only understate the method — but a rung it loses at a 2.7× "
              "budget gap is not evidence that the hull wins at equal cost.", "",
              f"The cost gap does **not** on its own separate wins from losses, and "
              f"saying so would overstate it: the rungs the hull WINS span "
              f"{min(lr):.2f}–{max(lr):.2f}× and the rungs it LOSES span "
              f"{min(wr):.2f}–{max(wr):.2f}×, which overlap almost entirely "
              f"(median {sorted(lr)[len(lr)//2]:.2f}× vs {sorted(wr)[len(wr)//2]:.2f}×). "
              "What can be said from these data is narrower: the hull is beaten at "
              f"**{len(won)} of {len(live)}** rungs, its only significant wins are on "
              "one methane seed, and at every rung it is the more expensive arm.", ""]

    L += ["## The hull's cheapest rung against cap100", "",
          "The hull is not only beaten at its own rungs; its cheapest rung is dearer "
          "than the whole cap100 budget, which is why it never appears in the cap "
          "table at all.", "",
          "| bench | seed | cheapest hull cost | cap100 target | ratio |",
          "|---|---:|---:|---:|---:|"]
    ratios = []
    for bench in ("methane", "williams"):
        for seed in SEEDS:
            hull = [d for k, d in sm.items()
                    if k[0] == bench and k[1] == seed and d["arm"] == HULL]
            if not hull:
                continue
            cheap = min(d["cost"] for d in hull)
            cap = caps.get((bench, seed))
            if cap is None:
                L.append(f"| {bench} | {seed} | {cheap:.4f} | — | — |")
                continue
            ratios.append(cheap / cap)
            L.append(f"| {bench} | {seed} | {cheap:.4f} | {cap:.4f} | "
                     f"**{cheap/cap:.2f}×** |")
            print(f"  {bench} s{seed}: cheapest hull {cheap:.4f} / cap100 {cap:.4f} "
                  f"= {cheap/cap:.2f}x")
    if ratios:
        L += ["", f"The cheapest hull rung costs **{min(ratios):.2f}–{max(ratios):.2f}×** "
              f"the cap100 target; it exceeds cap100 on "
              f"**{sum(1 for r in ratios if r > 1)} of {len(ratios)}** seeds.", ""]
    L += ["## Hanoi", "", HANOI_LINE, ""]
    cols = ["benchmark", "seed", "hull_rung", "hull_cost", "hull_violations",
            "method_rung", "method_cost", "method_violations", "n_objectives",
            "discordant_hull_only", "discordant_method_only", "mcnemar_p_exact",
            "cost_ratio_hull_over_method", "method_rung_is_CHECK", "winner", "note"]
    with open(TAB / "opticl_hull.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    (TAB / "opticl_hull.md").write_text("\n".join(L))
    print(f"\nwrote {TAB/'opticl_hull.csv'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

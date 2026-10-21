#!/usr/bin/env python3.12
"""Export the risk-cost frontier that stage [5] already computed, and plot it.

usage:  BENCHMARK=methane python3 frontier_export.py
        BENCHMARK=williams python3 frontier_export.py

WHY THIS EXISTS, AND WHAT IT IS NOT

The reported cap table shows one number per (arm, cap): the violation rate at a
deployment cost matched to `f * C*`. Those are 3-5 points read off a CURVE that
stage [5] computes in full and stores in every `seed<N>.json`. This script does
no new computation of any kind -- it reads the stored `frontier` dict and
reshapes it. Nothing here can change a reported number, and nothing here reads
the oracle: `viol_mask` and `dep_mask` were written during the run.

Two row families live in `frontier`, distinguished by key:

  "<arm>|<starts>|<dial>"   a SWEEP rung from [5]. The arm's dial was set to a
                            grid value, 200 objectives were solved, and cost and
                            risk were measured. These are the curve.
  "READOUT|<arm>|<cap>"     a REFINEMENT rung from [6]. The dial was root-found
                            so that the REALISED cost on D_perf hits `f * C*`
                            within REFINE_TOL. These are the cap-table cells.
  "CHECK|<arm>|<cap>"       the same rung re-solved at CHECK_STARTS (120). A
                            convergence check, not an independent measurement,
                            so it is exported but never plotted as a curve point.

SWEEP and READOUT rows carry the same measurement fields, so they are directly
comparable; a readout rung is simply a point whose dial was chosen by
root-finding against the realised cost rather than taken from the grid. CHECK
rows are DIFFERENT: run_seed.py records only arm/knob/cost/risk/masks for them
and omits dial, n_obj, solve_rate and the cost-validity counters, so they are
exported with those fields blank and are never drawn as curve points.

Two figures are written per benchmark:

  frontier_<bench>.png       x clipped to the DECISION REGION, [min cost, 1.6*C*].
                             Every cap lies in [0.25*C*, 1.0*C*] and the [4] grid
                             is grown toward 1.5*C*, so this is the whole region
                             in which any comparison is made. Rungs to the right
                             of the clip exist and are in the CSV; they are the
                             tails of arms whose dial can buy far more margin
                             than any cap permits, and they compress the region
                             that actually decides the result into a few pixels.
  frontier_<bench>_full.png  the same curves unclipped, so the reader can see
                             those tails and verify nothing was hidden by the clip.
"""
from __future__ import annotations
import csv, json, os, pathlib, re, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = os.environ.get("BENCHMARK", "methane")
HERE = pathlib.Path(__file__).parent
SUB = HERE / ("runs" if BENCH == "methane" else f"runs_{BENCH}")
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

# The method, then the published baselines, then our own controls. Order is the
# plotting/legend order and is fixed here so figures across benchmarks match.
METHOD = "law_blind_envelope"
ORDER = [METHOD, "cmicl", "cmicl_refit", "wmicl_p5", "wmicl_p25", "cqr_scaled",
         "scalar", "alpha_tightening", "trust_region_box", "trust_region_hull",
         "dose_response"]
# A context arm is not a published method and is excluded from the reported
# baseline tables, but it is still exported and still plotted, dashed, so a
# frontier figure shows everything that was measured.
CONTEXT = {"dose_response"}      # a context arm, never in the reported set

COLOR = {
    METHOD:              "#111111",
    "cmicl":             "#d62728",
    "cmicl_refit":       "#ff9896",
    "wmicl_p5":          "#1f77b4",
    "wmicl_p25":         "#aec7e8",
    "cqr_scaled":        "#2ca02c",
    "scalar":            "#9467bd",
    "alpha_tightening":  "#8c564b",
    "trust_region_box":  "#e377c2",
    "trust_region_hull": "#7f7f7f",
    "dose_response":     "#bcbd22",
}
LABEL = {METHOD: "ours (law-blind envelope)", "cmicl": "C-MICL",
         "cmicl_refit": "C-MICL (refit)", "wmicl_p5": "W-MICL (P=5)",
         "wmicl_p25": "W-MICL (P=25)", "cqr_scaled": "loc.-weighted conformal",
         "scalar": "constant pad", "alpha_tightening": "alpha tightening",
         "trust_region_box": "quantile box", "trust_region_hull": "OptiCL hull",
         "dose_response": "dose response"}

SWEEP_KEY = re.compile(r"^(?P<arm>.+)\|(?P<starts>\d+)\|(?P<dial>[-0-9.]+)$")


def discover():
    """Every finished seed on disk. Never hardcode a seed list: doing so once
    silently excluded williams seed 7 -- the weakest seed -- from every pooled
    mean, which inflated the numbers in our favour."""
    seeds = []
    for f in sorted(SUB.glob("seed*.json")):
        if f.name.count(".") > 1:      # .arms. / .partial. / .premerge.
            continue
        m = re.fullmatch(r"seed(\d+)", f.stem)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def load(seed):
    d = json.load(open(SUB / f"seed{seed}.json"))
    primary = int(d.get("primary_starts", 60))
    sweep, readout, check = [], [], []
    for k, r in d["frontier"].items():
        # CHECK rows carry a REDUCED schema -- they are a convergence re-solve of
        # an existing rung at CHECK_STARTS, not a new measurement, so run_seed.py
        # records only arm/knob/cost/risk/masks and omits dial, n_obj,
        # solve_rate, n_cost_valid, n_nesting_violations and per_law. Assuming a
        # uniform shape raises KeyError on the first CHECK row. Every optional
        # field is therefore read with .get and left blank when absent, rather
        # than defaulted to a number that would silently look like a measurement.
        rec = dict(seed=seed, arm=r["arm"], knob=r["knob"],
                   cost=r["cost"], risk=r["risk"], n_viol=r["n_viol"],
                   n_deployed=r["n_deployed"], starts=r["starts"],
                   dial=r.get("dial", ""), n_obj=r.get("n_obj", ""),
                   solve_rate=r.get("solve_rate", ""),
                   n_cost_valid=r.get("n_cost_valid", ""),
                   n_nesting_violations=r.get("n_nesting_violations", ""),
                   match_status=r.get("match_status", ""),
                   readout=r.get("readout", ""),
                   target_cost=r.get("target_cost", ""),
                   cost_error=r.get("cost_error", ""))
        if k.startswith("READOUT|"):
            rec["kind"] = "readout"; readout.append(rec)
        elif k.startswith("CHECK|"):
            rec["kind"] = "check"; check.append(rec)
        else:
            rec["kind"] = "sweep"
            # Only the primary budget defines the curve. CHECK_STARTS rungs, if
            # any were ever written into the sweep family, price the knob
            # differently and must not be mixed into one curve.
            if int(r["starts"]) == primary:
                sweep.append(rec)
    return d, sweep, readout, check


def main():
    seeds = discover()
    if not seeds:
        sys.exit(f"no finished seeds in {SUB}")
    rows, caps_by_seed, cstar = [], {}, {}
    for s in seeds:
        d, sw, ro, ck = load(s)
        rows += sw + ro + ck
        caps_by_seed[s] = d["caps"]
        cstar[s] = d["C_star"]

    csv_path = OUT / f"frontier_{BENCH}.csv"
    cols = ["seed", "kind", "arm", "starts", "dial", "knob", "cost", "risk",
            "n_viol", "n_deployed", "n_obj", "solve_rate", "n_cost_valid",
            "n_nesting_violations", "match_status", "readout", "target_cost",
            "cost_error"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    present = [a for a in ORDER if any(r["arm"] == a for r in rows)]
    extra = sorted({r["arm"] for r in rows} - set(ORDER))
    if extra:
        print(f"WARNING: arms not in ORDER, appended: {extra}")
        present += extra

    n = len(seeds)
    # Two framings of the SAME curves. The clip is a presentation choice, not a
    # filter: every point exists in the CSV and in the "_full" figure.
    for variant in ("decision", "full"):
        fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 4.3), squeeze=False,
                                 sharey=True)
        for ax, sd in zip(axes[0], seeds):
            xs_all = []
            for arm in present:
                sw = sorted([r for r in rows
                             if r["seed"] == sd and r["arm"] == arm
                             and r["kind"] == "sweep"
                             and np.isfinite(r["cost"]) and np.isfinite(r["risk"])],
                            key=lambda r: r["cost"])
                ro = sorted([r for r in rows
                             if r["seed"] == sd and r["arm"] == arm
                             and r["kind"] == "readout"
                             and np.isfinite(r["cost"]) and np.isfinite(r["risk"])],
                            key=lambda r: r["cost"])
                if not sw and not ro:
                    continue
                is_m = arm == METHOD
                col = COLOR.get(arm, "#999999")
                ls = "--" if arm in CONTEXT else "-"
                if sw:
                    ax.plot([r["cost"] for r in sw], [r["risk"] for r in sw],
                            color=col, lw=2.8 if is_m else 1.2, ls=ls,
                            marker="o", ms=3.2 if is_m else 2.0,
                            zorder=5 if is_m else 2,
                            alpha=1.0 if is_m else 0.8)
                    xs_all += [r["cost"] for r in sw]
                if ro:
                    ax.scatter([r["cost"] for r in ro], [r["risk"] for r in ro],
                               s=58 if is_m else 26, marker="D",
                               facecolor=col if is_m else "none",
                               edgecolor=col, lw=1.5,
                               zorder=6 if is_m else 3)
                    xs_all += [r["cost"] for r in ro]
            caps = caps_by_seed[sd]
            if caps:
                ax.axvspan(min(caps.values()), max(caps.values()),
                           color="#4c78a8", alpha=0.06, zorder=0)
            for cid, cv in sorted(caps.items()):
                ax.axvline(cv, color="#b0b0b0", lw=0.8, ls=":", zorder=1)
                ax.text(cv, 0.985, cid.replace("cap", ""), rotation=90,
                        fontsize=6, ha="right", va="top", color="#777777",
                        transform=ax.get_xaxis_transform())
            if variant == "decision" and xs_all:
                # Every cap lies in [0.25*C*, 1.0*C*] and [4] grows the grid
                # toward 1.5*C*, so 1.6*C* covers the entire region in which a
                # comparison can be made. The lower edge keeps any negative
                # rungs visible -- a cost below zero means the arm was CHEAPER
                # than `static`, which is real and worth seeing, not an error.
                # Bound the LEFT edge too. On methane seed 5 a single
                # cqr_scaled rung sits at cost -0.15 (it is genuinely cheaper
                # than `static` there); letting it set the limit pushes the
                # entire cap ladder into the right-hand fifth of the panel and
                # hides the region every comparison is made in. Clip to
                # -0.3*C*; anything further left is in the CSV and in the
                # "_full" figure.
                lo = max(min(0.0, min(xs_all)) * 1.05, -0.30 * cstar[sd]) - 1e-3
                ax.set_xlim(lo, 1.6 * cstar[sd])
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_title(f"seed {sd}   $C^*$ = {cstar[sd]:.5f}", fontsize=10)
            ax.set_xlabel("deployment cost", fontsize=9)
            ax.grid(alpha=0.22, lw=0.5)
            ax.tick_params(labelsize=8)
        axes[0][0].set_ylabel("violation rate among deployed\n"
                              r"$\Pr[\,h(x^*) > 0\,]$", fontsize=9)
        handles = [plt.Line2D([], [], color=COLOR.get(a, "#999999"),
                              lw=2.8 if a == METHOD else 1.4,
                              ls="--" if a in CONTEXT else "-",
                              label=LABEL.get(a, a)
                                    + (" [context]" if a in CONTEXT else ""))
                   for a in present]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(6, len(handles)), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.03))
        scope = (r"decision region, x clipped to $1.6\,C^*$"
                 if variant == "decision" else "full swept range")
        fig.suptitle(
            f"{BENCH} — risk–cost frontier, {len(seeds)} seeds ({scope})\n"
            "lines = [5] sweep rungs   ·   diamonds = [6] cost-matched readouts"
            "   ·   shaded = cap ladder   ·   cost = mean normalised objective "
            "excess over `static`", fontsize=10)
        fig.tight_layout(rect=(0, 0.11, 1, 0.93))
        png = OUT / (f"frontier_{BENCH}.png" if variant == "decision"
                     else f"frontier_{BENCH}_full.png")
        fig.savefig(png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"figure       -> {png}")

    print(f"seeds        : {seeds}")
    print(f"rows exported: {len(rows)}  "
          f"(sweep {sum(r['kind']=='sweep' for r in rows)}, "
          f"readout {sum(r['kind']=='readout' for r in rows)}, "
          f"check {sum(r['kind']=='check' for r in rows)})")
    print(f"arms plotted : {present}")
    print(f"csv          -> {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The certificate bound against the violation rate on fresh objectives.

One point per anchor with a fresh evaluation -- seeds 3, 5, 6, 7, 8 and 9 on both
benchmarks, 30 margins. x is the one-sided Clopper-Pearson bound Ubar at level
delta/K with K = 40, from the certification draw; y is the violation rate the same
locked margin realises on a fresh set of objectives. Circles are certified margins,
crosses rejected ones, colour is the benchmark. Seeds 4 and 10 have no fresh
evaluation of the certified margins.

Reference lines: gamma = 0.12 on both axes (dashed) and the diagonal y = x.

writes: figures/fig_certificate_bound.{pdf,png}, derived/certificate_bound_points.csv
"""
from __future__ import annotations
import csv, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.lines import Line2D                           # noqa: E402

from _figsave import savefig                                   # noqa: E402
from _paths import FIGURES, DERIVED                   # noqa: E402

SRC = DERIVED / "certification_anchors.json"
FIG = FIGURES
TAB = DERIVED

GAMMA = 0.12
LIM = 0.30
DROP_SEEDS = {10}
# Okabe-Ito, colour-blind safe, as used by the other paper figures.
COLOUR = {"methane": "#0072B2", "williams": "#D55E00"}
LABEL = {"methane": "methane", "williams": "Williams–Otto"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def points():
    rows = json.load(open(SRC))["rows"]
    out, dropped = [], []
    for r in rows:
        if r["fresh_risk"] in ("", None):
            continue
        if int(r["seed"]) in DROP_SEEDS:
            dropped.append((r["benchmark"], int(r["seed"]), r["anchor"]))
            continue
        out.append({
            "benchmark": r["benchmark"], "seed": int(r["seed"]), "anchor": r["anchor"],
            "arm": r["arm"], "selected_lambda": r["selected_lambda"],
            "cert_violations": int(r["first_violations"]), "cert_n": int(r["first_n"]),
            "U_bar": float(r["first_cp_upper"]),
            "decision": r["first_decision"],
            "fresh_violations": int(float(r["fresh_violations"])),
            "fresh_n": int(r["fresh_n"]),
            "fresh_risk": float(r["fresh_risk"]),
            "fresh_source": r["fresh_source"],
        })
    out.sort(key=lambda r: (r["benchmark"], r["seed"], r["anchor"]))
    return out, dropped


def _seg_hits_box(bb, p0, p1, n=600):
    """Does the segment p0->p1 (display coords) enter the bbox?"""
    for i in range(n + 1):
        t = i / n
        if bb.contains(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])):
            return True
    return False


def clashes(fig, ax, leg, pts):
    """Everything the legend is forbidden to touch. Returns a list of names."""
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    bad = []
    for p in pts:
        if bb.contains(*ax.transData.transform((p["U_bar"], p["fresh_risk"]))):
            bad.append(f"point {p['benchmark']} s{p['seed']} {p['anchor']}")
    T = ax.transData.transform
    lines = {
        "vertical dashed x=0.12": (T((GAMMA, 0.0)), T((GAMMA, LIM))),
        "horizontal dashed y=0.12": (T((0.0, GAMMA)), T((LIM, GAMMA))),
        "diagonal y=x": (T((0.0, 0.0)), T((LIM, LIM))),
    }
    for name, (p0, p1) in lines.items():
        if _seg_hits_box(bb, p0, p1):
            bad.append(name)
    return bb, bad


def main() -> int:
    pts, dropped = points()
    print(f"dropped {len(dropped)} seed-10 anchors: {dropped}")
    off = [p for p in pts if not (0 <= p["U_bar"] <= LIM and 0 <= p["fresh_risk"] <= LIM)]
    if off:
        print(f"!! {len(off)} point(s) outside [0,{LIM}]: "
              + str([(p['benchmark'], p['seed'], p['anchor']) for p in off]))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([0, LIM], [0, LIM], color="0.80", lw=0.6, ls=":", zorder=1)
    ax.axvline(GAMMA, color="0.40", lw=0.7, ls="--", zorder=2)
    ax.axhline(GAMMA, color="0.40", lw=0.7, ls="--", zorder=2)

    for bench in ("methane", "williams"):
        for dec, mk in (("certified", "o"), ("rejected", "x")):
            sel = [p for p in pts if p["benchmark"] == bench and p["decision"] == dec]
            if not sel:
                continue
            kw = (dict(color=COLOUR[bench]) if mk == "x"
                  else dict(facecolors=COLOUR[bench], edgecolors=COLOUR[bench]))
            ax.scatter([p["U_bar"] for p in sel], [p["fresh_risk"] for p in sel],
                       marker=mk, s=20, lw=0.9, zorder=4, clip_on=False, **kw)

    ax.set_xlim(0, LIM); ax.set_ylim(0, LIM)
    ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_xlabel(r"certificate bound $\bar{U}$")
    ax.set_ylabel("violation rate on fresh objectives")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=3.6, mew=0.9,
               mfc=COLOUR["methane"], mec=COLOUR["methane"],
               label=f"{LABEL['methane']}, certified"),
        Line2D([], [], ls="none", marker="x", ms=3.6, mew=0.9,
               color=COLOUR["methane"], label=f"{LABEL['methane']}, rejected"),
        Line2D([], [], ls="none", marker="o", ms=3.6, mew=0.9,
               mfc=COLOUR["williams"], mec=COLOUR["williams"],
               label=f"{LABEL['williams']}, certified"),
        Line2D([], [], ls="none", marker="x", ms=3.6, mew=0.9,
               color=COLOUR["williams"], label=f"{LABEL['williams']}, rejected"),
    ]

    # Preferred placement: inside the axes, left of x = 0.12 and above y = 0.12,
    # which is empty here. Fall back to two columns below the axes if the box
    # cannot be made to fit without touching a point, a dashed line or the
    # diagonal. The choice is made by measurement, not by eye.
    placement = "upper left, inside the axes"
    fig.subplots_adjust(left=0.165, right=0.955, top=0.975, bottom=0.155)
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                    handlelength=0.9, handletextpad=0.25, labelspacing=0.28,
                    borderpad=0.0, borderaxespad=0.25)
    bb, bad = clashes(fig, ax, leg, pts)
    if bad:
        print(f"in-axes legend touches {len(bad)}: {bad} -> falling back")
        leg.remove()
        placement = "below the axes, two columns"
        fig.subplots_adjust(left=0.165, right=0.955, top=0.975, bottom=0.295)
        leg = fig.legend(handles=handles, loc="lower center",
                         bbox_to_anchor=(0.5, 0.012), ncol=2, frameon=False,
                         handlelength=0.9, handletextpad=0.25,
                         columnspacing=1.2, labelspacing=0.28, borderpad=0.0)
        bb, bad = clashes(fig, ax, leg, pts)

    x0, y0 = ax.transData.inverted().transform((bb.x0, bb.y0))
    x1, y1 = ax.transData.inverted().transform((bb.x1, bb.y1))
    print(f"legend placement: {placement}")
    print(f"  display px : x {bb.x0:.1f}-{bb.x1:.1f}  y {bb.y0:.1f}-{bb.y1:.1f}")
    print(f"  data coords: x {x0:.4f}-{x1:.4f}  y {y0:.4f}-{y1:.4f}")
    xl = ax.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    if placement.startswith("below"):
        print(f"  gap below the x-label: {xl.y0 - bb.y1:.1f} px "
              f"(x-label y {xl.y0:.1f}-{xl.y1:.1f})")
    print(f"  touches: {len(bad)}" + ("" if not bad else f"  {bad}"))
    if bad:
        return 1

    for ext in ("pdf", "png"):
        savefig(fig, FIG / f"fig_certificate_bound.{ext}", dpi=300, omit_date=True)
    print(f"wrote {FIG/'fig_certificate_bound.pdf'} and .png  (3.4 x 2.6 in)")

    cols = ["benchmark", "seed", "anchor", "arm", "selected_lambda",
            "cert_violations", "cert_n", "U_bar", "decision",
            "fresh_violations", "fresh_n", "fresh_risk", "fresh_source"]
    with open(TAB / "certificate_bound_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(pts)
    print(f"wrote {TAB/'certificate_bound_points.csv'}  ({len(pts)} plotted points)")

    # ---- the numbers the caption needs -------------------------------------
    for b in ("methane", "williams"):
        c = [p for p in pts if p["benchmark"] == b and p["decision"] == "certified"]
        r = [p for p in pts if p["benchmark"] == b and p["decision"] == "rejected"]
        print(f"{LABEL[b]}: {len(c)} certified, {len(r)} rejected")
    cert = [p for p in pts if p["decision"] == "certified"]
    rej = [p for p in pts if p["decision"] == "rejected"]
    print(f"plotted {len(pts)} margins: {len(cert)} certified, {len(rej)} rejected")
    hi = max(cert, key=lambda p: p["fresh_risk"])
    print(f"highest fresh rate among certified: {hi['fresh_risk']:.4f} "
          f"({hi['fresh_violations']}/{hi['fresh_n']}, {LABEL[hi['benchmark']]} "
          f"s{hi['seed']} {hi['anchor']})")
    lo_r = min(rej, key=lambda p: p["fresh_risk"]); hi_r = max(rej, key=lambda p: p["fresh_risk"])
    print(f"rejected fresh rates range {lo_r['fresh_risk']:.4f} "
          f"({lo_r['fresh_violations']}/{lo_r['fresh_n']}, {LABEL[lo_r['benchmark']]} "
          f"s{lo_r['seed']}) to {hi_r['fresh_risk']:.4f} "
          f"({hi_r['fresh_violations']}/{hi_r['fresh_n']}, {LABEL[hi_r['benchmark']]} "
          f"s{hi_r['seed']})")
    below = [p for p in pts if p["fresh_risk"] <= p["U_bar"]]
    above = [p for p in pts if p["fresh_risk"] > p["U_bar"]]
    print(f"at or below the diagonal: {len(below)}/{len(pts)}; above: "
          + str([(LABEL[p['benchmark']], p['seed'], p['anchor'], p['decision']) for p in above]))
    broken = [p for p in cert if p["fresh_risk"] > GAMMA]
    print(f"certified anchors above gamma = {GAMMA}: {len(broken)}")
    print(f"seeds present: {sorted({p['seed'] for p in pts})}")
if __name__ == "__main__":
    raise SystemExit(main())

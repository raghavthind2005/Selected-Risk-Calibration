#!/usr/bin/env python3
"""The coverage diagnostic: coverage by configuration, and against the penalty.

Built from the 30 stored fits -- 6 configurations x 5 repetitions {0, 101, 102,
103, 104}.

Panel (a)  coverage by configuration, three quantities per configuration:
             random inputs (circles), matched-boundary points (triangles),
             returned decisions x* (diamonds).
           Large marker = mean over repetitions, small markers = the individual
           repetitions, bar = min..max. Undefined matched-boundary values are
           left out of the mean and the count is annotated.
Panel (b)  mean coverage at x* against the penalty weight lambda, with a
           min..max band.

The palette is Okabe-Ito, which is colour-blind safe. Every font is at least
8 pt at the 5.5 in text width.

writes: figures/fig_coverage_diagnostic.{pdf,png}
        derived/fig_coverage_diagnostic_values.csv  (mean/min/max behind every
                                                     plotted marker)
"""
from __future__ import annotations
import csv, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _figsave import savefig                                   # noqa: E402
from _paths import COVERAGE, FIGURES, DERIVED                  # noqa: E402
OUT = FIGURES
OUT.mkdir(parents=True, exist_ok=True)
FITS_M, FITS_W = COVERAGE / "methane", COVERAGE / "williams"
DRAWS = (0, 101, 102, 103, 104)

# ---------------------------------------------------------------- style
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.handlelength": 1.4, "legend.handletextpad": 0.45,
    "legend.labelspacing": 0.35, "legend.borderpad": 0.35,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
TEXTWIDTH_IN, FIG_H_IN = 6.5, 3.0
YLIM = (0.0, 1.02)
# Okabe-Ito, colour-blind safe
C_BLUE, C_VERM, C_GREEN = "#0072B2", "#D55E00", "#009E73"
C_PINK, C_SKY, C_ORANGE = "#CC79A7", "#56B4E9", "#E69F00"

# (key, display label, colour) -- the order the brief fixes
CONFIGS = [
    # (json key, full label for the legend, short two-line tick label, colour)
    ("methane:symbolic:scalar",     "methane / symb. / scalar",
     "methane\nsymb/scal",   C_BLUE),
    ("methane:mlp:scalar",          "methane / MLP / scalar",
     "methane\nMLP/scal",    C_VERM),
    ("methane:mlp:normalized",      "methane / MLP / norm.",
     "methane\nMLP/norm",    C_GREEN),
    ("methane:symbolic:normalized", "methane / symb. / norm.\u2020",
     "methane\nsymb/norm\u2020", C_PINK),
    ("williams:symbolic",           "W\u2013Otto / symb. / scalar",
     "W\u2013Otto\nsymb/scal",  C_SKY),
    ("williams:mlp",                "W\u2013Otto / MLP / scalar",
     "W\u2013Otto\nMLP/scal",   C_ORANGE),
]
QUANTITIES = [   # (json key, marker, label)
    ("sanity_random_coverage",             "o", "random inputs"),
    ("mechanism_matched_boundary_coverage", "^", "matched-boundary points"),
    ("reported_selected_coverage",         "D", "returned decisions $x^{\\star}$"),
]


def path_for(key: str, draw: int) -> pathlib.Path:
    bench, rest = key.split(":", 1)
    if bench == "methane":
        s_, c_ = rest.split(":")
        return FITS_M / f"draw{draw}" / f"coverage_{s_}_{c_}.json"
    return FITS_W / f"draw{draw}" / f"coverage_wo_{rest}.json"


def load_all() -> dict:
    fits: dict = {}
    for key, _lab, _short, _col in CONFIGS:
        for d in DRAWS:
            p = path_for(key, d)
            if not p.exists():
                raise SystemExit(f"missing stored fit: {p}")
            fits[(key, d)] = json.load(open(p))
    return fits


def finite(vals):
    return [v for v in vals if v is not None and np.isfinite(v)]


def main() -> int:
    fits = load_all()
    plt.rcParams.update(RC)
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, FIG_H_IN),
        gridspec_kw=dict(width_ratios=[1.78, 1.0], wspace=0.17))
    rows: list[dict] = []

    # ------------------------------------------------ panel (a)
    offs = (-0.26, 0.0, 0.26)
    partial_notes: list = []
    for xi, (key, lab, short, col) in enumerate(CONFIGS):
        if xi:
            axA.axvline(xi - 0.5, color="0.92", lw=0.5, zorder=0)
        for (jkey, mk, qlab), off in zip(QUANTITIES, offs):
            raw = [fits[(key, d)].get(jkey) for d in DRAWS]
            vals = finite(raw)
            n_def, n_tot = len(vals), len(raw)
            x = xi + off
            if vals:
                lo, hi, mu = min(vals), max(vals), float(np.mean(vals))
                axA.plot([x, x], [lo, hi], color=col, lw=1.0, zorder=2,
                         solid_capstyle="butt")
                axA.plot([x] * len(vals), vals, marker=mk, ms=2.6, ls="none",
                         color=col, mfc="white", mew=0.7, zorder=3)
                axA.plot([x], [mu], marker=mk, ms=6.4, ls="none", color=col,
                         mec="black", mew=0.5, zorder=4)
            else:
                lo = hi = mu = float("nan")
                axA.text(x, 0.04, "n/a", ha="center", va="bottom",
                         fontsize=8, color=col)
            if n_def < n_tot:          # say how many repetitions were usable
                partial_notes.append((xi, qlab, n_def, n_tot, col))
            rows.append(dict(panel="a", configuration=lab, quantity=qlab,
                             json_key=jkey, n_used=n_def, n_total=n_tot,
                             mean=("" if not vals else f"{mu:.6f}"),
                             min=("" if not vals else f"{lo:.6f}"),
                             max=("" if not vals else f"{hi:.6f}"),
                             values=";".join("undefined" if (v is None or not np.isfinite(v))
                                             else f"{v:.6f}" for v in raw)))
    axA.axhline(0.90, color="black", ls=(0, (4, 2)), lw=0.9, zorder=1)
    axA.text(1.012, 0.90, "0.90", transform=axA.get_yaxis_transform(),
             ha="left", va="center", fontsize=8)
    axA.set_xlim(-0.6, len(CONFIGS) - 0.4)
    axA.set_ylim(*YLIM)
    axA.set_xticks(range(len(CONFIGS)))
    axA.set_xticklabels([sh for _k, _l, sh, _c in CONFIGS], ha="center",
                        va="top", linespacing=1.3)
    axA.set_ylabel("empirical coverage")
    axA.set_title("(a)", pad=4)
    axA.yaxis.grid(True, color="0.9", linewidth=0.5, zorder=0)
    axA.set_axisbelow(True)
    marker_handles = [Line2D([], [], marker=mk, ls="none", ms=5.2, color="0.25",
                             mec="black", mew=0.5, label=qlab)
                      for _jk, mk, qlab in QUANTITIES]

    # ------------------------------------------------ panel (b)
    lam_ref = [e["lambda"] for e in fits[(CONFIGS[0][0], DRAWS[0])]["dose_response"]]
    for key, lab, short, col in CONFIGS:
        curves = []
        for d in DRAWS:
            dr = {e["lambda"]: e["coverage"] for e in fits[(key, d)]["dose_response"]}
            curves.append([dr.get(l, np.nan) for l in lam_ref])
        arr = np.asarray(curves, float)
        mu = np.nanmean(arr, axis=0)
        lo = np.nanmin(arr, axis=0)
        hi = np.nanmax(arr, axis=0)
        xs = np.arange(len(lam_ref))
        axB.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=1)
        axB.plot(xs, mu, lw=1.2, color=col, label=lab, zorder=3)
        for i, l in enumerate(lam_ref):
            rows.append(dict(panel="b", configuration=lab,
                             quantity="coverage at x* (dose-response)",
                             json_key=f"dose_response[lambda={l:g}]",
                             n_used=int(np.sum(np.isfinite(arr[:, i]))),
                             n_total=len(DRAWS),
                             mean=f"{mu[i]:.6f}", min=f"{lo[i]:.6f}", max=f"{hi[i]:.6f}",
                             values=";".join(f"{v:.6f}" for v in arr[:, i])))
    axB.axhline(0.90, color="black", ls=(0, (4, 2)), lw=0.9, zorder=2)
    axB.set_xticks(range(len(lam_ref)))
    axB.set_xticklabels([f"{l:g}" for l in lam_ref], rotation=45,
                        ha="right", rotation_mode="anchor")
    axB.set_ylim(*YLIM)
    axB.set_xlabel("$\\lambda$")
    axB.set_ylabel("coverage at $x^{\\star}$")
    axB.set_title("(b)", pad=4)
    axB.yaxis.grid(True, color="0.9", linewidth=0.5, zorder=0)
    axB.set_axisbelow(True)
    colour_handles = [Line2D([], [], color=col, lw=1.8, label=lab)
                      for _k, lab, _sh, col in CONFIGS]
    fig.legend(handles=marker_handles + colour_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=3, frameon=False, columnspacing=1.6, handletextpad=0.5,
               fontsize=8)

    fig.subplots_adjust(left=0.062, right=0.992, top=0.912, bottom=0.345)
    for ext in ("pdf", "png"):
        savefig(fig, OUT / f"fig_coverage_diagnostic.{ext}", dpi=400, omit_date=True)
    plt.close(fig)

    with open(DERIVED / "fig_coverage_diagnostic_values.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"wrote {OUT/'fig_coverage_diagnostic.pdf'} and .png  ({TEXTWIDTH_IN}in wide)")
    print(f"wrote {DERIVED/'fig_coverage_diagnostic_values.csv'}  ({len(rows)} rows)")
    und = [r for r in rows if r["panel"] == "a" and r["n_used"] < r["n_total"]]
    for r in und:
        print(f"  partial: {r['configuration']} / {r['quantity']}: "
              f"n = {r['n_used']} of {r['n_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

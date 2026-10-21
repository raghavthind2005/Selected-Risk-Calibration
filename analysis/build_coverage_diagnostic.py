#!/usr/bin/env python3
"""Per-fit and per-alpha tables of the coverage diagnostic, from the stored fits.

Each stored fit is one (configuration, repetition) of the diagnostic. Stored values
are floats, so each is converted to the exact decimal it stands for before any
arithmetic; the CSVs carry the exact value.

writes derived/coverage_diagnostic_fits.csv, derived/alpha_sweep.csv
"""
from __future__ import annotations
import csv, json, sys, pathlib
from fractions import Fraction as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import exact                                            # noqa: E402
from _paths import COVERAGE, DERIVED                              # noqa: E402

REPS = (0, 101, 102, 103, 104)
N_RANDOM = 200          # test_clean.csv holds 200 held-out rows on both benchmarks

CONFIGS = [
    ("methane:symbolic:scalar",     "methane / symbolic / scalar"),
    ("methane:mlp:scalar",          "methane / MLP / scalar"),
    ("methane:mlp:normalized",      "methane / MLP / normalized"),
    ("methane:symbolic:normalized", "methane / symbolic / normalized"),
    ("williams:symbolic",           "Williams--Otto / symbolic / scalar"),
    ("williams:mlp",                "Williams--Otto / MLP / scalar"),
]


def fit_path(key: str, rep: int) -> pathlib.Path:
    bench, rest = key.split(":", 1)
    if bench == "methane":
        surrogate, margin = rest.split(":")
        return COVERAGE / "methane" / f"draw{rep}" / f"coverage_{surrogate}_{margin}.json"
    return COVERAGE / "williams" / f"draw{rep}" / f"coverage_wo_{rest}.json"


rows = []
for key, label in CONFIGS:
    for rep in REPS:
        path = fit_path(key, rep)
        if not path.exists():
            raise SystemExit(f"missing stored fit: {path}")
        j = json.load(open(path))
        n_matched = int(j.get("n_matched") or 0)
        matched_raw = j.get("mechanism_matched_boundary_coverage")
        defined = n_matched > 0 and matched_raw is not None and matched_raw == matched_raw
        xstar = exact(j["reported_selected_coverage"])
        matched = exact(matched_raw) if defined else None
        rows.append(dict(configuration=label, repetition=rep,
                         random_coverage=exact(j["sanity_random_coverage"]),
                         n_random=N_RANDOM,
                         xstar_coverage=xstar, n_xstar=int(j["n_selected"]),
                         matched_coverage=matched, n_matched=n_matched,
                         drop_matched_minus_xstar=(matched - xstar) if defined else None))

with open(DERIVED / "coverage_diagnostic_fits.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["configuration", "repetition", "random_coverage", "n_random",
                "xstar_coverage", "n_xstar", "matched_coverage", "n_matched",
                "drop_matched_minus_xstar"])
    for r in rows:
        w.writerow([r["configuration"], r["repetition"], r["random_coverage"],
                    r["n_random"], r["xstar_coverage"], r["n_xstar"],
                    "undefined" if r["matched_coverage"] is None else r["matched_coverage"],
                    r["n_matched"],
                    "undefined" if r["drop_matched_minus_xstar"] is None
                    else r["drop_matched_minus_xstar"]])

sweep = []
for surrogate in ("symbolic", "mlp"):
    for rep in REPS:
        j = json.load(open(COVERAGE / "alpha_sweep" / f"draw{rep}" / f"alpha_sweep_{surrogate}.json"))
        for r in j["rows"]:
            sweep.append(dict(surrogate=surrogate, repetition=rep,
                              alpha=exact(r["alpha"]), target=exact(r["target"]),
                              sanity=exact(r["sanity"]), selected=exact(r["selected"]),
                              gap=exact(r["target"]) - exact(r["selected"]),
                              objective=exact(r["obj"]), n=int(r["n"])))
alphas = sorted({r["alpha"] for r in sweep}, reverse=True)

agg = []
for surrogate in ("symbolic", "mlp"):
    for a in alphas:
        sub = [r for r in sweep if r["surrogate"] == surrogate and r["alpha"] == a]
        row = dict(surrogate=surrogate, alpha=a, target=1 - a, n_reps=len(sub))
        for name in ("sanity", "selected", "gap", "objective"):
            vs = [r[name] for r in sub]
            row[f"{name}_min"], row[f"{name}_max"] = min(vs), max(vs)
        agg.append(row)

with open(DERIVED / "alpha_sweep.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(list(agg[0].keys()))
    for r in agg:
        w.writerow([str(v) for v in r.values()])

nd = sum(1 for r in rows if r["drop_matched_minus_xstar"] is None)
pos = sum(1 for r in rows if r["drop_matched_minus_xstar"] is not None
          and r["drop_matched_minus_xstar"] > 0)
print(f"coverage_diagnostic_fits.csv: {len(rows)} fits "
      f"({nd} matched-boundary undefined, {pos} with a positive drop)")
print(f"alpha_sweep.csv: {len(agg)} rows over {len(alphas)} values of alpha")

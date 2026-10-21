#!/usr/bin/env python3
"""Violation rates at matched cost: per seed, paired, and pooled.

Reads the export rows (frontier.csv filtered to the matched-cost readouts;
per-objective masks from objective_level.csv.gz) and the extended-refinement
top-up, and writes three CSVs.

Admissibility rule:
  "equal-cost"    both readouts have match_status == "matched"
  "conservative"  they do not, but the method's cost <= the peer's cost
  anything else   refused; the seed does not enter the cell

writes: derived/matched_cost_per_seed.csv   one row per benchmark x budget x arm x seed
        derived/matched_cost_paired.csv     method vs baseline on admissible seeds,
                                            exact two-sided McNemar on objectives
                                            deployed by both, markers for p<0.05 and
                                            for Bonferroni over the table
        derived/matched_cost_pooled.csv     mean of per-seed rates per cell, with the
                                            seed range and a Clopper-Pearson interval
"""
from __future__ import annotations
import collections, csv, gzip, json, math, pathlib, sys
from scipy.stats import beta as _beta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _paths import EXPORT, MATCHED_COST, DERIVED            # noqa: E402
OUT = DERIVED

from scipy.stats import binomtest                          # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fmt import half_up, mean_fraction, rate_fraction, ROUNDING_RULE  # noqa: E402
import _config                                              # noqa: E402
from fractions import Fraction                               # noqa: E402

# The comparison set comes from config/reported_arms.json and nowhere else.
_CFG = _config.study("matched_cost_table")
BENCH = tuple(_CFG["benchmarks"])
BUDGETS = tuple(_CFG["budgets"])                  # .25 / .55 / 1.0
METHOD = _CFG["method"]
METHOD_LABEL = _config.load()["method_label"]
ARMS = list(_CFG["baselines"])
SEEDS = tuple(_CFG["seeds"])


def fnum(s):
    try: return float(s)
    except Exception: return float("nan")


def load_readouts() -> dict:
    """(bench, seed, cap, arm) -> readout dict. Same filter as t05_load.py."""
    ro = {}
    with open(EXPORT / "frontier.csv") as fh:
        for r in csv.DictReader(fh):
            if r["benchmark"] not in BENCH:
                continue
            if r["is_readout"] != "True":
                continue
            k = (r["benchmark"], int(r["seed"]), r["readout_cap"], r["arm"])
            assert k not in ro, k
            ro[k] = dict(cost=fnum(r["cost"]), risk=fnum(r["risk"]),
                         nv=int(fnum(r["n_viol"])) if r["n_viol"] not in ("", "None") else None,
                         nd=int(fnum(r["n_deployed"])) if r["n_deployed"] not in ("", "None") else None,
                         ms=r["match_status"], tgt=fnum(r["derived_cap_target"]))
    return ro


def load_masks() -> dict:
    """(bench, seed, cap, arm) -> {objective_id: (viol, dep)}. Same filter as t05_masks.py."""
    masks: dict = {}
    with gzip.open(EXPORT / "objective_level.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["source_kind"] != "matched_cost_frontier_mask":
                continue
            if r["reported_run"] != "True":
                continue
            ex = json.loads(r["extra_json"])
            fk = ex.get("frontier_key", "")
            if not fk.startswith("READOUT|") or ex.get("run_variant") != "stage2":
                continue
            _, arm, cap = fk.split("|")
            d = masks.setdefault((r["benchmark"], int(r["seed"]), cap, arm), {})
            dep = ex.get("dep_mask")
            d[r["objective_id"]] = (
                r["violation_fail_closed"] in ("True", "1", "true"),
                dep in (True, "True", "1", "true", None))
    return masks


def exact_rate(d) -> Fraction | None:
    """A cell's risk as an exact Fraction, n_viol / n_deployed.

    The stored `risk` float is checked against it; a mismatch would mean the
    denominator is not n_deployed and is reported rather than absorbed.
    """
    if d is None or d.get("nv") is None or not d.get("nd"):
        return None
    fr = rate_fraction(d["nv"], d["nd"])
    if d.get("risk") == d["risk"] and abs(float(fr) - d["risk"]) > 1e-9:
        raise AssertionError(
            f"stored risk {d['risk']!r} != n_viol/n_deployed "
            f"{d['nv']}/{d['nd']} = {float(fr)!r}")
    return fr


def pooled_mean_inputs(pooled_rows, bench, arm, cap):
    """The rates that enter a pooled cell.

    The pooling rule is the mean of per-seed rates over admissible seeds. A
    baseline's cell therefore uses only the seeds on which the pair was
    admissible; averaging over all five would credit a baseline on seeds whose
    comparison is refused. The method's own row has no pair to be admissible
    against, so it uses every seed whose readout exists.
    """
    rs = [r for r in pooled_rows if r["benchmark"] == bench and r["arm"] == arm
          and r["budget"] == cap and r["rate_exact"] != ""]
    if arm != METHOD:
        rs = [r for r in rs if r["admissible_vs_method"] not in ("None", "")]
    return [Fraction(r["rate_exact"]) for r in rs]


def admissible(m, p):
    if m is None or p is None: return None
    if m["ms"] == "matched" and p["ms"] == "matched": return "equal-cost"
    if m["cost"] <= p["cost"] * (1 + 1e-12): return "conservative"
    return None


def b1_overrides() -> dict:
    """Extended-refinement readouts, keyed like load_readouts()."""
    ov = {}
    root = MATCHED_COST / "extended_refinement"
    for d in sorted(root.glob("*/seed*.arms.json")) if root.exists() else []:
        j = json.load(open(d))
        seed = int(j["seed"])
        for key, v in j["frontier"].items():
            if not key.startswith("READOUT|"):
                continue
            _, arm, cap = key.split("|")
            ov[("methane", seed, cap, arm)] = dict(
                cost=v["cost"], risk=v["risk"], nv=v["n_viol"],
                nd=v["n_deployed"], ms=v["match_status"], tgt=v["target_cost"],
                _from_topup=True, _viol_mask=v.get("viol_mask"), _dep_mask=v.get("dep_mask"))
    return ov


def main() -> int:
    ro = load_readouts()
    ov = b1_overrides()
    n_ov = 0
    for k, v in ov.items():
        if k in ro and (ro[k]["ms"] != v["ms"] or ro[k]["cost"] != v["cost"]):
            n_ov += 1
        ro[k] = v
    masks = load_masks()

    paired_rows, pooled_rows = [], []
    pvals = []
    for bench in BENCH:
        for arm in ARMS:
            for cap in BUDGETS:
                adm_seeds, kinds = [], []
                B = C = 0
                mr_seed, pr_seed = [], []
                for s in SEEDS:
                    m = ro.get((bench, s, cap, METHOD))
                    p = ro.get((bench, s, cap, arm))
                    kind = admissible(m, p)
                    if kind is None:
                        continue
                    adm_seeds.append(s); kinds.append(kind)
                    mr_seed.append(exact_rate(m)); pr_seed.append(exact_rate(p))
                    mm = masks.get((bench, s, cap, METHOD)) or {}
                    pp = masks.get((bench, s, cap, arm)) or {}
                    if m.get("_from_topup") and m.get("_viol_mask"):
                        mm = {str(i): (bool(x), bool(y)) for i, (x, y)
                              in enumerate(zip(m["_viol_mask"], m["_dep_mask"]))}
                    if p.get("_from_topup") and p.get("_viol_mask"):
                        pp = {str(i): (bool(x), bool(y)) for i, (x, y)
                              in enumerate(zip(p["_viol_mask"], p["_dep_mask"]))}
                    for oid in set(mm) & set(pp):
                        mv, md = mm[oid]; pv, pd = pp[oid]
                        if not (md and pd):
                            continue          # deployed by BOTH only
                        if mv and not pv: B += 1
                        elif pv and not mv: C += 1
                pv_ = binomtest(B, B + C, 0.5).pvalue if B + C > 0 else 1.0
                if adm_seeds:
                    pvals.append(pv_)
                paired_rows.append({
                    "benchmark": bench, "budget": cap, "baseline": arm,
                    "method_rate_exact": (str(mean_fraction(mr_seed)) if mr_seed else ""),
                    "baseline_rate_exact": (str(mean_fraction(pr_seed)) if pr_seed else ""),
                    "method_rate": (half_up(mean_fraction(mr_seed), 3) if mr_seed else ""),
                    "baseline_rate": (half_up(mean_fraction(pr_seed), 3) if pr_seed else ""),
                    "n_seeds": len(adm_seeds),
                    "seeds": ";".join(map(str, adm_seeds)),
                    "admissibility": ";".join(kinds),
                    "b_method_only_fails": B, "c_peer_only_fails": C,
                    "n_discordant": B + C,
                    "mcnemar_p_exact_two_sided": (f"{pv_:.6g}" if adm_seeds else ""),
                })
                for s in SEEDS:
                    p = ro.get((bench, s, cap, arm))
                    m = ro.get((bench, s, cap, METHOD))
                    pooled_rows.append({
                        "benchmark": bench, "budget": cap, "arm": arm, "seed": s,
                        "rate": (p["risk"] if p else ""),
                        "rate_exact": (str(exact_rate(p)) if exact_rate(p) is not None else ""),
                        "n_viol": (p["nv"] if p else ""), "n_deployed": (p["nd"] if p else ""),
                        "match_status": (p["ms"] if p else ""),
                        "admissible_vs_method": str(admissible(m, p)),
                        "from_extended_refinement": str(bool(p.get("_from_topup"))) if p else "",
                    })
        for cap in BUDGETS:
            for s in SEEDS:
                m = ro.get((bench, s, cap, METHOD))
                pooled_rows.append({
                    "benchmark": bench, "budget": cap, "arm": METHOD, "seed": s,
                    "rate": (m["risk"] if m else ""),
                    "rate_exact": (str(exact_rate(m)) if exact_rate(m) is not None else ""),
                    "n_viol": (m["nv"] if m else ""), "n_deployed": (m["nd"] if m else ""),
                    "match_status": (m["ms"] if m else ""),
                    "admissible_vs_method": "", "from_extended_refinement": str(bool(m.get("_from_topup"))) if m else "",
                })

    n_tests = sum(1 for r in paired_rows if r["n_seeds"] > 0)
    bonf = 0.05 / max(n_tests, 1)
    for r in paired_rows:
        p = float(r["mcnemar_p_exact_two_sided"]) if r["mcnemar_p_exact_two_sided"] else None
        r["sig_p05"] = "" if p is None else ("*" if p < 0.05 else "")
        r["sig_bonferroni"] = "" if p is None else ("**" if p < bonf else "")
        r["bonferroni_alpha"] = f"{bonf:.3e}"
        r["n_tests_in_table"] = n_tests

    pc = ["benchmark", "budget", "baseline", "method_rate", "baseline_rate",
          "method_rate_exact", "baseline_rate_exact",
          "n_seeds", "seeds", "admissibility", "b_method_only_fails",
          "c_peer_only_fails", "n_discordant", "mcnemar_p_exact_two_sided",
          "sig_p05", "sig_bonferroni", "bonferroni_alpha", "n_tests_in_table"]
    with open(OUT / "matched_cost_paired.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pc); w.writeheader(); w.writerows(paired_rows)
    lc = ["benchmark", "budget", "arm", "seed", "rate", "rate_exact", "n_viol", "n_deployed",
          "match_status", "admissible_vs_method", "from_extended_refinement"]
    with open(OUT / "matched_cost_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=lc); w.writeheader(); w.writerows(pooled_rows)

    write_pooled(pooled_rows, paired_rows)
    return 0


def clopper_pearson(v: int, n: int):
    """Two-sided 95% interval on a binomial proportion."""
    if n == 0:
        return "", ""
    lo = 0.0 if v == 0 else _beta.ppf(0.025, v, n - v + 1)
    hi = 1.0 if v == n else _beta.ppf(0.975, v + 1, n - v)
    return f"{lo:.6f}", f"{hi:.6f}"


def write_pooled(pooled_rows, paired_rows):
    """One row per benchmark x budget x arm: the pooled rate and its spread."""
    by = collections.defaultdict(list)
    for r in pooled_rows:
        by[(r["benchmark"], r["budget"], r["arm"])].append(r)
    out = []
    for bench in BENCH:
        for cap in BUDGETS:
            for arm in [METHOD] + ARMS:
                rs = by.get((bench, cap, arm), [])
                adm = [r for r in rs if arm == METHOD
                       or r["admissible_vs_method"] in ("equal-cost", "conservative")]
                row = dict(benchmark=bench, budget=cap, arm=arm,
                           role="method (reference)" if arm == METHOD else "baseline")
                if not adm:
                    row.update(pooled_rate_exact="", n_admissible_seeds="0",
                               admissible_seeds="", min_seed_rate="", max_seed_rate="",
                               pooled_objectives_v="0", pooled_objectives_n="0",
                               cp95_lo="", cp95_hi="")
                    out.append(row)
                    continue
                rates = [Fraction(r["rate_exact"]) for r in adm]
                mean = mean_fraction(rates)
                v = sum(int(r["n_viol"]) for r in adm)
                n = sum(int(r["n_deployed"]) for r in adm)
                lo, hi = clopper_pearson(v, n)
                row.update(pooled_rate_exact=f"{mean.numerator}/{mean.denominator}",
                           n_admissible_seeds=str(len(adm)),
                           admissible_seeds=";".join(str(r["seed"]) for r in adm),
                           min_seed_rate=f"{float(min(rates)):.4f}",
                           max_seed_rate=f"{float(max(rates)):.4f}",
                           pooled_objectives_v=str(v), pooled_objectives_n=str(n),
                           cp95_lo=lo, cp95_hi=hi)
                out.append(row)
    cols = ["benchmark", "budget", "arm", "role", "pooled_rate_exact",
            "n_admissible_seeds", "admissible_seeds", "min_seed_rate", "max_seed_rate",
            "pooled_objectives_v", "pooled_objectives_n", "cp95_lo", "cp95_hi"]
    with open(OUT / "matched_cost_pooled.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"  wrote matched_cost_pooled.csv ({len(out)} cells)")


if __name__ == "__main__":
    raise SystemExit(main())

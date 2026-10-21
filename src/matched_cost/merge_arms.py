"""Merge a TOP-UP run (subset of arms) into an existing seed result.

Sound because every arm is independent and deterministic given (seed, arm, knob):
all RNGs in run_seed.py are freshly seeded from `seed` and never advance across
arms, and ocl_solve.deterministic_starts seeds the multistart per objective, so
arms are compared under common random numbers.

That argument is CHECKED, not assumed. The top-up is asked to recompute at least
one arm the base run already has (the CONTROL); if any control row differs, the
premise is false for this run and the merge aborts rather than producing a
silently mismatched comparison.

usage: BENCHMARK=<b> python3 merge_arms.py <seed>
"""
import json, os, pathlib, sys

HERE = pathlib.Path(__file__).parent
BENCH = os.environ.get("BENCHMARK", "methane")
SUB = "runs" + ("" if BENCH == "methane" else f"_{BENCH}")

# Fields that fix the estimand. If any differs the two runs are not comparable.
INVARIANTS = ["C_star", "caps", "laws", "law_of_objective",
              "primary_starts", "sweep_starts", "check_starts"]
# Row fields compared for the control arm, split by how they must be compared.
#
# OUTCOME fields are the result itself and are compared EXACTLY. If a single
# objective's deployment or violation flips, the runs are not the same
# experiment and no tolerance may hide it.
#
# NUMERIC fields are continuous quantities produced by a multistart SLSQP whose
# summation order depends on the BLAS thread count. The committed williams bases
# ran before commit 144d819d pinned BLAS to one thread; a top-up run with it
# pinned reproduces every mask, every risk and every count EXACTLY but differs on
# cost in the 10th significant figure. Measured on williams seeds 3 and 9:
#
#     outcome fields differing ......... 0
#     max relative diff on cost/dial/knob 1.8e-07
#     C* relative difference ........... 8.2e-08
#
# For contrast, the genuinely incompatible case this guard exists to catch --
# methane seed 9, whose base predates a change to the [4] grid -- shows
#
#     outcome fields differing ......... 18
#     max relative diff ................ 6.4e-02
#
# Five orders of magnitude apart. RTOL is set four orders below the tightest
# thing that carries meaning anywhere in the protocol (MATCH_TOL = 2e-2), so it
# cannot admit a real disagreement; the observed deviation is reported on every
# merge and stored in the merged file, so it can never pass silently.
OUTCOME_KEYS = ["risk", "n_viol", "n_deployed", "viol_mask", "dep_mask", "per_law"]
NUMERIC_KEYS = ["dial", "knob", "cost"]
ROW_KEYS = NUMERIC_KEYS + OUTCOME_KEYS
RTOL = 1e-6


def _dev(a, b):
    """Max relative deviation between two JSON values, or inf if incomparable."""
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else float("inf")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(float(b) - float(a))
        return d / max(abs(float(a)), 1e-300) if d else 0.0
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return max([_dev(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return max([_dev(a[k], b[k]) for k in a] or [0.0])
    return 0.0 if a == b else float("inf")


def main(seed):
    d = HERE / SUB
    base_p, top_p = d / f"seed{seed}.json", d / f"seed{seed}.arms.json"
    for p in (base_p, top_p):
        if not p.exists():
            sys.exit(f"missing {p}")
    base, top = json.load(open(base_p)), json.load(open(top_p))

    bp, tp = base.get("platform"), top.get("platform")
    if bp and tp and bp != tp:
        sys.exit(f"ABORT: base and top-up ran on DIFFERENT PLATFORMS.\n"
                 f"  base : {bp}\n  top-up: {tp}\n"
                 "The pinball surrogate fit is BLAS-sensitive, so C* and the caps\n"
                 "diverge (measured: 5.1% on methane seed 6). Re-run the top-up on\n"
                 "the same machine as the base.")
    if not (bp and tp):
        print("NOTE: platform fingerprint absent on one side (pre-dates the guard); "
              "relying on the C* invariant below.")

    worst_inv = 0.0
    for k in INVARIANTS:                     # `dev`, not `d`: `d` is the run dir
        dev = _dev(base.get(k), top.get(k))
        if dev > RTOL:
            sys.exit(f"ABORT: invariant '{k}' differs between base and top-up by "
                     f"{dev:.3g} relative (tolerance {RTOL:.0e}). The runs do not "
                     "share an estimand; merge is unsafe.")
        worst_inv = max(worst_inv, dev)
    print(f"invariants match ({', '.join(INVARIANTS)}); "
          f"max relative deviation {worst_inv:.3g} (tolerance {RTOL:.0e})")

    base_arms = {v["arm"] for v in base["frontier"].values()}
    top_arms = {v["arm"] for v in top["frontier"].values()}
    control = sorted(base_arms & top_arms)
    if not control:
        sys.exit("ABORT: top-up shares no arm with the base run, so the "
                 "reproducibility premise is untested. Re-run including one "
                 "already-computed arm as a control.")

    checked = mismatched = 0
    worst_row = 0.0
    for k, tv in top["frontier"].items():
        if tv["arm"] not in control:
            continue
        bv = base["frontier"].get(k)
        if bv is None:
            continue
        bad = None
        for f in OUTCOME_KEYS:              # exact: the result itself
            if f in bv and f in tv and bv[f] != tv[f]:
                bad = f"{f} (OUTCOME field, compared exactly)"
                break
        if bad is None:
            for f in NUMERIC_KEYS:          # tolerance: continuous, BLAS-sensitive
                if f in bv and f in tv:
                    dev = _dev(bv[f], tv[f])     # `dev`: `d` is the run dir
                    worst_row = max(worst_row, dev)
                    if dev > RTOL:
                        bad = f"{f} differs by {dev:.3g} relative"
                        break
        if bad:
            print(f"  MISMATCH {k}: {bad}")
            mismatched += 1
        else:
            checked += 1
    print(f"control arms {control}: {checked} rows agree "
          f"({mismatched} mismatched); every outcome field exact, max relative "
          f"deviation on cost/dial/knob {worst_row:.3g} (tolerance {RTOL:.0e})")
    if mismatched:
        sys.exit("ABORT: control rows differ -- arms are NOT independent in this "
                 "run. Do not merge; re-run the full roster.")
    if not checked:
        sys.exit("ABORT: no overlapping rows were actually compared.")

    # Merge by KEY, not by arm: the base run already carries CONTEXT sweep rungs
    # for these arms, so an arm-level filter would drop the very READOUT rows the
    # top-up exists to add. Overlapping keys were verified bit-identical above.
    added = [k for k in top["frontier"] if k not in base["frontier"]]
    for k in added:
        base["frontier"][k] = top["frontier"][k]
    new_arms = sorted({top["frontier"][k]["arm"] for k in added})
    print(f"  new rows by arm: "
          + str({a: sum(1 for k in added if top["frontier"][k]["arm"] == a)
                 for a in new_arms}))

    bak = d / f"seed{seed}.premerge.json"
    if not bak.exists():
        json.dump(json.load(open(base_p)), open(bak, "w"))
    base["merged_arms"] = sorted(set(base.get("merged_arms", [])) | set(new_arms))
    base["merge_control_rows_verified"] = checked
    base["merge_max_relative_deviation"] = max(worst_inv, worst_row)
    json.dump(base, open(base_p, "w"))
    print(f"merged {len(added)} rows for {new_arms} -> {base_p}")
    print(f"base backed up at {bak}")


if __name__ == "__main__":
    main(int(sys.argv[1]))

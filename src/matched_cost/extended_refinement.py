#!/usr/bin/env python3
"""Extended refinement -- raise only `REFINE_ITERS`, nothing else.

Imports run_seed.py, asserts that every quantity the readout depends on is
exactly the one the base runs used, sets `REFINE_ITERS` to the extended value,
and calls `run_seed.main()`.

`REFINE_ITERS` is the iteration budget of the root-find that drives a baseline's
knob to the readout's cost target. Raising it cannot change any baseline's
risk at a given cost, because the fit, the calibration, the correction, C*, the
knob grid and the frontier sweep are bit-identical; only the cost-match search
is given more steps. The loop bisects once a bracket exists, so each extra
iteration at worst halves the cost error, and it is self-limiting: it breaks on
tolerance, on a stalled dial, and on knob saturation.

Four conditions are enforced rather than assumed:

* It does not touch the method's rows. An arms-restricted run emits no
  `law_blind_envelope` frontier key at all, so `ARMS` is required here and
  naming the method is refused.
* It does not change the readout rule. `REFINE_TOL` and `analyze.MATCH_TOL` are
  asserted to be 0.02, so "matched" means what the matched-cost table means.
* It does not resume. run_seed.py's resume gate hashes its own bytes, which this
  file does not change, so a checkpoint from a shorter pass would pass that gate
  and splice into a longer one. `RESUME` is refused outright.
* It does not write into the directory holding the base runs.

usage:
    BENCHMARK=methane ARMS=scalar python3 extended_refinement.py <seed>

writes: runs/seed<N>.arms.json          (run_seed's own writer)
        runs/seed<N>.arms.extref.json   (a sidecar recording what was changed)
"""
from __future__ import annotations
import json, os, pathlib, sys, time

# ---- the one number this pass changes ------------------------------------
EXTENDED_REFINE_ITERS = 12

# os.path.abspath, not .resolve(): if this file is reached through a symlink,
# resolving it would put the original directory on sys.path instead of this one.
HERE = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")


def refuse(why: str) -> None:
    print(f"EXTENDED REFINEMENT REFUSED: {why}", file=sys.stderr)
    raise SystemExit(3)


# ---- gates that must hold BEFORE run_seed is imported --------------------
if os.environ.get("RESUME"):
    refuse("RESUME is set. run_seed.py's resume gate hashes its own "
           "bytes, which this file does not change, so a REFINE_ITERS=4 "
           "checkpoint would pass the gate and mix two estimands.")
if os.environ.get("DRY"):
    refuse("DRY is set; extended refinement produces reported numbers.")
_arms = os.environ.get("ARMS")
if not _arms:
    refuse("ARMS is required. An unrestricted run would recompute the "
           "method's own readout rows at a refinement budget the published "
           "table did not use.")
_armlist = [a.strip() for a in _arms.split(",") if a.strip()]
if not _armlist:
    refuse("ARMS is empty.")
# Several arms per run is allowed. The fit, the calibration, the correction and
# C* depend on the seed, not on the arm, so batching the arms of one seed does
# that shared work once; and an arm's rows do not depend on which other arms
# were selected -- a four-arm restricted run reproduces the eleven-arm run's
# readout rows byte for byte, every float by repr(), both 200-entry masks and
# the per-family rates, with identical fit, C* and cap targets. Every frontier
# key names its own arm and the sidecar records the arm list.
if "law_blind_envelope" in _armlist:
    refuse("ARMS names the method: extended "
           "refinement never touches the method's rows.")
if "alpha_tightening" in _armlist:
    refuse("alpha_tightening's knob is an order statistic "
           "(run_seed.py:983-986, np.quantile(..., method='higher')) and "
           "saturates at min(t,1.0), so its unmatched cells are a structural "
           "step gap: more iterations cannot reach between two adjacent order "
           "statistics.")

import run_seed as R                                     # noqa: E402
import analyze as A                                      # noqa: E402

# ---- everything the readout depends on must be the committed value -------
HELD = {
    "REFINE_TOL":      (R.REFINE_TOL, 0.02),
    "MATCH_TOL":       (A.MATCH_TOL, 0.02),
    "ALPHA":           (R.ALPHA, 0.10),
    "FIT_TAU":         (R.FIT_TAU, 0.90),
    "N_FIT":           (R.N_FIT, 1000),
    "N_CAL":           (R.N_CAL, 300),
    "N_ENV":           (R.N_ENV, 400),
    "N_SANE":          (R.N_SANE, 800),
    "N_PROP_PL":       (R.N_PROP_PL, 20),
    "N_PERF_PL":       (R.N_PERF_PL, 40),
    "PRIMARY_STARTS":  (R.PRIMARY_STARTS, 60),
    "CHECK_STARTS":    (R.CHECK_STARTS, 120),
    "N_SPAN_RUNGS":    (R.N_SPAN_RUNGS, 3),
    "LAWS":            (list(R.LAWS), ["random_linear", "corner_pressure",
                                       "boundary_pressure",
                                       "residual_hotspot_pressure",
                                       "boundary_residual_pressure"]),
    "CAP_FRACTIONS":   (list(R.CAP_FRACTIONS), [0.25, 0.40, 0.55, 0.70, 1.00]),
    "REFINE_ITERS_as_committed": (R.REFINE_ITERS, 4),
}
bad = {k: v for k, (v, want) in HELD.items() if v != want}
if bad:
    refuse("run_seed.py/analyze.py do not carry the values the base runs "
           f"used, so this run would not be comparable: {bad}")

# run_seed writes to (its own dir)/runs. Set EXTENDED_REFINEMENT_DIR to a
# directory of your own so the base runs' outputs are left alone.
_guard = os.environ.get("EXTENDED_REFINEMENT_DIR")
if not _guard:
    refuse("set EXTENDED_REFINEMENT_DIR to the directory this pass may write "
           "into, so that the base runs' outputs are left alone.")
R.HERE = pathlib.Path(_guard).resolve()

# ---- the ONLY mutation ---------------------------------------------------
_before = R.REFINE_ITERS
R.REFINE_ITERS = EXTENDED_REFINE_ITERS

print(f"[extref] output dir     : {R.HERE.resolve()}", flush=True)
print(f"[extref] benchmark      : {R.BENCH}", flush=True)
print(f"[extref] arms           : {_armlist}"
      f"{'' if len(_armlist) == 1 else '  (batched; arm-set independence measured)'}",
      flush=True)
print(f"[extref] REFINE_ITERS   : {_before} -> {R.REFINE_ITERS}  (the only change)",
      flush=True)
print(f"[extref] REFINE_TOL     : {R.REFINE_TOL}  (held)", flush=True)
print(f"[extref] MATCH_TOL      : {A.MATCH_TOL}  (held)", flush=True)
print(f"[extref] run_seed md5   : {R._code_md5()}  (bytes unchanged)", flush=True)
print(f"[extref] {len(HELD)} held constants asserted equal to committed", flush=True)

if __name__ == "__main__":
    seed = int(sys.argv[1])
    t0 = time.time()
    try:
        R.main()
    finally:
        sub = "runs" + ("" if R.BENCH == "methane" else f"_{R.BENCH}")
        out = R.HERE / sub / f"seed{seed}.arms.extref.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "driver": "extended_refinement.py",
            "benchmark": R.BENCH, "seed": seed,
            "arm": _armlist[0] if len(_armlist) == 1 else None,
            "arms": _armlist,
            "armset_independence": "a four-arm restricted run reproduces the "
                "eleven-arm run's readout rows byte-identically, so batching "
                "arms does not change any arm's rows.",
            "refine_iters_base": _before,
            "refine_iters_used": R.REFINE_ITERS,
            "only_change": "REFINE_ITERS",
            "held_constants": {k: v for k, (v, _) in HELD.items()},
            "run_seed_code_md5": R._code_md5(),
            "output_dir": str(R.HERE.resolve()),
            "emits_method_rows": False,
            "emits_method_rows_reason":
                "ARMS restricts the roster; an arms-restricted run "
                "writes no law_blind_envelope frontier key.",
            "label_for_filled_cells": "extended refinement",
            "elapsed_s": time.time() - t0,
        }, open(out, "w"), indent=1)
        print(f"[extref] provenance -> {out}", flush=True)

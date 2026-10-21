#!/usr/bin/env python3
"""Prove that `config/reported_arms.json` actually decides the comparison set.

A config file that no script reads is decoration. This test makes the claim
falsifiable:

  1. run the analysis, keep the tables;
  2. mutate `config/reported_arms.json` -- drop a baseline from the matched-cost
     table, and drop a control from the certification study;
  3. run again. At least one table MUST change, or the config is not load-bearing;
  4. restore the config and run a third time. Every table MUST come back
     byte-identical to step 1, or the mutation left residue.

Exit code 0 only if both 3 and 4 hold. The config is restored on every exit path,
including a crash, so a failed run never leaves the repo mutated.

usage: python3 smoke/mutation_test.py [--python PY]
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
CONFIG = REPO / "config" / "reported_arms.json"
TABLES = REPO / "tables"
ANALYSIS = REPO / "analysis"


def digests() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(TABLES.glob("*")) if p.is_file()}


def run_all(py: str) -> None:
    env = dict(os.environ)
    r = subprocess.run(["bash", "run_all.sh"],
                       cwd=REPO, env=dict(env, PYTHON=py),
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise SystemExit(f"run_all.sh failed ({r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    args = ap.parse_args()
    py = args.python

    original = CONFIG.read_bytes()
    backup = pathlib.Path(tempfile.mkdtemp()) / "reported_arms.json"
    backup.write_bytes(original)
    ok = True
    try:
        print("[1] baseline run")
        run_all(py)
        before = digests()
        print(f"    {len(before)} table files")

        print("[2] mutating config/reported_arms.json")
        cfg = json.loads(original)
        dropped_baseline = cfg["studies"]["matched_cost_table"]["baselines"].pop()
        dropped_control = cfg["studies"]["certification_study"]["controls"].pop()
        CONFIG.write_text(json.dumps(cfg, indent=1))
        print(f"    dropped baseline {dropped_baseline!r} from matched_cost_table")
        print(f"    dropped control  {dropped_control!r} from certification_study")

        print("[3] run with the mutated config")
        run_all(py)
        after = digests()
        changed = sorted(k for k in before if before.get(k) != after.get(k))
        changed += sorted(set(after) - set(before))
        if changed:
            print(f"    CHANGED {len(changed)} of {len(before)} tables: {', '.join(changed)}")
        else:
            print("    FAIL: no table changed -- the config is NOT load-bearing")
            ok = False

        print("[4] restore the config and run again")
        CONFIG.write_bytes(original)
        run_all(py)
        restored = digests()
        drift = sorted(k for k in before if before.get(k) != restored.get(k))
        if drift:
            print(f"    FAIL: {len(drift)} table(s) did not return byte-identical: {', '.join(drift)}")
            ok = False
        else:
            print(f"    all {len(before)} tables byte-identical to the baseline run")
    finally:
        CONFIG.write_bytes(backup.read_bytes())
        shutil.rmtree(backup.parent, ignore_errors=True)

    print("\nMUTATION TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

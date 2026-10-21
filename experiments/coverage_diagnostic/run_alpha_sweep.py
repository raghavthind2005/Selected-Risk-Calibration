#!/usr/bin/env python3
"""Run the methane alpha sweep at a chosen seed draw.

Same discipline as run_coverage_fit.py: reads `alpha_sweep_methane.py`, applies
four declared substitutions, checks the diff (each pattern exactly once, every
changed line a declared pair, exactly four lines changed), and refuses otherwise.
Three substitutions are seeds; the fourth is the output path.

The solver's own stream is deliberately not substituted: it is derived from the
cost vector's contents and so follows SEED_COST.

usage:  python3 run_alpha_sweep.py --draw <d> [<symbolic|mlp> ...]
"""
from __future__ import annotations
import argparse, difflib, json, os, pathlib, sys

HERE = pathlib.Path(os.path.abspath(__file__)).parent
REPO = HERE.parent.parent
SRC = HERE / "alpha_sweep_methane.py"
OUTROOT = pathlib.Path(os.environ.get(
    "ALPHA_SWEEP_OUT", str(REPO / "rerun" / "coverage_diagnostic" / "alpha_sweep")))

ORIGINAL_SEEDS = {"global": 0, "mlp_h": 0, "cost": 0}
CONFIGS = ("symbolic", "mlp")

SUBS = [
    ("rng_global = np.random.default_rng(0)",
     "rng_global = np.random.default_rng(SEED_GLOBAL)", "seed:global"),
    ("max_iter=4000, random_state=0).fit(_scaler.transform(Xtr), ytr)",
     "max_iter=4000, random_state=SEED_MLP_H).fit(_scaler.transform(Xtr), ytr)",
     "seed:mlp_h"),
    ("COST = make_cost_vectors(N_COST, seed=0)",
     "COST = make_cost_vectors(N_COST, seed=SEED_COST)", "seed:cost"),
    ('fn = f"{RUN}/alpha_sweep_{SURROGATE}.json"',
     'fn = f"{OUT_DIR}/alpha_sweep_{SURROGATE}.json"',
     "output-path (NOT a seed; disclosed)"),
]


def seeds_for_draw(d: int) -> dict:
    return {k: v + d for k, v in ORIGINAL_SEEDS.items()}


def rewrite():
    orig = SRC.read_text(); out = orig
    for old, new, what in SUBS:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"ALPHA SWEEP REFUSED: '{what}' matches {n} times, not "
                             f"once.\n  pattern: {old!r}\nalpha_sweep.py has changed; "
                             f"the wrapper must be re-derived, not guessed.")
        out = out.replace(old, new)
    diff = list(difflib.unified_diff(orig.splitlines(), out.splitlines(),
                                     "alpha_sweep.py (committed)",
                                     "rewritten", lineterm="", n=0))
    rem = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
    add = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    for l in rem:
        if not any(o in l for o in (o for o, _, _ in SUBS)):
            raise SystemExit(f"ALPHA SWEEP REFUSED: undeclared line removed:\n  {l}")
    for l in add:
        if not any(n in l for n in (n for _, n, _ in SUBS)):
            raise SystemExit(f"ALPHA SWEEP REFUSED: undeclared line added:\n  {l}")
    if len(rem) != len(SUBS) or len(add) != len(SUBS):
        raise SystemExit(f"ALPHA SWEEP REFUSED: diff changes {len(rem)} lines, "
                         f"expected exactly {len(SUBS)}")
    return orig, out, "\n".join(diff)


def run(cfg: str, seeds: dict, outdir: pathlib.Path, src: str) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    g = {"__name__": "__main__", "__file__": str(SRC),
         "SEED_GLOBAL": seeds["global"], "SEED_MLP_H": seeds["mlp_h"],
         "SEED_COST": seeds["cost"], "OUT_DIR": str(outdir)}
    argv = sys.argv
    sys.argv = ["alpha_sweep.py", cfg]
    try:
        exec(compile(src, str(SRC), "exec"), g)
    finally:
        sys.argv = argv
    return outdir / f"alpha_sweep_{cfg}.json"



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int)
    ap.add_argument("configs", nargs="*", default=[])
    a = ap.parse_args()
    orig, src, diff = rewrite()
    print(f"{len(SUBS)} declared substitutions, diff touches exactly "
          f"{len(SUBS)} lines, no undeclared change", flush=True)
    cfgs = a.configs or list(CONFIGS)
    for c in cfgs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c!r}; choose from {list(CONFIGS)}")
    draws = [a.draw] if a.draw is not None else None
    if draws is None:
        raise SystemExit("give --draw <N>")
    for d in draws:
        seeds = seeds_for_draw(d)
        outdir = OUTROOT / f"draw{d}"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "wrapper_diff.patch").write_text(diff + "\n")
        for c in cfgs:
            out = outdir / f"alpha_sweep_{c}.json"
            if out.exists():
                print(f"  SKIP (exists) {c} draw {d}", flush=True); continue
            print(f"\n=== alpha_sweep {c}  draw {d}  seeds {seeds} ===", flush=True)
            p = run(c, seeds, outdir, src)
            print(f"  -> {p}", flush=True)
        json.dump({"draw": d, "seeds": seeds, "configs": cfgs},
                  open(outdir / "draw_seeds.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

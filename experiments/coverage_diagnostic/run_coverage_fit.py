#!/usr/bin/env python3
"""Run the coverage experiment at a chosen set of seeds.

`coverage_experiment_<benchmark>.py` is a top-level script: it has no main(), no
seed parameter, and five seed literals in its body. This wrapper reads that
file's source, applies a declared set of textual substitutions that replace those
literals and the output path, checks that the diff between the original and the
rewritten source touches only the declared lines, and execs the result. The diff
is written next to the output.

The wrapper refuses if any declared pattern does not appear exactly once, or if
the diff contains a changed line that is not one of the declared pairs.

The six substitutions:

    rng_global = np.random.default_rng(0)                  -> SEED_GLOBAL
    MLPRegressor(... random_state=0).fit(scaled Xtr, ytr)  -> SEED_MLP_H
    GradientBoostingRegressor(... random_state=0)          -> SEED_GBM
    MLPRegressor(... random_state=1).fit(_us..., _abs_tr)  -> SEED_MLP_U
    make_cost_vectors(N_COST, seed=0)                      -> SEED_COST
    fn = f"{RUN}/coverage_...json"                         -> OUT_DIR

The last is the output path, not a seed: it cannot change any number, being the
argument of the final json.dump.

One seed site is deliberately not substituted: the experiment's solver draws its
restarts from `np.random.default_rng(int(abs(cv.sum()) * 1e4) % 2**31)`, derived
from the cost vector's contents. Changing SEED_COST changes the cost vectors and
so that stream follows automatically; patching it too would decouple the solver
stream from the objectives it belongs to.

A draw shifts every seed by the draw number, so draw 0 is the experiment at its
own seeds. The five repetitions the diagnostic reports are draws 0, 101, 102,
103 and 104.

usage:
  BENCHMARK=methane python3 run_coverage_fit.py --draw 0 [<config> ...]
      where <config> is one of
        symbolic:scalar  symbolic:normalized  mlp:scalar  mlp:normalized
      and Williams-Otto takes symbolic and mlp
"""
from __future__ import annotations
import argparse, difflib, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(os.path.abspath(__file__)).parent
REPO = HERE.parent.parent
BENCH = os.environ.get("BENCHMARK", "methane")            # methane | williams
if BENCH not in ("methane", "williams"):
    raise SystemExit(
        f"BENCHMARK={BENCH!r}: the coverage diagnostic covers methane and "
        "Williams-Otto. There is no Hanoi coverage experiment.")
SRC = HERE / f"coverage_experiment_{BENCH}.py"
OUTROOT = pathlib.Path(os.environ.get(
    "COVERAGE_OUT_ROOT", str(REPO / "rerun" / "coverage_diagnostic" / BENCH)))

# The committed script's seed assignment. NOTE the sites are NOT all equal:
# mlp_u is 1 while everything else is 0 (fcs_experiment.py:47,74,78,101,155).
_ORIGINAL = {
    # methane: NOTE the sites are NOT all equal -- mlp_u is 1
    # (fcs_experiment.py:47,74,78,101,155)
    "methane":  {"global": 0, "mlp_h": 0, "gbm": 0, "mlp_u": 1, "cost": 0},
    # williams: uniform; no GradientBoostingRegressor and no normalising u_hat
    # (fcs_experiment.py:44,78,128)
    "williams": {"global": 0, "mlp_h": 0, "cost": 0},
}
ORIGINAL_SEEDS = _ORIGINAL[BENCH]


def seeds_for_draw(draw: int) -> dict:
    """Draw `d` offsets every site from its committed value: seed = original + d.

    So draw 0 reproduces the committed assignment EXACTLY (0, 0, 0, 1, 0) and is
    therefore the reproduction test. Setting every site to one number instead
    would put mlp_u at 0 rather than 1 and would fail the reproduction test for
    a reason that has nothing to do with the environment -- which is why the
    offset, not a flat assignment, is used.

    Draw 101's mlp_u (102) coincides with draw 102's mlp_h (102). That is
    harmless: mlp_h is fit on (scaled Xtr -> ytr) and mlp_u on
    (scaled Xtr -> |ytr - h_hat(Xtr)|), different targets, so the two networks
    differ regardless of sharing a random_state.
    """
    return {k: v + draw for k, v in ORIGINAL_SEEDS.items()}

# (old, new, what) -- each must occur EXACTLY ONCE in the source.
_SUBS_METHANE = [
    ("rng_global = np.random.default_rng(0)",
     "rng_global = np.random.default_rng(SEED_GLOBAL)", "seed:global"),
    ("max_iter=4000, random_state=0).fit(_scaler.transform(Xtr), ytr)",
     "max_iter=4000, random_state=SEED_MLP_H).fit(_scaler.transform(Xtr), ytr)",
     "seed:mlp_h"),
    ("random_state=0).fit(Xtr, ytr)",
     "random_state=SEED_GBM).fit(Xtr, ytr)", "seed:gbm"),
    ("max_iter=4000, random_state=1).fit(_us.transform(Xtr), _abs_tr)",
     "max_iter=4000, random_state=SEED_MLP_U).fit(_us.transform(Xtr), _abs_tr)",
     "seed:mlp_u"),
    ("COST = make_cost_vectors(N_COST, seed=0)",
     "COST = make_cost_vectors(N_COST, seed=SEED_COST)", "seed:cost"),
    ('fn = f"{RUN}/coverage_{SURROGATE}_{CONFORMAL}.json"',
     'fn = f"{OUT_DIR}/coverage_{SURROGATE}_{CONFORMAL}.json"',
     "output-path (NOT a seed; disclosed)"),
]

# Williams: three seed sites and one output path. It has no GBM and no u_hat, so
# those two substitutions simply do not exist here -- four, not six.
_SUBS_WILLIAMS = [
    ("rng_global = np.random.default_rng(0)",
     "rng_global = np.random.default_rng(SEED_GLOBAL)", "seed:global"),
    ("random_state=0).fit(",
     "random_state=SEED_MLP_H).fit(", "seed:mlp_h"),
    ("COST = make_cost_vectors(N_COST, seed=0)",
     "COST = make_cost_vectors(N_COST, seed=SEED_COST)", "seed:cost"),
    ('fn = f"{RUN}/coverage_wo_{SURROGATE}.json"',
     'fn = f"{OUT_DIR}/coverage_wo_{SURROGATE}.json"',
     "output-path (NOT a seed; disclosed)"),
]

SUBS = _SUBS_METHANE if BENCH == "methane" else _SUBS_WILLIAMS

CONFIGS = ({"symbolic:scalar", "mlp:scalar", "symbolic:normalized", "mlp:normalized"}
           if BENCH == "methane" else {"symbolic", "mlp"})

# Fields present in every generation, so comparable even across the rename.
COMMON = ["alpha", "target", "sanity_random_coverage",
          "reported_selected_coverage", "n_selected",
          "mechanism_matched_boundary_coverage", "n_matched", "margin_band",
          "dose_response"]


def rewrite() -> tuple[str, str, str]:
    orig = SRC.read_text()
    out = orig
    for old, new, what in SUBS:
        n = out.count(old)
        if n != 1:
            raise SystemExit(
                f"refused: substitution '{what}' matches {n} times, not once.\n"
                f"  pattern: {old!r}\n"
                f"fcs_experiment.py has changed; the wrapper must be re-derived "
                f"rather than guessed.")
        out = out.replace(old, new)

    # every changed line must be one of the declared pairs
    diff = list(difflib.unified_diff(orig.splitlines(), out.splitlines(),
                                     "fcs_experiment.py (committed)",
                                     "rewritten", lineterm="", n=0))
    removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    ok_old = {o for o, _, _ in SUBS}
    ok_new = {n for _, n, _ in SUBS}
    for l in removed:
        if not any(o in l for o in ok_old):
            raise SystemExit(f"refused: undeclared line removed:\n  {l}")
    for l in added:
        if not any(n in l for n in ok_new):
            raise SystemExit(f"refused: undeclared line added:\n  {l}")
    if len(removed) != len(SUBS) or len(added) != len(SUBS):
        raise SystemExit(
            f"refused: diff changes {len(removed)} lines, expected exactly "
            f"{len(SUBS)}")
    return orig, out, "\n".join(diff)


def out_name(config: str) -> str:
    if BENCH == "methane":
        s_, c_ = config.split(":")
        return f"fcs_experiment_{s_}_{c_}.json"
    return f"fcs_experiment_wo_{config}.json"


def run(config: str, seeds: dict, outdir: pathlib.Path, src: str) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    g = {"__name__": "__main__", "__file__": str(SRC),
         "OUT_DIR": str(outdir)}
    for k, v in seeds.items():
        g[{"global": "SEED_GLOBAL", "mlp_h": "SEED_MLP_H",
           "gbm": "SEED_GBM", "mlp_u": "SEED_MLP_U",
           "cost": "SEED_COST"}[k]] = v
    argv = sys.argv
    sys.argv = (["fcs_experiment.py", *config.split(":")] if BENCH == "methane"
                else ["fcs_experiment.py", config])
    try:
        exec(compile(src, str(SRC), "exec"), g)
    finally:
        sys.argv = argv
    return outdir / out_name(config)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", type=int, action="append", default=[],
                    help="seed draw; every site = its committed value + draw. "
                         "0 is the committed assignment. Repeatable.")
    ap.add_argument("configs", nargs="*", default=[])
    a = ap.parse_args()

    orig, src, diff = rewrite()
    print(f"{len(SUBS)} declared substitutions, diff touches "
          f"exactly {len(SUBS)} lines, no undeclared change")

    configs = a.configs or sorted(CONFIGS)
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c!r}; choose from {sorted(CONFIGS)}")

    if not a.draw:
        raise SystemExit("give --draw <N> (repeatable)")
    for d in a.draw:
        seeds = seeds_for_draw(d)
        outdir = OUTROOT / f"draw{d}"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "wrapper_diff.patch").write_text(diff + "\n")
        for c in configs:
            print(f"\n=== {c}  draw {d}  seeds {seeds} ===", flush=True)
            p = run(c, seeds, outdir, src)
            print(f"  -> {p}", flush=True)
        json.dump({"draw": d, "seeds": seeds, "configs": configs},
                  open(outdir / "draw_seeds.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

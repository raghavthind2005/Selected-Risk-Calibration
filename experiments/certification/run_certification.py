#!/usr/bin/env python3
"""Certify one locked margin of the candidate family, end to end.

A standalone implementation of the procedure described in COMMANDS.md, for a
single locked margin:

  1. build the surrogate h_hat, its one-sided conformal quantile q_hat and the
     local-uncertainty estimate sigma_hat with its calibration median sigma_bar;
  2. lock the margin m(x) = eta * ((1 - theta) + theta * sigma_hat(x) / sigma_bar);
  3. draw the certification objectives and solve each one under the deployed
     constraint h_hat(x) + m(x) <= 0;
  4. evaluate the true constraint at each returned decision and count the
     violations, counting a solve that returns no feasible point as one;
  5. certify with src/ocl/certify.py.

Steps 2 to 5 are the whole of the procedure and use only src/ocl and the
benchmark.  Step 1 is the context the two studies share, so it is built by the
routine in src/matched_cost/run_seed.py and both are fitted the same way.

A margin is locked before its certification objectives are drawn.  Here the
lock is the invocation: theta and eta are given on the command line, and
nothing downstream of them is tuned.

Examples
--------
    # a locked margin on the smallest setting, into a scratch folder
    python run_certification.py --benchmark methane --seed 3 \
        --theta 0.5 --eta 0.4 --per-law 2 --starts 8 --out /tmp/scratch

    # re-solve a stored matched-cost rung whose deployed constraint is a member
    # of this family, and compare the per-objective violation flags
    python run_certification.py --benchmark methane --seed 3 \
        --replay 'cmicl|60|0.266107'

gamma, delta, K and n come from config/reported_arms.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _import_pipeline(benchmark: str):
    """Import the shared context builder for one benchmark.

    ``run_seed`` selects its benchmark from the environment at import time, so
    this has to run before the module is first imported.
    """
    os.environ["BENCHMARK"] = benchmark
    sys.path.insert(0, str(ROOT / "src" / "matched_cost"))
    sys.path.insert(0, str(ROOT / "src"))
    import run_seed                                               # noqa: PLC0415
    return run_seed


# ----------------------------------------------------------------- solving

_JOB: dict = {}


def _init(job):
    _JOB.update(job)


def _solve_one(i: int) -> dict:
    import numpy as np
    from ocl.ocl_solve import solve_ocl_multistart
    w = _JOB["W"][i]
    result = solve_ocl_multistart(
        lambda x: float(np.dot(np.asarray(x).ravel(), w)),
        _JOB["constraint"], _JOB["bounds"], n_multistart=_JOB["starts"],
        seed=int(_JOB["seeds"][i]), max_iter=200)
    x_star = np.asarray(result["x_star"], dtype=float)
    return {"i": i, "deployed": bool(result["feasible"]),
            "x_star": x_star.tolist()}


def solve_objectives(W, seeds, constraint, bounds, starts, processes):
    """Solve every objective under one deployed constraint."""
    job = {"W": W, "seeds": seeds, "constraint": constraint,
           "bounds": bounds, "starts": starts}
    if processes <= 1:
        _init(job)
        return [_solve_one(i) for i in range(len(W))]
    with mp.get_context("fork").Pool(processes, initializer=_init,
                                     initargs=(job,)) as pool:
        out = pool.map(_solve_one, range(len(W)), chunksize=2)
    return sorted(out, key=lambda r: r["i"])


def count_violations(pipeline, rows, tolerance):
    """Per-objective violation and deployment flags.

    A decision violates when the true constraint exceeds ``tolerance`` there.
    A solve that returns no feasible point counts as a violation, and so does
    one whose decision the true constraint cannot be evaluated at.
    """
    import numpy as np
    violated, deployed = [], []
    for r in rows:
        if not r["deployed"]:
            violated.append(True)
            deployed.append(False)
            continue
        h = float(pipeline.BM.true_h(np.asarray(r["x_star"], dtype=float)))
        finite = bool(np.isfinite(h))
        deployed.append(finite)
        violated.append(not finite or h > tolerance)
    return violated, deployed


# ------------------------------------------------------------- the context

def build_context(pipeline, seed, quiet):
    """Fit h_hat, q_hat and sigma_hat; return the context and sigma_bar."""
    import numpy as np

    def log(message):
        if not quiet:
            print("   ", message, flush=True)

    context = pipeline.build_context(seed, log)
    sigma_bar = float(np.median(pipeline.mlp(pipeline.G["X_cal"], "u")))
    return context, sigma_bar


def deployed_constraint(pipeline, margin):
    """``max(h_hat(x) + m(x), operability(x)) <= 0``, what the solver is given."""
    surrogate, operability = pipeline.hs, pipeline.oper

    def constraint(x):
        return max(surrogate(x) + margin(x), operability(x))

    return constraint


def objective_weights(pipeline, seed, per_law, seed_base, draws_before=0):
    """The objective weights, their families, and one solver seed per objective.

    ``draws_before`` advances the solver-seed generator past earlier objective
    sets, which a run that draws several sets from one generator needs in order
    to land on the same seeds.
    """
    import numpy as np
    W = np.vstack([pipeline.laws_weights(law, per_law, seed_base + seed * 17 + i)
                   for i, law in enumerate(pipeline.LAWS)])
    laws = [law for law in pipeline.LAWS for _ in range(per_law)]
    rng = np.random.default_rng(seed)
    if draws_before:
        rng.integers(0, 2**31, size=int(draws_before))
    return W, laws, rng.integers(0, 2**31, size=len(W))


# ------------------------------------------------------------------ the run

def run(pipeline, args, setting) -> int:
    from ocl.certify import certify
    from ocl.decision_envelope import CandidateMargin

    per_law = args.per_law if args.per_law else setting.n // len(pipeline.LAWS)
    print(f"[1] context: {args.benchmark} seed {args.seed}")
    context, sigma_bar = build_context(pipeline, args.seed, args.quiet)
    print(f"    q_hat {context['q_hat']!r}   sigma_bar {sigma_bar!r}")

    eta = context["q_hat"] if args.eta is None else args.eta
    margin = CandidateMargin(args.theta, eta,
                             lambda X: pipeline.mlp(X, "u"), sigma_bar)
    constraint = deployed_constraint(pipeline, margin)
    print(f"[2] locked margin: theta = {args.theta}, eta = {eta!r}")

    W, _, seeds = objective_weights(pipeline, args.seed, per_law,
                                    args.objective_seed_base)
    print(f"[3] solving {len(W)} certification objectives "
          f"({per_law} from each of {len(pipeline.LAWS)} families), "
          f"{args.starts} starts each")
    started = time.perf_counter()
    rows = solve_objectives(W, seeds, constraint, pipeline.BOUNDS,
                            args.starts, args.processes)
    print(f"    solved in {time.perf_counter() - started:.1f}s")

    violated, deployed = count_violations(pipeline, rows, args.tolerance)
    violations = sum(violated)
    print(f"[4] violations {violations}/{len(violated)} "
          f"(deployed on {sum(deployed)}; a failed solve counts as a violation)")

    certificate = certify(violations, setting, n=len(violated))
    print(f"[5] {certificate}")

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"certification_{args.benchmark}_seed{args.seed}.json"
        json.dump({
            "benchmark": args.benchmark, "seed": args.seed,
            "margin": margin.spec(), "q_hat": context["q_hat"],
            "n_objectives": len(violated), "objectives_per_law": per_law,
            "objective_laws": list(pipeline.LAWS), "starts": args.starts,
            "violation_tolerance": args.tolerance,
            "violations": violations, "n_deployed": sum(deployed),
            "bound": certificate.bound, "certified": certificate.certified,
            "threshold": certificate.threshold, "gamma": setting.gamma,
            "epsilon": setting.epsilon, "familywise_k": setting.K,
        }, open(path, "w"), indent=1)
        print(f"    wrote {path}")
    return 0


# ---------------------------------------------------------------- the replay

#: Stored matched-cost arms whose deployed constraint is a member of the
#: candidate family, with the (theta, eta) that reproduce them.  ``scalar`` adds
#: the constant ``knob``, which is m at theta = 0, eta = knob; ``cmicl`` adds
#: ``knob * q_norm * sigma_hat(x)``, which is m at theta = 1,
#: eta = knob * q_norm * sigma_bar.  Replay a READOUT or CHECK rung: those
#: store the knob at full precision, which a sweep rung need not.
#:
#: At theta = 0 the reparametrisation is exact: the bracket collapses to 1 and
#: the margin is eta itself.  At theta = 1 it is not bit-exact, because it
#: multiplies by sigma_bar to form eta and divides by it again to evaluate the
#: margin; the two forms then differ by about one unit in the last place (a
#: relative 3e-16 measured over the box).  That is enough to move a decision
#: sitting on the deployed boundary into a different local optimum, so a handful
#: of the 200 violation flags can differ on a theta = 1 replay without anything
#: being wrong in the procedure.
REPLAYABLE = {
    "scalar": lambda knob, context, sigma_bar: (0.0, knob),
    "cmicl": lambda knob, context, sigma_bar:
        (1.0, knob * context["q_norm"] * sigma_bar),
}


def replay(pipeline, args, setting) -> int:
    """Re-solve a stored rung and compare its per-objective violation flags."""
    from ocl.certify import certify
    from ocl.decision_envelope import CandidateMargin

    stored = json.load(open(ROOT / "results" / "matched_cost"
                            / f"{args.benchmark}_seed{args.seed}.json"))
    if args.replay not in stored["frontier"]:
        raise SystemExit(f"no rung {args.replay!r} in that run; the replayable "
                         f"ones are "
                         f"{sorted(k for k in stored['frontier'] if k.split('|')[0] in REPLAYABLE)}")
    rung = stored["frontier"][args.replay]
    arm = rung["arm"]
    if arm not in REPLAYABLE:
        raise SystemExit(f"arm {arm!r} is not a member of the candidate family; "
                         f"replayable arms are {sorted(REPLAYABLE)}")

    print(f"[1] context: {args.benchmark} seed {args.seed}")
    context, sigma_bar = build_context(pipeline, args.seed, args.quiet)
    for key in ("q_hat", "q_norm"):
        same = repr(context[key]) == repr(stored["context"][key])
        print(f"    {key:7} re-solved {context[key]!r}")
        print(f"    {key:7} stored    {stored['context'][key]!r}   "
              f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")
        if not same:
            print("    the stored context does not reproduce on this platform, so "
                  "the replay would compare two different margins")
            return 1

    knob = float(rung["knob"])
    # Some sweep rungs store their knob rounded to 8 decimals, so the margin
    # that was actually deployed is only known to ~1e-8. Objectives whose
    # decision sits that close to the boundary can then flip, and a mismatch
    # would say nothing about the code. The READOUT and CHECK rungs -- the ones
    # the tables read -- store the knob at full precision.
    rounded = knob != 0.0 and round(knob, 8) == knob and repr(knob) != repr(round(knob, 2))
    if rounded and not args.allow_rounded_knob:
        raise SystemExit(
            f"rung {args.replay!r} stores its knob as {knob!r}, rounded to 8 "
            f"decimals, so the deployed margin is not recoverable exactly. Pick a "
            f"rung whose knob is stored at full precision (the READOUT|... and "
            f"CHECK|... rungs are), or pass --allow-rounded-knob to compare anyway.")
    theta, eta = REPLAYABLE[arm](knob, context, sigma_bar)
    margin = CandidateMargin(theta, eta,
                             lambda X: pipeline.mlp(X, "u"), sigma_bar)
    constraint = deployed_constraint(pipeline, margin)
    print(f"[2] rung {args.replay}: arm {arm}, knob {rung['knob']!r}")
    print(f"    as a member of the family: theta = {theta}, eta = {eta!r}")

    per_law = len(rung["viol_mask"]) // len(pipeline.LAWS)
    # The stored run draws one solver seed per objective from a single generator,
    # for its proposal and cost-matching sets before this one, so the generator
    # has to be advanced past those draws to land on the same seeds.
    n_laws = len(pipeline.LAWS)
    draws_before = n_laws * pipeline.N_PROP_PL + n_laws * 20
    W, _, seeds = objective_weights(pipeline, args.seed, per_law,
                                    args.objective_seed_base, draws_before)
    starts = args.starts if args.starts != 500 else int(rung["starts"])
    print(f"[3] re-solving the rung's {len(W)} objectives, {starts} starts each")
    rows = solve_objectives(W, seeds, constraint, pipeline.BOUNDS,
                            starts, args.processes)

    # The stored masks use the matched-cost study's violation rule, h(x*) > 0,
    # and mark only deployed decisions, so compare on the same footing.
    violated, deployed = count_violations(pipeline, rows, 0.0)
    violated = [v and d for v, d in zip(violated, deployed)]
    stored_violated = [bool(v) for v in rung["viol_mask"]]
    stored_deployed = [bool(d) for d in rung["dep_mask"]]
    bad_dep = [i for i, (a, b) in enumerate(zip(deployed, stored_deployed)) if a != b]
    bad_viol = [i for i, (a, b) in enumerate(zip(violated, stored_violated)) if a != b]
    print(f"[4] per objective: {len(violated)} compared")
    print(f"    deployment flag mismatches: {len(bad_dep)}"
          + (f"  at {bad_dep[:10]}" if bad_dep else ""))
    print(f"    violation  flag mismatches: {len(bad_viol)}"
          + (f"  at {bad_viol[:10]}" if bad_viol else ""))
    print(f"    violations re-solved {sum(violated)}/{sum(deployed)}   "
          f"stored {rung['n_viol']}/{rung['n_deployed']}")
    print(f"[5] {certify(sum(violated), setting, n=len(violated))}")

    ok = not bad_dep and not bad_viol and sum(violated) == rung["n_viol"]
    print(f"\nVERDICT: {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="methane",
                   choices=("methane", "williams", "hanoi"))
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--theta", type=float, default=0.0,
                   help="the locked shape, in [0, 1]")
    p.add_argument("--eta", type=float, default=None,
                   help="the locked size; defaults to q_hat, the plain conformal margin")
    p.add_argument("--per-law", type=int, default=None,
                   help="certification objectives per family; default n / 5 from config")
    p.add_argument("--starts", type=int, default=500,
                   help="multistart points per objective")
    p.add_argument("--tolerance", type=float, default=1e-8,
                   help="a decision violates when the true constraint exceeds this")
    p.add_argument("--objective-seed-base", type=int, default=2000,
                   help="base of the objective-family seeds")
    p.add_argument("--processes", type=int,
                   default=int(os.environ.get("N_PROC", 0)) or (os.cpu_count() or 4))
    p.add_argument("--out", default=None, help="directory to write the receipt into")
    p.add_argument("--replay", default=None, metavar="RUNG",
                   help="instead re-solve this stored matched-cost rung and "
                        "compare its per-objective violation flags")
    p.add_argument("--allow-rounded-knob", action="store_true",
                   help="replay a rung whose stored knob is rounded to 8 decimals")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    pipeline = _import_pipeline(args.benchmark)
    from ocl.certify import load_setting
    setting = load_setting()
    print(f"certification setting: {setting.describe()}")
    return replay(pipeline, args, setting) if args.replay else run(pipeline, args, setting)


if __name__ == "__main__":
    raise SystemExit(main())

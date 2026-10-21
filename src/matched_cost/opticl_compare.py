#!/usr/bin/env python3.12
"""Direct method-vs-OptiCL comparison at OptiCL's OWN cost point.

The cap ladder cannot host this comparison: `trust_region_hull`'s cost FLOOR --
t = 0, which IS OptiCL's published CH(X), and the dial only tightens from there
-- is 0.069-0.072 on williams and 0.044 on methane, against ladders topping out
at 0.023 and 0.035. So the pre-registered table reports it REFUSED, correctly.

That is not the same as having no comparison. For every hull rung, take the
method's own swept rungs and find the CHEAPEST one whose cost is still <= the
hull's. If the method has fewer violations while spending no more, the
conservative basis applies exactly as written in `analyze.py:admissible()`:
the cost gap runs AGAINST us, so a win cannot be an artefact of it.

Reported for every hull rung on every seed, win or lose, with the paired test.
Selection is by COST only, which is oracle-free, so no outcome enters it.

usage: BENCHMARK=<bench> python3 opticl_compare.py [seed ...]
"""
from __future__ import annotations
import contextlib, io, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A                                            # noqa: E402
BENCH = os.environ.get("BENCHMARK", "methane")


def main(seeds):
    print(f"## {BENCH} — method vs trust_region_hull at the hull's own cost\n")
    print("| seed | hull cost | hull risk | our cost | our risk | b | c | p | "
          "CI-disj | verdict |")
    print("|" + "---|" * 10)
    rows = []
    for s in seeds:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                D = A.load(s)
        except SystemExit:
            continue
        ps = D["primary_starts"]
        hull = A.rungs(D, "trust_region_hull", ps)
        meth = A.rungs(D, A.METHOD, ps)
        if not hull or not meth:
            continue
        for h in sorted(hull, key=lambda v: v["cost"]):
            cands = [m for m in meth if m["cost"] <= h["cost"] + 1e-12]
            if not cands:
                print(f"| {s} | {h['cost']:.4f} | {h['risk']:.3f} | – | – | | | | "
                      "| no method rung at or below this cost |")
                continue
            # Deterministic and oracle-free: the DEAREST method rung that still
            # costs no more than the hull. Cheaper ones would only flatter us by
            # understating what we spent.
            m = max(cands, key=lambda v: v["cost"])
            mv, md = np.array(m["viol_mask"]), np.array(m["dep_mask"])
            pv, pd = np.array(h["viol_mask"]), np.array(h["dep_mask"])
            both = md & pd
            b = int((pv & ~mv & both).sum()); c = int((mv & ~pv & both).sum())
            n = int(both.sum())
            mk, pk = int((mv & both).sum()), int((pv & both).sum())
            p = A.mcnemar(b, c)
            mlo, mhi = A.cp(mk, n); plo, phi = A.cp(pk, n)
            dis = mhi < plo
            verdict = ("BETTER, CI-disjoint" if mk < pk and dis else
                       "better, CIs overlap" if mk < pk else
                       "worse" if mk > pk else "tie")
            rows.append((mk < pk and dis and p < 0.05, p))
            print(f"| {s} | {h['cost']:.4f} | {pk/max(n,1):.3f} | {m['cost']:.4f} "
                  f"| {mk/max(n,1):.3f} | {b} | {c} | {p:.3g} | "
                  f"{'yes' if dis else 'no'} | {verdict} |")
    if rows:
        w = sum(1 for ok, _ in rows if ok)
        print(f"\n{w} of {len(rows)} hull rungs beaten with CI-disjointness and "
              f"p < 0.05, while spending no more than the hull did.")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [3, 5, 6, 9])

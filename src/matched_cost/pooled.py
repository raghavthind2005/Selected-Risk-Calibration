#!/usr/bin/env python3.12
"""Cross-seed mean violation rate at each readout cost.

The matrix answers "what did this baseline do on this seed at this cap". This
answers "what does it do on average at this cost", which is the number you can
put in a paper table.

ONLY COST-MATCHED CELLS ARE POOLED. A cell marked `'`, `^` or `v` in
`matrix.py` is a measurement at a DIFFERENT cost -- the arm saturated below the
cap, or its cheapest configuration already costs more than the cap, or the
refinement landed short. Averaging those into a per-cost mean would report a
number at a cost nobody paid. They are counted in the `off` column instead, so
the omission is visible rather than silent.

Cells admitted on the CONSERVATIVE basis are pooled: there the realised cost is
no higher than the peer's, so the mean is if anything unfavourable to us.

usage: BENCHMARK=<bench> python3 pooled.py [seed ...]
"""
from __future__ import annotations
import contextlib, io, os, pathlib, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A                                            # noqa: E402

BENCH = os.environ.get("BENCHMARK", "methane")


def main(seeds):
    PEERS = A.PEERS_STRONG + A.PEERS_CHEAP
    pool, off, mcost, mrisk, caps_seen = {}, {}, {}, {}, []
    for s in seeds:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                D = A.load(s)
        except SystemExit:
            continue
        ps = D["primary_starts"]
        for cap, target in D["caps"].items():
            if cap not in caps_seen:
                caps_seen.append(cap)
            mr = A.readout(D, A.METHOD, cap) or A.nearest(A.rungs(D, A.METHOD, ps), target)
            if mr is None:
                continue
            mv, md = np.array(mr["viol_mask"]), np.array(mr["dep_mask"])
            mrisk.setdefault(cap, []).append(float(np.mean(mv[md])))
            mcost.setdefault(cap, []).append(float(mr["cost"]))
            for peer in PEERS:
                pr = A.readout(D, peer, cap) or A.nearest(A.rungs(D, peer, ps), target)
                if A.admissible(mr, pr) is None:
                    off[(peer, cap)] = off.get((peer, cap), 0) + 1
                    continue
                pv, pd = np.array(pr["viol_mask"]), np.array(pr["dep_mask"])
                pool.setdefault((peer, cap), []).append(
                    (float(np.mean(pv[pd])), float(pr["cost"])))
    caps_seen.sort()
    print(f"## {BENCH} — mean violation rate across seeds, at each readout cost\n")
    print(f"Seeds pooled: {', '.join(str(s) for s in seeds)}. "
          "`n` = seeds contributing a cost-matched cell; "
          "`off` = seeds whose cell sits at a different cost and is therefore "
          "excluded (marked `'`/`^`/`v` in the matrix).\n")
    hdr = "| baseline |" + "".join(f" {c} mean (n/off) |" for c in caps_seen)
    print(hdr); print("|" + "---|" * (1 + len(caps_seen)))
    row = "| **METHOD** |"
    for c in caps_seen:
        v = mrisk.get(c, [])
        row += f" **{np.mean(v):.3f}** (n={len(v)}) |" if v else " – |"
    print(row)
    row = "| *(its mean cost)* |"
    for c in caps_seen:
        v = mcost.get(c, [])
        row += f" *{np.mean(v):.4f}* |" if v else " – |"
    print(row)
    for peer in PEERS:
        row = f"| {peer} |"
        for c in caps_seen:
            v = pool.get((peer, c), [])
            o = off.get((peer, c), 0)
            row += (f" {np.mean([x for x, _ in v]):.3f} ({len(v)}/{o}) |" if v
                    else f" – (0/{o}) |")
        print(row)


if __name__ == "__main__":
    # NEVER hardcode the seed list. A default of [3, 5, 6, 9] silently dropped
    # williams seed 7 the moment it landed -- and seed 7 is the WEAKEST williams
    # seed, so omitting it inflated every pooled mean in our favour. A pooled
    # total that quietly excludes a seed reads as a stronger result than it is.
    # Same defect and same fix as raw_scores.discover().
    import re as _re
    def _discover():
        d = pathlib.Path(__file__).parent / ("runs" if BENCH == "methane"
                                             else f"runs_{BENCH}")
        out = set()
        for f in d.glob("seed*.json"):
            if f.name.count(".") > 1:      # skip .arms./.partial./.premerge.
                continue
            m = _re.fullmatch(r"seed(\d+)", f.stem)
            if m:
                out.add(int(m.group(1)))
        return sorted(out)
    main([int(a) for a in sys.argv[1:]] or _discover())

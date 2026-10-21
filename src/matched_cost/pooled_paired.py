#!/usr/bin/env python3.12
"""Cross-seed means, PAIRED: every method-vs-peer number is over the SAME seeds.

Why this exists
---------------
`pooled.py` averages each arm over whatever seeds gave it a cost-matched cell,
independently. That is not a like-for-like comparison. Measured on methane at
cap100: the METHOD is averaged over seeds [3, 6, 9], `cmicl` over [6, 9], and
`scalar` and `cmicl_refit` over [3, 6, 7, 9]. Seed 3 is a weak seed and seed 7
has no method rungs at all, so each arm's mean is taken on a different, and
differently-favourable, subset.

The direction of the distortion is not constant -- it flatters us against arms
that happen to skip a strong seed and penalises us against arms measured on a
seed we are not measured on -- so it cannot be waved away as conservative.

This script instead restricts EACH comparison to the seeds where the method and
that peer BOTH have a cost-matched cell, and reports the paired means side by
side with the seed list. A cell backed by fewer seeds is not comparable to one
backed by more, so the count is printed, never hidden.

usage: BENCHMARK=<bench> python3 pooled_paired.py [seed ...]
"""
from __future__ import annotations
import json, os, pathlib, re, sys
import numpy as np

HERE = pathlib.Path(__file__).parent
BENCH = os.environ.get("BENCHMARK", "methane")
SUB = HERE / ("runs" if BENCH == "methane" else f"runs_{BENCH}")
METHOD = "law_blind_envelope"
PEERS = ["cmicl", "cmicl_refit", "alpha_tightening",
         "dose_response", "wmicl_p5", "wmicl_p25", "cqr_scaled", "scalar",
         "trust_region_box", "trust_region_hull"]


def discover():
    out = set()
    for f in SUB.glob("seed*.json"):
        if f.name.count(".") > 1:
            continue
        m = re.fullmatch(r"seed(\d+)", f.stem)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def main(seeds):
    runs, caps = {}, []
    for s in seeds:
        p = SUB / f"seed{s}.json"
        if not p.exists():
            continue
        runs[s] = json.load(open(p))
        for c in runs[s]["caps"]:
            if c not in caps:
                caps.append(c)
    caps.sort(key=lambda c: int(c[3:]))

    def risk(s, arm, cap):
        r = runs[s]["frontier"].get(f"READOUT|{arm}|{cap}")
        return None if r is None else float(r["risk"])

    print(f"## {BENCH} — PAIRED cross-seed means (same seeds on both sides)\n")
    print("Each cell: method mean vs peer mean, over the seeds where BOTH have a "
          "cost-matched rung. `n` is that number of seeds; the seeds themselves "
          "are listed. Cells with different `n` are not comparable to each other.\n")
    head = "| peer | " + " | ".join(caps) + " |"
    print(head); print("|" + "---|" * (len(caps) + 1))
    for peer in PEERS:
        row = f"| {peer} |"
        for cap in caps:
            both = [s for s in runs
                    if risk(s, METHOD, cap) is not None
                    and risk(s, peer, cap) is not None]
            if not both:
                row += " – |"
                continue
            m = np.mean([risk(s, METHOD, cap) for s in both])
            q = np.mean([risk(s, peer, cap) for s in both])
            mark = "**" if m < q else ""
            row += (f" {mark}{m:.3f}{mark} vs {q:.3f} "
                    f"(n={len(both)}: {','.join(str(s) for s in sorted(both))}) |")
        print(row)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or discover())

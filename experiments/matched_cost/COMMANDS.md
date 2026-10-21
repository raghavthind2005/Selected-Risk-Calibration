# Matched-cost comparison

One run per benchmark and seed. Each run fits the surrogate, calibrates the
conformal margin, fits the correction, sweeps every arm's knob, and reads the
violation rate at each budget.

    cd src/matched_cost
    BENCHMARK=methane  SEED=3 python3 run_seed.py
    BENCHMARK=williams SEED=3 python3 run_seed.py
    BENCHMARK=hanoi    SEED=3 python3 run_seed.py

Seeds 3, 5, 6, 7 and 9 on each of the three benchmarks give the 15 runs the
comparison reports.

Environment variables:

| variable | meaning |
|---|---|
| `BENCHMARK` | `methane`, `williams` or `hanoi` (default `methane`) |
| `SEED` | the seed, passed as the first positional argument instead if preferred |
| `N_PROC` | worker pool size (default: the core count) |
| `RESUME` | `1` continues from this run's own checkpoint |

A run writes one JSON per seed, holding the context, the per-arm frontier with
the per-objective violation and deployment masks, and the realised costs. The
stored copies of those files are in `results/matched_cost/`.

## The extended-refinement pass

Some seed slots needed more refinement iterations to reach an equal-cost match.
`src/matched_cost/extended_refinement.py` reruns only those slots, and
`analysis/build_matched_cost.py` merges the result over the base readouts. Its
stored output is in `results/matched_cost/extended_refinement/`.

## Exporting

`frontier_export.py` turns the per-seed JSONs into the flat row files the
analysis reads (`results/export/frontier.csv` and
`results/export/objective_level.csv.gz`); `merge_arms.py` merges a partial run
into a base run.

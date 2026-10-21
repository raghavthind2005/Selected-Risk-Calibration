# Coverage diagnostic

Each fit measures the coverage of one margin on three sets of points: random
held-out inputs, the decisions returned for 100 random objectives, and
matched-boundary points — random inputs the optimizer did not choose that lie as
close to the boundary as the returned decisions do.

Six configurations: two surrogates (a symbolic expression and an MLP) crossed
with two margins (scalar and normalized) on methane, and the scalar margin with
both surrogates on Williams-Otto. Each is repeated five times.

    cd experiments/coverage_diagnostic
    python3 run_coverage_fit.py --benchmark methane  --draw 0
    python3 run_coverage_fit.py --benchmark williams --draw 0

`--draw` selects the repetition; the five repetitions the diagnostic reports are
draws 0, 101, 102, 103 and 104. A draw shifts every seed inside the experiment
by that amount, so draw 0 is the experiment at its committed seeds.

The alpha sweep repeats the methane fits with the scalar margin at
alpha = 0.20, 0.15, 0.10, 0.05, 0.02 and 0.01:

    python3 run_alpha_sweep.py --draw 0

Each fit writes one JSON. The stored copies are in
`results/coverage_diagnostic/`, one directory per benchmark and per draw, plus
`alpha_sweep/`.

The diagnostic has its own training, calibration and test splits and its own
margins, so its numbers are not comparable with the other studies'.

# Certification study

SRC runs on methane and Williams-Otto with eight seeds each. At each target
cost — three per seed on methane, two on Williams-Otto, giving 40 margins — the
five candidate shapes are sized to that cost, the shape with the fewest
violations on 300 held-out objectives is selected, and the selected margin is
certified on 300 new objectives.

With gamma = 0.12, delta = 0.05 and K = 40, a margin is certified when at most
19 of its 300 certification objectives end in a violation. Every margin is
certified a second time on a further 300 objectives, and the selected margins
are then solved on 200 fresh objectives.

A returned decision counts as a violation when h(x) > 1e-8; a solve that returns
no feasible point also counts as a violation. Each objective is solved from 500
starting points.

The deployed constraint, the candidate family and the solver are
`src/ocl/decision_envelope.py`, `src/ocl/conformal.py` and
`src/ocl/ocl_solve.py` — the same modules the matched-cost runs use. The
certification test itself — the Clopper-Pearson bound, the threshold, and the
choice of the cheapest certified margin — is `src/ocl/certify.py`, which reads
gamma, delta, K and n from `config/reported_arms.json`.

## Running one locked margin

`run_certification.py` is a standalone implementation of the procedure used in
the study, checked against the stored results. It certifies one locked margin
end to end: it builds the surrogate and its conformal quantile, locks
`m(x) = eta * ((1 - theta) + theta * sigma_hat(x) / sigma_bar)`, solves the
certification objectives under `h_hat + m <= 0`, evaluates the true constraint
at each returned decision, counts the violations and certifies.

    cd experiments/certification
    python3 run_certification.py --benchmark methane --seed 3 \
        --theta 0.5 --eta 0.4 --out /tmp/scratch

`--theta 0 --eta q_hat` (the default `--eta`) is the plain conformal deployment.
`--per-law` and `--starts` set the size of the run; the defaults are the study's
300 objectives and 500 starts, which takes hours, so pass smaller ones for a
quick check.

`--replay RUNG` instead re-solves a stored matched-cost rung whose deployed
constraint is a member of the candidate family — `scalar` is the family at
theta = 0 and `cmicl` at theta = 1 — and compares the violation flag of every
returned decision against the one stored with that rung.

    python3 run_certification.py --benchmark methane --seed 3 \
        --replay 'READOUT|scalar|cap025'

`smoke/certify_check.py` recomputes every bound and decision of all 40 margins
from the stored violation counts and compares them with `tables/tab_cert_all.tex`.

The stored results are in `results/certification/`: one directory per benchmark
and seed with its per-anchor certification rows and its fresh performance rows,
plus `second_certification/` for the second draw and the two
`*_equal_cost_frontier/` directories for the cost frontiers the anchors are
placed on.

Seeds 4 and 10 have no fresh evaluation of the certified margins.

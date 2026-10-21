# REPRODUCE.md

Two levels: rebuild every output from the stored results, or rerun a study from
scratch.

## Level 1 — rebuild every table and figure from `results/`

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    SOURCE_DATE_EPOCH=1767225600 ./run_all.sh

Measured on an 8-core arm64 machine, CPython 3.13.12: **82 seconds** for the
whole of `run_all.sh`. Most of it is one step:

| step | writes | measured |
|---|---|---|
| `build_matched_cost.py` | `derived/matched_cost_{per_seed,paired,pooled}.csv` | 8.5 s |
| `build_certification.py` | `derived/certification_anchors.{csv,md,json}` | 1.3 s |
| `build_equal_cost.py` | `derived/equal_cost_panel.{csv,md}` | 0.2 s |
| `build_opticl.py` | `derived/opticl_hull.{csv,md}` | 56 s |
| `build_coverage_diagnostic.py` | `derived/{coverage_diagnostic_fits,alpha_sweep}.csv` | 0.1 s |
| the nine `tab_*.py` | `tables/*.tex` | under 0.1 s each |
| `fig_coverage_diagnostic.py` | `figures/fig_coverage_diagnostic.{pdf,png}` | 1.4 s |
| `fig_certificate_bound.py` | `figures/fig_certificate_bound.{pdf,png}` | 0.9 s |

`build_opticl.py` dominates because it reads every per-objective row of
`results/export/objective_level.csv.gz` (616k rows).

### Determinism

Every `.tex` and `.csv` output is a pure function of the stored results.

matplotlib stamps a `CreationDate` into every PDF, so `analysis/_figsave.py`
writes the figures with that field left out. Both PDFs then reproduce
byte-for-byte across runs, and `MANIFEST.sha256` holds.

The matplotlib pin in `requirements.txt` is load-bearing: the same data rendered
by matplotlib 3.10.9 and 3.11.2 differs in tens of thousands of pixels while
every data table stays byte-identical. An unpinned install reproduces the tables
and misses the figures.

## Level 2 — rerun a study from `experiments/`

    python3 -m venv .venv-exp
    source .venv-exp/bin/activate
    pip install -r requirements-experiments.txt

Reruns write into their own output directory and do not touch `results/`.

| study | entry point | runtime |
|---|---|---|
| matched cost, one benchmark and seed | `experiments/matched_cost/COMMANDS.md` | 4-10 h |
| matched cost, all 15 runs | as above | 60-150 h |
| coverage diagnostic, one configuration and repetition | `experiments/coverage_diagnostic/COMMANDS.md` | 5-20 min |
| coverage diagnostic, all 30 fits plus the alpha sweep | as above | 6-12 h |

These are the wall-clock times the runs themselves recorded, and they are
approximate: the matched-cost figure comes from the recorded elapsed time of
single-seed runs (for example 11,251 s for one Williams-Otto seed), and the
coverage-diagnostic figure from the recorded elapsed time of single fits. They
depend on the number of cores; `N_PROC` controls the pool size.

The Hanoi runs additionally need the network file — see
`benchmarks/hanoi_network/README.md`.

### Versions

`requirements.txt` pins the analysis exactly. `requirements-experiments.txt`
carries the versions the runs recorded: python 3.12.7, numpy 2.2.6 and
scipy 1.14.1 for the methane matched-cost runs, python 3.12.13 and numpy 2.0.2
for the Williams-Otto and Hanoi runs, and scikit-learn 1.5.2 and wntr 1.5.0
where recorded. The pandas and scikit-learn versions of the methane
matched-cost runs were not recorded, so they are left unpinned rather than
guessed.

The results in `results/` were produced on the platforms recorded in each
result file, and rerunning elsewhere can change the fitted surrogate and
therefore the margin.

A rerun will not reproduce a stored run bit-for-bit on a different platform.
The surrogate fit and the conformal quantile both depend on the oracle's
floating-point output, and the methane oracle integrates an ODE: a different
scipy or BLAS moves the quantile by a few per cent, which moves the deployed
margin and so the violation counts. The stored runs each record their own
`platform` block.

## Fast checks

    python3 smoke/smoke_test.py      # rebuild twice, compare, verify the manifest
    python3 smoke/mutation_test.py   # prove config/reported_arms.json is load-bearing
    python3 smoke/certify_check.py   # recompute all 40 certificates from their counts

`certify_check.py` recomputes the Clopper-Pearson bound, the certified flag and
the threshold b for every one of the 40 margins from the stored violation counts
alone, and compares them with both the stored decisions and every cell of
`tables/tab_cert_all.tex`. It needs only `requirements.txt`.

A single locked margin can be certified end to end, which takes a context fit
and so belongs to level 2:

    python3 experiments/certification/run_certification.py \
        --benchmark methane --seed 3 --theta 0.5 --eta 0.4 \
        --per-law 2 --starts 8 --out /tmp/scratch

## Quoted numbers

Every number in a table is rounded half up from the exact rational value.
`analysis/_fmt.py` implements that rule and self-tests.

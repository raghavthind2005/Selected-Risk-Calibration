# Convex-hull trust region

The OptiCL trust region keeps the decision inside the convex hull of the
training inputs. It is compared against SRC's correction at each of its three
settings per seed, on methane and Williams-Otto.

    cd src/matched_cost
    BENCHMARK=methane SEED=3 python3 opticl_compare.py

The hull cannot be built at the Hanoi benchmark's dimension: a convex hull on
1,000 points in eight dimensions exhausts memory, and 400 points already give
474,818 facets, with every constraint evaluation taking a maximum over all of
them. `run_seed.py` therefore excludes the arm on that benchmark.

The comparison is read from the stored per-objective rows by
`analysis/build_opticl.py`; the hull's own swept settings are in
`results/export/objective_level.csv.gz` alongside the other arms.

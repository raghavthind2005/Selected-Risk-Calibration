"""
Generate the LLM-OCL `methane_reactor` dataset.

EXACT C-MICL (NeurIPS 2025) membrane-reactor replication. Ground-truth ODE,
variables, bounds, and constraint are ported verbatim from the authors'
released `regression.py` (see ode_model.py).

True constraint: FC6H6(x) >= 50  ↔  h(x) = 50 - FC6H6(x) <= 0
where FC6H6 is the benzene outlet flow from the reactor ODE.

5 decision variables  [v0 (cm^3/h), vHe (cm^3/h), T (K), dt (cm), L (cm)]:
    sampled uniformly from VARIABLE_BOUNDS in ode_model.py.

Outputs CSVs under out_dir with columns [x1, x2, x3, x4, x5, h]
so that LLM-SR main.py ingests them unchanged.

We also write test_id.csv (noisy, same distribution) and test_clean.csv
(noise-free) for our conformal coverage evaluation — beyond C-MICL's protocol
but required for LLM-OCL's benchmark reporting.

Failed ODE solves are retried up to 3 extra samples per slot to keep
the dataset full. Points that still fail are dropped (logged to stderr).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# ── locate ode_model relative to this file ──────────────────────────────────
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from ode_model import h_true_vec, simulate, VARIABLE_BOUNDS, VARIABLE_NAMES

# ── column names ────────────────────────────────────────────────────────────
FEATURE_COLS = ['x1', 'x2', 'x3', 'x4', 'x5']   # intentionally anonymous
TARGET_COL   = 'h'


def _sample(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n uniform samples from the 5D parameter space."""
    lo = np.array([b[0] for b in VARIABLE_BOUNDS])
    hi = np.array([b[1] for b in VARIABLE_BOUNDS])
    return rng.uniform(lo, hi, size=(n, 5))


def _evaluate_batch(X: np.ndarray, sigma: float, rng: np.random.Generator,
                    noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate h(x) for each row in X, optionally adding Gaussian noise.

    Returns (X_valid, h_valid) excluding any rows where the ODE failed.
    """
    h_clean = h_true_vec(X)
    valid   = np.isfinite(h_clean)
    X_v     = X[valid]
    h_v     = h_clean[valid]
    if noise and sigma > 0:
        h_v = h_v + rng.normal(0.0, sigma, size=len(h_v))
    return X_v, h_v


def _make_split(
    n_target: int,
    rng: np.random.Generator,
    sigma: float,
    noise: bool,
    max_attempts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n_target valid (X, h) pairs, retrying if the ODE fails.
    """
    rows_X, rows_h = [], []
    collected = 0
    attempt   = 0

    while collected < n_target and attempt < max_attempts:
        needed  = (n_target - collected) * 2   # over-sample to cover failures
        X_raw   = _sample(rng, needed)
        X_v, h_v = _evaluate_batch(X_raw, sigma, rng, noise)
        rows_X.append(X_v[:n_target - collected])
        rows_h.append(h_v[:n_target - collected])
        collected += len(rows_X[-1])
        attempt   += 1

    if collected < n_target:
        print(f'  WARNING: only {collected}/{n_target} valid samples generated',
              file=sys.stderr)

    X_all = np.vstack(rows_X)
    h_all = np.concatenate(rows_h)
    return X_all, h_all


def _write_csv(path: pathlib.Path, X: np.ndarray, h: np.ndarray) -> None:
    df = pd.DataFrame(
        np.column_stack([X, h]),
        columns=FEATURE_COLS + [TARGET_COL],
    )
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir', type=str,
                        default=str(HERE))
    # C-MICL protocol: 1000 points, 80/20 train/cal
    parser.add_argument('--n_train',       type=int, default=800)
    parser.add_argument('--n_calibration', type=int, default=200)
    # Extra splits for LLM-OCL conformal evaluation
    parser.add_argument('--n_test_id',     type=int, default=200)
    parser.add_argument('--n_test_clean',  type=int, default=200)
    parser.add_argument('--sigma',         type=float, default=1.0,
                        help='Gaussian noise std on h (benzene-flow units). C-MICL '
                             'adds noise to benzene flows; their exact sigma is not '
                             'in the released artifacts, so we use 1.0 (~9%% of the '
                             'benzene std of 11.3, ~2%% of the threshold of 50).')
    parser.add_argument('--seed',          type=int, default=42)
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seeds = {
        'train':       args.seed + 0,
        'calibration': args.seed + 1,
        'test_id':     args.seed + 2,
        'test_clean':  args.seed + 3,
    }

    splits = [
        ('train.csv',       args.n_train,       seeds['train'],       True),
        ('calibration.csv', args.n_calibration, seeds['calibration'], True),
        ('test_id.csv',     args.n_test_id,     seeds['test_id'],     True),
        ('test_clean.csv',  args.n_test_clean,  seeds['test_clean'],  False),
    ]

    print(f'Generating methane reactor dataset → {out}')
    feas_counts = {}
    for fname, n, seed, noisy in splits:
        t0  = time.perf_counter()
        rng = np.random.default_rng(seed)
        X, h = _make_split(n, rng, args.sigma, noisy)
        _write_csv(out / fname, X, h)
        feas = int(np.sum(h <= 0))
        tag  = 'noisy' if noisy else 'CLEAN'
        elapsed = time.perf_counter() - t0
        print(f'  {fname:<18} {len(X):>4} rows  ({tag})  '
              f'feasible={feas}/{len(X)} ({100*feas/len(X):.0f}%)  '
              f'[{elapsed:.1f}s]')
        feas_counts[fname] = feas

    metadata = {
        'true_form':          'ODE: DMA membrane reactor (C-MICL regression.py port)',
        'reference':          'Carrasco & Lima; opyrability repo (per C-MICL appendix)',
        'cmicl_benchmark':    'Ovalle et al., NeurIPS 2025 (arXiv:2506.03531)',
        'replication':        'EXACT: ODE, 5 variables, bounds, and benzene>=50 '
                              'constraint ported verbatim from C-MICL regression.py',
        'variables':          dict(zip(['x1','x2','x3','x4','x5'], VARIABLE_NAMES)),
        'variable_bounds':    {f'x{i+1}': list(b)
                               for i, b in enumerate(VARIABLE_BOUNDS)},
        'feature_columns':    FEATURE_COLS,
        'target_column':      TARGET_COL,
        'constraint':         'h(x) = 50 - FC6H6(x) <= 0  [feasible when benzene '
                              'outlet flow FC6H6 >= 50]',
        'objective':          'C-MICL: random cost vectors uniform(-4,4) with '
                              'negatives shrunk /10, + operability constraints',
        'sigma_noise':        args.sigma,
        'seeds':              seeds,
        'split_sizes': {k: v for k, v in zip(
            ['train', 'calibration', 'test_id', 'test_clean'],
            [args.n_train, args.n_calibration, args.n_test_id, args.n_test_clean])},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'  metadata.json written')
    print(f'Done.')


if __name__ == '__main__':
    main()

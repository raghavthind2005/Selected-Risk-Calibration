"""Generate the Williams-Otto (5D, full-plant-with-recycle) OCL dataset.

Constraint: F_P(x) >= F_P_MIN  <->  h(x) = F_P_MIN - F_P(x) <= 0.
Gaussian measurement noise added to h (per standard practice / C-MICL).
Columns: x1..x5 = (V, T, F_A, F_B, eta), h.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from plant_model import h_true_vec, VARIABLE_BOUNDS, VARIABLE_NAMES, F_P_MIN

FEATURE_COLS = ['x1', 'x2', 'x3', 'x4', 'x5']
TARGET_COL = 'h'
LO = np.array([b[0] for b in VARIABLE_BOUNDS])
HI = np.array([b[1] for b in VARIABLE_BOUNDS])


def _make_split(n, rng, sigma, noise):
    rows_X, rows_h, collected, attempt = [], [], 0, 0
    while collected < n and attempt < 6:
        X = rng.uniform(LO, HI, size=((n - collected) * 2, 5))
        h = h_true_vec(X)
        valid = np.isfinite(h)
        Xv, hv = X[valid], h[valid]
        if noise and sigma > 0:
            hv = hv + rng.normal(0, sigma, size=len(hv))
        take = n - collected
        rows_X.append(Xv[:take]); rows_h.append(hv[:take])
        collected += len(rows_X[-1]); attempt += 1
    return np.vstack(rows_X), np.concatenate(rows_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=str(HERE))
    ap.add_argument('--n_train', type=int, default=800)
    ap.add_argument('--n_calibration', type=int, default=200)
    ap.add_argument('--n_test_id', type=int, default=200)
    ap.add_argument('--n_test_clean', type=int, default=200)
    ap.add_argument('--sigma', type=float, default=0.05,
                    help='Gaussian noise std on h (F_P units; ~9%% of F_P std)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    seeds = {'train': args.seed, 'calibration': args.seed + 1,
             'test_id': args.seed + 2, 'test_clean': args.seed + 3}
    splits = [('train.csv', args.n_train, seeds['train'], True),
              ('calibration.csv', args.n_calibration, seeds['calibration'], True),
              ('test_id.csv', args.n_test_id, seeds['test_id'], True),
              ('test_clean.csv', args.n_test_clean, seeds['test_clean'], False)]
    print(f'Generating Williams-Otto dataset -> {out}')
    for fname, n, seed, noisy in splits:
        t0 = time.perf_counter(); rng = np.random.default_rng(seed)
        X, h = _make_split(n, rng, args.sigma, noisy)
        pd.DataFrame(np.column_stack([X, h]),
                     columns=FEATURE_COLS + [TARGET_COL]).to_csv(out / fname, index=False)
        feas = int(np.sum(h <= 0))
        print(f'  {fname:<18}{len(X):>4} rows  ({"noisy" if noisy else "CLEAN"})  '
              f'feasible={feas}/{len(X)} ({100*feas/len(X):.0f}%)  '
              f'[{time.perf_counter()-t0:.1f}s]')
    meta = {
        'true_form': 'Williams-Otto full plant w/ recycle (arXiv 2511.18998 AN6)',
        'replication': 'EXACT: reactions, recycle fixed-point, 5 decisions from AN6',
        'variables': dict(zip(FEATURE_COLS, VARIABLE_NAMES)),
        'variable_bounds': {f'x{i+1}': list(b) for i, b in enumerate(VARIABLE_BOUNDS)},
        'constraint': f'h(x) = {F_P_MIN} - F_P(x) <= 0  [min production F_P >= {F_P_MIN}]',
        'objective': 'minimize feed+energy operating cost (parallel to C-MICL methane)',
        'sigma_noise': args.sigma, 'seeds': seeds,
    }
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('  metadata.json written\nDone.')


if __name__ == '__main__':
    main()

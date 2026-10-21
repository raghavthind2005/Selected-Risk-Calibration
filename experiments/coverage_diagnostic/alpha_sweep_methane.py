"""
Alpha sweep: coverage at the returned decision as alpha is tightened.

For each alpha:
  - recompute q_hat, the (1-alpha)-quantile of the one-sided calibration
    residuals (q_hat grows as alpha shrinks, so the margin is more conservative),
  - re-solve  min c(x) s.t. h_hat(x) + q_hat <= 0 and the operating constraints,
    over 100 cost vectors,
  - measure:
      * SANITY    coverage on random test points        (tracks 1-alpha)
      * SELECTED  coverage at the returned decision x*
      * FEAS      true feasibility at x*, h(x*) <= 0
      * gap       (1-alpha) minus the coverage at x*
      * slack     median h_hat(x*) + q_hat: about 0 means x* is still on the
                  boundary, below 0 means it has retreated into the interior

Usage:  python alpha_sweep.py [symbolic|mlp]
"""
import json, glob, sys, time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import os
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "benchmarks", "methane_reactor")
RUN = HERE
N_COST = 100
ALPHAS = [0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
SURROGATE = sys.argv[1] if len(sys.argv) > 1 else 'symbolic'
sys.path.insert(0, BASE)
from benchmark import (make_cost_vectors, objective, operability_constraints,
                       true_h, _lo, _hi)

rng_global = np.random.default_rng(0)


def load(split):
    d = pd.read_csv(f"{BASE}/{split}.csv")
    return d[['x1', 'x2', 'x3', 'x4', 'x5']].values, d['h'].values


def q_finite(scores, alpha):
    n = len(scores); rank = int(np.ceil((1 - alpha) * (n + 1)))
    return float('inf') if rank > n else float(np.sort(scores)[rank - 1])


# ── surrogate ─────────────────────────────────────────────────────────────────
Xtr, ytr   = load('train')
Xcal, ycal = load('calibration')
Xtest, ytest = load('test_clean')

if SURROGATE == 'symbolic':
    _rec = json.load(open(os.path.join(HERE, "symbolic_surrogate_methane.json")))
    p = np.array(_rec['optimized_params'])
    ns = {'np': np}; exec(_rec['function'], ns); eq = ns['equation']
    def h_hat_vec(X): return eq(np.atleast_2d(X), p)
elif SURROGATE == 'mlp':
    _scaler = StandardScaler().fit(Xtr)
    _mlp = MLPRegressor(hidden_layer_sizes=(32, 32), activation='relu', alpha=0.01,
                        max_iter=4000, random_state=0).fit(_scaler.transform(Xtr), ytr)
    def h_hat_vec(X): return _mlp.predict(_scaler.transform(np.atleast_2d(X)))
else:
    raise SystemExit(f"unknown surrogate {SURROGATE!r}")

def h_hat(x): return float(h_hat_vec(np.asarray(x, float).reshape(1, 5))[0])

r_cal = ycal - h_hat_vec(Xcal)
ops = [{'type': 'ineq', 'fun': (lambda x, k=k: -operability_constraints(x)[k])}
       for k in range(7)]


def solve(cv, Q):
    c_con = {'type': 'ineq', 'fun': lambda x: -(h_hat(x) + Q)}
    cons = ops + [c_con]
    rng = np.random.default_rng(int(abs(cv.sum()) * 1e4) % 2**31)
    br = None
    for _ in range(12):
        x0 = rng.uniform(_lo, _hi)
        try:
            r = minimize(lambda x: objective(x, cv), x0, method='SLSQP',
                         bounds=list(zip(_lo, _hi)), constraints=cons,
                         options={'ftol': 1e-8, 'maxiter': 200})
            if r.success and (br is None or r.fun < br.fun):
                br = r
        except Exception:
            pass
    return None if br is None else br.x


COST = make_cost_vectors(N_COST, seed=0)
print(f"surrogate = {SURROGATE}   n_cost = {N_COST}\n")
print(f"{'1-alpha':>8} {'q_hat':>8} {'sanity':>7} {'selected':>9} {'feas':>6} "
      f"{'gap':>7} {'slack':>8} {'obj':>8} {'n':>4}")
print("-" * 78)

rows = []
t0 = time.time()
for alpha in ALPHAS:
    target = 1 - alpha
    Q = q_finite(r_cal, alpha)

    # sanity: random test-point coverage (must track target)
    cov_rand = float(np.mean((ytest - h_hat_vec(Xtest)) <= Q))

    # selected: solve over cost vectors
    Xstar = [x for cv in COST if (x := solve(cv, Q)) is not None]
    Xstar = np.array(Xstar)
    n_sel = len(Xstar)
    if n_sel:
        true_hs = np.array([true_h(x) for x in Xstar])
        hhat_s  = h_hat_vec(Xstar)
        cov_sel = float(np.mean((true_hs - hhat_s) <= Q))    # conformal coverage at x*
        feas    = float(np.mean(true_hs <= 0.0))             # true feasibility at x*
        slack   = float(np.median(hhat_s + Q))               # ~0 boundary, <0 interior
        objs    = np.array([objective(x, cv) for x, cv in zip(Xstar, COST[:n_sel])])
        obj_m   = float(np.mean(objs))
    else:
        cov_sel = feas = slack = obj_m = float('nan')

    gap = target - cov_sel
    rows.append({'alpha': alpha, 'target': target, 'q_hat': Q,
                 'sanity': cov_rand, 'selected': cov_sel, 'feas': feas,
                 'gap': gap, 'slack': slack, 'obj': obj_m, 'n': n_sel})
    print(f"{target:>8.2f} {Q:>8.3f} {cov_rand:>7.3f} {cov_sel:>9.3f} {feas:>6.3f} "
          f"{gap:>+7.3f} {slack:>8.3f} {obj_m:>8.2f} {n_sel:>4d}", flush=True)

print(f"\nElapsed {(time.time()-t0)/60:.1f} min")
out = {'surrogate': SURROGATE, 'n_cost': N_COST, 'rows': rows}
fn = f"{RUN}/alpha_sweep_{SURROGATE}.json"
json.dump(out, open(fn, 'w'), indent=2)
print(f"Wrote {fn}")

# ── summary ───────────────────────────────────────────────────────────────────
gaps = [r['gap'] for r in rows if not np.isnan(r['gap'])]
sel  = [r['selected'] for r in rows if not np.isnan(r['selected'])]
print("\n" + "=" * 60)
print("SUMMARY OVER ALPHA")
print("=" * 60)
print(f"gap (target - selected) across alpha: {[f'{g:+.3f}' for g in gaps]}")
print(f"selected coverage across alpha:       {[f'{s:.3f}' for s in sel]}")

"""
FCS (Feedback Covariate Shift) diagnostic for Williams-Otto 5D plant.

Parallel to coverage_experiment_methane.py.
Tests whether the FCS collapse at optimizer-selected x* is a phenomenon
specific to methane or generalises across different process structures.

Williams-Otto differs from methane in:
  - Implicit constraint (recycle fixed-point), not an ODE integration
  - No known operability constraints beyond box bounds
  - x5 (recycle ratio eta) creates non-monotone behaviour via feedback

4-part diagnostic:
  (1) SANITY     coverage on random test points        -- expect >= 0.90
  (2) coverage at the optimizer's returned decision x*
  (3) MECHANISM  matched-boundary control               -- FCS vs heteroscedasticity
  (4) DOSE-RESP  lambda sweep on selection aggressiveness
"""
import json
import glob
import os, sys
import warnings
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "benchmarks", "williams_otto")
RUN = HERE
ALPHA  = 0.10
TARGET = 1 - ALPHA
N_COST = 100

SURROGATE = sys.argv[1] if len(sys.argv) > 1 else 'symbolic'   # symbolic | mlp
sys.path.insert(0, BASE)
from plant_model import simulate, product_flow, VARIABLE_BOUNDS, F_P_MIN
from benchmark   import make_cost_vectors, objective

rng_global = np.random.default_rng(0)

_lo = np.array([b[0] for b in VARIABLE_BOUNDS])
_hi = np.array([b[1] for b in VARIABLE_BOUNDS])
RANGE = _hi - _lo


def load(split):
    d = pd.read_csv(f"{BASE}/{split}.csv")
    return d[['x1', 'x2', 'x3', 'x4', 'x5']].values, d['h'].values


def q_finite(scores, alpha):
    n = len(scores)
    rank = int(np.ceil((1 - alpha) * (n + 1)))
    return float('inf') if rank > n else float(np.sort(scores)[rank - 1])


# ── surrogate ─────────────────────────────────────────────────────────────────
Xtr, ytr   = load('train')
Xcal, ycal = load('calibration')

if SURROGATE == 'symbolic':
    _rec = json.load(open(os.path.join(HERE, "symbolic_surrogate_williams.json")))
    p = np.array(_rec['optimized_params'])
    ns = {'np': np}
    exec(_rec['function'], ns)
    eq = ns['equation']
    def h_hat_vec(X): return eq(np.atleast_2d(X), p)
elif SURROGATE == 'mlp':
    _scaler = StandardScaler().fit(Xtr)
    _mlp = MLPRegressor(hidden_layer_sizes=(32, 32), activation='relu',
                        alpha=0.01, max_iter=4000, random_state=0).fit(
                        _scaler.transform(Xtr), ytr)
    def h_hat_vec(X): return _mlp.predict(_scaler.transform(np.atleast_2d(X)))
else:
    raise SystemExit(f"unknown surrogate {SURROGATE!r}")

def h_hat(x): return float(h_hat_vec(np.asarray(x, float).reshape(1, 5))[0])

r_cal = ycal - h_hat_vec(Xcal)
Q = q_finite(r_cal, ALPHA)
train_r2 = 1 - np.var(ytr - h_hat_vec(Xtr)) / np.var(ytr)
print(f"surrogate={SURROGATE}  R²(train)={train_r2:.4f}  q̂={Q:+.4f}  target={TARGET:.2f}\n")


# ── solver (no operability constraints — box only + learned) ──────────────────
def solve(cv, lam=0.0, x_safe=None):
    c_con = {'type': 'ineq', 'fun': lambda x: -(h_hat(x) + Q)}
    cons = [c_con]
    def obj(x):
        v = objective(x, cv)
        if lam > 0.0:
            v += lam * float(np.mean(((x - x_safe) / RANGE) ** 2))
        return v
    rng = np.random.default_rng(int(abs(cv.sum()) * 1e4) % 2**31)
    br = None
    for _ in range(12):
        x0 = rng.uniform(_lo, _hi)
        try:
            r = minimize(obj, x0, method='SLSQP',
                         bounds=list(zip(_lo, _hi)), constraints=cons,
                         options={'ftol': 1e-8, 'maxiter': 200})
            if r.success and (br is None or r.fun < br.fun):
                br = r
        except Exception:
            pass
    return None if br is None else br.x


def coverage_at(Xpts):
    cov = []
    for x in Xpts:
        cov.append((product_flow(*x) - F_P_MIN) >= -1e-6)  # feasible iff F_P >= F_P_MIN
    return float(np.mean(cov)), len(cov)


def h_true(x):
    return float(simulate(*x)['h_ocl'])


# ── run ───────────────────────────────────────────────────────────────────────
COST = make_cost_vectors(N_COST, seed=0)
out = {'surrogate': SURROGATE, 'q': float(Q), 'alpha': ALPHA, 'target': TARGET}
t0 = time.time()

# (1) SANITY
Xtest, ytest = load('test_clean')
cov_rand = float(np.mean((ytest - h_hat_vec(Xtest)) <= Q))
out['sanity_random_coverage'] = cov_rand
print(f"(1) SANITY   random coverage = {cov_rand:.3f}  "
      f"({'PASS' if cov_rand >= TARGET - 0.02 else 'FAIL'} >= {TARGET:.2f})\n")

# (2) coverage at the returned decision
Xstar = [x for cv in COST if (x := solve(cv)) is not None]
Xstar = np.array(Xstar)
cov_sel, n_sel = coverage_at(Xstar)
margin_sel = h_hat_vec(Xstar) + Q
out['reported_selected_coverage'] = cov_sel
out['n_selected'] = n_sel
print(f"(2) coverage at the returned decision = {cov_sel:.3f}  (n={n_sel})  "
      f"[{'BELOW' if cov_sel < TARGET else 'at/above'} target {TARGET:.2f}]")
print(f"    conformal slack h_hat+Q: median={np.median(margin_sel):+.4f}\n")

# (3) MECHANISM
band_lo, band_hi = np.quantile(margin_sel, [0.05, 0.95])
band_lo = min(band_lo, -1e-3)
matched, tries = [], 0
while len(matched) < 400 and tries < 200_000:
    tries += 1
    x = rng_global.uniform(_lo, _hi)
    m = h_hat(x) + Q
    if band_lo <= m <= band_hi:
        matched.append(x)
cov_match, n_match = coverage_at(np.array(matched)) if matched else (float('nan'), 0)
out['mechanism_matched_boundary_coverage'] = cov_match
out['n_matched'] = n_match
out['margin_band'] = [float(band_lo), float(band_hi)]
gap = cov_match - cov_sel
print(f"(3) MECHANISM matched-boundary (not selected) = {cov_match:.3f}  (n={n_match})")
print(f"    selected {cov_sel:.3f}  vs  boundary-not-selected {cov_match:.3f}"
      f"  -> gap {gap:+.3f}  ({'FCS evidence' if gap > 0.03 else 'no FCS-specific gap'})\n")

# (4) DOSE-RESPONSE
feas_tr = Xtr[ytr < np.quantile(ytr, 0.15)]
x_safe  = feas_tr.mean(axis=0)
h_safe  = h_true(x_safe)
print(f"(4) DOSE-RESPONSE  x_safe h={h_safe:+.3f}")
print(f"    {'lambda':>8s}  {'coverage':>9s}  {'mean_dist':>12s}  {'n':>4s}")
dose = []
for lam in [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
    Xl = [x for cv in COST if (x := solve(cv, lam=lam, x_safe=x_safe)) is not None]
    if not Xl:
        dose.append({'lambda': lam, 'coverage': float('nan'), 'mean_dist': float('nan'), 'n': 0})
        print(f"    {lam:>8.1f}  {'nan':>9s}  {'nan':>12s}  {'0':>4s}", flush=True)
        continue
    Xl = np.array(Xl)
    cov_l, n_l = coverage_at(Xl)
    dist = float(np.mean(np.linalg.norm((Xl - x_safe) / RANGE, axis=1)))
    dose.append({'lambda': lam, 'coverage': cov_l, 'mean_dist': dist, 'n': n_l})
    print(f"    {lam:>8.1f}  {cov_l:>9.3f}  {dist:>12.3f}  {n_l:>4d}", flush=True)
out['dose_response'] = dose

print(f"\nElapsed {(time.time()-t0)/60:.1f} min")
fn = f"{RUN}/coverage_wo_{SURROGATE}.json"
json.dump(out, open(fn, 'w'), indent=2)
print(f"Wrote {fn}")

"""
Feedback-Covariate-Shift (FCS) experiment for conformal OCL.

Measures the coverage of a conformal margin at the decisions the optimizer returns.
induces feedback covariate shift (Fannjiang et al. 2022), so the conformal
feasibility guarantee that holds MARGINALLY (on random points) can FAIL at the
optimizer-SELECTED solution. C-MICL's Thm 4.1 needs Assumption 4.1 (coverage
independent of selection); the optimizer is the feedback loop that violates it.

One-sided conformal (matches the OCL constraint h<=0):
    r_i   = h_true_i - h_hat(x_i)
    q_hat = sorted(r_cal)[ceil((1-a)(n+1)) - 1]              (finite-sample)
    embedded constraint:  h_hat(x) + q_hat <= 0
    coverage at x  <=>  h_true(x) <= h_hat(x) + q_hat   <=>  r(x) <= q_hat

Quantities:
  (1) SANITY     coverage on random test points        -- expect >= 1-alpha
  (2) coverage at the optimizer's returned decision x*
  (3) MECHANISM  coverage at random points matched to the selected points'
                 conformal-boundary proximity (near h_hat+q=0 but NOT selected)
                 -- if selected << matched-boundary, that is FCS, not just
                    boundary heteroscedasticity
  (4) DOSE-RESP  add lambda * ||(x-x_safe)/range||^2 to the objective; sweep
                 lambda (selection aggressiveness). Coverage should rise toward
                 the marginal level as lambda grows (selection eased).
"""
import json, glob, sys, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import os
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "benchmarks", "methane_reactor")
RUN = HERE
ALPHA = 0.10
TARGET = 1 - ALPHA
N_COST = 100
SURROGATE = sys.argv[1] if len(sys.argv) > 1 else 'symbolic'   # symbolic|mlp|gbm
CONFORMAL = sys.argv[2] if len(sys.argv) > 2 else 'scalar'      # scalar|normalized
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


# --- surrogate predictor h_hat (symbolic | mlp | gbm) ------------------------
Xtr, ytr = load('train')
Xcal, ycal = load('calibration')

if SURROGATE == 'symbolic':
    _rec = json.load(open(os.path.join(HERE, "symbolic_surrogate_methane.json")))
    p = np.array(_rec['optimized_params'])
    ns = {'np': np}; exec(_rec['function'], ns); eq = ns['equation']
    def h_hat_vec(X): return eq(np.atleast_2d(X), p)
elif SURROGATE == 'mlp':                       # matches C-MICL: 2x32 ReLU, L2=0.01
    _scaler = StandardScaler().fit(Xtr)
    _mlp = MLPRegressor(hidden_layer_sizes=(32, 32), activation='relu', alpha=0.01,
                        max_iter=4000, random_state=0).fit(_scaler.transform(Xtr), ytr)
    def h_hat_vec(X): return _mlp.predict(_scaler.transform(np.atleast_2d(X)))
elif SURROGATE == 'gbm':                       # C-MICL also tests gradient boosting
    _gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                                     random_state=0).fit(Xtr, ytr)
    def h_hat_vec(X): return _gbm.predict(np.atleast_2d(X))
else:
    raise SystemExit(f"unknown surrogate {SURROGATE!r}")

def h_hat(x): return float(h_hat_vec(np.asarray(x, float).reshape(1, 5))[0])

cleanR2 = 1 - np.var(ytr - h_hat_vec(Xtr)) / np.var(ytr)   # train fit, sanity only
r_cal = ycal - h_hat_vec(Xcal)

# --- conformal margin: scalar one-sided (ours) | normalized symmetric (C-MICL)-
# Both embed the upper bound  h_hat(x) + margin(x) <= 0  and we measure realized
# one-sided coverage  P( h_true(x) <= h_hat(x) + margin(x) ).
if CONFORMAL == 'scalar':
    Q = q_finite(r_cal, ALPHA)                       # one-sided residual score
    def margin_vec(X): return np.full(len(np.atleast_2d(X)), Q)
    def margin_sc(x):  return Q
    q_report = float(Q)
    qdesc = f"q_hat(scalar one-sided) = {Q:+.4f}"
elif CONFORMAL == 'normalized':                      # C-MICL: |r|/u_hat, MLP u_hat
    _us = StandardScaler().fit(Xtr)
    _abs_tr = np.abs(ytr - h_hat_vec(Xtr))
    _um = MLPRegressor(hidden_layer_sizes=(32, 32), activation='relu', alpha=0.01,
                       max_iter=4000, random_state=1).fit(_us.transform(Xtr), _abs_tr)
    _UFLOOR = 0.2 * float(np.median(_abs_tr))
    def u_hat_vec(X): return np.maximum(_um.predict(_us.transform(np.atleast_2d(X))), _UFLOOR)
    q_norm = q_finite(np.abs(r_cal) / u_hat_vec(Xcal), ALPHA)
    def margin_vec(X): return q_norm * u_hat_vec(X)
    def margin_sc(x):  return q_norm * float(u_hat_vec(np.asarray(x, float).reshape(1, 5))[0])
    q_report = float(q_norm)
    qdesc = f"q_norm = {q_norm:+.4f} x u_hat(x)  (symmetric normalized)"
else:
    raise SystemExit(f"unknown conformal {CONFORMAL!r}")

print(f"surrogate = {SURROGATE}   conformal = {CONFORMAL}   {qdesc}   "
      f"target coverage = {TARGET:.2f}\n")

RANGE = _hi - _lo
ops = [{'type': 'ineq', 'fun': (lambda x, k=k: -operability_constraints(x)[k])}
       for k in range(7)]


def feasible_ops(x):
    return all(g <= 1e-6 for g in operability_constraints(x))


def solve(cv, lam=0.0, x_safe=None):
    """min objective(+lambda penalty) s.t. operability and h_hat+Q<=0."""
    c_con = {'type': 'ineq', 'fun': lambda x: -(h_hat(x) + margin_sc(x))}
    cons = ops + [c_con]
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
            r = minimize(obj, x0, method='SLSQP', bounds=list(zip(_lo, _hi)),
                         constraints=cons, options={'ftol': 1e-8, 'maxiter': 200})
            if r.success and (br is None or r.fun < br.fun):
                br = r
        except Exception:
            pass
    return None if br is None else br.x


def coverage_at(Xpts):
    """one-sided coverage  mean( h_true(x) <= h_hat(x) + margin(x) )."""
    cov = []
    for x in Xpts:
        cov.append((true_h(x) - h_hat(x)) <= margin_sc(x))
    return float(np.mean(cov)), len(cov)


COST = make_cost_vectors(N_COST, seed=0)
out = {'surrogate': SURROGATE, 'conformal': CONFORMAL, 'q': q_report,
       'alpha': ALPHA, 'target': TARGET}
t0 = time.time()

# (1) SANITY: coverage on random held-out points -------------------------------
Xtest, ytest = load('test_clean')
cov_rand = float(np.mean((ytest - h_hat_vec(Xtest)) <= margin_vec(Xtest)))
out['sanity_random_coverage'] = cov_rand
print(f"(1) SANITY   random test-point coverage = {cov_rand:.3f}  "
      f"({'PASS >=' if cov_rand >= TARGET-0.02 else 'FAIL <'} {TARGET:.2f})\n")

# (2) coverage at the optimizer's returned decision x* ------------------------
Xstar = [x for cv in COST if (x := solve(cv)) is not None]
Xstar = np.array(Xstar)
cov_sel, n_sel = coverage_at(Xstar)
margin_sel = h_hat_vec(Xstar) + margin_vec(Xstar)   # conformal slack at selected (~0)
out['reported_selected_coverage'] = cov_sel
out['n_selected'] = n_sel
print(f"(2) coverage at the returned decision   = {cov_sel:.3f}  (n={n_sel})  "
      f"[{'BELOW' if cov_sel < TARGET else 'at/above'} target {TARGET:.2f}]")
print(f"    selected conformal slack h_hat+margin: median={np.median(margin_sel):+.3f} "
      f"(0 = active boundary)\n")

# (3) MECHANISM: random points matched to selected boundary-proximity ----------
# Selected points sit at h_hat+Q ~ 0. Sample random ops-feasible points whose
# conformal margin falls in the same band, but that were NOT selected.
band_lo, band_hi = np.quantile(margin_sel, [0.05, 0.95])
band_lo = min(band_lo, -1e-3)            # ensure we include just-inside points
matched, tries = [], 0
while len(matched) < 400 and tries < 200000:
    tries += 1
    x = rng_global.uniform(_lo, _hi)
    if not feasible_ops(x):
        continue
    m = h_hat(x) + margin_sc(x)
    if band_lo <= m <= band_hi:
        matched.append(x)
cov_match, n_match = coverage_at(np.array(matched)) if matched else (float('nan'), 0)
out['mechanism_matched_boundary_coverage'] = cov_match
out['n_matched'] = n_match
out['margin_band'] = [float(band_lo), float(band_hi)]
print(f"(3) MECHANISM matched-boundary (not selected) coverage = {cov_match:.3f} "
      f"(n={n_match}, margin band [{band_lo:+.3f},{band_hi:+.3f}])")
gap = cov_match - cov_sel
print(f"    selected {cov_sel:.3f}  vs  boundary-not-selected {cov_match:.3f}  "
      f"-> gap {gap:+.3f}  ({'FCS evidence' if gap > 0.03 else 'no FCS-specific gap (heteroscedastic boundary)'})\n")

# (4) DOSE-RESPONSE: sweep selection aggressiveness ----------------------------
# x_safe = deep-interior feasible centroid (training pts with large benzene margin)
feas_tr = Xtr[ytr < np.quantile(ytr, 0.15)]      # most-feasible 15% (h very negative)
x_safe = feas_tr.mean(axis=0)
print(f"(4) DOSE-RESPONSE  x_safe={np.array2string(x_safe, precision=1)} "
      f"(true_h={true_h(x_safe):+.1f})")
print(f"    {'lambda':>8s} {'coverage':>9s} {'mean||x-x_safe||/range':>24s} {'n':>4s}")
dose = []
for lam in [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
    Xl = [x for cv in COST if (x := solve(cv, lam=lam, x_safe=x_safe)) is not None]
    Xl = np.array(Xl)
    cov_l, n_l = coverage_at(Xl)
    dist = float(np.mean(np.linalg.norm((Xl - x_safe) / RANGE, axis=1)))
    dose.append({'lambda': lam, 'coverage': cov_l, 'mean_dist': dist, 'n': n_l})
    print(f"    {lam:>8.3f} {cov_l:>9.3f} {dist:>24.3f} {n_l:>4d}", flush=True)
out['dose_response'] = dose

print(f"\nElapsed {(time.time()-t0)/60:.1f} min")
out['conformal'] = CONFORMAL
fn = f"{RUN}/coverage_{SURROGATE}_{CONFORMAL}.json"
json.dump(out, open(fn, 'w'), indent=2)
print(f"Wrote {fn}")

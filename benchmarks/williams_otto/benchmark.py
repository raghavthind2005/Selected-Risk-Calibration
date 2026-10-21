"""
LLM-OCL benchmark harness for Williams-Otto (5D full plant).

    min_x  c(x)   s.t.   F_P(x) >= F_P_MIN   (LEARNED),   x in box bounds

c(x): linear cost over normalised decisions (random cost vectors, C-MICL style),
so different instances bind on different variables. The box is the only known
constraint (the flowsheet/recycle is inside the oracle). Oracle = true F_P.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from plant_model import product_flow, simulate, VARIABLE_BOUNDS, F_P_MIN

bounds = VARIABLE_BOUNDS
_lo = np.array([b[0] for b in bounds])
_hi = np.array([b[1] for b in bounds])
_DEFAULT_COST = np.array([1.0, 1.0, 1.0, 1.0, 1.0])


def objective(x, cost_vec=None):
    """Linear cost over normalised decisions; minimised."""
    if cost_vec is None:
        cost_vec = _DEFAULT_COST
    xn = (np.asarray(x, float) - _lo) / (_hi - _lo)
    return float(np.dot(cost_vec, xn))


def true_h(x):
    return float(simulate(*x)['h_ocl'])


def make_cost_vectors(n=100, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.uniform(-4.0, 4.0, size=(n, 5))
    C[C < 0] /= 10.0
    return C


def solve_oracle(cost_vec=None, n_starts=12, seed=1):
    if cost_vec is None:
        cost_vec = _DEFAULT_COST
    rng = np.random.default_rng(seed)
    cons = [{'type': 'ineq', 'fun': lambda x: product_flow(*x) - F_P_MIN}]
    best_val, best_x = np.inf, None
    for _ in range(n_starts):
        x0 = rng.uniform(_lo, _hi)
        try:
            r = minimize(lambda x: objective(x, cost_vec), x0, method='SLSQP',
                         bounds=list(zip(_lo, _hi)), constraints=cons,
                         options={'ftol': 1e-7, 'maxiter': 100})
            if r.success and r.fun < best_val and product_flow(*r.x) >= F_P_MIN - 1e-2:
                best_val, best_x = float(r.fun), r.x.copy()
        except Exception:
            continue
    if best_x is None:
        return {'success': False}
    return {'success': True, 'x_star': best_x.tolist(), 'obj_value': best_val,
            'F_P': float(product_flow(*best_x))}

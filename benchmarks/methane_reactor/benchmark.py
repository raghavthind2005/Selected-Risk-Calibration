"""
LLM-OCL benchmark module for `methane_reactor` — EXACT C-MICL replication.

Mirrors the optimization problem from the C-MICL `regression.py` `build_base_model`:

    min_x  c(x)
    s.t.   FC6H6(x) >= 50               (LEARNED constraint; ODE ground truth)
           10*dt <= L  <= 150*dt        (geometry)
           0.75*vHe <= v0 <= 3*vHe      (gas feed ratio)
           20*L  <= v0 <= 120*L         (residence time)
           v0 <= 1.1*T                  (flow vs temperature)
           x in box bounds

c(x) is a random linear cost over the C-MICL-SCALED variables
[v0/100, vHe/100, T/100, dt, L/100], drawn as uniform(-4, 4) with negative
weights shrunk by /10 (exactly as in C-MICL regression.py).

x = [v0, vHe, T, dt, L]  in PHYSICAL units (matching ode_model.VARIABLE_BOUNDS).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from ode_model import benzene_flow, simulate, VARIABLE_BOUNDS, MIN_BENZENE

# ---------------------------------------------------------------------------
# Bounds  (physical units: v0, vHe, T, dt, L)
# ---------------------------------------------------------------------------
bounds = VARIABLE_BOUNDS
_lo = np.array([b[0] for b in bounds])
_hi = np.array([b[1] for b in bounds])

# C-MICL internal scaling: v0,vHe,T,L divided by 100; dt unchanged.
_SCALE = np.array([100.0, 100.0, 100.0, 1.0, 100.0])


# ---------------------------------------------------------------------------
# Objective (C-MICL scaled linear cost)
# ---------------------------------------------------------------------------
# Default cost vector: a representative draw from the C-MICL distribution.
_DEFAULT_COST = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

def objective(x: np.ndarray, cost_vec: np.ndarray | None = None) -> float:
    """Linear cost over C-MICL-scaled variables:  sum_i c_i * (x_i / scale_i)."""
    if cost_vec is None:
        cost_vec = _DEFAULT_COST
    x = np.asarray(x, dtype=float)
    return float(np.dot(cost_vec, x / _SCALE))


# ---------------------------------------------------------------------------
# True constraint (ODE oracle):  h(x) = 50 - FC6H6(x) <= 0
# ---------------------------------------------------------------------------
def true_h(x: np.ndarray) -> float:
    return float(simulate(*x)['h_ocl'])


# ---------------------------------------------------------------------------
# Known operability constraints, returned in g(x) <= 0 form.
# (These are KNOWN to the optimiser — not learned — exactly as in C-MICL.)
# ---------------------------------------------------------------------------
def operability_constraints(x: np.ndarray) -> list[float]:
    v0, vHe, T, dt, L = x
    return [
        10.0 * dt - L,        # L >= 10*dt
        L - 150.0 * dt,       # L <= 150*dt
        0.75 * vHe - v0,      # v0 >= 0.75*vHe
        v0 - 3.0 * vHe,       # v0 <= 3*vHe
        20.0 * L - v0,        # v0 >= 20*L
        v0 - 120.0 * L,       # v0 <= 120*L
        v0 - 1.1 * T,         # v0 <= 1.1*T
    ]


def _scipy_operability():
    """Operability constraints as scipy ineq dicts (fun >= 0)."""
    return [{'type': 'ineq', 'fun': (lambda x, k=k: -operability_constraints(x)[k])}
            for k in range(7)]


# ---------------------------------------------------------------------------
# C-MICL cost vectors: uniform(-4, 4), negatives shrunk /10, seeded.
# ---------------------------------------------------------------------------
def make_cost_vectors(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = rng.uniform(-4.0, 4.0, size=(n, 5))
    C[C < 0] /= 10.0
    return C


# ---------------------------------------------------------------------------
# Oracle solve: min c(x) s.t. true benzene >= 50 + operability + box
# ---------------------------------------------------------------------------
def solve_oracle(cost_vec: np.ndarray | None = None,
                 n_starts: int = 12, seed: int = 1) -> dict:
    if cost_vec is None:
        cost_vec = _DEFAULT_COST
    rng = np.random.default_rng(seed)
    cons = _scipy_operability() + [
        {'type': 'ineq', 'fun': lambda x: benzene_flow(*x) - MIN_BENZENE}]
    best_val, best_x = np.inf, None
    for _ in range(n_starts):
        x0 = rng.uniform(_lo, _hi)
        try:
            r = minimize(lambda x: objective(x, cost_vec), x0, method='SLSQP',
                         bounds=list(zip(_lo, _hi)), constraints=cons,
                         options={'ftol': 1e-7, 'maxiter': 100})
            if r.success and r.fun < best_val:
                bz = benzene_flow(*r.x)
                if bz >= MIN_BENZENE - 1e-3:
                    best_val, best_x = float(r.fun), r.x.copy()
        except Exception:
            continue
    if best_x is None:
        return {'success': False}
    return {'success': True, 'x_star': best_x.tolist(), 'obj_value': best_val,
            'benzene': float(benzene_flow(*best_x))}


# ---------------------------------------------------------------------------
# LLM-OCL solve: min c(x) s.t. h_hat(x)+q_hat <= 0 + operability + box
# ---------------------------------------------------------------------------
def solve_llm(equation_fn, params: np.ndarray, q_hat: float,
              cost_vec: np.ndarray | None = None,
              n_starts: int = 16, seed: int = 0) -> dict:
    """Solve the LLM-OCL problem with the learned symbolic constraint.

    equation_fn(X, params) takes X of shape (n,5) and returns (n,) predictions.
    """
    if cost_vec is None:
        cost_vec = _DEFAULT_COST
    rng = np.random.default_rng(seed)

    def h_hat_scalar(x):
        return float(equation_fn(np.asarray(x, float).reshape(1, 5), params)[0])

    cons = _scipy_operability() + [
        {'type': 'ineq', 'fun': lambda x: -(h_hat_scalar(x) + q_hat)}]
    best_val, best_x = np.inf, None
    for _ in range(n_starts):
        x0 = rng.uniform(_lo, _hi)
        try:
            r = minimize(lambda x: objective(x, cost_vec), x0, method='SLSQP',
                         bounds=list(zip(_lo, _hi)), constraints=cons,
                         options={'ftol': 1e-8, 'maxiter': 200})
            if r.success and r.fun < best_val:
                best_val, best_x = float(r.fun), r.x.copy()
        except Exception:
            continue
    if best_x is None:
        return {'success': False}
    return {'success': True, 'x_star': best_x.tolist(), 'obj_value': best_val,
            'h_hat': h_hat_scalar(best_x),
            'true_h': true_h(best_x),
            'true_benzene': float(benzene_flow(*best_x)),
            'true_feasible': bool(benzene_flow(*best_x) >= MIN_BENZENE)}

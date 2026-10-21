"""
Williams-Otto FULL PLANT (with recycle) — exact port of the grey-box benchmark
in arXiv 2511.18998 Annexe AN6 (the 5-decision version used for benchmarking,
NOT the 2D textbook reactor).

Flowsheet: feeds F_A, F_B + recycle F_R -> CSTR -> heat exchanger -> decanter
(removes waste gas G) -> distillation column (product F_P + bottoms; 10 wt% of
E remains, azeotrope) -> split into purge F_purge and recycle F_R back to reactor.

Reactions:  A+B->C,  B+C->P+E,  C+P->G
Rates:  y1=a1 e^(-120/T) xA xB V rho ;  y2=a2 e^(-150/T) xB xC V rho ;
        y3=a3 e^(-200/T) xP xC V rho      (T scaled, in [5.8,6.8]; rho=50)

The recycle creates a fixed-point: effluent flows depend on reaction (-> on
compositions -> on effluent), solved here numerically. This is the complexity
the 2D version lacks.

OCL framing (parallel to C-MICL's methane benzene>=50): the LEARNED constraint
is a minimum-production requirement  F_P(x) >= F_P_MIN, i.e.
    h(x) = F_P_MIN - F_P(x) <= 0
over the 5 decisions x = (V, T, F_A, F_B, eta). Objective = minimize feed+energy
operating cost.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve

# Kinetics / constants (exact, AN6)
A1, A2, A3 = 5.9755e9, 2.5962e12, 9.6283e15
RHO = 50.0

VARIABLE_NAMES = ['V_reactor', 'T_scaled', 'FA_feed', 'FB_feed', 'eta_recycle']
VARIABLE_BOUNDS = [
    (0.03, 0.10),   # V    reactor volume
    (5.8,  6.8),    # T    temperature (scaled)
    (1.0,  5.0),    # F_A  fresh feed of A   (>=1 per AN6)
    (1.0,  5.0),    # F_B  fresh feed of B   (>=1 per AN6)
    (0.10, 0.90),   # eta  purge/recycle ratio (bounded off 0: eta->0 blows up recycle)
]

# OCL constraint threshold: minimum product flow (tuned for ~50% feasibility).
F_P_MIN = 0.64


def _rates(Feff, V, T):
    x = Feff / Feff.sum()
    xA, xB, xC, xE, xP, xG = x
    k1 = A1 * np.exp(-120.0 / T)
    k2 = A2 * np.exp(-150.0 / T)
    k3 = A3 * np.exp(-200.0 / T)
    y1 = k1 * xA * xB * V * RHO
    y2 = k2 * xB * xC * V * RHO
    y3 = k3 * xP * xC * V * RHO
    return y1, y2, y3


def solve_plant(V, T, FA, FB, eta):
    """Solve the recycle fixed-point. Returns (F_eff[6], converged?)."""
    def resid(Feff):
        Feff = np.maximum(Feff, 1e-9)
        y1, y2, y3 = _rates(Feff, V, T)
        FRE = (1 - eta) * Feff[3]
        return [
            Feff[0] * eta - (FA - y1),
            Feff[1] * eta - (FB - (y1 + y2)),
            Feff[2] * eta - (2 * y1 - 2 * y2 - y3),
            Feff[3] * eta - (2 * y2),
            Feff[4] - (0.1 * FRE + y2 - 0.5 * y3),
            Feff[5] - (1.5 * y3),
        ]
    sol, info, ier, msg = fsolve(resid, [FA / eta, FB / eta, 0.1, 0.1, 0.1, 0.1],
                                 full_output=True)
    ok = (ier == 1) and np.all(sol > -1e-4) and np.all(np.isfinite(sol))
    return sol, ok


def product_flow(V, T, FA, FB, eta):
    """Product F_P (component P effluent). NaN if the plant fails to converge."""
    sol, ok = solve_plant(V, T, FA, FB, eta)
    return float(sol[4]) if ok else float('nan')


def simulate(V, T, FA, FB, eta):
    fp = product_flow(V, T, FA, FB, eta)
    h = F_P_MIN - fp if np.isfinite(fp) else float('nan')
    return dict(F_P=fp, h_ocl=h, feasible=(np.isfinite(h) and h <= 0.0))


def h_true_vec(X: np.ndarray) -> np.ndarray:
    """h(x) = F_P_MIN - F_P(x) for X of shape (n,5)."""
    return np.array([simulate(*row)['h_ocl'] for row in X])


if __name__ == '__main__':
    s, ok = solve_plant(0.06, 6.3, 3.0, 3.0, 0.3)
    print(f"Sample (V=.06,T=6.3,FA=3,FB=3,eta=.3): F_eff={np.round(s,3)} ok={ok}")
    print(f"  F_P={s[4]:.3f}  h={F_P_MIN - s[4]:+.3f}  feasible={F_P_MIN - s[4] <= 0}")

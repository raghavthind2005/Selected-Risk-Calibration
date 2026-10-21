"""
Direct methane aromatization (DMA) membrane reactor — EXACT C-MICL replication.

This is a faithful scipy port of the ground-truth ODE used in the regression
case study of C-MICL (Ovalle et al., NeurIPS 2025), taken verbatim from the
authors' released `regression.py` (`dma_mr_model` + `evaluate_ode_system`).
Verified to reproduce their Appendix Table 1 benzene-flow values within ~1
unit (their published points are noisy; the clean ODE matches to ~1-2%).

Single H2-selective membrane with helium sweep; NO air/O2 oxidative pathway.
Membrane permeance Q and selectivity are FIXED constants (not decision vars).

Decision variables (5), with C-MICL bounds:
    v0   inlet methane volumetric flow   U(450, 1500)        cm^3/h
    vHe  helium sweep volumetric flow     U(450, 1500)        cm^3/h
    T    operating temperature            U(997.18, 1348.12)  K
    dt   tube diameter                    U(0.5, 2.0)         cm
    L    reactor length                   U(10, 100)          cm

Output: benzene outlet flow FC6H6.  OCL constraint:  h(x) = 50 - FC6H6 <= 0
(feasible when benzene production >= 50, i.e. a minimum-throughput requirement).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

# ---------------------------------------------------------------------------
# Decision variable specification (anonymised as x1..x5 in spec/data)
# ---------------------------------------------------------------------------
VARIABLE_NAMES = ['v0_cm3_h', 'vHe_cm3_h', 'T_K', 'dt_cm', 'L_cm']
VARIABLE_BOUNDS = [
    (450.0,   1500.0),    # v0   methane feed flow      [cm^3/h]
    (450.0,   1500.0),    # vHe  helium sweep flow       [cm^3/h]
    (997.18,  1348.12),   # T    temperature             [K]
    (0.5,     2.0),       # dt   tube diameter           [cm]
    (10.0,    100.0),     # L    reactor length          [cm]
]

# OCL constraint threshold: benzene outlet flow must be >= MIN_BENZENE.
MIN_BENZENE = 50.0


# ---------------------------------------------------------------------------
# Ground-truth ODE (verbatim port of C-MICL dma_mr_model)
# ---------------------------------------------------------------------------
def _dma_mr_model(F, z, dt, v_He, v0, F_He, Ft0):
    """8-state plug-flow membrane reactor RHS, dF/dz along reactor length."""
    R = 8.314e6           # ideal gas constant [cm^3*Pa/mol*K]
    k1 = 0.04             # reaction-1 forward rate constant
    k1_Inv = 6.40e6       # reaction-1 reverse rate constant
    k2 = 4.20             # reaction-2 forward rate constant
    k2_Inv = 56.38        # reaction-2 reverse rate constant
    Q = 3600 * 0.01e-4    # H2-membrane permeation coefficient (FIXED)
    selec = 1500          # membrane selectivity (FIXED)
    Pt = 101325.0         # tube-side pressure  [Pa]
    Ps = 101325.0         # shell-side pressure [Pa]

    At = 0.25 * np.pi * (dt ** 2)                # tube cross-section
    F = np.where(F <= 1e-9, 1e-9, F)             # avoid division by zero

    Ft = F[0:4].sum()                            # tube-side total molar flow
    Fs = F[4:].sum() + F_He                      # shell-side total molar flow
    v = v0 * (Ft / Ft0)                          # volumetric flow (flow-corrected)
    C = F[:4] / v                                # tube-side concentrations

    P0t, P1t, P2t, P3t = (Pt / 101325) * (F[0:4] / Ft)
    P0s, P1s, P2s, P3s = (Ps / 101325) * (F[4:] / Fs)

    r0 = 3600 * k1 * C[0] * (1 - ((k1_Inv * C[1] * C[2] ** 2) / (k1 * C[0] ** 2)))
    r0 = np.where(C[0] <= 1e-9, 0.0, r0)
    r1 = 3600 * k2 * C[1] * (1 - ((k2_Inv * C[3] * C[2] ** 3) / (k2 * C[1] ** 3)))
    r1 = np.where(C[1] <= 1e-9, 0.0, r1)

    eff = 0.9
    vb = 0.5
    Cat = (1 - vb) * eff                         # effective catalyst fraction

    return np.array([
        -Cat * r0 * At - (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * np.pi * dt,
        0.5 * Cat * r0 * At - Cat * r1 * At - (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * np.pi * dt,
        Cat * r0 * At + Cat * r1 * At - Q * ((P2t ** 0.25) - (P2s ** 0.25)) * np.pi * dt,
        (1 / 3) * Cat * r1 * At - (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * np.pi * dt,
        (Q / selec) * ((P0t ** 0.25) - (P0s ** 0.25)) * np.pi * dt,
        (Q / selec) * ((P1t ** 0.25) - (P1s ** 0.25)) * np.pi * dt,
        Q * ((P2t ** 0.25) - (P2s ** 0.25)) * np.pi * dt,
        (Q / selec) * ((P3t ** 0.25) - (P3s ** 0.25)) * np.pi * dt,
    ])


def benzene_flow(v0: float, v_He: float, T: float, dt: float, L: float) -> float:
    """Outlet benzene flow FC6H6 (C-MICL units). Verbatim port of
    `evaluate_ode_system`."""
    R = 8.314e6
    MM_B = 78.00          # benzene molecular weight [g/mol]
    Pt = 101325.0
    Ps = 101325.0

    Ft0 = Pt * v0 / (R * T)          # inlet methane molar flow
    F_He = Ps * v_He / (R * T)       # helium sweep molar flow
    y0 = np.hstack((Ft0, np.zeros(7)))
    z = np.linspace(0.0, L, 2000)
    try:
        F = odeint(_dma_mr_model, y0, z, args=(dt, v_He, v0, F_He, Ft0),
                   rtol=1e-10, atol=1e-10)
    except Exception:
        return float('nan')
    return float((F[-1, 3] * 1000) * MM_B)


# ---------------------------------------------------------------------------
# OCL wrappers
# ---------------------------------------------------------------------------
def simulate(v0: float, v_He: float, T: float, dt: float, L: float) -> dict:
    """Return benzene flow and the OCL constraint h = MIN_BENZENE - FC6H6."""
    fb = benzene_flow(v0, v_He, T, dt, L)
    h = MIN_BENZENE - fb if np.isfinite(fb) else float('nan')
    return dict(FC6H6=fb, h_ocl=h, feasible=(np.isfinite(h) and h <= 0.0))


def h_true_vec(X: np.ndarray) -> np.ndarray:
    """h(x) = 50 - FC6H6(x) for array X of shape (n, 5). Returns (n,)."""
    return np.array([simulate(*row)['h_ocl'] for row in X])


# ---------------------------------------------------------------------------
# Self-test: reproduce C-MICL Appendix Table 1 (their noisy published points)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    table1 = [
        (1157.75, 845.25, 1272.93, 0.57, 30.67, 37.45),
        (773.89, 1319.69, 998.56, 0.75, 88.10, 42.57),
        (1484.77, 476.68, 1201.21, 1.64, 77.38, 40.70),
        (817.68, 536.65, 1123.28, 0.72, 86.07, 28.26),
        (1162.73, 1262.03, 1256.59, 1.95, 92.29, 36.58),
    ]
    print('=== Verify vs C-MICL Table 1 (benzene FC6H6) ===')
    for v0, vHe, T, dt, L, truth in table1:
        got = benzene_flow(v0, vHe, T, dt, L)
        print(f'  theirs={truth:6.2f}  ours={got:6.2f}  diff={got - truth:+.2f}')

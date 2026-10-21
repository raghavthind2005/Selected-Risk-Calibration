"""Surrogate models for selected-feasibility experiments.

Provides two surrogate families:

  symbolic_pinball — Polynomial features + quantile regression at tau = 1-alpha.
      Emulates the LLM-OCL pipeline's asymmetric pinball loss without requiring
      an LLM-discovered expression. Biases predictions upward, which helps
      one-sided conformal coverage.

  mlp — Standard MLP regressor with MSE loss.
      Represents a flexible black-box surrogate that may under-predict in
      regions far from training data, creating pockets the optimizer can
      exploit.

  mlp_early_stop — Larger MLP regressor with deterministic Adam early stopping.
      Used for Williams-Otto MLP coverage repair while preserving the original
      mlp profile for prior-run provenance.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_surrogate(
    surrogate_type: str,
    alpha: float = 0.10,
    seed: int = 42,
    n_vars: int = 5,
) -> Pipeline:
    """Build a surrogate model.

    Args:
        surrogate_type: 'symbolic_pinball', 'mlp', or 'mlp_early_stop'.
        alpha: target miscoverage rate; pinball quantile = 1 - alpha.
        seed: random seed for reproducibility.
        n_vars: number of input features (used for polynomial degree selection).

    Returns:
        A sklearn Pipeline with fit/predict.
    """
    if surrogate_type == 'symbolic_pinball':
        # Degree 2 polynomial + quantile regression at tau=1-alpha.
        # This biases predictions upward (over-predict h), which is the
        # same asymmetric loss principle LLM-OCL uses during SR search.
        quantile = max(0.51, min(0.99, 1.0 - alpha))
        # For high-dim problems (>=5 vars), degree-2 interaction terms
        # can blow up; cap at degree 2 or n_vars-dependent.
        degree = 2 if n_vars <= 6 else 1
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=True,
                                        interaction_only=False)),
            ('scaler', StandardScaler()),
            ('reg', QuantileRegressor(
                quantile=quantile, alpha=0.0, solver='highs',
            )),
        ])
    elif surrogate_type == 'mlp':
        # Two-hidden-layer MLP, matches C-MICL baseline sizing.
        # LBFGS solver for deterministic convergence on small data.
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='lbfgs',
                max_iter=2000,
                random_state=seed,
                tol=1e-6,
            )),
        ])
    elif surrogate_type in {'mlp_early_stop', 'mlp_repaired'}:
        # Coverage-repair profile. The wider MLP and explicit validation
        # stopping fixed the Williams-Otto random coverage precondition in
        # local smoke tests without changing the downstream selected metric.
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                max_iter=1500,
                random_state=seed,
                tol=1e-5,
            )),
        ])
    else:
        raise ValueError(f'Unknown surrogate_type: {surrogate_type!r}')


def compute_residuals(
    surrogate: Pipeline,
    X: np.ndarray,
    h_true: np.ndarray,
) -> np.ndarray:
    """One-sided residuals r_i = h_true_i - h_hat(x_i)."""
    h_pred = surrogate.predict(X).reshape(-1)
    h_true = np.asarray(h_true, dtype=float).reshape(-1)
    return h_true - h_pred


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """One-sided split-conformal upper quantile with finite-sample correction.

    Returns q such that at least ceil((1-alpha)(n+1))/n of residuals
    are <= q.  See conformal.py:conformal_quantile for the original.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    if n < 1:
        raise ValueError('Need at least 1 residual')
    rank = int(np.ceil((1.0 - alpha) * (n + 1)))
    if rank > n:
        return float('inf')
    return float(np.sort(residuals)[rank - 1])


def empirical_coverage(residuals: np.ndarray, q_hat: float) -> float:
    """Fraction of residuals <= q_hat."""
    return float(np.mean(residuals <= q_hat))

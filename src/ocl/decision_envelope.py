"""Decision-aware max-affine residual upper envelopes for OCL corrections.

Motivation
----------
An earlier deployment used ``h_hat(x) + q_hat + gamma * s_knn(x) <= 0`` with a
*k*-nearest neighbour local scale.  That constraint turned out not to be usable
as a measurement instrument: ``s_knn`` is piecewise constant and discontinuous, so
SLSQP sees a zero (or meaningless) gradient almost everywhere, and adjacent
``gamma`` values — whose feasible sets are provably nested — produce *improving*
optimal values on thousands of objectives.

This module replaces the correction shape with an object the solver can
actually differentiate while keeping the same deployed contract:

    g(x) = h_hat(x) + q_hat + strength * c(x) <= 0
    c(x) = (1 - floor_weight) * E_theta(x) + floor_weight * floor_level

``E_theta`` is a smoothed max-affine upper envelope of the residual excess

    e(x) = h_true(x) - h_hat(x) - q_hat

fitted by weighted pinball regression on the proposal split.  ``e`` is the right
target: when the deployed constraint is active at the returned decision ``x*``,
``h_true(x*) = e(x*) - strength * c(x*)``, so the decision is oracle-feasible
exactly when the correction dominates the residual excess it walked into.

Properties that the downstream contract depends on
--------------------------------------------------
* ``c(x) >= 0`` everywhere (the log-sum-exp includes a zero piece, and the
  floor level is non-negative), so the feasible sets are nested in ``strength``.
* ``E_theta`` is ``C^inf`` and convex with an exact analytic gradient.
* ``E_theta(x) >= max(0, max_l a_l . x + b_l)``: the smoothing is an upper
  bound of the exact max-affine envelope, never a relaxation of it.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "CandidateMargin",
    "ConstantFloorCorrection",
    "DeployedCorrection",
    "SmoothMaxAffineEnvelope",
    "fit_constant_upper_level",
    "fit_smooth_max_affine_envelope",
    "pinball_loss",
    "smooth_pinball_loss",
    "weighted_pinball_loss",
]


def _as2d(X: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(X, dtype=float))


def _softplus(z: np.ndarray) -> np.ndarray:
    """``log(1 + exp(z))`` without overflow."""
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def pinball_loss(residual: np.ndarray, level: float) -> np.ndarray:
    """Exact quantile (pinball) loss ``rho_level(u)`` for ``u = y - pred``."""
    u = np.asarray(residual, dtype=float)
    return level * np.maximum(u, 0.0) + (1.0 - level) * np.maximum(-u, 0.0)


def smooth_pinball_loss(
    residual: np.ndarray, level: float, epsilon: float
) -> np.ndarray:
    """``level * u + epsilon * softplus(-u / epsilon)``.

    Converges to :func:`pinball_loss` as ``epsilon -> 0`` and is ``C^inf`` for
    every ``epsilon > 0``, so the fit has an exact gradient everywhere.
    """
    u = np.asarray(residual, dtype=float)
    return level * u + epsilon * _softplus(-u / epsilon)


class SmoothMaxAffineEnvelope:
    """``E(x) = tau * log(1 + sum_l exp((a_l . z(x) + b_l) / tau))``.

    ``z(x) = (x - lower) / span`` maps the decision box to the unit cube, so the
    slope regulariser has the same meaning on every coordinate.  The constant
    zero piece inside the log-sum-exp is what makes the envelope non-negative.
    """

    def __init__(
        self,
        slopes: np.ndarray,
        intercepts: np.ndarray,
        smoothing: float,
        lower: np.ndarray,
        span: np.ndarray,
    ) -> None:
        self.slopes = np.asarray(slopes, dtype=float).reshape(
            -1, np.asarray(lower).size
        )
        self.intercepts = np.asarray(intercepts, dtype=float).reshape(-1)
        if self.slopes.shape[0] != self.intercepts.shape[0]:
            raise ValueError("slopes and intercepts must have equal length")
        if float(smoothing) <= 0.0:
            raise ValueError("smoothing must be positive")
        self.smoothing = float(smoothing)
        self.lower = np.asarray(lower, dtype=float).reshape(-1)
        self.span = np.maximum(
            np.asarray(span, dtype=float).reshape(-1), 1e-12
        )

    # -- geometry ----------------------------------------------------------

    @property
    def n_pieces(self) -> int:
        return int(self.slopes.shape[0])

    def scaled(self, X: np.ndarray) -> np.ndarray:
        return (_as2d(X) - self.lower) / self.span

    def _affine(self, X: np.ndarray) -> np.ndarray:
        """Affine piece values with the constant zero piece appended."""
        z = self.scaled(X)
        values = z @ self.slopes.T + self.intercepts
        return np.concatenate(
            [np.zeros((values.shape[0], 1), dtype=float), values], axis=1
        )

    def exact_max_affine(self, X: np.ndarray) -> np.ndarray:
        """``max(0, max_l a_l . z + b_l)`` — the unsmoothed envelope."""
        return np.max(self._affine(X), axis=1)

    def batch(self, X: np.ndarray) -> np.ndarray:
        values = self._affine(X)
        shift = np.max(values, axis=1, keepdims=True)
        weighted = np.exp((values - shift) / self.smoothing)
        return np.asarray(
            shift.reshape(-1)
            + self.smoothing * np.log(np.sum(weighted, axis=1)),
            dtype=float,
        )

    def __call__(self, x: np.ndarray) -> float:
        return float(self.batch(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def softmax_weights(self, X: np.ndarray) -> np.ndarray:
        values = self._affine(X)
        shift = np.max(values, axis=1, keepdims=True)
        weighted = np.exp((values - shift) / self.smoothing)
        return weighted / np.sum(weighted, axis=1, keepdims=True)

    def gradient_batch(self, X: np.ndarray) -> np.ndarray:
        """Exact ``dE/dx``; the zero piece contributes no slope."""
        weights = self.softmax_weights(X)[:, 1:]
        return (weights @ self.slopes) / self.span

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.gradient_batch(np.atleast_2d(np.asarray(x, dtype=float)))[0],
            dtype=float,
        )

    # -- receipts ----------------------------------------------------------

    def parameter_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.slopes.reshape(-1), self.intercepts.reshape(-1)]
        )

    def parameter_sha256(self) -> str:
        payload = np.concatenate(
            [
                self.parameter_vector(),
                np.array([self.smoothing], dtype=float),
                self.lower,
                self.span,
            ]
        ).astype("<f8")
        return hashlib.sha256(payload.tobytes()).hexdigest()

    def spec(self) -> dict[str, Any]:
        return {
            "form": "smooth_max_affine_upper_envelope",
            "expression": (
                "E(x) = tau * log(1 + sum_l exp((a_l . z + b_l)/tau)), "
                "z = (x - lower) / span"
            ),
            "n_pieces": self.n_pieces,
            "smoothing": self.smoothing,
            "slopes": self.slopes.tolist(),
            "intercepts": self.intercepts.tolist(),
            "lower": self.lower.tolist(),
            "span": self.span.tolist(),
            "parameter_sha256": self.parameter_sha256(),
            "smoothing_upper_bound_gap": float(
                self.smoothing * np.log(self.n_pieces + 1)
            ),
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> SmoothMaxAffineEnvelope:
        return cls(
            np.asarray(spec["slopes"], dtype=float),
            np.asarray(spec["intercepts"], dtype=float),
            float(spec["smoothing"]),
            np.asarray(spec["lower"], dtype=float),
            np.asarray(spec["span"], dtype=float),
        )


class DeployedCorrection:
    """``c(x) = (1 - w) * E(x) + w * floor_level``, scaled by ``strength``.

    ``strength`` is the single monotone axis of the deployed family: because
    ``c >= 0``, increasing it can only shrink the feasible set, which is the
    nestedness property the kNN correction was measured to violate.
    """

    def __init__(
        self,
        envelope: SmoothMaxAffineEnvelope,
        floor_level: float,
        floor_weight: float,
        strength: float,
    ) -> None:
        if not 0.0 <= float(floor_weight) <= 1.0:
            raise ValueError("floor_weight must lie in [0, 1]")
        if float(floor_level) < 0.0:
            raise ValueError("floor_level must be non-negative")
        if float(strength) < 0.0:
            raise ValueError("strength must be non-negative")
        self.envelope = envelope
        self.floor_level = float(floor_level)
        self.floor_weight = float(floor_weight)
        self.strength = float(strength)

    def batch(self, X: np.ndarray) -> np.ndarray:
        blended = (
            1.0 - self.floor_weight
        ) * self.envelope.batch(X) + self.floor_weight * self.floor_level
        return self.strength * blended

    def __call__(self, x: np.ndarray) -> float:
        return float(self.batch(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return (
            self.strength
            * (1.0 - self.floor_weight)
            * self.envelope.gradient(x)
        )

    def spec(self) -> dict[str, Any]:
        return {
            "form": "convex_combination_of_envelope_and_additive_floor",
            "expression": (
                "c(x) = strength * ((1 - floor_weight) * E(x) "
                "+ floor_weight * floor_level)"
            ),
            "floor_level": self.floor_level,
            "floor_weight": self.floor_weight,
            "strength": self.strength,
            "envelope": self.envelope.spec(),
        }


class ConstantFloorCorrection:
    """``c(x) = strength * level`` — the same deployed family without a shape.

    This is the scalar counterpart of :class:`DeployedCorrection`: identical
    contract (non-negative, nested in ``strength``, exactly differentiable) with
    the ``x``-dependence removed, so a frontier difference between the two is a
    property of the shape and not of the deployment mechanics.

    ``DeployedCorrection(envelope, level, floor_weight=1.0, strength)`` computes
    the same number — ``(1 - 1) * E(x) + 1 * level`` is ``level`` in exact IEEE
    arithmetic — so an arm built here reproduces the ``floor_weight == 1``
    ablation bit for bit while being constructible without an envelope.
    """

    def __init__(self, level: float, strength: float) -> None:
        if float(level) < 0.0:
            raise ValueError("floor level must be non-negative")
        if float(strength) < 0.0:
            raise ValueError("strength must be non-negative")
        self.level = float(level)
        self.strength = float(strength)

    def batch(self, X: np.ndarray) -> np.ndarray:
        return np.full(_as2d(X).shape[0], self.strength * self.level, dtype=float)

    def __call__(self, x: np.ndarray) -> float:  # noqa: ARG002 - constant by design
        return float(self.strength * self.level)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(x, dtype=float).reshape(-1).size, dtype=float)

    def parameter_sha256(self) -> str:
        payload = np.array([self.level], dtype="<f8")
        return hashlib.sha256(payload.tobytes()).hexdigest()

    def spec(self) -> dict[str, Any]:
        return {
            "form": "constant_additive_floor",
            "expression": "c(x) = strength * level",
            "level": self.level,
            "strength": self.strength,
            "parameter_sha256": self.parameter_sha256(),
        }


class CandidateMargin:
    """``m(x) = eta * ((1 - theta) + theta * sigma_hat(x) / sigma_bar)``.

    The candidate margin family the certification study locks one member of.
    ``sigma_hat(x) > 0`` is a local-uncertainty estimate and ``sigma_bar`` its
    median over the calibration inputs, so the bracket has median one and
    ``eta`` is the margin's overall level on the scale of the constraint.

    The shape ``theta`` in ``[0, 1]`` decides how closely the margin follows
    ``sigma_hat``: ``theta = 0`` is a constant margin, which at ``eta = q_hat``
    is the plain conformal deployment, and ``theta = 1`` a margin proportional
    to the uncertainty estimate.  Deploying ``h_hat(x) + m(x) <= 0`` is the
    correction ``g = m - q_hat`` in the deployment of
    :class:`DeployedCorrection`.

    ``sigma_hat`` is any callable mapping an ``(n, d)`` array to ``n`` positive
    values; the family is a reparametrisation, not a model, so nothing here
    depends on how that estimate was fitted.
    """

    def __init__(self, theta: float, eta: float,
                 sigma_hat: Any, sigma_bar: float) -> None:
        if not 0.0 <= float(theta) <= 1.0:
            raise ValueError(f"shape theta must lie in [0, 1]; got {theta!r}")
        if float(eta) < 0.0:
            raise ValueError(f"size eta must be non-negative; got {eta!r}")
        if float(sigma_bar) <= 0.0:
            raise ValueError(f"sigma_bar must be positive; got {sigma_bar!r}")
        self.theta = float(theta)
        self.eta = float(eta)
        self.sigma_bar = float(sigma_bar)
        self._sigma_hat = sigma_hat

    def batch(self, X: np.ndarray) -> np.ndarray:
        X = _as2d(X)
        if self.theta == 0.0:            # no need to evaluate the estimate
            return np.full(X.shape[0], self.eta, dtype=float)
        s = np.asarray(self._sigma_hat(X), dtype=float).reshape(-1)
        if s.size != X.shape[0]:
            raise ValueError(f"sigma_hat returned {s.size} values for "
                             f"{X.shape[0]} inputs")
        return self.eta * ((1.0 - self.theta) + self.theta * s / self.sigma_bar)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.batch(np.atleast_2d(np.asarray(x, dtype=float)))[0])

    def spec(self) -> dict[str, Any]:
        return {
            "form": "candidate_margin",
            "expression": "m(x) = eta * ((1 - theta) + theta * sigma_hat(x) / sigma_bar)",
            "theta": self.theta,
            "eta": self.eta,
            "sigma_bar": self.sigma_bar,
        }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def weighted_pinball_loss(
    y: np.ndarray, prediction: np.ndarray | float, level: float, weights: np.ndarray
) -> float:
    """Weight-normalised exact pinball loss, the objective both fits minimise."""
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    residual = y - np.asarray(prediction, dtype=float)
    return float(
        np.sum(weights * pinball_loss(residual, level)) / float(np.sum(weights))
    )


def fit_constant_upper_level(
    y: np.ndarray,
    *,
    coverage_level: float,
    weights: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Exact minimiser of the weighted pinball loss over *constant* predictors.

    The weighted ``coverage_level``-quantile of ``y`` is that minimiser: the
    subgradient of ``sum_i w_i rho_tau(y_i - c)`` changes sign exactly where the
    cumulative weight below ``c`` first reaches ``tau`` of the total, and the
    optimum is always attained at a sample point.  Fitting the constant this way
    — rather than by re-running the max-affine optimiser with one flat piece —
    keeps the scalar arm free of the envelope's search noise, so a floor that
    fails cannot be blamed on a worse fit of the same objective.

    The returned level is clipped at zero so the deployed correction stays
    non-negative and its feasible sets stay nested in ``strength``.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        raise ValueError("Cannot fit a constant level on an empty sample")
    weights = (
        np.ones(y.size, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if weights.size != y.size:
        raise ValueError("weights must align with y")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must be non-negative with a positive sum")
    tau = float(coverage_level)
    if not 0.0 < tau < 1.0:
        raise ValueError("coverage_level must lie strictly inside (0, 1)")

    order = np.argsort(y, kind="stable")
    sorted_y = y[order]
    sorted_w = weights[order]
    total = float(np.sum(sorted_w))
    cumulative = np.cumsum(sorted_w)
    index = int(np.searchsorted(cumulative, tau * total, side="left"))
    index = min(index, sorted_y.size - 1)
    raw_level = float(sorted_y[index])
    level = float(max(0.0, raw_level))
    info = {
        "n_points": int(y.size),
        "total_weight": total,
        "coverage_level": tau,
        "raw_weighted_quantile": raw_level,
        "level": level,
        "clipped_at_zero": bool(raw_level < 0.0),
        "weighted_exact_pinball_at_raw": weighted_pinball_loss(
            y, raw_level, tau, weights
        ),
        "weighted_exact_pinball_at_level": weighted_pinball_loss(
            y, level, tau, weights
        ),
        "empirical_coverage": float(np.mean(y <= level)),
        "weighted_empirical_coverage": float(
            np.sum(weights * (y <= level)) / total
        ),
        "max_y": float(np.max(y)),
        "mean_y": float(np.mean(y)),
    }
    return level, info


def _unpack(theta: np.ndarray, n_pieces: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    slopes = theta[: n_pieces * dim].reshape(n_pieces, dim)
    intercepts = theta[n_pieces * dim :].reshape(n_pieces)
    return slopes, intercepts


def _objective_and_gradient(
    theta: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_pieces: int,
    level: float,
    smoothing: float,
    l2: float,
) -> tuple[float, np.ndarray]:
    dim = Z.shape[1]
    slopes, intercepts = _unpack(theta, n_pieces, dim)
    values = np.concatenate(
        [
            np.zeros((Z.shape[0], 1), dtype=float),
            Z @ slopes.T + intercepts,
        ],
        axis=1,
    )
    shift = np.max(values, axis=1, keepdims=True)
    exp_values = np.exp((values - shift) / smoothing)
    denominator = np.sum(exp_values, axis=1, keepdims=True)
    envelope = (shift + smoothing * np.log(denominator)).reshape(-1)
    softmax = (exp_values / denominator)[:, 1:]

    u = y - envelope
    total_weight = float(np.sum(weights))
    loss = float(
        np.sum(weights * smooth_pinball_loss(u, level, smoothing))
        / total_weight
    )
    penalty = float(l2 * np.mean(slopes**2)) if slopes.size else 0.0

    # d/du of the smooth pinball, then chain through u = y - E(x).
    d_loss_du = level - _sigmoid(-u / smoothing)
    coefficient = -(weights * d_loss_du) / total_weight
    grad_intercepts = coefficient @ softmax
    grad_slopes = (softmax * coefficient[:, None]).T @ Z
    if slopes.size:
        grad_slopes = grad_slopes + (2.0 * l2 / slopes.size) * slopes
    return loss + penalty, np.concatenate(
        [grad_slopes.reshape(-1), grad_intercepts.reshape(-1)]
    )


def _initial_thetas(
    Z: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_pieces: int,
    n_restarts: int,
    seed: int,
) -> list[np.ndarray]:
    """Deterministic starts: a flat envelope plus weighted partition fits."""
    dim = Z.shape[1]
    positive_level = float(np.quantile(y, 0.9)) if y.size else 0.0
    flat = np.zeros(n_pieces * dim + n_pieces, dtype=float)
    flat[n_pieces * dim :] = positive_level
    thetas = [flat]

    rng = np.random.default_rng(seed)
    order = np.argsort(y)
    for restart in range(max(0, n_restarts - 1)):
        slopes = np.zeros((n_pieces, dim), dtype=float)
        intercepts = np.zeros(n_pieces, dtype=float)
        if restart == 0:
            # Partition by residual severity: each piece owns one y-stratum, so
            # the initial envelope already separates safe from risky regions.
            groups = np.array_split(order, n_pieces)
        else:
            groups = np.array_split(rng.permutation(Z.shape[0]), n_pieces)
        for index, group in enumerate(groups):
            if group.size <= dim:
                intercepts[index] = positive_level
                continue
            design = np.concatenate(
                [Z[group], np.ones((group.size, 1), dtype=float)], axis=1
            )
            sqrt_w = np.sqrt(weights[group])[:, None]
            solution, *_ = np.linalg.lstsq(
                design * sqrt_w, y[group] * sqrt_w.reshape(-1), rcond=None
            )
            slopes[index] = solution[:dim]
            intercepts[index] = solution[dim]
        thetas.append(
            np.concatenate([slopes.reshape(-1), intercepts.reshape(-1)])
        )
    return thetas


def fit_smooth_max_affine_envelope(
    X: np.ndarray,
    y: np.ndarray,
    *,
    bounds: list[tuple[float, float]],
    n_pieces: int,
    l2: float,
    smoothing: float,
    coverage_level: float,
    weights: np.ndarray | None = None,
    seed: int = 0,
    n_restarts: int = 3,
    max_iter: int = 500,
) -> tuple[SmoothMaxAffineEnvelope, dict[str, Any]]:
    """Weighted pinball fit of a smooth max-affine upper envelope of ``y``.

    ``y`` is the residual excess ``e(x)``.  ``coverage_level`` is the quantile
    level the envelope targets (``1 - alpha`` for the deployed contract); the
    weights carry the decision-aware part: selected false positives and their
    matched safe-boundary controls are up-weighted by the caller.
    """
    X = _as2d(X)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] != y.size:
        raise ValueError("X and y must have equal length")
    if X.shape[0] == 0:
        raise ValueError("Cannot fit an envelope on an empty sample")
    weights = (
        np.ones(y.size, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if weights.size != y.size:
        raise ValueError("weights must align with y")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must be non-negative with a positive sum")

    lower = np.array([bound[0] for bound in bounds], dtype=float)
    span = np.maximum(
        np.array([bound[1] - bound[0] for bound in bounds], dtype=float), 1e-12
    )
    Z = (X - lower) / span

    best_theta: np.ndarray | None = None
    best_loss = float("inf")
    attempts: list[dict[str, Any]] = []
    for index, theta0 in enumerate(
        _initial_thetas(Z, y, weights, n_pieces, n_restarts, seed)
    ):
        result = minimize(
            _objective_and_gradient,
            theta0,
            args=(Z, y, weights, n_pieces, coverage_level, smoothing, l2),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-10},
        )
        attempts.append(
            {
                "init": index,
                "success": bool(result.success),
                "loss": float(result.fun),
                "n_iterations": int(result.nit),
            }
        )
        if float(result.fun) < best_loss:
            best_loss = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)

    if best_theta is None:  # pragma: no cover - minimize always returns x
        raise RuntimeError("Envelope fit produced no candidate")
    slopes, intercepts = _unpack(best_theta, n_pieces, Z.shape[1])
    envelope = SmoothMaxAffineEnvelope(
        slopes, intercepts, smoothing, lower, span
    )
    fitted = envelope.batch(X)
    info = {
        "n_points": int(y.size),
        "total_weight": float(np.sum(weights)),
        "coverage_level": float(coverage_level),
        "l2": float(l2),
        "smoothing": float(smoothing),
        "n_pieces": int(n_pieces),
        "seed": int(seed),
        "objective_value": best_loss,
        "attempts": attempts,
        "weighted_exact_pinball": float(
            np.sum(weights * pinball_loss(y - fitted, coverage_level))
            / float(np.sum(weights))
        ),
        "empirical_coverage": float(np.mean(y <= fitted)),
        "weighted_empirical_coverage": float(
            np.sum(weights * (y <= fitted)) / float(np.sum(weights))
        ),
        "mean_envelope_on_fit": float(np.mean(fitted)),
        "max_envelope_on_fit": float(np.max(fitted)),
        "min_envelope_on_fit": float(np.min(fitted)),
    }
    return envelope, info

"""One-sided split conformal prediction for LLM-OCL Mode A.

This is the only genuinely new component versus LLM-SR. Given a fitted
constraint predictor h_hat(x) (the LLM-discovered + BFGS-tuned `equation`)
and a held-out calibration set {(x_i, h_i)}_{i=1}^n that is exchangeable
with future deployment data, this module computes a slack q_hat such that

    P( h_true(x*) <= h_hat(x*) + q_hat ) >= 1 - alpha

for any future test point x*. The downstream OCL solver embeds the
calibrated constraint as

    h_hat(x) + q_hat <= 0     (equivalently  h_hat(x) <= -q_hat)

instead of the naive `h_hat(x) <= 0`, providing a marginal coverage
guarantee on feasibility regardless of which symbolic skeleton the LLM
discovered, how well BFGS fit the parameters, or how heteroskedastic
the noise is.

References:
    - Vovk et al., "Algorithmic Learning in a Random World" (2005)
    - Lei et al., "Distribution-free predictive inference for regression",
      JASA 2018  -- the (1-alpha)(n+1)/n finite-sample correction.
    - Maragno et al., C-MICL framework for conformal MIO constraint
      learning (the OCL-specific instantiation we're following here).

The "one-sided upper" form is critical for OCL: a two-sided interval
would also widen `h_hat` from below, which is operationally meaningless
(over-feasibility doesn't violate safety). Pairing this with the
asymmetric pinball-tau loss (tau = 1 - alpha) used at SR time biases the
predictor toward over-prediction, which keeps q_hat small (often near
zero or negative -- conformal can TIGHTEN, not just loosen, the naive
constraint when the LLM did a good job).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
from typing import Callable, Optional

import numpy as np

from ocl import code_manipulation


# Type alias: a calibrated constraint predictor takes (X, params) and
# returns h_hat(X). This matches the signature of the `equation` function
# in every LLM-OCL spec.
EquationFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class ConformalResult:
    """Output of `calibrate(...)`.

    Attributes:
        q_hat: The calibrated additive slack.  Embed `h_hat(x) + q_hat <= 0`
            as the OCL constraint to obtain >= 1-alpha marginal coverage.
        alpha: User-specified target miscoverage rate (e.g. 0.10 for 90%).
        n_calibration: Number of held-out points used.
        realized_coverage_calibration: Fraction of calibration residuals
            satisfying r <= q_hat. By construction this is >= 1-alpha
            (with the (n+1)/n correction it slightly exceeds the target).
        residual_min/max/mean/std: Summary statistics of the residual
            distribution, useful for diagnosing predictor quality
            independently of the conformal step.
        params: The BFGS-optimised parameter vector that was used to
            evaluate h_hat. Echoed for downstream reproducibility.
    """
    q_hat: float
    alpha: float
    n_calibration: int
    realized_coverage_calibration: float
    residual_min: float
    residual_max: float
    residual_mean: float
    residual_std: float
    # LLM-OCL: full residual quantile profile so we can compare the
    # *shape* of the conformal nonconformity distribution against
    # CMICL baselines (linear / MLP / tree). A predictor that yields
    # the same q_hat at the configured alpha but a tighter overall
    # residual spread is genuinely better -- it would dominate at
    # every alpha, not just the operating one.
    residual_q05: float
    residual_q25: float
    residual_q50: float
    residual_q75: float
    residual_q95: float
    residual_skewness: float       # Pearson moment skew of residuals
    width_relative: float          # |q_hat| / mean(|h_cal|), unitless
    params: list[float]


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """One-sided split-conformal upper quantile with finite-sample correction.

    Returns q such that the empirical fraction of `residuals` <= q is
    at least ceil((1-alpha)(n+1))/n. Per Lei et al. 2018, this rank
    achieves marginal coverage >= 1-alpha for an exchangeable test point.

    Edge cases:
        - If the (1-alpha)(n+1) rank exceeds n, returns +inf to signal
          that the calibration set is too small for the requested alpha
          (alpha < 1/(n+1)). The caller should either lower alpha or
          collect more calibration data; we refuse to silently degrade.

    Args:
        residuals: 1-D array of nonconformity scores r_i = h_true_i - h_hat_i.
        alpha: target miscoverage rate in (0, 1).

    Returns:
        The calibrated upper-quantile slack q_hat.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0, 1), got {alpha}')
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1:
        raise ValueError(f'residuals must be 1-D, got shape {residuals.shape}')
    n = residuals.size
    if n < 1:
        raise ValueError('Need at least 1 calibration residual')
    if not np.all(np.isfinite(residuals)):
        bad = int(np.sum(~np.isfinite(residuals)))
        raise ValueError(
            f'{bad}/{n} calibration residuals are non-finite; '
            'the equation likely crashes or overflows on some calibration '
            'points. Filter or refit before calibrating.'
        )

    rank = int(np.ceil((1.0 - alpha) * (n + 1)))
    if rank > n:
        # alpha < 1/(n+1): conformal cannot give the requested coverage.
        return float('inf')
    sorted_r = np.sort(residuals)
    return float(sorted_r[rank - 1])


def empirical_coverage(residuals: np.ndarray, q_hat: float) -> float:
    """Fraction of residuals satisfying r <= q_hat (one-sided coverage)."""
    residuals = np.asarray(residuals, dtype=float)
    return float(np.mean(residuals <= q_hat))


def compute_residuals(
    equation_fn: EquationFn,
    params: np.ndarray,
    X: np.ndarray,
    h_true: np.ndarray,
) -> np.ndarray:
    """One-sided residuals  r_i = h_true_i - h_hat(x_i).

    Sign convention: a *positive* residual means the predictor
    UNDER-predicted the true h, which in OCL terms means the predictor
    declared a point safer than it actually was. The conformal upper
    quantile therefore inflates h_hat just enough to absorb this
    optimism with probability 1-alpha.
    """
    h_pred = np.asarray(equation_fn(X, params), dtype=float).reshape(-1)
    h_true = np.asarray(h_true, dtype=float).reshape(-1)
    if h_pred.shape != h_true.shape:
        raise ValueError(
            f'shape mismatch: h_pred {h_pred.shape} vs h_true {h_true.shape}'
        )
    return h_true - h_pred


def calibrate(
    equation_fn: EquationFn,
    params: np.ndarray,
    X_cal: np.ndarray,
    h_cal: np.ndarray,
    alpha: float,
) -> ConformalResult:
    """Run split-conformal calibration end-to-end.

    Args:
        equation_fn: The fitted predictor `(X, params) -> h_hat(X)`.
        params: BFGS-optimised parameter vector for `equation_fn`.
        X_cal: calibration features, shape (n, d).
        h_cal: calibration targets, shape (n,).
        alpha: miscoverage rate (e.g. 0.10 for 90% coverage).

    Returns:
        A `ConformalResult` containing the calibrated slack and diagnostics.
    """
    residuals = compute_residuals(equation_fn, params, X_cal, h_cal)
    q_hat = conformal_quantile(residuals, alpha)
    cov = empirical_coverage(residuals, q_hat)

    # LLM-OCL: extra distributional diagnostics for the paper. Computed
    # in pure numpy (no scipy dependency) to keep the conformal module
    # importable in minimal environments.
    qs = np.quantile(residuals, [0.05, 0.25, 0.50, 0.75, 0.95])
    r_mean = float(residuals.mean())
    r_std = float(residuals.std())
    if r_std > 1e-12:
        # Pearson's moment coefficient of skewness; matches
        # scipy.stats.skew(..., bias=True) within float precision.
        skew = float(np.mean(((residuals - r_mean) / r_std) ** 3))
    else:
        skew = 0.0
    h_cal_arr = np.asarray(h_cal, dtype=float).reshape(-1)
    h_scale = float(np.maximum(np.abs(h_cal_arr).mean(), 1e-12))
    width_rel = float(abs(q_hat) / h_scale) if np.isfinite(q_hat) else float('inf')

    return ConformalResult(
        q_hat=q_hat,
        alpha=alpha,
        n_calibration=int(residuals.size),
        realized_coverage_calibration=cov,
        residual_min=float(residuals.min()),
        residual_max=float(residuals.max()),
        residual_mean=r_mean,
        residual_std=r_std,
        residual_q05=float(qs[0]),
        residual_q25=float(qs[1]),
        residual_q50=float(qs[2]),
        residual_q75=float(qs[3]),
        residual_q95=float(qs[4]),
        residual_skewness=skew,
        width_relative=width_rel,
        params=[float(p) for p in np.asarray(params).ravel()],
    )


def evaluate_test_coverage(
    equation_fn: EquationFn,
    params: np.ndarray,
    X_test: np.ndarray,
    h_test: np.ndarray,
    q_hat: float,
) -> dict:
    """Evaluate the calibrated predictor on a held-out test set.

    Reports realised coverage, mean slack relative to |h|, and how much
    of the slack is consumed (mean residual / q_hat). Used in PDF §5
    experiments to compare conformal validity on ID vs OOD splits.
    """
    residuals = compute_residuals(equation_fn, params, X_test, h_test)
    cov = empirical_coverage(residuals, q_hat)
    return {
        'q_hat': float(q_hat),
        'n_test': int(residuals.size),
        'realized_coverage': cov,
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'residual_max': float(residuals.max()),
        'mean_relative_slack': float(
            np.abs(q_hat) / np.maximum(np.abs(h_test).mean(), 1e-12)
        ),
    }


def build_equation_from_source(
    function_source: str,
    template: code_manipulation.Program,
    function_to_evolve: str = 'equation',
) -> EquationFn:
    """Splice a discovered function body into the template and exec it.

    The result is a callable equivalent to what the LLM-SR sandbox
    produced at SR time, but bound in the current Python process so we
    can call it directly (no subprocess, no timeout).

    Args:
        function_source: full `def equation(...): ...` text from a
            persisted sample JSON's `function` field.
        template: the Program parsed from the original spec.
        function_to_evolve: name of the function to splice (default
            `equation`).

    Returns:
        A callable matching the `EquationFn` signature.
    """
    discovered = code_manipulation.text_to_function(function_source)
    program = copy.deepcopy(template)
    program.get_function(function_to_evolve).body = discovered.body

    program_str = str(program)
    namespace: dict = {}
    # Mirror the sandbox's __llmocl_source__ injection (Step 4) so any
    # AST-aware helpers in the spec keep working when re-exec'd here.
    namespace['__llmocl_source__'] = program_str
    exec(program_str, namespace)

    fn = namespace.get(function_to_evolve)
    if fn is None:
        raise ValueError(
            f'After exec, namespace has no `{function_to_evolve}` symbol.'
        )
    return fn


def select_best_sample(
    log_dir: pathlib.Path,
    require_feedback_key: Optional[str] = None,
) -> tuple[pathlib.Path, dict]:
    """Pick the highest-scoring sample JSON under `log_dir/samples/`.

    Args:
        log_dir: the `--log_path` from main.py / smoke_openai.py.
        require_feedback_key: if set, skips samples whose `feedback` dict
            does not contain this key. Used by the conformal CLI to
            exclude legacy samples that pre-date `optimized_params`
            persistence.

    Returns:
        (path_to_sample_json, parsed_json_dict)
    """
    log_dir = pathlib.Path(log_dir)
    sample_files = sorted((log_dir / 'samples').glob('samples_*.json'))
    if not sample_files:
        raise FileNotFoundError(f'No sample JSONs under {log_dir}/samples/')

    best_path: Optional[pathlib.Path] = None
    best_obj: Optional[dict] = None
    best_score: float = -np.inf
    skipped_no_key = 0
    for path in sample_files:
        try:
            obj = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        score = obj.get('score')
        if not isinstance(score, (int, float)):
            continue
        if require_feedback_key is not None:
            fb = obj.get('feedback')
            if not isinstance(fb, dict) or require_feedback_key not in fb:
                skipped_no_key += 1
                continue
        if score > best_score:
            best_score = score
            best_path = path
            best_obj = obj

    if best_path is None:
        msg = f'No legal samples found in {log_dir}/samples/.'
        if require_feedback_key:
            msg += (
                f' All {skipped_no_key} legal samples lacked '
                f'`feedback.{require_feedback_key}` -- they were produced '
                'before the spec persisted that field. Re-run main.py with '
                'the updated spec.'
            )
        raise ValueError(msg)
    return best_path, best_obj


def load_calibrated_predictor_from_log(
    log_dir: pathlib.Path,
    spec_path: pathlib.Path,
    function_to_evolve: str = 'equation',
) -> tuple[EquationFn, np.ndarray, dict]:
    """Convenience: load the best LLM-discovered equation from a run.

    Picks the best-scoring sample (with persisted `optimized_params`),
    rebuilds the merged program from `spec_path`, exec's it, and returns
    a callable + the saved params.

    Returns:
        (equation_fn, params, sample_obj_dict)
    """
    spec_text = pathlib.Path(spec_path).read_text()
    template = code_manipulation.text_to_program(spec_text)

    sample_path, obj = select_best_sample(
        pathlib.Path(log_dir),
        require_feedback_key='optimized_params',
    )
    equation_fn = build_equation_from_source(
        obj['function'], template, function_to_evolve=function_to_evolve,
    )
    params = np.asarray(obj['feedback']['optimized_params'], dtype=float)
    return equation_fn, params, {'sample_path': str(sample_path), **obj}

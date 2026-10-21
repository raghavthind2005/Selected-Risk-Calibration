"""Multi-start SLSQP solver for linear OCL problems, with honest bookkeeping.

Why this exists
---------------
An earlier multi-start loop, carried as a private ``_solve_ocl`` in each script
that needed it, kept a single ``(best_val, best_x, best_feasible)`` triple and
updated ``best_val`` and ``best_x`` from *infeasible* starts too::

    if feasible and res.fun < best_val:
        best_val, best_x, best_feasible = res.fun, res.x, True
    elif not feasible and isfinite(g_val) and res.fun < best_val:
        best_val, best_x = res.fun, res.x          # best_feasible stays True

So once any start is feasible, a later infeasible start with a smaller
objective silently overwrites the reported decision while the run keeps
``solver_feasible=True``.  The consequence was measured directly on stored
decisions of an earlier deployment: 1413 of 23400 are reported feasible but
violate their own constraint at the stored ``x*``.

This module keeps the feasible and the infeasible incumbent separate, verifies
the exact constraint at the returned point before calling it feasible, and
returns the receipt fields a run records.  Every arm of a run
shares the same solver so that differences between arms come from the
constraint geometry, not from the search.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "central_difference_gradient",
    "deterministic_starts",
    "solve_linear_ocl",
    "solve_ocl",
    "solve_ocl_multistart",
]


def deterministic_starts(
    bounds: list[tuple[float, float]], n_starts: int, seed: int
) -> np.ndarray:
    """Box centre plus ``n_starts - 1`` reproducible uniform draws.

    Callers pass the same ``seed`` for every method evaluated on one objective,
    so arms are compared under common random numbers and a monotonicity
    violation cannot be an artefact of different search points.
    """
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    centre = 0.5 * (lower + upper)
    if n_starts <= 1:
        return centre.reshape(1, -1)
    rng = np.random.default_rng(int(seed))
    draws = rng.uniform(lower, upper, size=(int(n_starts) - 1, lower.size))
    return np.concatenate([centre.reshape(1, -1), draws], axis=0)


def central_difference_gradient(
    function: Callable[[np.ndarray], float],
    x: np.ndarray,
    step: np.ndarray,
) -> np.ndarray:
    """Central differences with a per-coordinate step.

    Central rather than the forward differences SciPy uses internally: on the
    smooth arms it is an order more accurate, and on the non-smooth arms it at
    least samples both sides of the query point instead of one.
    """
    x = np.asarray(x, dtype=float)
    gradient = np.zeros(x.size, dtype=float)
    for index in range(x.size):
        forward = x.copy()
        backward = x.copy()
        forward[index] += step[index]
        backward[index] -= step[index]
        gradient[index] = (
            float(function(forward)) - float(function(backward))
        ) / (2.0 * step[index])
    return gradient


def solve_ocl(
    objective: Callable[[np.ndarray], float],
    constraint: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    starts: np.ndarray,
    *,
    objective_jac: Callable[[np.ndarray], np.ndarray] | None = None,
    max_iter: int = 200,
    feasibility_tolerance: float = 1e-6,
    fd_relative_step: float = 1e-6,
    constraint_gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    ftol: float = 1e-9,
    collect_starts: bool = False,
) -> dict[str, Any]:
    """Minimise ``objective(x)`` subject to ``constraint(x) <= 0`` on a box.

    Returns a receipt with the feasible incumbent (if any), the unconstrained
    best-effort point, the exact constraint value at the returned decision, and
    per-start counters.  ``solver_feasible`` is true only when the returned
    ``x_star`` itself satisfies ``constraint(x_star) <= feasibility_tolerance``.

    This is the single search/bookkeeping path for the project.  The legacy
    per-script ``_solve_ocl`` helpers are thin adapters over it, so a run cannot
    reintroduce the single-incumbent defect by copying a solver.

    ``collect_starts`` adds a ``"starts"`` list with one row per start (its
    ``x0``, converged point, retained per-start incumbent, exact constraint,
    SLSQP status and effort counters).  A feasible ``x0`` is already a valid
    incumbent: if SLSQP leaves the feasible set, the start is retained instead
    of being silently discarded.  It is off by default so the receipt of a
    production solve stays small; solver-budget diagnostics turn it on.
    """
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    span = np.array([bound[1] - bound[0] for bound in bounds], dtype=float)
    step = np.maximum(np.abs(span) * float(fd_relative_step), 1e-12)

    def negative_constraint(x: np.ndarray) -> float:
        return -float(constraint(x))

    if constraint_gradient is None:

        def negative_constraint_jac(x: np.ndarray) -> np.ndarray:
            return -central_difference_gradient(constraint, x, step)

    else:

        def negative_constraint_jac(x: np.ndarray) -> np.ndarray:
            return -np.asarray(
                constraint_gradient(x), dtype=float
            ).reshape(-1)

    best_feasible_value = float("inf")
    best_feasible_x: np.ndarray | None = None
    best_any_value = float("inf")
    best_any_x: np.ndarray | None = None
    n_success = 0
    n_feasible = 0
    n_exception = 0
    n_function_evaluations = 0
    n_iterations = 0
    start_rows: list[dict[str, Any]] = []

    def record(index: int, start: np.ndarray, **fields: Any) -> None:
        if collect_starts:
            start_rows.append(
                {
                    "start_index": int(index),
                    "x0": np.asarray(start, dtype=float).tolist(),
                    **fields,
                }
            )

    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)

    def evaluate_endpoint(point: np.ndarray) -> tuple[dict[str, Any] | None, str | None]:
        candidate = np.clip(np.asarray(point, dtype=float), lower, upper)
        try:
            value = float(objective(candidate))
            constraint_value = float(constraint(candidate))
        except Exception as error:  # noqa: BLE001 - endpoint failure is data
            return None, repr(error)
        return (
            {
                "x": candidate,
                "objective": value,
                "constraint": constraint_value,
                "finite": bool(
                    math.isfinite(value) and math.isfinite(constraint_value)
                ),
                "feasible": bool(
                    math.isfinite(constraint_value)
                    and constraint_value <= float(feasibility_tolerance)
                ),
            },
            None,
        )

    for start_index, start in enumerate(starts):
        initial, initial_error = evaluate_endpoint(start)
        result = None
        optimizer_error = None
        try:
            with warnings.catch_warnings():
                # SLSQP warns every time a trial step leaves the box and it
                # clips back.  That is normal, it is not an error, and the
                # candidate is clipped again below before it is scored.  Left
                # unsuppressed it emitted 2,026,582 of these into a 242 MB log
                # for a 35-minute run.
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds",
                    category=RuntimeWarning,
                )
                result = minimize(
                    fun=objective,
                    x0=np.asarray(start, dtype=float),
                    jac=objective_jac,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": negative_constraint,
                            "jac": negative_constraint_jac,
                        }
                    ],
                    options={"maxiter": int(max_iter), "ftol": float(ftol)},
                )
        except Exception as error:  # noqa: BLE001 - a failed start is data
            n_exception += 1
            optimizer_error = repr(error)

        optimizer = None
        optimizer_endpoint_error = None
        if result is not None:
            n_function_evaluations += int(getattr(result, "nfev", 0) or 0)
            n_iterations += int(getattr(result, "nit", 0) or 0)
            if bool(result.success):
                n_success += 1
            optimizer, optimizer_endpoint_error = evaluate_endpoint(result.x)
            if optimizer_endpoint_error is not None:
                n_exception += 1

        endpoints = [
            (source, endpoint)
            for source, endpoint in (
                ("initial_start", initial),
                ("optimizer_endpoint", optimizer),
            )
            if endpoint is not None and bool(endpoint["finite"])
        ]
        if not endpoints:
            record(
                start_index,
                start,
                status=("minimize_exception" if result is None else "nonfinite"),
                error=optimizer_error or optimizer_endpoint_error or initial_error,
            )
            continue

        feasible_endpoints = [
            item for item in endpoints if bool(item[1]["feasible"])
        ]
        incumbent_source, incumbent = min(
            feasible_endpoints or endpoints,
            key=lambda item: float(item[1]["objective"]),
        )

        if feasible_endpoints:
            n_feasible += 1
            _, best_from_start = min(
                feasible_endpoints,
                key=lambda item: float(item[1]["objective"]),
            )
            if float(best_from_start["objective"]) < best_feasible_value:
                best_feasible_value = float(best_from_start["objective"])
                best_feasible_x = np.asarray(best_from_start["x"], dtype=float).copy()

        _, best_endpoint = min(
            endpoints,
            key=lambda item: float(item[1]["objective"]),
        )
        if float(best_endpoint["objective"]) < best_any_value:
            best_any_value = float(best_endpoint["objective"])
            best_any_x = np.asarray(best_endpoint["x"], dtype=float).copy()

        record(
            start_index,
            start,
            status="evaluated",
            x=np.asarray(incumbent["x"], dtype=float).tolist(),
            objective=float(incumbent["objective"]),
            constraint=float(incumbent["constraint"]),
            constraint_satisfied=bool(incumbent["feasible"]),
            incumbent_source=incumbent_source,
            initial_objective=(
                float(initial["objective"])
                if initial is not None and bool(initial["finite"])
                else None
            ),
            initial_constraint=(
                float(initial["constraint"])
                if initial is not None and bool(initial["finite"])
                else None
            ),
            initial_constraint_satisfied=(
                bool(initial["feasible"]) if initial is not None else None
            ),
            optimizer_objective=(
                float(optimizer["objective"])
                if optimizer is not None and bool(optimizer["finite"])
                else None
            ),
            optimizer_constraint=(
                float(optimizer["constraint"])
                if optimizer is not None and bool(optimizer["finite"])
                else None
            ),
            optimizer_constraint_satisfied=(
                bool(optimizer["feasible"]) if optimizer is not None else None
            ),
            optimizer_success=(bool(result.success) if result is not None else False),
            optimizer_error=optimizer_error or optimizer_endpoint_error,
            message=(str(result.message) if result is not None else optimizer_error),
            nfev=int(getattr(result, "nfev", 0) or 0) if result is not None else 0,
            nit=int(getattr(result, "nit", 0) or 0) if result is not None else 0,
        )

    if best_feasible_x is not None:
        x_star = best_feasible_x
        optimal_value = best_feasible_value
        solver_feasible = True
    elif best_any_x is not None:
        x_star = best_any_x
        optimal_value = best_any_value
        solver_feasible = False
    else:
        x_star = 0.5 * (lower + upper)
        optimal_value = float(objective(x_star))
        solver_feasible = False

    constraint_at_x_star = float(constraint(x_star))
    receipt: dict[str, Any] = {
        "x_star": x_star,
        "optimal_value": float(optimal_value),
        "solver_feasible": bool(solver_feasible),
        "constraint_at_x_star": constraint_at_x_star,
        "constraint_satisfied_at_x_star": bool(
            constraint_at_x_star <= float(feasibility_tolerance)
        ),
        "n_starts": int(starts.shape[0]),
        "n_starts_success": int(n_success),
        "n_starts_feasible": int(n_feasible),
        "n_starts_exception": int(n_exception),
        "n_function_evaluations": int(n_function_evaluations),
        "n_iterations": int(n_iterations),
        "best_infeasible_value": (
            float(best_any_value)
            if best_any_x is not None and not solver_feasible
            else (float(best_any_value) if best_any_x is not None else None)
        ),
    }
    if collect_starts:
        receipt["starts"] = start_rows
    return receipt


def solve_linear_ocl(
    weights: np.ndarray,
    constraint: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    starts: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Minimise ``weights . x`` subject to ``constraint(x) <= 0`` on a box.

    Thin specialisation of :func:`solve_ocl` that supplies the exact objective
    gradient, so SLSQP never spends constraint evaluations approximating a
    gradient it already knows.  Every other behaviour, including the receipt
    fields, is :func:`solve_ocl`'s.
    """
    frozen = np.asarray(weights, dtype=float).reshape(-1)

    def objective(x: np.ndarray) -> float:
        return float(np.dot(np.asarray(x, dtype=float).ravel(), frozen))

    def objective_jac(x: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return frozen

    return solve_ocl(
        objective,
        constraint,
        bounds,
        starts,
        objective_jac=objective_jac,
        **kwargs,
    )


def solve_ocl_multistart(
    objective: Callable[[np.ndarray], float],
    constraint: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_multistart: int,
    seed: int,
    max_iter: int = 200,
    *,
    feasibility_tolerance: float = 1e-6,
    **kwargs: Any,
) -> dict[str, Any]:
    """Adapter for the historical per-script ``_solve_ocl`` call signature.

    The scripts that predate :func:`solve_ocl` each carried their own copy of a
    multistart SLSQP loop with a single ``(best_val, best_x, best_feasible)``
    incumbent; six of those copies let an infeasible start overwrite the
    reported decision while the feasibility flag stayed true (see the module
    docstring).  They now call this function instead, so the search, the
    feasible/infeasible separation and the exact re-verification are shared.

    ``deterministic_starts`` reproduces the legacy start points exactly (box
    centre plus ``n_multistart - 1`` draws from ``default_rng(seed)``), so the
    only behavioural change is the bookkeeping and the analytic constraint
    gradient.  The receipt carries both the legacy key names (``feasible``,
    ``solve_wall_clock``) and the receipt fields.
    """
    started = time.perf_counter()
    solved = solve_ocl(
        objective,
        constraint,
        bounds,
        deterministic_starts(bounds, int(n_multistart), int(seed)),
        max_iter=int(max_iter),
        feasibility_tolerance=float(feasibility_tolerance),
        **kwargs,
    )
    x_star = np.asarray(solved["x_star"], dtype=float)
    return {
        **solved,
        "x_star": x_star.tolist(),
        "x_star_array": x_star,
        "feasible": bool(solved["solver_feasible"]),
        "constraint_satisfied": bool(solved["constraint_satisfied_at_x_star"]),
        "reported_feasible_but_constraint_violated": bool(
            solved["solver_feasible"]
            and not solved["constraint_satisfied_at_x_star"]
        ),
        "solve_wall_clock": time.perf_counter() - started,
        "multistart_seed": int(seed),
        "n_multistart": int(n_multistart),
    }

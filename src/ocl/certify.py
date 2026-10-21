"""Certification of a locked margin, and SRC's final selection step.

A margin is *locked* before its certification objectives are drawn.  Solving
those ``n`` objectives under the locked margin yields a violation count ``N``,
and certification is a statement about ``N`` alone::

    U_n(v; eps) = sup{ r in [0,1] : Pr(Bin(n, r) <= v) >= eps }
    Ubar        = U_n(N; eps)
    certified   <=>  Ubar <= gamma

``U_n(., eps)`` is the one-sided Clopper-Pearson upper confidence bound at level
``eps``.  For ``v < n`` it is the ``(1 - eps)``-quantile of ``Beta(v + 1, n - v)``
and ``U_n(n; eps) = 1``.  It is non-decreasing in ``v``, so the test
``Ubar <= gamma`` is equivalent to a comparison of counts against a single
integer threshold

    b = max{ v in {0..n} : U_n(v; eps) <= gamma }      (b = -1 if there is none)

which is why a certification decision can be read off a count without
recomputing a quantile.  With the study's constants ``b = 19`` of ``n = 300``.

Several margins are certified in one run, one per target cost.  The familywise
error budget ``delta`` is spent equally over the ``K`` margins, so each is
certified at ``eps = delta / K``; the validity theorem then holds for all ``K``
at once, which is what lets the final step choose among them by cost.  That
final step is :func:`select_margin`: run the certification at each target cost,
return the cheapest certified margin, and return nothing -- abstain -- when no
margin is certified.

The constants come from ``config/reported_arms.json`` and nowhere else; this
module carries no defaults for them.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from scipy.stats import beta

__all__ = [
    "CertificationSetting",
    "Certificate",
    "Selection",
    "certify",
    "certification_threshold",
    "clopper_pearson_upper",
    "load_setting",
    "select_margin",
]

CONFIG_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent
               / "config" / "reported_arms.json")


@dataclass(frozen=True)
class CertificationSetting:
    """The constants of the certification procedure.

    ``gamma`` is the risk target, ``delta`` the familywise error budget, ``K``
    the number of margins the budget is spent over, and ``n`` the number of
    certification objectives behind each margin.
    """

    gamma: float
    delta: float
    K: int
    n: int

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must lie in (0, 1); got {self.gamma!r}")
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {self.delta!r}")
        if int(self.K) < 1:
            raise ValueError(f"K must be at least 1; got {self.K!r}")
        if int(self.n) < 1:
            raise ValueError(f"n must be at least 1; got {self.n!r}")

    @property
    def epsilon(self) -> float:
        """``eps = delta / K``, the level each single margin is certified at."""
        return self.delta / self.K

    @property
    def threshold(self) -> int:
        """``b``: the largest violation count that still certifies."""
        return certification_threshold(self.n, self.epsilon, self.gamma)

    @property
    def guarantee_applies(self) -> bool:
        """Whether the validity theorem's side condition ``b <= n*gamma - 1`` holds.

        The bound on the selected risk is stated under this condition.  It is a
        property of the constants alone, so it can be checked before any
        objective is solved.
        """
        return self.threshold <= self.n * self.gamma - 1.0

    def describe(self) -> str:
        return (f"gamma={self.gamma} delta={self.delta} K={self.K} n={self.n} "
                f"eps=delta/K={self.epsilon!r} b={self.threshold}")


def load_setting(path: str | pathlib.Path | None = None) -> CertificationSetting:
    """Read the certification constants from the repository's config file."""
    path = pathlib.Path(path) if path is not None else CONFIG_PATH
    with open(path) as fh:
        cfg = json.load(fh)
    try:
        study = cfg["studies"]["certification_study"]
    except KeyError as exc:                                    # pragma: no cover
        raise SystemExit(f"{path}: no certification_study block") from exc
    missing = [k for k in ("gamma", "delta", "familywise_k", "n_certification")
               if k not in study]
    if missing:
        raise SystemExit(f"{path}: certification_study is missing {missing}")
    return CertificationSetting(gamma=float(study["gamma"]),
                                delta=float(study["delta"]),
                                K=int(study["familywise_k"]),
                                n=int(study["n_certification"]))


def clopper_pearson_upper(v: int, n: int, epsilon: float) -> float:
    """``U_n(v; eps)``, the one-sided Clopper-Pearson upper bound.

    ``sup{r : Pr(Bin(n, r) <= v) >= eps}``, which for ``v < n`` is the
    ``(1 - eps)``-quantile of ``Beta(v + 1, n - v)``, and ``1`` at ``v = n``.
    """
    v, n = int(v), int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    if not 0 <= v <= n:
        raise ValueError(f"violation count {v} is outside 0..{n}")
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must lie in (0, 1); got {epsilon!r}")
    if v == n:
        return 1.0
    return float(beta.ppf(1.0 - epsilon, v + 1, n - v))


def certification_threshold(n: int, epsilon: float, gamma: float) -> int:
    """``b``: the largest ``v`` in ``0..n`` with ``U_n(v; eps) <= gamma``.

    Returns ``-1`` when even a clean run of ``n`` objectives does not certify,
    i.e. when ``U_n(0; eps) > gamma``.  ``U_n(., eps)`` is non-decreasing in
    ``v``, so a bisection on the predicate is exact rather than approximate.
    """
    n = int(n)
    if clopper_pearson_upper(0, n, epsilon) > gamma:
        return -1
    low, high = 0, n                        # invariant: U(low) <= gamma
    while low < high:
        mid = (low + high + 1) // 2
        if clopper_pearson_upper(mid, n, epsilon) <= gamma:
            low = mid
        else:
            high = mid - 1
    return low


@dataclass(frozen=True)
class Certificate:
    """The outcome of certifying one locked margin."""

    violations: int
    n: int
    bound: float          # Ubar = U_n(violations; eps)
    certified: bool
    threshold: int        # b
    gamma: float
    epsilon: float

    def __str__(self) -> str:
        return (f"{self.violations}/{self.n} violations  "
                f"Ubar={self.bound:.6f}  "
                f"{'certified' if self.certified else 'rejected'} "
                f"(gamma={self.gamma}, b={self.threshold})")


def certify(violations: int, setting: CertificationSetting,
            n: int | None = None) -> Certificate:
    """Certify a locked margin from its violation count.

    ``n`` defaults to the setting's number of certification objectives; pass it
    only when a particular margin was certified on a different number.
    """
    n = setting.n if n is None else int(n)
    bound = clopper_pearson_upper(violations, n, setting.epsilon)
    threshold = certification_threshold(n, setting.epsilon, setting.gamma)
    return Certificate(violations=int(violations), n=n, bound=bound,
                       certified=bound <= setting.gamma, threshold=threshold,
                       gamma=setting.gamma, epsilon=setting.epsilon)


@dataclass(frozen=True)
class Selection:
    """What the final step returns: a margin, or an abstention."""

    chosen: Any | None                       # the cheapest certified candidate
    certificates: Sequence[tuple[Any, Certificate]]

    @property
    def abstained(self) -> bool:
        return self.chosen is None

    def __str__(self) -> str:
        if self.abstained:
            return (f"abstain: none of {len(self.certificates)} margins certified")
        cost = self.chosen.get("cost") if isinstance(self.chosen, dict) else None
        return (f"selected the cheapest certified margin"
                + (f" at cost {cost}" if cost is not None else ""))


def select_margin(candidates: Iterable[Any],
                  setting: CertificationSetting) -> Selection:
    """SRC's final step: the cheapest certified margin, or abstain.

    Each candidate is a mapping carrying at least ``cost`` (the target cost the
    margin was sized to) and ``violations`` (its count on that margin's ``n``
    certification objectives); anything else it carries is passed through
    untouched.  An optional ``n`` overrides the setting's count for that
    candidate.

    Candidates are considered in increasing cost and the first certified one is
    returned.  Because the familywise budget already covers all ``K`` margins,
    choosing among them by cost does not weaken the guarantee.  When nothing
    certifies, ``chosen`` is ``None``: the procedure abstains rather than
    deploying an uncertified margin.
    """
    ordered = sorted(candidates, key=lambda c: float(c["cost"]))
    certificates = [(c, certify(c["violations"], setting, c.get("n")))
                    for c in ordered]
    chosen = next((c for c, cert in certificates if cert.certified), None)
    return Selection(chosen=chosen, certificates=certificates)


if __name__ == "__main__":
    s = load_setting()
    print(f"certification setting: {s.describe()}")
    print(f"  theorem side condition b <= n*gamma - 1: {s.guarantee_applies} "
          f"({s.threshold} <= {s.n * s.gamma - 1.0})")
    print(f"  U_n(v; eps) at v = b, b+1: "
          f"{clopper_pearson_upper(s.threshold, s.n, s.epsilon):.6f}, "
          f"{clopper_pearson_upper(s.threshold + 1, s.n, s.epsilon):.6f}")

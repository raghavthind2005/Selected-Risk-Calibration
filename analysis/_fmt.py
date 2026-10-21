#!/usr/bin/env python3
"""The single rounding rule for every paper-facing table and figure label.

**Round half up, from the exact value.**

Two things this must get right, and Python's defaults get both wrong.

1. `round()` and `format(x, '.3f')` use round-half-to-EVEN. An exact 0.4775
   becomes 0.477, but Table 1 in the manuscript prints 0.478. So the built-ins
   cannot be used for paper-facing output.

2. Rounding half-up from a **float** is still wrong, because the float is not the
   exact value. float64 0.4775 is 0.477499999999999980015985556747182272374629974365234375,
   which is strictly below the halfway point, so even a half-up rule applied to the
   float rounds it DOWN to 0.477. The rule has to be applied to the exact rational.

Every rate in these tables is exact: a risk is `n_viol / n_deployed` with both
integers, and a pooled cell is the mean of k such rates. So the exact value is a
`Fraction`, and `half_up()` takes one.

    >>> half_up(Fraction(191, 400), 3)      # exact 0.4775
    '0.478'
    >>> half_up(Fraction(33, 80), 3)        # exact 0.4125
    '0.413'
    >>> half_up(Fraction(7, 80), 3)         # exact 0.0875
    '0.088'

CSV files keep the exact value (as a repr or as numerator/denominator); only
display output is rounded. That way the rounding rule can be re-applied or changed
without recomputing anything.
"""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction

__all__ = ["half_up", "mean_fraction", "rate_fraction", "exact", "ROUNDING_RULE"]

ROUNDING_RULE = ("round half up, applied to the exact rational value "
                 "(not to its float64 approximation, and not Python's "
                 "round-half-to-even)")


def exact(value) -> Fraction:
    """The exact decimal a stored float stands for, as written in the file."""
    return Fraction(repr(float(value)))


def rate_fraction(n_viol, n_deployed) -> Fraction:
    """One cell's risk, exactly."""
    return Fraction(int(n_viol), int(n_deployed))


def mean_fraction(fracs) -> Fraction | None:
    """The pooling rule -- mean of per-seed rates -- kept exact."""
    fs = list(fracs)
    if not fs:
        return None
    return sum(fs, Fraction(0)) / len(fs)


def half_up(value, places: int = 3) -> str:
    """Format `value` with `places` decimals, rounding half AWAY FROM ZERO.

    `value` must be a Fraction, an int, a Decimal, or a numeric string -- anything
    whose exact value is known. A float is accepted only if it is exactly
    representable at `places` decimals; otherwise it raises, because silently
    rounding a float is the bug this function exists to prevent.
    """
    if isinstance(value, float):
        f = Fraction(value).limit_denominator(10 ** (places + 6))
        if Fraction(value) != f:
            raise TypeError(
                f"half_up() refuses a float ({value!r}): its exact binary value "
                f"is not the decimal you mean, so the half-up decision may flip. "
                f"Pass a Fraction -- e.g. rate_fraction(n_viol, n_deployed) or "
                f"mean_fraction([...]).")
        value = f
    if isinstance(value, Fraction):
        with localcontext() as ctx:
            ctx.prec = max(50, places + 30)
            d = Decimal(value.numerator) / Decimal(value.denominator)
    else:
        d = Decimal(str(value))
    q = Decimal(1).scaleb(-places)
    return str(d.quantize(q, rounding=ROUND_HALF_UP))


if __name__ == "__main__":
    cases = [(Fraction(191, 400), "0.478", "exact 0.4775, manuscript prints 0.478"),
             (Fraction(33, 80), "0.413", "exact 0.4125"),
             (Fraction(7, 80), "0.088", "exact 0.0875"),
             (Fraction(151, 400), "0.378", "exact 0.3775"),
             (Fraction(3, 32), "0.094", "exact 0.09375"),
             (Fraction(0), "0.000", "zero")]
    bad = 0
    for frac, want, why in cases:
        got = half_up(frac, 3)
        py = f"{float(frac):.3f}"
        flag = "" if got == want else "  <-- WRONG"
        if got != want: bad += 1
        print(f"  {why:<38} half_up={got}  python%.3f={py}  "
              f"{'differ' if got != py else 'same  '}{flag}")
    try:
        half_up(0.4775, 3); print("  FAIL: should have refused a float")
        bad += 1
    except TypeError:
        print("  refuses a non-exact float, as designed")
    raise SystemExit(1 if bad else 0)

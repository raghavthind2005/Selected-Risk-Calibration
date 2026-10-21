#!/usr/bin/env python3
"""The one place the paper's comparison set is read from.

`config/reported_arms.json` decides which arms the analysis compares. No analysis
script carries an arm list of its own; each one calls `study()` here and uses what
it gets back. That is what makes the config load-bearing rather than decorative --
`smoke/mutation_test.py` proves it by changing the file and watching the outputs
move.

Every accessor VALIDATES. A study that asks for a roster which is empty, or which
names an excluded arm, raises rather than silently reporting a different set.
"""
from __future__ import annotations
import functools, json, pathlib

CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "reported_arms.json"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)
    for key in ("method", "reported_baselines", "excluded_arms", "budgets", "seeds", "studies"):
        if key not in cfg:
            raise SystemExit(f"{CONFIG_PATH}: missing required key {key!r}")
    return cfg


def excluded_arms() -> tuple:
    return tuple(load()["excluded_arms"])


def study(name: str) -> dict:
    """One study's roster, validated against the top-level exclusion list."""
    cfg = load()
    if name not in cfg["studies"]:
        raise SystemExit(f"{CONFIG_PATH}: no study named {name!r}; "
                         f"have {sorted(cfg['studies'])}")
    s = dict(cfg["studies"][name])
    banned = set(cfg["excluded_arms"])
    for key, val in s.items():
        if key.startswith("_"):
            continue
        names = val if isinstance(val, list) else [val]
        for n in names:
            if isinstance(n, str) and n in banned:
                raise SystemExit(
                    f"{CONFIG_PATH}: study {name!r} field {key!r} names {n!r}, "
                    f"which is in excluded_arms. Remove it from one or the other.")
    return s


def roster(name: str, *keys: str) -> list:
    """Concatenate the named roster fields of a study, in order, de-duplicated.

    Raises if the result is empty: an empty roster means a table with no columns,
    which is a config error, not a result.
    """
    s = study(name)
    out: list = []
    for k in keys:
        if k not in s:
            raise SystemExit(f"{CONFIG_PATH}: study {name!r} has no field {k!r}")
        v = s[k]
        for n in (v if isinstance(v, list) else [v]):
            if n not in out:
                out.append(n)
    if not out:
        raise SystemExit(f"{CONFIG_PATH}: study {name!r} roster {keys} is empty")
    return out


def drop_excluded(arms):
    """Filter a data-discovered arm list through excluded_arms, order preserved."""
    banned = set(excluded_arms())
    return [a for a in arms if a not in banned]


if __name__ == "__main__":
    cfg = load()
    print(f"config: {CONFIG_PATH}")
    print(f"  method            : {cfg['method']}")
    print(f"  reported baselines: {len(cfg['reported_baselines'])}")
    print(f"  excluded arms     : {list(cfg['excluded_arms'])}")
    print(f"  studies           : {sorted(cfg['studies'])}")
    for name in sorted(cfg["studies"]):
        s = study(name)
        shown = {k: v for k, v in s.items() if not k.startswith("_")}
        print(f"    {name}: {shown}")

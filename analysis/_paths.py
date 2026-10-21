#!/usr/bin/env python3
"""Where everything lives, relative to the repository root.

Every analysis script imports these instead of carrying a path of its own, so
the repository can be cloned anywhere and run from any directory.
"""
from __future__ import annotations
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
EXPORT = RESULTS / "export"
CERTIFICATION = RESULTS / "certification"
COVERAGE = RESULTS / "coverage_diagnostic"
MATCHED_COST = RESULTS / "matched_cost"

DERIVED = ROOT / "derived"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
CONFIG = ROOT / "config"

for _d in (DERIVED, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

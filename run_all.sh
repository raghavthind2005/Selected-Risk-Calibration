#!/bin/bash
# Rebuild every table and figure from results/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c '' 2>/dev/null; then
  echo "error: PYTHON=$PYTHON is not runnable." >&2
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  cat >&2 <<'WARN'
warning: no virtual environment is active.
         The committed figures reproduce byte-for-byte only at the pinned
         versions in requirements.txt. To build one:

             python3 -m venv .venv
             source .venv/bin/activate
             pip install -r requirements.txt

         Set PYTHON=/path/to/python to use a specific interpreter instead.
WARN
fi

echo "python: $("$PYTHON" -c 'import sys;print(sys.executable, sys.version.split()[0])')"
"$PYTHON" - <<'PY' || true
import importlib
for m in ("numpy", "scipy", "matplotlib"):
    try:
        print(f"  {m}: {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"  {m}: NOT INSTALLED ({e})")
PY

cd "$HERE/analysis"
"$PYTHON" _fmt.py
"$PYTHON" _config.py

echo "--- results -> derived CSVs"
"$PYTHON" build_matched_cost.py
"$PYTHON" build_certification.py
"$PYTHON" build_equal_cost.py
"$PYTHON" build_opticl.py
"$PYTHON" build_coverage_diagnostic.py

echo "--- derived CSVs -> tables"
"$PYTHON" tab_no_margin.py
"$PYTHON" tab_coverage_summary.py
"$PYTHON" tab_alpha_sweep.py
"$PYTHON" tab_certification_all.py
"$PYTHON" tab_equal_cost_panel.py
"$PYTHON" tab_matched_cost.py
"$PYTHON" tab_matched_cost_paired.py
"$PYTHON" tab_pooled_rates.py
"$PYTHON" tab_opticl.py

echo "--- figures"
"$PYTHON" fig_coverage_diagnostic.py
"$PYTHON" fig_certificate_bound.py

echo "all analysis scripts completed"

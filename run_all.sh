#!/usr/bin/env bash
# One command: build data, run tests, run the full evaluation.
set -euo pipefail
python -m catalogue.generate
python -m evals.build_testset
python -m pytest -q
python -m evals.run_eval
echo "Open results/report.md for the numbers."

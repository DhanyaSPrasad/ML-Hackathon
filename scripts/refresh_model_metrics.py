"""Deprecated compatibility entry point.

The current scripts/train.py computes all training and student-test metrics
from the same feature schema. Running a second refresh step is no longer
required and could make results inconsistent.
"""
raise SystemExit("This script is deprecated. Run `python scripts/train.py` to regenerate model_comparison.csv.")
